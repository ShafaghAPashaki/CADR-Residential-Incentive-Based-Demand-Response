"""Diagnostic-only 12x125 calibration sensitivity exercise (NOT production).

This reproduces the formula with a NEW deterministic scenario sample, not the
unarchived original selection. It MUST NOT change the frozen production lambda
or any seed config. Its output is diagnostic and is not a selection rule.

Scientific rule (Appendix B):
    lambda_c = Q95(abs(R_stakeholder)) / Q95(J_track_unweighted)

Sampling basis: twelve fixed, deterministic training-load quantiles, chosen
before examining any validation/test controller output. Prefix each scenario
with zero actions, then evaluate every one of 125 joint actions independently.
The historical audit used 12 scenarios x 125 actions but did not preserve the
original exact scenario IDs in the current project. This script records and
fixes the newly specified deterministic scenario-selection protocol.
"""
from __future__ import annotations

import argparse
import copy
import csv
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from utils.final_run_protocol import (
    PROJECT_ROOT,
    require_real_holidays, sha256_file,
)

PERCENTILE = 95.0
N_SCENARIOS = 12
# Nonproduction sensitivity diagnostic only. This label must never be copied
# into the frozen production configurations.
LAMBDA_SOURCE = "train_p95_quantile_sample_diagnostic_nonproduction_v1"
CALIBRATION_FILE = PROJECT_ROOT / "results/calibration_diagnostics/lambda_c_train_only_diagnostic.json"


def pick_scenarios(config):
    general = config["general"]
    training = config["training"]
    houses = [int(h) for h in config["environment"]["house_ids"]]
    frame = pd.read_csv(PROJECT_ROOT / general["load_file"])
    frame["dt"] = pd.to_datetime(frame["time"], format="%d/%m/%Y %H:%M", errors="raise")
    frame = frame.loc[
        (frame["dt"].dt.year == int(general["year"]))
        & frame["dataid"].isin(houses)
    ]
    train_days = {
        day for start, end in training["train_ranges"]
        for day in range(int(start), int(end) + 1)
    }
    frame = frame.loc[frame["dt"].dt.dayofyear.isin(train_days)]
    counts = frame.groupby("dt")["dataid"].nunique()
    if not (counts == len(houses)).all():
        raise ValueError("Incomplete household data in training calendar.")
    aggregate = frame.groupby("dt", sort=True)["total"].sum()
    # Stable sorting breaks ties by timestamp; deterministic, train-only.
    sorted_hours = sorted(aggregate.items(), key=lambda p: (float(p[1]), p[0]))
    if len(sorted_hours) != len(train_days) * 24:
        raise ValueError("Training calendar does not have 24 hourly rows/day.")
    index_positions = [
        int(np.floor((i + 0.5) * len(sorted_hours) / N_SCENARIOS))
        for i in range(N_SCENARIOS)
    ]
    scenarios = []
    for i, position in enumerate(index_positions):
        timestamp, total = sorted_hours[position]
        scenarios.append({
            "scenario_id": i + 1,
            "training_quantile": float((i + 0.5) / N_SCENARIOS),
            "day": int(timestamp.dayofyear),
            "hour": int(timestamp.hour),
            "timestamp": timestamp.isoformat(),
            "training_aggregate_load_kw": float(total),
        })
    if len({(s["day"], s["hour"]) for s in scenarios}) != N_SCENARIOS:
        raise RuntimeError("Calibration scenario timestamps are not unique.")
    return scenarios


def calibrate(config):
    holidays_meta = require_real_holidays()
    from env import Environment
    env = Environment(config["environment"]["house_ids"], cfg_override=config)
    if len(env.all_actions) != 125:
        raise RuntimeError("Calibration requires the unchanged 125 joint actions.")
    zeros = np.flatnonzero(np.all(np.isclose(env.all_actions, 0), axis=1))
    if len(zeros) != 1:
        raise RuntimeError("Unique zero-action index not found.")
    zero_action = int(zeros[0])
    scenarios = pick_scenarios(config)

    observations = []
    for scenario in scenarios:
        day, h = scenario["day"], scenario["hour"]
        env.reset(day=day, mode="train")
        for _ in range(h):
            env.step(zero_action)
        if env.curr_step != h:
            raise RuntimeError("Invalid zero-prefix scenario hour.")
        for action in range(env.num_actions):
            branch = copy.deepcopy(env)
            branch.step(action)
            stakeholder = float(branch.stakeholder_reward_component[h])
            tracking = float(branch.symmetric_tracking_loss[h])
            if not np.isfinite(stakeholder) or not np.isfinite(tracking) or tracking < 0:
                raise RuntimeError("Invalid calibration reward component.")
            observations.append({
                "scenario_id": scenario["scenario_id"],
                "day": day, "hour": h, "action_index": action,
                "stakeholder_reward": stakeholder,
                "absolute_stakeholder_reward": abs(stakeholder),
                "unweighted_tracking_loss": tracking,
            })

    if len(observations) != 1500:
        raise RuntimeError(f"Expected 1500 scenario-action observations; got {len(observations)}.")
    economic_p95 = float(np.percentile(
        [row["absolute_stakeholder_reward"] for row in observations], PERCENTILE
    ))
    tracking_p95 = float(np.percentile(
        [row["unweighted_tracking_loss"] for row in observations], PERCENTILE
    ))
    if not np.isfinite(economic_p95) or tracking_p95 <= 0:
        raise RuntimeError("Calibration cannot compute a finite positive lambda_c.")
    lam = float(economic_p95 / tracking_p95)
    return {
        "status": "DIAGNOSTIC_NOT_FOR_PRODUCTION",
        "lambda_source": LAMBDA_SOURCE,
        "method": "Q95(abs(stakeholder_reward)) / Q95(unweighted_smooth_L1_tracking_loss)",
        "selection_protocol": "12 evenly spaced training-baseline quantiles, stable timestamp ties, zero-action prefixes; all 125 joint actions at each prefix",
        "selection_precommitment": "train data only; no validation/test CADR performance used",
        "scenario_count": N_SCENARIOS,
        "actions_per_scenario": 125,
        "scenario_action_count": len(observations),
        "percentile": PERCENTILE,
        "economic_p95": economic_p95,
        "tracking_p95": tracking_p95,
        "recommended_lambda_c": lam,
        "previous_lambda_c": float(config["environment"]["symmetric_tracking"]["lambda_c"]),
        "capacity_kw": float(config["environment"]["capacity_threshold"]),
        "forecast_file_sha256": sha256_file(
            PROJECT_ROOT / config["forecast"]["ddqn_state_input_file"]
        ),
        "environment_version": env.VERSION,
        **holidays_meta,
        "scenarios": scenarios,
    }, observations


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write-config", action="store_true",
                        help="Unsupported: production configs are frozen. Always raises an error.")
    args = parser.parse_args()
    if args.write_config:
        parser.error("Diagnostic-only calibration may not modify frozen production configs")
    cfg = yaml.safe_load((PROJECT_ROOT / "config_seed_0.yaml").read_text(encoding="utf-8"))
    report, rows = calibrate(cfg)
    CALIBRATION_FILE.parent.mkdir(parents=True, exist_ok=True)
    audit_csv = CALIBRATION_FILE.parent / "lambda_c_scenario_action_rows.csv"
    scenarios_csv = CALIBRATION_FILE.parent / "lambda_c_training_scenarios.csv"
    with scenarios_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(report["scenarios"][0]))
        writer.writeheader(); writer.writerows(report["scenarios"])
    with audit_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    CALIBRATION_FILE.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: report[k] for k in ["economic_p95", "tracking_p95", "recommended_lambda_c", "scenario_action_count", "status"]}, indent=2))
    print(f"Audit: {CALIBRATION_FILE}")
    print("Diagnostic-only: frozen seed configs were not modified.")


if __name__ == "__main__":
    main()
