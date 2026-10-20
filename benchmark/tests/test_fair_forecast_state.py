"""Regression checks for causal/fair EBDR state and incentive reference."""
from __future__ import annotations

import copy
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from benchmark.env_benchmark import Environment  # noqa: E402


def main() -> None:
    cfg_path = ROOT / "benchmark" / "config_benchmark_seed_0.yaml"
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    house_ids = cfg["environment"]["house_ids"]

    forecast_env = Environment(house_ids, cfg_override=cfg)

    oracle_cfg = copy.deepcopy(cfg)
    oracle_cfg["forecast"]["state_input_mode"] = "oracle_actual"
    oracle_env = Environment(house_ids, cfg_override=oracle_cfg)

    intended = {0, 1, 2, 3, 6}
    changed_outside = 0
    oracle_reward_diff = 0.0
    reference_error = 0.0
    cases = 0

    rng = np.random.default_rng(20261006)
    for day in [2, 50, 100, 168, 182, 203, 243, 305, 334]:
        mode = "train" if day <= 167 else ("val" if day <= 181 else "test")
        forecast_env.reset(day=day, mode=mode)
        oracle_env.reset(day=day, mode=mode)

        expected_ref = forecast_env._previous_day_positive_price_reference(day)
        reference_error = max(
            reference_error,
            abs(forecast_env.previous_day_reference_price - expected_ref),
            float(np.max(np.abs(
                forecast_env.lambda_min
                - forecast_env.lambda_min_multiplier * expected_ref
            ))),
            float(np.max(np.abs(
                forecast_env.lambda_max
                - forecast_env.lambda_max_multiplier * expected_ref
            ))),
        )

        actions = rng.integers(0, forecast_env.num_actions, size=24)
        for h, action in enumerate(actions):
            sf = forecast_env.get_state().copy()
            so = oracle_env.get_state().copy()
            changed = {
                i for i, (a, b) in enumerate(zip(sf, so))
                if abs(float(a) - float(b)) > 1e-9
            }
            if changed - intended:
                changed_outside += 1

            _, rf, _, _ = forecast_env.step(int(action))
            _, ro, _, _ = oracle_env.step(int(action))
            oracle_reward_diff = max(oracle_reward_diff, abs(float(rf) - float(ro)))
            cases += 1

    if reference_error > 1e-12:
        raise AssertionError(f"Previous-day reference error {reference_error}")
    if changed_outside:
        raise AssertionError(
            f"Forecasting changed unintended state fields in {changed_outside} cases."
        )
    if oracle_reward_diff > 1e-12:
        raise AssertionError(
            f"Forecast/oracle reward mismatch under fixed actions: {oracle_reward_diff}"
        )

    print(
        {
            "cases": cases,
            "changed_outside_intended_fields": changed_outside,
            "oracle_reward_max_diff": oracle_reward_diff,
            "previous_day_reference_max_error": reference_error,
            "status": "PASS",
        }
    )


if __name__ == "__main__":
    main()
