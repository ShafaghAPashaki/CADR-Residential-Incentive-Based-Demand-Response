"""Regression checks for non-incentivised physical-conflict accounting.

These checks ensure that forced physical deferrals/unserved energy are costed
causally without being paid as demand-response relief.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from env.env import Environment  # noqa: E402


def main() -> None:
    cfg = yaml.safe_load((ROOT / "config_seed_0.yaml").read_text(encoding="utf-8"))
    env = Environment(cfg["environment"]["house_ids"], cfg_override=cfg)
    rng = np.random.default_rng(20261005)

    late_discomfort_writes = 0
    conflict_only_payment_violations = 0
    settlement_exceeds_action_relief = 0
    unserved_events = 0
    unserved_without_discomfort = 0
    total_unserved = 0.0
    total_conflict_discomfort = 0.0

    for day in range(2, 244):
        mode = "train" if day <= 167 else ("val" if day <= 181 else "test")
        env.reset(day=day, mode=mode)
        actions = rng.integers(0, env.num_actions, size=24)

        for h, action in enumerate(actions):
            # Rewards for hours < h have already been finalised. No later
            # physical-conflict resolution may write discomfort back to them.
            frozen_past = env.discomforts[:h].copy()
            env.step(int(action))
            if h and not np.allclose(env.discomforts[:h], frozen_past, atol=1e-12, rtol=0.0):
                late_discomfort_writes += 1

        conflict_out = env.device_conflict_deferred_out.sum(axis=2)
        incentivised_relief = env.device_raw_settlement_relief.sum(axis=2)
        conflict_only = (
            (conflict_out > env.EPS)
            & (incentivised_relief <= env.EPS)
        )
        conflict_only_payment_violations += int(
            np.count_nonzero(
                conflict_only
                & (
                    (env.settlement_dr_energy > env.EPS)
                    | (env.incentive_payment > env.EPS)
                )
            )
        )
        settlement_exceeds_action_relief += int(
            np.count_nonzero(
                env.settlement_dr_energy > incentivised_relief + 1e-9
            )
        )

        for i in range(env.N):
            for d in range(env.num_devices):
                unserved = np.asarray(env.device_conflict_unserved[:, i, d], dtype=float)
                total_unserved += float(unserved.sum())
                total_conflict_discomfort += float(
                    env.device_conflict_discomfort[:, i, d].sum()
                )
                for h in range(env.T):
                    starts_event = (
                        unserved[h] > env.EPS
                        and (h == 0 or unserved[h - 1] <= env.EPS)
                    )
                    if not starts_event:
                        continue
                    unserved_events += 1
                    if env.device_conflict_discomfort[h, i, d] <= env.EPS:
                        unserved_without_discomfort += 1

    failures = {
        "late_discomfort_writes": late_discomfort_writes,
        "conflict_only_payment_violations": conflict_only_payment_violations,
        "settlement_exceeds_action_relief": settlement_exceeds_action_relief,
        "unserved_without_discomfort": unserved_without_discomfort,
    }
    if any(failures.values()):
        raise AssertionError(failures)

    print(
        {
            **failures,
            "unserved_events": unserved_events,
            "total_unserved_energy": total_unserved,
            "total_conflict_discomfort": total_conflict_discomfort,
            "status": "PASS",
        }
    )


if __name__ == "__main__":
    main()
