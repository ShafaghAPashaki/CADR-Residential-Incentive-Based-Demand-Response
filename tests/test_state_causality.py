"""Regression test for direct state causality.

Future realised non-shiftable load, appliance load, or price must not alter
the state observed at the current decision hour when the past/current data
and existing commitments are unchanged.
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
    zero_action = int(np.where((env.all_actions == 0).all(axis=1))[0][0])

    summary = {
        "future_nonshiftable_load": 0,
        "future_appliance_load": 0,
        "future_price": 0,
    }
    cases = 0
    max_abs_diff = {name: 0.0 for name in summary}

    for day in range(182, 244):
        env.reset(day=day, mode="test")
        # Test-only writable copy: the source price view may be read-only.
        env.prices = np.array(env.prices, copy=True)
        for h in range(23):
            if env.curr_step != h:
                raise AssertionError((day, h, env.curr_step))

            baseline_state = env.get_state().copy()

            original = env.nonshift_baseline_per_house[h + 1 :].copy()
            env.nonshift_baseline_per_house[h + 1 :] += 1.5
            perturbed = env.get_state().copy()
            env.nonshift_baseline_per_house[h + 1 :] = original
            diff = float(np.max(np.abs(baseline_state - perturbed)))
            max_abs_diff["future_nonshiftable_load"] = max(max_abs_diff["future_nonshiftable_load"], diff)
            summary["future_nonshiftable_load"] += int(diff > 1e-9)

            original = env.device_demands[h + 1 :].copy()
            env.device_demands[h + 1 :] *= 1.5
            perturbed = env.get_state().copy()
            env.device_demands[h + 1 :] = original
            diff = float(np.max(np.abs(baseline_state - perturbed)))
            max_abs_diff["future_appliance_load"] = max(max_abs_diff["future_appliance_load"], diff)
            summary["future_appliance_load"] += int(diff > 1e-9)

            original = env.prices[h + 1 :].copy()
            # Test-only writable copy of the future-price array.
            env.prices = env.prices.copy()
            env.prices[h + 1 :] *= 3.0
            perturbed = env.get_state().copy()
            env.prices[h + 1 :] = original
            diff = float(np.max(np.abs(baseline_state - perturbed)))
            max_abs_diff["future_price"] = max(max_abs_diff["future_price"], diff)
            summary["future_price"] += int(diff > 1e-9)

            cases += 1
            env.step(zero_action)

    for name, changed in summary.items():
        if changed != 0:
            raise AssertionError(f"{name}: current state changed in {changed}/{cases} cases; max_abs_diff={max_abs_diff[name]}")

    print({"cases": cases, "state_changes": summary, "max_abs_diff": max_abs_diff, "status": "PASS"})


if __name__ == "__main__":
    main()
