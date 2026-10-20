"""Forecasted state fields must not read realised current whole-load or price."""
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
    cfg["forecast"]["state_input_mode"] = "forecast"
    env = Environment(cfg["environment"]["house_ids"], cfg_override=cfg)
    rng = np.random.default_rng(23)

    cases = 0
    changes = 0
    for day, mode in [(100, "train"), (208, "test")]:
        env.reset(day=day, mode=mode)
        env.prices = np.array(env.prices, copy=True)
        env.baseline_per_house = np.array(env.baseline_per_house, copy=True)
        env.nonshift_baseline_per_house = np.array(env.nonshift_baseline_per_house, copy=True)
        actions = rng.integers(0, env.num_actions, size=24)
        for h in range(24):
            before = env.get_state().copy()
            price = float(env.prices[h])
            nonshift = env.nonshift_baseline_per_house[h].copy()
            baseline = env.baseline_per_house[h].copy()

            # Perturb realised current whole-house/non-shiftable load and price,
            # but leave forecast inputs, appliance availability, and commitments fixed.
            env.prices[h] = price * 3.0 + 11.0
            env.nonshift_baseline_per_house[h] = nonshift + 5.0
            env.baseline_per_house[h] = baseline + 5.0
            after = env.get_state().copy()

            env.prices[h] = price
            env.nonshift_baseline_per_house[h] = nonshift
            env.baseline_per_house[h] = baseline

            diff = float(np.max(np.abs(before.astype(float) - after.astype(float))))
            changes += int(diff > 1e-8)
            if diff > 1e-8:
                raise AssertionError(f"State read realised current load/price at day={day}, h={h}")
            cases += 1
            if h < 23:
                env.step(int(actions[h]))
        env.step(int(actions[23]))

    print({"cases": cases, "state_changes": changes, "status": "PASS"})


if __name__ == "__main__":
    main()
