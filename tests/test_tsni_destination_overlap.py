"""Regression test for TS-NI device double-booking."""
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

    overlap_count = 0
    overlap_energy = 0.0

    for day in range(2, 244):
        mode = "train" if day <= 167 else ("val" if day <= 181 else "test")
        env.reset(day=day, mode=mode)
        actions = rng.integers(0, env.num_actions, size=24)
        for action in actions:
            env.step(int(action))

        for t in range(env.T):
            for i in range(env.N):
                for d in np.where(env.TS_NI_MASK)[0]:
                    original_remaining = max(
                        0.0,
                        float(
                            env.device_demands[t, i, d]
                            - env.device_shifted_out[t, i, d]
                        ),
                    )
                    shifted_in = float(env.device_future_inj[t, i, d])
                    if original_remaining > env.EPS and shifted_in > env.EPS:
                        overlap_count += 1
                        overlap_energy += min(original_remaining, shifted_in)

    if overlap_count:
        raise AssertionError(
            {
                "overlap_count": overlap_count,
                "overlap_energy": overlap_energy,
            }
        )

    print(
        {
            "overlap_count": overlap_count,
            "overlap_energy": overlap_energy,
            "status": "PASS",
        }
    )


if __name__ == "__main__":
    main()
