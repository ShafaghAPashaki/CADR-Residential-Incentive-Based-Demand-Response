"""Regression test for locked-rate settlement below later wholesale prices.

Locked rates are contractual decision-hour commitments. A later lower realised
price must not invalidate settlement. Any resulting loss is recorded explicitly
and must already be present in the SP economic accounting.
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

    above_device_hours = 0
    above_hours = 0
    above_energy = 0.0
    negative_margin = 0.0
    max_reconstruction_error = 0.0

    for day in range(2, 244):
        mode = "train" if day <= 167 else ("val" if day <= 181 else "test")
        env.reset(day=day, mode=mode)
        actions = rng.integers(0, env.num_actions, size=24)
        for action in actions:
            env.step(int(action))

        flags = env.device_settlement_rate_above_price
        expected_loss = np.maximum(
            0.0,
            env.device_settlement_payment
            - env.prices[:, None, None] * env.device_verified_settlement_energy,
        )
        max_reconstruction_error = max(
            max_reconstruction_error,
            float(
                np.max(
                    np.abs(expected_loss - env.device_settlement_negative_margin),
                    initial=0.0,
                )
            ),
        )

        above_device_hours += int(flags.sum())
        above_hours += int(np.any(flags, axis=(1, 2)).sum())
        above_energy += float(env.device_settlement_above_price_energy.sum())
        negative_margin += float(env.device_settlement_negative_margin.sum())

    if max_reconstruction_error > 1e-10:
        raise AssertionError(
            f"Negative-margin reconstruction error {max_reconstruction_error:.3e}"
        )

    print(
        {
            "above_price_device_hours": above_device_hours,
            "above_price_hours": above_hours,
            "above_price_energy": above_energy,
            "negative_margin": negative_margin,
            "max_reconstruction_error": max_reconstruction_error,
            "status": "PASS",
        }
    )


if __name__ == "__main__":
    main()
