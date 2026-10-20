"""Regression test for the EV shifted-energy commitment cap.

Both known shifted EV commitments and realised EV charging must respect the
configured per-hour charger rating. Newly observed original charging requests
that conflict with earlier commitments are physically deferred before the
current state/action is evaluated.
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

    shifted_cap_violations = 0
    realised_cap_violations = 0
    max_shifted_commitment = 0.0
    max_realised_ev_load = 0.0
    forced_deferral_energy = 0.0
    unserved_energy = 0.0

    for day in range(2, 244):
        mode = "train" if day <= 167 else ("val" if day <= 181 else "test")
        env.reset(day=day, mode=mode)
        for action in rng.integers(0, env.num_actions, size=24):
            env.step(int(action))

        if env.idx_car is None:
            continue
        d = env.idx_car
        limit = float(env.DEVICE_CONSUMPTION[d])
        shifted = (
            env.device_shifted_in[:, :, d]
            + env.device_conflict_deferred_in[:, :, d]
        )
        realised = env.device_after[:, :, d]
        shifted_cap_violations += int((shifted > limit + env.EPS).sum())
        realised_cap_violations += int((realised > limit + env.EPS).sum())
        max_shifted_commitment = max(max_shifted_commitment, float(shifted.max()))
        max_realised_ev_load = max(max_realised_ev_load, float(realised.max()))
        forced_deferral_energy += float(
            env.device_conflict_deferred_out[:, :, d].sum()
        )
        unserved_energy += float(env.device_conflict_unserved[:, :, d].sum())

    if shifted_cap_violations or realised_cap_violations:
        raise AssertionError(
            {
                "shifted_cap_violations": shifted_cap_violations,
                "realised_cap_violations": realised_cap_violations,
                "max_shifted_commitment": max_shifted_commitment,
                "max_realised_ev_load": max_realised_ev_load,
            }
        )

    print(
        {
            "shifted_cap_violations": shifted_cap_violations,
            "max_shifted_commitment": max_shifted_commitment,
            "realised_cap_violations": realised_cap_violations,
            "max_realised_ev_load": max_realised_ev_load,
            "forced_deferral_energy": forced_deferral_energy,
            "unserved_energy": unserved_energy,
            "status": "PASS",
        }
    )


if __name__ == "__main__":
    main()
