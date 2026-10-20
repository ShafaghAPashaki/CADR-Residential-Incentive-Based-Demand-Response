"""Regression test for same-appliance TS-NI physical overlap.

A remaining original TS-NI programme must never execute in the same
household/device/hour as an incentive-shifted or feasibility-deferred programme.
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

    overlap_count = 0
    overlap_energy = 0.0
    conflict_deferral_energy = 0.0
    unserved_conflict_energy = 0.0

    for day in range(2, 244):
        mode = "train" if day <= 167 else ("val" if day <= 181 else "test")
        env.reset(day=day, mode=mode)
        for action in rng.integers(0, env.num_actions, size=24):
            env.step(int(action))

        remaining_original = np.maximum(
            env.device_demands
            - env.device_shifted_out
            - env.device_conflict_deferred_out,
            0.0,
        )
        scheduled = env.device_shifted_in + env.device_conflict_deferred_in
        tsni = env.TS_NI_MASK[None, None, :]

        mask = (
            (remaining_original > env.EPS)
            & (scheduled > env.EPS)
            & tsni
        )
        overlap_count += int(mask.sum())
        if np.any(mask):
            overlap_energy += float(
                np.minimum(remaining_original, scheduled)[mask].sum()
            )

        mask = (
            (env.device_shifted_in > env.EPS)
            & (env.device_conflict_deferred_in > env.EPS)
            & tsni
        )
        overlap_count += int(mask.sum())
        if np.any(mask):
            overlap_energy += float(
                np.minimum(
                    env.device_shifted_in, env.device_conflict_deferred_in
                )[mask].sum()
            )

        conflict_deferral_energy += float(
            (env.device_conflict_deferred_out * tsni).sum()
        )
        unserved_conflict_energy += float(
            (env.device_conflict_unserved * tsni).sum()
        )

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
            "conflict_deferral_energy": conflict_deferral_energy,
            "unserved_conflict_energy": unserved_conflict_energy,
            "status": (
                "PASS" if unserved_conflict_energy <= env.EPS
                else "PASS_WITH_UNSERVED_CONFLICT_DIAGNOSTIC"
            ),
        }
    )


if __name__ == "__main__":
    main()
