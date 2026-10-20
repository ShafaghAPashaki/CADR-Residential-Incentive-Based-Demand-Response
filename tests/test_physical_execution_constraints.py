"""Regression test for physical appliance execution constraints."""
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

    ev_rating_violations = 0
    tsni_rating_violations = 0
    max_energy_balance_error = 0.0
    conflict_deferral_energy = 0.0
    conflict_unserved_energy = 0.0
    conflict_discomfort = 0.0

    for day in range(2, 244):
        mode = "train" if day <= 167 else ("val" if day <= 181 else "test")
        env.reset(day=day, mode=mode)
        for action in rng.integers(0, env.num_actions, size=24):
            env.step(int(action))

        if env.idx_car is not None:
            d = env.idx_car
            limit = float(env.DEVICE_CONSUMPTION[d])
            ev_rating_violations += int(
                (env.device_after[:, :, d] > limit + env.EPS).sum()
            )

        for d in np.where(env.TS_NI_MASK)[0]:
            limit = float(env.DEVICE_CONSUMPTION[d])
            tsni_rating_violations += int(
                (env.device_after[:, :, d] > limit + env.EPS).sum()
            )

        audit = env.get_energy_audit()
        max_energy_balance_error = max(
            max_energy_balance_error,
            float(np.max(np.abs(audit["balance_error"]))),
        )
        conflict_deferral_energy += float(env.device_conflict_deferred_out.sum())
        conflict_unserved_energy += float(env.device_conflict_unserved.sum())
        conflict_discomfort += float(env.device_conflict_discomfort.sum())

    if ev_rating_violations or tsni_rating_violations:
        raise AssertionError(
            {
                "ev_rating_violations": ev_rating_violations,
                "tsni_rating_violations": tsni_rating_violations,
            }
        )
    if max_energy_balance_error > 1e-8:
        raise AssertionError(
            {"max_energy_balance_error": max_energy_balance_error}
        )
    if conflict_deferral_energy > env.EPS and conflict_discomfort <= env.EPS:
        raise AssertionError("Physical conflict deferrals carry no dissatisfaction.")

    print(
        {
            "ev_rating_violations": ev_rating_violations,
            "tsni_rating_violations": tsni_rating_violations,
            "max_energy_balance_error": max_energy_balance_error,
            "conflict_deferral_energy": conflict_deferral_energy,
            "conflict_unserved_energy": conflict_unserved_energy,
            "conflict_discomfort": conflict_discomfort,
            "status": "PASS",
        }
    )


if __name__ == "__main__":
    main()
