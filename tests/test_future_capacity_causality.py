"""Regression test for causal future-capacity scheduling.

Future realised current-day baseline/appliance observations must not alter the
TS-NI destination or EV allocation chosen from information already available
at the current decision hour. The declared current TS-NI programme is held
fixed because it is assumed to be known to the household HEMS at request time.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from env.env import Environment  # noqa: E402


def _allocation_equal(a, b, tol=1e-9):
    if len(a) != len(b):
        return False
    return all(
        x[0] == y[0] and abs(float(x[1]) - float(y[1])) <= tol
        for x, y in zip(a, b)
    )


def main() -> None:
    cfg = yaml.safe_load((ROOT / "config_seed_0.yaml").read_text(encoding="utf-8"))
    env = Environment(cfg["environment"]["house_ids"], cfg_override=cfg)
    rng = np.random.default_rng(20261005)

    tsni_cases = tsni_changes = 0
    ev_cases = ev_changes = 0

    for day in range(2, 244):
        mode = "train" if day <= 167 else ("val" if day <= 181 else "test")
        env.reset(day=day, mode=mode)
        actions = rng.integers(0, env.num_actions, size=24)

        for h in range(23):
            jobs = []
            for d in range(env.num_devices):
                if not env.TS_NI_MASK[d]:
                    continue
                for i in range(env.N):
                    block = env._find_original_tsni_block(h, i, d)
                    if block is None:
                        continue
                    t0, _, profile = block
                    profile = profile.copy()
                    before = env._find_tsni_destination(t0, profile, i, d)
                    jobs.append((i, d, t0, profile, before))

            ev_requests = []
            if env.idx_car is not None:
                d = env.idx_car
                for i in range(env.N):
                    amount = max(
                        0.0,
                        float(
                            env.device_demands[h, i, d]
                            - env.device_shifted_out[h, i, d]
                            - env.device_conflict_deferred_out[h, i, d]
                        ),
                    )
                    if amount > env.EPS:
                        before = env._plan_ev_allocation(amount, h, i, d)
                        ev_requests.append((i, d, amount, before))

            original_nonshift = env.nonshift_baseline_per_house[h + 1 :].copy()
            original_devices = env.device_demands[h + 1 :].copy()

            # Perturb only future realised current-day observations. Known
            # commitments and declared current programmes are left unchanged.
            env.nonshift_baseline_per_house[h + 1 :] = original_nonshift * 0.05 + 8.0
            env.device_demands[h + 1 :] = original_devices * 0.1 + 2.0

            for i, d, t0, profile, before in jobs:
                after = env._find_tsni_destination(t0, profile.copy(), i, d)
                tsni_cases += 1
                tsni_changes += int(after != before)

            for i, d, amount, before in ev_requests:
                after = env._plan_ev_allocation(amount, h, i, d)
                ev_cases += 1
                ev_changes += int(not _allocation_equal(before, after))

            env.nonshift_baseline_per_house[h + 1 :] = original_nonshift
            env.device_demands[h + 1 :] = original_devices
            env.step(int(actions[h]))

        env.step(int(actions[23]))

    if tsni_changes or ev_changes:
        raise AssertionError(
            {
                "tsni_cases": tsni_cases,
                "tsni_changes": tsni_changes,
                "ev_cases": ev_cases,
                "ev_changes": ev_changes,
            }
        )

    print(
        {
            "tsni_cases": tsni_cases,
            "tsni_changes": tsni_changes,
            "ev_cases": ev_cases,
            "ev_changes": ev_changes,
            "status": "PASS",
        }
    )


if __name__ == "__main__":
    main()
