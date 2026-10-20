"""End-to-end no-look-ahead regression for future realised load.

For fixed actions, rerun July-August after perturbing realised data strictly
after each cut-off hour. States and HEMS decisions through the cut-off must be
identical. The future remainder of any TS-NI programme already declared by the
cut-off is restored after perturbation because that programme profile is an
explicit HEMS input assumption, not an unknown future observation.

This test is intentionally exhaustive and may take several minutes.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from env.env import Environment  # noqa: E402

cfg = yaml.safe_load((ROOT / "config_seed_0.yaml").read_text(encoding="utf-8"))
env = Environment(cfg["environment"]["house_ids"], cfg_override=cfg)
A = env.all_actions
DECISION_ARRAYS = [
    "device_shifted_out",
    "device_future_inj",
    "device_shifted_in",
    "device_curtailed",
    "shifted_out_locked_rate",
    "new_shifted_out_at_action",
    "device_unshifted_request",
    "incentives",
]


def refresh_derived(e):
    e.device_before = e.device_demands.copy()
    e.baseline_per_house = e.nonshift_baseline_per_house + e.device_demands.sum(axis=2)
    e.total_baseline = e.baseline_per_house.sum(axis=1)
    e.device_baseline_raw = e.device_before.sum(axis=1)
    e.lam_min, e.lam_max = e.get_lam_bounds()
    e._refresh_planned_load()


def protect_declared_tsni_remainders(clean_demands, perturbed_demands, cut, e):
    """Restore profile slices already declared to the HEMS by hour ``cut``."""
    for d in range(e.num_devices):
        if not e.TS_NI_MASK[d]:
            continue
        for i in range(e.N):
            if clean_demands[cut, i, d] <= e.EPS:
                continue
            t0 = cut
            while t0 - 1 >= 0 and clean_demands[t0 - 1, i, d] > e.EPS:
                t0 -= 1
            t1 = cut
            while t1 + 1 < e.T and clean_demands[t1 + 1, i, d] > e.EPS:
                t1 += 1
            if t0 <= cut < t1:
                perturbed_demands[cut + 1 : t1 + 1, i, d] = clean_demands[
                    cut + 1 : t1 + 1, i, d
                ]


def run(day, actions, upto, perturb=None, cut=None):
    env.reset(day=day, mode="test")
    if perturb is not None:
        clean_demands = np.array(env.device_demands, dtype=float, copy=True)
        env.prices = np.array(env.prices, dtype=float, copy=True)
        env.nonshift_baseline_per_house = np.array(
            env.nonshift_baseline_per_house, dtype=float, copy=True
        )
        env.device_demands = clean_demands.copy()
        perturb(env, cut)
        protect_declared_tsni_remainders(
            clean_demands, env.device_demands, cut, env
        )
        refresh_derived(env)

    states, snapshots = [], []
    for k in range(upto + 1):
        states.append(env.get_state().copy())
        env.step(int(actions[k]))
        snapshot = {
            name: np.asarray(getattr(env, name)).copy() for name in DECISION_ARRAYS
        }
        # Conflict resolution for hour k+1 is intentionally executed at the
        # end of step k, after that next-hour request becomes observable. It is
        # legitimate next-hour bookkeeping, not part of the decision at k.
        # Current/past discomfort is still part of the current decision record.
        snapshot["discomforts_through_current"] = env.discomforts[: k + 1].copy()
        snapshots.append(snapshot)
    return states, snapshots


def perturb_future_nonshiftable(e, h):
    e.nonshift_baseline_per_house[h + 1 :] += 1.5


def perturb_future_appliances(e, h):
    e.device_demands[h + 1 :] *= 1.5


def maxdiff(a, b):
    return float(np.max(np.abs(np.asarray(a, dtype=float) - np.asarray(b, dtype=float))))


def main():
    perturbations = {
        "future_nonshiftable_load": perturb_future_nonshiftable,
        "future_appliance_load": perturb_future_appliances,
    }
    summary = {
        name: {"cases": 0, "state_changed": 0, "decision_changed": 0}
        for name in perturbations
    }
    rng = np.random.default_rng(0)

    for day in range(182, 244):
        actions = rng.integers(0, len(A), size=24)
        clean_states, clean_snapshots = run(day, actions, 22)
        for cut in range(23):
            for name, perturb in perturbations.items():
                states, snapshots = run(day, actions, cut, perturb, cut)
                state_diff = max(
                    maxdiff(clean_states[k], states[k]) for k in range(cut + 1)
                )
                decision_changed = any(
                    maxdiff(clean_snapshots[cut][key], snapshots[cut][key]) > 1e-9
                    for key in snapshots[cut].keys()
                )
                summary[name]["cases"] += 1
                summary[name]["state_changed"] += int(state_diff > 1e-9)
                summary[name]["decision_changed"] += int(decision_changed)

    failures = {
        name: values
        for name, values in summary.items()
        if values["state_changed"] or values["decision_changed"]
    }
    if failures:
        raise AssertionError(failures)
    print({"summary": summary, "status": "PASS"})


if __name__ == "__main__":
    main()
