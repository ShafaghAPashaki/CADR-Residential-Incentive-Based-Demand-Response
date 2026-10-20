"""Stage-1 state-feature equivalence regression.

With fixed actions, replacing the causal TS-NI state feature by the legacy
whole-cycle calculation must not change physical transitions, HEMS arrays,
settlement arrays, or rewards. State differences are permitted only in the
three TS-NI feature columns.
"""
from __future__ import annotations

import sys
import types
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from env.env import Environment  # noqa: E402

ARRAYS = [
    "device_shifted_out", "device_future_inj", "device_shifted_in",
    "device_curtailed", "shifted_out_locked_rate",
    "new_shifted_out_at_action", "device_unshifted_request",
    "discomforts", "incentives", "settlement_dr_energy", "rewards_total",
]


def legacy_tsni_feature(self, h):
    values = np.zeros(self.N, dtype=float)
    for d in range(self.num_devices):
        if not self.TS_NI_MASK[d]:
            continue
        for i in range(self.N):
            block = self._find_original_tsni_block(h, i, d)
            if block is not None:
                values[i] += float(block[2].sum())
    return values


def maxdiff(a, b):
    return float(np.max(np.abs(np.asarray(a, dtype=float) - np.asarray(b, dtype=float))))


def main() -> None:
    cfg = yaml.safe_load((ROOT / "config_seed_0.yaml").read_text(encoding="utf-8"))
    causal = Environment(cfg["environment"]["house_ids"], cfg_override=cfg)
    legacy = Environment(cfg["environment"]["house_ids"], cfg_override=cfg)
    legacy._current_tsni_startable_energy_per_house = types.MethodType(legacy_tsni_feature, legacy)

    names = causal.get_state_feature_names()
    tsni_idx = [i for i, name in enumerate(names) if name.startswith("current_tsni_startable_energy_norm_house_")]
    if len(tsni_idx) != causal.N:
        raise AssertionError(f"Expected {causal.N} TS-NI columns; got {tsni_idx}.")
    other_idx = [i for i in range(len(names)) if i not in tsni_idx]

    rng = np.random.default_rng(0)
    cases = 0
    tsni_state_differences = 0

    for day in range(2, 244):
        mode = "test" if day >= 182 else ("val" if day >= 168 else "train")
        causal.reset(day=day, mode=mode)
        legacy.reset(day=day, mode=mode)
        actions = rng.integers(0, causal.num_actions, size=24)

        for h in range(24):
            s_new = causal.get_state().copy()
            s_old = legacy.get_state().copy()
            if maxdiff(s_new[other_idx], s_old[other_idx]) > 1e-12:
                changed = [names[i] for i in other_idx if abs(float(s_new[i]) - float(s_old[i])) > 1e-12]
                raise AssertionError(f"Unexpected non-TSNI state difference on day={day}, hour={h}: {changed[:10]}")
            if maxdiff(s_new[tsni_idx], s_old[tsni_idx]) > 1e-12:
                tsni_state_differences += 1

            _, r_new, done_new, _ = causal.step(int(actions[h]))
            _, r_old, done_old, _ = legacy.step(int(actions[h]))
            if abs(float(r_new) - float(r_old)) > 1e-12:
                raise AssertionError(f"Reward changed on day={day}, hour={h}: {r_new} vs {r_old}")
            if bool(done_new) != bool(done_old):
                raise AssertionError(f"Done flag changed on day={day}, hour={h}.")
            for name in ARRAYS:
                if maxdiff(getattr(causal, name), getattr(legacy, name)) > 1e-12:
                    raise AssertionError(f"{name} changed on day={day}, hour={h} under fixed actions.")
            cases += 1

    if tsni_state_differences == 0:
        raise AssertionError("The causal TS-NI feature never differed from the legacy feature.")

    print({
        "cases": cases,
        "tsni_state_hours_changed": tsni_state_differences,
        "other_state_features_changed": 0,
        "physical_or_reward_arrays_changed": 0,
        "status": "PASS",
    })


if __name__ == "__main__":
    main()
