"""End-to-end no-look-ahead regression for future realised price.

Future prices strictly after decision hour h must not alter the state or any
current-hour HEMS/action outcome. The next-state price itself is intentionally
not compared because it becomes observable at the next decision hour.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from env.env import Environment  # noqa: E402

DECISION_ARRAYS = [
    "device_shifted_out",
    "device_shifted_in",
    "device_future_inj",
    "device_curtailed",
    "shifted_out_locked_rate",
    "shifted_out_decision_hour",
    "shifted_out_has_ledger",
    "new_shifted_out_at_action",
    "device_unshifted_request",
    "device_settlement_ineligible_request",
    "device_discomfort",
    "discomforts",
    "incentives",
    "settlement_dr_energy",
    "incentive_payment",
    "rewards_service_provider",
    "rewards_customers",
    "rewards_total",
]


def maxdiff(a, b):
    a = np.asarray(a)
    b = np.asarray(b)
    if a.dtype == bool or b.dtype == bool:
        return float(np.any(a != b))
    return float(np.max(np.abs(a.astype(float) - b.astype(float)), initial=0.0))


def main() -> None:
    cfg = yaml.safe_load((ROOT / "config_seed_0.yaml").read_text(encoding="utf-8"))
    clean = Environment(cfg["environment"]["house_ids"], cfg_override=cfg)
    pert = Environment(cfg["environment"]["house_ids"], cfg_override=cfg)
    rng = np.random.default_rng(0)

    cases = 0
    state_changes = 0
    decision_changes = 0

    for day in range(182, 244):
        clean.reset(day=day, mode="test")
        pert.reset(day=day, mode="test")
        # Test-only writable copy, leaving production data untouched.
        pert.prices = np.array(pert.prices, copy=True)
        actions = rng.integers(0, clean.num_actions, size=24)

        for h in range(23):
            # The histories are identical before the perturbation.
            clean_state = clean.get_state().copy()
            original_future = pert.prices[h + 1 :].copy()

            # Deliberately large deterministic perturbation of unknown future
            # prices. Current price is unchanged.
            # Test-only writable copy of the future-price array.
            pert.prices = pert.prices.copy()
            pert.prices[h + 1 :] = original_future * 4.0 + 17.0
            pert_state = pert.get_state().copy()
            state_changes += int(maxdiff(clean_state, pert_state) > 1e-9)

            action = int(actions[h])
            clean.step(action)
            pert.step(action)

            changed = False
            for name in DECISION_ARRAYS:
                # Compare records through the current decision hour only.
                a = np.asarray(getattr(clean, name))
                b = np.asarray(getattr(pert, name))
                if a.ndim >= 1 and a.shape[0] == clean.T:
                    a = a[: h + 1]
                    b = b[: h + 1]
                if maxdiff(a, b) > 1e-9:
                    changed = True
                    break
            decision_changes += int(changed)

            # Restore future prices before advancing to the next decision hour.
            pert.prices[h + 1 :] = clean.prices[h + 1 :]
            cases += 1

        # Complete final hour with identical prices/actions so each new day
        # starts from a clean reset and the test does not leave partial episodes.
        clean.step(int(actions[23]))
        pert.step(int(actions[23]))

    if state_changes or decision_changes:
        raise AssertionError(
            {
                "cases": cases,
                "state_changes": state_changes,
                "decision_changes": decision_changes,
            }
        )

    print(
        {
            "cases": cases,
            "state_changes": state_changes,
            "decision_changes": decision_changes,
            "status": "PASS",
        }
    )


if __name__ == "__main__":
    main()
