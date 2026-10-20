"""Verify that forecast integration changes only the documented state fields."""
from __future__ import annotations

import copy
import sys
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from env.env import Environment  # noqa: E402


def main() -> None:
    base = yaml.safe_load((ROOT / "config_seed_0.yaml").read_text(encoding="utf-8"))
    forecast_cfg = copy.deepcopy(base)
    oracle_cfg = copy.deepcopy(base)
    forecast_cfg["forecast"]["state_input_mode"] = "forecast"
    oracle_cfg["forecast"]["state_input_mode"] = "oracle_actual"

    fenv = Environment(forecast_cfg["environment"]["house_ids"], cfg_override=forecast_cfg)
    oenv = Environment(oracle_cfg["environment"]["house_ids"], cfg_override=oracle_cfg)
    names = fenv.get_state_feature_names()
    allowed = {
        i for i, name in enumerate(names)
        if name.startswith("forecast_pre_action_house_load_norm_")
        or name in {
            "log_clipped_forecast_wholesale_price",
            "forecast_total_pre_action_load_norm",
            "forecast_capacity_need_flag",
            "forecast_needed_reduction_norm",
        }
    }
    expected_allowed = 3 + 4
    if len(allowed) != expected_allowed:
        raise AssertionError((allowed, [names[i] for i in sorted(allowed)]))

    rng = np.random.default_rng(17)
    cases = 0
    outside_changes = 0
    intended_change_cases = 0

    # Source-boundary and later-diagnostic reset checks.
    for day, mode in [(2, "train"), (167, "train"), (168, "val"),
                      (181, "val"), (182, "test"), (243, "test"),
                      (305, "test"), (334, "test")]:
        sf = fenv.reset(day=day, mode=mode).copy()
        so = oenv.reset(day=day, mode=mode).copy()
        diff = np.abs(sf.astype(float) - so.astype(float))
        changed = set(np.flatnonzero(diff > 1e-8).tolist())
        if changed - allowed:
            outside_changes += 1
            raise AssertionError(
                f"Unexpected reset-state change day={day}: "
                f"{[names[i] for i in sorted(changed - allowed)]}"
            )
        intended_change_cases += int(bool(changed & allowed))
        cases += 1

    # One complete day verifies the same mapping after commitments accumulate.
    day, mode = 208, "test"
    fenv.reset(day=day, mode=mode)
    oenv.reset(day=day, mode=mode)
    actions = rng.integers(0, fenv.num_actions, size=24)
    for h in range(24):
        sf = fenv.get_state().copy()
        so = oenv.get_state().copy()
        diff = np.abs(sf.astype(float) - so.astype(float))
        changed = set(np.flatnonzero(diff > 1e-8).tolist())
        if changed - allowed:
            outside_changes += 1
            raise AssertionError(
                f"Unexpected state change day={day} hour={h}: "
                f"{[names[i] for i in sorted(changed - allowed)]}"
            )
        intended_change_cases += int(bool(changed & allowed))
        cases += 1
        action = int(actions[h])
        fenv.step(action)
        oenv.step(action)

    if intended_change_cases == 0:
        raise AssertionError("Forecast and oracle states never differed.")
    print({
        "cases": cases,
        "allowed_state_indices": sorted(allowed),
        "allowed_state_features": [names[i] for i in sorted(allowed)],
        "outside_changes": outside_changes,
        "cases_with_intended_forecast_changes": intended_change_cases,
        "status": "PASS",
    })


if __name__ == "__main__":
    main()
