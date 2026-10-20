"""Oracle diagnostic mode reconstructs the pre-integration state exactly."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from env.env import Environment  # noqa: E402


def legacy_state(env):
    h = env.curr_step
    price = float(env.prices[h])
    committed_house = env._current_pre_action_house_load(h)
    pc = env._current_pc_available_per_house(h)
    ev = env._current_ev_available_per_house(h)
    tsni = env._current_tsni_startable_energy_per_house(h)
    next_in, rem_in = env._shifted_in_commitments_per_house(h)
    cur_relief, cur_payment = env._current_settlement_commitments_per_house(h)
    next_relief, next_payment, rem_relief, rem_payment = env._pending_settlement_commitments_per_house(h)
    total = float(committed_house.sum())
    cap = env.capacity_threshold
    upper = 80.0
    needed = max(0.0, total - cap)
    flag = 1.0 if needed > env.EPS else 0.0
    pay_scale = max(cap * upper, env.EPS)
    values = [
        committed_house / cap, pc / cap, ev / cap, tsni / cap,
        next_in / cap, rem_in / cap,
        cur_relief / cap, cur_payment / pay_scale,
        next_relief / cap, next_payment / pay_scale,
        rem_relief / cap, rem_payment / pay_scale,
        [np.log1p(np.clip(price, 0.0, upper))],
        [h / max(env.max_steps - 1, 1)], [total / cap],
        [env.is_holiday], [env.is_weekend], [flag], [needed / cap],
    ]
    return np.concatenate([np.atleast_1d(v) for v in values]).astype(np.float32)


def main() -> None:
    cfg = yaml.safe_load((ROOT / "config_seed_0.yaml").read_text(encoding="utf-8"))
    cfg["forecast"]["state_input_mode"] = "oracle_actual"
    env = Environment(cfg["environment"]["house_ids"], cfg_override=cfg)
    rng = np.random.default_rng(31)
    max_diff = 0.0
    cases = 0
    for day, mode in [(2, "train"), (80, "train"), (167, "train"), (168, "val"), (208, "test"), (243, "test")]:
        env.reset(day=day, mode=mode)
        actions = rng.integers(0, env.num_actions, size=24)
        for h in range(24):
            observed = env.get_state().copy()
            expected = legacy_state(env)
            diff = float(np.max(np.abs(observed.astype(float) - expected.astype(float))))
            max_diff = max(max_diff, diff)
            if diff > 0.0:
                raise AssertionError(f"Oracle reconstruction mismatch day={day} h={h}: {diff}")
            cases += 1
            if h < 23:
                env.step(int(actions[h]))
        env.step(int(actions[23]))
    print({"cases": cases, "max_abs_diff": max_diff, "status": "PASS"})


if __name__ == "__main__":
    main()
