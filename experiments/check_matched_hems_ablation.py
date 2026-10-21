"""Pre-training gates for the matched-HEMS ablation (CBSP); no training is performed.
Run from the project root:  python experiments/check_matched_hems_ablation.py
"""
import copy, hashlib, sys
from pathlib import Path
import numpy as np, yaml
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(1, str(Path(__file__).resolve().parent))
from utils.final_run_protocol import require_real_holidays
require_real_holidays()
from env import Environment
from matched_hems_ablation import EconomicSPEnvironment

ORIGINAL_ENV_SHA = "5ae8182054e42be741a2450a5650f8f320f48e0aade06b8f3cb378af622ad9a6"
PHYSICAL = ["after_total_per_house", "pre_action_total_per_house", "device_after", "device_curtailed",
            "device_shifted_out", "device_shifted_in", "settlement_dr_energy", "incentive_payment",
            "discomforts", "device_discomfort", "capacity_overrun", "incentives",
            "useful_settlement_dr_energy", "net_curtailment_per_house", "signed_load_change"]
fails = []
def check(cond, msg):
    print(("PASS " if cond else "FAIL ") + msg)
    if not cond: fails.append(msg)

sha = hashlib.sha256((ROOT / "env/env.py").read_bytes()).hexdigest()
check(sha == ORIGINAL_ENV_SHA, "env/env.py SHA-256 unchanged")
rng = np.random.default_rng(20261015)
for seed in (0, 1, 2):
    cfg = yaml.safe_load((ROOT / f"config_seed_{seed}.yaml").read_text(encoding="utf-8"))
    base = Environment(data_ids=cfg["environment"]["house_ids"], cfg_override=copy.deepcopy(cfg))
    abl = EconomicSPEnvironment(data_ids=cfg["environment"]["house_ids"], cfg_override=copy.deepcopy(cfg))
    for mode, day in (("train", 20), ("train", 120), ("val", 170), ("test", 200), ("test", 230)):
        if mode == "test":
            for e in (base, abl): e.test_range = [182, 243]
        s0 = base.reset(day=day, mode=mode); s1 = abl.reset(day=day, mode=mode)
        ok_state = s0.shape == s1.shape == (43,) and np.array_equal(s0[:-2], s1[:-2]) and not s1[-2:].any()
        max_r = 0.0; phys_ok = True; done = False
        while not done:
            a = int(rng.integers(base.num_actions))
            n0, r0, done, _ = base.step(a); n1, r1, d1, _ = abl.step(a)
            h = base.curr_step - 1
            if not done:
                ok_state &= np.array_equal(n0[:-2], n1[:-2]) and not n1[-2:].any()
            p = float(abl.prices[h]); S = float(abl.reward_economic_scale)
            sp = (p * abl.settlement_dr_energy[h].sum() - abl.incentive_payment[h].sum()) / S
            eu = abl.rho * abl.incentive_payment[h].sum() / S - (1 - abl.rho) * abl.normalized_discomfort_component[h]
            hand = 0.65 * sp + 0.35 * eu
            max_r = max(max_r, abs(r1 - hand))
            for name in PHYSICAL:
                if not np.array_equal(getattr(base, name)[h], getattr(abl, name)[h]):
                    phys_ok = False; print("   differs:", name, mode, day, h)
        check(ok_state, f"seed {seed} {mode} day {day}: state equal except two zeroed need features")
        check(phys_ok, f"seed {seed} {mode} day {day}: physics/settlement identical under matched actions")
        check(max_r < 1e-12, f"seed {seed} {mode} day {day}: reward equals hand formula (max err {max_r:.2e})")
    check(abs(abl.reward_economic_scale - 32.8526) < 1e-4 and abl.stakeholder_sp_weight == 0.65 and abl.rho == 0.9,
          f"seed {seed}: S={abl.reward_economic_scale:.4f}, omega, rho as preregistered")
print("\nALL GATES PASS" if not fails else f"\n{len(fails)} GATE(S) FAILED")
sys.exit(1 if fails else 0)
