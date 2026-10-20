"""Shared helpers for the predeclared R2 sensitivity protocol (R2-SENS-2026-10-14).

New file only: no existing project file is modified. The frozen final-run
protocol (utils/final_run_protocol.py) and env/env.py stay byte-identical,
so their SHA-256 checks continue to pass.
"""
import copy
import hashlib
import itertools
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parent
PROTOCOL_ID = "R2-SENS-2026-10-14"
OUT_ROOT = ROOT / "results" / "r2_sensitivity"

# Predeclared runs. Nothing else may be run under this protocol.
RUNS = {
    "cap6":  {"capacity_kw": 6.0, "levels": 5,  "purpose": "R2-C3"},
    "cap8":  {"capacity_kw": 8.0, "levels": 5,  "purpose": "R2-C3"},
    "lev10": {"capacity_kw": 7.0, "levels": 10, "purpose": "R2-M7"},
    # Optional: timed re-run of the published Seed-0 configuration.
    "cap7_repro": {"capacity_kw": 7.0, "levels": 5, "purpose": "R2-C1 timing / determinism check"},
}


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def build_config(run):
    spec = RUNS[run]
    cfg = yaml.safe_load((ROOT / "config_seed_0.yaml").read_text(encoding="utf-8"))
    cfg["general"]["seed"] = 0
    cfg["general"]["results_root"] = f"results/r2_sensitivity/{run}"
    cfg["environment"]["capacity_threshold"] = float(spec["capacity_kw"])
    # Everything else (splits, 2500 episodes, eval interval, lambda_c, eta_NN,
    # rho, omega, appliance data, forecasts, coefficients) is kept verbatim.
    assert cfg["DDQN"]["episodes"] == 2500 and cfg["DDQN"]["eval_interval"] == 50
    assert cfg["training"]["train_ranges"] == [[2, 167]]
    assert cfg["training"]["val_range"] == [168, 181]
    cfg["r2_sensitivity"] = {"protocol": PROTOCOL_ID, "run": run,
                             "capacity_kw": float(spec["capacity_kw"]),
                             "levels_per_household": int(spec["levels"])}
    return cfg


def environment_class(base, levels):
    """Return `base` unchanged for 5 levels, or a subclass with a uniform
    `levels`-point grid on [0, 1] for each household (joint size levels**N)."""
    if int(levels) == 5:
        return base

    class LevelGridEnvironment(base):
        ACTION_LEVELS = int(levels)

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.discrete_actions = np.linspace(0.0, 1.0, self.ACTION_LEVELS)
            self.all_actions = np.array(
                list(itertools.product(self.discrete_actions, repeat=self.N)),
                dtype=float,
            )
            self.num_actions = len(self.all_actions)

    LevelGridEnvironment.__name__ = f"Environment{int(levels)}Levels"
    return LevelGridEnvironment


def capacity_gate(original, declared_capacity):
    """Wrap enforce_lambda_calibration: every frozen check (lambda_c value and
    source, retention audit, forecast SHA, env.py SHA) is still run on the real
    configuration; only the 7-kW equality is replaced by an equality with the
    predeclared capacity of this run."""
    def gate(cfg):
        cap = float(cfg["environment"]["capacity_threshold"])
        if cap != float(declared_capacity):
            raise RuntimeError(f"R2 protocol: capacity {cap} != predeclared {declared_capacity}")
        if cfg.get("r2_sensitivity", {}).get("protocol") != PROTOCOL_ID:
            raise RuntimeError("R2 protocol: configuration is not tagged with the predeclared protocol")
        probe = copy.deepcopy(cfg)
        probe["environment"]["capacity_threshold"] = 7.0
        return original(probe)
    return gate
