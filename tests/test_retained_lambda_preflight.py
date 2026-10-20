"""Unit regressions for the frozen original-lambda training gate."""
from __future__ import annotations

import copy
import sys
from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from utils.final_run_protocol import enforce_lambda_calibration


def main():
    for seed in (0, 1, 2):
        cfg = yaml.safe_load((ROOT / f"config_seed_{seed}.yaml").read_text(encoding="utf-8"))
        record = enforce_lambda_calibration(cfg)
        assert record["status"] == "ORIGINAL_RETAINED"
        assert int(cfg["DDQN"]["episodes"]) == 2500

        for invalid_key, invalid_value in (("lambda_c", 0.19906),
                                           ("lambda_source", "calibrated_changed")):
            tampered = copy.deepcopy(cfg)
            tampered["environment"]["symmetric_tracking"][invalid_key] = invalid_value
            try:
                enforce_lambda_calibration(tampered)
            except RuntimeError:
                pass
            else:
                raise AssertionError(f"Seed {seed}: tampered {invalid_key} passed")
    print("PASS: 3 seed configs; 6 tampering attempts rejected")

if __name__ == "__main__":
    main()
