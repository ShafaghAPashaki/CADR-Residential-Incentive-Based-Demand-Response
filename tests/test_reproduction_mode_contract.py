"""Regression contract for explicit regenerated-forecast training opt-in.

Run from repository root:
    python -m unittest tests/test_reproduction_mode_contract.py
"""
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from utils import final_run_protocol as protocol


def file_sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


class ReproductionModeContract(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        (self.root / "env").mkdir()
        (self.root / "forecast").mkdir()
        self.environment = self.root / "env" / "env.py"
        self.environment.write_text("frozen scientific environment", encoding="utf-8")
        self.forecast = self.root / "forecast" / "inputs.csv"
        self.forecast.write_text("new independently regenerated forecast", encoding="utf-8")
        self.audit = self.root / "lambda_retention.json"
        self.record = {
            "status": "ORIGINAL_RETAINED",
            "lambda_source": protocol.LAMBDA_SOURCE,
            "retained_lambda_c": protocol.RETAINED_LAMBDA_C,
            "capacity_reference_kw": 7.0,
            "environment_version": "forecast_state_v1",
            "training_forecast_file_sha256": hashlib.sha256(b"historic forecast").hexdigest(),
            "frozen_cadr_environment_sha256": file_sha(self.environment),
        }
        self.audit.write_text(json.dumps(self.record), encoding="utf-8")
        self.config = {
            "environment": {
                "capacity_threshold": 7.0,
                "symmetric_tracking": {
                    "lambda_c": protocol.RETAINED_LAMBDA_C,
                    "lambda_source": protocol.LAMBDA_SOURCE,
                },
            },
            "forecast": {"ddqn_state_input_file": "forecast/inputs.csv"},
        }
        for attribute, value in [
            ("PROJECT_ROOT", self.root), ("CALIBRATION_FILE", self.audit)
        ]:
            p = patch.object(protocol, attribute, value)
            p.start()
            self.addCleanup(p.stop)

    def test_default_rejects_regenerated_forecasts(self):
        with self.assertRaisesRegex(RuntimeError, "frozen forecast SHA changed"):
            protocol.enforce_lambda_calibration(self.config)

    def test_explicit_opt_in_allows_only_forecast_change(self):
        received = protocol.enforce_lambda_calibration(
            self.config, allow_regenerated_forecasts=True
        )
        self.assertEqual(received["retained_lambda_c"], protocol.RETAINED_LAMBDA_C)

    def test_opt_in_does_not_bypass_lambda_or_environment(self):
        self.config["environment"]["symmetric_tracking"]["lambda_c"] += 0.01
        with self.assertRaisesRegex(RuntimeError, "changed from frozen value"):
            protocol.enforce_lambda_calibration(
                self.config, allow_regenerated_forecasts=True
            )
        self.config["environment"]["symmetric_tracking"]["lambda_c"] = (
            protocol.RETAINED_LAMBDA_C
        )
        self.environment.write_text("changed environment", encoding="utf-8")
        with self.assertRaisesRegex(RuntimeError, "frozen scientific environment changed"):
            protocol.enforce_lambda_calibration(
                self.config, allow_regenerated_forecasts=True
            )


if __name__ == "__main__":
    unittest.main()
