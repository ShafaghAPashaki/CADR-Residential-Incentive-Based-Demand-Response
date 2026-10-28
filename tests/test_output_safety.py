"""Exercise isolated report outputs with synthetic, data-free examples."""
import contextlib
import csv
import io
import json
import runpy
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import openpyxl
import pandas as pd

from diagnostics import evaluate_forecast_confusion as confusion
from experiments import run_reward_sensitivity as reward
from tools import export_aggregate_metrics as exporter
import summarize_results as summary


class ReportOutputTests(unittest.TestCase):
    def test_aggregate_export_uses_new_destination(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            for seed in range(3):
                for month in ("july", "august"):
                    for kind, filename in (("cadr", "audit.xlsx"), ("cbsp", "audit.xlsx"), ("ebdr", "results.xlsx")):
                        path = root / kind / f"seed_{seed}" / month / filename
                        path.parent.mkdir(parents=True)
                        path.touch()
            out = root / "output.csv"
            args = ["export", "--cadr-root", str(root / "cadr"), "--cbsp-root", str(root / "cbsp"),
                    "--ebdr-root", str(root / "ebdr"), "--skip-sensitivity", "--out", str(out)]
            with patch.object(sys, "argv", args), patch.object(exporter, "cadr_like", return_value={"metric": 2.}), \
                 patch.object(exporter, "ebdr", return_value={"metric": 3.}), contextlib.redirect_stdout(io.StringIO()):
                exporter.main()
                with out.open(newline="") as handle:
                    rows = list(csv.DictReader(handle))
                self.assertEqual(len(rows), 18)
                before = out.read_bytes()
                with self.assertRaises(FileExistsError):
                    exporter.main()
                self.assertEqual(out.read_bytes(), before)

    def test_confusion_creates_report_and_refuses_overwrite(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            fore = root / "results/forecast/ddqn_state_forecasts.csv"
            fore.parent.mkdir(parents=True)
            pd.DataFrame({"timestamp": ["2018-07-01 00:00", "2018-07-01 01:00"],
                          "doy": [182, 182], "forecast_aggregate_load_kw": [9., 6.]}).to_csv(fore, index=False)
            load = root / "data/load_hourly_2018.csv"
            load.parent.mkdir()
            pd.DataFrame([{"time": day, "dataid": house, "total": 3.}
                          for day in ("01/07/2018 00:00", "01/07/2018 01:00")
                          for house in (661, 3039, 8565)]).to_csv(load, index=False)
            out = root / "new/report.json"
            with patch.object(confusion, "ROOT", root), patch.object(sys, "argv", ["confusion", "--output", str(out)]), \
                 contextlib.redirect_stdout(io.StringIO()):
                confusion.main()
                report = json.loads(out.read_text())
                self.assertEqual(report["splits"]["july"]["true_positives"], 1)
                self.assertEqual(report["splits"]["july"]["false_negatives"], 1)
                before = out.read_bytes()
                with self.assertRaises(FileExistsError):
                    confusion.main()
                self.assertEqual(out.read_bytes(), before)

    def test_summary_writes_workbook_and_refuses_overwrite(self):
        with tempfile.TemporaryDirectory() as temp:
            out = Path(temp) / "reports/summary.xlsx"
            metrics = {name: 1. for name in (
                "remaining_violation_rate_percent", "cumulative_excess_kWh", "maximum_overrun_kW",
                "unnecessary_incentive_rate_percent", "no_response_rate_percent", "maximum_action_hours",
                "rebound_violation_hours")}
            with patch.object(summary, "PERIODS", ("july",)), patch.object(summary, "load_audit", return_value=(None, None)), \
                 patch.object(summary, "seed_metrics", return_value=metrics), patch.object(summary, "signal_rows", return_value=[]), \
                 patch.object(summary, "q_summary", return_value=pd.DataFrame()), \
                 patch.object(sys, "argv", ["summary", "--output", str(out)]), contextlib.redirect_stdout(io.StringIO()):
                summary.main()
                wb = openpyxl.load_workbook(out, read_only=True)
                try:
                    self.assertEqual(wb.sheetnames, ["Seed_Performance", "Mean_SD", "Signal_Stability", "Failure_Cases", "Q_Diagnostic"])
                finally:
                    wb.close()
                before = out.read_bytes()
                with self.assertRaises(FileExistsError):
                    summary.main()
                self.assertEqual(out.read_bytes(), before)

    def test_sensitivity_wrapper_can_import_its_helper(self):
        project = Path(__file__).resolve().parents[1]
        module = runpy.run_path(str(project / "experiments/evaluate_capacity_incentive_sensitivity.py"), run_name="audit_import")
        self.assertTrue(callable(module["main"]))

    def test_reward_sensitivity_uses_new_output_root(self):
        cfg, _ = reward.configure("eta", 0)
        self.assertTrue(cfg["general"]["results_root"].startswith("repro_runs/sensitivity_eta/runs/"))


if __name__ == "__main__":
    unittest.main()
