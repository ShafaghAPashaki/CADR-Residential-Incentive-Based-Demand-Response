"""Check generated forecast installation never rewrites archived records."""
import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

import build_ddqn_forecast_inputs as builder
from forecast_regeneration import install_outputs as installer


class ForecastBuilderTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.forecast = self.root / "results/forecast"
        (self.forecast / "crossfit").mkdir(parents=True)
        (self.forecast / "load").mkdir()
        (self.forecast / "price").mkdir()
        self.gate = self.forecast / "crossfit/crossfit_gate.txt"
        self.gate.write_bytes(b"READY\n")
        self.historical = self.forecast / "ddqn_state_forecasts_provenance.json"
        self.historical.write_bytes(b'{"historical": true}\n')
        self.output = self.forecast / "ddqn_state_forecasts.csv"
        self.provenance = self.root / "repro_runs/forecast/ddqn_state_forecasts_provenance.json"
        self.cf = self.forecast / "crossfit/ddqn_crossfit_forecasts_days_002_167.csv"
        self.load = self.forecast / "load/lstm_load_predictions_all.csv"
        self.price = self.forecast / "price/lstm_price_predictions_all.csv"
        timestamps = pd.date_range("2018-01-02", "2018-12-31 23:00", freq="h")
        train = timestamps[timestamps.dayofyear <= 167]
        full = timestamps[timestamps.dayofyear >= 168]
        pd.DataFrame({
            "timestamp": train, "doy": train.dayofyear,
            "forecast_load_eu1_kw": 1., "forecast_load_eu2_kw": 2.,
            "forecast_load_eu3_kw": 3., "forecast_aggregate_load_kw": 6.,
            "forecast_price_cent_per_kwh": 2., "inputs_padding_free_all_targets": True,
        }).to_csv(self.cf, index=False)
        pd.concat([pd.DataFrame({"timestamp": full, "doy": full.dayofyear,
                               "house_id": house, "forecast_load_kw": value})
                   for house, value in ((661, 1.), (3039, 2.), (8565, 3.))]).to_csv(self.load, index=False)
        pd.DataFrame({"timestamp": full, "doy": full.dayofyear,
                      "forecast_price_cent_per_kWh": 2., "padding_free": True}).to_csv(self.price, index=False)
        values = dict(ROOT=self.root, FORECAST_DIR=self.forecast, CROSSFIT=self.cf,
                      LOAD_FULL=self.load, PRICE_FULL=self.price, CROSSFIT_GATE=self.gate,
                      OUTPUT=self.output, PROVENANCE=self.provenance)
        self.patcher = patch.multiple(builder, **values)
        self.patcher.start()
        self.addCleanup(self.patcher.stop)

    def test_full_synthetic_year_preserves_historical_provenance(self):
        with contextlib.redirect_stdout(io.StringIO()):
            builder.main()
        data = pd.read_csv(self.output)
        record = json.loads(self.provenance.read_text())
        self.assertEqual(len(data), 8736)
        self.assertEqual(record["crossfit_rows"], 3984)
        self.assertEqual(record["full_model_rows"], 4752)
        self.assertEqual(self.historical.read_bytes(), b'{"historical": true}\n')
        self.assertEqual(set(data["forecast_source"]), {"crossfit", "full_model"})

    def test_existing_forecast_is_not_overwritten(self):
        self.output.write_bytes(b"original forecast")
        with self.assertRaises(FileExistsError):
            builder.main()
        self.assertEqual(self.output.read_bytes(), b"original forecast")
        self.assertFalse(self.provenance.exists())

    def test_existing_generated_provenance_is_not_overwritten(self):
        self.provenance.parent.mkdir(parents=True)
        self.provenance.write_bytes(b"existing")
        with self.assertRaises(FileExistsError):
            builder.main()
        self.assertFalse(self.output.exists())
        self.assertEqual(self.provenance.read_bytes(), b"existing")


class ForecastInstallerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.here = self.root / "forecast_regeneration"
        gate = self.here / "crossfit_results/audit/crossfit_gate.txt"
        gate.parent.mkdir(parents=True)
        gate.write_bytes(b"READY\r\n")
        self.installed_gate = self.root / "results/forecast/crossfit/crossfit_gate.txt"
        self.installed_gate.parent.mkdir(parents=True)
        self.installed_gate.write_bytes(b"READY\n")
        self.historical = self.root / "results/forecast/ddqn_state_forecasts_provenance.json"
        self.historical.write_bytes(b"historical provenance")
        copies = []
        for name in ("load", "price", "crossfit"):
            src = self.here / f"source_{name}.csv"
            src.write_text(f"{name} regenerated")
            copies.append((src, self.root / f"results/forecast/{name}/generated.csv"))
        for name in ("load_hourly_2018.csv", "ercot_hourly_price.csv"):
            p = self.here / "data" / name
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text("synthetic input")
        self.copies = copies
        self.state = self.root / "results/forecast/ddqn_state_forecasts.csv"
        self.generated = self.root / "repro_runs/forecast/ddqn_state_forecasts_provenance.json"
        self.patcher = patch.multiple(installer, ROOT=self.root, HERE=self.here, COPIES=copies)
        self.patcher.start()
        self.addCleanup(self.patcher.stop)

    def fake_builder(self, *args, **kwargs):
        self.state.write_text("synthetic state")
        self.generated.parent.mkdir(parents=True)
        self.generated.write_text("generated provenance")

    def test_installer_preserves_original_gate_and_provenance(self):
        with patch.object(installer.subprocess, "run", side_effect=self.fake_builder) as run:
            with contextlib.redirect_stdout(io.StringIO()):
                installer.main()
        self.assertEqual(run.call_count, 1)
        self.assertEqual(self.installed_gate.read_bytes(), b"READY\n")
        self.assertEqual(self.historical.read_bytes(), b"historical provenance")
        self.assertEqual(self.generated.read_text(), "generated provenance")
        for src, dst in self.copies:
            self.assertEqual(src.read_bytes(), dst.read_bytes())

    def test_existing_forecast_stops_before_copying(self):
        self.state.write_text("existing")
        with patch.object(installer.subprocess, "run") as run:
            with self.assertRaisesRegex(SystemExit, "Refusing to overwrite"):
                installer.main()
        self.assertEqual(self.state.read_text(), "existing")
        self.assertFalse(any(dst.exists() for _, dst in self.copies))
        run.assert_not_called()

    def test_existing_generated_provenance_stops_before_copying(self):
        self.generated.parent.mkdir(parents=True)
        self.generated.write_text("existing")
        with self.assertRaisesRegex(SystemExit, "Refusing to overwrite"):
            installer.main()
        self.assertFalse(any(dst.exists() for _, dst in self.copies))
        self.assertEqual(self.historical.read_bytes(), b"historical provenance")


if __name__ == "__main__":
    unittest.main()
