"""Exercise the CADR launcher without loading data or training models."""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from tools import run_cadr


class LauncherTests(unittest.TestCase):
    def parse(self, *args):
        return run_cadr.parser().parse_args(list(args))

    def test_all_seeds_and_periods_map_to_expected_commands(self):
        args = self.parse("evaluate", "--months", "july", "august", "november")
        commands, _, outputs = run_cadr.make_plan(args)
        self.assertEqual(len(commands), 9)
        self.assertEqual(len(set(outputs)), 9)
        self.assertIn("tools/evaluate_with_regenerated_forecasts.py", commands[0])
        for month, expected in run_cadr.PERIODS.items():
            cmd = next(c for c in commands if Path(c[c.index("--output_dir") + 1]).parts[-2:] == ("seed_2", month))
            self.assertEqual(tuple(int(cmd[cmd.index(x) + 1]) for x in
                                   ("--test_start", "--test_end", "--target_plot_days")), expected)

    def test_new_checkpoint_uses_strict_evaluator_and_saved_config(self):
        args = self.parse("evaluate", "--source", "trained", "--seeds", "1", "--months", "august")
        commands, required, _ = run_cadr.make_plan(args)
        self.assertEqual(commands[0][1], "test.py")
        self.assertIn(str(Path("repro_runs/train/seed_1/training/config.yaml")), commands[0])
        self.assertEqual(len(required), 2)

    def test_custom_period_and_bounds(self):
        args = self.parse("evaluate", "--seeds", "1", "--days", "244", "273", "--plot-day", "250")
        commands, _, outputs = run_cadr.make_plan(args)
        cmd = commands[0]
        self.assertEqual(Path(cmd[cmd.index("--output_dir") + 1]).name, "days_244_273")
        self.assertEqual(tuple(int(cmd[cmd.index(x) + 1]) for x in
                               ("--test_start", "--test_end", "--target_plot_days")), (244, 273, 250))
        self.assertEqual(len(outputs), 1)
        with self.assertRaises(ValueError):
            run_cadr.make_plan(self.parse("evaluate", "--days", "1", "31"))
        with self.assertRaises(ValueError):
            run_cadr.make_plan(self.parse("evaluate", "--plot-day", "200"))

    def test_training_keeps_the_forecast_override_explicit(self):
        default = run_cadr.make_plan(self.parse("train", "--seeds", "2"))[0][0]
        regenerated = run_cadr.make_plan(self.parse("train", "--seeds", "2", "--regenerated-forecasts"))[0][0]
        self.assertNotIn("--allow_regenerated_forecasts", default)
        self.assertIn("--allow_regenerated_forecasts", regenerated)
        self.assertIn("config_seed_2.yaml", regenerated)

    def test_preflight_rejects_existing_output_before_starting(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(run_cadr, "ROOT", Path(tmp)):
            root = Path(tmp)
            (root / "config_seed_0.yaml").touch()
            (root / "config_seed_1.yaml").touch()
            existing = root / "repro_runs/train/seed_1/training"
            existing.mkdir(parents=True)
            args = self.parse("train", "--seeds", "0", "1")
            _, required, outputs = run_cadr.make_plan(args)
            with self.assertRaises(FileExistsError):
                run_cadr.check_plan(required, outputs)

    def test_missing_checkpoint_stops_before_evaluation(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(run_cadr, "ROOT", Path(tmp)):
            args = self.parse("evaluate", "--seeds", "0", "--months", "july")
            _, required, outputs = run_cadr.make_plan(args)
            with self.assertRaises(FileNotFoundError):
                run_cadr.check_plan(required, outputs)

    def test_execution_uses_repo_working_directory_and_stops_on_error(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(run_cadr, "ROOT", Path(tmp)):
            root = Path(tmp)
            (root / "config_seed_0.yaml").touch()
            (root / "config_seed_1.yaml").touch()
            failure = run_cadr.subprocess.CalledProcessError(1, "main.py")
            with patch.object(run_cadr.subprocess, "run", side_effect=failure) as run:
                with self.assertRaises(run_cadr.subprocess.CalledProcessError):
                    run_cadr.main(["train", "--seeds", "0", "1"])
                self.assertEqual(run.call_count, 1)
                self.assertEqual(run.call_args.kwargs, {"cwd": root, "check": True})


if __name__ == "__main__":
    unittest.main()
