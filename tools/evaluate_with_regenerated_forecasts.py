"""Evaluate released DDQN checkpoints with a locally regenerated forecast-state file.

The released checkpoints record the SHA-256 of the original forecast-state file
(results/forecast/ddqn_state_forecasts.csv). That file holds household-level
forecasts derived from licensed Pecan Street data and is not distributed, so
test.py refuses a regenerated file whose bytes differ, even slightly.

This wrapper is for licensed users who have regenerated the file with the
documented pipeline (forecast_regeneration/REGENERATE.md). It runs the
unmodified evaluation entry point and relaxes exactly one check: the
forecast-state input hash. Every other checkpoint/environment check in test.py
still applies. When the hashes differ, the output folder receives
APPROXIMATE_FORECAST_INPUTS.json and the results must be reported as an
approximate regeneration, not an exact reproduction.

Usage, from the project root (arguments after the target are passed through):

    python tools/evaluate_with_regenerated_forecasts.py test.py \
        --model_path results/seed_0/training/ddqn_best.pth --config_path config_seed_0.yaml \
        --output_dir eval_regenerated/cadr_s0_july --test_start 182 --test_end 212 --target_plot_days 208

    python tools/evaluate_with_regenerated_forecasts.py experiments/evaluate_matched_hems_ablation.py --seed 0 --month july --root <fresh_root>
    python tools/evaluate_with_regenerated_forecasts.py experiments/evaluate_capacity_incentive_sensitivity.py --run lev10

EBDR (benchmark/test_benchmark.py) does not check the forecast-state hash and
can be run directly; its outputs are approximate in the same sense.
"""
from __future__ import annotations

import json
import os
import runpy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FIELD = "forecast_state_input_sha256"
TARGETS = {
    "test.py": "test.py",
    "evaluate_matched_hems_ablation.py": "experiments/evaluate_matched_hems_ablation.py",
    "evaluate_capacity_incentive_sensitivity.py": "experiments/evaluate_capacity_incentive_sensitivity.py",
}


def main() -> None:
    if len(sys.argv) < 2 or Path(sys.argv[1]).name not in TARGETS:
        raise SystemExit(f"First argument must be one of {sorted(TARGETS)}")
    if Path.cwd().resolve() != ROOT:
        raise SystemExit("Run from the project root.")
    sys.path.insert(0, str(ROOT))
    import test as tester_module  # the project's test.py, not the stdlib package

    if Path(tester_module.__file__).resolve() != ROOT / "test.py":
        raise SystemExit("Could not import the project's test.py")

    cls = tester_module.ModelTester
    original_validate = cls._validate_checkpoint_metadata
    original_init = cls.__init__
    record: dict = {}

    def validate(self, checkpoint, state_dim, action_dim):
        active = self.env.get_forecast_metadata()[FIELD]
        saved = checkpoint.get(FIELD)
        if saved is not None and saved != active:
            record.update({
                "status": "APPROXIMATE_FORECAST_INPUTS",
                "checkpoint_forecast_state_sha256": saved,
                "active_forecast_state_sha256": active,
                "statement": (
                    "Evaluated with a regenerated forecast-state file whose bytes differ from the "
                    "file used to train and select this checkpoint. Only this hash check was "
                    "relaxed. Results are an approximate regeneration, not an exact reproduction."
                ),
            })
            checkpoint = dict(checkpoint)
            checkpoint[FIELD] = active
        return original_validate(self, checkpoint, state_dim, action_dim)

    def init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        if record:
            os.makedirs(self.results_dir, exist_ok=True)
            marker = Path(self.results_dir) / "APPROXIMATE_FORECAST_INPUTS.json"
            marker.write_text(json.dumps(record, indent=2), encoding="utf-8")
            print(f"WARNING: forecast-state hash differs; wrote {marker}")

    cls._validate_checkpoint_metadata = validate
    cls.__init__ = init

    target = ROOT / TARGETS[Path(sys.argv[1]).name]
    sys.argv = [str(target)] + sys.argv[2:]
    if target.name == "test.py":
        # Call the imported (patched) module; re-executing the file would redefine the class.
        tester_module.main()
    else:
        # The evaluation drivers import the same, already patched, test module.
        runpy.run_path(str(target), run_name="__main__")


if __name__ == "__main__":
    main()
