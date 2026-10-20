"""Install regenerated forecasts into the project and build the DDQN forecast-state file.

Run from forecast_regeneration/ after the full-model LSTMs, cross-fitting and
verify_crossfit.py have finished (see REGENERATE.md):

    python install_outputs.py

The script:
1. refuses to continue unless crossfit_results/audit/crossfit_gate.txt reads READY;
2. copies the regenerated full-model predictions and the cross-fit outputs into
   ../results/forecast/ (local, untracked files; never commit them);
3. runs ../build_ddqn_forecast_inputs.py, the same builder used for the paper;
4. compares every stage with the SHA-256 values recorded for the published run
   and prints EXACT or APPROXIMATE for each stage.

It never edits checkpoints, configurations or recorded provenance files.
"""
from __future__ import annotations

import hashlib
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent

# SHA-256 values recorded for the run that produced the released checkpoints
# (results/forecast/ddqn_state_forecasts_provenance.json and the checkpoints).
RECORDED = {
    "load data (data/load_hourly_2018.csv)": "c183ce463f4364a9f517d4b0eab24133445f1772c12a3621023ea2cdb999947e",
    "price data (data/ercot_hourly_price.csv)": "56fdd5b489269d7a95cd6a3816b278a8b46eed02a30688e61712ce3bd8c90593",
    "full-model load predictions": "e2a149ac610ddd897a81e33007766b2538af97d2e9d1eb16d1f2481255c17c3d",
    "full-model price predictions": "c7446d11066e8cdfc394700306385449d8e67b6b1f69dc51403266d9e24bea65",
    "cross-fit training-day forecasts": "7ffbed0c3af69b9421fc534fc4998df5b1cbeef7eb3968863f7aa955e1e4d11b",
    "DDQN forecast-state file": "b86e09baae30d3093ecb4cf8efb71dd273104739fc690120807edda8df88cd17",
}

COPIES = [
    (HERE / "results/forecast/load/lstm_load_predictions_all.csv",
     ROOT / "results/forecast/load/lstm_load_predictions_all.csv"),
    (HERE / "results/forecast/price/lstm_price_predictions_all.csv",
     ROOT / "results/forecast/price/lstm_price_predictions_all.csv"),
    (HERE / "crossfit_results/combined/ddqn_crossfit_forecasts_days_002_167.csv",
     ROOT / "results/forecast/crossfit/ddqn_crossfit_forecasts_days_002_167.csv"),
    (HERE / "crossfit_results/audit/crossfit_gate.txt",
     ROOT / "results/forecast/crossfit/crossfit_gate.txt"),
]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    gate = HERE / "crossfit_results/audit/crossfit_gate.txt"
    if not gate.exists() or gate.read_text(encoding="utf-8").strip() != "READY":
        raise SystemExit("Cross-fit gate is not READY. Run verify_crossfit.py first.")
    for src, _ in COPIES:
        if not src.exists():
            raise SystemExit(f"Missing regenerated output: {src}")

    for src, dst in COPIES:
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
    subprocess.run([sys.executable, str(ROOT / "build_ddqn_forecast_inputs.py")], cwd=ROOT, check=True)

    actual = {
        "load data (data/load_hourly_2018.csv)": HERE / "data/load_hourly_2018.csv",
        "price data (data/ercot_hourly_price.csv)": HERE / "data/ercot_hourly_price.csv",
        "full-model load predictions": COPIES[0][1],
        "full-model price predictions": COPIES[1][1],
        "cross-fit training-day forecasts": COPIES[2][1],
        "DDQN forecast-state file": ROOT / "results/forecast/ddqn_state_forecasts.csv",
    }
    print("\nStage comparison with the published run")
    exact_all = True
    for stage, recorded in RECORDED.items():
        got = sha256(actual[stage])
        same = got == recorded
        exact_all &= same
        print(f"  {'EXACT      ' if same else 'APPROXIMATE'}  {stage}: {got[:16]}... (recorded {recorded[:16]}...)")
    if exact_all:
        print("\nThe forecast-state file is byte-identical to the published one. "
              "Evaluate the released checkpoints with test.py directly.")
    else:
        print("\nThe forecast-state file differs from the published one. Released checkpoints can only be "
              "evaluated approximately, with tools/evaluate_with_regenerated_forecasts.py, and results must "
              "be reported as an approximate regeneration. See REPRODUCIBILITY.md.")


if __name__ == "__main__":
    main()
