"""Run all four ETA configurations into a new isolated rerun directory."""
from __future__ import annotations

import argparse
import datetime as dt
import subprocess
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]
RUNNER = BASE / "scripts" / "run_sensitivity.py"
ARMS = ("eta_0p00000", "eta_0p01429", "eta_0p03571", "eta_0p07143")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", default=dt.datetime.now().strftime("%Y%m%d_%H%M%S"))
    parser.add_argument("--seed", type=int, default=None, help="Optional seed override for all four runs.")
    args = parser.parse_args()

    root = BASE / "reruns" / args.tag
    for slug in ARMS:
        command = [
            sys.executable,
            str(RUNNER),
            "--config_path",
            str(BASE / "configs" / f"{slug}.yaml"),
            "--output_dir",
            str(root / slug / "training"),
        ]
        if args.seed is not None:
            command.extend(["--seed", str(args.seed)])
        print("Running:", " ".join(command), flush=True)
        subprocess.run(command, check=True)

    print(f"Completed all ETA runs in: {root}")


if __name__ == "__main__":
    main()
