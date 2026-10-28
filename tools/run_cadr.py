"""Train or evaluate CADR for selected seeds and test periods.

Runs the existing training and evaluation entry points in separate processes.
For published checkpoints only, evaluation uses the documented forecast-hash
wrapper; newly trained checkpoints retain the normal strict hash check.
"""
from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PERIODS = {
    "july": (182, 212, 208),
    "august": (213, 243, 220),
    "november": (305, 334, 312),
}


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="action", required=True)

    def common(q: argparse.ArgumentParser) -> None:
        q.add_argument("--seeds", type=int, nargs="+", choices=(0, 1, 2),
                       default=[0, 1, 2], help="Seeds to run (default: 0 1 2).")
        q.add_argument("--dry-run", action="store_true",
                       help="Print commands without checking inputs or running models.")

    train = sub.add_parser("train", help="Train validation-selected CADR checkpoints.")
    common(train)
    train.add_argument("--train-root", default="repro_runs/train",
                       help="New output root, relative to the project root by default.")
    train.add_argument("--regenerated-forecasts", action="store_true",
                       help="Explicitly allow a regenerated forecast-state file.")

    evaluate = sub.add_parser("evaluate", help="Evaluate published or newly trained checkpoints.")
    common(evaluate)
    evaluate.add_argument("--source", choices=("published", "trained"), default="published")
    periods = evaluate.add_mutually_exclusive_group()
    periods.add_argument("--months", nargs="+", choices=tuple(PERIODS),
                         help="Published periods (default: july august).")
    periods.add_argument("--days", nargs=2, type=int, metavar=("START", "END"),
                         help="Custom inclusive 2018 day-of-year range, within 2–365.")
    evaluate.add_argument("--plot-day", type=int,
                          help="Plot day for a custom range (default: its first day).")
    evaluate.add_argument("--train-root", default="repro_runs/train",
                          help="Location of newly trained checkpoints; used with --source trained.")
    evaluate.add_argument("--output-root", default=None,
                          help="New evaluation root (default: repro_runs/eval/<source>).")
    return p


def unique(values):
    return list(dict.fromkeys(values))


def project_path(path: str | Path) -> Path:
    return ROOT / Path(path)


def selected_periods(args):
    if args.days is not None:
        start, end = args.days
        plot = args.plot_day if args.plot_day is not None else start
        if not (2 <= start <= end <= 365 and start <= plot <= end):
            raise ValueError("Custom days must satisfy 2 <= START <= plot day <= END <= 365.")
        return [(f"days_{start}_{end}", (start, end, plot))]
    if args.plot_day is not None:
        raise ValueError("--plot-day requires --days START END.")
    return [(month, PERIODS[month]) for month in unique(args.months or ["july", "august"])]


def make_plan(args):
    """Return (commands, required files, output directories)."""
    commands, required, outputs = [], [], []
    seeds = unique(args.seeds)
    if args.action == "train":
        train_root = Path(args.train_root)
        for seed in seeds:
            cfg = Path(f"config_seed_{seed}.yaml")
            required.append(project_path(cfg))
            outputs.append(project_path(train_root / f"seed_{seed}" / "training"))
            cmd = [sys.executable, "main.py", "--config_path", str(cfg),
                   "--results_root", str(train_root)]
            if args.regenerated_forecasts:
                cmd.append("--allow_regenerated_forecasts")
            commands.append(cmd)
    else:
        base = Path("results") if args.source == "published" else Path(args.train_root)
        out_root = Path(args.output_root or f"repro_runs/eval/{args.source}")
        for seed in seeds:
            train_dir = base / f"seed_{seed}" / "training"
            model, config = train_dir / "ddqn_best.pth", train_dir / "config.yaml"
            required.extend((project_path(model), project_path(config)))
            for period, (start, end, plot) in selected_periods(args):
                out = out_root / f"seed_{seed}" / period
                outputs.append(project_path(out))
                entry = (["tools/evaluate_with_regenerated_forecasts.py", "test.py"]
                         if args.source == "published" else ["test.py"])
                commands.append([sys.executable, *entry, "--model_path", str(model),
                                 "--config_path", str(config), "--output_dir", str(out),
                                 "--test_start", str(start), "--test_end", str(end),
                                 "--target_plot_days", str(plot)])
    return commands, required, outputs


def check_plan(required, outputs):
    for path in unique(required):
        if not path.is_file():
            raise FileNotFoundError(f"Required configuration or checkpoint is missing: {path}")
    for path in outputs:
        if path.exists():
            raise FileExistsError(f"Output already exists: {path}. Choose a new output root.")


def main(argv=None) -> int:
    args = parser().parse_args(argv)
    commands, required, outputs = make_plan(args)
    if not args.dry_run:
        check_plan(required, outputs)
    else:
        print("Preview only: inputs are not checked and no models will run.", flush=True)
    for index, cmd in enumerate(commands, 1):
        print(f"[{index}/{len(commands)}] {shlex.join(cmd)}", flush=True)
        if not args.dry_run:
            subprocess.run(cmd, cwd=ROOT, check=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, FileExistsError, ValueError, subprocess.CalledProcessError) as exc:
        print(f"CADR run stopped: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
