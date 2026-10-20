"""Evaluate a trained matched-HEMS economic-only SP ablation policy with the unmodified test.py.

From the project root:
  python eval_r1c3_r2c2_ablation.py --seed 0 --month july
  python eval_r1c3_r2c2_ablation.py --seed 0 --month august
Writes results/r1c3_r2c2_ablation/seed_<s>/<month>/ and never overwrites.
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
MONTHS = {"july": (182, 212, 208), "august": (213, 243, 220)}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, choices=(0, 1, 2), required=True)
    p.add_argument("--month", choices=sorted(MONTHS), required=True)
    p.add_argument("--root", default="results/r1c3_r2c2_ablation",
                   help="results root (use the _smoke root only for smoke checks)")
    args = p.parse_args()
    if Path.cwd().resolve() != ROOT.resolve():
        raise SystemExit("Open project root and run from that directory")
    train_dir = ROOT / args.root / f"seed_{args.seed}" / "training"
    out = ROOT / args.root / f"seed_{args.seed}" / args.month
    if out.exists():
        raise SystemExit(f"Refusing to overwrite {out}")
    start, end, plot_day = MONTHS[args.month]

    import test as tester_module
    from r1c3_r2c2_ablation import EconomicSPEnvironment
    tester_module.Environment = EconomicSPEnvironment
    sys.argv = ["test.py", "--model_path", str(train_dir / "ddqn_best.pth"),
                "--config_path", str(train_dir / "config.yaml"), "--output_dir", str(out),
                "--test_start", str(start), "--test_end", str(end),
                "--target_plot_days", str(plot_day)]
    tester_module.main()


if __name__ == "__main__":
    main()
