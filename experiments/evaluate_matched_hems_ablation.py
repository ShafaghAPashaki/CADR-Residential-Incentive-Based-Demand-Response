"""Evaluate a trained matched-HEMS ablation (CBSP) policy with the unmodified test.py.

From the project root:
  python experiments/evaluate_matched_hems_ablation.py --seed 0 --month july
  python experiments/evaluate_matched_hems_ablation.py --seed 0 --month august
Writes <root>/seed_<s>/<month>/ (default root results/matched_hems_ablation) and never overwrites.
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(1, str(Path(__file__).resolve().parent))
MONTHS = {"july": (182, 212, 208), "august": (213, 243, 220)}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, choices=(0, 1, 2), required=True)
    p.add_argument("--month", choices=sorted(MONTHS), required=True)
    p.add_argument("--root", default="results/matched_hems_ablation",
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
    from matched_hems_ablation import EconomicSPEnvironment
    tester_module.Environment = EconomicSPEnvironment
    sys.argv = ["test.py", "--model_path", str(train_dir / "ddqn_best.pth"),
                "--config_path", str(train_dir / "config.yaml"), "--output_dir", str(out),
                "--test_start", str(start), "--test_end", str(end),
                "--target_plot_days", str(plot_day)]
    tester_module.main()


if __name__ == "__main__":
    main()
