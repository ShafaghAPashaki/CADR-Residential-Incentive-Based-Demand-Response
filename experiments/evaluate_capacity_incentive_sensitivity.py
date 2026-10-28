"""Evaluate a capacity-target or incentive-discretisation run on July 2018 (days 182-212)
with the unmodified test.py.

    python experiments/evaluate_capacity_incentive_sensitivity.py --run cap6
    python experiments/evaluate_capacity_incentive_sensitivity.py --run ref7   # published Seed-0 checkpoint
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from capacity_incentive_sensitivity import OUT_ROOT, ROOT, RUNS, environment_class

sys.path.insert(0, str(ROOT))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", choices=sorted(RUNS) + ["ref7"], required=True)
    p.add_argument("--output-root", default="repro_runs/sensitivity")
    args = p.parse_args()
    if args.run == "ref7":
        model = ROOT / "results/seed_0/training/ddqn_best.pth"
        config = ROOT / "config_seed_0.yaml"
        levels = 5
    else:
        train_dir = OUT_ROOT / args.run / "seed_0" / "training"
        model, config, levels = train_dir / "ddqn_best.pth", train_dir / "config.yaml", RUNS[args.run]["levels"]
    out = Path(args.output_root) / args.run / "july"
    if out.exists():
        raise SystemExit(f"Refusing to overwrite {out}")

    import test as tester_module
    tester_module.Environment = environment_class(tester_module.Environment, levels)
    sys.argv = ["test.py", "--model_path", str(model), "--config_path", str(config),
                "--output_dir", str(out), "--test_start", "182", "--test_end", "212",
                "--target_plot_days", "208"]
    tester_module.main()


if __name__ == "__main__":
    main()
