"""Evaluate a run on July 2018 (days 182-212) with the unmodified test.py.

    python eval_r2_july.py --run cap6
    python eval_r2_july.py --run ref7      # published Seed-0 checkpoint, re-evaluated with current test.py
"""
import argparse
import sys

from r2_common import OUT_ROOT, ROOT, RUNS, environment_class

sys.path.insert(0, str(ROOT))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", choices=sorted(RUNS) + ["ref7"], required=True)
    args = p.parse_args()
    if args.run == "ref7":
        model = ROOT / "results/seed_0/training/ddqn_best.pth"
        config = ROOT / "config_seed_0.yaml"
        levels = 5
    else:
        train_dir = OUT_ROOT / args.run / "seed_0" / "training"
        model, config, levels = train_dir / "ddqn_best.pth", train_dir / "config.yaml", RUNS[args.run]["levels"]
    out = OUT_ROOT / args.run / "july"
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
