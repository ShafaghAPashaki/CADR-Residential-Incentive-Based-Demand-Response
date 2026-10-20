"""Train one predeclared R2 sensitivity run (Seed 0, 2500 episodes, validation-only selection).

    python run_r2_sensitivity.py --run cap6 [--dry-run]

Writes only to results/r2_sensitivity/<run>/ and refuses to touch an existing folder.
Records wall-clock training time (including the 50 validation passes) in timing.json.
"""
import argparse
import datetime
import json
import os
import platform
import sys
import time

import yaml

from r2_common import (OUT_ROOT, PROTOCOL_ID, ROOT, RUNS, build_config,
                       capacity_gate, environment_class, sha256_file)

sys.path.insert(0, str(ROOT))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", choices=sorted(RUNS), required=True)
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()
    spec = RUNS[args.run]
    run_dir = OUT_ROOT / args.run
    if run_dir.exists() and any(run_dir.iterdir()):
        raise SystemExit(f"Refusing to reuse existing folder: {run_dir}")
    cfg = build_config(args.run)
    print("PLAN" if args.dry_run else "TRAIN", args.run, spec, "->", run_dir)
    if args.dry_run:
        return

    import torch
    import main as trainer_module                      # runs the holidays==0.100 gate
    from utils.final_run_protocol import enforce_lambda_calibration
    trainer_module.enforce_lambda_calibration = capacity_gate(enforce_lambda_calibration, spec["capacity_kw"])
    trainer_module.Environment = environment_class(trainer_module.Environment, spec["levels"])

    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "run_config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")

    trainer_module.set_global_seeds(seed=0)
    start = datetime.datetime.now(datetime.timezone.utc)
    t0 = time.perf_counter()
    trainer = trainer_module.DDQNTrainer(cfg_override=cfg)
    t_init = time.perf_counter()
    trainer.train()
    t1 = time.perf_counter()
    end = datetime.datetime.now(datetime.timezone.utc)

    timing = {
        "protocol": PROTOCOL_ID, "run": args.run, **spec,
        "action_dim": int(trainer.action_dim), "state_dim": int(trainer.state_dim),
        "episodes": int(cfg["DDQN"]["episodes"]),
        "start_utc": start.isoformat(), "end_utc": end.isoformat(),
        "setup_seconds": t_init - t0,
        "train_call_seconds_including_validation": t1 - t_init,
        "total_wall_clock_seconds": t1 - t0,
        "device": str(trainer.device), "torch_version": torch.__version__,
        "torch_num_threads": torch.get_num_threads(),
        "cpu": platform.processor() or platform.machine(), "logical_cpus": os.cpu_count(),
        "platform": platform.platform(), "python": platform.python_version(),
        "driver_sha256": {f: sha256_file(ROOT / f) for f in ("run_r2_sensitivity.py", "r2_common.py")},
        "note": "Record whether any other heavy process ran concurrently.",
    }
    (run_dir / "timing.json").write_text(json.dumps(timing, indent=2), encoding="utf-8")
    print(json.dumps(timing, indent=2))


if __name__ == "__main__":
    main()
