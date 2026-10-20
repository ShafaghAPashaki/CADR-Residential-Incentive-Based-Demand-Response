"""Train the predeclared matched-HEMS economic-only SP ablation.

From project root:
  python run_r1c3_r2c2_ablation.py --dry-run
  python run_r1c3_r2c2_ablation.py --smoke
  python run_r1c3_r2c2_ablation.py --seed 0
  python run_r1c3_r2c2_ablation.py --seed 1
  python run_r1c3_r2c2_ablation.py --seed 2

Never overwrites existing output directories.
"""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import platform
import sys
import time
import yaml

ROOT = Path(__file__).resolve().parent
PROTOCOL = 'MATCHED_HEMS_SP_ABLATION_2026_10_15_V1'


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def config_for(seed, smoke=False):
    path = ROOT / f'config_seed_{seed}.yaml'
    cfg = yaml.safe_load(path.read_text(encoding='utf-8'))
    assert int(cfg['general']['seed']) == seed
    assert float(cfg['environment']['capacity_threshold']) == 7.0
    assert cfg['environment']['reward_mode'] == 'weighted_stakeholder_tracking_no_need'
    assert int(cfg['DDQN']['episodes']) == 2500
    assert cfg['training']['val_range'] == [168,181]
    cfg['general']['results_root'] = ('results/r1c3_r2c2_ablation/_smoke' if smoke
                                        else 'results/r1c3_r2c2_ablation')
    if smoke:
        cfg['DDQN']['episodes'] = 20
        cfg['DDQN']['eval_interval'] = 10
    cfg['matched_hems_sp_ablation'] = {'protocol': PROTOCOL, 'arm':'economic_only_sp',
      'description':'Same capacity-aware HEMS, economic-only SP policy; two need features masked',
      'original_configuration':path.name}
    return cfg


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--seed', type=int, choices=(0,1,2))
    p.add_argument('--dry-run', action='store_true')
    p.add_argument('--smoke', action='store_true')
    args = p.parse_args()
    if args.seed is None and not args.dry_run:
        p.error('specify --seed 0/1/2 or --dry-run')
    if Path.cwd().resolve() != ROOT.resolve():
        raise SystemExit('Open project root and run from that directory')
    seeds = (0,1,2) if args.dry_run else (args.seed,)
    for seed in seeds:
        cfg = config_for(seed, args.smoke)
        print('PLAN',seed, cfg['general']['results_root'], 'episodes',cfg['DDQN']['episodes'])
        if args.dry_run: continue
        import torch
        import main as train_module
        from r1c3_r2c2_ablation import EconomicSPEnvironment
        train_module.Environment = EconomicSPEnvironment
        train_module.set_global_seeds(seed=seed)
        output = ROOT / cfg['general']['results_root']
        run_dir = output / f'seed_{seed}' / 'training'
        if run_dir.exists():
            raise SystemExit(f'Output already exists: {run_dir}')
        # Frozen preregistration must be present before real training.
        prereg = ROOT / 'R1C3_R2C2_ABLATION_PREREGISTRATION.md'
        if not args.smoke and not prereg.is_file():
            raise SystemExit('Missing frozen preregistration')
        output.mkdir(parents=True, exist_ok=True)
        (output / f'run_config_seed_{seed}.yaml').write_text(yaml.safe_dump(cfg,sort_keys=False),encoding='utf-8')
        t0=time.perf_counter()
        start=datetime.datetime.now(datetime.timezone.utc).isoformat()
        trainer=train_module.DDQNTrainer(cfg_override=cfg)
        trainer.train()
        end=datetime.datetime.now(datetime.timezone.utc).isoformat()
        timing={'protocol':PROTOCOL,'seed':seed,'start_utc':start,'end_utc':end,
           'elapsed_seconds':time.perf_counter()-t0,'device':str(trainer.device),
           'python':platform.python_version(),'torch':torch.__version__,
           'original_env_sha256':digest(ROOT/'env/env.py'),
           'ablation_sha256':digest(ROOT/'r1c3_r2c2_ablation.py'),
           'driver_sha256':digest(ROOT/'run_r1c3_r2c2_ablation.py')}
        (output / f'timing_seed_{seed}.json').write_text(json.dumps(timing,indent=2),encoding='utf-8')
        print('COMPLETE',seed, timing['elapsed_seconds'])

if __name__ == '__main__': main()
