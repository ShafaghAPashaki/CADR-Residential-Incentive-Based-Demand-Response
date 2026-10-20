"""Final-code single-seed parameter sensitivities. Run from project root.

python run_sensitivity.py --group eta
python run_sensitivity.py --group rho
python run_sensitivity.py --group both
python run_sensitivity.py --group both --dry-run
"""
import argparse
import copy
import csv
import hashlib
import json
from pathlib import Path
import sys

import yaml

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

VALUES = {"eta": [0, 0.01429, 0.03571, 0.07143],
          "rho": [0.1, 0.3, 0.5, 0.7, 0.9]}

FIELDS = ["parameter", "value", "seed", "best_episode", "validation_days",
          "violation_steps", "cumulative_excess_kwh", "useful_dr_kwh",
          "tracking_mae_kwh", "unnecessary_offers", "incentive_payment_cent",
          "raw_dissatisfaction", "best_mean_validation_reward", "forecast_state_mode",
          "source_config_sha256"]


def dirname(group, value):
    return ("eta_" if group == "eta" else "rho_") + str(value).replace('.', 'p')


def configure(group, value):
    source = ROOT / 'config_seed_0.yaml'
    raw = source.read_bytes()
    cfg = yaml.safe_load(raw)
    cfg['general']['seed'] = 0
    cfg['DDQN']['episodes'] = 2500
    cfg['DDQN']['eval_interval'] = 50
    cfg['training']['train_ranges'] = [[2, 167]]
    cfg['training']['val_range'] = [168, 181]
    # Preserve all other final-method parameters and forecast source verbatim.
    cfg.setdefault('forecast', {})['state_input_mode'] = 'forecast'
    if not cfg['forecast'].get('ddqn_state_input_file'):
        raise RuntimeError('Final config lacks the DDQN forecast state file')
    if group == 'eta':
        cfg['environment']['no_need_offer_regularization']['enabled'] = value > 0
        cfg['environment']['no_need_offer_regularization']['coefficient'] = float(value)
        # The final environment explicitly forbids the *_no_need reward mode
        # when eta == 0. Use its mathematically equivalent no-penalty mode.
        cfg['environment']['reward_mode'] = (
            'weighted_stakeholder_tracking_no_need' if value > 0
            else 'weighted_stakeholder_tracking'
        )
    else:
        cfg['environment']['rho'] = float(value)
    rel = f'results/sensitivity_{group}/runs/{dirname(group, value)}'
    cfg['general']['results_root'] = rel
    return cfg, hashlib.sha256(raw).hexdigest()


def check_inputs(cfg):
    for field in ('load_file', 'price_file'):
        name = cfg['general'][field]
        if not (ROOT / name).is_file():
            raise FileNotFoundError(f'Missing {field}: {name}')
    fore = cfg['forecast']
    if fore['state_input_mode'] != 'forecast':
        raise ValueError('state_input_mode must be forecast')
    if not (ROOT / fore['ddqn_state_input_file']).is_file():
        raise FileNotFoundError(f"Missing forecast state: {fore['ddqn_state_input_file']}")
    prov = fore.get('ddqn_state_provenance_file')
    if prov and not (ROOT / prov).is_file():
        raise FileNotFoundError(f'Missing forecast provenance: {prov}')


def selected_row(group, value, sha, cfg):
    run_root = ROOT / cfg['general']['results_root']
    train = run_root / 'seed_0' / 'training'
    csvfile = train / 'validation_metrics.csv'
    if not csvfile.exists():
        return None
    with csvfile.open(newline='', encoding='utf-8') as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        return None
    # Match main.py: max mean_reward, min std_reward, earliest episode.
    best = sorted(rows, key=lambda r: (-float(r['mean_reward']), float(r['std_reward']), int(r['episode'])))[0]
    if len(rows) != 50 or int(rows[-1]['episode']) != 2500:
        raise ValueError(f'Incomplete validation history: {csvfile} ({len(rows)} rows)')
    pick = lambda key: best.get(key, '')
    return {'parameter': group, 'value': value, 'seed': 0,
            'best_episode': int(best['episode']), 'validation_days': pick('validation_days'),
            'violation_steps': pick('violation_steps'),
            'cumulative_excess_kwh': pick('cumulative_excess'),
            'useful_dr_kwh': pick('useful_settlement_dr_energy'),
            'tracking_mae_kwh': pick('target_tracking_mae_kwh'),
            'unnecessary_offers': pick('unnecessary_incentive_steps'),
            'incentive_payment_cent': pick('incentive_payment'),
            'raw_dissatisfaction': pick('raw_discomfort'),
            'best_mean_validation_reward': pick('mean_reward'),
            'forecast_state_mode': cfg['forecast']['state_input_mode'],
            'source_config_sha256': sha}


def write_summary(group):
    records = []
    for value in VALUES[group]:
        cfg, sha = configure(group, value)
        rec = selected_row(group, value, sha, cfg)
        if rec:
            records.append(rec)
    target = ROOT / 'results' / f'sensitivity_{group}' / 'summary.csv'
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(records)
    print(f'{target.relative_to(ROOT)} ({len(records)}/{len(VALUES[group])} complete)')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--group', choices=['eta', 'rho', 'both'], required=True)
    p.add_argument('--dry-run', action='store_true')
    args = p.parse_args()
    groups = ['eta', 'rho'] if args.group == 'both' else [args.group]
    for group in groups:
        for value in VALUES[group]:
            cfg, sha = configure(group, value)
            check_inputs(cfg)
            output = ROOT / cfg['general']['results_root'] / 'seed_0' / 'training'
            rec = selected_row(group, value, sha, cfg)
            if rec:
                print('SKIP completed:', group, value)
                continue
            if output.exists() and any(output.iterdir()):
                raise RuntimeError(f'Incomplete existing run: {output}\nMove this folder aside manually to restart; nothing was overwritten.')
            cfgfile = ROOT / cfg['general']['results_root'] / 'run_config.yaml'
            print('PLAN' if args.dry_run else 'TRAIN', group, value, '->', output)
            if args.dry_run:
                continue
            cfgfile.parent.mkdir(parents=True, exist_ok=True)
            cfgfile.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding='utf-8')
            from main import DDQNTrainer, set_global_seeds
            from utils.final_run_protocol import require_real_holidays, enforce_lambda_calibration
            require_real_holidays()
            enforce_lambda_calibration(cfg)
            set_global_seeds(seed=0)
            DDQNTrainer(cfg_override=copy.deepcopy(cfg)).train()
            if not selected_row(group, value, sha, cfg):
                raise RuntimeError('Run finished but results are incomplete')
            write_summary(group)
        if not args.dry_run:
            write_summary(group)


if __name__ == '__main__':
    main()
