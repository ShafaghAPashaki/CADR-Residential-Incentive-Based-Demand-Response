"""Check 14-day validation energy accounting for rho sensitivity checkpoints.
Run from the project root in the existing virtual environment; no retraining.
"""
import csv
import json
import sys
from pathlib import Path
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from env import Environment  # noqa: E402
from agent.agent_ddqn import DDQNAgent  # noqa: E402

BASE = ROOT / 'results' / 'sensitivity_rho'
OUT = BASE / 'diagnostics'
OUT.mkdir(parents=True, exist_ok=True)
RUNS = ['rho_0p1', 'rho_0p3', 'rho_0p5', 'rho_0p7', 'rho_0p9']
daily_rows = []
summary_rows = []

for run in RUNS:
    folder = BASE / 'runs' / run / 'seed_0' / 'training'
    cfg = yaml.safe_load((folder / 'config.yaml').read_text(encoding='utf-8'))
    env = Environment(data_ids=cfg['environment']['house_ids'], cfg_override=cfg)
    state = env.reset(day=168, mode='val')
    agent = DDQNAgent(len(state), env.num_actions, cfg_override=cfg,
                      action_costs=np.mean(env.all_actions, axis=1))
    checkpoint = agent.load(str(folder / 'ddqn_best.pth'), load_optimizer=False)
    if checkpoint.get('household_coefficients') is not None:
        env.set_household_coefficients(np.asarray(checkpoint['household_coefficients'], dtype=float))
    agent.policy_net.eval()
    run_rows = []
    for day in range(168, 182):
        state = env.reset(day=day, mode='val')
        for hour in range(24):
            action = int(agent.greedy_action(state))
            state, _, _, _ = env.step(action)
        raw = float(env.baseline_per_house.sum() - env.after_total_per_house.sum()
                    - env.net_curtailment_per_house.sum())
        audit = env.get_energy_audit()
        unmet = float(audit['unmet'].sum())
        corrected = raw - unmet
        record = {'run': run, 'day': day, 'raw_balance_error_kwh': raw,
                  'unmet_ev_energy_kwh': unmet, 'corrected_balance_error_kwh': corrected,
                  'device_audit_balance_error_kwh': float(audit['balance_error'].sum()),
                  'max_device_balance_abs_kwh': float(np.max(np.abs(audit['balance_error'])))}
        run_rows.append(record)
        daily_rows.append(record)
    summary = {'run': run, 'validation_days': len(run_rows),
               'max_abs_raw_error_kwh': max(abs(r['raw_balance_error_kwh']) for r in run_rows),
               'total_unmet_ev_kwh': sum(r['unmet_ev_energy_kwh'] for r in run_rows),
               'max_abs_corrected_error_kwh': max(abs(r['corrected_balance_error_kwh']) for r in run_rows),
               'max_abs_device_balance_error_kwh': max(r['max_device_balance_abs_kwh'] for r in run_rows)}
    summary_rows.append(summary)
    print('RESULT', json.dumps(summary), flush=True)

for name, rows in [('rho_daily_energy_balance.csv', daily_rows),
                   ('rho_energy_audit_summary.csv', summary_rows)]:
    with (OUT / name).open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
print('Saved diagnostics to:', OUT)
