"""Frozen, inference-only forecast versus oracle value-of-information comparison.

This evaluator NEVER updates a network, changes a checkpoint or selects a
model. Evaluation windows and metric definitions are fixed before final tests.
Run only against newly trained checkpoints from this project's final configs.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
import yaml

from agent.agent_ddqn import DDQNAgent
from utils.final_run_protocol import collect_run_provenance, require_real_holidays

ROOT = Path(__file__).resolve().parent
PERIODS = {"july": (182,212), "august": (213,243), "november": (305,334)}
EPS=1e-9


def load_environment(method, cfg):
    if method == 'cadr':
        from env.env import Environment
    else:
        from benchmark.env_benchmark import Environment
    return Environment(cfg['environment']['house_ids'], cfg_override=cfg)


def evaluate_one(method, cfg, checkpoint, mode, period):
    env=load_environment(method, cfg)
    if env.VERSION != checkpoint.get('environment_version'):
        raise ValueError('Checkpoint environment version mismatch')
    if checkpoint.get('forecast_state_mode') not in (None, 'forecast'):
        raise ValueError('Only a forecast-trained production policy may be evaluated')
    if method == 'cadr' and 'household_coefficients' in checkpoint:
        coef=checkpoint['household_coefficients']
        if isinstance(coef, torch.Tensor): coef=coef.cpu().numpy()
        env.set_household_coefficients(coef)
    state=env.reset(day=PERIODS[period][0],mode='test')
    agent=DDQNAgent(len(state), env.num_actions, cfg_override=cfg,
                    action_costs=np.mean(env.all_actions,axis=1))
    agent.policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
    agent.policy_net.eval()
    # Controlled inference-only change: keep configuration and policy fixed.
    env.forecast_state_mode=mode
    rows=[]
    for day in range(PERIODS[period][0],PERIODS[period][1]+1):
        state=env.reset(day=day,mode='test')
        rewards=[]
        done=False
        with torch.no_grad():
            while not done:
                action=int(agent.greedy_action(state))
                state,r,done,_=env.step(action)
                rewards.append(float(r))
        before=env.baseline_per_house.sum(axis=1)
        if method=='cadr':
            after=env.after_total_per_house.sum(axis=1)
            shifted_in=env.shifted_in_per_house.sum(axis=1)
            rebound=(shifted_in>EPS)&(before<=7+EPS)&(after>7+EPS)&(after-shifted_in<=7+EPS)
            conflict_in=env.device_conflict_deferred_in.sum(axis=(1,2))
            # Attribution counterfactual: remove only conflict-deferred arrivals.
            # Categories may overlap if both arrival types contribute to a peak.
            conflict_rebound=(conflict_in>EPS)&(before<=7+EPS)&(after>7+EPS)&(after-conflict_in<=7+EPS)
            unmet=float(env.device_unmet.sum())
            conflict_unserved=float(env.device_conflict_unserved.sum())
            forced=float(env.device_conflict_deferred_out.sum())
            forced_received=float(env.device_conflict_deferred_in.sum())
            conflict_discomfort=float(env.device_conflict_discomfort.sum())
            above=float(env.device_settlement_above_price_energy.sum())
            above_device_hours=int(env.device_settlement_rate_above_price.sum())
            margin=float(env.device_settlement_negative_margin.sum())

            audit=env.get_energy_audit()
            max_abs_energy_balance_error=float(
                np.max(np.abs(audit['balance_error']), initial=0.0)
            )

            ev_rating_violations=0
            if env.idx_car is not None:
                d=env.idx_car
                limit=float(env.DEVICE_CONSUMPTION[d])
                ev_rating_violations=int(
                    (env.device_after[:,:,d] > limit + env.EPS).sum()
                )

            tsni_rating_violations=0
            for d in np.where(env.TS_NI_MASK)[0]:
                limit=float(env.DEVICE_CONSUMPTION[d])
                tsni_rating_violations += int(
                    (env.device_after[:,:,d] > limit + env.EPS).sum()
                )

            remaining_original=np.maximum(
                env.device_demands
                - env.device_shifted_out
                - env.device_conflict_deferred_out,
                0.0,
            )
            scheduled=env.device_shifted_in + env.device_conflict_deferred_in
            tsni=env.TS_NI_MASK[None,None,:]
            overlap_original_scheduled=(
                (remaining_original > env.EPS)
                & (scheduled > env.EPS)
                & tsni
            )
            overlap_two_scheduled=(
                (env.device_shifted_in > env.EPS)
                & (env.device_conflict_deferred_in > env.EPS)
                & tsni
            )
            tsni_double_booking_count=int(
                overlap_original_scheduled.sum() + overlap_two_scheduled.sum()
            )
        else:
            after=env.after_per_house.sum(axis=1)
            rebound=np.zeros(len(after),dtype=bool)
            conflict_rebound=np.zeros(len(after),dtype=bool)
            unmet=conflict_unserved=forced=forced_received=0.0
            conflict_discomfort=above=margin=0.0
            above_device_hours=0
            ev_rating_violations=0
            tsni_rating_violations=0
            tsni_double_booking_count=0
            max_abs_energy_balance_error=float(
                np.max(
                    np.abs(
                        env.baseline_per_house.sum(axis=1)
                        - env.after_per_house.sum(axis=1)
                        - env.reductions.sum(axis=1)
                    ),
                    initial=0.0,
                )
            )
        capacity=7.0  # descriptive common 7-kW reporting reference only
        rows.append({
            'method':method, 'seed':int(cfg['general']['seed']),
            'period':period,'day':day,'state_mode':mode,
            'reward_sum':sum(rewards),
            'cumulative_excess_kwh':float(np.maximum(after-capacity,0).sum()),
            'exceedance_hours':int(np.sum(after>capacity+EPS)),
            'peak_kw':float(after.max()),
            'par':float(after.max()/after.mean()),
            'rebound_induced_hours':int(rebound.sum()),
            'rebound_induced_excess_kwh':float(np.maximum(after-capacity,0)[rebound].sum()),
            'conflict_induced_rebound_hours':int(conflict_rebound.sum()),
            'conflict_induced_rebound_excess_kwh':float(np.maximum(after-capacity,0)[conflict_rebound].sum()),
            'unserved_energy_kwh':unmet,
            'unserved_conflict_energy_kwh':conflict_unserved,
            'forced_conflict_deferral_kwh':forced,
            'forced_conflict_deferral_received_kwh':forced_received,
            'conflict_discomfort':conflict_discomfort,
            'settlement_above_price_device_hours':above_device_hours,
            'settlement_above_price_energy_kwh':above,
            'settlement_negative_margin':margin,
            'ev_rating_violations':ev_rating_violations,
            'tsni_rating_violations':tsni_rating_violations,
            'tsni_double_booking_count':tsni_double_booking_count,
            'max_abs_energy_balance_error':max_abs_energy_balance_error,
            'incentive_payment':float(env.incentive_payment.sum()),
            'sp_value':float(env.rewards_service_provider.sum()),
            'eu_utility':float(env.rewards_customers.sum()),
        })
    return rows


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--method',choices=['cadr','ebdr'],required=True)
    parser.add_argument('--seed',type=int,choices=[0,1,2],required=True)
    parser.add_argument('--period',choices=PERIODS,required=True)
    args=parser.parse_args()
    require_real_holidays()
    if args.method=='cadr':
        path=ROOT/f'config_seed_{args.seed}.yaml'
        ckpt=ROOT/f'results/seed_{args.seed}/training/ddqn_best.pth'
    else:
        path=ROOT/f'benchmark/config_benchmark_seed_{args.seed}.yaml'
        ckpt=ROOT/f'results/benchmark/seed_{args.seed}/training/ddqn_best.pth'
    cfg=yaml.safe_load(path.read_text(encoding='utf-8'))
    checkpoint=torch.load(ckpt,map_location='cpu',weights_only=False)
    rows=[]
    for mode in ('forecast','oracle_actual'):
        rows.extend(evaluate_one(args.method,cfg,checkpoint,mode,args.period))
    out=ROOT/'results'/'final_diagnostics'/args.method/f'seed_{args.seed}'
    out.mkdir(parents=True,exist_ok=True)
    destination=out/f'value_of_information_{args.period}.csv'
    if destination.exists():
        raise FileExistsError(f'Immutable evaluation output already exists: {destination}')
    with destination.open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    print(f'Wrote {destination}; inference only, no policy update or tuning')


if __name__=='__main__':
    main()
