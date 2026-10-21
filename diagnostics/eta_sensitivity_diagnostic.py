import sys, csv, json
from pathlib import Path
import numpy as np, yaml, torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from env import Environment  # noqa: E402
from agent.agent_ddqn import DDQNAgent  # noqa: E402
ETA=ROOT/'results'/'sensitivity_eta'
OUT=ETA/'diagnostics';OUT.mkdir(parents=True,exist_ok=True)
RUNS=['eta_0','eta_0p01429','eta_0p03571','eta_0p07143']

def init(r):
    p=ETA/'runs'/r/'seed_0'/'training'
    cfg=yaml.safe_load((p/'config.yaml').read_text(encoding='utf8'))
    env=Environment(data_ids=cfg['environment']['house_ids'],cfg_override=cfg)
    zero=int(np.where((env.all_actions==0).all(axis=1))[0][0])
    st=env.reset(day=168,mode='val')
    ag=DDQNAgent(len(st),env.num_actions,cfg_override=cfg,action_costs=np.mean(env.all_actions,axis=1))
    ckpt=ag.load(str(p/'ddqn_best.pth'),load_optimizer=False)
    if ckpt.get('household_coefficients') is not None: env.set_household_coefficients(np.asarray(ckpt['household_coefficients'],dtype=float))
    ag.policy_net.eval()
    return cfg,env,ag,zero
# Fixed state identification on a no-incentive trajectory, common to all eta arms.
_,env,ag,zero=init(RUNS[0])
probes=[]
for day in range(168,182):
    st=env.reset(day=day,mode='val')
    for hour in range(24):
        pre=float(env._current_pre_action_house_load(hour).sum())
        # fixed control-state identification: no-need gate on a zero-action prefix
        if pre <= 7.0+1e-9:
            probes.append((day,hour,pre))
        st,_,done,_=env.step(zero)
probes=probes[:20]
print('PROBES',len(probes),probes[:4],flush=True)
with (OUT/'controlled_state_specs.csv').open('w',newline='') as f:
    w=csv.writer(f);w.writerow(['day','hour','pre_action_kw']);w.writerows(probes)
rows=[];balances=[]; probe_rows=[]
for r in RUNS:
    cfg,env,ag,zero=init(r)
    decisions=[]
    for day,hour,pre in probes:
        st=env.reset(day=day,mode='val')
        for _ in range(hour):
            st,_,_,_=env.step(zero)
        action=int(ag.greedy_action(st)); no_need=int(np.all(env.all_actions[action]==0.0))
        decisions.append(no_need)
        probe_rows.append(dict(run=r,day=day,hour=hour,pre_action_kw=pre,action_index=action,is_zero_action=no_need))
    daily=[]
    for day in range(168,182):
        st=env.reset(day=day,mode='val')
        for hour in range(24):
            a=int(ag.greedy_action(st));st,_,done,_=env.step(a)
        raw=float(env.baseline_per_house.sum()-env.after_total_per_house.sum()-env.net_curtailment_per_house.sum())
        audit=env.get_energy_audit();unmet=float(audit['unmet'].sum());full=float(audit['balance_error'].sum());maximum=float(np.max(np.abs(audit['balance_error'])))
        rec=dict(run=r,day=day,raw_balance_error=raw,unmet_energy=unmet,corrected_balance_error=raw-unmet,device_audit_balance_error=full,max_device_balance_abs=maximum)
        daily.append(rec);balances.append(rec)
    result=dict(run=r,zero_actions=sum(decisions),total_probes=len(decisions),max_abs_raw_error=max(abs(x['raw_balance_error']) for x in daily),max_abs_corrected_error=max(abs(x['corrected_balance_error']) for x in daily),total_unmet=sum(x['unmet_energy'] for x in daily))
    rows.append(result);print('RESULT',json.dumps(result),flush=True)
for name,data in [('eta_audit_summary.csv',rows),('eta_daily_energy_balance.csv',balances),('eta_controlled_20_actions.csv',probe_rows)]:
    with (OUT/name).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(data[0]));w.writeheader();w.writerows(data)
