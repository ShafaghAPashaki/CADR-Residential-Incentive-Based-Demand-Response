"""Read-only timing of Seed-0 DDQN inference, July 2018, on local CPU.

network_only: greedy_action(state) for a fixed state, excluding environment.
policy_on_july: greedy_action(state) for each of the 744 actual July states.
policy_plus_simulator: greedy_action(state) + env.step(action); NOT deployment latency.

Run from project root: python measure_inference_local.py
"""
import argparse
import json
import os
import platform
import statistics
import sys
import time

import numpy as np
import torch
import yaml

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
from utils.final_run_protocol import require_real_holidays  # noqa: E402

require_real_holidays()
from env import Environment  # noqa: E402
from agent.agent_ddqn import DDQNAgent  # noqa: E402


def summary(ms):
    a = np.asarray(ms)
    return {"n": int(a.size), "median_ms": float(np.median(a)), "mean_ms": float(a.mean()),
            "p95_ms": float(np.percentile(a, 95))}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--threads", type=int, default=None)
    p.add_argument("--repeats", type=int, default=10000)
    args = p.parse_args()
    if args.threads:
        torch.set_num_threads(args.threads)
    cfg = yaml.safe_load(open(os.path.join(ROOT, "config_seed_0.yaml"), encoding="utf-8"))
    cfg["training"]["test_range"] = [182, 212]
    env = Environment(data_ids=cfg["environment"]["house_ids"], cfg_override=cfg)
    state = env.reset(day=182, mode="test")
    agent = DDQNAgent(len(state), env.num_actions, cfg_override=cfg,
                      action_costs=np.mean(env.all_actions, axis=1))
    agent.device = torch.device("cpu")
    agent.policy_net.to("cpu")
    ckpt = agent.load(os.path.join(ROOT, "results/seed_0/training/ddqn_best.pth"), load_optimizer=False)
    env.set_household_coefficients(np.asarray(ckpt["household_coefficients"], dtype=float))
    agent.policy_net.eval()

    for _ in range(1000):
        agent.greedy_action(state)
    net = []
    for _ in range(args.repeats):
        t = time.perf_counter(); agent.greedy_action(state); net.append(1e3 * (time.perf_counter() - t))

    full, decide = [], []
    for day in range(182, 213):
        state = env.reset(day=day, mode="test")
        done = False
        while not done:
            t = time.perf_counter()
            a = agent.greedy_action(state)
            t_mid = time.perf_counter()
            state, _, done, _ = env.step(a)
            t_end = time.perf_counter()
            decide.append(1e3 * (t_mid - t)); full.append(1e3 * (t_end - t))
    out = {"network_only": summary(net), "policy_plus_simulator": summary(full),
           "policy_on_july": summary(decide),
           "action_dim": int(env.num_actions), "state_dim": int(len(state)),
           "parameters": int(sum(x.numel() for x in agent.policy_net.parameters())),
           "torch": torch.__version__, "threads": torch.get_num_threads(),
           "cpu": platform.processor() or platform.machine(), "logical_cpus": os.cpu_count(),
           "platform": platform.platform()}
    out["notes"] = {"network_only": "Fixed-state DDQN forward + greedy tie-break; excludes simulator", "policy_on_july": "DDQN forward + greedy tie-break on each real July test state", "policy_plus_simulator": "Includes env.step (HEMS simulation, settlement, reward, next-state). Do NOT report as just inference or deployment decision latency."}
    target = os.path.join(ROOT, "local_validation", "inference_timing_seed0.json")
    os.makedirs(os.path.dirname(target), exist_ok=True)
    if os.path.exists(target):
        raise SystemExit(f"Refusing to overwrite {target}")
    with open(target, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
