"""Diagnostic-only audit of learned DDQN Q-value ordering.

This script does not train or modify a model.  It evaluates all joint actions
from cloned validation states and compares:
  * exact immediate reward,
  * zero-action continuation return,
  * greedy-policy continuation return,
  * policy/target-network Bellman quantities,
  * next-state differences relative to the zero action.

The goal is to distinguish a reward near-tie from a genuine learned-Q
misranking or a next-state/temporal-credit effect.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional

import numpy as np
import pandas as pd
import torch
import yaml


# Allow this script to be launched from any working directory.
PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from env import Environment
from reward_counterfactual_audit import (
    EPS,
    _maximum_action_index,
    _zero_action_index,
    evaluate_action,
    set_global_seeds,
)

from agent.agent_ddqn import DDQNAgent


SCRIPT_VERSION = "final_ddqn_q_diagnostic"
AUDIT_TOL = 1e-9


@dataclass(frozen=True)
class StateSpec:
    state_id: str
    day: int
    hour: int
    prefix_kind: str
    source_pool: str
    selection_tags: str
    illustrative_only: bool = False


def load_yaml(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    if not isinstance(cfg, dict):
        raise TypeError(f"Configuration is not a dictionary: {path}")
    return cfg


def safe_torch_load(path: str, device: torch.device) -> dict:
    try:
        checkpoint = torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(path, map_location=device)
    if not isinstance(checkpoint, dict):
        raise TypeError("Checkpoint must be a dictionary.")
    if "policy_net_state_dict" not in checkpoint:
        raise KeyError("Checkpoint does not contain policy_net_state_dict.")
    return checkpoint


def stable_hash(values: Iterable[float], decimals: int = 8) -> str:
    rounded = np.round(np.asarray(list(values), dtype=float), decimals=decimals)
    return hashlib.sha256(rounded.tobytes()).hexdigest()[:16]


def greedy_action(agent: DDQNAgent, state: np.ndarray) -> int:
    return agent.greedy_action(state)


def q_values(agent: DDQNAgent, state: np.ndarray) -> np.ndarray:
    tensor = torch.as_tensor(
        state, dtype=torch.float32, device=agent.device
    ).unsqueeze(0)
    with torch.no_grad():
        values = agent.policy_net(tensor).squeeze(0)
    return values.detach().cpu().numpy().astype(float)


def build_environment(cfg: dict, checkpoint: dict) -> Environment:
    house_ids = list(cfg["environment"]["house_ids"])
    env = Environment(data_ids=house_ids, cfg_override=cfg)
    coefficients = checkpoint.get("household_coefficients")
    if coefficients is not None:
        if torch.is_tensor(coefficients):
            coefficients = coefficients.detach().cpu().numpy()
        env.set_household_coefficients(np.asarray(coefficients, dtype=float))
    return env


def build_agent(cfg: dict, env: Environment, checkpoint_path: str) -> tuple[DDQNAgent, dict]:
    initial_state = env.reset(mode="val")
    agent = DDQNAgent(
        len(initial_state),
        env.num_actions,
        cfg_override=cfg,
        action_costs=np.mean(env.all_actions, axis=1),
    )
    checkpoint = agent.load(checkpoint_path, load_optimizer=False)
    agent.policy_net.eval()
    agent.target_net.eval()
    return agent, checkpoint


def validate_provenance(
    cfg: dict,
    env: Environment,
    checkpoint: dict,
    allow_reward_override: bool = False,
) -> None:
    expected_state_dim = len(env.reset(mode="val"))
    expected_action_dim = env.num_actions
    checks = {
        "state_dim": (checkpoint.get("state_dim"), expected_state_dim),
        "action_dim": (checkpoint.get("action_dim"), expected_action_dim),
        "reward_mode": (checkpoint.get("reward_mode"), env.reward_mode),
        "reward_definition_version": (
            checkpoint.get("reward_definition_version"),
            env.REWARD_DEFINITION_VERSION,
        ),
    }
    hard_mismatches = []
    reward_mismatches = []
    for name, (saved, active) in checks.items():
        if saved is None or saved == active:
            continue
        target = (
            reward_mismatches
            if name in {"reward_mode", "reward_definition_version"}
            else hard_mismatches
        )
        target.append((name, saved, active))

    if hard_mismatches:
        details = "\n".join(
            f"  {name}: checkpoint={saved!r}, active={active!r}"
            for name, saved, active in hard_mismatches
        )
        raise ValueError(f"Checkpoint/config mismatch:\n{details}")

    if reward_mismatches and not allow_reward_override:
        details = "\n".join(
            f"  {name}: checkpoint={saved!r}, active={active!r}"
            for name, saved, active in reward_mismatches
        )
        raise ValueError(
            "Reward-definition mismatch. Use --allow_reward_override only for "
            "pre-training counterfactual evaluation of a frozen checkpoint:\n"
            + details
        )
    if reward_mismatches:
        print("[COUNTERFACTUAL] Reward override accepted:")
        for name, saved, active in reward_mismatches:
            print(f"  {name}: checkpoint={saved!r} -> active={active!r}")


def per_house_flexibility(env: Environment, hour: int) -> dict[str, np.ndarray]:
    max_pc_rate = float(np.max(env.POWER_RATE, initial=0.0))
    pc_raw = env._current_pc_available_per_house(hour)
    pc_reducible = np.asarray(pc_raw, dtype=float) * max_pc_rate
    ev = np.asarray(env._current_ev_available_per_house(hour), dtype=float)
    tsni = np.asarray(
        env._current_tsni_startable_energy_per_house(hour), dtype=float
    )
    total = pc_reducible + ev + tsni
    return {
        "pc_reducible": pc_reducible,
        "ev": ev,
        "tsni": tsni,
        "total": total,
    }


def need_bucket(need: float, capacity: float) -> str:
    if need <= EPS:
        return "none"
    if need <= 0.10 * capacity:
        return "slight"
    if need <= 0.30 * capacity:
        return "moderate"
    return "severe"


def flex_tier(value: float, deadband: float, capacity: float) -> str:
    if value <= EPS:
        return "zero"
    threshold = max(float(deadband), 0.10 * capacity)
    if value <= threshold:
        return "low"
    return "high"


def collect_zero_prefix_validation_catalog(env: Environment) -> pd.DataFrame:
    start, end = [int(v) for v in env.val_range]
    zero = _zero_action_index(env)
    rows: list[dict[str, Any]] = []
    for day in range(start, end + 1):
        env.reset(day=day, mode="val")
        done = False
        while not done:
            h = int(env.curr_step)
            pre_house = np.asarray(env._current_pre_action_house_load(h), dtype=float)
            pre_total = float(pre_house.sum())
            needed = max(0.0, pre_total - env.capacity_threshold)
            flex = per_house_flexibility(env, h)
            prior_relief, prior_payment = env._current_settlement_commitments_per_house(h)
            row: dict[str, Any] = {
                "day": day,
                "hour": h,
                "prefix_kind": "zero",
                "pre_action_load": pre_total,
                "needed_reduction": needed,
                "price": float(env.prices[h]),
                "capacity": float(env.capacity_threshold),
                "need_bucket": need_bucket(needed, env.capacity_threshold),
                "total_flexibility": float(flex["total"].sum()),
                "prior_committed_relief": float(np.sum(prior_relief)),
                "prior_locked_payment": float(np.sum(prior_payment)),
            }
            for i, house_id in enumerate(env.data_ids):
                row[f"pre_load_house_{house_id}"] = float(pre_house[i])
                row[f"pc_flex_house_{house_id}"] = float(flex["pc_reducible"][i])
                row[f"ev_flex_house_{house_id}"] = float(flex["ev"][i])
                row[f"tsni_flex_house_{house_id}"] = float(flex["tsni"][i])
                row[f"total_flex_house_{house_id}"] = float(flex["total"][i])
                row[f"flex_tier_house_{house_id}"] = flex_tier(
                    float(flex["total"][i]),
                    env.track_deadband_kwh,
                    env.capacity_threshold,
                )
            rows.append(row)
            _, _, done, _ = env.step(zero)
    return pd.DataFrame(rows)


def create_state_manifest(
    catalog: pd.DataFrame,
    house_ids: list[int],
    samples_per_stratum: int,
    seed: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    selections: list[dict[str, Any]] = []


    boundary_rules = {
        "lowest_load": catalog["pre_action_load"].idxmin(),
        "highest_price_no_need": catalog.loc[
            catalog["needed_reduction"] <= EPS, "price"
        ].idxmax(),
        "smallest_positive_need": catalog.loc[
            catalog["needed_reduction"] > EPS, "needed_reduction"
        ].idxmin(),
        "largest_need": catalog["needed_reduction"].idxmax(),
        "largest_flex_no_need": catalog.loc[
            catalog["needed_reduction"] <= EPS, "total_flexibility"
        ].idxmax(),
    }
    for tag, idx in boundary_rules.items():
        row = catalog.loc[idx]
        selections.append({
            "day": int(row.day),
            "hour": int(row.hour),
            "prefix_kind": "zero",
            "source_pool": "boundary_validation",
            "selection_tag": tag,
            "illustrative_only": False,
        })


    long_rows: list[dict[str, Any]] = []
    for row in catalog.itertuples(index=False):
        data = row._asdict()
        for house_id in house_ids:
            long_rows.append({
                "day": int(data["day"]),
                "hour": int(data["hour"]),
                "need_bucket": str(data["need_bucket"]),
                "house_id": int(house_id),
                "flex_tier": str(data[f"flex_tier_house_{house_id}"]),
            })
    long_df = pd.DataFrame(long_rows)
    for key, group in long_df.groupby(
        ["need_bucket", "house_id", "flex_tier"], sort=True
    ):
        if group.empty:
            continue
        count = min(int(samples_per_stratum), len(group))
        chosen = rng.choice(group.index.to_numpy(), size=count, replace=False)
        for idx in chosen:
            row = long_df.loc[int(idx)]
            selections.append({
                "day": int(row.day),
                "hour": int(row.hour),
                "prefix_kind": "zero",
                "source_pool": "stratified_validation",
                "selection_tag": (
                    f"need={row.need_bucket}|house={int(row.house_id)}|"
                    f"flex={row.flex_tier}"
                ),
                "illustrative_only": False,
            })

    selected = pd.DataFrame(selections)
    selected["key"] = (
        selected["day"].astype(str)
        + "_"
        + selected["hour"].astype(str)
        + "_"
        + selected["prefix_kind"]
    )
    grouped = []
    for key, group in selected.groupby("key", sort=True):
        grouped.append({
            "state_id": f"VAL_{int(group.day.iloc[0]):03d}_{int(group.hour.iloc[0]):02d}_ZERO",
            "day": int(group.day.iloc[0]),
            "hour": int(group.hour.iloc[0]),
            "prefix_kind": str(group.prefix_kind.iloc[0]),
            "source_pool": "+".join(sorted(set(group.source_pool))),
            "selection_tags": ";".join(sorted(set(group.selection_tag))),
            "illustrative_only": bool(group.illustrative_only.any()),
        })
    return pd.DataFrame(grouped).sort_values(["day", "hour"]).reset_index(drop=True)


def save_manifest(path: str, manifest: pd.DataFrame) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    manifest.to_csv(path, index=False)


def load_manifest(path: str) -> pd.DataFrame:
    manifest = pd.read_csv(path)
    required = {
        "state_id", "day", "hour", "prefix_kind", "source_pool",
        "selection_tags", "illustrative_only",
    }
    missing = required.difference(manifest.columns)
    if missing:
        raise ValueError(f"State manifest is missing columns: {sorted(missing)}")
    return manifest


def build_prefix_environment(
    template: Environment,
    spec: StateSpec,
    agent: DDQNAgent,
) -> Environment:
    env = copy.deepcopy(template)
    zero = _zero_action_index(env)
    env.reset(day=spec.day, mode="val" if not spec.illustrative_only else "test")
    for _ in range(spec.hour):
        state = env.get_state()
        action = zero if spec.prefix_kind == "zero" else greedy_action(agent, state)
        _, _, done, _ = env.step(action)
        if done:
            raise RuntimeError(
                f"Prefix terminated before {spec.state_id} at hour {spec.hour}."
            )
    if int(env.curr_step) != int(spec.hour):
        raise RuntimeError(f"Failed to reconstruct state {spec.state_id}.")
    return env


def discounted_rollout(
    env_after_first_step: Environment,
    agent: DDQNAgent,
    first_reward: float,
    first_done: bool,
    continuation: str,
    gamma: float,
) -> tuple[float, float, int]:
    env = copy.deepcopy(env_after_first_step)
    discounted = float(first_reward)
    undiscounted = float(first_reward)
    discount = float(gamma)
    steps = 1
    done = bool(first_done)
    zero = _zero_action_index(env)
    while not done:
        state = env.get_state()
        if continuation == "zero":
            action = zero
        elif continuation == "greedy":
            action = greedy_action(agent, state)
        else:
            raise ValueError(f"Unknown continuation: {continuation}")
        _, reward, done, _ = env.step(action)
        discounted += discount * float(reward)
        undiscounted += float(reward)
        discount *= gamma
        steps += 1
    return discounted, undiscounted, steps


def next_state_feature_differences(
    state_id: str,
    action_index: int,
    comparison_label: str,
    next_state: np.ndarray,
    zero_next_state: np.ndarray,
    feature_names: list[str],
) -> list[dict[str, Any]]:
    delta = np.asarray(next_state, dtype=float) - np.asarray(zero_next_state, dtype=float)
    rows = []
    for idx, value in enumerate(delta):
        if abs(float(value)) <= 1e-12:
            continue
        rows.append({
            "state_id": state_id,
            "action_index": int(action_index),
            "comparison_label": comparison_label,
            "feature_index": int(idx),
            "feature_name": feature_names[idx],
            "zero_next_value": float(zero_next_state[idx]),
            "action_next_value": float(next_state[idx]),
            "delta": float(value),
            "abs_delta": abs(float(value)),
        })
    return rows


def evaluate_state(
    prefix_env: Environment,
    spec: StateSpec,
    agent: DDQNAgent,
    gamma: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    state = np.asarray(prefix_env.get_state(), dtype=float)
    q = q_values(agent, state)


    network_selected = greedy_action(agent, state)
    zero = _zero_action_index(prefix_env)
    max_action = _maximum_action_index(prefix_env)
    feature_names = list(prefix_env.get_state_feature_names())
    h = int(prefix_env.curr_step)
    flex = per_house_flexibility(prefix_env, h)
    prior_relief, prior_payment = prefix_env._current_settlement_commitments_per_house(h)

    rows: list[dict[str, Any]] = []
    next_states: dict[int, np.ndarray] = {}

    for action_index in range(prefix_env.num_actions):

        details = evaluate_action(prefix_env, action_index)

        env = copy.deepcopy(prefix_env)
        next_state, reward, done, _ = env.step(action_index)
        next_state = np.asarray(next_state, dtype=float)
        next_states[action_index] = next_state.copy()

        if done:
            next_policy_action = -1
            next_policy_q = 0.0
            next_target_q = 0.0
            bellman_target = float(reward)
        else:
            next_tensor = torch.as_tensor(
                next_state, dtype=torch.float32, device=agent.device
            ).unsqueeze(0)
            with torch.no_grad():
                policy_next_values = agent.policy_net(next_tensor).squeeze(0)
                next_policy_action = int(
                    agent._greedy_indices_from_q(policy_next_values).item()
                )
                next_policy_q = float(policy_next_values[next_policy_action].item())
                target_next_values = agent.target_net(next_tensor).squeeze(0)
                next_target_q = float(target_next_values[next_policy_action].item())
            bellman_target = float(reward) + gamma * next_target_q

        greedy_discounted, greedy_undiscounted, greedy_steps = discounted_rollout(
            env, agent, float(reward), bool(done), "greedy", gamma
        )
        zero_discounted, zero_undiscounted, zero_steps = discounted_rollout(
            env, agent, float(reward), bool(done), "zero", gamma
        )

        details.update({
            "state_id": spec.state_id,
            "day": spec.day,
            "hour": spec.hour,
            "prefix_kind": spec.prefix_kind,
            "source_pool": spec.source_pool,
            "selection_tags": spec.selection_tags,
            "illustrative_only": spec.illustrative_only,
            "network_selected_action": network_selected,
            "is_network_selected": action_index == network_selected,
            "is_zero_action": action_index == zero,
            "is_maximum_action_index": action_index == max_action,
            "predicted_Q": float(q[action_index]),
            "next_policy_action": int(next_policy_action),
            "next_policy_Q": float(next_policy_q),
            "next_target_Q_for_policy_action": float(next_target_q),
            "ddqn_one_step_target": float(bellman_target),
            "signed_bellman_residual": float(q[action_index] - bellman_target),
            "abs_bellman_residual": abs(float(q[action_index] - bellman_target)),
            "greedy_followup_discounted_return": float(greedy_discounted),
            "greedy_followup_undiscounted_return": float(greedy_undiscounted),
            "zero_followup_discounted_return": float(zero_discounted),
            "zero_followup_undiscounted_return_v2": float(zero_undiscounted),
            "greedy_followup_steps": int(greedy_steps),
            "zero_followup_steps": int(zero_steps),
            "Q_minus_greedy_return": float(q[action_index] - greedy_discounted),
            "abs_Q_minus_greedy_return": abs(float(q[action_index] - greedy_discounted)),
            "next_state_hash": stable_hash(next_state),
            "next_state_L1_from_current": float(np.abs(next_state - state).sum()),
            "next_state_L2_from_current": float(np.linalg.norm(next_state - state)),
            "pre_action_prior_committed_relief_total": float(np.sum(prior_relief)),
            "pre_action_prior_locked_payment_total": float(np.sum(prior_payment)),
        })
        for i, house_id in enumerate(prefix_env.data_ids):
            details[f"pc_flex_house_{house_id}"] = float(flex["pc_reducible"][i])
            details[f"ev_flex_house_{house_id}"] = float(flex["ev"][i])
            details[f"tsni_flex_house_{house_id}"] = float(flex["tsni"][i])
            details[f"total_flex_house_{house_id}"] = float(flex["total"][i])
            details[f"prior_relief_house_{house_id}"] = float(prior_relief[i])
            details[f"prior_payment_house_{house_id}"] = float(prior_payment[i])
        rows.append(details)

    df = pd.DataFrame(rows)
    for metric in [
        "predicted_Q",
        "immediate_reward",
        "greedy_followup_discounted_return",
        "zero_followup_discounted_return",
    ]:
        df[f"rank_{metric}"] = (
            df[metric].rank(method="min", ascending=False).astype(int)
        )

    zero_row = df.loc[df["is_zero_action"]].iloc[0]
    for metric in [
        "predicted_Q",
        "immediate_reward",
        "greedy_followup_discounted_return",
        "zero_followup_discounted_return",
    ]:
        df[f"gap_vs_zero_{metric}"] = df[metric] - float(zero_row[metric])

    residual_scale = float(df["abs_bellman_residual"].median())
    rollout_error_scale = float(df["abs_Q_minus_greedy_return"].median())
    best_q = float(df["predicted_Q"].max())
    best_immediate = float(df["immediate_reward"].max())
    best_greedy = float(df["greedy_followup_discounted_return"].max())
    best_zero_rollout = float(df["zero_followup_discounted_return"].max())
    for eps in [1e-6, 1e-4, 1e-3, 1e-2]:
        df[f"within_{eps:g}_of_best_Q"] = (best_q - df["predicted_Q"]) <= eps
        df[f"within_{eps:g}_of_best_greedy_return"] = (
            best_greedy - df["greedy_followup_discounted_return"]
        ) <= eps

    zero_next = next_states[zero]
    df["next_state_L1_vs_zero"] = [
        float(np.abs(next_states[int(a)] - zero_next).sum())
        for a in df["action_index"]
    ]
    df["next_state_L2_vs_zero"] = [
        float(np.linalg.norm(next_states[int(a)] - zero_next))
        for a in df["action_index"]
    ]
    df["next_state_max_abs_vs_zero"] = [
        float(np.max(np.abs(next_states[int(a)] - zero_next), initial=0.0))
        for a in df["action_index"]
    ]

    selected_candidates = {
        "network_selected": network_selected,
        "best_immediate": int(df.loc[df["immediate_reward"].idxmax(), "action_index"]),
        "best_greedy_rollout": int(
            df.loc[df["greedy_followup_discounted_return"].idxmax(), "action_index"]
        ),
    }
    positive_no_response = df[
        (df["raw_action_sum"] > EPS)
        & (df["immediate_capacity_relief"].abs() <= 1e-9)
        & (df["incentive_payment"].abs() <= 1e-9)
        & (df["raw_discomfort"].abs() <= 1e-9)
    ]
    if not positive_no_response.empty:
        selected_candidates["best_positive_no_response_Q"] = int(
            positive_no_response.loc[positive_no_response["predicted_Q"].idxmax(), "action_index"]
        )

    diff_rows: list[dict[str, Any]] = []
    for label, action_index in selected_candidates.items():
        if action_index == zero:
            continue
        diff_rows.extend(
            next_state_feature_differences(
                spec.state_id,
                action_index,
                label,
                next_states[action_index],
                zero_next,
                feature_names,
            )
        )

    best_action = int(
        df.loc[df["greedy_followup_discounted_return"].idxmax(), "action_index"]
    )
    best_raw = np.asarray(prefix_env.all_actions[best_action], dtype=float)
    action_lookup = {
        tuple(np.round(np.asarray(action, dtype=float), 10)): idx
        for idx, action in enumerate(prefix_env.all_actions)
    }
    marginal_rows: list[dict[str, Any]] = []
    levels = sorted(set(float(v) for v in np.unique(prefix_env.all_actions)))
    for house_pos, house_id in enumerate(prefix_env.data_ids):
        for level in levels:
            candidate = best_raw.copy()
            candidate[house_pos] = level
            action_index = action_lookup[tuple(np.round(candidate, 10))]
            row = df.loc[df["action_index"] == action_index].iloc[0]
            marginal_rows.append({
                "state_id": spec.state_id,
                "house_id": int(house_id),
                "held_other_components_at_action": best_action,
                "varied_raw_level": float(level),
                "action_index": int(action_index),
                "available_flexibility": float(row[f"total_flex_house_{house_id}"]),
                "incentive_rate": float(row[f"incentive_house_{house_id}"]),
                "house_relief": float(row[f"relief_house_{house_id}"]),
                "house_payment": float(row[f"payment_house_{house_id}"]),
                "predicted_Q": float(row["predicted_Q"]),
                "immediate_reward": float(row["immediate_reward"]),
                "greedy_followup_discounted_return": float(
                    row["greedy_followup_discounted_return"]
                ),
                "zero_followup_discounted_return": float(
                    row["zero_followup_discounted_return"]
                ),
            })

    summary = pd.DataFrame([{
        "state_id": spec.state_id,
        "day": spec.day,
        "hour": spec.hour,
        "prefix_kind": spec.prefix_kind,
        "source_pool": spec.source_pool,
        "selection_tags": spec.selection_tags,
        "illustrative_only": spec.illustrative_only,
        "pre_action_load": float(df["pre_action_load"].iloc[0]),
        "needed_reduction": float(df["needed_reduction"].iloc[0]),
        "network_selected_action": network_selected,
        "zero_action": zero,
        "network_selected_is_zero": network_selected == zero,
        "zero_rank_Q": int(zero_row["rank_predicted_Q"]),
        "zero_rank_immediate": int(zero_row["rank_immediate_reward"]),
        "zero_rank_greedy_rollout": int(
            zero_row["rank_greedy_followup_discounted_return"]
        ),
        "selected_Q_gap_vs_zero": float(
            df.loc[df["action_index"] == network_selected, "gap_vs_zero_predicted_Q"].iloc[0]
        ),
        "selected_immediate_gap_vs_zero": float(
            df.loc[df["action_index"] == network_selected, "gap_vs_zero_immediate_reward"].iloc[0]
        ),
        "selected_greedy_return_gap_vs_zero": float(
            df.loc[
                df["action_index"] == network_selected,
                "gap_vs_zero_greedy_followup_discounted_return",
            ].iloc[0]
        ),
        "best_immediate_action": int(df.loc[df["immediate_reward"].idxmax(), "action_index"]),
        "best_greedy_rollout_action": best_action,
        "best_zero_followup_action": int(
            df.loc[df["zero_followup_discounted_return"].idxmax(), "action_index"]
        ),
        "median_abs_bellman_residual": residual_scale,
        "p95_abs_bellman_residual": float(df["abs_bellman_residual"].quantile(0.95)),
        "median_abs_Q_minus_greedy_return": rollout_error_scale,
        "p95_abs_Q_minus_greedy_return": float(
            df["abs_Q_minus_greedy_return"].quantile(0.95)
        ),
        "near_ties_Q_1e-3": int(((best_q - df["predicted_Q"]) <= 1e-3).sum()),
        "near_ties_greedy_return_1e-3": int(
            ((best_greedy - df["greedy_followup_discounted_return"]) <= 1e-3).sum()
        ),
        "near_ties_immediate_1e-3": int(
            ((best_immediate - df["immediate_reward"]) <= 1e-3).sum()
        ),
        "near_ties_zero_rollout_1e-3": int(
            ((best_zero_rollout - df["zero_followup_discounted_return"]) <= 1e-3).sum()
        ),
        "positive_no_response_action_count": int(len(positive_no_response)),
        "selected_positive_no_response": bool(
            df.loc[df["action_index"] == network_selected, "positive_incentive_no_response"].iloc[0]
        ),
        "selected_unnecessary_incentive": bool(
            df.loc[df["action_index"] == network_selected, "unnecessary_incentive"].iloc[0]
        ),
    }])

    return df, pd.DataFrame(diff_rows), pd.DataFrame(marginal_rows), summary


def collect_policy_failure_specs(
    template: Environment,
    agent: DDQNAgent,
    seed: int,
    per_category: int,
) -> tuple[list[StateSpec], pd.DataFrame]:
    start, end = [int(v) for v in template.val_range]
    candidates: list[dict[str, Any]] = []
    for day in range(start, end + 1):
        env = copy.deepcopy(template)
        state = env.reset(day=day, mode="val")
        done = False
        while not done:
            h = int(env.curr_step)
            pre_total = float(env._current_pre_action_house_load(h).sum())
            need = max(0.0, pre_total - env.capacity_threshold)
            action = greedy_action(agent, state)
            raw_sum = float(np.asarray(env.all_actions[action], dtype=float).sum())
            state, _, done, _ = env.step(action)
            relief = float(env.hourly_capacity_relief[h].sum())
            post_total = float(env.after_total_per_house[h].sum())
            category: Optional[str] = None
            if need <= EPS and raw_sum > EPS and relief <= 1e-9:
                category = "policy_no_need_positive_no_response"
            elif need <= EPS and relief > 1e-9:
                category = "policy_no_need_real_response"
            elif need > EPS and post_total > env.capacity_threshold + 1e-6:
                category = "policy_overload_residual_violation"
            elif need > EPS and post_total < env.capacity_threshold - env.track_deadband_kwh - 1e-6:
                category = "policy_overload_excessive_response"
            if category:
                candidates.append({
                    "day": day,
                    "hour": h,
                    "category": category,
                    "action": action,
                    "raw_action_sum": raw_sum,
                    "needed_reduction": need,
                    "relief": relief,
                    "post_total": post_total,
                })
    candidate_df = pd.DataFrame(candidates)
    if candidate_df.empty:
        return [], candidate_df
    rng = np.random.default_rng(seed)
    specs: list[StateSpec] = []
    for category, group in candidate_df.groupby("category", sort=True):
        count = min(per_category, len(group))
        chosen = rng.choice(group.index.to_numpy(), size=count, replace=False)
        for idx in chosen:
            row = candidate_df.loc[int(idx)]
            specs.append(StateSpec(
                state_id=f"POL_{category}_{int(row.day):03d}_{int(row.hour):02d}",
                day=int(row.day),
                hour=int(row.hour),
                prefix_kind="greedy",
                source_pool="policy_visited_validation",
                selection_tags=category,
                illustrative_only=False,
            ))
    return specs, candidate_df


def build_equivalent_groups(audit: pd.DataFrame) -> pd.DataFrame:
    work = audit.copy()
    physical_columns = [
        "post_DR_load",
        "immediate_capacity_relief",
        "incentive_payment",
        "raw_discomfort",
        "next_state_hash",
    ]
    for col in physical_columns[:-1]:
        work[f"key_{col}"] = work[col].round(8)
    group_cols = ["state_id"] + [f"key_{c}" for c in physical_columns[:-1]] + ["next_state_hash"]
    rows = []
    for keys, group in work.groupby(group_cols, dropna=False, sort=False):
        if len(group) < 2:
            continue
        rows.append({
            "state_id": group["state_id"].iloc[0],
            "group_size": int(len(group)),
            "action_indices": ",".join(str(int(v)) for v in sorted(group["action_index"])),
            "raw_action_sum_min": float(group["raw_action_sum"].min()),
            "raw_action_sum_max": float(group["raw_action_sum"].max()),
            "predicted_Q_range": float(group["predicted_Q"].max() - group["predicted_Q"].min()),
            "immediate_reward_range": float(
                group["immediate_reward"].max() - group["immediate_reward"].min()
            ),
            "greedy_return_range": float(
                group["greedy_followup_discounted_return"].max()
                - group["greedy_followup_discounted_return"].min()
            ),
            "contains_zero_action": bool(group["is_zero_action"].any()),
            "contains_network_selected": bool(group["is_network_selected"].any()),
        })
    return pd.DataFrame(rows)


def training_metrics_context(path: Optional[str], checkpoint_episode: Any) -> pd.DataFrame:
    if not path:
        return pd.DataFrame()
    payload = np.load(path, allow_pickle=True).item()
    losses = np.asarray(payload.get("episode_losses", []), dtype=float)
    rewards = np.asarray(payload.get("episode_rewards", []), dtype=float)
    if losses.size == 0:
        return pd.DataFrame()
    try:
        episode = int(checkpoint_episode)
    except (TypeError, ValueError):
        episode = len(losses)
    start = max(0, episode - 50)
    end = min(len(losses), episode + 50)
    window_losses = losses[start:end]
    window_rewards = rewards[start:end] if rewards.size else np.asarray([])
    return pd.DataFrame([{
        "checkpoint_episode": episode,
        "context_window_start_episode": start + 1,
        "context_window_end_episode": end,
        "loss_mean_context_only": float(np.mean(window_losses)),
        "loss_std_context_only": float(np.std(window_losses)),
        "loss_p95_context_only": float(np.quantile(window_losses, 0.95)),
        "reward_mean_context": float(np.mean(window_rewards)) if window_rewards.size else np.nan,
        "reward_std_context": float(np.std(window_rewards)) if window_rewards.size else np.nan,
        "note": (
            "Training loss is reported for context only. It is not used as the Q-noise floor; "
            "state-specific Bellman residual and Q-vs-greedy-rollout error are used instead."
        ),
    }])


def write_workbook(path: str, sheets: dict[str, pd.DataFrame]) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        for name, frame in sheets.items():
            safe_name = name[:31]
            frame.to_excel(writer, sheet_name=safe_name, index=False)
            worksheet = writer.book[safe_name]
            worksheet.freeze_panes = "A2"
            worksheet.auto_filter.ref = worksheet.dimensions
            for column_cells in worksheet.columns:
                max_length = max(
                    len(str(cell.value)) if cell.value is not None else 0
                    for cell in column_cells[: min(len(column_cells), 500)]
                )
                worksheet.column_dimensions[column_cells[0].column_letter].width = min(
                    max(max_length + 2, 10), 45
                )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--config_path", required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--state_manifest", required=True)
    parser.add_argument("--samples_per_stratum", type=int, default=2)
    parser.add_argument("--policy_failure_samples", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20260727)
    parser.add_argument("--training_metrics", default=None)
    parser.add_argument(
        "--illustrative_day",
        type=int,
        default=None,
        help="Optional test day for illustration only; excluded from validation summaries.",
    )
    parser.add_argument(
        "--recreate_manifest",
        action="store_true",
        help="Overwrite the shared validation state manifest.",
    )
    parser.add_argument(
        "--skip_policy_failures",
        action="store_true",
        help="Audit only controlled zero-prefix validation states.",
    )
    parser.add_argument(
        "--allow_reward_override",
        action="store_true",
        help=(
            "Allow an intentional reward-metadata override for a diagnostic-only run. "
            "State and action dimensions must still match."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_global_seeds(args.seed)
    cfg = load_yaml(args.config_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    raw_checkpoint = safe_torch_load(args.model_path, device)
    template = build_environment(cfg, raw_checkpoint)
    agent, checkpoint = build_agent(cfg, template, args.model_path)
    validate_provenance(
        cfg,
        template,
        checkpoint,
        allow_reward_override=args.allow_reward_override,
    )

    if args.recreate_manifest or not os.path.exists(args.state_manifest):
        catalog = collect_zero_prefix_validation_catalog(copy.deepcopy(template))
        manifest = create_state_manifest(
            catalog,
            list(template.data_ids),
            args.samples_per_stratum,
            args.seed,
        )
        save_manifest(args.state_manifest, manifest)
        print(f"Created shared validation manifest: {os.path.abspath(args.state_manifest)}")
    else:
        catalog = collect_zero_prefix_validation_catalog(copy.deepcopy(template))
        manifest = load_manifest(args.state_manifest)
        print(f"Loaded shared validation manifest: {os.path.abspath(args.state_manifest)}")

    controlled_specs = [
        StateSpec(
            state_id=str(row.state_id),
            day=int(row.day),
            hour=int(row.hour),
            prefix_kind=str(row.prefix_kind),
            source_pool=str(row.source_pool),
            selection_tags=str(row.selection_tags),
            illustrative_only=bool(row.illustrative_only),
        )
        for row in manifest.itertuples(index=False)
    ]

    policy_specs: list[StateSpec] = []
    policy_candidates = pd.DataFrame()
    if not args.skip_policy_failures:
        policy_specs, policy_candidates = collect_policy_failure_specs(
            template, agent, args.seed, args.policy_failure_samples
        )

    illustrative_specs: list[StateSpec] = []
    if args.illustrative_day is not None:


        day = int(args.illustrative_day)
        env = copy.deepcopy(template)
        state = env.reset(day=day, mode="test")
        done = False
        while not done:
            h = int(env.curr_step)
            action = greedy_action(agent, state)
            raw_sum = float(np.asarray(env.all_actions[action]).sum())
            pre_total = float(env._current_pre_action_house_load(h).sum())
            need = max(0.0, pre_total - env.capacity_threshold)
            state, _, done, _ = env.step(action)
            relief = float(env.hourly_capacity_relief[h].sum())
            post_total = float(env.after_total_per_house[h].sum())
            categories = []
            if need <= EPS and raw_sum > EPS and relief <= 1e-9:
                categories.append("day208_no_need_positive_no_response")
            if need <= EPS and relief > 1e-9:
                categories.append("day208_no_need_real_response")
            if need > EPS and post_total > env.capacity_threshold + 1e-6:
                categories.append("day208_residual_violation")
            if need > EPS and post_total < env.capacity_threshold - env.track_deadband_kwh - 1e-6:
                categories.append("day208_excessive_response")
            for category in categories:
                illustrative_specs.append(StateSpec(
                    state_id=f"ILL_{day:03d}_{h:02d}_{category}",
                    day=day,
                    hour=h,
                    prefix_kind="greedy",
                    source_pool="illustrative_test_day",
                    selection_tags=category,
                    illustrative_only=True,
                ))

    specs = controlled_specs + policy_specs + illustrative_specs
    unique_specs: list[StateSpec] = []
    seen = set()
    for spec in specs:
        key = (spec.state_id, spec.day, spec.hour, spec.prefix_kind)
        if key not in seen:
            unique_specs.append(spec)
            seen.add(key)
    gamma = float(cfg["DDQN"]["gamma"])

    all_audit: list[pd.DataFrame] = []
    all_diffs: list[pd.DataFrame] = []
    all_marginals: list[pd.DataFrame] = []
    all_summaries: list[pd.DataFrame] = []

    if not unique_specs:
        raise RuntimeError("No diagnostic states were selected.")

    for idx, spec in enumerate(unique_specs, start=1):
        if idx == 1 or idx % 10 == 0 or idx == len(unique_specs):
            print(f"Q diagnostic state {idx}/{len(unique_specs)}")
        prefix_env = build_prefix_environment(template, spec, agent)
        audit, diffs, marginals, summary = evaluate_state(
            prefix_env, spec, agent, gamma
        )
        all_audit.append(audit)
        all_diffs.append(diffs)
        all_marginals.append(marginals)
        all_summaries.append(summary)

    action_audit = pd.concat(all_audit, ignore_index=True)
    next_diffs = pd.concat(all_diffs, ignore_index=True) if all_diffs else pd.DataFrame()
    marginals = pd.concat(all_marginals, ignore_index=True) if all_marginals else pd.DataFrame()
    state_summary = pd.concat(all_summaries, ignore_index=True)
    equivalent = build_equivalent_groups(action_audit)

    catalog_selected = catalog.merge(
        manifest[["day", "hour", "state_id", "source_pool", "selection_tags"]],
        on=["day", "hour"],
        how="left",
    )
    catalog_selected = catalog_selected[catalog_selected["state_id"].notna()].copy()

    metadata = pd.DataFrame([{
        "script_version": SCRIPT_VERSION,
        "label": args.label,
        "model_path": os.path.abspath(args.model_path),
        "config_path": os.path.abspath(args.config_path),
        "state_manifest": os.path.abspath(args.state_manifest),
        "checkpoint_episode": checkpoint.get("episode", "unknown"),
        "checkpoint_model_type": checkpoint.get("model_type", "unknown"),
        "reward_mode": template.reward_mode,
        "reward_definition_version": template.REWARD_DEFINITION_VERSION,
        "allow_reward_override": bool(args.allow_reward_override),
        "no_need_offer_regularization_enabled": bool(
            template.no_need_offer_regularization_enabled
        ),
        "no_need_offer_coefficient": float(
            template.no_need_offer_coefficient
        ),
        "no_need_gate_tolerance_kw": float(
            template.no_need_gate_tolerance_kw
        ),
        "state_dim": len(template.get_state_feature_names()),
        "action_dim": template.num_actions,
        "gamma": gamma,
        "tie_break_enabled": bool(agent.tie_break_enabled),
        "tie_break_q_tolerance": float(agent.tie_break_q_tolerance),
        "sampling_seed": args.seed,
        "controlled_state_count": len(controlled_specs),
        "policy_failure_state_count": len(policy_specs),
        "illustrative_state_count": len(illustrative_specs),
        "classification_note": (
            "Use exact discounted greedy-followup return and state-specific Bellman residual/Q-vs-rollout "
            "error to distinguish reward near-ties from learned-Q misranking. Training loss is contextual only."
        ),
    }])

    readme = pd.DataFrame({"Notes": [
        "All 125 joint actions are evaluated from cloned states.",
        "Controlled states come from a seeded, stratified sample of the fixed validation range under a zero-action prefix and are shared across checkpoints through State_Manifest.csv.",
        "Policy-visited validation states are sampled separately and are not used as cross-checkpoint frequency estimates.",
        "Illustrative test-day states are explicitly labelled illustrative_only and must not be used to tune coefficients.",
        "Predicted_Q is compared with the exact deterministic full-day discounted return obtained by taking the candidate action and then following the checkpoint's greedy policy.",
        "ddqn_one_step_target uses the policy network for next-action selection and the target network for evaluation, matching DDQN training.",
        "Training loss is not used as the Q-noise floor because Smooth-L1 loss is not a state-specific Q uncertainty measure.",
        "A reward-side near-tie is indicated when exact zero-vs-positive return gaps are tiny relative to state-specific Bellman residual/Q-vs-rollout errors.",
        "A learning-side misranking is indicated when the exact greedy rollout clearly prefers zero but the network Q ordering prefers a positive ineffective action by a material margin.",
        "Next_State_Differences identifies whether previous-incentive or settlement-commitment features create future-value differences between otherwise physically equivalent actions.",
        "When --allow_reward_override is used, checkpoint Q-values remain frozen while exact immediate and rollout returns are recomputed under the active diagnostic reward configuration. This is not a trained-policy performance claim.",
    ]})

    context = training_metrics_context(
        args.training_metrics, checkpoint.get("episode")
    )

    sheets = {
        "Run_Metadata": metadata,
        "State_Manifest": manifest,
        "Selected_State_Catalog": catalog_selected,
        "State_Summary": state_summary,
        "Action_Q_Audit": action_audit.sort_values(
            ["state_id", "rank_predicted_Q", "action_index"]
        ),
        "Zero_Action_Comparison": state_summary,
        "Equivalent_Action_Groups": equivalent,
        "Marginal_Household_Slices": marginals,
        "Next_State_Differences": next_diffs,
        "Policy_Failure_Candidates": policy_candidates,
        "Training_Metrics_Context": context,
        "README": readme,
    }
    write_workbook(args.output, sheets)
    print(f"Q diagnostic saved: {os.path.abspath(args.output)}")


if __name__ == "__main__":
    main()