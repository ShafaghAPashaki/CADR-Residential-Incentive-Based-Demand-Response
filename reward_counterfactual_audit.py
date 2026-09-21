from __future__ import annotations

import copy
import random

import numpy as np
import torch

from env import Environment

EPS = 1e-9


def set_global_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _zero_action_index(env: Environment) -> int:
    matches = np.flatnonzero(np.all(np.isclose(env.all_actions, 0.0), axis=1))
    if len(matches) != 1:
        raise RuntimeError("Could not identify the unique all-zero action.")
    return int(matches[0])


def _maximum_action_index(env: Environment) -> int:
    matches = np.flatnonzero(np.all(np.isclose(env.all_actions, 1.0), axis=1))
    if len(matches) != 1:
        raise RuntimeError("Could not identify the unique all-maximum action.")
    return int(matches[0])


def evaluate_action(prefix_env: Environment, action_index: int) -> dict:
    env = copy.deepcopy(prefix_env)
    h = int(env.curr_step)
    raw_action = np.asarray(env.all_actions[action_index], dtype=float)
    pre_total = float(env._current_pre_action_house_load(h).sum())
    price = float(env.prices[h])
    prior_relief_house, prior_payment_house = env._current_settlement_commitments_per_house(h)

    _, immediate_reward, done, _ = env.step(action_index)
    post_total = float(env.after_total_per_house[h].sum())
    incentive_rates = env.incentives[h].copy()

    result = {
        "action_index": int(action_index),
        "raw_action_sum": float(raw_action.sum()),
        "price": price,
        "pre_action_load": pre_total,
        "post_DR_load": post_total,
        "capacity": float(env.capacity_threshold),
        "needed_reduction": float(env.needed_reduction[h]),
        "immediate_capacity_relief": float(env.hourly_capacity_relief[h].sum()),
        "action_attributed_DR": float(env.action_attributed_dr_energy[h].sum()),
        "settlement_DR": float(env.settlement_dr_energy[h].sum()),
        "useful_settlement_DR": float(env.useful_settlement_dr_energy[h]),
        "excess_settlement_DR": float(env.excess_settlement_dr_energy[h]),
        "incentive_payment": float(env.incentive_payment[h].sum()),
        "pre_action_prior_committed_relief": float(prior_relief_house.sum()),
        "pre_action_prior_locked_payment": float(prior_payment_house.sum()),
        "raw_discomfort": float(env.discomforts[h].sum()),
        "control_target_load": float(env.control_target_load[h]),
        "control_over_error": float(env.control_over_error[h]),
        "control_under_error": float(env.control_under_error[h]),
        "target_tracking_penalty": float(env.target_tracking_penalty[h]),
        "normalised_economic_component": float(env.normalized_economic_component[h]),
        "normalised_discomfort_component": float(env.normalized_discomfort_component[h]),
        "raw_discomfort_PC_AC": float(env.raw_discomfort_by_class[h, 0]),
        "raw_discomfort_EV_TS_I": float(env.raw_discomfort_by_class[h, 1]),
        "raw_discomfort_TS_NI": float(env.raw_discomfort_by_class[h, 2]),
        "normalised_discomfort_PC_AC": float(env.normalized_discomfort_by_class[h, 0]),
        "normalised_discomfort_EV_TS_I": float(env.normalized_discomfort_by_class[h, 1]),
        "normalised_discomfort_TS_NI": float(env.normalized_discomfort_by_class[h, 2]),
        "incentive_offer_intensity": float(env.incentive_offer_intensity[h]),
        "incentive_offer_penalty": float(env.incentive_offer_penalty[h]),
        "no_need_gate": bool(env.no_need_gate_flag[h]),
        "no_need_offer_penalty": float(env.no_need_offer_penalty[h]),
        "combined_offer_penalty": float(env.combined_offer_penalty[h]),
        "reward_base_component": float(env.reward_base_component[h]),
        "reward_tracking_component": float(env.reward_tracking_component[h]),
        "reward_no_need_component": float(env.reward_no_need_component[h]),
        "immediate_reward": float(immediate_reward),
        "unnecessary_incentive": bool(env.unnecessary_incentive_flag[h]),
        "positive_incentive_no_response": bool(env.positive_incentive_no_response_flag[h]),
        "maximum_action": bool(env.maximum_action_flag[h]),
    }

    decision_hour = env.device_settlement_decision_hour[h]
    settled_energy = env.device_verified_settlement_energy[h]
    settled_payment = env.device_settlement_payment[h]
    prior_mask = (decision_hour >= 0) & (decision_hour < h)
    current_mask = decision_hour == h
    result.update({
        "settlement_DR_from_prior_decisions": float(settled_energy[prior_mask].sum()),
        "payment_from_prior_decisions": float(settled_payment[prior_mask].sum()),
        "settlement_DR_from_current_decision": float(settled_energy[current_mask].sum()),
        "payment_from_current_decision": float(settled_payment[current_mask].sum()),
    })

    for i, house_id in enumerate(env.data_ids):
        result[f"raw_action_house_{house_id}"] = float(raw_action[i])
        result[f"incentive_house_{house_id}"] = float(incentive_rates[i])
        result[f"relief_house_{house_id}"] = float(env.hourly_capacity_relief[h, i])
        result[f"payment_house_{house_id}"] = float(env.incentive_payment[h, i])
        result[f"discomfort_house_{house_id}"] = float(env.discomforts[h, i])

    rollout_reward = float(immediate_reward)
    discounted_rollout_reward = float(immediate_reward)
    gamma = float(env.cfg["DDQN"]["gamma"])
    discount = gamma
    zero_action = _zero_action_index(env)
    while not done:
        _, reward, done, _ = env.step(zero_action)
        rollout_reward += float(reward)
        discounted_rollout_reward += discount * float(reward)
        discount *= gamma

    future_after = env.after_total_per_house[h:].sum(axis=1)
    future_overrun = np.maximum(future_after - env.capacity_threshold, 0.0)
    result.update({
        "zero_followup_rollout_return": rollout_reward,
        "zero_followup_discounted_return": discounted_rollout_reward,
        "future_violation_steps": int(np.sum(future_overrun > 1e-6)),
        "future_cumulative_excess": float(future_overrun.sum()),
        "future_maximum_overrun": float(future_overrun.max(initial=0.0)),
        "future_shifted_in_energy": float(env.shifted_in_per_house[h:].sum()),
        "future_total_payment": float(env.incentive_payment[h:].sum()),
        "future_total_discomfort": float(env.discomforts[h:].sum()),
    })
    return result
