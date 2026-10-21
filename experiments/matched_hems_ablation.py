"""Matched-HEMS ablation environment (CBSP): economic-only service-provider objective.

The original env/env.py remains unchanged; this subclass only masks the two
capacity-need state features and replaces the learning reward.

NOT a capacity-blind HEMS: appliance scheduling keeps the original 7-kW rules.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from env import Environment  # noqa: E402

class EconomicSPEnvironment(Environment):
    """Same household physics/settlement, no capacity-need features in SP state.

    The economic reward is computed from verified settled relief, not useful
    capacity-capped relief. Reported capacity metrics retain their definition.
    """
    def get_state(self):
        state = super().get_state().copy()
        if state.size != 12 * self.N + 7:
            raise RuntimeError('Unexpected state dimension')
        state[-2:] = 0.0
        return state

    def compute_total_reward(self):
        # Parent computes *all original diagnostic and reporting arrays*.
        # This does not affect the household physical transition.
        super().compute_total_reward()
        h = self.curr_step
        price = float(self.prices[h])
        settled = float(self.settlement_dr_energy[h].sum())
        payment = float(self.incentive_payment[h].sum())
        scale = max(float(self.reward_economic_scale), float(self.EPS))
        discomfort = float(self.normalized_discomfort_component[h])
        sp = (price * settled - payment) / scale
        eu = self.rho * payment / scale - (1 - self.rho) * discomfort
        economic_reward = self.stakeholder_sp_weight * sp + self.stakeholder_eu_weight * eu
        if not np.isfinite(economic_reward):
            raise FloatingPointError('Nonfinite ablation reward')
        # Keep the parent's stakeholder_* and normalized_economic_* arrays
        # unchanged: they are CADR-definition diagnostics that test.py audits
        # independently. Only the reward actually used for learning changes.
        self.reward_base_component[h] = economic_reward
        self.reward_shaping_component[h] = 0.0
        self.reward_tracking_component[h] = 0.0
        self.reward_no_need_component[h] = 0.0
        self.rewards_total[h] = economic_reward
        return float(economic_reward)
