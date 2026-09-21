import argparse
import os
import random
import shutil
import warnings
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import yaml

from env import Environment
from agent.agent_ddqn import DDQNAgent
from utils.config_loader import load_config


plt.rcParams.update({
    "font.size": 24,
    "axes.labelsize": 24,
    "xtick.labelsize": 22,
    "ytick.labelsize": 22,
    "legend.fontsize": 20,
    "legend.title_fontsize": 22,
})
mpl.rcParams["hatch.linewidth"] = 3
BIN_SIZE = 2
EPS = 1e-9
AUDIT_TOL = 1e-8
ARTICLE_PLOT_DAY = 208
SCRIPT_VERSION = "final_ddqn_testing"


class ModelTester:
    """Evaluate one trained DDQN model on the full configured test range.

    This version is designed for the verified-ledger Environment. It reports
    physical capacity effects, non-negative hourly settlement energy, locked
    incentive settlement, action-attributed DR, signed procurement diagnostics,
    true curtailment, shifting, and unshifted EV requests separately.
    """

    def __init__(
        self,
        model_path,
        cfg_override=None,
        target_plot_day=ARTICLE_PLOT_DAY,
        target_plot_days=None,
        output_dir=None,
        allow_legacy_checkpoint=False,
        test_range_override=None,
        overwrite=False,
    ):
        self.cfg = cfg_override or load_config()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_path = model_path
        if target_plot_days is None:
            target_plot_days = [target_plot_day]
        self.target_plot_days = []
        for value in target_plot_days:
            day_value = int(value)
            if day_value not in self.target_plot_days:
                self.target_plot_days.append(day_value)
        if not self.target_plot_days:
            raise ValueError("At least one target plot day must be provided.")


        self.target_plot_day = int(self.target_plot_days[0])
        self.allow_legacy_checkpoint = bool(allow_legacy_checkpoint)

        cfg_test_start, cfg_test_end = self.cfg["training"]["test_range"]
        if test_range_override is None:
            self.active_test_range = (int(cfg_test_start), int(cfg_test_end))
        else:
            override_start, override_end = test_range_override
            self.active_test_range = (int(override_start), int(override_end))
            if self.active_test_range[0] > self.active_test_range[1]:
                raise ValueError(
                    f"Invalid overridden test range: {self.active_test_range[0]}-{self.active_test_range[1]}"
                )

        self.checkpoint = self._read_checkpoint(model_path)

        house_ids = list(self.cfg["environment"]["house_ids"])
        self.house_ids = house_ids
        self.env = Environment(data_ids=house_ids, cfg_override=self.cfg)
        self._restore_household_coefficients(self.checkpoint)

        state = self.env.reset(day=int(self.active_test_range[0]), mode="test")
        state_dim = len(state)
        action_dim = self.env.num_actions
        self.agent = DDQNAgent(
            state_dim,
            action_dim,
            cfg_override=self.cfg,
            action_costs=np.mean(self.env.all_actions, axis=1),
        )
        self._validate_checkpoint_metadata(
            self.checkpoint, state_dim=state_dim, action_dim=action_dim
        )

        self.load_model(self.checkpoint, model_path)

        plot_day_tag = "_".join(str(day) for day in self.target_plot_days)
        default_output_dir = os.path.join(
            os.path.dirname(os.path.abspath(model_path)),
            f"test_outputs_{self.active_test_range[0]}_{self.active_test_range[1]}_days_{plot_day_tag}",
        )
        self.results_dir = os.path.abspath(output_dir or default_output_dir)
        if os.path.isdir(self.results_dir) and os.listdir(self.results_dir):
            if not overwrite:
                raise FileExistsError(
                    f"Test output already exists: {self.results_dir}. "
                    "Use --overwrite only when you intentionally want to replace it."
                )
            shutil.rmtree(self.results_dir)
        os.makedirs(self.results_dir, exist_ok=True)

        test_start, test_end = self.active_test_range
        self.test_days = list(range(int(test_start), int(test_end) + 1))
        configured_test_start, configured_test_end = self.cfg["training"]["test_range"]
        for plot_day in self.target_plot_days:
            if plot_day not in self.test_days:
                print(
                    f"[WARN] Target plot day {plot_day} is outside the active "
                    f"test range {test_start}-{test_end}; its detailed plots will be skipped."
                )

        self.capacity = float(self.env.capacity_threshold)
        self.rho = float(self.cfg["environment"]["rho"])
        self.day_results = {}

    def _read_checkpoint(self, model_path):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        try:
            checkpoint = torch.load(
                model_path, map_location=self.device, weights_only=False
            )
        except TypeError:
            checkpoint = torch.load(model_path, map_location=self.device)
        if not isinstance(checkpoint, dict):
            raise TypeError("The model checkpoint must be a dictionary.")
        if "policy_net_state_dict" not in checkpoint:
            raise KeyError("Checkpoint does not contain 'policy_net_state_dict'.")
        return checkpoint

    def _restore_household_coefficients(self, checkpoint):
        values = checkpoint.get("household_coefficients")
        if values is None:
            message = (
                "Checkpoint does not contain household_coefficients; test-time "
                "preferences cannot be proven identical to training preferences."
            )
            if not self.allow_legacy_checkpoint:
                raise ValueError(message + " Use --allow_legacy_checkpoint only for diagnostics.")
            warnings.warn(message, RuntimeWarning)
            return

        if torch.is_tensor(values):
            values = values.detach().cpu().numpy()
        values = np.asarray(values, dtype=float)
        expected_shape = (self.env.N, self.env.num_devices)
        if values.shape != expected_shape:
            raise ValueError(
                "household_coefficients shape mismatch: "
                f"checkpoint={values.shape}, environment={expected_shape}."
            )
        if not np.isfinite(values).all():
            raise ValueError("Checkpoint household_coefficients contain non-finite values.")
        self.env.set_household_coefficients(values)

    @staticmethod
    def _same_sequence(left, right):
        return list(left) == list(right)

    def _validate_checkpoint_metadata(self, checkpoint, state_dim, action_dim):
        required = [
            "environment_version",
            "accounting_version",
            "state_dim",
            "state_feature_names",
            "action_dim",
            "house_ids",
            "device_names",
            "capacity_threshold",
            "power_rate",
            "device_non_interruptible",
            "ts_deadline_hour",
            "time_steps_test",
            "expected_hours_per_day",
            "rho",
            "reward_mode",
            "c_cap",
            "c_over",
            "reward_definition_version",
            "reward_normalization_enabled",
            "reward_economic_scale",
            "reward_discomfort_scale",
            "discomfort_normalization_mode",
            "reward_discomfort_class_scales",
            "discomfort_weight",
            "offer_regularization_enabled",
            "offer_regularization_fraction",
            "incentive_offer_coefficient",
            "incentive_offer_coefficient_source",
            "track_c_violation",
            "track_c_violation_source",
            "track_c_under",
            "track_c_under_source",
            "track_deadband_kwh",
            "no_need_offer_regularization_enabled",
            "no_need_offer_coefficient",
            "no_need_gate_tolerance_kw",
            "no_need_gate_definition",
            "no_need_offer_cost_definition",
            "reward_discomfort_p95_dominance_ratio",
            "stakeholder_sp_weight",
            "stakeholder_eu_weight",
            "effective_payment_cost_coefficient",
            "effective_payment_cost_fraction",
            "minimum_effective_payment_cost_fraction",
            "symmetric_tracking_lambda",
            "symmetric_tracking_lambda_source",
            "symmetric_tracking_loss",
            "symmetric_tracking_delta",
            "symmetric_tracking_delta_source",
            "tie_break_enabled",
            "tie_break_q_tolerance",
        ]
        missing = [field for field in required if field not in checkpoint]
        if missing:
            message = f"Legacy/incomplete checkpoint metadata: missing {missing}."
            if not self.allow_legacy_checkpoint:
                raise ValueError(message + " Final paper testing requires a new checkpoint.")
            warnings.warn(message, RuntimeWarning)

        checks = []
        if "environment_version" in checkpoint:
            checks.append((
                "environment_version",
                checkpoint["environment_version"],
                self.env.VERSION,
            ))
        if "accounting_version" in checkpoint:
            checks.append((
                "accounting_version",
                checkpoint["accounting_version"],
                self.env.ACCOUNTING_VERSION,
            ))
        if "state_dim" in checkpoint:
            checks.append(("state_dim", int(checkpoint["state_dim"]), int(state_dim)))
        if "action_dim" in checkpoint:
            checks.append(("action_dim", int(checkpoint["action_dim"]), int(action_dim)))
        if "house_ids" in checkpoint:
            checks.append(("house_ids", list(checkpoint["house_ids"]), self.house_ids))
        if "device_names" in checkpoint:
            checks.append((
                "device_names", list(checkpoint["device_names"]), list(self.env.DEVICES)
            ))
        if "state_feature_names" in checkpoint:
            checks.append((
                "state_feature_names",
                list(checkpoint["state_feature_names"]),
                self.env.get_state_feature_names(),
            ))
        if "power_rate" in checkpoint:
            checks.append((
                "power_rate",
                [float(value) for value in checkpoint["power_rate"]],
                [float(value) for value in self.env.POWER_RATE],
            ))
        if "device_non_interruptible" in checkpoint:
            checks.append((
                "device_non_interruptible",
                list(checkpoint["device_non_interruptible"]),
                list(self.cfg["environment"]["DEVICE_NON_INTERRUPTIBLE"]),
            ))
        if "ts_deadline_hour" in checkpoint:
            checks.append((
                "ts_deadline_hour",
                dict(checkpoint["ts_deadline_hour"]),
                dict(self.env.ts_deadline_hour),
            ))
        if "time_steps_test" in checkpoint:
            checks.append((
                "time_steps_test",
                int(checkpoint["time_steps_test"]),
                int(self.env.time_steps_test),
            ))
        if "expected_hours_per_day" in checkpoint:
            checks.append((
                "expected_hours_per_day",
                int(checkpoint["expected_hours_per_day"]),
                int(self.env.expected_hours_per_day),
            ))
        if "reward_mode" in checkpoint:
            checks.append((
                "reward_mode",
                str(checkpoint["reward_mode"]),
                str(self.env.reward_mode),
            ))
        if "reward_definition_version" in checkpoint:
            checks.append((
                "reward_definition_version",
                str(checkpoint["reward_definition_version"]),
                str(self.env.REWARD_DEFINITION_VERSION),
            ))
        if "reward_normalization_enabled" in checkpoint:
            checks.append((
                "reward_normalization_enabled",
                bool(checkpoint["reward_normalization_enabled"]),
                bool(self.env.reward_normalization_enabled),
            ))
        if "track_c_violation_source" in checkpoint:
            checks.append((
                "track_c_violation_source",
                str(checkpoint["track_c_violation_source"]),
                str(self.env.track_c_violation_source),
            ))
        if "track_c_under_source" in checkpoint:
            checks.append((
                "track_c_under_source",
                str(checkpoint["track_c_under_source"]),
                str(self.env.track_c_under_source),
            ))
        if "discomfort_normalization_mode" in checkpoint:
            checks.append((
                "discomfort_normalization_mode",
                str(checkpoint["discomfort_normalization_mode"]),
                str(self.env.discomfort_normalization_mode),
            ))
        if "incentive_offer_coefficient_source" in checkpoint:
            checks.append((
                "incentive_offer_coefficient_source",
                str(checkpoint["incentive_offer_coefficient_source"]),
                str(self.env.incentive_offer_coefficient_source),
            ))
        if "offer_regularization_enabled" in checkpoint:
            checks.append((
                "offer_regularization_enabled",
                bool(checkpoint["offer_regularization_enabled"]),
                bool(self.env.offer_regularization_enabled),
            ))
        if "no_need_offer_regularization_enabled" in checkpoint:
            checks.append((
                "no_need_offer_regularization_enabled",
                bool(checkpoint["no_need_offer_regularization_enabled"]),
                bool(self.env.no_need_offer_regularization_enabled),
            ))
        if "no_need_gate_definition" in checkpoint:
            checks.append((
                "no_need_gate_definition",
                str(checkpoint["no_need_gate_definition"]),
                str(self.env.get_reward_metadata()["no_need_gate_definition"]),
            ))
        if "no_need_offer_cost_definition" in checkpoint:
            checks.append((
                "no_need_offer_cost_definition",
                str(checkpoint["no_need_offer_cost_definition"]),
                str(self.env.get_reward_metadata()["no_need_offer_cost_definition"]),
            ))
        for field, active_value in {
            "symmetric_tracking_lambda_source": self.env.symmetric_tracking_lambda_source,
            "symmetric_tracking_loss": self.env.symmetric_tracking_loss_name,
            "symmetric_tracking_delta_source": self.env.symmetric_tracking_delta_source,
        }.items():
            if field in checkpoint:
                checks.append((field, str(checkpoint[field]), str(active_value)))
        if "tie_break_enabled" in checkpoint:
            checks.append((
                "tie_break_enabled",
                bool(checkpoint["tie_break_enabled"]),
                bool(self.agent.tie_break_enabled),
            ))

        if "reward_discomfort_class_scales" in checkpoint:
            saved_scales = {
                str(k): float(v)
                for k, v in dict(checkpoint["reward_discomfort_class_scales"]).items()
            }
            active_scales = {
                str(k): float(v)
                for k, v in self.env.reward_discomfort_class_scales.items()
            }
            if saved_scales.keys() != active_scales.keys() or any(
                not np.isclose(saved_scales[k], active_scales[k], rtol=1e-9, atol=1e-10)
                for k in active_scales
            ):
                checks.append((
                    "reward_discomfort_class_scales",
                    saved_scales,
                    active_scales,
                ))

        mismatches = [
            (name, saved, active)
            for name, saved, active in checks
            if saved != active
        ]
        if "capacity_threshold" in checkpoint and not np.isclose(
            float(checkpoint["capacity_threshold"]), self.env.capacity_threshold
        ):
            mismatches.append((
                "capacity_threshold",
                checkpoint["capacity_threshold"],
                self.env.capacity_threshold,
            ))
        if "rho" in checkpoint and not np.isclose(
            float(checkpoint["rho"]), self.env.rho
        ):
            mismatches.append(("rho", checkpoint["rho"], self.env.rho))
        if "c_cap" in checkpoint and not np.isclose(
            float(checkpoint["c_cap"]), float(self.env.c_cap)
        ):
            mismatches.append(("c_cap", checkpoint["c_cap"], self.env.c_cap))
        if "c_over" in checkpoint and not np.isclose(
            float(checkpoint["c_over"]), float(self.env.c_over)
        ):
            mismatches.append(("c_over", checkpoint["c_over"], self.env.c_over))
        reward_float_checks = {
            "reward_economic_scale": self.env.reward_economic_scale,
            "reward_discomfort_scale": self.env.reward_discomfort_scale,
            "discomfort_weight": self.env.discomfort_weight,
            "offer_regularization_fraction": self.env.offer_regularization_fraction,
            "incentive_offer_coefficient": self.env.incentive_offer_coefficient,
            "track_c_violation": self.env.track_c_violation,
            "track_c_under": self.env.track_c_under,
            "track_deadband_kwh": self.env.track_deadband_kwh,
            "no_need_offer_coefficient": self.env.no_need_offer_coefficient,
            "no_need_gate_tolerance_kw": self.env.no_need_gate_tolerance_kw,
            "reward_discomfort_p95_dominance_ratio": (
                self.env.reward_discomfort_p95_dominance_ratio
            ),
            "stakeholder_sp_weight": self.env.stakeholder_sp_weight,
            "stakeholder_eu_weight": self.env.stakeholder_eu_weight,
            "effective_payment_cost_coefficient": (
                self.env.effective_payment_cost_coefficient
            ),
            "effective_payment_cost_fraction": (
                self.env.effective_payment_cost_fraction
            ),
            "minimum_effective_payment_cost_fraction": (
                self.env.minimum_effective_payment_cost_fraction
            ),
            "symmetric_tracking_lambda": self.env.symmetric_tracking_lambda,
            "symmetric_tracking_delta": self.env.symmetric_tracking_delta,
            "tie_break_q_tolerance": self.agent.tie_break_q_tolerance,
        }
        for field, active_value in reward_float_checks.items():
            if field in checkpoint and not np.isclose(
                float(checkpoint[field]), float(active_value), rtol=1e-9, atol=1e-10
            ):
                mismatches.append((field, checkpoint[field], active_value))
        if mismatches:
            details = "\n".join(
                f"  {name}: checkpoint={saved!r}, active={active!r}"
                for name, saved, active in mismatches
            )
            raise ValueError(
                f"Checkpoint/environment incompatibility:\n{details}"
            )

        warning_fields = {
            "config_signature": self.env.get_config_signature(),
            "train_ranges": self.cfg["training"]["train_ranges"],
            "val_range": self.cfg["training"]["val_range"],
            "test_range": self.cfg["training"]["test_range"],
        }
        for field, active_value in warning_fields.items():
            if field in checkpoint and checkpoint[field] != active_value:
                warnings.warn(
                    f"Checkpoint {field} differs from the active configuration: "
                    f"saved={checkpoint[field]!r}, active={active_value!r}",
                    RuntimeWarning,
                )

    def load_model(self, checkpoint, model_path):
        self.agent.policy_net.load_state_dict(checkpoint["policy_net_state_dict"])
        self.agent.policy_net.eval()
        if "model_type" in checkpoint:
            model_type = str(checkpoint["model_type"])
            if model_type != "best_validation":
                warnings.warn(
                    "The selected checkpoint is not labelled 'best_validation'. "
                    "Final paper results should normally use "
                    "ddqn_best.pth.",
                    RuntimeWarning,
                )

    @staticmethod
    def _longest_true_run(mask):
        longest = 0
        current = 0
        for value in np.asarray(mask, dtype=bool):
            if value:
                current += 1
                longest = max(longest, current)
            else:
                current = 0
        return int(longest)

    @staticmethod
    def _day_to_date(day_of_year):
        return pd.to_datetime(f"2018-{int(day_of_year)}", format="%Y-%j").date().isoformat()

    def _select_greedy_action(self, state):
        return self.agent.greedy_action(state)

    def simulate_day(self, day):
        """Run one test day exactly once and copy all available Environment buffers."""
        state = self.env.reset(day=day, mode="test")
        actions = []
        raw_actions = []
        rewards = []

        done = False
        steps = 0
        while not done:
            action = self._select_greedy_action(state)
            raw_action = np.asarray(self.env.all_actions[action], dtype=float).copy()
            next_state, reward, done, _ = self.env.step(action)

            actions.append(action)
            raw_actions.append(raw_action)
            rewards.append(float(reward))

            state = next_state
            steps += 1
            if steps > self.env.max_steps:
                raise RuntimeError(
                    f"Environment exceeded max_steps={self.env.max_steps} on day {day}."
                )

        valid_steps = len(actions)
        if valid_steps != self.env.expected_hours_per_day:
            raise RuntimeError(
                f"Test day {day} produced {valid_steps} steps; expected "
                f"{self.env.expected_hours_per_day}."
            )

        result = {
            "day": int(day),
            "date": self._day_to_date(day),
            "actions": np.asarray(actions, dtype=int),
            "raw_actions": np.asarray(raw_actions, dtype=float),
            "total_rewards": np.asarray(rewards, dtype=float),
            "baseline_per_house": self.env.baseline_per_house[:valid_steps].copy(),
            "after_total_per_house": self.env.after_total_per_house[:valid_steps].copy(),
            "pre_action_total_per_house": (
                self.env.pre_action_total_per_house[:valid_steps].copy()
            ),
            "hourly_capacity_relief": self.env.hourly_capacity_relief[:valid_steps].copy(),
            "signed_load_change": self.env.signed_load_change[:valid_steps].copy(),
            "settlement_dr_energy": self.env.settlement_dr_energy[:valid_steps].copy(),
            "settlement_effective_rate": (
                self.env.settlement_effective_rate[:valid_steps].copy()
            ),
            "action_attributed_dr_energy": (
                self.env.action_attributed_dr_energy[:valid_steps].copy()
            ),
            "incentive_payment": self.env.incentive_payment[:valid_steps].copy(),
            "sp_settlement_wholesale_value": (
                self.env.sp_wholesale_value[:valid_steps].copy()
            ),
            "sp_settlement_incentive_cost": (
                self.env.sp_incentive_cost[:valid_steps].copy()
            ),
            "net_wholesale_procurement_impact": (
                self.env.net_wholesale_procurement_impact[:valid_steps].copy()
            ),
            "reward_shaping_component": self.env.reward_shaping_component[:valid_steps].copy(),
            "reward_base_component": self.env.reward_base_component[:valid_steps].copy(),
            "reward_tracking_component": self.env.reward_tracking_component[:valid_steps].copy(),
            "reward_no_need_component": self.env.reward_no_need_component[:valid_steps].copy(),
            "corrected_economic_reward_component": (
                self.env.corrected_economic_reward_component[:valid_steps].copy()
            ),
            "normalized_economic_component": (
                self.env.normalized_economic_component[:valid_steps].copy()
            ),
            "normalized_discomfort_component": (
                self.env.normalized_discomfort_component[:valid_steps].copy()
            ),
            "raw_discomfort_by_class": (
                self.env.raw_discomfort_by_class[:valid_steps].copy()
            ),
            "normalized_discomfort_by_class": (
                self.env.normalized_discomfort_by_class[:valid_steps].copy()
            ),
            "incentive_offer_intensity": (
                self.env.incentive_offer_intensity[:valid_steps].copy()
            ),
            "incentive_offer_penalty": (
                self.env.incentive_offer_penalty[:valid_steps].copy()
            ),
            "no_need_gate_flag": self.env.no_need_gate_flag[:valid_steps].copy(),
            "no_need_offer_penalty": (
                self.env.no_need_offer_penalty[:valid_steps].copy()
            ),
            "combined_offer_penalty": (
                self.env.combined_offer_penalty[:valid_steps].copy()
            ),
            "baseline_capacity_need": self.env.baseline_capacity_need[:valid_steps].copy(),
            "pre_action_capacity_need": (
                self.env.pre_action_capacity_need[:valid_steps].copy()
            ),
            "baseline_useful_settlement_dr_energy": (
                self.env.baseline_useful_settlement_dr_energy[:valid_steps].copy()
            ),
            "pre_action_useful_settlement_dr_energy": (
                self.env.pre_action_useful_settlement_dr_energy[:valid_steps].copy()
            ),
            "stakeholder_sp_normalized_component": (
                self.env.stakeholder_sp_normalized_component[:valid_steps].copy()
            ),
            "stakeholder_eu_normalized_component": (
                self.env.stakeholder_eu_normalized_component[:valid_steps].copy()
            ),
            "stakeholder_reward_component": (
                self.env.stakeholder_reward_component[:valid_steps].copy()
            ),
            "signed_tracking_error_normalized": (
                self.env.signed_tracking_error_normalized[:valid_steps].copy()
            ),
            "symmetric_tracking_loss": (
                self.env.symmetric_tracking_loss[:valid_steps].copy()
            ),
            "symmetric_tracking_penalty": (
                self.env.symmetric_tracking_penalty[:valid_steps].copy()
            ),
            "useful_settlement_dr_energy": (
                self.env.useful_settlement_dr_energy[:valid_steps].copy()
            ),
            "excess_settlement_dr_energy": (
                self.env.excess_settlement_dr_energy[:valid_steps].copy()
            ),
            "control_target_load": self.env.control_target_load[:valid_steps].copy(),
            "control_over_error": self.env.control_over_error[:valid_steps].copy(),
            "control_under_error": self.env.control_under_error[:valid_steps].copy(),
            "target_tracking_penalty": (
                self.env.target_tracking_penalty[:valid_steps].copy()
            ),
            "unnecessary_incentive_flag": (
                self.env.unnecessary_incentive_flag[:valid_steps].copy()
            ),
            "positive_incentive_no_response_flag": (
                self.env.positive_incentive_no_response_flag[:valid_steps].copy()
            ),
            "maximum_action_flag": self.env.maximum_action_flag[:valid_steps].copy(),
            "net_curtailment_per_house": self.env.net_curtailment_per_house[:valid_steps].copy(),
            "shifted_out_per_house": self.env.shifted_out_per_house[:valid_steps].copy(),
            "shifted_in_per_house": self.env.shifted_in_per_house[:valid_steps].copy(),
            "device_curtailed": self.env.device_curtailed[:valid_steps].copy(),
            "device_shifted_out": self.env.device_shifted_out[:valid_steps].copy(),
            "device_shifted_in": self.env.device_shifted_in[:valid_steps].copy(),
            "device_unshifted_request": self.env.device_unshifted_request[:valid_steps].copy(),
            "device_settlement_ineligible_request": (
                self.env.device_settlement_ineligible_request[:valid_steps].copy()
            ),
            "device_raw_settlement_relief": (
                self.env.device_raw_settlement_relief[:valid_steps].copy()
            ),
            "device_verified_settlement_energy": (
                self.env.device_verified_settlement_energy[:valid_steps].copy()
            ),
            "device_settlement_payment": (
                self.env.device_settlement_payment[:valid_steps].copy()
            ),
            "device_settlement_rate": (
                self.env.device_settlement_rate[:valid_steps].copy()
            ),
            "device_settlement_decision_hour": (
                self.env.device_settlement_decision_hour[:valid_steps].copy()
            ),
            "shifted_out_locked_rate": (
                self.env.shifted_out_locked_rate[:valid_steps].copy()
            ),
            "shifted_out_decision_hour": (
                self.env.shifted_out_decision_hour[:valid_steps].copy()
            ),
            "shifted_out_has_ledger": (
                self.env.shifted_out_has_ledger[:valid_steps].copy()
            ),
            "settlement_allocation_factor": (
                self.env.settlement_allocation_factor[:valid_steps].copy()
            ),
            "device_discomfort": self.env.device_discomfort[:valid_steps].copy(),
            "capacity_overrun_env": self.env.capacity_overrun[:valid_steps].copy(),
            "incentives": self.env.incentives[:valid_steps].copy(),
            "prices": self.env.prices[:valid_steps].copy(),
            "discomforts": self.env.discomforts[:valid_steps].copy(),
            "customer_rewards": self.env.rewards_customers[:valid_steps].copy(),
            "sp_rewards": self.env.rewards_service_provider[:valid_steps].copy(),
            "device_before": self.env.device_before[:valid_steps].copy(),
            "device_after": self.env.device_after[:valid_steps].copy(),
            "nonshift": self.env.nonshift_baseline_per_house[:valid_steps].copy(),
        }

        self.day_results[int(day)] = result
        return result

    def simulate_all_test_days(self):
        print(f"Testing days {self.test_days[0]}-{self.test_days[-1]} | output={self.results_dir}")
        for day in self.test_days:
            self.simulate_day(day)

    def build_report_tables(self):
        """Build one internally consistent physical/economic audit."""
        daily_rows = []
        household_rows = []
        hourly_rows = []
        appliance_rows = []
        settlement_rows = []
        reward_target_rows = []

        for day in self.test_days:
            r = self.day_results[day]
            baseline = r["baseline_per_house"]
            after = r["after_total_per_house"]
            relief = r["hourly_capacity_relief"]
            signed_change = r["signed_load_change"]
            settlement = r["settlement_dr_energy"]
            settlement_rate = r["settlement_effective_rate"]
            action_attributed = r["action_attributed_dr_energy"]
            payment = r["incentive_payment"]
            curtailment_house = r["net_curtailment_per_house"]
            shifted_out_house = r["shifted_out_per_house"]
            shifted_in_house = r["shifted_in_per_house"]
            unshifted_request = r["device_unshifted_request"]
            settlement_ineligible = r["device_settlement_ineligible_request"]
            device_raw_settlement = r["device_raw_settlement_relief"]
            device_verified_settlement = r["device_verified_settlement_energy"]
            device_settlement_payment = r["device_settlement_payment"]
            device_settlement_rate = r["device_settlement_rate"]
            device_settlement_decision_hour = r["device_settlement_decision_hour"]
            incentives = r["incentives"]
            prices = r["prices"]
            discomforts = r["discomforts"]
            customer_rewards = r["customer_rewards"]
            sp_rewards = r["sp_rewards"]
            shaping = r["reward_shaping_component"]
            reward_base = r["reward_base_component"]
            reward_tracking = r["reward_tracking_component"]
            reward_no_need = r["reward_no_need_component"]
            corrected_reward = r["corrected_economic_reward_component"]
            normalized_economic = r["normalized_economic_component"]
            normalized_discomfort = r["normalized_discomfort_component"]
            baseline_capacity_need = r["baseline_capacity_need"]
            pre_action_capacity_need = r["pre_action_capacity_need"]
            baseline_useful_settlement = r[
                "baseline_useful_settlement_dr_energy"
            ]
            pre_action_useful_settlement = r[
                "pre_action_useful_settlement_dr_energy"
            ]
            stakeholder_sp_component = r[
                "stakeholder_sp_normalized_component"
            ]
            stakeholder_eu_component = r[
                "stakeholder_eu_normalized_component"
            ]
            stakeholder_component = r["stakeholder_reward_component"]
            signed_tracking_error = r["signed_tracking_error_normalized"]
            symmetric_tracking_loss = r["symmetric_tracking_loss"]
            symmetric_tracking_penalty = r["symmetric_tracking_penalty"]
            useful_settlement = r["useful_settlement_dr_energy"]
            excess_settlement = r["excess_settlement_dr_energy"]
            control_target = r["control_target_load"]
            control_over_error = r["control_over_error"]
            control_under_error = r["control_under_error"]
            tracking_penalty = r["target_tracking_penalty"]
            unnecessary_incentive = r["unnecessary_incentive_flag"]
            positive_incentive_no_response = r["positive_incentive_no_response_flag"]
            maximum_action = r["maximum_action_flag"]
            no_need_gate = r["no_need_gate_flag"]
            no_need_offer_penalty = r["no_need_offer_penalty"]
            combined_offer_penalty = r["combined_offer_penalty"]
            actions = r["actions"]
            raw_actions = r["raw_actions"]

            if len(prices) != self.env.expected_hours_per_day:
                raise RuntimeError(
                    f"Day {day} contains {len(prices)} test hours; expected "
                    f"{self.env.expected_hours_per_day}."
                )

            agg_baseline = baseline.sum(axis=1)
            agg_after = after.sum(axis=1)
            hourly_capacity_relief = relief.sum(axis=1)
            signed_total = signed_change.sum(axis=1)
            settlement_total = settlement.sum(axis=1)
            action_attributed_total = action_attributed.sum(axis=1)
            payment_total = payment.sum(axis=1)
            net_curtailment = curtailment_house.sum(axis=1)
            shifted_out = shifted_out_house.sum(axis=1)
            shifted_in = shifted_in_house.sum(axis=1)
            unshifted_ev_request = unshifted_request.sum(axis=(1, 2))
            settlement_ineligible_request = settlement_ineligible.sum(axis=(1, 2))

            baseline_overrun = np.maximum(agg_baseline - self.capacity, 0.0)
            overrun = np.maximum(agg_after - self.capacity, 0.0)
            baseline_violation = baseline_overrun > 1e-6
            violation_mask = overrun > 1e-6
            resolved_violation = baseline_violation & ~violation_mask
            remaining_violation = baseline_violation & violation_mask
            newly_created_violation = ~baseline_violation & violation_mask

            shifted_in_present = shifted_in > EPS
            after_without_shifted_in = agg_after - shifted_in
            shifted_in_capacity_safe = shifted_in_present & (
                agg_after <= self.capacity + 1e-6
            )
            rebound_induced_violation = (
                shifted_in_present
                & (agg_baseline <= self.capacity + 1e-6)
                & (agg_after > self.capacity + 1e-6)
                & (after_without_shifted_in <= self.capacity + 1e-6)
            )
            safe_shifted_in_energy = float(
                shifted_in[shifted_in_capacity_safe].sum()
            )
            unsafe_shifted_in_energy = float(
                shifted_in[shifted_in_present & ~shifted_in_capacity_safe].sum()
            )
            total_shifted_in_energy = float(shifted_in.sum())
            safe_shifted_in_ratio = (
                100.0 * safe_shifted_in_energy / total_shifted_in_energy
                if total_shifted_in_energy > EPS
                else np.nan
            )
            maximum_secondary_peak = (
                float(agg_after[shifted_in_present].max())
                if np.any(shifted_in_present)
                else np.nan
            )


            verified_from_physics = np.maximum(0.0, baseline - after)
            physics_to_settlement_error = float(
                np.max(
                    np.abs(settlement - verified_from_physics),
                    initial=0.0,
                )
            )

            independently_reconstructed_wholesale = prices * settlement_total
            independently_reconstructed_payment = device_settlement_payment.sum(axis=(1, 2))
            expected_sp = (
                independently_reconstructed_wholesale
                - independently_reconstructed_payment
            )


            ledger_sp_by_hour = (
                prices[:, None, None] * device_verified_settlement
                - device_settlement_payment
            ).sum(axis=(1, 2))
            expected_customer = (
                self.rho * payment - (1.0 - self.rho) * discomforts
            )

            expected_stakeholder_sp = (
                prices * pre_action_useful_settlement - payment_total
            ) / max(self.env.reward_economic_scale, EPS)
            expected_stakeholder_eu = (
                self.rho * payment_total
                / max(self.env.reward_economic_scale, EPS)
                - (1.0 - self.rho) * normalized_discomfort
            )
            expected_stakeholder = (
                self.env.stakeholder_sp_weight * expected_stakeholder_sp
                + self.env.stakeholder_eu_weight * expected_stakeholder_eu
            )
            expected_signed_tracking_error = (
                agg_after - control_target
            ) / max(self.capacity, EPS)
            abs_tracking_error = np.abs(expected_signed_tracking_error)
            delta = max(self.env.symmetric_tracking_delta, EPS)
            if self.env.symmetric_tracking_loss_name == "smooth_l1":
                expected_symmetric_loss = np.where(
                    abs_tracking_error <= delta,
                    0.5 * expected_signed_tracking_error ** 2 / delta,
                    abs_tracking_error - 0.5 * delta,
                )
            else:
                expected_symmetric_loss = np.where(
                    abs_tracking_error <= delta,
                    0.5 * expected_signed_tracking_error ** 2,
                    delta * (abs_tracking_error - 0.5 * delta),
                )
            expected_symmetric_penalty = (
                self.env.symmetric_tracking_lambda * expected_symmetric_loss
            )

            if self.env.reward_mode in {
                "economic_only", "capacity_penalty", "legacy_shaping"
            }:
                expected_total = (
                    sp_rewards + customer_rewards.sum(axis=1) + shaping
                )
            else:
                expected_total = (
                    reward_base + reward_tracking + reward_no_need + shaping
                )
            stakeholder_sp_reconstruction_error = float(
                np.max(
                    np.abs(stakeholder_sp_component - expected_stakeholder_sp),
                    initial=0.0,
                )
            )
            stakeholder_eu_reconstruction_error = float(
                np.max(
                    np.abs(stakeholder_eu_component - expected_stakeholder_eu),
                    initial=0.0,
                )
            )
            stakeholder_reconstruction_error = float(
                np.max(
                    np.abs(stakeholder_component - expected_stakeholder),
                    initial=0.0,
                )
            )
            signed_tracking_reconstruction_error = float(
                np.max(
                    np.abs(
                        signed_tracking_error - expected_signed_tracking_error
                    ),
                    initial=0.0,
                )
            )
            symmetric_tracking_loss_reconstruction_error = float(
                np.max(
                    np.abs(symmetric_tracking_loss - expected_symmetric_loss),
                    initial=0.0,
                )
            )
            symmetric_tracking_penalty_reconstruction_error = float(
                np.max(
                    np.abs(
                        symmetric_tracking_penalty
                        - expected_symmetric_penalty
                    ),
                    initial=0.0,
                )
            )

            sp_reconstruction_error = float(
                np.max(np.abs(sp_rewards - expected_sp), initial=0.0)
            )
            ledger_sp_value_reconstruction_error = float(
                np.max(
                    np.abs(sp_rewards - ledger_sp_by_hour),
                    initial=0.0,
                )
            )
            customer_reconstruction_error = float(
                np.max(np.abs(customer_rewards - expected_customer), initial=0.0)
            )
            total_reconstruction_error = float(
                np.max(
                    np.abs(r["total_rewards"] - expected_total), initial=0.0
                )
            )
            settlement_energy_reconstruction_error = float(
                np.max(
                    np.abs(
                        settlement - device_verified_settlement.sum(axis=2)
                    ),
                    initial=0.0,
                )
            )
            settlement_payment_reconstruction_error = float(
                np.max(
                    np.abs(payment - device_settlement_payment.sum(axis=2)),
                    initial=0.0,
                )
            )
            wholesale_reconstruction_error = float(
                np.max(
                    np.abs(
                        r["sp_settlement_wholesale_value"]
                        - independently_reconstructed_wholesale
                    ),
                    initial=0.0,
                )
            )
            procurement_reconstruction_error = float(
                np.max(
                    np.abs(
                        r["net_wholesale_procurement_impact"]
                        - prices * signed_total
                    ),
                    initial=0.0,
                )
            )


            shifted_origin_mask = r["device_shifted_out"] > EPS
            missing_ledger_mask = shifted_origin_mask & (
                ~r["shifted_out_has_ledger"]
            )
            invalid_decision_mask = shifted_origin_mask & (
                r["shifted_out_decision_hour"] < 0
            )
            invalid_rate_mask = shifted_origin_mask & (
                ~np.isfinite(r["shifted_out_locked_rate"])
                | (r["shifted_out_locked_rate"] < -EPS)
            )

            missing_ledger_count = int(np.count_nonzero(missing_ledger_mask))
            invalid_decision_count = int(np.count_nonzero(invalid_decision_mask))
            invalid_locked_rate_count = int(np.count_nonzero(invalid_rate_mask))

            audit_errors = {
                "physics_to_settlement": physics_to_settlement_error,
                "ledger_energy": settlement_energy_reconstruction_error,
                "ledger_payment": settlement_payment_reconstruction_error,
                "ledger_SP_value": ledger_sp_value_reconstruction_error,
                "SP_reward": sp_reconstruction_error,
                "settlement_wholesale": wholesale_reconstruction_error,
                "procurement_impact": procurement_reconstruction_error,
                "customer_reward": customer_reconstruction_error,
                "stakeholder_SP": stakeholder_sp_reconstruction_error,
                "stakeholder_EU": stakeholder_eu_reconstruction_error,
                "stakeholder_combined": stakeholder_reconstruction_error,
                "signed_tracking_error": signed_tracking_reconstruction_error,
                "symmetric_tracking_loss": (
                    symmetric_tracking_loss_reconstruction_error
                ),
                "symmetric_tracking_penalty": (
                    symmetric_tracking_penalty_reconstruction_error
                ),
                "total_reward": total_reconstruction_error,
            }
            failed_numeric = {
                name: value
                for name, value in audit_errors.items()
                if (not np.isfinite(value)) or value > AUDIT_TOL
            }
            if failed_numeric:
                details = ", ".join(
                    f"{name}={value:.3e}"
                    for name, value in failed_numeric.items()
                )
                raise RuntimeError(
                    f"Independent settlement/reward audit failed on day {day}: "
                    f"{details}; tolerance={AUDIT_TOL:.1e}."
                )
            if missing_ledger_count or invalid_decision_count or invalid_locked_rate_count:
                raise RuntimeError(
                    f"Settlement ledger integrity failed on day {day}: "
                    f"missing_ledger={missing_ledger_count}, "
                    f"invalid_decision_hour={invalid_decision_count}, "
                    f"invalid_locked_rate={invalid_locked_rate_count}."
                )

            baseline_energy = float(agg_baseline.sum())
            after_energy = float(agg_after.sum())
            signed_change_energy = float(signed_total.sum())

            daily_rows.append({
                "day": day,
                "date": r["date"],
                "peak_baseline": float(agg_baseline.max()),
                "peak_after": float(agg_after.max()),
                "mean_baseline": float(agg_baseline.mean()),
                "mean_after": float(agg_after.mean()),
                "PAR_baseline": float(agg_baseline.max() / max(agg_baseline.mean(), EPS)),
                "PAR_after": float(agg_after.max() / max(agg_after.mean(), EPS)),
                "baseline_violation_steps": int(baseline_violation.sum()),
                "resolved_violation_steps": int(resolved_violation.sum()),
                "remaining_violation_steps": int(remaining_violation.sum()),
                "newly_created_violation_steps": int(newly_created_violation.sum()),
                "violation_steps": int(violation_mask.sum()),
                "violation_percentage": float(100.0 * violation_mask.mean()),
                "baseline_cumulative_excess": float(baseline_overrun.sum()),
                "cumulative_excess": float(overrun.sum()),
                "maximum_overrun": float(overrun.max()),
                "longest_consecutive_violation": self._longest_true_run(violation_mask),
                "baseline_energy": baseline_energy,
                "after_energy": after_energy,
                "signed_load_change": signed_change_energy,
                "energy_identity_error": float(
                    baseline_energy - after_energy - signed_change_energy
                ),
                "positive_baseline_relative_relief": float(
                    np.maximum(signed_total, 0.0).sum()
                ),
                "rebound_energy": float(np.maximum(-signed_total, 0.0).sum()),
                "immediate_action_capacity_relief": float(
                    hourly_capacity_relief.sum()
                ),
                "settlement_DR_energy": float(settlement_total.sum()),
                "action_attributed_DR_energy": float(action_attributed_total.sum()),
                "baseline_capacity_need": float(baseline_capacity_need.sum()),
                "pre_action_capacity_need": float(
                    pre_action_capacity_need.sum()
                ),
                "baseline_useful_settlement_DR_energy": float(
                    baseline_useful_settlement.sum()
                ),
                "pre_action_useful_settlement_DR_energy": float(
                    pre_action_useful_settlement.sum()
                ),
                "SP_pre_action_useful_wholesale_value": float(
                    (prices * pre_action_useful_settlement).sum()
                ),
                "SP_pre_action_useful_net_value": float(
                    (prices * pre_action_useful_settlement).sum()
                    - payment_total.sum()
                ),
                "useful_settlement_DR_energy": float(useful_settlement.sum()),
                "excess_settlement_DR_energy": float(excess_settlement.sum()),
                "target_tracking_MAE_kWh": float(
                    np.mean(control_over_error + control_under_error)
                ),
                "target_tracking_penalty": float(tracking_penalty.sum()),
                "unnecessary_incentive_hours": int(unnecessary_incentive.sum()),
                "positive_incentive_no_response_hours": int(
                    positive_incentive_no_response.sum()
                ),
                "maximum_action_hours": int(maximum_action.sum()),
                "average_offered_incentive_rate": float(incentives.mean()),
                "true_net_curtailment": float(net_curtailment.sum()),
                "shifted_out_energy": float(shifted_out.sum()),
                "shifted_in_energy": float(shifted_in.sum()),
                "capacity_safe_shifted_in_energy": safe_shifted_in_energy,
                "capacity_unsafe_shifted_in_energy": unsafe_shifted_in_energy,
                "capacity_safe_shifted_in_ratio_percentage": float(safe_shifted_in_ratio),
                "rebound_induced_violation_hours": int(rebound_induced_violation.sum()),
                "maximum_post_DR_load_during_shifted_in_hours": maximum_secondary_peak,
                "unshifted_EV_request": float(unshifted_ev_request.sum()),
                "settlement_ineligible_TSNI_request": float(
                    settlement_ineligible_request.sum()
                ),
                "incentive_payment": float(payment_total.sum()),
                "SP_settlement_wholesale_value": float(
                    r["sp_settlement_wholesale_value"].sum()
                ),
                "SP_settlement_incentive_cost": float(
                    r["sp_settlement_incentive_cost"].sum()
                ),
                "SP_settlement_value": float(sp_rewards.sum()),
                "net_wholesale_procurement_impact": float(
                    r["net_wholesale_procurement_impact"].sum()
                ),
                "weighted_customer_utility": float(customer_rewards.sum()),
                "legacy_SP_plus_EU_objective": float(
                    sp_rewards.sum() + customer_rewards.sum()
                ),
                "active_reward_base_before_penalties": float(reward_base.sum()),
                "reward_base_component": float(reward_base.sum()),
                "reward_shaping_component": float(shaping.sum()),
                "reward_tracking_component": float(reward_tracking.sum()),
                "reward_no_need_component": float(reward_no_need.sum()),
                "no_need_gate_hours": int(no_need_gate.sum()),
                "no_need_offer_penalty": float(no_need_offer_penalty.sum()),
                "combined_offer_penalty": float(combined_offer_penalty.sum()),
                "corrected_economic_reward_component": float(corrected_reward.sum()),
                "normalised_economic_component": float(normalized_economic.sum()),
                "normalised_discomfort_component": float(normalized_discomfort.sum()),
                "stakeholder_SP_normalized_component": float(
                    stakeholder_sp_component.sum()
                ),
                "stakeholder_EU_normalized_component": float(
                    stakeholder_eu_component.sum()
                ),
                "stakeholder_reward_component": float(
                    stakeholder_component.sum()
                ),
                "symmetric_tracking_loss": float(
                    symmetric_tracking_loss.sum()
                ),
                "symmetric_tracking_penalty": float(
                    symmetric_tracking_penalty.sum()
                ),
                "environment_total_reward_sum": float(r["total_rewards"].sum()),
                "raw_discomfort_sum": float(discomforts.sum()),
                "max_abs_physics_to_settlement_energy_error": (
                    physics_to_settlement_error
                ),
                "max_abs_SP_reward_reconstruction_error": sp_reconstruction_error,
                "max_abs_ledger_SP_value_reconstruction_error": (
                    ledger_sp_value_reconstruction_error
                ),
                "max_abs_settlement_energy_reconstruction_error": (
                    settlement_energy_reconstruction_error
                ),
                "max_abs_settlement_payment_reconstruction_error": (
                    settlement_payment_reconstruction_error
                ),
                "max_abs_settlement_wholesale_reconstruction_error": (
                    wholesale_reconstruction_error
                ),
                "max_abs_procurement_impact_reconstruction_error": (
                    procurement_reconstruction_error
                ),
                "max_abs_customer_reward_reconstruction_error": customer_reconstruction_error,
                "max_abs_stakeholder_SP_reconstruction_error": (
                    stakeholder_sp_reconstruction_error
                ),
                "max_abs_stakeholder_EU_reconstruction_error": (
                    stakeholder_eu_reconstruction_error
                ),
                "max_abs_stakeholder_combined_reconstruction_error": (
                    stakeholder_reconstruction_error
                ),
                "max_abs_signed_tracking_reconstruction_error": (
                    signed_tracking_reconstruction_error
                ),
                "max_abs_symmetric_tracking_loss_reconstruction_error": (
                    symmetric_tracking_loss_reconstruction_error
                ),
                "max_abs_symmetric_tracking_penalty_reconstruction_error": (
                    symmetric_tracking_penalty_reconstruction_error
                ),
                "max_abs_total_reward_reconstruction_error": total_reconstruction_error,
                "missing_shifted_origin_ledger_entries": missing_ledger_count,
                "invalid_shifted_origin_decision_hours": invalid_decision_count,
                "invalid_shifted_origin_locked_rates": invalid_locked_rate_count,
            })

            for i, hid in enumerate(self.house_ids):
                raw_discomfort_i = float(discomforts[:, i].sum())
                payment_i = float(payment[:, i].sum())
                settlement_wholesale_i = float((prices * settlement[:, i]).sum())
                procurement_impact_i = float((prices * signed_change[:, i]).sum())
                household_rows.append({
                    "day": day,
                    "date": r["date"],
                    "house_id": hid,
                    "baseline_energy": float(baseline[:, i].sum()),
                    "after_energy": float(after[:, i].sum()),
                    "signed_load_change": float(signed_change[:, i].sum()),
                    "immediate_action_capacity_relief": float(relief[:, i].sum()),
                    "settlement_DR_energy": float(settlement[:, i].sum()),
                    "action_attributed_DR_energy": float(
                        action_attributed[:, i].sum()
                    ),
                    "true_net_curtailment": float(curtailment_house[:, i].sum()),
                    "shifted_out_energy": float(shifted_out_house[:, i].sum()),
                    "shifted_in_energy": float(shifted_in_house[:, i].sum()),
                    "unshifted_EV_request": float(unshifted_request[:, i, :].sum()),
                    "average_offered_incentive_rate": float(incentives[:, i].mean()),
                    "average_effective_settlement_rate": float(
                        payment_i / max(float(settlement[:, i].sum()), EPS)
                    ),
                    "incentive_payment": payment_i,
                    "rho_weighted_incentive_benefit": float(self.rho * payment_i),
                    "raw_discomfort": raw_discomfort_i,
                    "one_minus_rho_weighted_discomfort": float(
                        (1.0 - self.rho) * raw_discomfort_i
                    ),
                    "weighted_customer_utility": float(customer_rewards[:, i].sum()),
                    "SP_settlement_wholesale_value_contribution": (
                        settlement_wholesale_i
                    ),
                    "net_wholesale_procurement_impact_contribution": (
                        procurement_impact_i
                    ),
                })

            for h in range(len(prices)):
                row = {
                    "day": day,
                    "date": r["date"],
                    "hour": h + 1,
                    "action_index": int(actions[h]),
                    "price": float(prices[h]),
                    "capacity": self.capacity,
                    "baseline_total": float(agg_baseline[h]),
                    "after_total": float(agg_after[h]),
                    "signed_load_change": float(signed_total[h]),
                    "immediate_action_capacity_relief": float(hourly_capacity_relief[h]),
                    "settlement_DR_energy": float(settlement_total[h]),
                    "action_attributed_DR_energy": float(
                        action_attributed_total[h]
                    ),
                    "incentive_payment": float(payment_total[h]),
                    "SP_settlement_wholesale_value": float(
                        r["sp_settlement_wholesale_value"][h]
                    ),
                    "SP_settlement_incentive_cost": float(
                        r["sp_settlement_incentive_cost"][h]
                    ),
                    "SP_settlement_value": float(sp_rewards[h]),
                    "net_wholesale_procurement_impact": float(
                        r["net_wholesale_procurement_impact"][h]
                    ),
                    "weighted_customer_utility": float(customer_rewards[h].sum()),
                    "baseline_capacity_need": float(baseline_capacity_need[h]),
                    "pre_action_capacity_need": float(
                        pre_action_capacity_need[h]
                    ),
                    "baseline_useful_settlement_DR_energy": float(
                        baseline_useful_settlement[h]
                    ),
                    "pre_action_useful_settlement_DR_energy": float(
                        pre_action_useful_settlement[h]
                    ),
                    "SP_pre_action_useful_wholesale_value": float(
                        prices[h] * pre_action_useful_settlement[h]
                    ),
                    "SP_pre_action_useful_net_value": float(
                        prices[h] * pre_action_useful_settlement[h]
                        - payment_total[h]
                    ),
                    "useful_settlement_DR_energy": float(useful_settlement[h]),
                    "excess_settlement_DR_energy": float(excess_settlement[h]),
                    "control_target_load": float(control_target[h]),
                    "control_over_error": float(control_over_error[h]),
                    "control_under_error": float(control_under_error[h]),
                    "target_tracking_penalty": float(tracking_penalty[h]),
                    "unnecessary_incentive": bool(unnecessary_incentive[h]),
                    "positive_incentive_no_response": bool(
                        positive_incentive_no_response[h]
                    ),
                    "maximum_action": bool(maximum_action[h]),
                    "normalised_economic_component": float(normalized_economic[h]),
                    "normalised_discomfort_component": float(normalized_discomfort[h]),
                    "stakeholder_SP_normalized_component": float(
                        stakeholder_sp_component[h]
                    ),
                    "stakeholder_EU_normalized_component": float(
                        stakeholder_eu_component[h]
                    ),
                    "stakeholder_reward_component": float(
                        stakeholder_component[h]
                    ),
                    "signed_tracking_error_normalized": float(
                        signed_tracking_error[h]
                    ),
                    "symmetric_tracking_loss": float(
                        symmetric_tracking_loss[h]
                    ),
                    "symmetric_tracking_penalty": float(
                        symmetric_tracking_penalty[h]
                    ),
                    "corrected_economic_reward_component": float(corrected_reward[h]),
                    "reward_base_component": float(reward_base[h]),
                    "reward_shaping_component": float(shaping[h]),
                    "reward_tracking_component": float(reward_tracking[h]),
                    "reward_no_need_component": float(reward_no_need[h]),
                    "no_need_gate": bool(no_need_gate[h]),
                    "no_need_offer_penalty": float(no_need_offer_penalty[h]),
                    "combined_offer_penalty": float(combined_offer_penalty[h]),
                    "environment_total_reward": float(r["total_rewards"][h]),
                    "true_hourly_curtailment": float(net_curtailment[h]),
                    "shifted_out_this_hour": float(shifted_out[h]),
                    "shifted_in_this_hour": float(shifted_in[h]),
                    "shifted_in_present": bool(shifted_in_present[h]),
                    "post_DR_load_without_shifted_in": float(after_without_shifted_in[h]),
                    "shifted_in_capacity_safe": bool(shifted_in_capacity_safe[h]),
                    "rebound_induced_violation": bool(rebound_induced_violation[h]),
                    "unshifted_EV_request_this_hour": float(unshifted_ev_request[h]),
                    "settlement_ineligible_TSNI_request_this_hour": float(
                        settlement_ineligible_request[h]
                    ),
                    "baseline_capacity_overrun": float(baseline_overrun[h]),
                    "capacity_overrun": float(overrun[h]),
                    "baseline_capacity_violation": bool(baseline_violation[h]),
                    "capacity_violation": bool(violation_mask[h]),
                    "resolved_capacity_violation": bool(resolved_violation[h]),
                    "remaining_capacity_violation": bool(remaining_violation[h]),
                    "newly_created_capacity_violation": bool(newly_created_violation[h]),
                }
                for i, hid in enumerate(self.house_ids):
                    row[f"raw_action_house_{hid}"] = float(raw_actions[h, i])
                    row[f"incentive_house_{hid}"] = float(incentives[h, i])
                    row[f"baseline_house_{hid}"] = float(baseline[h, i])
                    row[f"after_house_{hid}"] = float(after[h, i])
                    row[f"signed_load_change_house_{hid}"] = float(signed_change[h, i])
                    row[f"capacity_relief_house_{hid}"] = float(relief[h, i])
                    row[f"settlement_DR_house_{hid}"] = float(settlement[h, i])
                    row[f"effective_settlement_rate_house_{hid}"] = float(
                        settlement_rate[h, i]
                    )
                    row[f"action_attributed_DR_house_{hid}"] = float(
                        action_attributed[h, i]
                    )
                    row[f"incentive_payment_house_{hid}"] = float(payment[h, i])
                    row[f"curtailment_house_{hid}"] = float(curtailment_house[h, i])
                    row[f"shifted_out_house_{hid}"] = float(shifted_out_house[h, i])
                    row[f"shifted_in_house_{hid}"] = float(shifted_in_house[h, i])
                    row[f"unshifted_EV_request_house_{hid}"] = float(
                        unshifted_request[h, i, :].sum()
                    )
                    row[f"raw_discomfort_house_{hid}"] = float(discomforts[h, i])
                    row[f"weighted_customer_utility_house_{hid}"] = float(
                        customer_rewards[h, i]
                    )
                hourly_rows.append(row)
                reward_target_rows.append({
                    "day": day,
                    "date": r["date"],
                    "hour": h + 1,
                    "reward_mode": self.env.reward_mode,
                    "action_index": int(actions[h]),
                    "baseline_total": float(agg_baseline[h]),
                    "pre_action_committed_total": float(
                        r["pre_action_total_per_house"][h].sum()
                    ),
                    "post_DR_total": float(agg_after[h]),
                    "capacity": float(self.capacity),
                    "control_target_load": float(control_target[h]),
                    "baseline_capacity_need": float(baseline_capacity_need[h]),
                    "pre_action_capacity_need": float(
                        pre_action_capacity_need[h]
                    ),
                    "baseline_useful_settlement_DR_energy": float(
                        baseline_useful_settlement[h]
                    ),
                    "pre_action_useful_settlement_DR_energy": float(
                        pre_action_useful_settlement[h]
                    ),
                    "SP_pre_action_useful_wholesale_value": float(
                        prices[h] * pre_action_useful_settlement[h]
                    ),
                    "SP_pre_action_useful_net_value": float(
                        prices[h] * pre_action_useful_settlement[h]
                        - payment_total[h]
                    ),
                    "useful_settlement_DR_energy": float(useful_settlement[h]),
                    "excess_settlement_DR_energy": float(excess_settlement[h]),
                    "control_over_error": float(control_over_error[h]),
                    "control_under_error": float(control_under_error[h]),
                    "target_tracking_penalty": float(tracking_penalty[h]),
                    "incentive_payment": float(payment_total[h]),
                    "raw_discomfort": float(discomforts[h].sum()),
                    "normalised_economic_component": float(normalized_economic[h]),
                    "normalised_discomfort_component": float(normalized_discomfort[h]),
                    "reward_base_component": float(reward_base[h]),
                    "reward_tracking_component": float(reward_tracking[h]),
                    "reward_no_need_component": float(reward_no_need[h]),
                    "no_need_gate": bool(no_need_gate[h]),
                    "no_need_offer_penalty": float(no_need_offer_penalty[h]),
                    "combined_offer_penalty": float(combined_offer_penalty[h]),
                    "total_reward": float(r["total_rewards"][h]),
                    "unnecessary_incentive": bool(unnecessary_incentive[h]),
                    "positive_incentive_no_response": bool(
                        positive_incentive_no_response[h]
                    ),
                    "maximum_action": bool(maximum_action[h]),
                })


            device_names = [str(value) for value in self.env.DEVICES]
            for h in range(len(prices)):
                for i, hid in enumerate(self.house_ids):
                    for d, device_name in enumerate(device_names):
                        raw_source = float(device_raw_settlement[h, i, d])
                        verified_source = float(
                            device_verified_settlement[h, i, d]
                        )
                        ineligible_source = float(
                            settlement_ineligible[h, i, d]
                        )
                        if (
                            raw_source <= EPS
                            and verified_source <= EPS
                            and ineligible_source <= EPS
                        ):
                            continue

                        curtailed_source = float(
                            r["device_curtailed"][h, i, d]
                        )
                        shifted_source = float(
                            r["device_shifted_out"][h, i, d]
                        )
                        locked_shift_rate = float(
                            r["shifted_out_locked_rate"][h, i, d]
                        )
                        raw_payment = (
                            curtailed_source * float(incentives[h, i])
                            + shifted_source * locked_shift_rate
                        )
                        raw_rate = (
                            raw_payment / raw_source
                            if raw_source > EPS
                            else np.nan
                        )
                        decision_hour = int(
                            device_settlement_decision_hour[h, i, d]
                        )
                        settlement_rows.append({
                            "day": day,
                            "date": r["date"],
                            "verification_hour": h + 1,
                            "house_id": hid,
                            "appliance": device_name,
                            "decision_hour": (
                                decision_hour + 1 if decision_hour >= 0 else np.nan
                            ),
                            "verification_hour_price": float(prices[h]),
                            "current_offered_incentive_rate": float(
                                incentives[h, i]
                            ),
                            "locked_or_source_rate": raw_rate,
                            "raw_attributed_relief": raw_source,
                            "household_allocation_factor": float(
                                r["settlement_allocation_factor"][h, i]
                            ),
                            "verified_settlement_energy": verified_source,
                            "settlement_payment": float(
                                device_settlement_payment[h, i, d]
                            ),
                            "SP_settlement_value_contribution": float(
                                prices[h] * verified_source
                                - device_settlement_payment[h, i, d]
                            ),
                            "current_curtailment_source": curtailed_source,
                            "shifted_out_origin_source": shifted_source,
                            "settlement_ineligible_TSNI_request": (
                                ineligible_source
                            ),
                        })

            before_dev = r["device_before"]
            after_dev = r["device_after"]
            curtailed_dev = r["device_curtailed"]
            shifted_out_dev = r["device_shifted_out"]
            shifted_in_dev = r["device_shifted_in"]
            unshifted_dev = r["device_unshifted_request"]
            ineligible_dev = r["device_settlement_ineligible_request"]
            verified_settlement_dev = r["device_verified_settlement_energy"]
            settlement_payment_dev = r["device_settlement_payment"]
            settlement_rate_dev = r["device_settlement_rate"]
            settlement_decision_dev = r["device_settlement_decision_hour"]
            device_names = [str(value) for value in self.env.DEVICES]
            for i, hid in enumerate(self.house_ids):
                for d, device_name in enumerate(device_names):
                    baseline_device = before_dev[:, i, d]
                    after_device = after_dev[:, i, d]
                    curtailed = float(curtailed_dev[:, i, d].sum())
                    shifted_out_d = float(shifted_out_dev[:, i, d].sum())
                    shifted_in_d = float(shifted_in_dev[:, i, d].sum())
                    unshifted_d = float(unshifted_dev[:, i, d].sum())
                    ineligible_d = float(ineligible_dev[:, i, d].sum())
                    verified_settlement_d = float(
                        verified_settlement_dev[:, i, d].sum()
                    )
                    settlement_payment_d = float(
                        settlement_payment_dev[:, i, d].sum()
                    )
                    positive_settlement_mask = (
                        verified_settlement_dev[:, i, d] > EPS
                    )
                    weighted_rate_d = (
                        settlement_payment_d / verified_settlement_d
                        if verified_settlement_d > EPS
                        else np.nan
                    )
                    decision_hours = settlement_decision_dev[:, i, d]
                    appliance_rows.append({
                        "day": day,
                        "date": r["date"],
                        "house_id": hid,
                        "appliance": device_name,
                        "baseline_energy": float(baseline_device.sum()),
                        "after_energy": float(after_device.sum()),
                        "true_curtailment": curtailed,
                        "shifted_out_energy": shifted_out_d,
                        "shifted_in_energy": shifted_in_d,
                        "unshifted_EV_request": unshifted_d,
                        "settlement_ineligible_TSNI_request": ineligible_d,
                        "verified_settlement_energy": verified_settlement_d,
                        "settlement_payment": settlement_payment_d,
                        "weighted_average_settlement_rate": weighted_rate_d,
                        "number_of_settled_hours": int(
                            positive_settlement_mask.sum()
                        ),
                        "earliest_settlement_decision_hour": (
                            int(decision_hours[decision_hours >= 0].min()) + 1
                            if np.any(decision_hours >= 0)
                            else np.nan
                        ),
                        "energy_balance_error": float(
                            baseline_device.sum() - after_device.sum() - curtailed
                        ),
                    })

        tables = {
            "Daily_Summary": pd.DataFrame(daily_rows),
            "Household_Daily": pd.DataFrame(household_rows),
            "Hourly_Audit": pd.DataFrame(hourly_rows),
            "Appliance_Diagnostic": pd.DataFrame(appliance_rows),
            "Settlement_Ledger": pd.DataFrame(settlement_rows),
            "Reward_Target_Audit": pd.DataFrame(reward_target_rows),
        }
        self._validate_report_row_counts(tables)
        return tables

    def _validate_report_row_counts(self, tables):
        n_days = len(self.test_days)
        expected = {
            "Daily_Summary": n_days,
            "Household_Daily": n_days * len(self.house_ids),
            "Hourly_Audit": n_days * self.env.expected_hours_per_day,
            "Appliance_Diagnostic": (
                n_days * len(self.house_ids) * self.env.num_devices
            ),
            "Reward_Target_Audit": n_days * self.env.expected_hours_per_day,
        }
        actual = {name: len(tables[name]) for name in expected}
        if actual != expected:
            raise RuntimeError(
                f"Report row-count validation failed: expected={expected}, actual={actual}"
            )


        expected_ledger_rows = 0
        for result in self.day_results.values():
            include_mask = (
                (result["device_raw_settlement_relief"] > EPS)
                | (result["device_verified_settlement_energy"] > EPS)
                | (result["device_settlement_ineligible_request"] > EPS)
            )
            expected_ledger_rows += int(np.count_nonzero(include_mask))

        actual_ledger_rows = len(tables["Settlement_Ledger"])
        if actual_ledger_rows != expected_ledger_rows:
            raise RuntimeError(
                "Settlement_Ledger row-count validation failed: "
                f"expected={expected_ledger_rows}, actual={actual_ledger_rows}."
            )

        any_source_activity = expected_ledger_rows > 0
        if any_source_activity and tables["Settlement_Ledger"].empty:
            raise RuntimeError(
                "Settlement_Ledger is empty despite source-level settlement "
                "or ineligible-request activity."
            )


    def save_excel_report(self, tables):
        """Save the complete 31-day audit workbook used for traceability.

        This is the only detailed Excel output.  It preserves the full daily,
        household, hourly, appliance, action, reward, and diagnostic records.
        """
        excel_path = os.path.join(
            self.results_dir,
            "audit.xlsx",
        )

        notes = pd.DataFrame({
            "Important_notes": [                "This workbook covers the complete configured test range with exactly 24 hourly observations per day.",
                "immediate_action_capacity_relief is the non-negative relief caused by the current action and is used only for capacity analysis and reward shaping.",
                "signed_load_change equals baseline minus realised post-DR load; it may be negative during shifted-in hours and is used only for energy-balance, rebound, and net-procurement diagnostics.",
                "settlement_DR_energy equals max(0, baseline minus realised post-DR load) at household-hour level and is allocated proportionally across the physical relief sources active in that hour.",
                "Multi-hour shifted jobs use the incentive rate locked when the job was moved; each origin-hour slice is valued at its own verification-hour wholesale price.",
                "action_attributed_DR_energy equals current curtailment plus the full energy of jobs newly shifted by the current action and is a diagnostic only.",
                "SP_settlement_value is the origin-hour capacity-relief settlement value: verified relief valued at the origin-hour price minus incentive payment. net_wholesale_procurement_impact separately includes shifted-in destination effects.",
                "weighted_customer_utility equals rho*incentive_payment minus (1-rho)*discomfort.",
                "For corrected_economic and target_tracking, the training objective is separate from the reporting-only weighted_customer_utility.",
                "Useful settlement DR equals min(total verified settlement DR, baseline capacity need); the full verified incentive payment is still charged.",
                "The control target equals min(pre-action committed load, capacity). The dead-band is active only in overload hours.",
                "The no-need regularizer applies a non-monetary action-parsimony cost only when pre-action committed load is at or below capacity plus the numerical tolerance.",
                "The no-need regularizer changes the training objective but does not alter settlement energy, incentive payment, capacity-relief value, customer utility, discomfort, or physical accounting definitions.",
                "No social-welfare column is reported because a complete bill-saving and system-cost welfare definition has not yet been adopted.",
                "The independent physics-to-settlement check recomputes max(0, baseline_house - after_house) without using Environment settlement arrays.",
                "Ledger energy, payment, SP-value, customer-reward, and total-reward reconstruction errors must not exceed the strict audit tolerance.",
                f"Strict numerical audit tolerance: {AUDIT_TOL:.1e}.",
                "Detailed figures are generated only for day(s): "
                + ", ".join(str(day) for day in self.target_plot_days) + ".",
                f"Configured test range: {self.cfg['training']['test_range'][0]}-{self.cfg['training']['test_range'][1]}; active test range: {self.test_days[0]}-{self.test_days[-1]}.",
            ]
        })

        for sheet_name, df in tables.items():
            if df.empty and sheet_name != "Settlement_Ledger":
                raise ValueError(
                    f"Report table '{sheet_name}' is empty; audit workbook was not written."
                )

        try:
            from openpyxl import load_workbook
            from openpyxl.styles import Alignment, Border, Font, PatternFill, Side

            with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
                for sheet_name, df in tables.items():
                    df.to_excel(writer, sheet_name=sheet_name[:31], index=False)
                notes.to_excel(writer, sheet_name="README", index=False)

                header_fill = PatternFill("solid", fgColor="1F4E78")
                header_font = Font(color="FFFFFF", bold=True)
                thin_gray = Side(style="thin", color="D9E1F2")
                header_border = Border(bottom=thin_gray)

                for ws in writer.book.worksheets:
                    ws.freeze_panes = "A2"
                    ws.auto_filter.ref = ws.dimensions
                    ws.sheet_view.showGridLines = False

                    for cell in ws[1]:
                        cell.fill = header_fill
                        cell.font = header_font
                        cell.alignment = Alignment(
                            horizontal="center", vertical="center", wrap_text=True
                        )
                        cell.border = header_border

                    ws.row_dimensions[1].height = 36
                    for column_cells in ws.columns:
                        letter = column_cells[0].column_letter
                        max_len = 0
                        for cell in column_cells[:250]:
                            value = "" if cell.value is None else str(cell.value)
                            max_len = max(max_len, len(value))
                        ws.column_dimensions[letter].width = min(
                            max(max_len + 2, 11), 34
                        )

                    for row in ws.iter_rows(min_row=2):
                        for cell in row:
                            cell.alignment = Alignment(vertical="center")
                            if isinstance(cell.value, float):
                                cell.number_format = "0.0000"

                writer.book.active = writer.book.sheetnames.index("Daily_Summary")

            check_wb = load_workbook(excel_path, read_only=True, data_only=True)
            counts = {
                name: max(check_wb[name].max_row - 1, 0)
                for name in [
                    "Daily_Summary",
                    "Household_Daily",
                    "Hourly_Audit",
                    "Appliance_Diagnostic",
                    "Settlement_Ledger",
                    "Reward_Target_Audit",
                ]
            }
            check_wb.close()
        except ImportError as exc:
            raise ImportError(
                "openpyxl is required to create the Excel reports. "
                "Install it with: pip install openpyxl"
            ) from exc

        return excel_path

    def plot_day_profiles(self, result):
        day = result["day"]
        baseline = result["baseline_per_house"]
        after = result["after_total_per_house"]
        incentives = result["incentives"]
        prices = result["prices"]


        d_color = "#FFDD00"
        c_color = "#919291"
        price_color = "#D62728"
        inc_color = "#1F77B4"

        hours = np.arange(1, len(prices) + 1)
        xticks = np.arange(1, len(prices) + 1, BIN_SIZE)

        for i, hid in enumerate(self.house_ids):
            original_load = baseline[:, i]
            consumption = after[:, i]
            actual_incentive = incentives[:, i]

            fig, ax1 = plt.subplots(figsize=(10, 6))
            ax1.fill_between(
                hours, original_load, 0,
                color=d_color, alpha=0.7, label="Demand", zorder=1,
            )
            ax1.fill_between(
                hours, consumption, 0,
                color=c_color, alpha=0.2, label="_nolegend_", zorder=2,
            )
            ax1.fill_between(
                hours, consumption, 0,
                facecolor="none", edgecolor="black", hatch="/", linewidth=2,
                label="Consumption", zorder=3,
            )
            ax1.plot(hours, original_load, color=d_color, linewidth=2, zorder=4)
            ax1.plot(hours, consumption, color=c_color, linewidth=1.5, zorder=4)
            ax1.set_xlabel("Hour")
            ax1.set_ylabel("Energy (kWh)")
            ax1.set_xticks(xticks)
            ax1.set_xlim(1, len(prices))
            ymax_left = max(original_load.max(), consumption.max(), EPS) * 1.1
            ax1.set_ylim(0, ymax_left)
            ax1.grid(False)

            ax2 = ax1.twinx()
            ax2.plot(
                hours, prices, marker="o", color=price_color,
                linewidth=3, markersize=6, label="Wholesale Price", zorder=5,
            )
            ax2.plot(
                hours, actual_incentive, marker="s", color=inc_color,
                linewidth=3, markersize=6, label="Incentive Rate", zorder=5,
            )
            ax2.set_ylabel("Rate (¢/kWh)")
            ymax_right = max(float(prices.max()), float(actual_incentive.max()), EPS)
            ax2.set_ylim(0, ymax_right * 1.2)

            h1, l1 = ax1.get_legend_handles_labels()
            h2, l2 = ax2.get_legend_handles_labels()
            ax1.legend(h1 + h2, l1 + l2, loc="upper left", framealpha=0.9)

            plt.tight_layout()
            out = os.path.join(
                self.results_dir,
                f"energy_profile_house_{hid}_day_{day}_rho{self.rho}.png",
            )
            plt.savefig(out, dpi=300, bbox_inches="tight")
            plt.close(fig)

    def plot_aggregated_load(self, result):
        day = result["day"]
        agg_baseline = result["baseline_per_house"].sum(axis=1)
        agg_after = result["after_total_per_house"].sum(axis=1)
        hours = np.arange(1, len(agg_baseline) + 1)
        xticks = np.arange(1, len(agg_baseline) + 1, BIN_SIZE)

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(
            hours, agg_baseline, marker="o", color="red", linewidth=2,
            markersize=6, label="Aggregated Original Load",
        )
        ax.plot(
            hours, agg_after, marker="o", color="blue", linestyle="--",
            linewidth=2, markersize=6, label="Aggregated Load After Reduction",
        )
        ax.hlines(
            self.capacity, xmin=hours.min(), xmax=hours.max(), colors="green",
            linestyles="--", linewidth=2,
            label=f"Capacity Threshold ({self.capacity:g} kW)",
        )
        ax.set_xlabel("Hour")
        ax.set_ylabel("Aggregated load (kW)")
        ax.set_xticks(xticks)
        ax.set_xlim(1, len(agg_baseline))
        ymax = max(agg_baseline.max(), agg_after.max(), self.capacity, EPS) * 1.1
        ax.set_ylim(0, ymax)
        ax.grid(False)
        ax.legend(loc="upper right")
        plt.tight_layout()

        out = os.path.join(
            self.results_dir,
            f"aggregated_load_day_{day}_rho{self.rho}.png",
        )
        plt.savefig(out, dpi=300, bbox_inches="tight")
        plt.close(fig)

    @staticmethod
    def _device_display_name(name):
        """Return compact appliance abbreviations used in the manuscript."""
        key = str(name).strip().lower().replace("_", " ").replace("-", " ")
        compact = "".join(key.split())
        aliases = {
            "air": "AC",
            "ac": "AC",
            "airconditioner": "AC",
            "car": "EV",
            "ev": "EV",
            "electricvehicle": "EV",
            "clotheswasher": "WM",
            "washingmachine": "WM",
            "washer": "WM",
            "dishwasher": "DW",
            "dry": "DR",
            "dryer": "DR",
            "clothesdryer": "DR",
        }
        return aliases.get(compact, str(name))

    def plot_device_schedule_by_house(self, result):
        """Create one clear appliance-level figure for each household on day 208.

        Each household figure contains one panel per appliance and reports hourly
        energy in kWh.  The legend distinguishes the original appliance demand,
        realised demand after DR, true curtailment, shifted-out energy, and
        shifted-in energy.  Empty appliance panels are explicitly marked as
        "No operation" instead of displaying misleading scientific notation.
        """
        from matplotlib.patches import Patch
        from matplotlib.ticker import MaxNLocator, ScalarFormatter

        day = int(result["day"])
        if day != self.target_plot_day:
            return

        before = np.asarray(result["device_before"], dtype=float)
        after = np.asarray(result["device_after"], dtype=float)
        curtailed = np.asarray(result["device_curtailed"], dtype=float)
        shifted_out = np.asarray(result["device_shifted_out"], dtype=float)
        shifted_in = np.asarray(result["device_shifted_in"], dtype=float)

        if before.ndim != 3:
            raise ValueError(
                "device_before must have shape (hours, households, devices)."
            )
        if not (
            before.shape == after.shape == curtailed.shape
            == shifted_out.shape == shifted_in.shape
        ):
            raise ValueError("All device-level result arrays must have identical shapes.")

        hours = np.arange(1, before.shape[0] + 1)
        xticks = np.arange(1, before.shape[0] + 1, BIN_SIZE)
        device_names = [self._device_display_name(x) for x in self.env.DEVICES]
        num_devices = before.shape[2]

        if len(device_names) != num_devices:
            raise ValueError(
                "The number of configured device names does not match the result arrays."
            )

        legend_handles = [
            Patch(
                facecolor="#D9D9D9", edgecolor="#7A7A7A",
                linewidth=1.2, label="Baseline",
            ),
            Patch(
                facecolor="#1F77B4", edgecolor="black",
                linewidth=1.1, label="After DR",
            ),
            Patch(
                facecolor="white", edgecolor="#D62728",
                hatch="///", linewidth=1.7, label="Curtailed",
            ),
            Patch(
                facecolor="white", edgecolor="#FF7F0E",
                hatch="xx", linewidth=1.7, label="Shifted out",
            ),
            Patch(
                facecolor="white", edgecolor="#2CA02C",
                hatch="\\", linewidth=1.7, label="Shifted in",
            ),
        ]

        for house_index, house_id in enumerate(self.house_ids):
            fig, axes = plt.subplots(
                num_devices,
                1,
                figsize=(10, 6),
                sharex=True,
                gridspec_kw={"hspace": 0.08},
            )
            axes = np.atleast_1d(axes)

            for device_index, (ax, device_label) in enumerate(
                zip(axes, device_names)
            ):
                baseline_hd = before[:, house_index, device_index]
                after_hd = after[:, house_index, device_index]
                curtailed_hd = curtailed[:, house_index, device_index]
                shifted_out_hd = shifted_out[:, house_index, device_index]
                shifted_in_hd = shifted_in[:, house_index, device_index]

                baseline_mask = baseline_hd > EPS
                after_mask = after_hd > EPS
                curtailed_mask = curtailed_hd > EPS
                shifted_out_mask = shifted_out_hd > EPS
                shifted_in_mask = shifted_in_hd > EPS

                bar_width = 0.36
                baseline_x = hours - bar_width / 2
                after_x = hours + bar_width / 2

                ax.bar(
                    baseline_x[baseline_mask],
                    baseline_hd[baseline_mask],
                    width=bar_width,
                    color="#D9D9D9",
                    edgecolor="#7A7A7A",
                    linewidth=1.1,
                    zorder=1,
                    align="center",
                )
                ax.bar(
                    after_x[after_mask],
                    after_hd[after_mask],
                    width=bar_width,
                    color="#1F77B4",
                    edgecolor="black",
                    linewidth=1.0,
                    zorder=2,
                    align="center",
                )
                ax.bar(
                    baseline_x[curtailed_mask],
                    curtailed_hd[curtailed_mask],
                    width=bar_width * 0.42,
                    facecolor="none",
                    edgecolor="#D62728",
                    hatch="///",
                    linewidth=1.6,
                    zorder=4,
                    align="center",
                )
                ax.bar(
                    baseline_x[shifted_out_mask],
                    shifted_out_hd[shifted_out_mask],
                    width=bar_width * 0.42,
                    facecolor="none",
                    edgecolor="#FF7F0E",
                    hatch="xx",
                    linewidth=1.6,
                    zorder=4,
                    align="center",
                )
                ax.bar(
                    after_x[shifted_in_mask],
                    shifted_in_hd[shifted_in_mask],
                    width=bar_width * 0.42,
                    facecolor="none",
                    edgecolor="#2CA02C",
                    hatch="\\",
                    linewidth=1.6,
                    zorder=4,
                    align="center",
                )

                local_max = max(
                    float(np.max(baseline_hd, initial=0.0)),
                    float(np.max(after_hd, initial=0.0)),
                    float(np.max(curtailed_hd, initial=0.0)),
                    float(np.max(shifted_out_hd, initial=0.0)),
                    float(np.max(shifted_in_hd, initial=0.0)),
                )

                if local_max <= EPS:
                    ax.set_ylim(0.0, 1.0)
                    ax.set_yticks([0.0])
                    ax.text(
                        0.5,
                        0.48,
                        "No operation",
                        transform=ax.transAxes,
                        ha="center",
                        va="center",
                        fontsize=16,
                        fontweight="bold",
                        color="#555555",
                    )
                else:
                    ax.set_ylim(0.0, local_max * 1.22)
                    ax.yaxis.set_major_locator(MaxNLocator(nbins=3, min_n_ticks=2))

                formatter = ScalarFormatter(useOffset=False)
                formatter.set_scientific(False)
                ax.yaxis.set_major_formatter(formatter)
                ax.ticklabel_format(axis="y", style="plain", useOffset=False)

                ax.set_xlim(0.5, len(hours) + 0.5)
                ax.grid(False)
                ax.text(
                    0.012,
                    0.84,
                    device_label,
                    transform=ax.transAxes,
                    ha="left",
                    va="top",
                    fontsize=21,
                    fontweight="bold",
                    bbox={
                        "facecolor": "white",
                        "edgecolor": "none",
                        "alpha": 0.86,
                        "pad": 1.0,
                    },
                    zorder=6,
                )

                ax.tick_params(
                    axis="y",
                    which="major",
                    labelsize=17,
                    width=1.8,
                    length=6,
                    direction="out",
                )
                ax.tick_params(
                    axis="x",
                    which="major",
                    labelsize=21,
                    width=1.8,
                    length=7,
                    direction="out",
                )
                for tick in ax.get_yticklabels():
                    tick.set_fontweight("bold")
                for spine in ax.spines.values():
                    spine.set_linewidth(1.8)

            axes[-1].set_xticks(xticks)
            axes[-1].set_xlabel("Hour", fontsize=26, fontweight="bold")
            for tick in axes[-1].get_xticklabels():
                tick.set_fontweight("bold")

            fig.supylabel(
                "Energy (kWh)",
                x=0.012,
                fontsize=26,
                fontweight="bold",
            )
            fig.legend(
                handles=legend_handles,
                loc="upper center",
                bbox_to_anchor=(0.56, 0.995),
                ncol=5,
                frameon=False,
                fontsize=15,
                handlelength=2.0,
                handleheight=1.2,
                columnspacing=1.1,
            )
            fig.subplots_adjust(
                left=0.12,
                right=0.985,
                bottom=0.13,
                top=0.90,
                hspace=0.08,
            )

            output_path = os.path.join(
                self.results_dir,
                f"device_schedule_house_{house_id}_day_{day}.png",
            )
            fig.savefig(output_path, dpi=300, bbox_inches="tight")
            plt.close(fig)

    def plot_total_incentive_rate(self, result):
        day = result["day"]
        incentives = result["incentives"]
        hours = np.arange(1, incentives.shape[0] + 1)
        xticks = np.arange(1, incentives.shape[0] + 1, BIN_SIZE)

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(hours, incentives.sum(axis=1), color="#1F77B4", linewidth=3)
        ax.set_xlabel("Hour")
        ax.set_ylabel("Incentives (¢/kWh)")
        ax.set_xticks(xticks)
        ax.set_xlim(1, len(hours))
        ax.grid(False)
        plt.tight_layout()

        out = os.path.join(self.results_dir, f"total_incentive_rate_day_{day}.png")
        plt.savefig(out, dpi=300, bbox_inches="tight")
        plt.close(fig)

    @staticmethod
    def _write_xlsx(path, sheets):
        from openpyxl.styles import Alignment, Font, PatternFill
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with pd.ExcelWriter(path, engine="openpyxl") as writer:
            for name, df in sheets.items():
                df.to_excel(writer, sheet_name=name[:31], index=False)
            for ws in writer.book.worksheets:
                ws.freeze_panes = "A2"
                ws.auto_filter.ref = ws.dimensions
                ws.sheet_view.showGridLines = False
                for cell in ws[1]:
                    cell.fill = PatternFill("solid", fgColor="1F4E78")
                    cell.font = Font(color="FFFFFF", bold=True)
                    cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
                for column in ws.columns:
                    letter = column[0].column_letter
                    width = max((len(str(c.value)) if c.value is not None else 0) for c in column[:250]) + 2
                    ws.column_dimensions[letter].width = min(max(width, 11), 34)
                for row in ws.iter_rows(min_row=2):
                    for cell in row:
                        if isinstance(cell.value, float):
                            cell.number_format = "0.0000"

    @staticmethod
    def _mean_pm_sd(values):
        values = np.asarray(values, dtype=float)
        return f"{values.mean():.2f} ± {values.std(ddof=0):.2f}"

    def _build_paper_ready_tables(self, tables):
        """Build the manuscript tables from the single Environment accounting."""
        daily = tables["Daily_Summary"].copy()
        household = tables["Household_Daily"].copy()
        hourly = tables["Hourly_Audit"].copy()
        appliance = tables["Appliance_Diagnostic"].copy()
        n_days = len(self.test_days)

        all_baseline = np.concatenate([
            self.day_results[day]["baseline_per_house"].sum(axis=1)
            for day in self.test_days
        ])
        all_after = np.concatenate([
            self.day_results[day]["after_total_per_house"].sum(axis=1)
            for day in self.test_days
        ])
        baseline_compliance = 100.0 * np.mean(
            all_baseline <= self.capacity + 1e-6
        )
        after_compliance = 100.0 * np.mean(
            all_after <= self.capacity + 1e-6
        )

        def reduction_percent(before_values, after_values):
            before_mean = float(np.mean(before_values))
            after_mean = float(np.mean(after_values))
            return 100.0 * (before_mean - after_mean) / max(before_mean, EPS)

        table_1 = pd.DataFrame([
            {
                "Metric": "Daily peak load (kW)",
                "No DR": self._mean_pm_sd(daily["peak_baseline"]),
                "CCRL-DR": self._mean_pm_sd(daily["peak_after"]),
                "Improvement": f"{reduction_percent(daily['peak_baseline'], daily['peak_after']):.2f}%",
            },
            {
                "Metric": "Mean load (kW)",
                "No DR": self._mean_pm_sd(daily["mean_baseline"]),
                "CCRL-DR": self._mean_pm_sd(daily["mean_after"]),
                "Improvement": f"{reduction_percent(daily['mean_baseline'], daily['mean_after']):.2f}%",
            },
            {
                "Metric": "Daily PAR",
                "No DR": self._mean_pm_sd(daily["PAR_baseline"]),
                "CCRL-DR": self._mean_pm_sd(daily["PAR_after"]),
                "Improvement": f"{reduction_percent(daily['PAR_baseline'], daily['PAR_after']):.2f}%",
            },
            {
                "Metric": "Capacity compliance rate (%)",
                "No DR": f"{baseline_compliance:.2f}",
                "CCRL-DR": f"{after_compliance:.2f}",
                "Improvement": f"+{after_compliance - baseline_compliance:.2f} percentage points",
            },
        ])

        device = appliance.copy()
        device["Appliance"] = device["appliance"].map(self._device_display_name)
        device_order = ["AC", "EV", "WM", "DW", "DR"]
        type_map = {
            "AC": "PC",
            "EV": "TS-I",
            "WM": "TS-NI",
            "DW": "TS-NI",
            "DR": "TS-NI",
        }
        grouped = device.groupby("Appliance", as_index=True).agg(
            baseline_energy=("baseline_energy", "sum"),
            after_energy=("after_energy", "sum"),
            true_curtailment=("true_curtailment", "sum"),
            shifted_out=("shifted_out_energy", "sum"),
            shifted_in=("shifted_in_energy", "sum"),
        )
        table_2_rows = []
        for appliance_name in device_order:
            if appliance_name not in grouped.index:
                continue
            row = grouped.loc[appliance_name]
            table_2_rows.append({
                "Appliance": appliance_name,
                "Type": type_map[appliance_name],
                "Controllable-device baseline energy (kWh)": float(row["baseline_energy"]),
                "After-DR energy (kWh)": float(row["after_energy"]),
                "True curtailment (kWh)": float(row["true_curtailment"]),
                "Shifted-out energy (kWh)": float(row["shifted_out"]),
                "Shifted-in energy (kWh)": float(row["shifted_in"]),
            })
        table_2 = pd.DataFrame(table_2_rows)
        total_row = {
            "Appliance": "Total",
            "Type": "—",
            "Controllable-device baseline energy (kWh)": float(table_2["Controllable-device baseline energy (kWh)"].sum()),
            "After-DR energy (kWh)": float(table_2["After-DR energy (kWh)"].sum()),
            "True curtailment (kWh)": float(table_2["True curtailment (kWh)"].sum()),
            "Shifted-out energy (kWh)": float(table_2["Shifted-out energy (kWh)"].sum()),
            "Shifted-in energy (kWh)": float(table_2["Shifted-in energy (kWh)"].sum()),
        }
        table_2 = pd.concat([table_2, pd.DataFrame([total_row])], ignore_index=True)

        total_shifted_in = float(hourly["shifted_in_this_hour"].sum())
        safe_shifted_in = float(
            hourly.loc[
                hourly["shifted_in_capacity_safe"].astype(bool),
                "shifted_in_this_hour",
            ].sum()
        )
        safe_shifted_in_ratio = (
            100.0 * safe_shifted_in / total_shifted_in
            if total_shifted_in > EPS
            else np.nan
        )
        rebound_hours = int(hourly["rebound_induced_violation"].astype(bool).sum())
        rebound_days = int(
            hourly.groupby("day")["rebound_induced_violation"]
            .apply(lambda values: bool(np.any(np.asarray(values, dtype=bool))))
            .sum()
        )
        shifted_in_hour_mask = hourly["shifted_in_present"].astype(bool)
        maximum_secondary_peak = (
            float(hourly.loc[shifted_in_hour_mask, "after_total"].max())
            if bool(shifted_in_hour_mask.any())
            else np.nan
        )
        table_2_rebound = pd.DataFrame([
            {
                "Rebound / secondary-peak indicator": "Capacity-safe shifted-in energy (%)",
                "Result": safe_shifted_in_ratio,
            },
            {
                "Rebound / secondary-peak indicator": "Rebound-induced violation hours",
                "Result": f"{rebound_hours}/{len(hourly)}",
            },
            {
                "Rebound / secondary-peak indicator": "Days with rebound-induced violations",
                "Result": f"{rebound_days}/{n_days}",
            },
            {
                "Rebound / secondary-peak indicator": "Maximum post-DR load during shifted-in hours (kW)",
                "Result": maximum_secondary_peak,
            },
        ])

        eu_rows = []
        fairness_rows = []
        for eu_index, house_id in enumerate(self.house_ids, start=1):
            hdf = household[household["house_id"] == house_id]
            settlement_total_house = float(hdf["settlement_DR_energy"].sum())
            weighted_benefit_total = float(
                hdf["rho_weighted_incentive_benefit"].sum()
            )
            weighted_discomfort_total = float(
                hdf["one_minus_rho_weighted_discomfort"].sum()
            )
            weighted_utility_total = float(
                hdf["weighted_customer_utility"].sum()
            )
            eu_rows.append({
                "End user": f"EU{eu_index} ({house_id})",
                "Verified settlement DR (kWh/day)": settlement_total_house / n_days,
                f"rho-weighted incentive benefit (cent/day), rho={self.rho:g}": weighted_benefit_total / n_days,
                "(1-rho)-weighted dissatisfaction (cent/day)": weighted_discomfort_total / n_days,
                "Weighted EU utility (cent/day)": weighted_utility_total / n_days,
            })
            fairness_rows.append({
                "End user": f"EU{eu_index} ({house_id})",
                "Weighted utility over test range, U_n (cent)": weighted_utility_total,
                "Verified settlement DR over test range, D_n (kWh)": (
                    settlement_total_house
                ),
                "x_n = U_n / D_n (cent/kWh)": (
                    weighted_utility_total / settlement_total_house
                    if settlement_total_house > EPS
                    else np.nan
                ),
            })

        panel_a = pd.DataFrame(eu_rows)
        panel_a_total = {column: None for column in panel_a.columns}
        panel_a_total["End user"] = "Total"
        for column in panel_a.columns[1:]:
            panel_a_total[column] = float(panel_a[column].sum())
        panel_a = pd.concat([panel_a, pd.DataFrame([panel_a_total])], ignore_index=True)

        panel_b = pd.DataFrame([
            {
                "Service-provider metric": "Wholesale value of verified settlement DR",
                "Average daily value (cent/day)": float(
                    daily["SP_settlement_wholesale_value"].mean()
                ),
            },
            {
                "Service-provider metric": "Incentive cost for verified settlement DR",
                "Average daily value (cent/day)": float(
                    daily["SP_settlement_incentive_cost"].mean()
                ),
            },
            {
                "Service-provider metric": "Capacity-relief settlement value",
                "Average daily value (cent/day)": float(
                    daily["SP_settlement_value"].mean()
                ),
            },
            {
                "Service-provider metric": "Net wholesale procurement impact",
                "Average daily value (cent/day)": float(
                    daily["net_wholesale_procurement_impact"].mean()
                ),
            },
        ])

        fairness_calculation = pd.DataFrame(fairness_rows)
        x_values = fairness_calculation["x_n = U_n / D_n (cent/kWh)"].to_numpy(dtype=float)
        finite_x = x_values[np.isfinite(x_values)]
        if len(finite_x) != len(self.house_ids) or np.any(finite_x < -EPS):
            jain_index = np.nan
            fairness_note = (
                "Jain's index was not computed because all participation-normalised "
                "utilities must be finite and non-negative."
            )
        else:
            sum_x = float(finite_x.sum())
            sum_x_sq = float(np.square(finite_x).sum())
            jain_index = (
                sum_x ** 2 / (len(finite_x) * sum_x_sq)
                if sum_x_sq > EPS
                else np.nan
            )
            fairness_note = (
                "Ex-post Jain index based on weighted EU utility per unit of "
                "verified settlement DR energy."
            )
        panel_c = pd.DataFrame([
            {
                "Fairness metric": "Participation-normalised Jain fairness index",
                "Value": jain_index,
            }
        ])
        sum_x = float(np.nansum(x_values))
        sum_x_sq = float(np.nansum(np.square(x_values)))
        fairness_summary = pd.DataFrame([
            {"Metric": "Sum of x_n", "Calculation": "x_1 + x_2 + x_3", "Value": sum_x},
            {"Metric": "Sum of x_n squared", "Calculation": "x_1^2 + x_2^2 + x_3^2", "Value": sum_x_sq},
            {"Metric": "Number of end users, N", "Calculation": "N", "Value": len(self.house_ids)},
            {
                "Metric": "Jain fairness index",
                "Calculation": "(sum x_n)^2 / [N * sum(x_n^2)]",
                "Value": jain_index,
            },
        ])

        reward_target_summary = pd.DataFrame([
            {
                "Reward / target metric": "Unnecessary incentive hours",
                "Result": int(hourly["unnecessary_incentive"].astype(bool).sum()),
            },
            {
                "Reward / target metric": "Positive-incentive/no-response hours",
                "Result": int(
                    hourly["positive_incentive_no_response"].astype(bool).sum()
                ),
            },
            {
                "Reward / target metric": "Maximum-action hours",
                "Result": int(hourly["maximum_action"].astype(bool).sum()),
            },
            {
                "Reward / target metric": "No-need gate hours",
                "Result": int(hourly["no_need_gate"].astype(bool).sum()),
            },
            {
                "Reward / target metric": "Cumulative no-need offer penalty",
                "Result": float(hourly["no_need_offer_penalty"].sum()),
            },
            {
                "Reward / target metric": "Useful settlement DR (kWh)",
                "Result": float(hourly["useful_settlement_DR_energy"].sum()),
            },
            {
                "Reward / target metric": "Excess settlement DR (kWh)",
                "Result": float(hourly["excess_settlement_DR_energy"].sum()),
            },
            {
                "Reward / target metric": "Target-tracking MAE (kWh)",
                "Result": float(
                    np.mean(
                        hourly["control_over_error"]
                        + hourly["control_under_error"]
                    )
                ),
            },
            {
                "Reward / target metric": "Average offered incentive rate (cent/kWh)",
                "Result": float(
                    hourly[[
                        column for column in hourly.columns
                        if column.startswith("incentive_house_")
                    ]].to_numpy(dtype=float).mean()
                ),
            },
        ])

        notes = {
            "table_1": (
                f"Peak load, mean load, and PAR are mean ± population standard "
                f"deviation across {n_days} test days. Capacity compliance is "
                "calculated over all hourly observations."
            ),
            "table_2": (
                f"Panel A values are totals across all households and {n_days} "
                "test days. Panel B uses the strict rebound-induced violation "
                f"definition. The aggregate capacity limit was {self.capacity:g} kW."
            ),
            "table_3": (
                "Official incentive settlement uses non-negative hourly baseline-verified "
                "DR energy. Multi-hour shifted jobs retain their decision-hour incentive "
                "rate, while each verified origin-hour slice uses that hour's wholesale "
                "price. The separate net wholesale procurement impact includes signed "
                "shifted-in destination effects and is diagnostic only. Weighted EU utility "
                "equals rho*incentive payment - (1-rho)*dissatisfaction. " + fairness_note
            ),
        }
        return {
            "table_1": table_1,
            "table_2": table_2,
            "table_2_rebound": table_2_rebound,
            "panel_a": panel_a,
            "panel_b": panel_b,
            "panel_c": panel_c,
            "fairness_calculation": fairness_calculation,
            "fairness_summary": fairness_summary,
            "reward_target_summary": reward_target_summary,
            "notes": notes,
        }

    def _cleanup_obsolete_excel_outputs(self):
        """Remove only obsolete Excel outputs created by earlier test versions."""
        overall_dir = os.path.join(self.results_dir, "overall_summary")
        if os.path.isdir(overall_dir):
            shutil.rmtree(overall_dir)

        old_audit = os.path.join(
            self.results_dir,
            f"test_range_{self.test_days[0]}_{self.test_days[-1]}_corrected_environment.xlsx",
        )
        if os.path.exists(old_audit):
            os.remove(old_audit)

        for plot_day in self.target_plot_days:
            day_dir = os.path.join(self.results_dir, f"day_{plot_day}")
            if os.path.isdir(day_dir):
                for filename in os.listdir(day_dir):
                    if filename.lower().endswith(".xlsx"):
                        path = os.path.join(day_dir, filename)
                        os.remove(path)

    def save_paper_ready_results(self, tables):
        """Save only the three agreed manuscript tables plus fairness support."""
        outputs = self._build_paper_ready_tables(tables)
        excel_path = os.path.join(
            self.results_dir,
            "results.xlsx",
        )

        try:
            from openpyxl import load_workbook
            from openpyxl.styles import Alignment, Border, Font, PatternFill, Side

            with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
                outputs["table_1"].to_excel(
                    writer, sheet_name="Table_1_Multiday", index=False
                )
                table_2_sheet = "Table_2_Appliances"
                outputs["table_2"].to_excel(
                    writer, sheet_name=table_2_sheet, index=False, startrow=2
                )
                rebound_title_row = len(outputs["table_2"]) + 6
                outputs["table_2_rebound"].to_excel(
                    writer, sheet_name=table_2_sheet, index=False,
                    startrow=rebound_title_row
                )


                sheet_name = "Table_3_Economic_Fairness"
                outputs["panel_a"].to_excel(
                    writer, sheet_name=sheet_name, index=False, startrow=3
                )
                outputs["panel_b"].to_excel(
                    writer, sheet_name=sheet_name, index=False, startrow=11
                )
                outputs["panel_c"].to_excel(
                    writer, sheet_name=sheet_name, index=False, startrow=18
                )

                outputs["fairness_calculation"].to_excel(
                    writer, sheet_name="Fairness_Calculation", index=False, startrow=2
                )
                outputs["fairness_summary"].to_excel(
                    writer, sheet_name="Fairness_Calculation", index=False, startrow=8
                )

                outputs["reward_target_summary"].to_excel(
                    writer, sheet_name="Reward_Target_Metrics", index=False
                )

                wb = writer.book
                dark_fill = PatternFill("solid", fgColor="1F4E78")
                panel_fill = PatternFill("solid", fgColor="D9EAF7")
                white_bold = Font(color="FFFFFF", bold=True)
                bold_font = Font(bold=True)
                thin = Side(style="thin", color="B4C6E7")
                border = Border(bottom=thin)


                for ws_name in ["Table_1_Multiday"]:
                    ws = wb[ws_name]
                    ws.freeze_panes = "A2"
                    ws.sheet_view.showGridLines = False
                    ws.auto_filter.ref = ws.dimensions
                    for cell in ws[1]:
                        cell.fill = dark_fill
                        cell.font = white_bold
                        cell.alignment = Alignment(
                            horizontal="center", vertical="center", wrap_text=True
                        )
                        cell.border = border
                    ws.row_dimensions[1].height = 42
                    for row in ws.iter_rows(min_row=2):
                        for cell in row:
                            cell.alignment = Alignment(vertical="center", wrap_text=True)
                            if isinstance(cell.value, float):
                                cell.number_format = "0.00"
                    for column_cells in ws.columns:
                        letter = column_cells[0].column_letter
                        max_len = max(
                            len(str(c.value)) if c.value is not None else 0
                            for c in column_cells
                        )
                        ws.column_dimensions[letter].width = min(max(max_len + 2, 12), 36)

                ws1 = wb["Table_1_Multiday"]
                foot_row_1 = ws1.max_row + 2
                ws1.merge_cells(start_row=foot_row_1, start_column=1, end_row=foot_row_1, end_column=4)
                ws1.cell(foot_row_1, 1, outputs["notes"]["table_1"])
                ws1.cell(foot_row_1, 1).alignment = Alignment(wrap_text=True)
                ws1.cell(foot_row_1, 1).font = Font(italic=True, color="555555")
                ws1.row_dimensions[foot_row_1].height = 45


                ws2 = wb["Table_2_Appliances"]
                ws2.sheet_view.showGridLines = False
                ws2.merge_cells("A1:G1")
                ws2["A1"] = (
                    f"Appliance-level demand-response decomposition and rebound "
                    f"assessment over {len(self.test_days)} test days"
                )
                ws2["A1"].fill = dark_fill
                ws2["A1"].font = white_bold
                ws2["A1"].alignment = Alignment(
                    horizontal="center", vertical="center"
                )
                ws2.row_dimensions[1].height = 30

                ws2.merge_cells("A2:G2")
                ws2["A2"] = "Panel A. Appliance-level action decomposition"
                ws2["A2"].fill = panel_fill
                ws2["A2"].font = bold_font
                ws2["A2"].alignment = Alignment(horizontal="left")

                appliance_header_row = 3
                rebound_title_excel_row = len(outputs["table_2"]) + 6
                rebound_header_row = rebound_title_excel_row + 1
                ws2.merge_cells(
                    start_row=rebound_title_excel_row, start_column=1,
                    end_row=rebound_title_excel_row, end_column=7
                )
                ws2.cell(
                    rebound_title_excel_row, 1,
                    "Panel B. Rebound and secondary-peak assessment"
                )
                ws2.cell(rebound_title_excel_row, 1).fill = panel_fill
                ws2.cell(rebound_title_excel_row, 1).font = bold_font
                ws2.cell(rebound_title_excel_row, 1).alignment = Alignment(
                    horizontal="left"
                )

                for header_row in [appliance_header_row, rebound_header_row]:
                    for cell in ws2[header_row]:
                        if cell.value is not None:
                            cell.fill = dark_fill
                            cell.font = white_bold
                            cell.alignment = Alignment(
                                horizontal="center", vertical="center",
                                wrap_text=True
                            )
                            cell.border = border
                    ws2.row_dimensions[header_row].height = 46

                for row in ws2.iter_rows(min_row=4, max_row=ws2.max_row):
                    for cell in row:
                        if cell.value is not None:
                            cell.alignment = Alignment(
                                vertical="center", wrap_text=True
                            )
                            if isinstance(cell.value, float):
                                cell.number_format = "0.00"


                for row in range(rebound_header_row + 1, ws2.max_row + 1):
                    result_cell = ws2.cell(row, 2)
                    if isinstance(result_cell.value, float):
                        result_cell.number_format = "0.00"

                for letter, width in {
                    "A": 36, "B": 17, "C": 24, "D": 21,
                    "E": 21, "F": 21, "G": 21
                }.items():
                    ws2.column_dimensions[letter].width = width

                foot_row_2 = ws2.max_row + 2
                ws2.merge_cells(
                    start_row=foot_row_2, start_column=1,
                    end_row=foot_row_2, end_column=7
                )
                ws2.cell(foot_row_2, 1, outputs["notes"]["table_2"])
                ws2.cell(foot_row_2, 1).alignment = Alignment(wrap_text=True)
                ws2.cell(foot_row_2, 1).font = Font(italic=True, color="555555")
                ws2.row_dimensions[foot_row_2].height = 78
                ws2.freeze_panes = "A3"


                ws3 = wb[sheet_name]
                ws3.sheet_view.showGridLines = False
                ws3.merge_cells("A1:E1")
                ws3["A1"] = (
                    f"Average daily economic performance and ex-post fairness "
                    f"over {len(self.test_days)} test days"
                )
                ws3["A1"].fill = dark_fill
                ws3["A1"].font = white_bold
                ws3["A1"].alignment = Alignment(horizontal="center", vertical="center")
                ws3.row_dimensions[1].height = 28

                panel_titles = {
                    3: "Panel A. End-user economic performance",
                    11: "Panel B. Service-provider economic performance",
                    18: "Panel C. Ex-post participation-normalised fairness",
                }
                for row_number, title in panel_titles.items():
                    ws3.merge_cells(
                        start_row=row_number, start_column=1,
                        end_row=row_number, end_column=5
                    )
                    cell = ws3.cell(row_number, 1, title)
                    cell.fill = panel_fill
                    cell.font = bold_font
                    cell.alignment = Alignment(horizontal="left", vertical="center")

                for header_row in [4, 12, 19]:
                    for cell in ws3[header_row]:
                        if cell.value is not None:
                            cell.fill = dark_fill
                            cell.font = white_bold
                            cell.alignment = Alignment(
                                horizontal="center", vertical="center", wrap_text=True
                            )
                            cell.border = border
                    ws3.row_dimensions[header_row].height = 48

                for row in ws3.iter_rows(min_row=5, max_row=20):
                    for cell in row:
                        if cell.value is not None:
                            cell.alignment = Alignment(vertical="center", wrap_text=True)
                            if isinstance(cell.value, float):
                                cell.number_format = "0.000"

                note_row = 23
                ws3.merge_cells(start_row=note_row, start_column=1, end_row=note_row, end_column=5)
                ws3.cell(note_row, 1, outputs["notes"]["table_3"])
                ws3.cell(note_row, 1).alignment = Alignment(wrap_text=True)
                ws3.cell(note_row, 1).font = Font(italic=True, color="555555")
                ws3.row_dimensions[note_row].height = 65
                widths = {"A": 38, "B": 22, "C": 29, "D": 31, "E": 24}
                for letter, width in widths.items():
                    ws3.column_dimensions[letter].width = width


                wsf = wb["Fairness_Calculation"]
                wsf.sheet_view.showGridLines = False
                wsf.merge_cells("A1:D1")
                wsf["A1"] = "Participation-normalised Jain fairness calculation"
                wsf["A1"].fill = dark_fill
                wsf["A1"].font = white_bold
                wsf["A1"].alignment = Alignment(horizontal="center")
                for header_row in [3, 9]:
                    for cell in wsf[header_row]:
                        if cell.value is not None:
                            cell.fill = dark_fill
                            cell.font = white_bold
                            cell.alignment = Alignment(
                                horizontal="center", vertical="center", wrap_text=True
                            )
                for row in wsf.iter_rows(min_row=4):
                    for cell in row:
                        if isinstance(cell.value, float):
                            cell.number_format = "0.000000"
                        cell.alignment = Alignment(vertical="center", wrap_text=True)
                for letter, width in {"A": 29, "B": 38, "C": 38, "D": 34}.items():
                    wsf.column_dimensions[letter].width = width

                wsr = wb["Reward_Target_Metrics"]
                wsr.sheet_view.showGridLines = False
                wsr.freeze_panes = "A2"
                wsr.auto_filter.ref = wsr.dimensions
                for cell in wsr[1]:
                    cell.fill = dark_fill
                    cell.font = white_bold
                    cell.alignment = Alignment(
                        horizontal="center", vertical="center", wrap_text=True
                    )
                for row in wsr.iter_rows(min_row=2):
                    for cell in row:
                        cell.alignment = Alignment(vertical="center", wrap_text=True)
                        if isinstance(cell.value, float):
                            cell.number_format = "0.0000"
                wsr.column_dimensions["A"].width = 48
                wsr.column_dimensions["B"].width = 22

                wb.active = wb.sheetnames.index("Table_1_Multiday")

            check_wb = load_workbook(excel_path, read_only=True, data_only=True)
            expected_sheets = [
                "Table_1_Multiday",
                "Table_2_Appliances",
                "Table_3_Economic_Fairness",
                "Fairness_Calculation",
                "Reward_Target_Metrics",
            ]
            missing = [name for name in expected_sheets if name not in check_wb.sheetnames]
            check_wb.close()
            if missing:
                raise RuntimeError(f"Paper workbook is missing sheets: {missing}")
        except ImportError as exc:
            raise ImportError(
                "openpyxl is required to create the Excel reports. "
                "Install it with: pip install openpyxl"
            ) from exc

        return excel_path

    def generate_detailed_plots(self):
        """Generate all retained article figures for each requested plot day."""
        original_target_day = self.target_plot_day
        try:
            for plot_day in self.target_plot_days:
                if plot_day not in self.day_results:
                    print(
                        f"[WARN] No figures generated because day {plot_day} "
                        "was not simulated."
                    )
                    continue

                self.target_plot_day = int(plot_day)
                fig_dir = os.path.join(
                    self.results_dir,
                    f"day_{plot_day}",
                    "figures",
                )
                os.makedirs(fig_dir, exist_ok=True)

                obsolete_payout_plot = os.path.join(
                    fig_dir,
                    f"total_payout_and_capacity_relief_per_house_day_{plot_day}.png",
                )
                if os.path.exists(obsolete_payout_plot):
                    os.remove(obsolete_payout_plot)

                original_results_dir = self.results_dir
                try:
                    self.results_dir = fig_dir
                    result = self.day_results[plot_day]
                    self.plot_day_profiles(result)
                    self.plot_aggregated_load(result)
                    self.plot_total_incentive_rate(result)
                    self.plot_device_schedule_by_house(result)
                finally:
                    self.results_dir = original_results_dir
        finally:
            self.target_plot_day = original_target_day

    def run_complete_analysis(self):
        self.simulate_all_test_days()
        tables = self.build_report_tables()


        self._cleanup_obsolete_excel_outputs()
        audit_path = self.save_excel_report(tables)
        paper_path = self.save_paper_ready_results(tables)
        self.generate_detailed_plots()

        print(
            f"Test complete | audit={audit_path} | results={paper_path}"
        )


def main():
    parser = argparse.ArgumentParser(
        description="Test a trained DDQN model over the complete configured test range."
    )
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to the trained model checkpoint (.pth)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Optional output directory. Default: a subfolder beside the model checkpoint.",
    )
    parser.add_argument(
        "--config_path",
        type=str,
        default=None,
        help=(
            "Optional YAML config used for this checkpoint. When omitted, "
            "the project's default config loader is used."
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace a non-empty output directory intentionally.",
    )
    parser.add_argument(
        "--allow_legacy_checkpoint",
        action="store_true",
        help=(
            "Allow diagnostic loading of a checkpoint without the new metadata. "
            "Do not use this option for final paper results."
        ),
    )
    parser.add_argument(
        "--target_plot_day",
        type=int,
        default=ARTICLE_PLOT_DAY,
        help=(
            "Illustrative day for detailed figures. Default: 208."
        ),
    )
    parser.add_argument(
        "--target_plot_days",
        type=int,
        nargs="+",
        default=None,
        help=(
            "One or more illustrative days for detailed figures, e.g. "
            "--target_plot_days 311 312. This overrides --target_plot_day."
        ),
    )
    parser.add_argument(
        "--test_start",
        type=int,
        default=None,
        help=(
            "Optional override for the first simulated test day. "
            "Use together with --test_end."
        ),
    )
    parser.add_argument(
        "--test_end",
        type=int,
        default=None,
        help=(
            "Optional override for the last simulated test day. "
            "Use together with --test_start."
        ),
    )
    args = parser.parse_args()

    if (args.test_start is None) ^ (args.test_end is None):
        parser.error("--test_start and --test_end must be provided together.")

    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)

    if args.config_path:
        with open(args.config_path, "r", encoding="utf-8") as handle:
            cfg = yaml.safe_load(handle)
        if not isinstance(cfg, dict):
            raise TypeError(f"Configuration is not a dictionary: {args.config_path}")
    else:
        cfg = load_config()

    test_range_override = None
    if args.test_start is not None and args.test_end is not None:
        test_range_override = (int(args.test_start), int(args.test_end))

    tester = ModelTester(
        model_path=args.model_path,
        cfg_override=cfg,
        target_plot_day=args.target_plot_day,
        target_plot_days=args.target_plot_days,
        output_dir=args.output_dir,
        allow_legacy_checkpoint=args.allow_legacy_checkpoint,
        test_range_override=test_range_override,
        overwrite=args.overwrite,
    )
    tester.run_complete_analysis()


if __name__ == "__main__":
    main()