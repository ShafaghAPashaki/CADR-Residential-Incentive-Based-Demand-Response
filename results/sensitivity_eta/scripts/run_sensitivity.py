import argparse
import csv
import json
import os
import random
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml


# Allow this script to be launched from any working directory.
PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

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

EPS = 1e-9
SCRIPT_VERSION = "ddqn_eta_sensitivity_v2"


class DDQNTrainer:
    """Train DDQN and select the reporting checkpoint using validation only.

    The configured train/validation/test date ranges are not changed here.
    Validation is deterministic:
      - a separate Environment is used;
      - every day in val_range is evaluated exactly once;
      - the policy is fully greedy (epsilon = 0);
      - the test range is never accessed.

    Best-checkpoint rule:
      1) highest deterministic mean validation reward for the active arm;
      2) lowest validation reward standard deviation only as a tie-breaker;
      3) earliest episode in the final tie.
    """

    def __init__(self, cfg_override=None, output_dir=None):
        self.cfg = cfg_override or load_config()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        house_ids = list(self.cfg["environment"]["house_ids"])


        self.train_env = Environment(data_ids=house_ids, cfg_override=self.cfg)
        self.val_env = Environment(data_ids=house_ids, cfg_override=self.cfg)


        if hasattr(self.train_env, "household_coefficients"):
            fixed_coefficients = self.train_env.household_coefficients.copy()
            self.val_env.set_household_coefficients(fixed_coefficients)

        initial_state = self.train_env.reset(mode="train")
        self.state_dim = len(initial_state)
        self.action_dim = self.train_env.num_actions

        self.agent = DDQNAgent(
            self.state_dim,
            self.action_dim,
            cfg_override=self.cfg,
            action_costs=np.mean(self.train_env.all_actions, axis=1),
        )


        self.episodes = int(self.cfg["DDQN"]["episodes"])

        configured_max_steps = int(self.cfg["DDQN"]["max_steps"])
        self.max_steps = int(self.train_env.max_steps)
        if configured_max_steps != self.max_steps:
            raise ValueError(
                "DDQN.max_steps must match the Environment episode length: "
                f"{configured_max_steps} != {self.max_steps}."
            )
        if int(self.val_env.max_steps) != self.max_steps:
            raise ValueError(
                "Training and validation environments must have the same max_steps."
            )

        self.eval_interval = int(self.cfg["DDQN"]["eval_interval"])
        if self.eval_interval <= 0:
            raise ValueError("DDQN.eval_interval must be a positive integer.")

        val_start, val_end = self.cfg["training"]["val_range"]
        self.val_days = list(range(int(val_start), int(val_end) + 1))
        if not self.val_days:
            raise ValueError("The configured validation range is empty.")

        self.capacity = float(self.cfg["environment"]["capacity_threshold"])


        self.episode_rewards = []
        self.episode_losses = []
        self.episode_lengths = []
        self.epsilon_history = []
        self.episode_q_values = []
        self.current_episode_q_values = []


        self.validation_history = []
        self.eval_rewards = []
        self.eval_timesteps = []
        self.eval_violation_steps = []
        self.eval_cumulative_excess = []
        self.eval_maximum_overrun = []
        self.eval_energy_balance_error = []


        self.best_selection_key = None
        self.best_validation_episode = None
        self.best_validation_metrics = None
        self.last_validation_episode = None

        self.run_seed = int(self.cfg.get("general", {}).get("seed", 0))
        if output_dir is None:
            raise ValueError(
                "Sensitivity training requires an explicit isolated output_dir. "
                "The primary training directory must never be used."
            )
        self.results_dir = os.path.abspath(output_dir)
        primary_training_dir = os.path.abspath(
            os.path.join("results", f"seed_{self.run_seed}", "training")
        )
        if os.path.normcase(self.results_dir) == os.path.normcase(primary_training_dir):
            raise ValueError(
                "Refusing to write the sensitivity run into the primary training directory: "
                f"{primary_training_dir}"
            )
        if os.path.isdir(self.results_dir) and os.listdir(self.results_dir):
            raise FileExistsError(
                f"Training output already exists: {self.results_dir}. "
                "Move or remove it before starting this seed again."
            )
        os.makedirs(self.results_dir, exist_ok=True)
        with open(os.path.join(self.results_dir, "config.yaml"), "w", encoding="utf-8") as handle:
            yaml.safe_dump(self.cfg, handle, sort_keys=False)
        print(
            f"Seed {self.run_seed} | device={self.device} | "
            f"episodes={self.episodes} | output={self.results_dir}"
        )
        reward_meta = self.train_env.get_reward_metadata()
        if (
            self.train_env.reward_mode.startswith("weighted_stakeholder")
            and reward_meta["symmetric_tracking_lambda_source"].startswith(
                "pre_audit"
            )
        ):
            raise RuntimeError(
                "Training is blocked because symmetric_tracking.lambda_c "
                "is still the pre-audit placeholder. Run the counterfactual "
                "calibration audit, write its recommended lambda_C into the "
                "training config, and update lambda_source before training."
            )

    def _greedy_action(self, state):
        return self.agent.greedy_action(state)

    def evaluate_policy(self, completed_episode):
        """Evaluate every configured validation day without changing training RNG."""
        was_training = self.agent.policy_net.training
        python_rng_state = random.getstate()
        numpy_rng_state = np.random.get_state()
        torch_rng_state = torch.get_rng_state()
        cuda_rng_states = (
            torch.cuda.get_rng_state_all()
            if torch.cuda.is_available()
            else None
        )

        self.agent.policy_net.eval()

        day_rewards = []
        total_violation_steps = 0
        total_evaluated_steps = 0
        total_cumulative_excess = 0.0
        maximum_overrun = 0.0
        total_gross_relief = 0.0
        total_true_curtailment = 0.0
        max_abs_energy_balance_error = 0.0
        total_useful_settlement_dr = 0.0
        total_excess_settlement_dr = 0.0
        total_tracking_abs_error = 0.0
        total_tracking_penalty = 0.0
        total_incentive_payment = 0.0
        total_discomfort = 0.0
        total_offered_incentive = 0.0
        total_incentive_steps = 0
        total_unnecessary_incentive_steps = 0
        total_positive_incentive_no_response_steps = 0
        total_maximum_action_steps = 0
        total_no_need_gate_steps = 0
        total_no_need_offer_penalty = 0.0
        total_combined_offer_penalty = 0.0
        total_stakeholder_sp_component = 0.0
        total_stakeholder_eu_component = 0.0
        total_stakeholder_reward_component = 0.0
        total_symmetric_tracking_loss = 0.0
        total_symmetric_tracking_penalty = 0.0

        try:
            with torch.no_grad():
                for day in self.val_days:
                    state = self.val_env.reset(day=day, mode="val")
                    done = False
                    steps = 0
                    day_reward = 0.0

                    while not done and steps < self.max_steps:
                        action = self._greedy_action(state)
                        state, reward, done, _ = self.val_env.step(action)
                        day_reward += float(reward)
                        steps += 1

                    if not done:
                        raise RuntimeError(
                            f"Validation day {day} did not terminate within "
                            f"max_steps={self.max_steps}."
                        )

                    day_rewards.append(day_reward)
                    total_evaluated_steps += int(steps)

                    final_load = (
                        self.val_env.after_total_per_house[:steps].sum(axis=1)
                    )
                    overrun = np.maximum(final_load - self.capacity, 0.0)

                    total_violation_steps += int(np.sum(overrun > 1e-6))
                    total_cumulative_excess += float(np.sum(overrun))
                    maximum_overrun = max(
                        maximum_overrun,
                        float(np.max(overrun)) if len(overrun) else 0.0,
                    )

                    if hasattr(self.val_env, "hourly_capacity_relief"):
                        total_gross_relief += float(
                            self.val_env.hourly_capacity_relief[:steps].sum()
                        )

                    if hasattr(
                        self.val_env,
                        "net_curtailment_per_house",
                    ):
                        true_curtailment = float(
                            self.val_env
                            .net_curtailment_per_house[:steps]
                            .sum()
                        )
                        total_true_curtailment += true_curtailment

                        baseline_energy = float(
                            self.val_env.baseline_per_house[:steps].sum()
                        )
                        final_energy = float(
                            self.val_env.after_total_per_house[:steps].sum()
                        )
                        balance_error = (
                            baseline_energy
                            - final_energy
                            - true_curtailment
                        )
                        max_abs_energy_balance_error = max(
                            max_abs_energy_balance_error,
                            abs(balance_error),
                        )

                    sl = slice(0, steps)
                    total_useful_settlement_dr += float(
                        self.val_env.useful_settlement_dr_energy[sl].sum()
                    )
                    total_excess_settlement_dr += float(
                        self.val_env.excess_settlement_dr_energy[sl].sum()
                    )
                    total_tracking_abs_error += float(
                        (
                            self.val_env.control_over_error[sl]
                            + self.val_env.control_under_error[sl]
                        ).sum()
                    )
                    total_tracking_penalty += float(
                        self.val_env.target_tracking_penalty[sl].sum()
                    )
                    total_incentive_payment += float(
                        self.val_env.incentive_payment[sl].sum()
                    )
                    total_discomfort += float(self.val_env.discomforts[sl].sum())
                    total_offered_incentive += float(
                        self.val_env.incentives[sl].sum()
                    )
                    incentive_mask = np.any(
                        self.val_env.incentives[sl] > EPS, axis=1
                    )
                    total_incentive_steps += int(incentive_mask.sum())
                    total_unnecessary_incentive_steps += int(
                        self.val_env.unnecessary_incentive_flag[sl].sum()
                    )
                    total_positive_incentive_no_response_steps += int(
                        self.val_env.positive_incentive_no_response_flag[sl].sum()
                    )
                    total_maximum_action_steps += int(
                        self.val_env.maximum_action_flag[sl].sum()
                    )
                    total_no_need_gate_steps += int(
                        self.val_env.no_need_gate_flag[sl].sum()
                    )
                    total_no_need_offer_penalty += float(
                        self.val_env.no_need_offer_penalty[sl].sum()
                    )
                    total_combined_offer_penalty += float(
                        self.val_env.combined_offer_penalty[sl].sum()
                    )
                    total_stakeholder_sp_component += float(
                        self.val_env.stakeholder_sp_normalized_component[sl].sum()
                    )
                    total_stakeholder_eu_component += float(
                        self.val_env.stakeholder_eu_normalized_component[sl].sum()
                    )
                    total_stakeholder_reward_component += float(
                        self.val_env.stakeholder_reward_component[sl].sum()
                    )
                    total_symmetric_tracking_loss += float(
                        self.val_env.symmetric_tracking_loss[sl].sum()
                    )
                    total_symmetric_tracking_penalty += float(
                        self.val_env.symmetric_tracking_penalty[sl].sum()
                    )
        finally:
            random.setstate(python_rng_state)
            np.random.set_state(numpy_rng_state)
            torch.set_rng_state(torch_rng_state)
            if cuda_rng_states is not None:
                torch.cuda.set_rng_state_all(cuda_rng_states)

            if was_training:
                self.agent.policy_net.train()
            else:
                self.agent.policy_net.eval()

        metrics = {
            "episode": int(completed_episode),
            "validation_days": len(self.val_days),
            "mean_reward": float(np.mean(day_rewards)),
            "std_reward": float(np.std(day_rewards)),
            "minimum_reward": float(np.min(day_rewards)),
            "maximum_reward": float(np.max(day_rewards)),
            "total_reward": float(np.sum(day_rewards)),
            "violation_steps": int(total_violation_steps),
            "evaluated_steps": int(total_evaluated_steps),
            "violation_percentage": float(
                100.0
                * total_violation_steps
                / max(total_evaluated_steps, 1)
            ),
            "cumulative_excess": float(total_cumulative_excess),
            "maximum_overrun": float(maximum_overrun),
            "gross_hourly_capacity_relief": float(total_gross_relief),
            "true_net_curtailment": float(total_true_curtailment),
            "max_abs_energy_balance_error": float(
                max_abs_energy_balance_error
            ),
            "useful_settlement_dr_energy": float(total_useful_settlement_dr),
            "excess_settlement_dr_energy": float(total_excess_settlement_dr),
            "target_tracking_mae_kwh": float(
                total_tracking_abs_error / max(total_evaluated_steps, 1)
            ),
            "target_tracking_penalty": float(total_tracking_penalty),
            "incentive_steps": int(total_incentive_steps),
            "unnecessary_incentive_steps": int(
                total_unnecessary_incentive_steps
            ),
            "positive_incentive_no_response_steps": int(
                total_positive_incentive_no_response_steps
            ),
            "maximum_action_steps": int(total_maximum_action_steps),
            "no_need_gate_steps": int(total_no_need_gate_steps),
            "no_need_offer_penalty": float(total_no_need_offer_penalty),
            "combined_offer_penalty": float(total_combined_offer_penalty),
            "stakeholder_sp_normalized_component": float(
                total_stakeholder_sp_component
            ),
            "stakeholder_eu_normalized_component": float(
                total_stakeholder_eu_component
            ),
            "stakeholder_reward_component": float(
                total_stakeholder_reward_component
            ),
            "symmetric_tracking_loss": float(total_symmetric_tracking_loss),
            "symmetric_tracking_penalty": float(
                total_symmetric_tracking_penalty
            ),
            "average_offered_incentive_rate": float(
                total_offered_incentive
                / max(total_evaluated_steps * self.train_env.N, 1)
            ),
            "incentive_payment": float(total_incentive_payment),
            "raw_discomfort": float(total_discomfort),
        }

        self.validation_history.append(metrics)
        self.eval_timesteps.append(int(completed_episode))
        self.eval_rewards.append(metrics["mean_reward"])
        self.eval_violation_steps.append(metrics["violation_steps"])
        self.eval_cumulative_excess.append(metrics["cumulative_excess"])
        self.eval_maximum_overrun.append(metrics["maximum_overrun"])
        self.eval_energy_balance_error.append(
            metrics["max_abs_energy_balance_error"]
        )
        self.last_validation_episode = int(completed_episode)

        print(
            f"Validation at episode {completed_episode}: "
            f"mean reward={metrics['mean_reward']:.2f} | "
            f"violations={metrics['violation_steps']} | "
            f"cumulative excess={metrics['cumulative_excess']:.4f} | "
            f"max overrun={metrics['maximum_overrun']:.4f}"
        )

        self._update_best_validation_model(metrics)
        self.save_metrics()
        return metrics

    @staticmethod
    def _selection_key(metrics):
        """Select within each reward arm by deterministic validation return."""
        return (
            -round(float(metrics["mean_reward"]), 10),
            round(float(metrics["std_reward"]), 10),
            int(metrics["episode"]),
        )

    def _update_best_validation_model(self, metrics):
        candidate_key = self._selection_key(metrics)

        if self.best_selection_key is None or candidate_key < self.best_selection_key:
            self.best_selection_key = candidate_key
            self.best_validation_episode = int(metrics["episode"])
            self.best_validation_metrics = dict(metrics)

            best_path = os.path.join(
                self.results_dir,
                "ddqn_best.pth",
            )
            self._save_model(
                best_path,
                episode=self.best_validation_episode,
                model_type="best_validation",
                validation_metrics=metrics,
            )

            summary_path = os.path.join(
                self.results_dir,
                "validation_summary.json",
            )
            with open(summary_path, "w", encoding="utf-8") as file:
                json.dump(
                    {
                        "selection_rule": [
                            "maximum deterministic mean validation reward",
                            "minimum validation reward standard deviation (tie-break)",
                            "earliest episode (final tie-break)",
                        ],
                        "run_seed": int(self.run_seed),
                        "config_signature": self.train_env.get_config_signature(),
                        "household_coefficients_source": str(
                            self.train_env.household_coefficients_source
                        ),
                        "household_coefficients_hash": str(
                            self.train_env.household_coefficients_hash
                        ),
                        "selection_key": list(candidate_key),
                        "best_metrics": metrics,
                    },
                    file,
                    indent=2,
                    default=str,
                )

            print(f"Best validation checkpoint: episode {self.best_validation_episode}")

    def train(self):

        for episode_index in range(self.episodes):
            completed_episode = episode_index + 1

            state = self.train_env.reset(mode="train")
            episode_reward = 0.0
            episode_loss = 0.0
            loss_count = 0
            self.current_episode_q_values = []

            step = -1
            for step in range(self.max_steps):
                action = self.agent.select_action(state, eval_mode=False)

                state_tensor = torch.as_tensor(
                    state,
                    dtype=torch.float32,
                    device=self.device,
                ).unsqueeze(0)
                with torch.no_grad():
                    q_values = self.agent.policy_net(state_tensor)
                    self.current_episode_q_values.append(
                        float(q_values.max().item())
                    )

                next_state, reward, done, _ = self.train_env.step(action)

                self.agent.memory.add(
                    state,
                    action,
                    reward,
                    next_state,
                    done,
                )

                loss = self.agent.train()
                if loss is not None:
                    episode_loss += float(loss)
                    loss_count += 1


                    self.agent.update_target_network()

                state = next_state
                episode_reward += float(reward)

                if done:
                    break

            avg_q = (
                float(np.mean(self.current_episode_q_values))
                if self.current_episode_q_values
                else 0.0
            )
            avg_loss = (
                episode_loss / loss_count
                if loss_count > 0
                else 0.0
            )


            self.episode_rewards.append(float(episode_reward))
            self.episode_losses.append(float(avg_loss))
            self.episode_lengths.append(int(step + 1))
            self.epsilon_history.append(float(self.agent.epsilon))
            self.episode_q_values.append(avg_q)

            self.agent.update_epsilon()

            if completed_episode % 50 == 0 or completed_episode == 1:
                self.log_progress(completed_episode)

            if completed_episode % self.eval_interval == 0:
                self.evaluate_policy(completed_episode)

            if completed_episode % 500 == 0:
                self.save_checkpoint(completed_episode)


        if self.last_validation_episode != self.episodes:
            self.evaluate_policy(self.episodes)

        self.finalize_training()

    def log_progress(self, completed_episode):
        window_size = min(50, len(self.episode_rewards))
        avg_reward = float(
            np.mean(self.episode_rewards[-window_size:])
        )
        avg_loss = float(
            np.mean(self.episode_losses[-window_size:])
        )

        print(
            f"Episode {completed_episode}/{self.episodes} | "
            f"Reward: {self.episode_rewards[-1]:.2f} | "
            f"Avg Reward: {avg_reward:.2f} | "
            f"Avg Loss: {avg_loss:.4f} | "
            f"Epsilon: {self.agent.epsilon:.4f}"
        )

    def _checkpoint_payload(
        self,
        episode,
        model_type,
        validation_metrics=None,
    ):
        payload = {

            "policy_net_state_dict": self.agent.policy_net.state_dict(),
            "target_net_state_dict": self.agent.target_net.state_dict(),
            "optimizer_state_dict": self.agent.optimizer.state_dict(),
            "training_history": self.agent.training_history,
            "epsilon": float(self.agent.epsilon),
            "steps_done": int(self.agent.steps_done),


            "episode": int(episode),
            "model_type": str(model_type),
            "script_version": SCRIPT_VERSION,
            "environment_version": self.train_env.VERSION,
            "accounting_version": self.train_env.ACCOUNTING_VERSION,
            "config_signature": self.train_env.get_config_signature(),
            "state_dim": int(self.state_dim),
            "state_feature_names": self.train_env.get_state_feature_names(),
            "action_dim": int(self.action_dim),
            "seed": int(self.run_seed),
            "household_coefficients_source": str(
                self.train_env.household_coefficients_source
            ),
            "household_coefficients_hash": str(
                self.train_env.household_coefficients_hash
            ),
            "train_ranges": self.cfg["training"]["train_ranges"],
            "val_range": self.cfg["training"]["val_range"],
            "test_range": self.cfg["training"]["test_range"],
            "time_steps_train": int(self.train_env.time_steps_train),
            "time_steps_test": int(self.train_env.time_steps_test),
            "expected_hours_per_day": int(self.train_env.expected_hours_per_day),
            "config": self.cfg,

            "household_coefficients": torch.as_tensor(
                self.train_env.household_coefficients, dtype=torch.float64
            ).cpu(),
            "house_ids": list(self.train_env.data_ids),
            "device_names": list(self.train_env.DEVICES),
            "device_non_interruptible": list(
                self.cfg["environment"]["DEVICE_NON_INTERRUPTIBLE"]
            ),
            "power_rate": [float(value) for value in self.train_env.POWER_RATE],
            "ts_deadline_hour": dict(self.train_env.ts_deadline_hour),
            "capacity_threshold": float(self.train_env.capacity_threshold),
            "rho": float(self.train_env.rho),
            "tie_break_enabled": bool(self.agent.tie_break_enabled),
            "tie_break_q_tolerance": float(
                self.agent.tie_break_q_tolerance
            ),
            "action_cost_definition": "mean(normalized_raw_joint_action)",


            "reward_mode": str(self.train_env.reward_mode),
            "c_cap": float(self.train_env.c_cap),
            "c_over": float(self.train_env.c_over),
            **self.train_env.get_reward_metadata(),

            "validation_metrics": validation_metrics,
            "best_validation_episode": self.best_validation_episode,
            "best_validation_metrics": self.best_validation_metrics,


            "torch_random_state": torch.get_rng_state(),
        }

        if torch.cuda.is_available():
            payload["torch_cuda_random_state_all"] = (
                torch.cuda.get_rng_state_all()
            )

        return payload

    def _save_model(
        self,
        path,
        episode,
        model_type,
        validation_metrics=None,
    ):
        torch.save(
            self._checkpoint_payload(
                episode=episode,
                model_type=model_type,
                validation_metrics=validation_metrics,
            ),
            path,
        )

    def save_checkpoint(self, completed_episode):
        path = os.path.join(
            self.results_dir,
            f"ddqn_episode_{completed_episode}.pth",
        )
        self._save_model(
            path,
            episode=completed_episode,
            model_type="periodic_checkpoint",
            validation_metrics=(
                self.validation_history[-1]
                if self.validation_history
                else None
            ),
        )
        self.save_metrics()

    def save_metrics(self):
        metrics = {
            "script_version": SCRIPT_VERSION,
            "environment_version": self.train_env.VERSION,
            "accounting_version": self.train_env.ACCOUNTING_VERSION,
            "config_signature": self.train_env.get_config_signature(),
            "run_seed": int(self.run_seed),
            "household_coefficients_source": str(
                self.train_env.household_coefficients_source
            ),
            "household_coefficients_hash": str(
                self.train_env.household_coefficients_hash
            ),
            "state_dim": self.state_dim,
            "state_feature_names": self.train_env.get_state_feature_names(),
            "episode_rewards": self.episode_rewards,
            "episode_losses": self.episode_losses,
            "episode_lengths": self.episode_lengths,
            "epsilon_history": self.epsilon_history,
            "episode_q_values": self.episode_q_values,
            "eval_rewards": self.eval_rewards,
            "eval_timesteps": self.eval_timesteps,
            "eval_violation_steps": self.eval_violation_steps,
            "eval_cumulative_excess": self.eval_cumulative_excess,
            "eval_maximum_overrun": self.eval_maximum_overrun,
            "eval_energy_balance_error": self.eval_energy_balance_error,
            "validation_history": self.validation_history,
            "best_validation_episode": self.best_validation_episode,
            "best_validation_metrics": self.best_validation_metrics,
            "best_selection_key": self.best_selection_key,
        }
        np.save(
            os.path.join(self.results_dir, "training_metrics.npy"),
            metrics,
            allow_pickle=True,
        )

        csv_path = os.path.join(
            self.results_dir,
            "validation_metrics.csv",
        )
        if self.validation_history:
            fieldnames = list(self.validation_history[0].keys())
            with open(csv_path, "w", newline="", encoding="utf-8") as file:
                writer = csv.DictWriter(file, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(self.validation_history)

    def finalize_training(self):
        final_model_path = os.path.join(
            self.results_dir,
            "ddqn_final.pth",
        )
        final_validation = (
            self.validation_history[-1]
            if self.validation_history
            else None
        )
        self._save_model(
            final_model_path,
            episode=self.episodes,
            model_type="final_episode",
            validation_metrics=final_validation,
        )

        self.save_metrics()
        self.plot_training_curves()

        if self.best_validation_episode is None:
            raise RuntimeError("No best validation checkpoint was created.")
        print(
            f"Seed {self.run_seed} complete | best_episode={self.best_validation_episode} | "
            f"checkpoint={os.path.join(self.results_dir, 'ddqn_best.pth')}"
        )

    @staticmethod
    def _moving_average(values, window=50):
        values = np.asarray(values, dtype=float)
        if len(values) < window:
            return np.array([]), np.array([])
        kernel = np.ones(window, dtype=float) / window
        averaged = np.convolve(values, kernel, mode="valid")
        x = np.arange(window, len(values) + 1)
        return x, averaged

    def _mark_best_episode(self):
        if self.best_validation_episode is not None:
            plt.axvline(
                self.best_validation_episode,
                color="black",
                linestyle="--",
                linewidth=2,
                label=(
                    f"Best validation episode "
                    f"{self.best_validation_episode}"
                ),
            )

    def plot_training_curves(self):
        try:
            if self.episode_rewards:
                plt.figure(figsize=(12, 8))
                episodes = np.arange(1, len(self.episode_rewards) + 1)
                plt.plot(
                    episodes,
                    self.episode_rewards,
                    color="blue",
                    linewidth=1.0,
                    alpha=0.45,
                    label="Episode reward",
                )
                x_ma, reward_ma = self._moving_average(
                    self.episode_rewards,
                    window=50,
                )
                if len(reward_ma):
                    plt.plot(
                        x_ma,
                        reward_ma,
                        color="darkblue",
                        linewidth=3,
                        label="50-episode moving average",
                    )
                self._mark_best_episode()
                plt.title("Training Reward")
                plt.xlabel("Training Episode")
                plt.ylabel("Reward")
                plt.legend(fontsize=12)
                plt.tight_layout()
                plt.savefig(
                    os.path.join(
                        self.results_dir,
                        "episode_reward.png",
                    ),
                    dpi=300,
                    bbox_inches="tight",
                )
                plt.close()

            if self.episode_losses:
                plt.figure(figsize=(12, 8))
                plt.plot(
                    np.arange(1, len(self.episode_losses) + 1),
                    self.episode_losses,
                    color="red",
                    linewidth=1.5,
                )
                self._mark_best_episode()
                plt.title("Training Loss")
                plt.xlabel("Training Episode")
                plt.ylabel("Average Loss")
                plt.legend(fontsize=12)
                plt.tight_layout()
                plt.savefig(
                    os.path.join(
                        self.results_dir,
                        "training_loss.png",
                    ),
                    dpi=300,
                    bbox_inches="tight",
                )
                plt.close()

            if self.epsilon_history:
                plt.figure(figsize=(12, 8))
                plt.plot(
                    np.arange(1, len(self.epsilon_history) + 1),
                    self.epsilon_history,
                    color="purple",
                    linewidth=2,
                )
                self._mark_best_episode()
                plt.title("Epsilon Decay")
                plt.xlabel("Training Episode")
                plt.ylabel("Epsilon")
                plt.legend(fontsize=12)
                plt.tight_layout()
                plt.savefig(
                    os.path.join(
                        self.results_dir,
                        "epsilon_decay.png",
                    ),
                    dpi=300,
                    bbox_inches="tight",
                )
                plt.close()

            if self.eval_rewards:
                plt.figure(figsize=(12, 8))
                plt.plot(
                    self.eval_timesteps,
                    self.eval_rewards,
                    "o-",
                    color="darkorange",
                    linewidth=3,
                    markersize=7,
                    label="Mean validation reward",
                )
                self._mark_best_episode()
                plt.title("Deterministic Validation Reward")
                plt.xlabel("Training Episode")
                plt.ylabel("Mean Validation Reward")
                plt.legend(fontsize=12)
                plt.tight_layout()
                plt.savefig(
                    os.path.join(
                        self.results_dir,
                        "evaluation_curve.png",
                    ),
                    dpi=300,
                    bbox_inches="tight",
                )
                plt.close()

                plt.figure(figsize=(12, 8))
                plt.plot(
                    np.arange(1, len(self.episode_rewards) + 1),
                    self.episode_rewards,
                    color="blue",
                    linewidth=1,
                    alpha=0.55,
                    label="Training reward",
                )
                plt.plot(
                    self.eval_timesteps,
                    self.eval_rewards,
                    "ro-",
                    linewidth=2.5,
                    markersize=6,
                    label="Mean validation reward",
                )
                self._mark_best_episode()
                plt.title("Training vs Validation Performance")
                plt.xlabel("Training Episode")
                plt.ylabel("Reward")
                plt.legend(fontsize=12)
                plt.tight_layout()
                plt.savefig(
                    os.path.join(
                        self.results_dir,
                        "train_vs_validation.png",
                    ),
                    dpi=300,
                    bbox_inches="tight",
                )
                plt.close()

                plt.figure(figsize=(12, 8))
                plt.plot(
                    self.eval_timesteps,
                    self.eval_violation_steps,
                    "o-",
                    linewidth=3,
                    label="Violation steps",
                )
                self._mark_best_episode()
                plt.title("Validation Capacity Violations")
                plt.xlabel("Training Episode")
                plt.ylabel("Violation Steps")
                plt.legend(fontsize=12)
                plt.tight_layout()
                plt.savefig(
                    os.path.join(
                        self.results_dir,
                        "validation_violation_steps.png",
                    ),
                    dpi=300,
                    bbox_inches="tight",
                )
                plt.close()

                plt.figure(figsize=(12, 8))
                plt.plot(
                    self.eval_timesteps,
                    self.eval_cumulative_excess,
                    "o-",
                    linewidth=3,
                    label="Cumulative excess",
                )
                self._mark_best_episode()
                plt.title("Validation Cumulative Capacity Excess")
                plt.xlabel("Training Episode")
                plt.ylabel("Cumulative Excess (kWh)")
                plt.legend(fontsize=12)
                plt.tight_layout()
                plt.savefig(
                    os.path.join(
                        self.results_dir,
                        "validation_cumulative_excess.png",
                    ),
                    dpi=300,
                    bbox_inches="tight",
                )
                plt.close()


        except Exception as exc:
            raise RuntimeError("Could not save training figures") from exc


def set_global_seeds(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main():
    parser = argparse.ArgumentParser(
        description="Train one DDQN run using an explicit configuration."
    )
    parser.add_argument(
        "--config_path",
        default=None,
        help=(
            "Optional YAML configuration path. When omitted, the project's "
            "default utils.config_loader.load_config() is used."
        ),
    )
    parser.add_argument(
        "--output_dir",
        required=True,
        help=(
            "Isolated training output directory for this ETA run. "
            "The official results/seed_0/training directory is rejected."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help=(
            "Optional training-seed override. The value is written into "
            "general.seed before environments and networks are created."
        ),
    )
    args = parser.parse_args()

    if args.config_path:
        with open(args.config_path, "r", encoding="utf-8") as handle:
            cfg = yaml.safe_load(handle)
        if not isinstance(cfg, dict):
            raise TypeError(f"Configuration is not a dictionary: {args.config_path}")
    else:
        cfg = load_config()

    if args.seed is not None:
        cfg.setdefault("general", {})["seed"] = int(args.seed)

    seed = int(cfg.get("general", {}).get("seed", 0))
    fixed_coefficients = (
        cfg.get("environment", {}).get(
            "fixed_household_coefficients", None
        )
    )
    if fixed_coefficients is None:
        raise ValueError(
            "Multi-seed replication requires "
            "environment.fixed_household_coefficients so household "
            "preferences remain identical across seeds."
        )

    set_global_seeds(seed=seed)
    trainer = DDQNTrainer(cfg_override=cfg, output_dir=args.output_dir)
    trainer.train()


if __name__ == "__main__":
    main()