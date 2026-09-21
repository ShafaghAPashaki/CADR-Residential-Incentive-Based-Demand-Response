"""Train the independent elasticity-based DDQN benchmark.

Run from the project root, for example::

    python -m benchmark.main_benchmark \
        --config benchmark/config_benchmark_seed_0.yaml

The best checkpoint is selected only from deterministic validation reward over
all days 168--181.  Capacity is not imported or consulted by this module.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml

from agent.agent_ddqn import DDQNAgent
from benchmark.env_benchmark import Environment
from utils.config_loader import load_config


SCRIPT_VERSION = "benchmark_ddqn_training_v2"
TIE_TOLERANCE = 1.0e-12


def set_reproducible_seed(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(int(seed))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


class BenchmarkTrainer:
    """Train three-seed-compatible benchmark with validation-only selection."""

    def __init__(self, cfg: dict[str, Any], smoke_test: bool = False) -> None:
        self.cfg = cfg
        self.seed = int(cfg["general"]["seed"])
        set_reproducible_seed(self.seed)

        house_ids = list(cfg["environment"]["house_ids"])
        self.train_env = Environment(house_ids, cfg_override=cfg, rng_stream=0)
        self.val_env = Environment(house_ids, cfg_override=cfg, rng_stream=100_000)
        initial_state = self.train_env.reset(mode="train")
        self.state_dim = int(len(initial_state))
        self.action_dim = int(self.train_env.num_actions)
        self.agent = DDQNAgent(
            self.state_dim,
            self.action_dim,
            cfg_override=cfg,
            action_costs=np.mean(self.train_env.all_actions, axis=1),
        )

        configured_steps = int(cfg["DDQN"]["max_steps"])
        if configured_steps != self.train_env.max_steps:
            raise ValueError(
                f"DDQN.max_steps={configured_steps} does not match environment "
                f"steps={self.train_env.max_steps}."
            )
        self.max_steps = configured_steps
        self.eval_interval = int(cfg["DDQN"]["eval_interval"])
        self.episodes = int(cfg["DDQN"]["episodes"])
        self.smoke_test = bool(smoke_test)
        if self.smoke_test:
            self.episodes = min(self.episodes, 2)
            self.eval_interval = 1

        val_start, val_end = map(int, cfg["training"]["val_range"])
        self.val_days = list(range(val_start, val_end + 1))
        if not self.val_days:
            raise ValueError("Validation range is empty.")

        output_root = "benchmark_smoke" if self.smoke_test else "benchmark"
        self.results_dir = PROJECT_ROOT / "results" / output_root / f"seed_{self.seed}" / "training"
        if self.results_dir.exists() and any(self.results_dir.iterdir()):
            raise FileExistsError(
                f"Training output already exists: {self.results_dir}. "
                "Move or remove it before rerunning this seed."
            )
        self.results_dir.mkdir(parents=True, exist_ok=True)
        with (self.results_dir / "config_snapshot.yaml").open("w", encoding="utf-8") as handle:
            yaml.safe_dump(cfg, handle, sort_keys=False)

        self.episode_rewards: list[float] = []
        self.episode_losses: list[float] = []
        self.episode_lengths: list[int] = []
        self.epsilon_history: list[float] = []
        self.episode_q_values: list[float] = []
        self.validation_history: list[dict[str, Any]] = []
        self.best_validation_episode: int | None = None
        self.best_validation_metrics: dict[str, Any] | None = None
        self.last_validation_episode: int | None = None

        print(
            f"Benchmark seed {self.seed} | device={self.agent.device} | "
            f"state_dim={self.state_dim} | action_dim={self.action_dim} | "
            f"episodes={self.episodes} | output={self.results_dir}"
        )

    def _checkpoint_metadata(
        self,
        episode: int,
        model_type: str,
        validation_metrics: dict[str, Any] | None,
    ) -> dict[str, Any]:
        return {
            "episode": int(episode),
            "model_type": str(model_type),
            "script_version": SCRIPT_VERSION,
            "environment_version": self.train_env.VERSION,
            "accounting_version": self.train_env.ACCOUNTING_VERSION,
            "response_model_version": self.train_env.RESPONSE_MODEL_VERSION,
            "reward_version": self.train_env.REWARD_VERSION,
            "negative_price_convention": self.train_env.NEGATIVE_PRICE_CONVENTION,
            "config_signature": self.train_env.get_config_signature(),
            "seed": int(self.seed),
            "state_feature_names": self.train_env.get_state_feature_names(),
            "house_ids": list(self.train_env.data_ids),
            "action_grid": [float(value) for value in self.train_env.discrete_actions],
            "train_ranges": self.cfg["training"]["train_ranges"],
            "val_range": self.cfg["training"]["val_range"],
            "config": self.cfg,
            "capacity_blind": True,
            "curtailment_only": True,
            "action_cost_definition": "mean(normalized_raw_joint_action)",
            **self.train_env.get_reward_metadata(),
            **self.train_env.get_response_metadata(),
            "validation_metrics": validation_metrics,
            "best_validation_episode": self.best_validation_episode,
            "best_validation_metrics": self.best_validation_metrics,
        }

    def _save_model(
        self,
        filename: str,
        episode: int,
        model_type: str,
        validation_metrics: dict[str, Any] | None = None,
    ) -> None:
        self.agent.save(
            str(self.results_dir / filename),
            metadata=self._checkpoint_metadata(episode, model_type, validation_metrics),
        )

    @staticmethod
    def _is_better(
        candidate: dict[str, Any], current: dict[str, Any] | None
    ) -> bool:
        if current is None:
            return True
        candidate_mean = float(candidate["mean_validation_reward"])
        current_mean = float(current["mean_validation_reward"])
        if candidate_mean > current_mean + TIE_TOLERANCE:
            return True
        if abs(candidate_mean - current_mean) <= TIE_TOLERANCE:
            candidate_sd = float(candidate["validation_reward_sd"])
            current_sd = float(current["validation_reward_sd"])
            if candidate_sd < current_sd - TIE_TOLERANCE:
                return True
        return False  # earliest episode is retained in a complete tie

    def evaluate_policy(self, completed_episode: int) -> dict[str, Any]:
        """Greedy evaluation on every validation day, without capacity metrics."""
        was_training = self.agent.policy_net.training
        python_state = random.getstate()
        numpy_state = np.random.get_state()
        torch_state = torch.get_rng_state()
        cuda_states = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        self.agent.policy_net.eval()

        day_rewards: list[float] = []
        total_curtailment = 0.0
        total_payment = 0.0
        total_avoided = 0.0
        intervention_hours = 0
        household_intervention_events = 0
        no_response_events = 0
        try:
            with torch.no_grad():
                for day in self.val_days:
                    state = self.val_env.reset(day=day, mode="val")
                    done = False
                    day_reward = 0.0
                    steps = 0
                    while not done and steps < self.max_steps:
                        action = self.agent.greedy_action(state)
                        state, reward, done, _ = self.val_env.step(action)
                        day_reward += float(reward)
                        steps += 1
                    if not done:
                        raise RuntimeError(
                            f"Validation day {day} did not finish within {self.max_steps} steps."
                        )
                    day_rewards.append(day_reward)
                    total_curtailment += float(self.val_env.reductions[:steps].sum())
                    total_payment += float(self.val_env.incentive_payment[:steps].sum())
                    total_avoided += float(self.val_env.wholesale_avoided_value[:steps].sum())
                    active = self.val_env.raw_actions[:steps] > 1.0e-12
                    intervention_hours += int(np.any(active, axis=1).sum())
                    household_intervention_events += int(active.sum())
                    no_response_events += int(
                        np.logical_and(active, self.val_env.reductions[:steps] <= 1.0e-12).sum()
                    )
        finally:
            random.setstate(python_state)
            np.random.set_state(numpy_state)
            torch.set_rng_state(torch_state)
            if cuda_states is not None:
                torch.cuda.set_rng_state_all(cuda_states)
            self.agent.policy_net.train(was_training)

        rewards = np.asarray(day_rewards, dtype=float)
        metrics = {
            "episode": int(completed_episode),
            "validation_days": int(len(self.val_days)),
            "mean_validation_reward": float(np.mean(rewards)),
            "validation_reward_sd": float(np.std(rewards, ddof=0)),
            "minimum_day_reward": float(np.min(rewards)),
            "maximum_day_reward": float(np.max(rewards)),
            "total_curtailment_kWh": float(total_curtailment),
            "incentive_payment_cent": float(total_payment),
            "avoided_wholesale_value_cent": float(total_avoided),
            "net_wholesale_after_payment_cent": float(total_avoided - total_payment),
            "intervention_hours": int(intervention_hours),
            "household_intervention_events": int(household_intervention_events),
            "no_response_events": int(no_response_events),
        }
        self.validation_history.append(metrics)
        self.last_validation_episode = int(completed_episode)

        if self._is_better(metrics, self.best_validation_metrics):
            self.best_validation_metrics = dict(metrics)
            self.best_validation_episode = int(completed_episode)
            self._save_model(
                "ddqn_best.pth",
                episode=completed_episode,
                model_type="best_validation",
                validation_metrics=metrics,
            )
            with (self.results_dir / "validation_summary.json").open(
                "w", encoding="utf-8"
            ) as handle:
                json.dump(
                    {
                        "selection_rule": (
                            "highest mean deterministic validation reward; lowest "
                            "population SD as tie-break; earliest episode in final tie"
                        ),
                        "capacity_used_for_selection": False,
                        "best_validation_episode": self.best_validation_episode,
                        "best_validation_metrics": self.best_validation_metrics,
                    },
                    handle,
                    indent=2,
                )
            print(
                f"  New benchmark best | episode={completed_episode} | "
                f"mean_val_reward={metrics['mean_validation_reward']:.6f} | "
                f"sd={metrics['validation_reward_sd']:.6f}"
            )
        return metrics

    def train(self) -> None:
        for episode in range(1, self.episodes + 1):
            state = self.train_env.reset(mode="train")
            episode_reward = 0.0
            loss_sum = 0.0
            loss_count = 0
            q_values: list[float] = []

            for step in range(self.max_steps):
                with torch.no_grad():
                    tensor = torch.as_tensor(
                        state, dtype=torch.float32, device=self.agent.device
                    ).unsqueeze(0)
                    q_values.append(float(self.agent.policy_net(tensor).max().item()))

                action = self.agent.select_action(state, eval_mode=False)
                next_state, reward, done, _ = self.train_env.step(action)
                self.agent.memory.add(state, action, reward, next_state, done)
                loss = self.agent.train()
                if loss is not None:
                    loss_sum += float(loss)
                    loss_count += 1
                    self.agent.update_target_network()
                state = next_state
                episode_reward += float(reward)
                if done:
                    break

            self.episode_rewards.append(float(episode_reward))
            self.episode_losses.append(float(loss_sum / loss_count if loss_count else 0.0))
            self.episode_lengths.append(int(step + 1))
            self.epsilon_history.append(float(self.agent.epsilon))
            self.episode_q_values.append(float(np.mean(q_values)) if q_values else 0.0)
            self.agent.update_epsilon()

            if episode == 1 or episode % 50 == 0:
                window = min(50, len(self.episode_rewards))
                print(
                    f"Episode {episode}/{self.episodes} | "
                    f"reward={episode_reward:.4f} | "
                    f"avg50={np.mean(self.episode_rewards[-window:]):.4f} | "
                    f"loss={self.episode_losses[-1]:.6f} | "
                    f"epsilon={self.agent.epsilon:.5f}"
                )

            if episode % self.eval_interval == 0:
                self.evaluate_policy(episode)
            if not self.smoke_test and episode % 500 == 0:
                self._save_model(
                    f"ddqn_episode_{episode}.pth",
                    episode=episode,
                    model_type="periodic_checkpoint",
                    validation_metrics=(self.validation_history[-1] if self.validation_history else None),
                )
                self.save_metrics()

        if self.last_validation_episode != self.episodes:
            self.evaluate_policy(self.episodes)
        self._save_model(
            "ddqn_final.pth",
            episode=self.episodes,
            model_type="final_episode",
            validation_metrics=(self.validation_history[-1] if self.validation_history else None),
        )
        self.save_metrics()
        self.plot_training_curves()
        if self.best_validation_episode is None:
            raise RuntimeError("No validation checkpoint was created.")
        print(
            f"Benchmark seed {self.seed} complete | "
            f"best_episode={self.best_validation_episode} | "
            f"checkpoint={self.results_dir / 'ddqn_best.pth'}"
        )

    def save_metrics(self) -> None:
        payload = {
            "script_version": SCRIPT_VERSION,
            "environment_version": self.train_env.VERSION,
            "accounting_version": self.train_env.ACCOUNTING_VERSION,
            "response_model_version": self.train_env.RESPONSE_MODEL_VERSION,
            "reward_version": self.train_env.REWARD_VERSION,
            "config_signature": self.train_env.get_config_signature(),
            "seed": self.seed,
            "state_dim": self.state_dim,
            "state_feature_names": self.train_env.get_state_feature_names(),
            "action_dim": self.action_dim,
            "episode_rewards": self.episode_rewards,
            "episode_losses": self.episode_losses,
            "episode_lengths": self.episode_lengths,
            "epsilon_history": self.epsilon_history,
            "episode_q_values": self.episode_q_values,
            "validation_history": self.validation_history,
            "best_validation_episode": self.best_validation_episode,
            "best_validation_metrics": self.best_validation_metrics,
        }
        np.save(self.results_dir / "training_metrics.npy", payload, allow_pickle=True)

        if self.validation_history:
            with (self.results_dir / "validation_metrics.csv").open(
                "w", newline="", encoding="utf-8"
            ) as handle:
                writer = csv.DictWriter(handle, fieldnames=list(self.validation_history[0]))
                writer.writeheader()
                writer.writerows(self.validation_history)

    def plot_training_curves(self) -> None:
        def save_line(values, ylabel, filename, x_values=None, marker=None):
            if not values:
                return
            x = np.asarray(x_values if x_values is not None else np.arange(1, len(values) + 1))
            plt.figure(figsize=(10, 6))
            plt.plot(x, np.asarray(values, dtype=float), marker=marker, linewidth=1.5)
            if self.best_validation_episode is not None:
                plt.axvline(self.best_validation_episode, linestyle="--", linewidth=1.5)
            plt.xlabel("Episode")
            plt.ylabel(ylabel)
            plt.tight_layout()
            plt.savefig(self.results_dir / filename, dpi=300, bbox_inches="tight")
            plt.close()

        save_line(self.episode_rewards, "Episode reward", "episode_reward.png")
        save_line(self.episode_losses, "Average loss", "training_loss.png")
        save_line(self.epsilon_history, "Epsilon", "epsilon_decay.png")
        if self.validation_history:
            save_line(
                [row["mean_validation_reward"] for row in self.validation_history],
                "Mean validation reward",
                "evaluation_curve.png",
                x_values=[row["episode"] for row in self.validation_history],
                marker="o",
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="benchmark/config_benchmark_seed_0.yaml",
        help="Benchmark YAML config path, relative to the project root or absolute.",
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Run two episodes into results/benchmark_smoke; never use for paper results.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = PROJECT_ROOT / config_path
    cfg = load_config(str(config_path))
    BenchmarkTrainer(cfg, smoke_test=args.smoke_test).train()


if __name__ == "__main__":
    main()
