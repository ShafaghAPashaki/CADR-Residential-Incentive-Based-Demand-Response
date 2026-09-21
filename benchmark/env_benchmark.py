"""Capacity-blind elasticity-based DDQN benchmark environment.

Scientific identity
-------------------
This benchmark is deliberately independent of the proposed appliance-aware,
capacity-aware method.  It performs curtailment only.  The capacity threshold
is not read by this module and is never used in state construction, reward,
response, action selection, validation, or checkpoint selection.

The response model is the algebraically simplified form of the archived code::

    delta_E[n,h] = clip(
        baseline[n,h] * elasticity[h] * raw_action[n,h],
        0,
        max_reduction_fraction * baseline[n,h],
    )

Incentive rates price the response in the reward/accounting ledger; they do not
change its physical magnitude beyond the discrete raw action selected by DDQN.
"""

from __future__ import annotations

import hashlib
import itertools
import json
from copy import deepcopy
from typing import Any, Iterable

import numpy as np

from utils.config_loader import load_config
from utils.load_demand import load_baselines, load_demand
from utils.load_price import load_price

try:  # Preserve the archived holiday convention when the dependency is present.
    import holidays as _holidays
except ImportError:  # pragma: no cover - only used in minimal development envs.
    _holidays = None
    from pandas.tseries.holiday import USFederalHolidayCalendar


EPS = 1.0e-12


class Environment:
    """Elasticity-based, capacity-blind, curtailment-only environment."""

    VERSION = "benchmark_elasticity_env_v2"
    ACCOUNTING_VERSION = "benchmark_curtailment_ledger_v2"
    RESPONSE_MODEL_VERSION = "direct_action_elasticity_v2"
    REWARD_VERSION = "raw_weighted_sp_eu_v1"
    NEGATIVE_PRICE_CONVENTION = "daily_min_strictly_positive_reference_v1"

    _PROPOSED_ONLY_ENV_KEYS = {
        "capacity_threshold",
        "reward_mode",
        "c_cap",
        "c_over",
        "reward_normalization",
        "incentive_offer_regularization",
        "weighted_stakeholder",
        "target_tracking",
        "symmetric_tracking",
        "no_need_offer_regularization",
        "DEVICES",
        "DEVICE_CONSUMPTION",
        "DISSATISFACTION_COEFFICIENTS",
        "DISSATISFACTION_COEFFICIENTS_STD",
        "DISSATISFACTION_COEFFICIENTS_MIN",
        "DEVICE_NON_INTERRUPTIBLE",
        "POWER_RATE",
        "ts_deadline_hour",
        "DISABLE_PC",
        "fixed_household_coefficients",
    }

    def __init__(
        self,
        data_ids: Iterable[int],
        cfg_override: dict[str, Any] | None = None,
        rng_stream: int = 0,
    ) -> None:
        incoming_cfg = deepcopy(cfg_override) if cfg_override is not None else load_config()
        # Keep only executable benchmark sections inside the environment.  The
        # external capacity reference and test-period metadata are deliberately
        # not retained, so the environment cannot inspect them accidentally.
        self.cfg = {
            key: incoming_cfg[key]
            for key in ("general", "environment", "training", "DDQN")
        }
        self.data_ids = [int(value) for value in data_ids]
        if not self.data_ids or len(set(self.data_ids)) != len(self.data_ids):
            raise ValueError("data_ids must contain unique household IDs.")
        self.N = len(self.data_ids)

        self._validate_config_boundary()
        general = self.cfg["general"]
        env = self.cfg["environment"]
        training = self.cfg["training"]

        self.year = int(general.get("year", 2018))
        self.expected_hours_per_day = int(general.get("expected_hours_per_day", 24))
        self.run_seed = int(general.get("seed", 0))
        self.rng = np.random.default_rng(self.run_seed + int(rng_stream))

        self.df_demand = load_demand(
            data_path=str(general["load_file"]), house_ids=self.data_ids
        )
        self.df_price = load_price(data_path=str(general["price_file"]))
        self.all_baselines = load_baselines(self.df_demand)

        self.train_ranges = [tuple(map(int, value)) for value in training["train_ranges"]]
        self.val_range = tuple(map(int, training["val_range"]))
        self.max_steps = int(training["time_steps_train"])
        self.time_steps_train = self.max_steps
        self.time_steps_test = int(training.get("time_steps_test", self.max_steps))
        if self.max_steps != self.expected_hours_per_day:
            raise ValueError(
                "Benchmark episodes require one complete day: "
                f"time_steps_train={self.max_steps}, "
                f"expected_hours_per_day={self.expected_hours_per_day}."
            )

        self.rho = float(env["rho"])
        self.alpha = float(env["sp_reward_weight"])
        self.mu = np.asarray(env["mu"], dtype=float)
        self.kappa = float(env["kappa"])
        self.max_reduction_fraction = float(env["max_reduction_fraction"])
        incentive_cfg = env.get("incentive_bounds", {}) or {}
        self.lambda_min_multiplier = float(incentive_cfg.get("lower_multiplier", 0.3))
        self.lambda_max_multiplier = float(incentive_cfg.get("upper_multiplier", 1.0))

        if self.mu.shape != (self.N,):
            raise ValueError(f"environment.mu must contain {self.N} values; found {self.mu}.")
        if not (0.0 <= self.rho <= 1.0):
            raise ValueError("environment.rho must lie in [0, 1].")
        if not (0.0 <= self.alpha <= 1.0):
            raise ValueError("environment.sp_reward_weight must lie in [0, 1].")
        if self.kappa < 0.0 or np.any(self.mu < 0.0):
            raise ValueError("Discomfort coefficients must be non-negative.")
        if not (0.0 <= self.max_reduction_fraction <= 1.0):
            raise ValueError("max_reduction_fraction must lie in [0, 1].")
        if not (0.0 <= self.lambda_min_multiplier < self.lambda_max_multiplier):
            raise ValueError(
                "Incentive multipliers must satisfy "
                "0 <= lower_multiplier < upper_multiplier."
            )

        self.off_peak_hours = {int(value) for value in env["off_peak_hours"]}
        self.mid_peak_hours = {int(value) for value in env["mid_peak_hours"]}
        self.on_peak_hours = {int(value) for value in env["on_peak_hours"]}
        covered = self.off_peak_hours | self.mid_peak_hours | self.on_peak_hours
        if covered != set(range(self.expected_hours_per_day)):
            raise ValueError(
                "Peak-period hour sets must cover each hour 0..23 exactly once."
            )
        if (
            self.off_peak_hours & self.mid_peak_hours
            or self.off_peak_hours & self.on_peak_hours
            or self.mid_peak_hours & self.on_peak_hours
        ):
            raise ValueError("Peak-period hour sets must be disjoint.")

        elasticity_cfg = env["elasticity"]
        self.elasticity_by_period = {
            "off_peak": float(elasticity_cfg["off_peak"]),
            "mid_peak": float(elasticity_cfg["mid_peak"]),
            "on_peak": float(elasticity_cfg["on_peak"]),
        }
        if any(value < 0.0 for value in self.elasticity_by_period.values()):
            raise ValueError("Elasticities must be non-negative.")

        self.discrete_actions = np.linspace(0.0, 1.0, 5, dtype=float)
        self.all_actions = np.asarray(
            list(itertools.product(self.discrete_actions, repeat=self.N)), dtype=float
        )
        self.num_actions = int(len(self.all_actions))

        if _holidays is not None:
            self.us_holidays = _holidays.US(years=[self.year])
        else:  # pragma: no cover
            calendar = USFederalHolidayCalendar()
            start = f"{self.year}-01-01"
            end = f"{self.year}-12-31"
            self.us_holidays = set(calendar.holidays(start=start, end=end).date)

        self.episode = 0
        self.day: int | None = None
        self.curr_step = 0
        self.done = False
        self.state_dim = len(self.get_state_feature_names())

    def _validate_config_boundary(self) -> None:
        required_sections = {"general", "environment", "training", "DDQN"}
        missing_sections = sorted(required_sections.difference(self.cfg))
        if missing_sections:
            raise ValueError(f"Benchmark config is missing sections: {missing_sections}")

        env_keys = set(self.cfg["environment"])
        leaked = sorted(env_keys.intersection(self._PROPOSED_ONLY_ENV_KEYS))
        if leaked:
            raise ValueError(
                "Proposed-method/capacity fields are forbidden in the active "
                f"benchmark environment config: {leaked}"
            )

        required_env = {
            "house_ids",
            "off_peak_hours",
            "mid_peak_hours",
            "on_peak_hours",
            "elasticity",
            "mu",
            "kappa",
            "rho",
            "sp_reward_weight",
            "max_reduction_fraction",
        }
        missing_env = sorted(required_env.difference(env_keys))
        if missing_env:
            raise ValueError(f"Benchmark environment config is missing: {missing_env}")

        configured_ids = [int(value) for value in self.cfg["environment"]["house_ids"]]
        if configured_ids != self.data_ids:
            raise ValueError(
                f"Configured house_ids {configured_ids} do not match data_ids {self.data_ids}."
            )

    def _select_day(self, day: int | None, mode: str) -> int:
        if day is not None:
            return int(day)
        if mode == "train":
            range_idx = int(self.rng.integers(0, len(self.train_ranges)))
            start, end = self.train_ranges[range_idx]
            return int(self.rng.integers(start, end + 1))
        if mode == "val":
            start, end = self.val_range
            return int(self.rng.integers(start, end + 1))
        raise ValueError("A day must be supplied for test mode.")

    def reset(self, day: int | None = None, mode: str = "train") -> np.ndarray:
        self.day = self._select_day(day, mode)
        self.curr_step = 0
        self.episode += 1
        self.done = False

        expected_index = self._expected_day_index(self.day)
        price_day = self.df_price.reindex(expected_index)
        if len(price_day) != self.expected_hours_per_day or price_day["price"].isna().any():
            raise ValueError(f"Day {self.day} does not have 24 aligned wholesale prices.")
        self.price_df = price_day.copy()
        self.prices = price_day["price"].to_numpy(dtype=float)
        self.hours = expected_index.hour.to_numpy(dtype=int)

        timestamp = expected_index[0]
        self.is_weekend = 1.0 if timestamp.weekday() >= 5 else 0.0
        date_value = timestamp.date()
        self.is_holiday = 1.0 if date_value in self.us_holidays else 0.0

        day_rows = self.all_baselines[self.all_baselines["timestamp"].isin(expected_index)]
        baseline_wide = day_rows.pivot(index="timestamp", columns="house_id", values="baseline_demand")
        baseline_wide = baseline_wide.reindex(index=expected_index, columns=self.data_ids)
        if baseline_wide.isna().any().any():
            raise ValueError(
                f"Day {self.day} has missing household baselines; no forward filling is allowed."
            )
        self.baseline_per_house = baseline_wide.to_numpy(dtype=float)
        self.total_baseline = self.baseline_per_house.sum(axis=1)

        positive_prices = self.prices[self.prices > 0.0]
        self.daily_reference_price = (
            float(np.min(positive_prices)) if positive_prices.size else 0.0
        )
        self.lambda_min = np.full(
            self.max_steps,
            self.lambda_min_multiplier * self.daily_reference_price,
            dtype=float,
        )
        self.lambda_max = np.full(
            self.max_steps,
            self.lambda_max_multiplier * self.daily_reference_price,
            dtype=float,
        )

        shape = (self.max_steps, self.N)
        self.raw_actions = np.zeros(shape, dtype=float)
        self.action_indices = np.zeros(self.max_steps, dtype=int)
        self.incentives = np.zeros(shape, dtype=float)
        self.incremental_incentives = np.zeros(shape, dtype=float)
        self.reductions = np.zeros(shape, dtype=float)
        self.after_per_house = np.zeros(shape, dtype=float)
        self.incentive_payment_per_house = np.zeros(shape, dtype=float)
        self.discomforts = np.zeros(shape, dtype=float)
        self.rewards_customers = np.zeros(shape, dtype=float)
        self.rewards_service_provider = np.zeros(self.max_steps, dtype=float)
        self.rewards_total = np.zeros(self.max_steps, dtype=float)
        self.wholesale_avoided_value = np.zeros(self.max_steps, dtype=float)
        self.incentive_payment = np.zeros(self.max_steps, dtype=float)
        self.elasticities = np.asarray(
            [self.get_elasticity(hour) for hour in self.hours], dtype=float
        )

        state = self.get_state()
        if state.shape != (self.state_dim,):
            raise RuntimeError(
                f"State shape changed unexpectedly: {state.shape} != {(self.state_dim,)}"
            )
        return state

    def _expected_day_index(self, day: int):
        import pandas as pd

        start = pd.to_datetime(f"{self.year}-{int(day)}", format="%Y-%j")
        return pd.date_range(start, periods=self.expected_hours_per_day, freq="h")

    def step(self, action: int):
        if self.done:
            raise RuntimeError("step() was called after the episode terminated.")
        action_index = int(action)
        if not (0 <= action_index < self.num_actions):
            raise IndexError(f"Action index {action_index} outside [0, {self.num_actions}).")

        h = self.curr_step
        raw_action = self.all_actions[action_index].copy()
        self.action_indices[h] = action_index
        self.raw_actions[h] = raw_action

        lam_min_h = float(self.lambda_min[h])
        lam_max_h = float(self.lambda_max[h])
        gap = lam_max_h - lam_min_h
        nominal_rate = lam_min_h + raw_action * gap
        self.incentives[h] = nominal_rate
        self.incremental_incentives[h] = raw_action * gap

        delta_e = self.compute_delta_E(
            baselines=self.baseline_per_house[h],
            raw_action=raw_action,
            elasticity=float(self.elasticities[h]),
        )
        self.reductions[h] = delta_e
        self.after_per_house[h] = self.baseline_per_house[h] - delta_e

        payment = nominal_rate * delta_e
        self.incentive_payment_per_house[h] = payment
        self.incentive_payment[h] = float(payment.sum())
        self.wholesale_avoided_value[h] = float(self.prices[h] * delta_e.sum())

        sp_reward = self.wholesale_avoided_value[h] - self.incentive_payment[h]
        discomfort = 0.5 * self.mu * np.square(delta_e) + self.kappa * delta_e
        eu_reward = self.rho * payment - (1.0 - self.rho) * discomfort
        total_reward = self.alpha * sp_reward + (1.0 - self.alpha) * float(eu_reward.sum())

        self.discomforts[h] = discomfort
        self.rewards_customers[h] = eu_reward
        self.rewards_service_provider[h] = sp_reward
        self.rewards_total[h] = total_reward

        self.curr_step += 1
        self.done = self.curr_step >= self.max_steps
        observation = (
            np.zeros(self.state_dim, dtype=np.float32)
            if self.done
            else self.get_state()
        )
        info = {
            "day": int(self.day),
            "hour": int(self.hours[h]),
            "action_index": action_index,
            "raw_action": raw_action.copy(),
            "daily_reference_price": float(self.daily_reference_price),
            "wholesale_avoided_value": float(self.wholesale_avoided_value[h]),
            "incentive_payment": float(self.incentive_payment[h]),
            "sp_reward": float(sp_reward),
            "eu_reward_sum": float(eu_reward.sum()),
        }
        return observation, float(total_reward), self.done, info

    def get_state(self) -> np.ndarray:
        if self.day is None:
            return np.zeros(len(self.get_state_feature_names()), dtype=np.float32)
        h = self.curr_step
        baselines = self.baseline_per_house[h]
        price = float(self.prices[h])
        price_for_state = float(np.log1p(np.clip(price, 0.0, 80.0)))
        elasticity = float(self.elasticities[h])
        hour_normalized = float(h / max(self.max_steps - 1, 1))
        total_baseline = float(baselines.sum())
        current_hour = int(self.hours[h])
        hour_type = np.zeros(3, dtype=float)  # on, mid, off
        if current_hour in self.on_peak_hours:
            hour_type[0] = 1.0
        elif current_hour in self.mid_peak_hours:
            hour_type[1] = 1.0
        else:
            hour_type[2] = 1.0

        values = np.concatenate(
            [
                baselines.astype(float),
                np.asarray(
                    [
                        price_for_state,
                        elasticity,
                        hour_normalized,
                        total_baseline,
                        self.is_holiday,
                        self.is_weekend,
                    ],
                    dtype=float,
                ),
                hour_type,
            ]
        )
        return values.astype(np.float32)

    def get_state_feature_names(self) -> list[str]:
        return [
            *[f"baseline_house_{house_id}_kW" for house_id in self.data_ids],
            "log_clipped_wholesale_price",
            "hourly_elasticity",
            "normalized_hour",
            "aggregate_baseline_kW",
            "holiday_flag",
            "weekend_flag",
            "on_peak_flag",
            "mid_peak_flag",
            "off_peak_flag",
        ]

    def get_elasticity(self, hour: int) -> float:
        hour_value = int(hour)
        if hour_value in self.off_peak_hours:
            return self.elasticity_by_period["off_peak"]
        if hour_value in self.mid_peak_hours:
            return self.elasticity_by_period["mid_peak"]
        if hour_value in self.on_peak_hours:
            return self.elasticity_by_period["on_peak"]
        raise ValueError(f"Hour {hour_value} is absent from peak-period definitions.")

    def compute_delta_E(
        self,
        baselines: np.ndarray,
        raw_action: np.ndarray,
        elasticity: float,
    ) -> np.ndarray:
        baselines = np.asarray(baselines, dtype=float)
        raw_action = np.asarray(raw_action, dtype=float)
        if baselines.shape != (self.N,) or raw_action.shape != (self.N,):
            raise ValueError("baselines and raw_action must each have one value per household.")
        if self.daily_reference_price <= 0.0:
            return np.zeros(self.N, dtype=float)
        raw_delta = baselines * float(elasticity) * raw_action
        upper = self.max_reduction_fraction * baselines
        return np.clip(raw_delta, 0.0, upper)

    def get_reward_metadata(self) -> dict[str, Any]:
        return {
            "reward_version": self.REWARD_VERSION,
            "reward_normalized": False,
            "sp_reward_weight_alpha": float(self.alpha),
            "rho": float(self.rho),
            "mu": [float(value) for value in self.mu],
            "kappa": float(self.kappa),
            "sp_reward_equation": "price*sum(delta_E)-sum(lambda*delta_E)",
            "eu_reward_equation": (
                "rho*lambda*delta_E-(1-rho)*(0.5*mu*delta_E^2+kappa*delta_E)"
            ),
            "total_reward_equation": "alpha*R_SP+(1-alpha)*sum(R_EU)",
        }

    def get_response_metadata(self) -> dict[str, Any]:
        return {
            "response_model_version": self.RESPONSE_MODEL_VERSION,
            "response_equation": (
                "clip(baseline*elasticity*raw_action,0,max_reduction_fraction*baseline)"
            ),
            "max_reduction_fraction": float(self.max_reduction_fraction),
            "negative_price_convention": self.NEGATIVE_PRICE_CONVENTION,
            "daily_reference_price": "minimum strictly positive hourly price of the day",
            "lambda_min_multiplier": float(self.lambda_min_multiplier),
            "lambda_max_multiplier": float(self.lambda_max_multiplier),
            "zero_action_convention": (
                "nominal_rate=lambda_min; response=0; actual_payment=0"
            ),
            "capacity_blind": True,
            "curtailment_only": True,
        }

    def get_config_signature(self) -> str:
        relevant = {
            "general": {
                "load_file": self.cfg["general"]["load_file"],
                "price_file": self.cfg["general"]["price_file"],
                "year": self.year,
                "expected_hours_per_day": self.expected_hours_per_day,
                "seed": self.run_seed,
            },
            "environment": self.cfg["environment"],
            "training": self.cfg["training"],
            "DDQN": self.cfg["DDQN"],
            "versions": {
                "environment": self.VERSION,
                "accounting": self.ACCOUNTING_VERSION,
                "response": self.RESPONSE_MODEL_VERSION,
                "reward": self.REWARD_VERSION,
            },
        }
        payload = json.dumps(relevant, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def compute_total_demand(self, step: int | None = None):
        if step is None:
            return self.total_baseline.copy()
        return float(self.total_baseline[int(step)])

    def compute_total_reduction(self, step: int | None = None):
        if step is None:
            return self.reductions.sum(axis=1)
        return float(self.reductions[int(step)].sum())

    def compute_total_consumption(self, step: int | None = None):
        if step is None:
            return self.after_per_house.sum(axis=1)
        return float(self.after_per_house[int(step)].sum())
