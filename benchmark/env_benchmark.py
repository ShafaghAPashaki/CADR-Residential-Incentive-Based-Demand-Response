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
import os
from copy import deepcopy
from typing import Any, Iterable

import numpy as np
import pandas as pd

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

    VERSION = "benchmark_elasticity_env_v3_fair_forecast"
    ACCOUNTING_VERSION = "benchmark_curtailment_ledger_v2"
    RESPONSE_MODEL_VERSION = "direct_action_elasticity_v2"
    REWARD_VERSION = "raw_weighted_sp_eu_v1"
    NEGATIVE_PRICE_CONVENTION = "previous_day_min_strictly_positive_reference_v1"

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
            for key in ("general", "environment", "training", "DDQN", "forecast")
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
        self._load_decision_state_forecasts()

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

    @staticmethod
    def _sha256_file(path: str) -> str:
        hasher = hashlib.sha256()
        with open(path, "rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                hasher.update(chunk)
        return hasher.hexdigest()

    def _load_decision_state_forecasts(self) -> None:
        """Load the same frozen one-step forecast inputs used by CADR."""
        forecast_cfg = self.cfg.get("forecast", {}) or {}
        self.forecast_state_mode = str(
            forecast_cfg.get("state_input_mode", "forecast")
        ).strip().lower()
        if self.forecast_state_mode not in {"forecast", "oracle_actual"}:
            raise ValueError(
                "forecast.state_input_mode must be 'forecast' or 'oracle_actual'."
            )

        self.forecast_state_input_file = str(
            forecast_cfg.get(
                "ddqn_state_input_file",
                "results/forecast/ddqn_state_forecasts.csv",
            )
        )
        self.forecast_state_provenance_file = str(
            forecast_cfg.get(
                "ddqn_state_provenance_file",
                "results/forecast/ddqn_state_forecasts_provenance.json",
            )
        )
        self.forecast_load_floor_kw = float(
            forecast_cfg.get("state_load_floor_kw", 0.0)
        )
        if (
            not np.isfinite(self.forecast_load_floor_kw)
            or self.forecast_load_floor_kw < 0.0
        ):
            raise ValueError(
                "forecast.state_load_floor_kw must be finite and non-negative."
            )

        path = self.forecast_state_input_file
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Benchmark decision-state forecast file not found: {path}"
            )
        table = pd.read_csv(path, parse_dates=["timestamp"])
        if any("actual" in str(column).lower() for column in table.columns):
            raise ValueError(
                "Benchmark decision-state forecast input must not contain "
                "audit/actual columns."
            )

        self.forecast_load_columns = [
            f"forecast_load_house_{int(house_id)}_kw"
            for house_id in self.data_ids
        ]
        required = [
            "timestamp",
            "doy",
            "forecast_source",
            *self.forecast_load_columns,
            "forecast_aggregate_load_kw",
            "forecast_price_cent_per_kwh",
        ]
        missing = [column for column in required if column not in table.columns]
        if missing:
            raise ValueError(
                f"Benchmark state-forecast file is missing columns: {missing}"
            )
        if table["timestamp"].duplicated().any():
            raise ValueError("Duplicate timestamps in benchmark state forecasts.")

        numeric = table[
            self.forecast_load_columns
            + ["forecast_aggregate_load_kw", "forecast_price_cent_per_kwh"]
        ].to_numpy(dtype=float)
        if not np.isfinite(numeric).all():
            raise ValueError("Non-finite benchmark state forecasts were found.")

        aggregate_error = np.max(
            np.abs(
                table[self.forecast_load_columns].sum(axis=1).to_numpy(dtype=float)
                - table["forecast_aggregate_load_kw"].to_numpy(dtype=float)
            )
        )
        if aggregate_error > 1e-8:
            raise ValueError(
                "Benchmark forecast aggregate is inconsistent with household "
                f"forecasts: max error={aggregate_error:.3e}."
            )

        train_days = {
            day
            for start, end in self.cfg["training"]["train_ranges"]
            for day in range(int(start), int(end) + 1)
        }
        for day in train_days:
            rows = table[table["doy"] == int(day)]
            if len(rows) != self.expected_hours_per_day:
                raise ValueError(
                    f"Forecast day {day} has {len(rows)} rows; expected "
                    f"{self.expected_hours_per_day}."
                )
            if set(rows["forecast_source"].astype(str)) != {"crossfit"}:
                raise ValueError(
                    f"Benchmark training day {day} must use cross-fitted forecasts."
                )

        val_start, _ = self.cfg["training"]["val_range"]
        max_forecast_day = int(table["doy"].max())
        for day in range(int(val_start), max_forecast_day + 1):
            rows = table[table["doy"] == int(day)]
            if len(rows) != self.expected_hours_per_day:
                raise ValueError(
                    f"Forecast day {day} has {len(rows)} rows; expected "
                    f"{self.expected_hours_per_day}."
                )
            if set(rows["forecast_source"].astype(str)) != {"full_model"}:
                raise ValueError(
                    "Every benchmark forecast day from validation onward must "
                    f"use full-model forecasts; day {day} does not."
                )

        self.decision_state_forecasts = table.sort_values("timestamp").set_index(
            "timestamp", drop=False
        )
        self.forecast_state_input_sha256 = self._sha256_file(path)
        self.forecast_state_provenance_sha256 = None
        if os.path.exists(self.forecast_state_provenance_file):
            self.forecast_state_provenance_sha256 = self._sha256_file(
                self.forecast_state_provenance_file
            )

    def get_forecast_metadata(self) -> dict[str, Any]:
        return {
            "forecast_state_mode": str(self.forecast_state_mode),
            "forecast_state_input_file": str(self.forecast_state_input_file),
            "forecast_state_input_sha256": str(self.forecast_state_input_sha256),
            "forecast_state_provenance_file": str(
                self.forecast_state_provenance_file
            ),
            "forecast_state_provenance_sha256": self.forecast_state_provenance_sha256,
            "forecast_training_source": "crossfit",
            "forecast_validation_test_source": "full_model",
            "forecast_load_floor_kw": float(self.forecast_load_floor_kw),
            "forecast_price_unit": "cent/kWh",
        }

    def _previous_day_positive_price_reference(self, day: int) -> float:
        """Return the minimum strictly positive price observed on day d-1."""
        previous_day = int(day) - 1
        if previous_day < 1:
            raise ValueError(
                "Previous-day price reference requires an episode day >= 2."
            )
        index = self._expected_day_index(previous_day)
        rows = self.df_price.reindex(index)
        if len(rows) != self.expected_hours_per_day or rows["price"].isna().any():
            raise ValueError(
                f"Previous day {previous_day} does not have 24 aligned prices."
            )
        positive = rows.loc[rows["price"] > 0.0, "price"].to_numpy(dtype=float)
        if positive.size == 0:
            raise ValueError(
                f"Previous day {previous_day} has no strictly positive price."
            )
        return float(np.min(positive))

    def _validate_config_boundary(self) -> None:
        required_sections = {"general", "environment", "training", "DDQN", "forecast"}
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

        forecast_day = self.decision_state_forecasts.reindex(expected_index)
        forecast_fields = self.forecast_load_columns + [
            "forecast_price_cent_per_kwh"
        ]
        if forecast_day[forecast_fields].isna().any().any():
            raise ValueError(
                f"Missing benchmark decision-state forecasts on day {self.day}."
            )
        self.state_forecast_load_per_house = forecast_day[
            self.forecast_load_columns
        ].to_numpy(dtype=float)
        self.state_forecast_price = forecast_day[
            "forecast_price_cent_per_kwh"
        ].to_numpy(dtype=float)
        self.state_forecast_source = tuple(
            forecast_day["forecast_source"].astype(str).tolist()
        )
        if len(set(self.state_forecast_source)) != 1:
            raise ValueError(
                f"Mixed benchmark forecast sources on day {self.day}: "
                f"{sorted(set(self.state_forecast_source))}."
            )

        # Minimal causal version of Appendix D.1: retain one flat daily
        # reference, but use the previous day's minimum strictly positive price,
        # which is known before the current day begins.
        self.previous_day_reference_price = (
            self._previous_day_positive_price_reference(self.day)
        )
        self.lambda_min = np.full(
            self.max_steps,
            self.lambda_min_multiplier * self.previous_day_reference_price,
            dtype=float,
        )
        self.lambda_max = np.full(
            self.max_steps,
            self.lambda_max_multiplier * self.previous_day_reference_price,
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
            "previous_day_reference_price": float(self.previous_day_reference_price),
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
        if self.forecast_state_mode == "oracle_actual":
            state_loads = self.baseline_per_house[h].astype(float)
            state_price = float(self.prices[h])
        else:
            state_loads = np.maximum(
                self.state_forecast_load_per_house[h],
                self.forecast_load_floor_kw,
            ).astype(float)
            state_price = float(self.state_forecast_price[h])

        price_for_state = float(np.log1p(np.clip(state_price, 0.0, 80.0)))
        elasticity = float(self.elasticities[h])
        hour_normalized = float(h / max(self.max_steps - 1, 1))
        total_state_load = float(state_loads.sum())
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
                state_loads,
                np.asarray(
                    [
                        price_for_state,
                        elasticity,
                        hour_normalized,
                        total_state_load,
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
            *[f"forecast_house_{house_id}_load_kW" for house_id in self.data_ids],
            "log_clipped_forecast_wholesale_price",
            "hourly_elasticity",
            "normalized_hour",
            "forecast_aggregate_load_kW",
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
            "incentive_reference_price": "minimum strictly positive price on previous day",
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
            "forecast": self.cfg["forecast"],
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
