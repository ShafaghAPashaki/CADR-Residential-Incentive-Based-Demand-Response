import hashlib
import itertools
import json
import os
import random

import holidays
import numpy as np
import pandas as pd
import torch

from utils.load_demand import (
    get_device_day_demands,
    get_device_demands,
    load_baselines,
    load_demand,
    load_device_demands,
)
from utils.load_price import load_price
from utils.config_loader import load_config


random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


class Environment:
    """
    DDQN residential demand-response environment with verified hourly
    settlement and explicit physical/economic diagnostics.

    Four quantities are intentionally kept separate:

    * ``hourly_capacity_relief`` is the non-negative relief caused by the
      CURRENT action relative to the physically committed pre-action load. It
      is used only for capacity control and reward shaping.
    * ``signed_load_change`` is baseline load minus realised post-DR load. It
      may be negative during shifted-in/rebound hours and is retained only for
      physical auditing and the optional net-procurement diagnostic.
    * ``settlement_dr_energy`` is the non-negative, baseline-verified hourly
      reduction ``max(0, baseline - realised load)``. It is the common energy
      quantity used in both terms of the official SP settlement objective and
      in the EU incentive benefit.
    * ``action_attributed_dr_energy`` is current curtailment plus the full
      profile of jobs newly shifted by the CURRENT action. It is retained as an
      action-level diagnostic only and never enters settlement.

    Multi-hour shifted jobs are settled through a source-level ledger. Each
    origin-hour slice is verified in the hour where the physical vacancy
    occurs, valued at that hour's wholesale price, and paid at the incentive
    rate locked when the job was shifted. If multiple sources contribute to one
    household's verified reduction in the same hour, the verified total is
    allocated proportionally across those sources.
    """

    VERSION = "forecast_state_v1"
    ACCOUNTING_VERSION = "verified_hourly_settlement"
    REWARD_DEFINITION_VERSION = "weighted_stakeholder_tracking_no_need"
    EPS = 1e-10

    def __init__(self, data_ids, cfg_override=None):
        self.cfg = cfg_override or load_config()
        self.data_ids = list(data_ids)
        self.N = len(self.data_ids)

        self.seed = int(
            self.cfg.get("general", {}).get(
                "seed", self.cfg.get("DDQN", {}).get("seed", 0)
            )
        )
        self.rng = np.random.default_rng(self.seed)

        self.disable_pc = bool(
            self.cfg["environment"].get("DISABLE_PC", False)
        )
        self.episode = 0
        self.year = int(self.cfg.get("general", {}).get("year", 2018))
        self.expected_hours_per_day = int(
            self.cfg.get("general", {}).get("expected_hours_per_day", 24)
        )
        self.data_tolerance = float(
            self.cfg.get("environment", {}).get("data_validation_tolerance", 1e-6)
        )

        self.DEVICES = list(self.cfg["environment"]["DEVICES"])
        self.num_devices = len(self.DEVICES)
        self.DEVICE_CONSUMPTION = np.asarray(
            self.cfg["environment"].get(
                "DEVICE_CONSUMPTION", [np.inf] * self.num_devices
            ),
            dtype=float,
        )
        if self.DEVICE_CONSUMPTION.shape != (self.num_devices,):
            raise ValueError("DEVICE_CONSUMPTION must match DEVICES length.")
        self.idx_car = (
            self.DEVICES.index("car") if "car" in self.DEVICES else None
        )

        self.df_demand = load_demand(
            data_path=self.cfg["general"]["load_file"],
            house_ids=self.data_ids,
        )
        self.df_device_demands = load_device_demands(
            data_path=self.cfg["general"]["load_file"],
            house_ids=self.data_ids,
            devices=self.DEVICES,
        )
        self.df_price = load_price(
            data_path=self.cfg["general"]["price_file"]
        )
        self.all_baselines = load_baselines(self.df_demand)

        # Decision-state forecasts are loaded once and routed by day. Training
        # days use out-of-fold/cross-fitted predictions; later days use the
        # fixed full-model forecasts. Physical response, reward, settlement,
        # and the incentive bound continue to use realised current quantities.
        self._load_decision_state_forecasts()

        self.train_ranges = self.cfg["training"]["train_ranges"]
        self.val_range = self.cfg["training"]["val_range"]
        self.test_range = self.cfg["training"]["test_range"]
        self._validate_temporal_split()
        self.time_steps_train = int(
            self.cfg["training"].get("time_steps_train", 24)
        )
        self.time_steps_test = int(
            self.cfg["training"].get("time_steps_test", self.time_steps_train)
        )
        self.max_steps = self.time_steps_train

        self.rho = float(self.cfg["environment"]["rho"])
        self.capacity_threshold = float(
            self.cfg["environment"]["capacity_threshold"]
        )


        reward_cfg = self.cfg["environment"]
        self.reward_mode = str(
            reward_cfg.get("reward_mode", "legacy_shaping")
        ).strip().lower()
        self.c_cap = float(reward_cfg.get("c_cap", 0.0))
        self.c_over = float(reward_cfg.get("c_over", 0.0))

        norm_cfg = reward_cfg.get("reward_normalization", {}) or {}
        self.reward_normalization_enabled = bool(norm_cfg.get("enabled", True))
        self.reward_price_percentile = float(norm_cfg.get("price_percentile", 95.0))
        self.reward_discomfort_percentile = float(
            norm_cfg.get("discomfort_percentile", 95.0)
        )
        self.configured_economic_scale = norm_cfg.get("economic_scale", None)
        self.configured_discomfort_scale = norm_cfg.get("discomfort_scale", None)
        self.discomfort_weight = float(norm_cfg.get("discomfort_weight", 1.0))
        self.discomfort_normalization_mode = str(
            norm_cfg.get("discomfort_mode", "per_class_mean")
        ).strip().lower()
        self.configured_discomfort_class_scales = (
            norm_cfg.get("discomfort_class_scales", {}) or {}
        )


        stakeholder_cfg = reward_cfg.get("weighted_stakeholder", {}) or {}
        self.stakeholder_sp_weight = float(
            stakeholder_cfg.get("sp_weight", 0.65)
        )
        self.stakeholder_eu_weight = 1.0 - self.stakeholder_sp_weight
        self.minimum_effective_payment_cost_fraction = float(
            stakeholder_cfg.get("minimum_effective_payment_cost_fraction", 0.5)
        )
        self.effective_payment_cost_coefficient = (
            self.stakeholder_sp_weight
            - self.stakeholder_eu_weight * self.rho
        )
        self.effective_payment_cost_fraction = (
            self.effective_payment_cost_coefficient
            / max(self.stakeholder_sp_weight, self.EPS)
        )

        offer_cfg = reward_cfg.get("incentive_offer_regularization", {}) or {}
        self.offer_regularization_enabled = bool(offer_cfg.get("enabled", True))
        self.offer_regularization_fraction = float(
            offer_cfg.get("fraction_of_1kwh_p95_value", 0.01)
        )
        configured_offer_coefficient = offer_cfg.get(
            "coefficient", "auto_one_percent_1kwh_p95"
        )
        if isinstance(configured_offer_coefficient, str):
            if configured_offer_coefficient.strip().lower() != "auto_one_percent_1kwh_p95":
                raise ValueError(
                    "incentive_offer_regularization.coefficient must be a "
                    "non-negative number or 'auto_one_percent_1kwh_p95'."
                )
            self.incentive_offer_coefficient = (
                self.offer_regularization_fraction
                / max(self.capacity_threshold, self.EPS)
            )
            self.incentive_offer_coefficient_source = (
                "auto_one_percent_1kwh_p95"
            )
        else:
            self.incentive_offer_coefficient = float(configured_offer_coefficient)
            self.incentive_offer_coefficient_source = "configured_numeric"

        track_cfg = reward_cfg.get("target_tracking", {}) or {}
        self.configured_track_c_violation = track_cfg.get(
            "track_c_violation", "auto_capacity_anchored"
        )
        self.configured_track_c_under = track_cfg.get(
            "track_c_under", "auto_economic_anchored"
        )
        self.track_c_violation, self.track_c_violation_source = (
            self._resolve_tracking_coefficient(
                self.configured_track_c_violation,
                kind="violation",
            )
        )
        self.track_c_under, self.track_c_under_source = (
            self._resolve_tracking_coefficient(
                self.configured_track_c_under,
                kind="under",
            )
        )
        self.configured_track_deadband_kwh = track_cfg.get(
            "deadband_kwh", "auto"
        )

        symmetric_cfg = reward_cfg.get("symmetric_tracking", {}) or {}
        self.symmetric_tracking_lambda = float(
            symmetric_cfg.get("lambda_c", 1.0)
        )
        self.symmetric_tracking_lambda_source = str(
            symmetric_cfg.get("lambda_source", "configured_numeric")
        )
        self.configured_huber_delta = symmetric_cfg.get(
            "huber_delta", "auto_deadband_normalized"
        )
        self.symmetric_tracking_loss_name = str(
            symmetric_cfg.get("loss", "smooth_l1")
        ).strip().lower()


        no_need_cfg = reward_cfg.get("no_need_offer_regularization", {}) or {}
        self.no_need_offer_regularization_enabled = bool(
            no_need_cfg.get("enabled", False)
        )
        self.no_need_offer_coefficient = float(
            no_need_cfg.get("coefficient", 0.0)
        )
        self.no_need_gate_tolerance_kw = float(
            no_need_cfg.get("gate_tolerance_kw", self.EPS)
        )

        valid_reward_modes = {
            "economic_only",
            "capacity_penalty",
            "legacy_shaping",
            "corrected_economic",
            "target_tracking",
            "target_tracking_no_need",
            "weighted_stakeholder_tracking",
            "weighted_stakeholder_tracking_no_need",
        }
        if self.reward_mode not in valid_reward_modes:
            raise ValueError(
                "environment.reward_mode must be one of "
                f"{sorted(valid_reward_modes)}; found {self.reward_mode!r}."
            )
        if self.reward_mode in {
            "target_tracking_no_need",
            "weighted_stakeholder_tracking_no_need",
        }:
            if not self.no_need_offer_regularization_enabled:
                raise ValueError(
                    f"{self.reward_mode} requires "
                    "no_need_offer_regularization.enabled=true."
                )
            if self.no_need_offer_coefficient <= 0.0:
                raise ValueError(
                    f"{self.reward_mode} requires a strictly positive "
                    "no_need_offer_regularization.coefficient."
                )
        if not 0.0 < self.stakeholder_sp_weight < 1.0:
            raise ValueError(
                "environment.weighted_stakeholder.sp_weight must be in (0, 1)."
            )
        theoretical_lower_bound = self.rho / (1.0 + self.rho)
        if self.stakeholder_sp_weight <= theoretical_lower_bound:
            raise ValueError(
                "weighted_stakeholder.sp_weight does not preserve incentive "
                "payment as a net cost: require omega > rho/(1+rho), found "
                f"omega={self.stakeholder_sp_weight}, rho={self.rho}."
            )
        if (
            self.effective_payment_cost_fraction + self.EPS
            < self.minimum_effective_payment_cost_fraction
        ):
            raise ValueError(
                "The effective incentive-payment cost is below the configured "
                "minimum fraction of the SP-value weight: "
                f"{self.effective_payment_cost_fraction:.6g} < "
                f"{self.minimum_effective_payment_cost_fraction:.6g}."
            )
        if self.symmetric_tracking_loss_name not in {"smooth_l1", "huber"}:
            raise ValueError(
                "environment.symmetric_tracking.loss must be 'smooth_l1' or 'huber'."
            )

        non_negative_values = {
            "c_cap": self.c_cap,
            "c_over": self.c_over,
            "reward_normalization.discomfort_weight": self.discomfort_weight,
            "incentive_offer_regularization.fraction_of_1kwh_p95_value": (
                self.offer_regularization_fraction
            ),
            "incentive_offer_regularization.coefficient": (
                self.incentive_offer_coefficient
            ),
            "target_tracking.track_c_violation": self.track_c_violation,
            "target_tracking.track_c_under": self.track_c_under,
            "symmetric_tracking.lambda_c": self.symmetric_tracking_lambda,
            "weighted_stakeholder.minimum_effective_payment_cost_fraction": (
                self.minimum_effective_payment_cost_fraction
            ),
            "no_need_offer_regularization.coefficient": (
                self.no_need_offer_coefficient
            ),
            "no_need_offer_regularization.gate_tolerance_kw": (
                self.no_need_gate_tolerance_kw
            ),
        }
        invalid = {name: value for name, value in non_negative_values.items() if value < 0.0}
        if invalid:
            raise ValueError(f"Reward coefficients must be non-negative; found {invalid}.")
        if not 0.0 < self.reward_price_percentile <= 100.0:
            raise ValueError("reward_normalization.price_percentile must be in (0, 100].")
        if not 0.0 < self.reward_discomfort_percentile <= 100.0:
            raise ValueError(
                "reward_normalization.discomfort_percentile must be in (0, 100]."
            )
        valid_discomfort_modes = {"global", "per_class_mean"}
        if self.discomfort_normalization_mode not in valid_discomfort_modes:
            raise ValueError(
                "reward_normalization.discomfort_mode must be one of "
                f"{sorted(valid_discomfort_modes)}; found "
                f"{self.discomfort_normalization_mode!r}."
            )

        self.discrete_actions = np.linspace(0.0, 1.0, 5)
        self.all_actions = np.array(
            list(itertools.product(self.discrete_actions, repeat=self.N)),
            dtype=float,
        )
        self.num_actions = len(self.all_actions)

        self.us_holidays = holidays.US(years=[self.year])

        dni = np.asarray(
            self.cfg["environment"]["DEVICE_NON_INTERRUPTIBLE"],
            dtype=int,
        )
        if dni.shape[0] != self.num_devices:
            raise ValueError(
                "DEVICE_NON_INTERRUPTIBLE length must match DEVICES length."
            )

        self.PC_MASK = dni == 0
        self.TS_NI_MASK = dni == 1
        self.TS_I_MASK = np.zeros(self.num_devices, dtype=bool)

        if self.idx_car is not None:
            self.TS_I_MASK[self.idx_car] = True
            self.PC_MASK[self.idx_car] = False
            self.TS_NI_MASK[self.idx_car] = False

        self.reward_discomfort_class_names = ("PC_AC", "EV_TS_I", "TS_NI")

        self.TS_MASK = self.TS_I_MASK | self.TS_NI_MASK
        if self.disable_pc:
            self.PC_MASK[:] = False

        self.ts_deadline_hour = self.cfg["environment"].get(
            "ts_deadline_hour", {}
        )

        power_rate = list(self.cfg["environment"]["POWER_RATE"])
        if 0.0 not in power_rate:
            power_rate.append(0.0)
        self.POWER_RATE = np.asarray(
            sorted({float(r) for r in power_rate if 0.0 <= float(r) <= 1.0}),
            dtype=float,
        )

        self.heterogeneous = True
        self.coeff_base = np.asarray(
            self.cfg["environment"]["DISSATISFACTION_COEFFICIENTS"],
            dtype=float,
        )
        self.coeff_std = np.asarray(
            self.cfg["environment"]["DISSATISFACTION_COEFFICIENTS_STD"],
            dtype=float,
        )
        self.coeff_min = np.asarray(
            self.cfg["environment"]["DISSATISFACTION_COEFFICIENTS_MIN"],
            dtype=float,
        )

        if not (
            len(self.coeff_base)
            == len(self.coeff_std)
            == len(self.coeff_min)
            == self.num_devices
        ):
            raise ValueError(
                "All dissatisfaction coefficient arrays must match DEVICES length."
            )


        sampled = self.rng.normal(
            loc=self.coeff_base,
            scale=self.coeff_std,
            size=(self.N, self.num_devices),
        )
        sampled = np.maximum(self.coeff_min, sampled).astype(float)

        configured_fixed = reward_cfg.get(
            "fixed_household_coefficients", None
        )
        if configured_fixed is None:
            self.household_coefficients = sampled
            self.household_coefficients_source = "sampled_from_run_seed"
        else:
            fixed = np.asarray(configured_fixed, dtype=float)
            expected_shape = (self.N, self.num_devices)
            if fixed.shape != expected_shape:
                raise ValueError(
                    "environment.fixed_household_coefficients must have shape "
                    f"{expected_shape}; found {fixed.shape}."
                )
            if not np.isfinite(fixed).all():
                raise ValueError(
                    "environment.fixed_household_coefficients contains "
                    "non-finite values."
                )
            if np.any(fixed < self.coeff_min[None, :] - self.EPS):
                raise ValueError(
                    "environment.fixed_household_coefficients contains a value "
                    "below DISSATISFACTION_COEFFICIENTS_MIN."
                )
            self.household_coefficients = fixed.copy()
            self.household_coefficients_source = "configured_fixed_matrix"

        self.household_coefficients_hash = hashlib.sha256(
            np.ascontiguousarray(
                self.household_coefficients, dtype=np.float64
            ).tobytes()
        ).hexdigest()
        self.dissatisfaction_coefficients = (
            self.household_coefficients.copy()
        )

        self._train_days = np.asarray(
            [
                day
                for start, end in self.train_ranges
                for day in range(int(start), int(end) + 1)
            ],
            dtype=int,
        )
        self._all_configured_days = sorted(
            set(self._train_days.tolist())
            | set(range(int(self.val_range[0]), int(self.val_range[1]) + 1))
            | set(range(int(self.test_range[0]), int(self.test_range[1]) + 1))
        )
        self._validate_configured_datasets()
        self._compute_reward_reference_scales()

    def _validate_temporal_split(self):
        """Validate config-driven day-of-year partitions without hard-coding dates."""
        if not isinstance(self.train_ranges, (list, tuple)) or not self.train_ranges:
            raise ValueError("training.train_ranges must contain at least one [start, end] pair.")

        def normalise_pair(value, name):
            if not isinstance(value, (list, tuple)) or len(value) != 2:
                raise ValueError(f"{name} must be a [start_day, end_day] pair.")
            start, end = int(value[0]), int(value[1])
            if not (1 <= start <= end <= 365):
                raise ValueError(f"Invalid {name}: [{start}, {end}].")
            return start, end

        train_pairs = [
            normalise_pair(pair, f"training.train_ranges[{idx}]")
            for idx, pair in enumerate(self.train_ranges)
        ]
        val_pair = normalise_pair(self.val_range, "training.val_range")
        test_pair = normalise_pair(self.test_range, "training.test_range")

        labelled_days = []
        for idx, (start, end) in enumerate(train_pairs):
            labelled_days.extend((day, f"train[{idx}]") for day in range(start, end + 1))
        labelled_days.extend((day, "validation") for day in range(val_pair[0], val_pair[1] + 1))
        labelled_days.extend((day, "test") for day in range(test_pair[0], test_pair[1] + 1))

        owner = {}
        for day, label in labelled_days:
            if day in owner:
                raise ValueError(
                    f"Temporal split overlap on day {day}: {owner[day]} and {label}."
                )
            owner[day] = label

        self.train_ranges = [[start, end] for start, end in train_pairs]
        self.val_range = list(val_pair)
        self.test_range = list(test_pair)

    def get_config_signature(self):
        """Return a stable hash of the active configuration for provenance."""
        payload = json.dumps(self.cfg, sort_keys=True, separators=(",", ":"), default=str)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    @staticmethod
    def _sha256_file(path):
        hasher = hashlib.sha256()
        with open(path, "rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                hasher.update(chunk)
        return hasher.hexdigest()

    def _load_decision_state_forecasts(self):
        """Load the precomputed DDQN decision-state forecast table.

        The table intentionally contains forecast columns only. Days 2--167
        must be out-of-fold/cross-fitted; later days use fixed full-model
        forecasts. Audit actuals are not permitted in this input file.
        """
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
        if not np.isfinite(self.forecast_load_floor_kw) or self.forecast_load_floor_kw < 0.0:
            raise ValueError("forecast.state_load_floor_kw must be finite and non-negative.")

        path = self.forecast_state_input_file
        if not os.path.exists(path):
            raise FileNotFoundError(f"DDQN state-forecast file not found: {path}")
        table = pd.read_csv(path, parse_dates=["timestamp"])
        if any("actual" in str(column).lower() for column in table.columns):
            raise ValueError(
                "Decision-state forecast input must not contain audit/actual columns."
            )

        load_columns = [
            f"forecast_load_house_{int(house_id)}_kw"
            for house_id in self.data_ids
        ]
        required = [
            "timestamp",
            "doy",
            "forecast_source",
            *load_columns,
            "forecast_aggregate_load_kw",
            "forecast_price_cent_per_kwh",
        ]
        missing = [column for column in required if column not in table.columns]
        if missing:
            raise ValueError(
                f"DDQN state-forecast file is missing required columns: {missing}"
            )
        if table["timestamp"].duplicated().any():
            raise ValueError("Duplicate timestamps in DDQN state-forecast file.")

        numeric = table[
            load_columns
            + ["forecast_aggregate_load_kw", "forecast_price_cent_per_kwh"]
        ].to_numpy(dtype=float)
        if not np.isfinite(numeric).all():
            raise ValueError("Non-finite values found in DDQN state forecasts.")

        aggregate_error = np.max(
            np.abs(
                table[load_columns].sum(axis=1).to_numpy(dtype=float)
                - table["forecast_aggregate_load_kw"].to_numpy(dtype=float)
            )
        )
        if aggregate_error > 1e-8:
            raise ValueError(
                "Forecast aggregate is inconsistent with the sum of household forecasts: "
                f"max error={aggregate_error:.3e}."
            )

        train_days = set(
            day
            for start, end in self.cfg["training"]["train_ranges"]
            for day in range(int(start), int(end) + 1)
        )
        for day in train_days:
            rows = table[table["doy"] == int(day)]
            if len(rows) != self.expected_hours_per_day:
                raise ValueError(
                    f"Forecast day {day} has {len(rows)} rows; expected "
                    f"{self.expected_hours_per_day}."
                )
            if set(rows["forecast_source"].astype(str)) != {"crossfit"}:
                raise ValueError(
                    f"Training day {day} must use cross-fitted forecasts only."
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
                    "Every forecast day from validation onward must use fixed "
                    f"full-model forecasts; day {day} does not."
                )

        self.forecast_load_columns = load_columns
        self.decision_state_forecasts = table.sort_values("timestamp").set_index(
            "timestamp", drop=False
        )
        self.forecast_state_input_sha256 = self._sha256_file(path)
        self.forecast_state_provenance_sha256 = None
        if os.path.exists(self.forecast_state_provenance_file):
            self.forecast_state_provenance_sha256 = self._sha256_file(
                self.forecast_state_provenance_file
            )

    def get_forecast_metadata(self):
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

    def get_state_feature_names(self):
        names = []
        groups = [
            "forecast_pre_action_house_load_norm",
            "current_pc_available_norm",
            "current_ev_available_norm",
            "current_tsni_startable_energy_norm",
            "next_hour_shifted_in_norm",
            "remaining_future_shifted_in_norm",
            "current_pending_origin_relief_norm",
            "current_pending_locked_payment_norm",
            "next_hour_pending_origin_relief_norm",
            "next_hour_pending_locked_payment_norm",
            "remaining_pending_origin_relief_norm",
            "remaining_pending_locked_payment_norm",
        ]
        for group in groups:
            names.extend(f"{group}_house_{house_id}" for house_id in self.data_ids)
        names.extend(
            [
                "log_clipped_forecast_wholesale_price",
                "normalized_hour",
                "forecast_total_pre_action_load_norm",
                "is_holiday",
                "is_weekend",
                "forecast_capacity_need_flag",
                "forecast_needed_reduction_norm",
            ]
        )
        return names


    @staticmethod
    def _optional_positive_float(value, name):
        """Parse an optional positive scalar from config."""
        if value is None:
            return None
        if isinstance(value, str) and value.strip().lower() in {"", "auto", "none", "null"}:
            return None
        parsed = float(value)
        if not np.isfinite(parsed) or parsed <= 0.0:
            raise ValueError(f"{name} must be a positive finite number or 'auto'.")
        return parsed


    def _resolve_tracking_coefficient(self, value, kind):
        """Resolve a transparent tracking coefficient from config.

        ``auto_capacity_anchored`` sets the quadratic violation coefficient to
        the capacity threshold. With the economic scale ``C * p95``, this makes
        a 1 kWh residual violation cost the same order as 1 kWh of useful relief
        valued at the 95th-percentile training price.

        ``auto_economic_anchored`` sets the linear under-target coefficient to
        one, giving the same 1/C normalized marginal scale. Numeric values are
        retained for sensitivity analysis and are labelled ``explicit``.
        """
        label = str(kind).strip().lower()
        if isinstance(value, str):
            token = value.strip().lower()
            if label == "violation" and token in {
                "auto", "auto_capacity_anchored", "capacity_anchored"
            }:
                return float(self.capacity_threshold), "auto_capacity_anchored"
            if label == "under" and token in {
                "auto", "auto_economic_anchored", "economic_anchored"
            }:
                return 1.0, "auto_economic_anchored"
        parsed = float(value)
        if not np.isfinite(parsed) or parsed < 0.0:
            raise ValueError(
                f"environment.target_tracking.track_c_{label} must be a "
                "non-negative finite number or a supported auto anchor."
            )
        return parsed, "explicit"

    def _compute_reward_reference_scales(self):
        """Compute fixed reward scales from training days only.

        Economic normalisation uses capacity multiplied by a training-price
        percentile. Discomfort normalisation uses a deterministic, train-only
        reference envelope built from actual appliance availability, the fixed
        household coefficient matrix, the maximum PC reduction rate, and the
        maximum feasible delay to each appliance deadline.
        """
        train_day_set = set(int(day) for day in self._train_days.tolist())
        train_prices = self.df_price[
            self.df_price.index.dayofyear.isin(train_day_set)
        ]["price"].to_numpy(dtype=float)
        train_prices = np.maximum(train_prices[np.isfinite(train_prices)], 0.0)
        if train_prices.size == 0:
            raise ValueError("No finite training prices are available for reward scaling.")

        self.reward_price_reference = max(
            float(np.percentile(train_prices, self.reward_price_percentile)),
            self.EPS,
        )
        auto_economic_scale = max(
            self.capacity_threshold * self.reward_price_reference,
            self.EPS,
        )
        configured_economic = self._optional_positive_float(
            self.configured_economic_scale,
            "environment.reward_normalization.economic_scale",
        )

        reference_discomfort = []
        reference_discomfort_by_class = {
            "PC_AC": [],
            "EV_TS_I": [],
            "TS_NI": [],
        }
        active_pc_loads = []
        max_pc_rate = float(np.max(self.POWER_RATE, initial=0.0))
        positive_rate_steps = np.diff(np.unique(np.sort(self.POWER_RATE)))
        positive_rate_steps = positive_rate_steps[positive_rate_steps > self.EPS]
        minimum_rate_step = (
            float(positive_rate_steps.min()) if positive_rate_steps.size else 0.0
        )

        for day in self._train_days:
            _, demands = get_device_day_demands(
                self.df_device_demands,
                self.data_ids,
                int(day),
                self.DEVICES,
                year=self.year,
                expected_hours=self.expected_hours_per_day,
            )
            for h in range(self.expected_hours_per_day):
                hourly_reference = 0.0
                hourly_by_class = {
                    "PC_AC": 0.0,
                    "EV_TS_I": 0.0,
                    "TS_NI": 0.0,
                }
                for i in range(self.N):
                    for d in range(self.num_devices):
                        demand = float(demands[h, i, d])
                        if demand <= self.EPS:
                            continue
                        coefficient = float(self.household_coefficients[i, d])
                        if self.PC_MASK[d]:
                            active_pc_loads.append(demand)
                            contribution = coefficient * (max_pc_rate ** 2)
                            hourly_reference += contribution
                            hourly_by_class["PC_AC"] += contribution
                        elif self.TS_I_MASK[d]:
                            name = str(self.DEVICES[d]).lower()
                            deadline = int(np.clip(
                                int(self.ts_deadline_hour.get(
                                    name, self.expected_hours_per_day - 1
                                )),
                                0,
                                self.expected_hours_per_day - 1,
                            ))
                            delay = max(0, deadline - h)
                            contribution = coefficient * (delay ** 2)
                            hourly_reference += contribution
                            hourly_by_class["EV_TS_I"] += contribution
                        elif self.TS_NI_MASK[d]:
                            starts_here = h == 0 or demands[h - 1, i, d] <= self.EPS
                            if starts_here:
                                name = str(self.DEVICES[d]).lower()
                                deadline = int(np.clip(
                                    int(self.ts_deadline_hour.get(
                                        name, self.expected_hours_per_day - 1
                                    )),
                                    0,
                                    self.expected_hours_per_day - 1,
                                ))
                                delay = max(0, deadline - h)
                                contribution = coefficient * (delay ** 2)
                                hourly_reference += contribution
                                hourly_by_class["TS_NI"] += contribution
                reference_discomfort.append(hourly_reference)
                for class_name, value in hourly_by_class.items():
                    reference_discomfort_by_class[class_name].append(value)

        positive_reference = np.asarray(reference_discomfort, dtype=float)
        positive_reference = positive_reference[
            np.isfinite(positive_reference) & (positive_reference > self.EPS)
        ]
        self.reward_discomfort_reference = (
            float(np.percentile(positive_reference, self.reward_discomfort_percentile))
            if positive_reference.size
            else 1.0
        )
        auto_discomfort_scale = max(self.reward_discomfort_reference, self.EPS)
        configured_discomfort = self._optional_positive_float(
            self.configured_discomfort_scale,
            "environment.reward_normalization.discomfort_scale",
        )

        if self.reward_normalization_enabled:
            self.reward_economic_scale = configured_economic or auto_economic_scale
            self.reward_discomfort_scale = configured_discomfort or auto_discomfort_scale
        else:
            self.reward_economic_scale = 1.0
            self.reward_discomfort_scale = 1.0

        self.reward_discomfort_class_diagnostics = {}
        class_p95_values = []
        for class_name, values in reference_discomfort_by_class.items():
            array = np.asarray(values, dtype=float)
            positive = array[np.isfinite(array) & (array > self.EPS)]
            if positive.size:
                p50 = float(np.percentile(positive, 50.0))
                p95 = float(np.percentile(positive, self.reward_discomfort_percentile))
                maximum = float(np.max(positive))
                class_p95_values.append(p95)
            else:
                p50 = p95 = maximum = 0.0
            self.reward_discomfort_class_diagnostics[class_name] = {
                "positive_hour_count": int(positive.size),
                "p50_reference_discomfort": p50,
                "p95_reference_discomfort": p95,
                "maximum_reference_discomfort": maximum,
                "p95_to_final_scale_ratio": float(
                    p95 / max(self.reward_discomfort_scale, self.EPS)
                ),
            }
        self.reward_discomfort_class_scales = {}
        for class_name in self.reward_discomfort_class_names:
            diagnostic = self.reward_discomfort_class_diagnostics[class_name]
            auto_scale = max(
                float(diagnostic["p95_reference_discomfort"]), self.EPS
            )
            configured_value = self.configured_discomfort_class_scales.get(
                class_name, None
            )
            if configured_value is None:
                class_scale = auto_scale
            else:
                class_scale = self._optional_positive_float(
                    configured_value,
                    f"environment.reward_normalization.discomfort_class_scales.{class_name}",
                )
            if not self.reward_normalization_enabled:
                class_scale = 1.0
            self.reward_discomfort_class_scales[class_name] = float(class_scale)
            diagnostic["class_reward_discomfort_scale"] = float(class_scale)

        positive_p95 = [value for value in class_p95_values if value > self.EPS]
        self.reward_discomfort_p95_dominance_ratio = (
            float(max(positive_p95) / min(positive_p95))
            if len(positive_p95) >= 2
            else 1.0
        )

        deadband_value = self.configured_track_deadband_kwh
        if isinstance(deadband_value, str) and deadband_value.strip().lower() == "auto":
            median_active_pc = (
                float(np.median(np.asarray(active_pc_loads, dtype=float)))
                if active_pc_loads
                else 0.0
            )
            self.track_deadband_kwh = max(
                0.0, minimum_rate_step * median_active_pc
            )
        else:
            self.track_deadband_kwh = float(deadband_value)
            if not np.isfinite(self.track_deadband_kwh) or self.track_deadband_kwh < 0.0:
                raise ValueError(
                    "environment.target_tracking.deadband_kwh must be non-negative or 'auto'."
                )

        delta_value = self.configured_huber_delta
        if isinstance(delta_value, str):
            token = delta_value.strip().lower()
            if token != "auto_deadband_normalized":
                raise ValueError(
                    "environment.symmetric_tracking.huber_delta must be a "
                    "positive number or 'auto_deadband_normalized'."
                )
            self.symmetric_tracking_delta = max(
                self.track_deadband_kwh / max(self.capacity_threshold, self.EPS),
                self.EPS,
            )
            self.symmetric_tracking_delta_source = "auto_deadband_normalized"
        else:
            self.symmetric_tracking_delta = float(delta_value)
            if (
                not np.isfinite(self.symmetric_tracking_delta)
                or self.symmetric_tracking_delta <= 0.0
            ):
                raise ValueError(
                    "environment.symmetric_tracking.huber_delta must be positive."
                )
            self.symmetric_tracking_delta_source = "configured_numeric"

    def get_discomfort_scale_diagnostics(self):
        """Return train-only per-class diagnostics for the discomfort scale."""
        rows = []
        for class_name, values in self.reward_discomfort_class_diagnostics.items():
            rows.append({
                "device_class": class_name,
                **values,
                "final_reward_discomfort_scale": float(self.reward_discomfort_scale),
                "combined_reference_p95": float(self.reward_discomfort_reference),
                "p95_dominance_ratio": float(
                    self.reward_discomfort_p95_dominance_ratio
                ),
                "discomfort_normalization_mode": str(
                    self.discomfort_normalization_mode
                ),
                "class_reward_discomfort_scale": float(
                    self.reward_discomfort_class_scales[class_name]
                ),
            })
        return rows

    def set_household_coefficients(self, values):
        """Set the persistent household preferences and refresh reward scales."""
        values = np.asarray(values, dtype=float)
        expected_shape = (self.N, self.num_devices)
        if values.shape != expected_shape:
            raise ValueError(
                f"household_coefficients must have shape {expected_shape}; found {values.shape}."
            )
        if not np.isfinite(values).all():
            raise ValueError("household_coefficients contain non-finite values.")
        self.household_coefficients = values.copy()
        self.dissatisfaction_coefficients = values.copy()
        self.household_coefficients_source = "set_explicitly"
        self.household_coefficients_hash = hashlib.sha256(
            np.ascontiguousarray(
                self.household_coefficients, dtype=np.float64
            ).tobytes()
        ).hexdigest()
        self._compute_reward_reference_scales()

    def get_reward_metadata(self):
        """Return immutable reward provenance for checkpoints and tests."""
        return {
            "reward_definition_version": self.REWARD_DEFINITION_VERSION,
            "reward_mode": str(self.reward_mode),
            "c_cap": float(self.c_cap),
            "c_over": float(self.c_over),
            "reward_normalization_enabled": bool(self.reward_normalization_enabled),
            "reward_price_percentile": float(self.reward_price_percentile),
            "reward_discomfort_percentile": float(self.reward_discomfort_percentile),
            "reward_price_reference": float(self.reward_price_reference),
            "reward_discomfort_reference": float(self.reward_discomfort_reference),
            "reward_economic_scale": float(self.reward_economic_scale),
            "reward_discomfort_scale": float(self.reward_discomfort_scale),
            "discomfort_normalization_mode": str(
                self.discomfort_normalization_mode
            ),
            "reward_discomfort_class_scales": dict(
                self.reward_discomfort_class_scales
            ),
            "discomfort_weight": float(self.discomfort_weight),
            "offer_regularization_enabled": bool(
                self.offer_regularization_enabled
            ),
            "offer_regularization_fraction": float(
                self.offer_regularization_fraction
            ),
            "incentive_offer_coefficient": float(
                self.incentive_offer_coefficient
            ),
            "incentive_offer_coefficient_source": str(
                self.incentive_offer_coefficient_source
            ),
            "track_c_violation": float(self.track_c_violation),
            "track_c_violation_source": str(self.track_c_violation_source),
            "track_c_under": float(self.track_c_under),
            "track_c_under_source": str(self.track_c_under_source),
            "track_deadband_kwh": float(self.track_deadband_kwh),
            "stakeholder_sp_weight": float(self.stakeholder_sp_weight),
            "stakeholder_eu_weight": float(self.stakeholder_eu_weight),
            "effective_payment_cost_coefficient": float(
                self.effective_payment_cost_coefficient
            ),
            "effective_payment_cost_fraction": float(
                self.effective_payment_cost_fraction
            ),
            "minimum_effective_payment_cost_fraction": float(
                self.minimum_effective_payment_cost_fraction
            ),
            "symmetric_tracking_lambda": float(self.symmetric_tracking_lambda),
            "symmetric_tracking_lambda_source": str(
                self.symmetric_tracking_lambda_source
            ),
            "symmetric_tracking_loss": str(self.symmetric_tracking_loss_name),
            "symmetric_tracking_delta": float(self.symmetric_tracking_delta),
            "symmetric_tracking_delta_source": str(
                self.symmetric_tracking_delta_source
            ),
            "no_need_offer_regularization_enabled": bool(
                self.no_need_offer_regularization_enabled
            ),
            "no_need_offer_coefficient": float(
                self.no_need_offer_coefficient
            ),
            "no_need_gate_tolerance_kw": float(
                self.no_need_gate_tolerance_kw
            ),
            "no_need_gate_definition": (
                "pre_action_committed_load <= capacity + numerical_tolerance"
            ),
            "no_need_offer_cost_definition": (
                "coefficient * no_need_gate * mean(normalized_raw_action)"
            ),
            "reward_discomfort_p95_dominance_ratio": float(
                self.reward_discomfort_p95_dominance_ratio
            ),
            "household_coefficients_source": str(
                self.household_coefficients_source
            ),
            "household_coefficients_hash": str(
                self.household_coefficients_hash
            ),
            "useful_relief_definition": (
                "legacy modes: min(settlement, max(0, baseline-capacity)); "
                "weighted stakeholder modes: min(settlement, max(0, pre_action-capacity))"
            ),
            "training_payment_definition": (
                "full verified settlement payment booked at the current verification hour"
            ),
            "tracking_target_definition": "min(pre_action_committed_load, capacity)",
            "stakeholder_reward_definition": (
                "omega*(price*preaction_useful_settlement-payment)/S_E + "
                "(1-omega)*(rho*payment/S_E-(1-rho)*D_norm)"
            ),
        }

    def _validate_configured_datasets(self):
        """Fail fast if any configured train/validation/test day is incomplete."""
        if self.expected_hours_per_day != 24:
            raise ValueError(
                "This study uses complete 24-hour days; "
                f"general.expected_hours_per_day={self.expected_hours_per_day}."
            )
        if self.time_steps_train != self.expected_hours_per_day:
            raise ValueError(
                "training.time_steps_train must equal the complete-day length "
                f"({self.expected_hours_per_day}); found {self.time_steps_train}."
            )
        if self.time_steps_test != self.expected_hours_per_day:
            raise ValueError(
                "training.time_steps_test must equal the complete-day length "
                f"({self.expected_hours_per_day}); found {self.time_steps_test}."
            )

        demand = self.df_demand.copy()
        devices = self.df_device_demands.copy()
        requested_ids = [int(value) for value in self.data_ids]

        for day in self._all_configured_days:
            start = np.datetime64(
                f"{self.year}-01-01T00:00:00"
            ) + np.timedelta64(int(day) - 1, "D")
            expected_index = np.asarray(
                [start + np.timedelta64(hour, "h") for hour in range(self.expected_hours_per_day)]
            )

            price_day = self.df_price[self.df_price.index.dayofyear == int(day)]
            if len(price_day) != self.expected_hours_per_day:
                raise ValueError(
                    f"Price data for day {day} contain {len(price_day)} rows; "
                    f"exactly {self.expected_hours_per_day} are required."
                )
            actual_price_index = price_day.sort_index().index.to_numpy(dtype="datetime64[ns]")
            if not np.array_equal(actual_price_index, expected_index.astype("datetime64[ns]")):
                raise ValueError(f"Price timestamps are incomplete or misaligned on day {day}.")

            demand_day = demand[demand["dt"].dt.dayofyear == int(day)]
            device_day = devices[devices["dt"].dt.dayofyear == int(day)]
            expected_rows = self.expected_hours_per_day * self.N
            if len(demand_day) != expected_rows:
                counts = demand_day.groupby("dataid").size().to_dict()
                raise ValueError(
                    f"Household totals for day {day} contain {len(demand_day)} rows; "
                    f"expected {expected_rows}. Counts: {counts}"
                )
            if len(device_day) != expected_rows:
                counts = device_day.groupby("dataid").size().to_dict()
                raise ValueError(
                    f"Appliance data for day {day} contain {len(device_day)} rows; "
                    f"expected {expected_rows}. Counts: {counts}"
                )

            for house_id in requested_ids:
                house_total = demand_day[demand_day["dataid"] == house_id].sort_values("dt")
                house_device = device_day[device_day["dataid"] == house_id].sort_values("dt")
                if len(house_total) != self.expected_hours_per_day:
                    raise ValueError(
                        f"House {house_id}, day {day}: expected "
                        f"{self.expected_hours_per_day} total-load rows; found {len(house_total)}."
                    )
                if len(house_device) != self.expected_hours_per_day:
                    raise ValueError(
                        f"House {house_id}, day {day}: expected "
                        f"{self.expected_hours_per_day} appliance rows; found {len(house_device)}."
                    )
                total_index = house_total["dt"].to_numpy(dtype="datetime64[ns]")
                device_index = house_device["dt"].to_numpy(dtype="datetime64[ns]")
                expected_ns = expected_index.astype("datetime64[ns]")
                if not np.array_equal(total_index, expected_ns):
                    raise ValueError(
                        f"Household-total timestamps are incomplete or misaligned for "
                        f"house {house_id}, day {day}."
                    )
                if not np.array_equal(device_index, expected_ns):
                    raise ValueError(
                        f"Appliance timestamps are incomplete or misaligned for "
                        f"house {house_id}, day {day}."
                    )

                total_values = house_total["total"].to_numpy(dtype=float)
                device_values = house_device[self.DEVICES].to_numpy(dtype=float)
                inconsistency = device_values.sum(axis=1) - total_values
                if np.any(inconsistency > self.data_tolerance):
                    first = int(np.flatnonzero(inconsistency > self.data_tolerance)[0])
                    raise ValueError(
                        f"Sum of appliance loads exceeds total household load for house "
                        f"{house_id}, day {day}, hour {first}: excess="
                        f"{inconsistency[first]:.8f}."
                    )


    def reset(self, day=None, mode="train"):
        if day is None:
            if mode == "train":

                self.day = int(self.rng.choice(self._train_days))
            elif mode == "val":
                self.day = int(
                    self.rng.integers(
                        int(self.val_range[0]), int(self.val_range[1]) + 1
                    )
                )
            elif mode == "test":
                self.day = int(
                    self.rng.integers(
                        int(self.test_range[0]), int(self.test_range[1]) + 1
                    )
                )
            else:
                self.day = int(self.train_ranges[0][0])
        else:
            self.day = int(day)

        self.curr_step = 0
        self.episode += 1
        self.done = False
        self.mode = mode

        configured_steps = (
            self.time_steps_test if mode in {"val", "test"} else self.time_steps_train
        )
        self.max_steps = int(configured_steps)
        self.T = self.expected_hours_per_day

        self.price_df = self.df_price[
            self.df_price.index.dayofyear == self.day
        ].copy().sort_index()
        if len(self.price_df) != self.expected_hours_per_day:
            raise ValueError(
                f"Day {self.day} contains {len(self.price_df)} price rows; "
                f"exactly {self.expected_hours_per_day} are required."
            )
        expected_index = pd.date_range(
            pd.to_datetime(f"{self.year}-{self.day}", format="%Y-%j"),
            periods=self.expected_hours_per_day,
            freq="h",
        )
        if not self.price_df.index.equals(expected_index):
            raise ValueError(f"Price timestamps are misaligned on day {self.day}.")

        self.prices = self.price_df["price"].to_numpy(dtype=float)
        self.hours = self.price_df.index.hour.to_numpy(dtype=int)
        if not np.isfinite(self.prices).all():
            raise ValueError(f"Non-finite prices were found on day {self.day}.")

        ts0 = self.price_df.index[0]
        self.is_weekend = 1.0 if ts0.weekday() >= 5 else 0.0
        self.is_holiday = 1.0 if ts0.date() in self.us_holidays else 0.0

        forecast_day = self.decision_state_forecasts.reindex(expected_index)
        if forecast_day[self.forecast_load_columns + ["forecast_price_cent_per_kwh"]].isna().any().any():
            raise ValueError(
                f"Missing DDQN state forecasts on day {self.day}; exact hourly "
                "coverage is required."
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
                f"Mixed forecast sources within day {self.day}: "
                f"{sorted(set(self.state_forecast_source))}."
            )

        day_mask = self.all_baselines["timestamp"].dt.dayofyear == self.day
        baselines_day = self.all_baselines[day_mask].copy()
        if baselines_day.duplicated(subset=["timestamp", "house_id"]).any():
            raise ValueError(f"Duplicate baseline rows were found on day {self.day}.")
        baseline_wide = baselines_day.set_index(
            ["timestamp", "house_id"]
        )["baseline_demand"].unstack("house_id")
        baseline_wide = baseline_wide.reindex(
            index=expected_index, columns=self.data_ids
        )
        if baseline_wide.isna().any().any():
            missing = np.argwhere(baseline_wide.isna().to_numpy())[:10]
            raise ValueError(
                f"Missing household totals on day {self.day}; no forward-fill or "
                f"zero-fill is allowed. Missing positions: {missing.tolist()}"
            )

        self.baseline_per_house = baseline_wide.to_numpy(dtype=float)
        self.total_baseline = self.baseline_per_house.sum(axis=1)

        # Causal future-capacity reference used only by the HEMS scheduler.
        # For each wall-clock hour, use the mean aggregate household baseline
        # over up to the preceding seven days.  Only observations strictly
        # before the current episode are eligible.  Known shift commitments
        # are applied separately when candidate destinations are screened.
        if self.day <= 1:
            raise ValueError(
                "Causal future-capacity screening requires historical observations."
            )
        history_start = expected_index[0] - pd.Timedelta(days=7)
        history = self.all_baselines[
            (self.all_baselines["timestamp"] < expected_index[0])
            & (self.all_baselines["timestamp"] >= history_start)
        ].copy()
        history["hour"] = history["timestamp"].dt.hour
        mean_by_hour = history.groupby(
            ["hour", "house_id"], sort=False
        )["baseline_demand"].mean().unstack("house_id")
        mean_by_hour = mean_by_hour.reindex(
            index=np.arange(self.expected_hours_per_day), columns=self.data_ids
        )

        # For early source hours with no same-hour history (only possible near
        # the start of the dataset), fall back to the latest genuinely
        # available observation before the episode begins.
        if mean_by_hour.isna().any().any():
            prior_rows = self.all_baselines[
                self.all_baselines["timestamp"] < expected_index[0]
            ].sort_values("timestamp")
            for house_id in self.data_ids:
                series = prior_rows[
                    prior_rows["house_id"] == house_id
                ]["baseline_demand"]
                if series.empty:
                    raise ValueError(
                        f"No historical household load is available before day {self.day} "
                        f"for household {house_id}."
                    )
                mean_by_hour[house_id] = mean_by_hour[house_id].fillna(
                    float(series.iloc[-1])
                )
        reference = mean_by_hour.to_numpy(dtype=float)
        self.future_capacity_reference_per_house = reference

        device_index, self.device_demands = get_device_day_demands(
            self.df_device_demands,
            self.data_ids,
            self.day,
            self.DEVICES,
            year=self.year,
            expected_hours=self.expected_hours_per_day,
        )
        if not device_index.equals(expected_index):
            raise ValueError(f"Appliance timestamps are misaligned on day {self.day}.")
        if not np.isfinite(self.device_demands).all():
            raise ValueError(f"Non-finite appliance demand was found on day {self.day}.")
        if np.any(self.device_demands < -self.EPS):
            raise ValueError(f"Negative appliance demand was found on day {self.day}.")

        self.ts_carry_over = np.zeros((self.N, self.num_devices), dtype=bool)
        if self.day > 1:
            prev_last = get_device_demands(
                self.df_device_demands,
                self.data_ids,
                self.day - 1,
                self.expected_hours_per_day - 1,
                self.DEVICES,
                year=self.year,
                allow_missing=True,
            )
            self.ts_carry_over = (
                (prev_last > self.EPS) & self.TS_NI_MASK[None, :]
            )

        self.device_before = self.device_demands.copy()
        raw_ctrl_sum = self.device_demands.sum(axis=2)
        raw_nonshift = self.baseline_per_house - raw_ctrl_sum
        minimum_nonshift = float(raw_nonshift.min(initial=0.0))
        if minimum_nonshift < -self.data_tolerance:
            location = np.unravel_index(np.argmin(raw_nonshift), raw_nonshift.shape)
            raise ValueError(
                f"Appliance demand exceeds household total on day {self.day}, "
                f"hour {location[0]}, house {self.data_ids[location[1]]}: "
                f"difference={raw_nonshift[location]:.8f}."
            )
        self.data_consistency_error = np.minimum(raw_nonshift, 0.0)
        self.nonshift_baseline_per_house = np.maximum(raw_nonshift, 0.0)

        self.lam_min, self.lam_max = self.get_lam_bounds()


        self.incentives = np.zeros((self.T, self.N), dtype=float)
        self.rewards_customers = np.zeros((self.T, self.N), dtype=float)
        self.rewards_service_provider = np.zeros(self.T, dtype=float)
        self.rewards_total = np.zeros(self.T, dtype=float)
        self.reward_shaping_component = np.zeros(self.T, dtype=float)
        self.reward_base_component = np.zeros(self.T, dtype=float)
        self.reward_tracking_component = np.zeros(self.T, dtype=float)
        self.reward_no_need_component = np.zeros(self.T, dtype=float)
        self.corrected_economic_reward_component = np.zeros(self.T, dtype=float)
        self.normalized_economic_component = np.zeros(self.T, dtype=float)
        self.normalized_discomfort_component = np.zeros(self.T, dtype=float)
        self.raw_discomfort_by_class = np.zeros((self.T, 3), dtype=float)
        self.normalized_discomfort_by_class = np.zeros((self.T, 3), dtype=float)
        self.incentive_offer_intensity = np.zeros(self.T, dtype=float)
        self.incentive_offer_penalty = np.zeros(self.T, dtype=float)
        self.no_need_gate_flag = np.zeros(self.T, dtype=bool)
        self.no_need_offer_penalty = np.zeros(self.T, dtype=float)
        self.combined_offer_penalty = np.zeros(self.T, dtype=float)
        self.useful_settlement_dr_energy = np.zeros(self.T, dtype=float)
        self.excess_settlement_dr_energy = np.zeros(self.T, dtype=float)
        self.baseline_capacity_need = np.zeros(self.T, dtype=float)
        self.pre_action_capacity_need = np.zeros(self.T, dtype=float)
        self.baseline_useful_settlement_dr_energy = np.zeros(self.T, dtype=float)
        self.pre_action_useful_settlement_dr_energy = np.zeros(self.T, dtype=float)
        self.stakeholder_sp_normalized_component = np.zeros(self.T, dtype=float)
        self.stakeholder_eu_normalized_component = np.zeros(self.T, dtype=float)
        self.stakeholder_reward_component = np.zeros(self.T, dtype=float)
        self.signed_tracking_error_normalized = np.zeros(self.T, dtype=float)
        self.symmetric_tracking_loss = np.zeros(self.T, dtype=float)
        self.symmetric_tracking_penalty = np.zeros(self.T, dtype=float)
        self.control_target_load = np.zeros(self.T, dtype=float)
        self.control_over_error = np.zeros(self.T, dtype=float)
        self.control_under_error = np.zeros(self.T, dtype=float)
        self.target_tracking_penalty = np.zeros(self.T, dtype=float)
        self.unnecessary_incentive_flag = np.zeros(self.T, dtype=bool)
        self.positive_incentive_no_response_flag = np.zeros(self.T, dtype=bool)
        self.maximum_action_flag = np.zeros(self.T, dtype=bool)
        self.selected_action_index = np.full(self.T, -1, dtype=int)
        self.selected_raw_action = np.zeros((self.T, self.N), dtype=float)


        self.hourly_capacity_relief = np.zeros((self.T, self.N), dtype=float)


        self.signed_load_change = np.zeros((self.T, self.N), dtype=float)

        self.discomforts = np.zeros((self.T, self.N), dtype=float)
        self.device_discomfort = np.zeros(
            (self.T, self.N, self.num_devices), dtype=float
        )


        self.new_shifted_out_at_action = np.zeros(
            (self.T, self.N, self.num_devices), dtype=float
        )
        self.action_attributed_dr_energy = np.zeros((self.T, self.N), dtype=float)


        self.contracted_dr_energy = self.action_attributed_dr_energy


        self.settlement_dr_energy = np.zeros((self.T, self.N), dtype=float)
        self.settlement_effective_rate = np.zeros((self.T, self.N), dtype=float)
        self.incentive_payment = np.zeros((self.T, self.N), dtype=float)
        self.sp_wholesale_value = np.zeros(self.T, dtype=float)
        self.sp_incentive_cost = np.zeros(self.T, dtype=float)
        self.net_wholesale_procurement_impact = np.zeros(self.T, dtype=float)
        self.customer_weighted_incentive_benefit = np.zeros(
            (self.T, self.N), dtype=float
        )


        shape = (self.T, self.N, self.num_devices)
        self.device_shifted_out = np.zeros(shape, dtype=float)
        self.device_shifted_in = np.zeros(shape, dtype=float)
        self.device_future_inj = np.zeros(shape, dtype=float)
        # Physical feasibility deferrals caused by an already committed
        # shifted-in TS-NI cycle occupying the appliance when a new original
        # programme is requested. These deferrals are not incentive-settled.
        self.device_conflict_deferred_out = np.zeros(shape, dtype=float)
        self.device_conflict_deferred_in = np.zeros(shape, dtype=float)
        self.device_conflict_future_inj = np.zeros(shape, dtype=float)
        self.device_conflict_unserved = np.zeros(shape, dtype=float)
        self.device_conflict_discomfort = np.zeros(shape, dtype=float)
        self.device_curtailed = np.zeros(shape, dtype=float)


        self.shifted_out_locked_rate = np.zeros(shape, dtype=float)
        self.shifted_out_decision_hour = np.full(shape, -1, dtype=int)
        self.shifted_out_has_ledger = np.zeros(shape, dtype=bool)


        self.device_raw_settlement_relief = np.zeros(shape, dtype=float)
        self.device_verified_settlement_energy = np.zeros(shape, dtype=float)
        self.device_settlement_payment = np.zeros(shape, dtype=float)
        self.device_settlement_rate = np.zeros(shape, dtype=float)
        self.device_settlement_decision_hour = np.full(shape, -1, dtype=int)
        # Locked incentive rates are set at the original decision hour. If the
        # realised wholesale price is lower at a later verification hour, the
        # settlement remains valid and the SP absorbs the resulting loss.
        self.device_settlement_rate_above_price = np.zeros(shape, dtype=bool)
        self.device_settlement_above_price_energy = np.zeros(shape, dtype=float)
        self.device_settlement_negative_margin = np.zeros(shape, dtype=float)
        self.settlement_allocation_factor = np.zeros((self.T, self.N), dtype=float)


        self.device_unmet = np.zeros(shape, dtype=float)


        self.device_unshifted_request = np.zeros(shape, dtype=float)


        self.device_settlement_ineligible_request = np.zeros(shape, dtype=float)

        self.device_after = np.zeros(shape, dtype=float)
        self._removed_mask = np.zeros(shape, dtype=bool)

        self.pre_action_device = np.zeros(shape, dtype=float)
        self.pre_action_total_per_house = np.zeros(
            (self.T, self.N), dtype=float
        )
        self.before_total_per_house = np.zeros(
            (self.T, self.N), dtype=float
        )
        self.after_total_per_house = np.zeros(
            (self.T, self.N), dtype=float
        )


        self.net_change_vs_baseline = self.signed_load_change
        self.capacity_overrun = np.zeros(self.T, dtype=float)
        self.capacity_violation = np.zeros(self.T, dtype=bool)
        self.needed_reduction = np.zeros(self.T, dtype=float)


        self.net_curtailment_per_house = np.zeros(
            (self.T, self.N), dtype=float
        )
        self.shifted_out_per_house = np.zeros(
            (self.T, self.N), dtype=float
        )
        self.shifted_in_per_house = np.zeros(
            (self.T, self.N), dtype=float
        )


        self.tsni_house_order = np.full(
            (self.T, self.num_devices, self.N), -1, dtype=int
        )
        self.ev_house_order = np.full(
            (self.T, self.N), -1, dtype=int
        )


        self.device_after_agg = np.zeros(
            (self.T, self.num_devices), dtype=float
        )
        self.device_baseline_raw = self.device_before.sum(axis=1)
        self.ts_exec_agg = np.zeros(
            (self.T, self.num_devices), dtype=float
        )
        self.reductions_by_device = np.zeros(
            (self.T, self.num_devices), dtype=float
        )
        self.planned_load = np.zeros(self.T, dtype=float)
        self.ts_backlog = np.zeros(
            (self.N, self.num_devices), dtype=float
        )
        self.ts_due_time = np.full(
            (self.N, self.num_devices), -1, dtype=int
        )

        self.dissatisfaction_coefficients = (
            self.household_coefficients.copy()
            if self.heterogeneous
            else np.tile(self.coeff_base, (self.N, 1)).astype(float)
        )

        self._refresh_planned_load()
        state = self.get_state()
        self.state_dim = state.shape[0]
        return state

    def _current_pre_action_device(self, h):
        """Committed controllable load before the current action."""
        original_remaining = np.maximum(
            self.device_demands[h]
            - self.device_shifted_out[h]
            - self.device_conflict_deferred_out[h],
            0.0,
        )
        return (
            original_remaining
            + self.device_future_inj[h]
            + self.device_conflict_future_inj[h]
        )

    def _current_pre_action_house_load(self, h):
        return (
            self._current_pre_action_device(h).sum(axis=1)
            + self.nonshift_baseline_per_house[h]
        )

    def _decision_state_pre_action_house_load(self, h):
        """Pre-action household load exposed to the DDQN decision state.

        Production mode uses the one-step-ahead household forecast for the
        current decision hour, adjusted only by commitments that were already
        created before the current action. ``oracle_actual`` is a diagnostic
        mode used solely to verify exact mapping against the pre-integration
        environment.
        """
        if self.forecast_state_mode == "oracle_actual":
            return self._current_pre_action_house_load(h)

        forecast_baseline = np.maximum(
            self.state_forecast_load_per_house[h],
            self.forecast_load_floor_kw,
        )
        known_origin_relief = (
            self.device_shifted_out[h].sum(axis=1)
            + self.device_conflict_deferred_out[h].sum(axis=1)
        )
        known_shifted_in = (
            self.device_future_inj[h].sum(axis=1)
            + self.device_conflict_future_inj[h].sum(axis=1)
        )
        return np.maximum(
            forecast_baseline - known_origin_relief + known_shifted_in,
            0.0,
        )

    def _decision_state_price(self, h):
        if self.forecast_state_mode == "oracle_actual":
            return float(self.prices[h])
        return float(self.state_forecast_price[h])

    def _committed_total_at(self, t):
        """Aggregate realised/current-day physical load at hour ``t``.

        This quantity is retained for physical accounting and diagnostics.
        Future scheduling decisions must use ``_causal_capacity_estimate_at``
        instead.
        """
        if t < 0 or t >= self.T:
            return 0.0
        return float(
            self.nonshift_baseline_per_house[t].sum()
            + self._current_pre_action_device(t).sum()
        )

    def _causal_capacity_estimate_at(self, t):
        """Return a causal estimate of aggregate load at a future hour.

        The baseline reference is the mean of the same wall-clock hour over
        up to the preceding seven days. Only commitments created by decisions
        already made in the current episode are then applied: known origin
        relief is subtracted and known shifted-in load is added. No realised
        current-day baseline or appliance observation after the decision hour
        is consulted.
        """
        if t < 0 or t >= self.T:
            return 0.0
        baseline_reference = float(self.future_capacity_reference_per_house[t].sum())
        known_origin_relief = float(
            self.device_shifted_out[t].sum()
            + self.device_conflict_deferred_out[t].sum()
        )
        known_shifted_in = float(
            self.device_future_inj[t].sum()
            + self.device_conflict_future_inj[t].sum()
        )
        return max(0.0, baseline_reference - known_origin_relief + known_shifted_in)

    def _refresh_planned_load(self):
        self.planned_load = np.asarray(
            [self._committed_total_at(t) for t in range(self.T)],
            dtype=float,
        )

    def _current_pc_available_per_house(self, h):
        if not np.any(self.PC_MASK):
            return np.zeros(self.N, dtype=float)
        return self._current_pre_action_device(h)[:, self.PC_MASK].sum(axis=1)

    def _current_ev_available_per_house(self, h):
        values = np.zeros(self.N, dtype=float)
        if self.idx_car is None:
            return values
        values[:] = np.maximum(
            0.0,
            self.device_demands[h, :, self.idx_car]
            - self.device_shifted_out[h, :, self.idx_car]
            - self.device_conflict_deferred_out[h, :, self.idx_car],
        )
        return values

    def _current_tsni_startable_energy_per_house(self, h):
        """Return TS-NI demand whose operating block starts at hour ``h``.

        This state feature is deliberately causal: it uses only the current
        hour, the immediately preceding observed hour, and commitments already
        created by earlier decisions.  It does not inspect the realised
        appliance profile after ``h``.  The complete non-interruptible block
        is handled by the HEMS scheduling logic; this state component only
        indicates demand that is currently available to start/reschedule.
        """
        values = np.zeros(self.N, dtype=float)
        for d in range(self.num_devices):
            if not self.TS_NI_MASK[d]:
                continue
            for i in range(self.N):
                current = float(self.device_demands[h, i, d])
                if current <= self.EPS:
                    continue
                if (
                    float(self.device_shifted_out[h, i, d]) > self.EPS
                    or float(self.device_conflict_deferred_out[h, i, d]) > self.EPS
                ):
                    continue

                if h == 0:
                    started_now = not bool(self.ts_carry_over[i, d])
                else:
                    started_now = (
                        float(self.device_demands[h - 1, i, d]) <= self.EPS
                    )

                if started_now:
                    values[i] += current
        return values

    def _shifted_in_commitments_per_house(self, h):
        if h + 1 >= self.T:
            zeros = np.zeros(self.N, dtype=float)
            return zeros, zeros.copy()
        future_known = self.device_future_inj + self.device_conflict_future_inj
        next_hour = future_known[h + 1].sum(axis=1)
        remaining = future_known[h + 1 :].sum(axis=(0, 2))
        return next_hour, remaining

    def _current_settlement_commitments_per_house(self, h):
        """Return current-hour relief/payment committed by earlier decisions.

        Multi-hour TS-NI/EV decisions may settle source-hour slices after the
        original decision hour. These quantities are known before the current
        action and are therefore exposed in the state to avoid aliasing states
        that have identical physical loads but different current settlement
        obligations. Newly created commitments from the current action are not
        present until after ``step`` and therefore cannot leak into the state.
        """
        if h < 0 or h >= self.T:
            zeros = np.zeros(self.N, dtype=float)
            return zeros, zeros.copy()
        mask = self.shifted_out_has_ledger[h]
        energy = self.device_shifted_out[h] * mask
        payment = energy * self.shifted_out_locked_rate[h]
        return energy.sum(axis=1), payment.sum(axis=1)

    def _pending_settlement_commitments_per_house(self, h):
        """Return future origin-relief and locked-payment commitments.

        These commitments are consequences of earlier TS-NI decisions. Exposing
        both their energy and locked-payment mass prevents two physically
        identical states with different future settlement obligations from
        appearing identical to the DDQN agent.
        """
        zeros = np.zeros(self.N, dtype=float)
        if h + 1 >= self.T:
            return zeros, zeros.copy(), zeros.copy(), zeros.copy()

        future_energy = self.device_shifted_out[h + 1 :]
        future_rates = self.shifted_out_locked_rate[h + 1 :]
        future_mask = self.shifted_out_has_ledger[h + 1 :]
        future_payment = future_energy * future_rates * future_mask

        next_energy = future_energy[0].sum(axis=1)
        next_payment = future_payment[0].sum(axis=1)
        remaining_energy = future_energy.sum(axis=(0, 2))
        remaining_payment = future_payment.sum(axis=(0, 2))
        return next_energy, next_payment, remaining_energy, remaining_payment

    def get_state(self):
        """Return the 12N+7 decision state using one-step forecasts where specified."""
        h = self.curr_step
        price = self._decision_state_price(h)
        if self.capacity_threshold <= self.EPS:
            raise ValueError("capacity_threshold must be strictly positive.")

        committed_house = self._decision_state_pre_action_house_load(h)
        pc_available = self._current_pc_available_per_house(h)
        ev_available = self._current_ev_available_per_house(h)
        tsni_startable = self._current_tsni_startable_energy_per_house(h)
        next_shifted_in, remaining_shifted_in = (
            self._shifted_in_commitments_per_house(h)
        )
        (
            current_pending_relief,
            current_pending_payment,
        ) = self._current_settlement_commitments_per_house(h)
        (
            next_pending_relief,
            next_pending_payment,
            remaining_pending_relief,
            remaining_pending_payment,
        ) = self._pending_settlement_commitments_per_house(h)
        total_committed = float(committed_house.sum())

        upper_clip = 80.0
        price_for_state = np.log1p(np.clip(price, 0.0, upper_clip))
        needed = max(0.0, total_committed - self.capacity_threshold)
        need_flag = 1.0 if needed > self.EPS else 0.0
        time_den = max(self.max_steps - 1, 1)
        cap = self.capacity_threshold


        payment_scale = max(cap * upper_clip, self.EPS)

        inputs = [
            committed_house / cap,
            pc_available / cap,
            ev_available / cap,
            tsni_startable / cap,
            next_shifted_in / cap,
            remaining_shifted_in / cap,
            current_pending_relief / cap,
            current_pending_payment / payment_scale,
            next_pending_relief / cap,
            next_pending_payment / payment_scale,
            remaining_pending_relief / cap,
            remaining_pending_payment / payment_scale,
            [price_for_state],
            [h / time_den],
            [total_committed / cap],
            [self.is_holiday],
            [self.is_weekend],
            [need_flag],
            [needed / cap],
        ]
        state = np.concatenate([np.atleast_1d(value) for value in inputs]).astype(
            np.float32
        )
        expected_dim = 12 * self.N + 7
        if state.shape != (expected_dim,):
            raise RuntimeError(
                f"State construction error: expected {expected_dim} values, "
                f"found {state.shape}."
            )
        return state


    def step(self, action):
        h = self.curr_step
        if not 0 <= int(action) < self.num_actions:
            raise IndexError(
                f"Action index {action} outside [0, {self.num_actions - 1}]."
            )

        raw_action = self.all_actions[int(action)]
        self.selected_action_index[h] = int(action)
        self.selected_raw_action[h] = np.asarray(raw_action, dtype=float)
        incentives = self.apply_incentives(raw_action)

        self.apply_demand_response(incentives, h)

        self.compute_service_provider_reward()
        self.compute_customers_reward()
        reward = self.compute_total_reward()

        self.curr_step += 1
        done = self.curr_step >= self.max_steps
        self.done = done

        if not done:
            self._resolve_physical_conflicts_at_hour(self.curr_step)

        observation = (
            np.zeros(self.state_dim, dtype=np.float32)
            if done
            else self.get_state()
        )
        info = {
            "day": self.day,
            "hour": h,
            "capacity_overrun": float(self.capacity_overrun[h]),
            "realised_total_load": float(
                self.after_total_per_house[h].sum()
            ),
            "needed_reduction": float(self.needed_reduction[h]),
            "hourly_capacity_relief": float(
                self.hourly_capacity_relief[h].sum()
            ),
            "signed_load_change": float(self.signed_load_change[h].sum()),
            "settlement_dr_energy": float(self.settlement_dr_energy[h].sum()),
            "action_attributed_dr_energy": float(
                self.action_attributed_dr_energy[h].sum()
            ),
            "incentive_payment": float(self.incentive_payment[h].sum()),
            "sp_settlement_wholesale_value": float(self.sp_wholesale_value[h]),
            "sp_settlement_value": float(self.rewards_service_provider[h]),
            "net_wholesale_procurement_impact": float(
                self.net_wholesale_procurement_impact[h]
            ),
            "net_curtailment": float(
                self.net_curtailment_per_house[h].sum()
            ),
            "unshifted_ev_request": float(
                self.device_unshifted_request[h].sum()
            ),
            "baseline_capacity_need": float(self.baseline_capacity_need[h]),
            "pre_action_capacity_need": float(self.pre_action_capacity_need[h]),
            "stakeholder_sp_normalized_component": float(
                self.stakeholder_sp_normalized_component[h]
            ),
            "stakeholder_eu_normalized_component": float(
                self.stakeholder_eu_normalized_component[h]
            ),
            "stakeholder_reward_component": float(
                self.stakeholder_reward_component[h]
            ),
            "signed_tracking_error_normalized": float(
                self.signed_tracking_error_normalized[h]
            ),
            "symmetric_tracking_loss": float(self.symmetric_tracking_loss[h]),
            "symmetric_tracking_penalty": float(
                self.symmetric_tracking_penalty[h]
            ),
            "useful_settlement_dr_energy": float(
                self.useful_settlement_dr_energy[h]
            ),
            "excess_settlement_dr_energy": float(
                self.excess_settlement_dr_energy[h]
            ),
            "control_target_load": float(self.control_target_load[h]),
            "control_over_error": float(self.control_over_error[h]),
            "control_under_error": float(self.control_under_error[h]),
            "target_tracking_penalty": float(
                self.target_tracking_penalty[h]
            ),
            "reward_base_component": float(self.reward_base_component[h]),
            "reward_tracking_component": float(
                self.reward_tracking_component[h]
            ),
            "reward_no_need_component": float(
                self.reward_no_need_component[h]
            ),
            "no_need_gate": bool(self.no_need_gate_flag[h]),
            "no_need_offer_penalty": float(
                self.no_need_offer_penalty[h]
            ),
            "combined_offer_penalty": float(
                self.combined_offer_penalty[h]
            ),
            "unnecessary_incentive": bool(
                self.unnecessary_incentive_flag[h]
            ),
            "positive_incentive_no_response": bool(
                self.positive_incentive_no_response_flag[h]
            ),
            "maximum_action": bool(self.maximum_action_flag[h]),
        }
        return observation, reward, done, info

    def apply_incentives(self, raw_action):
        h = self.curr_step
        incentives = self.lam_min[h] + raw_action * (
            self.lam_max[h] - self.lam_min[h]
        )
        incentives = np.maximum(incentives, 0.0)
        self.incentives[h] = incentives
        return incentives


    def _house_processing_order(self, h, device_idx=0):
        """Return a deterministic, balanced round-robin household order.

        The former fixed ``0, 1, ..., N-1`` order gave the first household
        permanent priority whenever households competed for the same future
        capacity.  Rotating the starting household by day, hour, and device
        removes that structural bias while keeping repeated evaluation of the
        same day deterministic.
        """
        if self.N <= 1:
            return np.arange(self.N, dtype=int)

        day = int(getattr(self, "day", 0))
        start = (day + int(h) + int(device_idx)) % self.N
        base = np.arange(self.N, dtype=int)
        return np.roll(base, -start)

    def _deadline_index(self, device_idx):
        name = str(self.DEVICES[device_idx]).lower()
        ddl = int(self.ts_deadline_hour.get(name, self.T - 1))
        return int(np.clip(ddl, 0, self.T - 1))

    def _find_original_tsni_block(self, h, i, d):
        """Return (start, end, profile) for an original TS-NI block."""
        if self.device_demands[h, i, d] <= self.EPS:
            return None
        if (
            self.device_shifted_out[h, i, d] > self.EPS
            or self.device_conflict_deferred_out[h, i, d] > self.EPS
        ):
            return None

        t0 = h
        while (
            t0 - 1 >= 0
            and self.device_demands[t0 - 1, i, d] > self.EPS
        ):
            t0 -= 1
        t1 = h
        while (
            t1 + 1 < self.T
            and self.device_demands[t1 + 1, i, d] > self.EPS
        ):
            t1 += 1

        if h != t0:
            return None
        if t0 == 0 and self.ts_carry_over[i, d]:
            return None

        profile = self.device_demands[t0 : t1 + 1, i, d].copy()
        if profile.size == 0 or profile.sum() <= self.EPS:
            return None
        return t0, t1, profile

    def _find_conflict_deferral_destination(self, h, profile, i, d):
        """Find the earliest causal device-feasible slot for a forced deferral."""
        length = len(profile)
        ddl = self._deadline_index(d)
        latest = min(self.T - length, ddl - length + 1)
        if h + 1 > latest:
            return None
        for start in range(h + 1, latest + 1):
            occupied = False
            for k in range(length):
                t = start + k
                known_inj = (
                    self.device_future_inj[t, i, d]
                    + self.device_conflict_future_inj[t, i, d]
                )
                if known_inj > self.EPS:
                    occupied = True
                    break
            if not occupied:
                return start
        return None

    def _record_conflict_discomfort(self, h, i, d, discomfort):
        """Record non-incentivised physical-delay discomfort at hour ``h``."""
        value = max(0.0, float(discomfort))
        if value <= self.EPS:
            return
        self.discomforts[h, i] += value
        self.device_discomfort[h, i, d] += value
        self.device_conflict_discomfort[h, i, d] += value

    def _resolve_tsni_conflicts_at_hour(self, h):
        """Defer a newly declared TS-NI programme if its appliance is occupied.

        The conflict is detected only when the original programme becomes
        observable at hour ``h``. The complete programme profile is treated as
        declared to the household HEMS at request time. Forced deferrals carry
        no incentive settlement. Their delay dissatisfaction is nevertheless
        included in the household utility/reward because the physical delay is
        caused by an earlier scheduling commitment.
        """
        if h < 0 or h >= self.T:
            return
        for d in range(self.num_devices):
            if not self.TS_NI_MASK[d]:
                continue
            for i in range(self.N):
                block = self._find_original_tsni_block(h, i, d)
                if block is None:
                    continue
                t0, t1, profile = block
                known_occupancy = (
                    self.device_future_inj[t0 : t1 + 1, i, d]
                    + self.device_conflict_future_inj[t0 : t1 + 1, i, d]
                )
                if not np.any(known_occupancy > self.EPS):
                    continue

                destination = self._find_conflict_deferral_destination(
                    h, profile, i, d
                )
                self.device_conflict_deferred_out[t0 : t1 + 1, i, d] += profile
                phi = float(self.dissatisfaction_coefficients[i, d])

                if destination is None:
                    # No device-feasible slot remains before the completion
                    # deadline. The energy is explicitly recorded as unmet,
                    # rather than silently disappearing from the load. Apply
                    # the maximum feasible within-day delay dissatisfaction.
                    self.device_conflict_unserved[t0 : t1 + 1, i, d] += profile
                    self.device_unmet[t0 : t1 + 1, i, d] += profile
                    delay = self.T - t0
                    self._record_conflict_discomfort(
                        h, i, d, phi * (delay ** 2)
                    )
                    continue

                self.device_conflict_future_inj[
                    destination : destination + len(profile), i, d
                ] += profile
                self.device_conflict_deferred_in[
                    destination : destination + len(profile), i, d
                ] += profile
                delay = destination - t0
                self._record_conflict_discomfort(
                    h, i, d, phi * (delay ** 2)
                )

    def _allocate_forced_ev_deferral(self, amount, h, i, d):
        """Place forced EV deferral in earliest charger-feasible future slots."""
        ddl = self._deadline_index(d)
        charger_limit = float(self.DEVICE_CONSUMPTION[d])
        remaining = max(0.0, float(amount))
        allocation = []
        for t in range(h + 1, ddl + 1):
            known = float(
                self.device_future_inj[t, i, d]
                + self.device_conflict_future_inj[t, i, d]
            )
            headroom = max(0.0, charger_limit - known)
            move = min(headroom, remaining)
            if move > self.EPS:
                allocation.append((t, move))
                remaining -= move
            if remaining <= self.EPS:
                break
        return allocation, remaining

    def _resolve_ev_charger_conflicts_at_hour(self, h):
        """Enforce the EV charger rating when a new original request appears.

        Earlier shifted-in commitments have priority because they were already
        fixed before hour ``h``. Any newly observed original EV request that
        would push the physical charging power above the configured rating is
        deferred causally to the earliest later charger-feasible slots, without
        incentive settlement. Remaining energy at the deadline is recorded as
        unmet.
        """
        if self.idx_car is None or h < 0 or h >= self.T:
            return
        d = self.idx_car
        limit = float(self.DEVICE_CONSUMPTION[d])
        ddl = self._deadline_index(d)
        for i in range(self.N):
            original = max(
                0.0,
                float(
                    self.device_demands[h, i, d]
                    - self.device_shifted_out[h, i, d]
                    - self.device_conflict_deferred_out[h, i, d]
                ),
            )
            incoming = float(
                self.device_future_inj[h, i, d]
                + self.device_conflict_future_inj[h, i, d]
            )
            excess = max(0.0, original + incoming - limit)
            if excess <= self.EPS:
                continue

            forced = min(original, excess)
            if forced <= self.EPS:
                # This should be impossible once known commitments are capped.
                raise RuntimeError(
                    "Known EV commitments exceed the configured charger rating "
                    f"at hour {h}, house {self.data_ids[i]}."
                )

            allocation, unserved = self._allocate_forced_ev_deferral(
                forced, h, i, d
            )
            self.device_conflict_deferred_out[h, i, d] += forced

            phi = float(self.dissatisfaction_coefficients[i, d])
            scale = max(original, self.EPS)
            for t, move in allocation:
                self.device_conflict_future_inj[t, i, d] += move
                self.device_conflict_deferred_in[t, i, d] += move
                delay = t - h
                self._record_conflict_discomfort(
                    h, i, d, phi * (move / scale) * (delay ** 2)
                )

            if unserved > self.EPS:
                self.device_conflict_unserved[h, i, d] += unserved
                self.device_unmet[h, i, d] += unserved
                max_delay = self.T - h
                self._record_conflict_discomfort(
                    h, i, d, phi * (unserved / scale) * (max_delay ** 2)
                )

    def _resolve_physical_conflicts_at_hour(self, h):
        """Apply causal appliance-level execution constraints before state h."""
        self._resolve_tsni_conflicts_at_hour(h)
        self._resolve_ev_charger_conflicts_at_hour(h)

    def _candidate_total_after_tsni_move(
        self, t, origin_start, origin_profile, dest_start, dest_profile
    ):
        total = self._causal_capacity_estimate_at(t)
        origin_end = origin_start + len(origin_profile) - 1
        dest_end = dest_start + len(dest_profile) - 1

        if origin_start <= t <= origin_end:
            total -= float(origin_profile[t - origin_start])
        if dest_start <= t <= dest_end:
            total += float(dest_profile[t - dest_start])
        return total

    def _find_tsni_destination(self, t0, profile, i, d):
        length = len(profile)
        ddl = self._deadline_index(d)
        earliest = t0 + 1
        latest = min(self.T - length, ddl - length + 1)
        if earliest > latest:
            return None

        best_start = None
        best_score = float("inf")

        for start in range(earliest, latest + 1):


            feasible = True
            window_totals = []
            for k, power in enumerate(profile):
                t = start + k
                # Avoid double-booking only against schedules already
                # committed by earlier decisions.  Future realised appliance
                # demand is not available at decision time and is therefore
                # not consulted here.
                if (
                    self.device_future_inj[t, i, d]
                    + self.device_conflict_future_inj[t, i, d]
                    > self.EPS
                ):
                    feasible = False
                    break

                candidate_total = self._candidate_total_after_tsni_move(
                    t,
                    t0,
                    profile,
                    start,
                    profile,
                )
                if candidate_total > self.capacity_threshold + self.EPS:
                    feasible = False
                    break
                window_totals.append(candidate_total)

            if feasible:
                score = float(np.mean(window_totals))
                if score < best_score:
                    best_score = score
                    best_start = start

        return best_start

    def _schedule_tsni_jobs(self, incentives, h, cons_eff):


        for d in range(self.num_devices):
            if not self.TS_NI_MASK[d]:
                continue

            order = self._house_processing_order(h, d)
            if hasattr(self, "tsni_house_order"):
                self.tsni_house_order[h, d, : len(order)] = order

            for i in order:
                i = int(i)
                if float(incentives[i]) <= self.EPS:
                    continue

                block = self._find_original_tsni_block(h, i, d)
                if block is None:
                    continue
                t0, t1, profile = block


                # The admissibility decision is made at the request hour.
                # Future realised prices are not available at that time.
                locked_rate = float(incentives[i])
                current_price = float(self.prices[h])
                admissible_rate = max(0.0, 0.95 * current_price)
                if locked_rate > admissible_rate + self.EPS:
                    self.device_settlement_ineligible_request[h, i, d] += float(
                        profile.sum()
                    )
                    continue

                destination = self._find_tsni_destination(
                    t0, profile, i, d
                )
                if destination is None:

                    continue

                self.device_shifted_out[t0 : t1 + 1, i, d] += profile
                self.shifted_out_locked_rate[t0 : t1 + 1, i, d] = locked_rate
                self.shifted_out_decision_hour[t0 : t1 + 1, i, d] = h
                self.shifted_out_has_ledger[t0 : t1 + 1, i, d] = True
                self._removed_mask[t0 : t1 + 1, i, d] = True
                self.device_future_inj[
                    destination : destination + len(profile), i, d
                ] += profile
                self.device_shifted_in[
                    destination : destination + len(profile), i, d
                ] += profile


                cons_eff[i, d] = max(
                    0.0, cons_eff[i, d] - float(profile[0])
                )

                delay = destination - t0
                phi = float(self.dissatisfaction_coefficients[i, d])
                discomfort = phi * (delay ** 2)
                self.discomforts[h, i] += discomfort
                self.device_discomfort[h, i, d] += discomfort
                self.new_shifted_out_at_action[h, i, d] += float(profile.sum())

    def _plan_ev_allocation(self, amount, h, i, d):
        ddl = self._deadline_index(d)
        if amount <= self.EPS or h >= ddl:
            return []

        allocation = []
        remaining = float(amount)
        temp_additions = {}
        charger_limit = float(self.DEVICE_CONSUMPTION[d])
        for t in range(h + 1, ddl + 1):
            committed = self._causal_capacity_estimate_at(t) + temp_additions.get(t, 0.0)
            capacity_headroom = max(0.0, self.capacity_threshold - committed)
            known_ev_commitment = float(
                self.device_future_inj[t, i, d]
                + self.device_conflict_future_inj[t, i, d]
            )
            charger_headroom = max(
                0.0,
                charger_limit - known_ev_commitment - temp_additions.get(t, 0.0),
            )
            move = min(capacity_headroom, charger_headroom, remaining)
            if move > self.EPS:
                allocation.append((t, move))
                temp_additions[t] = temp_additions.get(t, 0.0) + move
                remaining -= move
            if remaining <= self.EPS:
                break
        return allocation

    def _schedule_ev(self, incentives, h, cons_eff):
        if self.idx_car is None:
            return
        d = self.idx_car

        order = self._house_processing_order(h, d)
        if hasattr(self, "ev_house_order"):
            self.ev_house_order[h, : len(order)] = order

        for i in order:
            i = int(i)
            if float(incentives[i]) <= self.EPS:
                continue

            original_available = max(
                0.0,
                self.device_demands[h, i, d]
                - self.device_shifted_out[h, i, d]
                - self.device_conflict_deferred_out[h, i, d],
            )
            if original_available <= self.EPS:
                continue

            total_now = float(
                cons_eff.sum()
                + self.nonshift_baseline_per_house[h].sum()
            )
            overrun = max(0.0, total_now - self.capacity_threshold)
            requested_move = min(original_available, overrun)
            if requested_move <= self.EPS:
                continue

            allocation = self._plan_ev_allocation(requested_move, h, i, d)
            feasible_move = float(sum(move for _, move in allocation))
            unshifted_request = max(0.0, requested_move - feasible_move)
            if hasattr(self, "device_unshifted_request"):
                self.device_unshifted_request[h, i, d] += unshifted_request


            if feasible_move <= self.EPS:
                continue

            self.device_shifted_out[h, i, d] += feasible_move
            self.shifted_out_locked_rate[h, i, d] = float(incentives[i])
            self.shifted_out_decision_hour[h, i, d] = h
            self.shifted_out_has_ledger[h, i, d] = True
            self.new_shifted_out_at_action[h, i, d] += feasible_move
            self._removed_mask[h, i, d] = True
            cons_eff[i, d] = max(0.0, cons_eff[i, d] - feasible_move)

            phi = float(self.dissatisfaction_coefficients[i, d])
            original_scale = max(original_available, self.EPS)
            for t, move in allocation:
                self.device_future_inj[t, i, d] += move
                self.device_shifted_in[t, i, d] += move
                delay = t - h
                discomfort = phi * (move / original_scale) * (delay ** 2)
                self.discomforts[h, i] += discomfort
                self.device_discomfort[h, i, d] += discomfort


    def _populate_verified_settlement(self, h, incentives, after_house):
        """Populate source-level verified settlement for one hour.

        Household-level verified relief is ``max(0, baseline - realised)``.
        When several devices/decisions contribute in the same household-hour,
        that verified total is allocated proportionally across the raw physical
        relief sources. This preserves the household-baseline settlement rule
        while keeping each source tied to its causal decision-hour incentive.
        """
        current_curtailment = np.asarray(self.device_curtailed[h], dtype=float)
        shifted_origin_relief = np.asarray(self.device_shifted_out[h], dtype=float)

        missing_ledger = (shifted_origin_relief > self.EPS) & (
            ~self.shifted_out_has_ledger[h]
        )
        if np.any(missing_ledger):
            location = np.argwhere(missing_ledger)[0]
            raise RuntimeError(
                "Shifted-out settlement slice has no locked-rate ledger entry: "
                f"hour={h}, house={self.data_ids[int(location[0])]}, "
                f"device={self.DEVICES[int(location[1])]}."
            )

        raw_by_device = current_curtailment + shifted_origin_relief
        raw_payment_by_device = (
            current_curtailment * np.asarray(incentives, dtype=float)[:, None]
            + shifted_origin_relief * self.shifted_out_locked_rate[h]
        )
        raw_house = raw_by_device.sum(axis=1)
        # Forced device-conflict deferrals are physical scheduling corrections,
        # not incentive-settled DR. Remove their signed effect from the
        # settlement counterfactual before verifying incentivised relief.
        conflict_out = self.device_conflict_deferred_out[h].sum(axis=1)
        conflict_in = self.device_conflict_deferred_in[h].sum(axis=1)
        settlement_reference = (
            self.baseline_per_house[h] - conflict_out + conflict_in
        )
        verified_house = np.maximum(
            0.0, settlement_reference - np.asarray(after_house, dtype=float)
        )

        if np.any(verified_house > raw_house + self.data_tolerance):
            idx = int(np.argmax(verified_house - raw_house))
            raise RuntimeError(
                "Verified household reduction exceeds attributed physical relief "
                f"at hour {h}, house {self.data_ids[idx]}: "
                f"verified={verified_house[idx]:.8f}, raw={raw_house[idx]:.8f}."
            )

        factor = np.divide(
            verified_house,
            raw_house,
            out=np.zeros_like(verified_house),
            where=raw_house > self.EPS,
        )
        factor = np.clip(factor, 0.0, 1.0)
        verified_by_device = raw_by_device * factor[:, None]
        payment_by_device = raw_payment_by_device * factor[:, None]
        rate_by_device = np.divide(
            payment_by_device,
            verified_by_device,
            out=np.zeros_like(payment_by_device),
            where=verified_by_device > self.EPS,
        )


        price = float(self.prices[h])
        above_price = (verified_by_device > self.EPS) & (
            rate_by_device > price + self.data_tolerance
        )
        negative_margin = np.maximum(
            0.0,
            payment_by_device - price * verified_by_device,
        )

        decision_hour = np.full((self.N, self.num_devices), -1, dtype=int)
        decision_hour[current_curtailment > self.EPS] = h
        shifted_mask = shifted_origin_relief > self.EPS
        decision_hour[shifted_mask] = self.shifted_out_decision_hour[h][shifted_mask]

        self.device_raw_settlement_relief[h] = raw_by_device
        self.device_verified_settlement_energy[h] = verified_by_device
        self.device_settlement_payment[h] = payment_by_device
        self.device_settlement_rate[h] = rate_by_device
        self.device_settlement_decision_hour[h] = decision_hour
        self.device_settlement_rate_above_price[h] = above_price
        self.device_settlement_above_price_energy[h] = np.where(
            above_price, verified_by_device, 0.0
        )
        self.device_settlement_negative_margin[h] = negative_margin
        self.settlement_allocation_factor[h] = factor
        self.settlement_dr_energy[h] = verified_by_device.sum(axis=1)
        self.incentive_payment[h] = payment_by_device.sum(axis=1)
        self.settlement_effective_rate[h] = np.divide(
            self.incentive_payment[h],
            self.settlement_dr_energy[h],
            out=np.zeros(self.N, dtype=float),
            where=self.settlement_dr_energy[h] > self.EPS,
        )

    def apply_demand_response(self, incentives, h):
        """Apply the current action and populate all physical/economic logs."""
        pre_action = self._current_pre_action_device(h)
        cons_eff = pre_action.copy()

        self.pre_action_device[h] = pre_action
        pre_house = pre_action.sum(axis=1) + self.nonshift_baseline_per_house[h]
        self.pre_action_total_per_house[h] = pre_house
        self.before_total_per_house[h] = pre_house
        self.needed_reduction[h] = max(
            0.0, float(pre_house.sum()) - self.capacity_threshold
        )


        if not self.disable_pc:
            for i in range(self.N):
                lam_i = float(incentives[i])
                for d in range(self.num_devices):
                    if not self.PC_MASK[d]:
                        continue
                    cons_d = float(cons_eff[i, d])
                    if cons_d <= self.EPS:
                        continue

                    scale = max(cons_d, 1e-6)
                    kappa = float(self.dissatisfaction_coefficients[i, d])
                    best_value = 0.0
                    best_reduction = 0.0
                    for rate in self.POWER_RATE:
                        reduction = float(rate) * cons_d
                        discomfort = kappa * ((reduction / scale) ** 2)
                        value = lam_i * reduction - discomfort
                        if value > best_value + self.EPS:
                            best_value = value
                            best_reduction = reduction

                    if best_reduction > self.EPS:
                        cons_eff[i, d] -= best_reduction
                        self.device_curtailed[h, i, d] += best_reduction
                        discomfort = kappa * ((best_reduction / scale) ** 2)
                        self.discomforts[h, i] += discomfort
                        self.device_discomfort[h, i, d] += discomfort

        self._schedule_tsni_jobs(incentives, h, cons_eff)
        self._schedule_ev(incentives, h, cons_eff)

        self.device_after[h] = np.maximum(cons_eff, 0.0)
        self.device_after_agg[h] = self.device_after[h].sum(axis=0)
        self.ts_exec_agg[h] = self.device_shifted_in[h].sum(axis=0)

        after_house = (
            self.device_after[h].sum(axis=1)
            + self.nonshift_baseline_per_house[h]
        )
        self.after_total_per_house[h] = after_house


        self.hourly_capacity_relief[h] = np.maximum(0.0, pre_house - after_house)

        self.net_curtailment_per_house[h] = self.device_curtailed[h].sum(axis=1)
        self.shifted_out_per_house[h] = self.device_shifted_out[h].sum(axis=1)
        self.shifted_in_per_house[h] = self.device_shifted_in[h].sum(axis=1)


        self.action_attributed_dr_energy[h] = (
            self.net_curtailment_per_house[h]
            + self.new_shifted_out_at_action[h].sum(axis=1)
        )


        self.signed_load_change[h] = self.baseline_per_house[h] - after_house


        self._populate_verified_settlement(h, incentives, after_house)

        total_after = float(after_house.sum())
        self.capacity_overrun[h] = max(
            0.0, total_after - self.capacity_threshold
        )
        self.capacity_violation[h] = self.capacity_overrun[h] > self.EPS

        device_action_reduction = np.maximum(
            0.0, pre_action - self.device_after[h]
        )
        self.reductions_by_device[h] = device_action_reduction.sum(axis=0)


        self.device_future_inj[h] = 0.0
        self.device_conflict_future_inj[h] = 0.0
        self._refresh_planned_load()

    def compute_service_provider_reward(self):
        h = self.curr_step
        price = float(self.prices[h])


        wholesale_value = price * float(self.settlement_dr_energy[h].sum())
        incentive_cost = float(self.incentive_payment[h].sum())
        settlement_value = wholesale_value - incentive_cost


        self.net_wholesale_procurement_impact[h] = (
            price * float(self.signed_load_change[h].sum())
        )
        self.sp_wholesale_value[h] = wholesale_value
        self.sp_incentive_cost[h] = incentive_cost
        self.rewards_service_provider[h] = settlement_value
        return settlement_value

    def compute_customers_reward(self):
        h = self.curr_step
        weighted_benefit = self.rho * self.incentive_payment[h]
        weighted_discomfort = (1.0 - self.rho) * self.discomforts[h]
        rewards = weighted_benefit - weighted_discomfort

        self.customer_weighted_incentive_benefit[h] = weighted_benefit
        self.rewards_customers[h] = rewards
        return float(rewards.sum())

    def _compute_legacy_shaping(self, h):
        """Return the original heuristic shaping term for ablation only."""
        lambdas = self.incentives[h]
        no_incentive = bool(np.all(lambdas <= self.EPS))
        needed = float(self.needed_reduction[h])
        actual = float(self.hourly_capacity_relief[h].sum())
        violation = float(self.capacity_overrun[h])
        shaping = 0.0

        if violation > self.EPS:
            violation_penalty = 15.0 * violation
            if no_incentive:
                violation_penalty *= 2.0
            shaping -= violation_penalty
        elif needed <= self.EPS:
            if no_incentive:
                shaping += 5.0
            else:
                shaping -= 5.0 * float(np.sum(lambdas))
        else:
            over_reduction = max(0.0, actual - needed)
            shaping -= 0.5 * over_reduction

        return float(shaping)

    def _compute_corrected_objective_components(self, h):
        """Populate reward components and reporting diagnostics.

        The active weighted-stakeholder mode uses action-local pre-action
        capacity need, verified settlement, and a symmetric Smooth-L1 tracking
        cost around ``min(pre_action_load, capacity)``.
        """
        price = float(self.prices[h])
        baseline_total = float(self.total_baseline[h])
        pre_total = float(self.pre_action_total_per_house[h].sum())
        post_total = float(self.after_total_per_house[h].sum())
        settlement_total = float(self.settlement_dr_energy[h].sum())
        payment_total = float(self.incentive_payment[h].sum())
        discomfort_total = float(self.discomforts[h].sum())

        raw_discomfort_by_class = np.asarray([
            float(self.device_discomfort[h, :, self.PC_MASK].sum()),
            float(self.device_discomfort[h, :, self.TS_I_MASK].sum()),
            float(self.device_discomfort[h, :, self.TS_NI_MASK].sum()),
        ], dtype=float)
        class_scales = np.asarray([
            self.reward_discomfort_class_scales[name]
            for name in self.reward_discomfort_class_names
        ], dtype=float)
        normalized_by_class = raw_discomfort_by_class / np.maximum(
            class_scales, self.EPS
        )
        if self.discomfort_normalization_mode == "per_class_mean":
            normalized_discomfort = float(np.mean(normalized_by_class))
        else:
            normalized_discomfort = discomfort_total / max(
                self.reward_discomfort_scale, self.EPS
            )


        baseline_need = max(0.0, baseline_total - self.capacity_threshold)
        baseline_useful = min(settlement_total, baseline_need)
        baseline_excess = max(0.0, settlement_total - baseline_need)
        native_corrected_economic = price * baseline_useful - payment_total
        normalized_corrected_economic = native_corrected_economic / max(
            self.reward_economic_scale, self.EPS
        )

        offer_intensity = float(np.mean(self.selected_raw_action[h]))
        offer_penalty = (
            self.incentive_offer_coefficient * offer_intensity
            if self.offer_regularization_enabled
            else 0.0
        )
        corrected_base = (
            normalized_corrected_economic
            - self.discomfort_weight * normalized_discomfort
            - offer_penalty
        )


        pre_action_need = max(0.0, pre_total - self.capacity_threshold)
        pre_action_useful = min(settlement_total, pre_action_need)
        pre_action_excess = max(0.0, settlement_total - pre_action_need)
        sp_normalized = (
            price * pre_action_useful - payment_total
        ) / max(self.reward_economic_scale, self.EPS)
        eu_normalized = (
            self.rho * payment_total / max(self.reward_economic_scale, self.EPS)
            - (1.0 - self.rho) * normalized_discomfort
        )
        stakeholder_base = (
            self.stakeholder_sp_weight * sp_normalized
            + self.stakeholder_eu_weight * eu_normalized
        )

        no_need_gate = (
            pre_total
            <= self.capacity_threshold + self.no_need_gate_tolerance_kw
        )
        no_need_offer_penalty = (
            self.no_need_offer_coefficient * offer_intensity
            if self.no_need_offer_regularization_enabled and no_need_gate
            else 0.0
        )

        target = min(pre_total, self.capacity_threshold)


        overload_regime = pre_total > self.capacity_threshold + self.EPS
        deadband = self.track_deadband_kwh if overload_regime else 0.0
        legacy_over_error = max(0.0, post_total - self.capacity_threshold)
        legacy_under_error = max(0.0, target - deadband - post_total)
        legacy_tracking_penalty = (
            self.track_c_violation
            * (legacy_over_error / max(self.capacity_threshold, self.EPS)) ** 2
            + self.track_c_under
            * (legacy_under_error / max(self.capacity_threshold, self.EPS))
        )


        signed_error_normalized = (
            post_total - target
        ) / max(self.capacity_threshold, self.EPS)
        abs_error = abs(signed_error_normalized)
        delta = max(self.symmetric_tracking_delta, self.EPS)
        if self.symmetric_tracking_loss_name == "smooth_l1":
            symmetric_loss = (
                0.5 * signed_error_normalized ** 2 / delta
                if abs_error <= delta
                else abs_error - 0.5 * delta
            )
        else:
            symmetric_loss = (
                0.5 * signed_error_normalized ** 2
                if abs_error <= delta
                else delta * (abs_error - 0.5 * delta)
            )
        symmetric_penalty = self.symmetric_tracking_lambda * symmetric_loss

        weighted_mode = self.reward_mode in {
            "weighted_stakeholder_tracking",
            "weighted_stakeholder_tracking_no_need",
        }
        if weighted_mode:
            active_useful = pre_action_useful
            active_excess = pre_action_excess
            active_over_error = max(0.0, post_total - target)
            active_under_error = max(0.0, target - post_total)
            active_tracking_penalty = symmetric_penalty
            active_normalized_economic = sp_normalized
        else:
            active_useful = baseline_useful
            active_excess = baseline_excess
            active_over_error = legacy_over_error
            active_under_error = legacy_under_error
            active_tracking_penalty = legacy_tracking_penalty
            active_normalized_economic = normalized_corrected_economic

        has_incentive = bool(np.any(self.incentives[h] > self.EPS))
        current_response = max(
            float(self.hourly_capacity_relief[h].sum()),
            float(self.action_attributed_dr_energy[h].sum()),
        )

        self.baseline_capacity_need[h] = baseline_need
        self.pre_action_capacity_need[h] = pre_action_need
        self.baseline_useful_settlement_dr_energy[h] = baseline_useful
        self.pre_action_useful_settlement_dr_energy[h] = pre_action_useful
        self.useful_settlement_dr_energy[h] = active_useful
        self.excess_settlement_dr_energy[h] = active_excess
        self.normalized_economic_component[h] = active_normalized_economic
        self.normalized_discomfort_component[h] = normalized_discomfort
        self.raw_discomfort_by_class[h] = raw_discomfort_by_class
        self.normalized_discomfort_by_class[h] = normalized_by_class
        self.incentive_offer_intensity[h] = offer_intensity
        self.incentive_offer_penalty[h] = offer_penalty
        self.no_need_gate_flag[h] = bool(no_need_gate)
        self.no_need_offer_penalty[h] = float(no_need_offer_penalty)
        self.combined_offer_penalty[h] = float(
            offer_penalty + no_need_offer_penalty
        )
        self.corrected_economic_reward_component[h] = corrected_base
        self.stakeholder_sp_normalized_component[h] = sp_normalized
        self.stakeholder_eu_normalized_component[h] = eu_normalized
        self.stakeholder_reward_component[h] = stakeholder_base
        self.signed_tracking_error_normalized[h] = signed_error_normalized
        self.symmetric_tracking_loss[h] = symmetric_loss
        self.symmetric_tracking_penalty[h] = symmetric_penalty
        self.control_target_load[h] = target
        self.control_over_error[h] = active_over_error
        self.control_under_error[h] = active_under_error
        self.target_tracking_penalty[h] = active_tracking_penalty
        self.unnecessary_incentive_flag[h] = (
            has_incentive
            and pre_total <= self.capacity_threshold + self.EPS
        )
        self.positive_incentive_no_response_flag[h] = (
            has_incentive and current_response <= self.EPS
        )
        self.maximum_action_flag[h] = bool(
            np.all(self.selected_raw_action[h] >= 1.0 - self.EPS)
        )
        return (
            float(corrected_base),
            float(legacy_tracking_penalty),
            float(no_need_offer_penalty),
            float(stakeholder_base),
            float(symmetric_penalty),
        )

    def compute_total_reward(self):
        """Compute one hourly reward under the configured ablation mode."""
        h = self.curr_step
        legacy_economic_reward = float(self.rewards_service_provider[h]) + float(
            self.rewards_customers[h].sum()
        )
        (
            corrected_base,
            legacy_tracking_penalty,
            no_need_offer_penalty,
            stakeholder_base,
            symmetric_tracking_penalty,
        ) = self._compute_corrected_objective_components(h)

        shaping = 0.0
        tracking_component = 0.0
        no_need_component = 0.0

        if self.reward_mode == "economic_only":
            base_component = legacy_economic_reward

        elif self.reward_mode == "capacity_penalty":
            base_component = legacy_economic_reward
            violation = float(self.capacity_overrun[h])
            needed = float(self.needed_reduction[h])
            actual = float(self.hourly_capacity_relief[h].sum())
            over_reduction = max(0.0, actual - needed)
            shaping = (
                -self.c_cap * violation
                -self.c_over * over_reduction
            )

        elif self.reward_mode == "legacy_shaping":
            base_component = legacy_economic_reward
            shaping = self._compute_legacy_shaping(h)

        elif self.reward_mode == "corrected_economic":
            base_component = corrected_base

        elif self.reward_mode == "target_tracking":
            base_component = corrected_base
            tracking_component = -legacy_tracking_penalty

        elif self.reward_mode == "target_tracking_no_need":
            base_component = corrected_base
            tracking_component = -legacy_tracking_penalty
            no_need_component = -no_need_offer_penalty

        elif self.reward_mode == "weighted_stakeholder_tracking":
            base_component = stakeholder_base
            tracking_component = -symmetric_tracking_penalty

        elif self.reward_mode == "weighted_stakeholder_tracking_no_need":
            base_component = stakeholder_base
            tracking_component = -symmetric_tracking_penalty
            no_need_component = -no_need_offer_penalty

        else:
            raise RuntimeError(f"Unsupported reward mode: {self.reward_mode!r}")

        total_reward = (
            float(base_component)
            + float(shaping)
            + float(tracking_component)
            + float(no_need_component)
        )
        self.reward_base_component[h] = float(base_component)
        self.reward_shaping_component[h] = float(shaping)
        self.reward_tracking_component[h] = float(tracking_component)
        self.reward_no_need_component[h] = float(no_need_component)
        self.rewards_total[h] = total_reward
        return total_reward


    def compute_total_demand(self, step=None):
        if step is None:
            return self.baseline_per_house.sum(axis=1)
        return float(self.baseline_per_house[step].sum())

    def compute_total_reduction(self, step=None):
        """Return immediate action-induced capacity relief."""
        if step is None:
            return self.hourly_capacity_relief.sum(axis=1)
        return float(self.hourly_capacity_relief[int(step)].sum())

    def compute_signed_load_change(self, step=None):
        values = self.signed_load_change.sum(axis=1)
        if step is None:
            return values
        return float(values[int(step)])

    def compute_settlement_dr_energy(self, step=None):
        """Return non-negative baseline-verified settlement energy."""
        values = self.settlement_dr_energy.sum(axis=1)
        if step is None:
            return values
        return float(values[int(step)])

    def compute_action_attributed_dr_energy(self, step=None):
        """Return full DR energy newly attributed to each current action."""
        values = self.action_attributed_dr_energy.sum(axis=1)
        if step is None:
            return values
        return float(values[int(step)])

    def compute_contracted_dr_energy(self, step=None):
        """Compatibility wrapper for the action-attributed diagnostic."""
        return self.compute_action_attributed_dr_energy(step=step)

    def compute_total_curtailment(self, step=None):
        """Return true curtailed energy, excluding all load shifting."""
        values = self.net_curtailment_per_house.sum(axis=1)
        if step is None:
            return values
        return float(values[int(step)])

    def compute_total_shifted_out(self, step=None):
        values = self.shifted_out_per_house.sum(axis=1)
        if step is None:
            return values
        return float(values[int(step)])

    def compute_total_consumption(self, step=None):
        if step is None:
            return self.after_total_per_house.sum(axis=1)
        return float(self.after_total_per_house[step].sum())

    def get_lam_bounds(self):
        lam_min = np.zeros_like(self.prices, dtype=float)

        lam_max = np.maximum(0.0, 0.95 * self.prices)
        return lam_min, lam_max

    def get_step_data(self, step=None):
        h = self.curr_step if step is None else int(step)
        return (
            h,
            self.baseline_per_house[h],
            self.incentives[h],
            self.prices[h],
            self.signed_load_change[h],
        )

    def get_device_data(self, step=None):
        h = self.curr_step if step is None else int(step)
        return self.device_demands[h]

    def get_total_device_consumption(self, step=None):
        h = self.curr_step if step is None else int(step)
        return self.device_demands[h].sum(axis=1)

    def get_reporting_audit(self):
        """Return report-ready arrays without recomputing reward economics."""
        baseline_relief = np.maximum(0.0, self.signed_load_change)
        rebound = np.maximum(0.0, -self.signed_load_change)
        return {
            "reward_mode": self.reward_mode,
            "reward_metadata": self.get_reward_metadata(),
            "c_cap": float(self.c_cap),
            "c_over": float(self.c_over),
            "hourly_capacity_relief": self.hourly_capacity_relief.copy(),
            "signed_load_change": self.signed_load_change.copy(),
            "settlement_dr_energy": self.settlement_dr_energy.copy(),
            "settlement_effective_rate": self.settlement_effective_rate.copy(),
            "action_attributed_dr_energy": self.action_attributed_dr_energy.copy(),
            "incentive_payment": self.incentive_payment.copy(),
            "sp_settlement_wholesale_value": self.sp_wholesale_value.copy(),
            "sp_settlement_incentive_cost": self.sp_incentive_cost.copy(),
            "sp_settlement_value": self.rewards_service_provider.copy(),
            "net_wholesale_procurement_impact": (
                self.net_wholesale_procurement_impact.copy()
            ),
            "customer_weighted_utility": self.rewards_customers.copy(),
            "reward_base_component": self.reward_base_component.copy(),
            "reward_shaping_component": self.reward_shaping_component.copy(),
            "reward_tracking_component": self.reward_tracking_component.copy(),
            "reward_no_need_component": self.reward_no_need_component.copy(),
            "corrected_economic_reward_component": (
                self.corrected_economic_reward_component.copy()
            ),
            "normalized_economic_component": (
                self.normalized_economic_component.copy()
            ),
            "normalized_discomfort_component": (
                self.normalized_discomfort_component.copy()
            ),
            "raw_discomfort_by_class": self.raw_discomfort_by_class.copy(),
            "normalized_discomfort_by_class": (
                self.normalized_discomfort_by_class.copy()
            ),
            "incentive_offer_intensity": (
                self.incentive_offer_intensity.copy()
            ),
            "incentive_offer_penalty": self.incentive_offer_penalty.copy(),
            "no_need_gate_flag": self.no_need_gate_flag.copy(),
            "no_need_offer_penalty": self.no_need_offer_penalty.copy(),
            "combined_offer_penalty": self.combined_offer_penalty.copy(),
            "baseline_capacity_need": self.baseline_capacity_need.copy(),
            "pre_action_capacity_need": self.pre_action_capacity_need.copy(),
            "baseline_useful_settlement_dr_energy": (
                self.baseline_useful_settlement_dr_energy.copy()
            ),
            "pre_action_useful_settlement_dr_energy": (
                self.pre_action_useful_settlement_dr_energy.copy()
            ),
            "stakeholder_sp_normalized_component": (
                self.stakeholder_sp_normalized_component.copy()
            ),
            "stakeholder_eu_normalized_component": (
                self.stakeholder_eu_normalized_component.copy()
            ),
            "stakeholder_reward_component": (
                self.stakeholder_reward_component.copy()
            ),
            "signed_tracking_error_normalized": (
                self.signed_tracking_error_normalized.copy()
            ),
            "symmetric_tracking_loss": self.symmetric_tracking_loss.copy(),
            "symmetric_tracking_penalty": (
                self.symmetric_tracking_penalty.copy()
            ),
            "useful_settlement_dr_energy": (
                self.useful_settlement_dr_energy.copy()
            ),
            "excess_settlement_dr_energy": (
                self.excess_settlement_dr_energy.copy()
            ),
            "control_target_load": self.control_target_load.copy(),
            "control_over_error": self.control_over_error.copy(),
            "control_under_error": self.control_under_error.copy(),
            "target_tracking_penalty": self.target_tracking_penalty.copy(),
            "unnecessary_incentive_flag": (
                self.unnecessary_incentive_flag.copy()
            ),
            "positive_incentive_no_response_flag": (
                self.positive_incentive_no_response_flag.copy()
            ),
            "maximum_action_flag": self.maximum_action_flag.copy(),
            "selected_action_index": self.selected_action_index.copy(),
            "selected_raw_action": self.selected_raw_action.copy(),
            "baseline_relative_relief": baseline_relief,
            "rebound_energy": rebound,
            "curtailment": self.net_curtailment_per_house.copy(),
            "shifted_out": self.shifted_out_per_house.copy(),
            "shifted_in": self.shifted_in_per_house.copy(),
            "device_discomfort": self.device_discomfort.copy(),
            "device_conflict_discomfort": self.device_conflict_discomfort.copy(),
            "device_conflict_deferred_out": self.device_conflict_deferred_out.copy(),
            "device_conflict_deferred_in": self.device_conflict_deferred_in.copy(),
            "device_conflict_unserved": self.device_conflict_unserved.copy(),
            "device_unmet": self.device_unmet.copy(),
            "device_verified_settlement_energy": (
                self.device_verified_settlement_energy.copy()
            ),
            "device_settlement_payment": self.device_settlement_payment.copy(),
            "device_settlement_rate": self.device_settlement_rate.copy(),
            "device_settlement_decision_hour": (
                self.device_settlement_decision_hour.copy()
            ),
            "settlement_allocation_factor": (
                self.settlement_allocation_factor.copy()
            ),
            "settlement_ineligible_request": (
                self.device_settlement_ineligible_request.copy()
            ),
            "capacity_overrun": self.capacity_overrun.copy(),
        }

    def get_energy_audit(self):
        """Return episode-level energy-accounting arrays for diagnostics."""
        original = self.device_before.sum(axis=0)
        final = self.device_after.sum(axis=0)
        shifted_out = self.device_shifted_out.sum(axis=0)
        shifted_in = self.device_shifted_in.sum(axis=0)
        curtailed = self.device_curtailed.sum(axis=0)
        unmet = self.device_unmet.sum(axis=0)
        unshifted_request = self.device_unshifted_request.sum(axis=0)


        balance_error = original - final - curtailed - unmet
        return {
            "original": original,
            "final": final,
            "shifted_out": shifted_out,
            "shifted_in": shifted_in,
            "curtailed": curtailed,
            "unmet": unmet,
            "unshifted_request": unshifted_request,
            "conflict_deferred_out": self.device_conflict_deferred_out.sum(axis=0),
            "conflict_deferred_in": self.device_conflict_deferred_in.sum(axis=0),
            "conflict_unserved": self.device_conflict_unserved.sum(axis=0),
            "conflict_discomfort": self.device_conflict_discomfort.sum(axis=0),
            "balance_error": balance_error,
        }