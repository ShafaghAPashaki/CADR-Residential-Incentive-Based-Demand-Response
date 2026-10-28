"""Build the forecast-only input table consumed by the DDQN decision state.

Training days use the frozen out-of-fold/cross-fitted forecasts. Validation,
test, and later diagnostic days use the frozen full-model forecasts. Audit
actuals are deliberately excluded from the output table.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
FORECAST_DIR = ROOT / "results" / "forecast"
CROSSFIT = FORECAST_DIR / "crossfit" / "ddqn_crossfit_forecasts_days_002_167.csv"
LOAD_FULL = FORECAST_DIR / "load" / "lstm_load_predictions_all.csv"
PRICE_FULL = FORECAST_DIR / "price" / "lstm_price_predictions_all.csv"
CROSSFIT_GATE = FORECAST_DIR / "crossfit" / "crossfit_gate.txt"
OUTPUT = FORECAST_DIR / "ddqn_state_forecasts.csv"
PROVENANCE = ROOT / "repro_runs" / "forecast" / "ddqn_state_forecasts_provenance.json"
HOUSE_IDS = [661, 3039, 8565]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    for target in (OUTPUT, PROVENANCE):
        if target.exists():
            raise FileExistsError(f"Refusing to overwrite existing output: {target}")
    if CROSSFIT_GATE.read_text(encoding="utf-8").strip() != "READY":
        raise RuntimeError("Cross-fit gate is not READY.")

    cf = pd.read_csv(CROSSFIT, parse_dates=["timestamp"])
    cf = cf[
        [
            "timestamp",
            "doy",
            "forecast_load_eu1_kw",
            "forecast_load_eu2_kw",
            "forecast_load_eu3_kw",
            "forecast_aggregate_load_kw",
            "forecast_price_cent_per_kwh",
            "inputs_padding_free_all_targets",
        ]
    ].copy()
    cf = cf.rename(
        columns={
            "forecast_load_eu1_kw": "forecast_load_house_661_kw",
            "forecast_load_eu2_kw": "forecast_load_house_3039_kw",
            "forecast_load_eu3_kw": "forecast_load_house_8565_kw",
            "inputs_padding_free_all_targets": "inputs_padding_free",
        }
    )
    cf["forecast_source"] = "crossfit"

    load = pd.read_csv(LOAD_FULL, parse_dates=["timestamp"])
    price = pd.read_csv(PRICE_FULL, parse_dates=["timestamp"])
    load = load[load["doy"] >= 168].copy()
    price = price[price["doy"] >= 168].copy()

    wide = load.pivot(
        index=["timestamp", "doy"], columns="house_id", values="forecast_load_kw"
    ).reset_index()
    wide.columns = ["timestamp", "doy"] + [
        f"forecast_load_house_{int(column)}_kw" for column in wide.columns[2:]
    ]
    load_cols = [f"forecast_load_house_{house_id}_kw" for house_id in HOUSE_IDS]
    wide["forecast_aggregate_load_kw"] = wide[load_cols].sum(axis=1)
    full = wide.merge(
        price[
            ["timestamp", "doy", "forecast_price_cent_per_kWh", "padding_free"]
        ],
        on=["timestamp", "doy"],
        how="inner",
        validate="one_to_one",
    ).rename(
        columns={
            "forecast_price_cent_per_kWh": "forecast_price_cent_per_kwh",
            "padding_free": "inputs_padding_free",
        }
    )
    full["forecast_source"] = "full_model"

    columns = [
        "timestamp",
        "doy",
        "forecast_source",
        *load_cols,
        "forecast_aggregate_load_kw",
        "forecast_price_cent_per_kwh",
        "inputs_padding_free",
    ]
    combined = pd.concat([cf[columns], full[columns]], ignore_index=True)
    combined = combined.sort_values("timestamp").reset_index(drop=True)
    combined["forecast_load_clipped_count"] = (
        combined[load_cols] < 0.0
    ).sum(axis=1)

    if combined["timestamp"].duplicated().any():
        raise RuntimeError("Duplicate forecast timestamps.")
    counts = combined.groupby("doy").size()
    if set(counts.index) != set(range(2, 366)) or not (counts == 24).all():
        raise RuntimeError("Forecast coverage must be exactly 24 hours for days 2-365.")
    if any("actual" in column.lower() for column in combined.columns):
        raise RuntimeError("Audit actual columns must not enter DDQN forecast inputs.")
    numeric = combined[
        load_cols + ["forecast_aggregate_load_kw", "forecast_price_cent_per_kwh"]
    ].to_numpy(dtype=float)
    if not np.isfinite(numeric).all():
        raise RuntimeError("Non-finite decision-state forecasts.")
    aggregate_error = np.max(
        np.abs(
            combined[load_cols].sum(axis=1).to_numpy(dtype=float)
            - combined["forecast_aggregate_load_kw"].to_numpy(dtype=float)
        )
    )
    if aggregate_error > 1e-9:
        raise RuntimeError(f"Aggregate forecast mismatch: {aggregate_error:.3e}")

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    PROVENANCE.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT.open("x", encoding="utf-8", newline="") as handle:
        combined.to_csv(handle, index=False)
    provenance = {
        "version": "ddqn_state_forecasts_v1",
        "routing": {
            "days_2_167": "cross-fitted out-of-fold forecasts",
            "days_168_365": "fixed full-model forecasts",
        },
        "household_order": HOUSE_IDS,
        "load_unit": "kW",
        "price_unit": "cent/kWh",
        "state_load_rule": (
            "clip each household forecast at 0 kW before applying known current commitments"
        ),
        "state_price_rule": (
            "log1p(clip(price forecast, 0, 80 cent/kWh)) in the DDQN state"
        ),
        "rows": int(len(combined)),
        "crossfit_rows": int((combined["forecast_source"] == "crossfit").sum()),
        "full_model_rows": int((combined["forecast_source"] == "full_model").sum()),
        "negative_household_load_forecast_values": int(
            (combined[load_cols] < 0.0).to_numpy().sum()
        ),
        "negative_price_forecast_hours": int(
            (combined["forecast_price_cent_per_kwh"] < 0.0).sum()
        ),
        "source_sha256": {
            "crossfit_combined": sha256_file(CROSSFIT),
            "full_load_predictions": sha256_file(LOAD_FULL),
            "full_price_predictions": sha256_file(PRICE_FULL),
            "crossfit_gate": sha256_file(CROSSFIT_GATE),
        },
        "output_sha256": sha256_file(OUTPUT),
    }
    with PROVENANCE.open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(provenance, indent=2, sort_keys=True) + "\n")
    print(json.dumps(provenance, indent=2))


if __name__ == "__main__":
    main()
