"""Strict household-demand and appliance-demand loading helpers.

The loaders deliberately avoid silent imputation.  Missing timestamps,
households, appliance values, duplicates, NaNs, and negative loads are treated
as data errors so that training and paper results cannot be generated from an
incomplete day without an explicit failure.
"""

from __future__ import annotations

import os
from collections.abc import Sequence

import numpy as np
import pandas as pd


data_dir = "data/"
data_path = os.path.join(data_dir, "load_hourly_2018.csv")
max_steps = 24


def _normalise_devices(devices: Sequence[str] | None) -> list[str]:
    if devices is None:
        raise ValueError(
            "The appliance column list must be supplied explicitly. "
            "Do not reload config.yaml inside a data loader."
        )
    result = [str(device) for device in devices]
    if not result:
        raise ValueError("At least one appliance column must be supplied.")
    if len(set(result)) != len(result):
        raise ValueError(f"Duplicate appliance names were supplied: {result}")
    return result


def _read_load_csv(data_path: str, required_value_columns: Sequence[str]) -> pd.DataFrame:
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Household-load file not found: {data_path}")

    df = pd.read_csv(data_path)
    required_columns = ["time", "dataid", *required_value_columns]
    missing_columns = [column for column in required_columns if column not in df.columns]
    if missing_columns:
        raise ValueError(
            f"Household-load file is missing required columns: {missing_columns}"
        )

    df = df[required_columns].copy()
    df["dt"] = pd.to_datetime(
        df.pop("time"),
        format="%d/%m/%Y %H:%M",
        errors="raise",
    )
    df["dataid"] = pd.to_numeric(df["dataid"], errors="raise").astype(int)

    for column in required_value_columns:
        df[column] = pd.to_numeric(df[column], errors="raise")

    if df[["dt", "dataid", *required_value_columns]].isna().any().any():
        bad = df[df[["dt", "dataid", *required_value_columns]].isna().any(axis=1)]
        raise ValueError(
            "NaN values were found in the household-load data. "
            f"First affected rows:\n{bad.head()}"
        )

    duplicate_mask = df.duplicated(subset=["dt", "dataid"], keep=False)
    if duplicate_mask.any():
        duplicates = df.loc[duplicate_mask, ["dt", "dataid"]].head(10)
        raise ValueError(
            "Duplicate (timestamp, house_id) rows were found in the load file. "
            f"Examples:\n{duplicates}"
        )

    negative_columns = [
        column for column in required_value_columns if (df[column] < 0).any()
    ]
    if negative_columns:
        examples = df.loc[
            (df[negative_columns] < 0).any(axis=1),
            ["dt", "dataid", *negative_columns],
        ].head(10)
        raise ValueError(
            f"Negative demand values were found in columns {negative_columns}. "
            f"Examples:\n{examples}"
        )

    return df.sort_values(["dt", "dataid"]).reset_index(drop=True)


def _filter_houses_strict(df: pd.DataFrame, house_ids: Sequence[int] | None) -> pd.DataFrame:
    if house_ids is None:
        return df

    requested = [int(house_id) for house_id in house_ids]
    available = set(df["dataid"].unique().tolist())
    missing = [house_id for house_id in requested if house_id not in available]
    if missing:
        raise ValueError(f"Requested household IDs are absent from the load file: {missing}")

    return df[df["dataid"].isin(requested)].copy()


def load_demand(data_path: str, house_ids: Sequence[int] | None = None) -> pd.DataFrame:
    """Load strict hourly household totals.

    All available dates are retained.  The temporal split, rather than the data
    loader, decides which complete days are used for training, validation, and
    testing.  This also permits the environment to inspect the final hour of the
    preceding day when checking TS-NI carry-over.
    """

    df = _read_load_csv(data_path, required_value_columns=["total"])
    df = _filter_houses_strict(df, house_ids)
    return df.rename(columns={"dataid": "dataid"})


def load_day(df: pd.DataFrame, day_of_year: int, max_hours: int, year: int = 2018) -> pd.DataFrame:
    start = pd.to_datetime(f"{year}-{int(day_of_year)}", format="%Y-%j")
    end = start + pd.to_timedelta(int(max_hours), unit="h")
    return df[(df["dt"] >= start) & (df["dt"] < end)].copy()


def load_baselines(df: pd.DataFrame) -> pd.DataFrame:
    baselines = df[["dataid", "dt", "total"]].copy()
    baselines.columns = ["house_id", "timestamp", "baseline_demand"]
    return baselines


def get_peak_demand(df: pd.DataFrame) -> float:
    return float(df.resample("1h", on="dt")["total"].sum().max())


def load_device_demands(
    data_path: str,
    house_ids: Sequence[int] | None = None,
    devices: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Load appliance columns using the caller's active configuration."""

    device_columns = _normalise_devices(devices)
    df = _read_load_csv(data_path, required_value_columns=device_columns)
    df = _filter_houses_strict(df, house_ids)
    return df[["dt", "dataid", *device_columns]].copy()


def get_device_demands(
    df_devices: pd.DataFrame,
    data_ids: Sequence[int],
    day: int,
    h: int,
    devices: Sequence[str],
    year: int = 2018,
    allow_missing: bool = False,
) -> np.ndarray:
    """Return the exact household-by-device matrix for one wall-clock hour."""

    device_columns = _normalise_devices(devices)
    timestamp = pd.to_datetime(f"{year}-{int(day)}", format="%Y-%j") + pd.Timedelta(
        hours=int(h)
    )
    rows = df_devices[df_devices["dt"] == timestamp]

    expected_ids = [int(house_id) for house_id in data_ids]
    present_ids = set(rows["dataid"].astype(int).tolist())
    missing_ids = [house_id for house_id in expected_ids if house_id not in present_ids]
    unexpected_ids = sorted(present_ids.difference(expected_ids))

    if missing_ids:
        if allow_missing:
            return np.zeros((len(expected_ids), len(device_columns)), dtype=float)
        raise ValueError(
            f"Missing appliance record(s) at {timestamp}: households {missing_ids}."
        )
    if unexpected_ids:
        rows = rows[rows["dataid"].isin(expected_ids)]

    if rows.duplicated(subset=["dataid"]).any():
        raise ValueError(f"Duplicate appliance rows were found at {timestamp}.")

    usage = rows.set_index("dataid").reindex(expected_ids)[device_columns]
    if usage.isna().any().any():
        raise ValueError(
            f"NaN appliance values were produced at {timestamp}; no silent filling is allowed."
        )
    return usage.to_numpy(dtype=float)


def get_device_day_demands(
    df_devices: pd.DataFrame,
    data_ids: Sequence[int],
    day: int,
    devices: Sequence[str],
    year: int = 2018,
    expected_hours: int = 24,
) -> tuple[pd.DatetimeIndex, np.ndarray]:
    """Return one complete, exactly aligned day of appliance demand."""

    device_columns = _normalise_devices(devices)
    start = pd.to_datetime(f"{year}-{int(day)}", format="%Y-%j")
    expected_index = pd.date_range(start, periods=int(expected_hours), freq="h")
    day_rows = df_devices[df_devices["dt"].isin(expected_index)].copy()

    expected_ids = [int(house_id) for house_id in data_ids]
    expected_pairs = int(expected_hours) * len(expected_ids)
    if len(day_rows) != expected_pairs:
        counts = day_rows.groupby("dataid").size().to_dict()
        raise ValueError(
            f"Day {day} must contain exactly {expected_hours} appliance rows per "
            f"household ({expected_pairs} rows total); found {len(day_rows)}. "
            f"Counts by household: {counts}"
        )

    if day_rows.duplicated(subset=["dt", "dataid"]).any():
        raise ValueError(f"Duplicate appliance rows were found on day {day}.")

    matrices = [
        get_device_demands(
            day_rows,
            expected_ids,
            day,
            hour,
            device_columns,
            year=year,
        )
        for hour in range(int(expected_hours))
    ]
    return expected_index, np.asarray(matrices, dtype=float)


if __name__ == "__main__":
    raise SystemExit(
        "This module now requires the active appliance list from config.yaml. "
        "Run the project through main.py or test.py instead."
    )