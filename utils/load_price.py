"""Strict wholesale-price loading helpers."""

from __future__ import annotations

import os

import numpy as np
import pandas as pd


data_dir = "data/"
data_path = os.path.join(data_dir, "ercot_hourly_price.csv")


def load_price(data_path: str) -> pd.DataFrame:
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Wholesale-price file not found: {data_path}")

    df = pd.read_csv(data_path)
    required = ["timestamp", "Price"]
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"Wholesale-price file is missing required columns: {missing}")

    df = df[required].copy()
    df["dt"] = pd.to_datetime(df.pop("timestamp"), dayfirst=True, errors="raise")
    df["price"] = pd.to_numeric(df.pop("Price"), errors="raise") / 10.0

    if df[["dt", "price"]].isna().any().any():
        raise ValueError("NaN timestamps or prices were found in the wholesale-price file.")

    duplicate_mask = df.duplicated(subset=["dt"], keep=False)
    if duplicate_mask.any():
        examples = df.loc[duplicate_mask, ["dt", "price"]].head(10)
        raise ValueError(
            "Duplicate wholesale-price timestamps were found. "
            f"Examples:\n{examples}"
        )

    df = df.sort_values("dt").set_index("dt")
    if not df.index.is_monotonic_increasing:
        raise ValueError("Wholesale-price timestamps are not monotonic after sorting.")

    if not np.isfinite(df["price"].to_numpy(dtype=float)).all():
        raise ValueError("Non-finite wholesale prices were found.")


    df.attrs["negative_price_hours"] = int((df["price"] < 0.0).sum())
    return df