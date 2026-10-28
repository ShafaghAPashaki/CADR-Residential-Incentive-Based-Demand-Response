"""Prepare hourly CADR loads from an authorised Pecan Street 15-minute CSV.

Reimplements the appliance transformations described by MARL-iDR
(Tilburg, Siebert & Cremer, IEEE PowerTech 2023; MIT-licensed source).
The documented 15-minute rules followed by hourly arithmetic averaging
were verified locally against all 218,835 historical 2018 household-hours.
Actual Dataport readings must not be redistributed without authorisation.

python data/prepare_load.py --input downloads/15minute_data_austin.csv --output data/load_hourly_2018.csv --year 2018
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

AIR = ("air1", "air2", "air3", "airwindowunit1")
CAR = ("car1", "car2")
DRY = ("drye1", "dryg1")
NON_SHIFTABLE = (
    "bathroom1", "bathroom2", "bedroom1", "bedroom2", "bedroom3",
    "diningroom1", "disposal1", "furnace1", "furnace2", "garage1",
    "kitchen1", "kitchen2", "kitchenapp1", "kitchenapp2",
    "lights_plugs1", "lights_plugs2", "lights_plugs3", "lights_plugs4",
    "livingroom1", "microwave1", "office1", "oven1", "oven2",
    "range1", "refrigerator1", "refrigerator2", "utilityroom1",
    "venthood1", "waterheater1", "waterheater2", "winecooler1",
)
RAW = tuple(dict.fromkeys((*AIR, *CAR, *DRY, "dishwasher1", "clotheswasher1", *NON_SHIFTABLE)))
HOURLY = ("air", "car", "clotheswasher", "dishwasher", "dry", "non-shiftable", "total")
OUTPUT_COLUMNS = ("dataid", "time", *HOURLY)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def prepare(raw: pd.DataFrame, year: int = 2018) -> pd.DataFrame:
    """Apply historical 15-min rules and arithmetic hourly aggregation in UTC."""
    missing = sorted(set(("dataid", "local_15min", *RAW)).difference(raw.columns))
    if missing:
        raise ValueError(f"Wrong Dataport export: missing required columns: {missing}")
    if raw.empty:
        raise ValueError("Input data is empty")
    stamps = raw["local_15min"].astype(str)
    if not stamps.str.contains(r"(?:[+-]\d{2}(?::?\d{2})?|Z)$", regex=True).all():
        raise ValueError("local_15min must include a UTC offset, e.g. -05 or -06")
    work = raw.loc[:, ["dataid", "local_15min", *RAW]].copy()
    work["dataid"] = pd.to_numeric(work["dataid"], errors="raise").astype(int)
    work["time"] = pd.to_datetime(work.pop("local_15min"), utc=True, errors="raise").dt.tz_localize(None)
    for col in RAW:
        work[col] = pd.to_numeric(work[col], errors="raise")
    if work["time"].isna().any():
        raise ValueError("Invalid raw Dataport timestamps")
    frames = []
    for house_id, house in work.groupby("dataid", sort=True):
        source = house.drop(columns="dataid").set_index("time").sort_index()
        quarter = source.resample("15min").max().fillna(0.0)
        quarter = quarter.mask(quarter < 0.1, 0.0)
        out = pd.DataFrame(index=quarter.index)
        out["air"] = quarter[list(AIR)].sum(axis=1).clip(upper=4.0)
        out["car"] = (quarter[list(CAR)].sum(axis=1) >= 0.1).astype(float) * 4.0
        out["clotheswasher"] = (quarter["clotheswasher1"] >= 0.1).astype(float)
        out["dishwasher"] = (quarter["dishwasher1"] >= 0.1).astype(float) * 2.0
        out["dry"] = (quarter[list(DRY)].sum(axis=1) >= 0.1).astype(float) * 2.0
        out["non-shiftable"] = quarter[list(NON_SHIFTABLE)].sum(axis=1).clip(upper=5.0)
        out["total"] = out[list(HOURLY[:-1])].sum(axis=1)
        hourly = out.resample("1h").mean()
        hourly = hourly[hourly.index.year == year].copy()
        hourly.insert(0, "time", hourly.index.strftime("%d/%m/%Y %H:%M"))
        hourly.insert(0, "dataid", int(house_id))
        frames.append(hourly.reset_index(drop=True))
    combined = pd.concat(frames, ignore_index=True)[list(OUTPUT_COLUMNS)]
    if not np.isfinite(combined[list(HOURLY)].to_numpy(float)).all():
        raise ValueError("Generated non-finite hourly demand")
    if not np.allclose(combined[list(HOURLY[:-1])].sum(axis=1), combined["total"], atol=1e-10, rtol=0):
        raise ValueError("Generated total differs from sum of components")
    return combined


def compare(generated: pd.DataFrame, reference_file: Path) -> dict:
    reference = pd.read_csv(reference_file)
    if set(OUTPUT_COLUMNS).difference(reference.columns):
        raise ValueError("Reference file does not have the expected hourly schema")
    keys = ["dataid", "time"]
    a = generated.set_index(keys).sort_index()
    b = reference.set_index(keys).sort_index()
    if not a.index.equals(b.index):
        raise ValueError("Reference and generated household-hour keys differ")
    errors = np.abs(a[list(HOURLY)].to_numpy(float) - b[list(HOURLY)].to_numpy(float))
    return {"reference_rows": len(b), "max_abs_error": float(errors.max()),
            "mismatching_cells_above_1e_9": int((errors > 1e-9).sum())}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--year", type=int, default=2018)
    ap.add_argument("--reference", type=Path, default=None,
                    help="Optional locally held historical hourly file for comparison")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()
    if args.output.exists() and not args.overwrite:
        ap.error(f"Output exists: {args.output}; use --overwrite deliberately")
    if args.input.resolve() == args.output.resolve():
        ap.error("Input and output must be different")
    needed = ["dataid", "local_15min", *RAW]
    header = pd.read_csv(args.input, nrows=0).columns
    if any(x not in header for x in needed):
        ap.error("Wrong Dataport raw columns: see data/examples/raw_15min_schema.csv")
    result = prepare(pd.read_csv(args.input, usecols=needed, low_memory=False), year=args.year)
    comparison = compare(result, args.reference) if args.reference else None
    if comparison and comparison["mismatching_cells_above_1e_9"]:
        raise SystemExit("Reference comparison FAILED: " + json.dumps(comparison))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False)
    print(json.dumps({"status": "READY", "rows": len(result),
                      "households": result.dataid.nunique(), "year_utc": args.year,
                      "raw_sha256": sha256(args.input), "output_sha256": sha256(args.output),
                      "reference_comparison": comparison}, indent=2))


if __name__ == "__main__":
    main()
