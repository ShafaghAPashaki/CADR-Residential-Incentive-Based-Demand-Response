"""Prepare the historical CADR hourly ERCOT price input from an EnergyOnline export.

Supports raw EnergyOnline Date,Price,Zone (US AM/PM) or already-processed
timestamp,Price (day-first, USD/MWh). The historical series is LZ_AEN.
Three missing 2018 hours are forward-filled; the optional 2019-01-01 00:00
boundary must come from a separate, authorised 2019 export.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import shutil
from pathlib import Path
import numpy as np
import pandas as pd

HISTORICAL_ZONE = "LZ_AEN"
EXPECTED_MISSING_2018 = pd.DatetimeIndex([
    "2018-03-11 02:00:00", "2018-05-22 05:00:00", "2018-06-04 05:00:00"])
OUTPUT_COLUMNS = ["timestamp", "Price"]

def file_sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b""):
            h.update(chunk)
    return h.hexdigest()

def _raw_zone_data(df, zone):
    if not {"Date", "Price", "Zone"}.issubset(df.columns):
        raise ValueError("Raw EnergyOnline export must have Date,Price,Zone columns")
    sub = df.loc[df["Zone"].astype(str).str.strip() == zone, ["Date","Price"]].copy()
    if sub.empty:
        raise ValueError(f"Zone {zone} missing; zones: {sorted(df.Zone.dropna().unique())}")
    sub["dt"] = pd.to_datetime(sub.pop("Date"), format="%m/%d/%Y %I:%M:%S %p", errors="raise")
    sub["Price"] = pd.to_numeric(sub["Price"], errors="raise")
    if sub.dt.isna().any() or not np.isfinite(sub.Price.to_numpy(float)).all():
        raise ValueError("Invalid dates or prices")
    if sub.dt.duplicated().any() or (sub.dt.dt.minute != 0).any():
        raise ValueError("Duplicate or non-hourly records")
    return sub.set_index("dt").sort_index()[["Price"]]

def prepare_price(df, year=2018, zone=None, boundary_df=None):
    expected = pd.date_range(f"{year}-01-01 00:00", f"{year}-12-31 23:00", freq="h")
    if {"Date", "Price", "Zone"}.issubset(df.columns):
        if zone is None: raise ValueError("Raw EnergyOnline input requires --zone LZ_AEN")
        if year == 2018 and zone != HISTORICAL_ZONE:
            raise ValueError("Historical CADR zone must be LZ_AEN")
        raw = _raw_zone_data(df, zone)
        prices = raw.loc[(raw.index>=expected[0])&(raw.index<=expected[-1]),"Price"]
        missing = expected.difference(prices.index)
        allowed = EXPECTED_MISSING_2018 if year==2018 else pd.DatetimeIndex([])
        if not missing.isin(allowed).all():
            raise ValueError(f"Unexpected missing price hours: {missing.tolist()[:12]}")
        series = prices.reindex(expected).ffill()
        if series.isna().any():raise ValueError("Missing starting price")
        if boundary_df is not None:
            boundary = expected[-1] + pd.Timedelta(hours=1)
            ext = _raw_zone_data(boundary_df, zone)
            if boundary not in ext.index:
                raise ValueError(f"Boundary download lacks {boundary} for {zone}")
            series.loc[boundary] = float(ext.loc[boundary,"Price"])
    elif {"timestamp","Price"}.issubset(df.columns):
        if zone is not None or boundary_df is not None:
            raise ValueError("--zone and --boundary-file only apply to raw downloads")
        dates = pd.to_datetime(df["timestamp"],format="%d/%m/%Y %H:%M",errors="raise")
        prices = pd.to_numeric(df["Price"], errors="raise")
        if dates.isna().any() or dates.duplicated().any() or not np.isfinite(prices.to_numpy(float)).all():
            raise ValueError("Invalid or duplicate processed price values")
        series = pd.Series(prices.to_numpy(float),index=pd.DatetimeIndex(dates)).sort_index()
        if not expected.isin(series.index).all():
            raise ValueError("Processed prices have missing 2018 hours")
        extra = series.index.difference(expected)
        boundary=expected[-1]+pd.Timedelta(hours=1)
        if len(extra)>0 and not extra.equals(pd.DatetimeIndex([boundary])):
            raise ValueError("Unexpected extra processed price hours")
    else:raise ValueError("Expected raw Date,Price,Zone or processed timestamp,Price")
    return pd.DataFrame({"timestamp":series.index.strftime("%d/%m/%Y %H:%M"),
                         "Price":series.to_numpy(float)})[OUTPUT_COLUMNS]

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input",type=Path,required=True)
    ap.add_argument("--output",type=Path,required=True)
    ap.add_argument("--zone",default=None)
    ap.add_argument("--boundary-file",type=Path)
    ap.add_argument("--reference",type=Path)
    ap.add_argument("--year",type=int,default=2018)
    ap.add_argument("--overwrite",action="store_true")
    a=ap.parse_args()
    if a.input.resolve()==a.output.resolve():ap.error("Input/output must differ")
    if a.output.exists() and not a.overwrite:ap.error("Output already exists; use --overwrite")
    src=pd.read_csv(a.input)
    result=prepare_price(src,year=a.year,zone=a.zone,
                         boundary_df=pd.read_csv(a.boundary_file) if a.boundary_file else None)
    stats={"status":"READY","rows":len(result),"price_unit":"USD/MWh",
           "zone":a.zone or "processed","source_sha256":file_sha256(a.input)}
    if a.reference:
        ref=pd.read_csv(a.reference)
        actual=result.set_index("timestamp")["Price"]
        expected=ref.set_index("timestamp")["Price"]
        common=actual.index.intersection(expected.index)
        if not len(common) or ((actual.loc[common]-expected.loc[common]).abs()>1e-10).any():
            raise SystemExit("Reference values do not match")
        stats["reference_overlap_hours"]=len(common)
        stats["reference_missing_hours"]=len(expected)-len(common)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    if list(src.columns)==OUTPUT_COLUMNS and src[OUTPUT_COLUMNS].equals(result):
        shutil.copyfile(a.input,a.output)
    else:result.to_csv(a.output,index=False)
    stats["output_sha256"]=file_sha256(a.output)
    print(json.dumps(stats,indent=2))

if __name__=="__main__":main()
