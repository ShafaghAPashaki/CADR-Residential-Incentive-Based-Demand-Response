"""Validate a study-aligned hourly ERCOT price export (USD/MWh).
Not a conversion from arbitrary EnergyOnline price-zone exports.
python data/prepare_price.py --input downloads/prices.csv --output data/ercot_hourly_price.csv
"""
import argparse
import hashlib
import json
import shutil
from pathlib import Path
import numpy as np
import pandas as pd

def sha256(path):
    h=hashlib.sha256()
    with path.open("rb") as f:
        for part in iter(lambda:f.read(1024*1024),b""):h.update(part)
    return h.hexdigest()

def prepare_price(df,year=2018):
    if not {"timestamp","Price"}.issubset(df.columns):
        raise ValueError("Expected timestamp,Price; price zone/extraction not automatically selected")
    out=df[["timestamp","Price"]].copy()
    stamps=pd.to_datetime(out["timestamp"],dayfirst=True,errors="raise")
    if stamps.isna().any() or stamps.duplicated().any():
        raise ValueError("Missing or duplicate price timestamps")
    if not ((stamps.dt.minute==0)&(stamps.dt.second==0)).all():
        raise ValueError("Prices must be on the hour")
    out["Price"]=pd.to_numeric(out["Price"],errors="raise")
    if not np.isfinite(out["Price"].to_numpy(float)).all():
        raise ValueError("Non-finite prices")
    expected=pd.date_range(f"{year}-01-01",f"{year}-12-31 23:00",freq="h")
    if len(expected.difference(pd.DatetimeIndex(stamps))):
        raise ValueError("Hourly price coverage incomplete")
    if (stamps<expected[0]).any() or (stamps>expected[-1]+pd.Timedelta(hours=1)).any():
        raise ValueError("Price series outside expected annual range")
    out["dt"]=stamps
    out=out.sort_values("dt")
    out["timestamp"]=out["dt"].dt.strftime("%d/%m/%Y %H:%M")
    return out[["timestamp","Price"]].reset_index(drop=True)

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input",type=Path,required=True)
    ap.add_argument("--output",type=Path,required=True)
    ap.add_argument("--year",type=int,default=2018)
    ap.add_argument("--overwrite",action="store_true")
    a=ap.parse_args()
    if a.input.resolve()==a.output.resolve():ap.error("Input/output must differ")
    if a.output.exists() and not a.overwrite:ap.error("Output exists; use --overwrite deliberately")
    original=pd.read_csv(a.input)
    result=prepare_price(original,a.year)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    dates=pd.to_datetime(original["timestamp"],dayfirst=True)
    if list(original.columns)==["timestamp","Price"] and dates.is_monotonic_increasing:
        shutil.copyfile(a.input,a.output)
    else:result.to_csv(a.output,index=False)
    print(json.dumps({"status":"READY","rows":len(result),"negative_price_hours":int((result.Price<0).sum()),"source_sha256":sha256(a.input),"output_sha256":sha256(a.output)},indent=2))
if __name__=="__main__":main()
