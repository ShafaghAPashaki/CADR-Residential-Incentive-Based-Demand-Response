"""Predetermined capacity-need confusion counts from frozen one-step forecasts.

No controller training, validation model selection or test-period tuning.
Rows: training, validation, July, August, November; threshold fixed at 7 kW.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
RANGES={'train':(2,167),'validation':(168,181),'july':(182,212),'august':(213,243),'november':(305,334)}
CAPACITY=7.0


def main():
    forecast=pd.read_csv(ROOT/'results/forecast/ddqn_state_forecasts.csv',parse_dates=['timestamp'])
    real=pd.read_csv(ROOT/'data/load_hourly_2018.csv')
    real['timestamp']=pd.to_datetime(real['time'],format='%d/%m/%Y %H:%M',errors='raise')
    real=real.loc[(real['timestamp'].dt.year==2018)&real['dataid'].isin([661,3039,8565])]
    actual=real.groupby('timestamp')['total'].sum()
    aligned=forecast.set_index('timestamp').join(actual.rename('actual_aggregate_kw'),how='left',validate='one_to_one')
    if aligned['actual_aggregate_kw'].isna().any():
        raise RuntimeError('Frozen forecasts are not exactly aligned with actual hours.')
    outputs={}
    for name,(start,end) in RANGES.items():
        rows=aligned.loc[(aligned.doy>=start)&(aligned.doy<=end)]
        truth=(rows.actual_aggregate_kw.to_numpy()>CAPACITY)
        prediction=(rows.forecast_aggregate_load_kw.to_numpy()>CAPACITY)
        outputs[name]={
            'hours':len(rows),
            'actual_capacity_need_hours':int(truth.sum()),
            'forecast_capacity_need_hours':int(prediction.sum()),
            'true_positives':int((truth&prediction).sum()),
            'false_negatives':int((truth&~prediction).sum()),
            'false_positives':int((~truth&prediction).sum()),
            'true_negatives':int((~truth&~prediction).sum()),
        }
    destination=ROOT/'results/final_diagnostics/forecast_capacity_confusion.json'
    destination.parent.mkdir(parents=True,exist_ok=True)
    destination.write_text(json.dumps({
        'threshold_kw': CAPACITY,
        'definition':'strict forecast and actual > 7 kW',
        'forecast_aggregate_definition':(
            'Unfloored sum of archived per-household LSTM forecasts for a '
            'diagnostic confusion table. Controller state independently floors '
            'negative household load forecasts at zero before summing. The '
            'controller feature itself is not redefined by this diagnostic.'
        ),
        'splits':outputs,
    },indent=2)+'\n',encoding='utf-8')
    print(destination)
    for k,v in outputs.items(): print(k,v)

if __name__=='__main__': main()
