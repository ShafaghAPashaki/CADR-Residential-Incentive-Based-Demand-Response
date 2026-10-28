#!/usr/bin/env python3
"""Recompute the July capacity-target and incentive-discretisation summaries from the
private Hourly_Audit workbooks. Requires only the Python standard library.
Run from the project root after evaluating the runs (see TABLE_MAP.md).
DO NOT publish the audit workbooks, which include licensed household data.
"""
import argparse, sys, zipfile, xml.etree.ElementTree as ET, re, pathlib
M='{http://schemas.openxmlformats.org/spreadsheetml/2006/main}'

def hourly_rows(path):
    with zipfile.ZipFile(path) as z:
        # All supplied CADR workbooks store Hourly_Audit as third sheet.
        root=ET.fromstring(z.read('xl/worksheets/sheet3.xml'))
        rows=root.find(M+'sheetData')
        headers=None
        for rr in rows:
            cells={}
            for c in rr:
                ref=c.get('r',''); col=re.match(r'[A-Z]+',ref)
                if not col: continue
                label=col.group(); idx=0
                for ch in label: idx=idx*26+ord(ch)-64
                raw=c.find(M+'v')
                value=raw.text if raw is not None else ''.join((t.text or '') for t in c.iter(M+'t'))
                cells[idx-1]=value
            if headers is None:
                headers={str(v):i for i,v in cells.items()}; continue
            def get(n):
                x=cells.get(headers[n], '0')
                return float(x) if x not in ('',None) else 0.0
            yield {name:get(name) for name in (
              'capacity','baseline_total','after_total','baseline_capacity_need',
              'useful_settlement_DR_energy','settlement_DR_energy',
              'excess_settlement_DR_energy','incentive_payment',
              'SP_settlement_value','weighted_customer_utility',
              'unnecessary_incentive','baseline_capacity_violation',
              'capacity_violation','remaining_capacity_violation',
              'newly_created_capacity_violation','raw_action_house_661',
              'raw_action_house_3039','raw_action_house_8565',
              'incentive_house_661','incentive_house_3039','incentive_house_8565')}

def summarise(rows):
    assert len(rows)==744, f'Expected July 744 hourly rows, got {len(rows)}'
    cap=rows[0]['capacity']
    assert all(abs(r['capacity']-cap)<1e-8 for r in rows)
    tol=1e-6
    baseline=sum(r['baseline_total']>cap+tol for r in rows)
    post=sum(r['after_total']>cap+tol for r in rows)
    remaining=sum(r['baseline_total']>cap+tol and r['after_total']>cap+tol for r in rows)
    new=sum(r['baseline_total']<=cap+tol and r['after_total']>cap+tol for r in rows)
    need=sum(max(0,r['baseline_total']-cap) for r in rows)
    excess=sum(max(0,r['after_total']-cap) for r in rows)
    useful=sum(r['useful_settlement_DR_energy'] for r in rows)
    offered=sum(any(r[k]>1e-10 for k in ['incentive_house_661','incentive_house_3039','incentive_house_8565']) for r in rows)
    return dict(capacity_kw=cap,baseline_h=baseline,post_h=post,remaining_h=remaining,
      new_h=new,exceedance_reduction_pct=(baseline-post)*100/baseline,
      baseline_excess_kwh=need,post_excess_kwh=excess,
      cumulative_excess_reduction_pct=(need-excess)*100/need,
      useful_dr_kwh=useful,useful_share_baseline_need_pct=100*useful/need,
      settled_dr_kwh=sum(r['settlement_DR_energy'] for r in rows),
      excess_dr_kwh=sum(r['excess_settlement_DR_energy'] for r in rows),
      offer_hours=offered,unnecessary_offer_hours=sum(r['unnecessary_incentive']>0.5 for r in rows),
      incentive_payment_cent=sum(r['incentive_payment'] for r in rows),
      sp_value_cent=sum(r['SP_settlement_value'] for r in rows),
      eu_utility_cent=sum(r['weighted_customer_utility'] for r in rows))

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cadr-root',default='repro_runs/eval/published')
    parser.add_argument('--sensitivity-root',default='repro_runs/sensitivity')
    args=parser.parse_args()
    cadr=pathlib.Path(args.cadr_root)
    sensitivity=pathlib.Path(args.sensitivity_root)
    cases={
      'H: 6 kW':sensitivity/'cap6/july/audit.xlsx',
      'H: 7 kW / I: 5 levels':cadr/'seed_0/july/audit.xlsx',
      'H: 8 kW':sensitivity/'cap8/july/audit.xlsx',
      'I: 10 levels':sensitivity/'lev10/july/audit.xlsx',
    }
    for name,p in cases.items():
        if not p.exists():
            print(f'MISSING {name}: {p}',file=sys.stderr)
            return 1
        m=summarise(list(hourly_rows(p)))
        print(f'\n{name} — {p}')
        for k,v in m.items():print(f'  {k}: {v:.6f}' if isinstance(v,float) else f'  {k}: {v}')
    print('\nNote: checkpoint selections and training durations are checked from separate training records.')
    return 0
if __name__=='__main__':sys.exit(main())
