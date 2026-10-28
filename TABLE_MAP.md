# Manuscript tables and figures: where each number comes from

**Data file.** `results/aggregate_metrics/per_seed_monthly_metrics.csv` holds the seed-level monthly values behind the main tables. It has one row per (method, seed, month, metric). Every table reports the mean and sample standard deviation (ddof = 1) over seeds 0, 1 and 2.

**Regenerating it.** The file was produced by `tools/export_aggregate_metrics.py` from evaluation workbooks (`audit.xlsx`, `results.xlsx`). Detailed audits contain licensed household data and are not distributed. Recreate the CADR, CBSP, EBDR and sensitivity evaluations with licensed inputs, then point the exporter to their new roots. Choose a new output path: the exporter's default overwrites the archived CSV.

```
python tools/export_aggregate_metrics.py --cadr-root repro_runs/eval/published --cbsp-root repro_runs/cbsp --ebdr-root repro_runs/ebdr --sensitivity-root repro_runs/sensitivity --out repro_runs/per_seed_monthly_metrics.csv
```

**Metric conventions**
- An hour counts as a capacity exceedance when the aggregate load exceeds the target by more than 1e-6 kW.
- Mean daily peak and PAR are computed per day and then averaged over the month.

## Main text

| Item | Metrics in the CSV | Method rows | Produced by |
|---|---|---|---|
| Table 3, operational performance | `baseline_*`, `exceedance_hours`, `cumulative_excess_kWh`, `mean_load_kW`, `mean_daily_peak_kW`, `daily_PAR`, `useful_DR_kWh` | CADR | `test.py` via `test_all_seeds.ps1` |
| §5.1 percentage reductions | ratios of Table 3 means to the baselines | CADR | as for Table 3 |
| Table 4, appliance decomposition | `AC_curtailment_kWh`; `{EV,WM,DW,Dryer}_shifted_{out,in}_kWh`; `unserved_EV_kWh` | CADR | `test.py` (`Table_2_Appliances` sheet) |
| §5.2, new and rebound-induced violations | `newly_created_violation_hours`, `rebound_induced_violation_hours` | CADR | `test.py` |
| Table 5, economics (per day = monthly total ÷ `days`) | `EU{1,2,3}_settled_DR_kWh`, `EU{n}_incentive_payment_cent`, `EU{n}_utility_cent`, `SP_wholesale_value_cent`, `incentive_payment_cent`, `SP_economic_value_cent` | CADR | `test.py` |
| Table 6, participation | `EU{n}_DR_share_percent`, `EU{n}_participation_hours_percent` | CADR | `test.py` |
| Table 7, EBDR comparison | `exceedance_hours`, `cumulative_excess_kWh`, `mean_daily_peak_kW`, `daily_PAR`, `useful_DR_kWh`, `net_curtailment_kWh` | EBDR, CADR | `benchmark/test_benchmark.py`, `test.py` |
| Table 8, matched-HEMS ablation | Table 7 metrics plus `incentive_offer_hours`, `no_need_offer_hours`, `incentive_payment_cent` | CBSP, CADR | `experiments/evaluate_matched_hems_ablation.py`, `test.py` |
| §5.1, forecast vs. oracle inputs | — | — | `results/final_diagnostics/{cadr,ebdr}/seed_*/value_of_information_*.csv` (`diagnostics/evaluate_value_of_information.py`) |
| §5.1, missed overload hours | — | — | `results/final_diagnostics/forecast_capacity_confusion.json` (`diagnostics/evaluate_forecast_confusion.py`) |
| Fig. 3, validation reward | — | — | Data: `results/seed_*/training/validation_metrics.csv` (`mean_reward`); selected episodes in `results/checkpoint_selection.csv`. No plotting script is included. |
| Fig. 4, aggregate load on 27 July (Seed 0) | — | — | The CADR and EBDR series come from the hourly evaluation workbooks for CADR seed 0 and EBDR seed 0 (July, day 208). `test.py` and `benchmark/test_benchmark.py` plot each method separately; no script for the combined figure is included. |
| Fig. 5, device schedule on 27 July (Seed 0) | — | — | `figures/make_fig5_device_schedule.py` |

## Appendices

| Item | Source | Produced by |
|---|---|---|
| Table A.10, LSTM MAE | `results/forecast/forecast_metrics.csv` | `build_forecast_outputs.py` |
| Table B.11, normalisation constants | `reward_economic_scale` and `reward_discomfort_class_scales` stored in every checkpoint; printed at environment start-up | `env/env.py` |
| Table C.12, dissatisfaction coefficients | `environment.fixed_household_coefficients` in `config_seed_*.yaml` | — |
| Appendix D, EBDR checkpoints | `results/checkpoint_selection.csv` | — |
| Tables E.13 and E.14, reward sensitivity | `results/sensitivity_{eta,rho}/summary.csv`; 20-state column in `results/sensitivity_eta/diagnostics/eta_audit_summary.csv` | `experiments/run_reward_sensitivity.py`; `diagnostics/eta_sensitivity_diagnostic.py` |
| Table G.15, seed-wise performance | CSV rows for method CADR, per seed | `test.py` |
| Table H.16, capacity sensitivity | CSV rows for methods `CADR_cap6` and `CADR_cap8`; the 7-kW column is CADR seed 0, July | `experiments/evaluate_capacity_incentive_sensitivity.py --run cap6 \| cap8` |
| Table I.17, ten incentive levels | CSV rows for method `CADR_lev10`; the five-level column is CADR seed 0, July; parameter counts printed by `experiments/run_capacity_incentive_sensitivity.py` | `experiments/evaluate_capacity_incentive_sensitivity.py --run lev10` |

November outputs under `results/*/november/` are retained for completeness only. November is not part of the revised manuscript.
