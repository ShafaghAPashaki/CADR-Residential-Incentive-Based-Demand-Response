# Results

The trained checkpoints, validation curves, monthly evaluation summaries and provenance records behind the manuscript. `TABLE_MAP.md` maps each table and figure to a file here. Hourly evaluation workbooks (`audit.xlsx`) and household-level forecast series contain licensed data and are not distributed.

| Folder or file | Contents |
|---|---|
| `seed_0/`, `seed_1/`, `seed_2/` | CADR: `training/` holds the selected checkpoint `ddqn_best.pth`, its `config.yaml`, `validation_metrics.csv` and `run_provenance.json`; `july/`, `august/` and `november/` hold the monthly evaluation summaries |
| `benchmark/` | EBDR, organised in the same way |
| `matched_hems_ablation/` | CBSP matched-HEMS ablation, seeds 0–2, July and August |
| `capacity_incentive_sensitivity/` | Seed-0 runs `cap6`, `cap8` (capacity target 6 and 8 kW) and `lev10` (ten incentive levels), each evaluated on July |
| `sensitivity_eta/`, `sensitivity_rho/` | Reward-parameter sensitivity summaries and diagnostics |
| `final_diagnostics/` | Inference-only value-of-information results and forecast-confusion counts |
| `forecast/` | Forecast metrics, figures, training histories, cross-fit records and forecast-state provenance |
| `calibration/` | The λc retention audit checked before every final training run |
| `aggregate_metrics/per_seed_monthly_metrics.csv` | Seed-level monthly values behind the manuscript tables |
| `checkpoint_selection.csv` | Validation-selected episodes of the CADR and EBDR checkpoints (each run's `validation_metrics.csv` covers the others) |
| `summary.xlsx`, `method_comparison.xlsx` | Three-seed summary and CADR–EBDR comparison workbooks |

November outputs are retained for completeness. November is not part of the revised manuscript.

**Former folder names.** `matched_hems_ablation/` and `capacity_incentive_sensitivity/` were created as `r1c3_r2c2_ablation/` and `r2_sensitivity/`. The files inside them were not edited, so their recorded configurations, logs and timing files still contain those original paths and the original script names (see `docs/DEVELOPMENT_HISTORY.md`). Evaluation does not use these recorded paths.

The `training/config.yaml` directly under each sensitivity run folder is the configuration written at launch. The copy in `seed_0/training/` sits next to the checkpoint and is the one used for evaluation; the two are identical.
