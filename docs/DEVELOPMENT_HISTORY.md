# Development history and provenance

The labels below identify the frozen implementation recorded in checkpoints and run metadata. They are not installation steps; do not rename them in code or checkpoints.

| Label | Meaning |
|---|---|
| `causal_state_v1` | State features no longer use realised appliance observations after the decision hour. |
| `causal_capacity_v1`–`causal_capacity_v3` | Successive causal scheduling, conflict deferral, charging limits and unmet-energy accounting changes. |
| `causal_price_v1` | Request admissibility uses the price known at the request hour; locked-rate settlement is retained. |
| `forecast_state_v1` | One-step LSTM forecasts are used in the CADR decision state. |
| `benchmark_fair_forecast_v1` | EBDR uses the same forecast inputs as CADR and a previous-day price reference. |
| `final_preflight_retained_lambda_v1` | Training safeguards retain the historical λc. |
| `public_release_v4` | Public packaging and reorganisation of the research release. |

The release groups experiment drivers in `experiments/`, diagnostics in `diagnostics/`, figures in `figures/` and supplementary notes in `docs/`. Historical results were moved to `results/matched_hems_ablation/` and `results/capacity_incentive_sensitivity/`. Their original contents, recorded paths, script names and hashes remain unchanged; the former and current names are recorded in [FILE_RENAME_MAP.csv](FILE_RENAME_MAP.csv). The frozen `R2-SENS-2026-10-14` identifier and `r2_sensitivity` configuration key are retained.

The original forecast-source snapshots and their manifests are in `forecast_regeneration/`. See [REPRODUCIBILITY.md](../REPRODUCIBILITY.md) for the verification and limits of regenerating the inputs. The repository's Git history retains earlier versions of this document.
