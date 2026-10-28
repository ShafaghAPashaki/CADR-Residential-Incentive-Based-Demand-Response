# Ablation, sensitivity and diagnostic experiments

All commands are run from the repository root. Obtain the authorised data and forecast-state input as described in the [main guide](../README.md). For published checkpoints, report evaluations using regenerated forecasts as approximate. The examples write to new folders rather than replacing archived outputs.

## EBDR benchmark evaluation

The published EBDR checkpoints cover seeds 0–2 and July, August and November. For example, in PowerShell, after setting `$py` as in the main guide:

```powershell
& $py -m benchmark.test_benchmark --model_path results/benchmark/seed_1/training/ddqn_best.pth --config_path benchmark/config_benchmark_seed_1.yaml --period august --output_dir repro_runs/ebdr/seed_1/august
```

Replace both instances of the seed and the period/output folder to select another run. EBDR does not itself check the checkpoint's forecast hash, so a run with regenerated forecasts is approximate even though it does not write the CADR warning file. See [benchmark/README.md](../benchmark/README.md) for the benchmark and training details.

## Matched-HEMS ablation (CBSP)

**What it tests.** Whether the capacity outcomes come from the capacity-aware service-provider objective or from the capacity-aware HEMS alone. CBSP keeps the HEMS physics, scheduling rules, settlement and data unchanged. The service provider instead learns an economic-only objective (service-provider margin and end-user utility, with no tracking or no-need term), and the two capacity-need state features are set to zero. It is **not** a fully capacity-blind household controller. The protocol was frozen before training (`docs/protocols/matched_hems_ablation_protocol.md`).

| File | Role |
|---|---|
| `experiments/matched_hems_ablation.py` | `EconomicSPEnvironment`, a subclass of the CADR environment |
| `experiments/check_matched_hems_ablation.py` | Pre-training gates: state mask, hand-computed reward, identical physics under matched actions |
| `experiments/run_matched_hems_ablation.py` | Training (`--dry-run`, `--smoke`, `--seed 0/1/2`) |
| `experiments/evaluate_matched_hems_ablation.py` | July or August evaluation with the unmodified `test.py` |
| `results/matched_hems_ablation/` | Checkpoints, validation metrics and monthly summaries for seeds 0–2 |

To evaluate a published checkpoint, first copy its `training/` folder into a new output root, then use the CADR forecast wrapper. For example, in PowerShell:

```powershell
New-Item -ItemType Directory -Path repro_runs/cbsp/seed_1 | Out-Null
Copy-Item results/matched_hems_ablation/seed_1/training repro_runs/cbsp/seed_1/training -Recurse
& $py tools/evaluate_with_regenerated_forecasts.py experiments/evaluate_matched_hems_ablation.py --seed 1 --month august --root repro_runs/cbsp
```

Choose seeds 0, 1 or 2 and `july` or `august`; use a new root for each independent rerun. The copied checkpoint must match the seed specified to the evaluator. For training or pre-training checks, see the drivers listed above; the historical training path expects the original forecast-state file and refuses to overwrite the released training folder.

## Capacity-target and incentive-discretisation sensitivity

**What it tests.** CADR Seed 0 retrained with a 6-kW or 8-kW capacity target (`cap6`, `cap8`), or with ten incentive levels per household instead of five (`lev10`, 1,000 joint actions). Everything else is unchanged, including λc, the data splits, the 2,500 episodes and validation-only checkpoint selection. The predeclared protocol identifier `R2-SENS-2026-10-14` is recorded in each run configuration.

| File | Role |
|---|---|
| `experiments/capacity_incentive_sensitivity.py` | Run definitions, configuration builder, ten-level action grid, capacity gate |
| `experiments/run_capacity_incentive_sensitivity.py` | Training (`--run cap6 / cap8 / lev10`, `--dry-run`) |
| `experiments/evaluate_capacity_incentive_sensitivity.py` | July evaluation (`--run cap6 / cap8 / lev10`, or `ref7` for the published 7-kW Seed-0 model) |
| `tools/summarize_capacity_incentive_sensitivity.py` | Recomputes the July summaries from the private hourly workbooks |
| `results/capacity_incentive_sensitivity/` | Checkpoints, validation metrics, timing and July summaries |

```powershell
& $py tools/evaluate_with_regenerated_forecasts.py experiments/evaluate_capacity_incentive_sensitivity.py --run cap6
```

Replace `cap6` with `cap8`, `lev10`, or `ref7`. New evaluations are written to `repro_runs/sensitivity/<run>/july/`; choose a new `--output-root` for a repeat. After evaluating all three sensitivity runs and CADR Seed 0 for July, `& $py tools/summarize_capacity_incentive_sensitivity.py` reads the new audit workbooks. Historical sensitivity retraining uses a fixed `results/` location and refuses to reuse stored runs; run it only in a separate working copy.

## Reward-parameter sensitivity (η, ρ)

**What it tests.** Single-seed retraining with the no-need offer coefficient η ∈ {0, 0.01429, 0.03571, 0.07143} or the end-user weight ρ ∈ {0.1, 0.3, 0.5, 0.7, 0.9}, with all other settings unchanged.

```
python experiments/run_reward_sensitivity.py --group both --dry-run
python experiments/run_reward_sensitivity.py --group eta
```

The archived summaries are in `results/sensitivity_eta/summary.csv` and `results/sensitivity_rho/summary.csv`. New runs and summaries are written under `repro_runs/sensitivity_{eta,rho}/`; the ρ run configurations are recorded in `configs/rho_sensitivity/`. The per-run checkpoints are not distributed. The two sensitivity diagnostics require newly trained runs and write to `repro_runs/diagnostics/sensitivity_{eta,rho}/`.

## Diagnostics

| Script | Output | Purpose |
|---|---|---|
| `diagnostics/evaluate_value_of_information.py --method cadr --seed 0 --period july` | `repro_runs/diagnostics/{cadr,ebdr}/seed_*/value_of_information_*.csv` | Inference-only comparison of forecast and oracle state inputs; no retraining or reselection |
| `diagnostics/evaluate_forecast_confusion.py` | `repro_runs/diagnostics/forecast_capacity_confusion.json` | Fixed 7-kW confusion counts for the active forecast inputs |
| `diagnostics/verify_retained_lambda.py` | console | Checks the retained λc, configurations and frozen hashes |
| `diagnostics/measure_inference_time.py` | `local_validation/inference_timing_seed0.json` | CPU timing of Seed-0 inference |
| `q_action_diagnostic.py` | configurable | Q-value ordering audit |
| `reward_counterfactual_audit.py` | (module) | Counterfactual reward evaluation used by the Q-value audit |
| `summarize_results.py` | `repro_runs/summary.xlsx` | Three-seed mean and standard deviation summary |
| `calibrate_lambda_c.py` | `results/calibration_diagnostics/` | λc calibration sensitivity; never changes the frozen configurations |

`run_inference_only_oracle_checks.ps1 -Python $py` runs the value-of-information evaluation for both methods, all seeds and all periods. Results produced with regenerated forecasts inherit the reproducibility limitation described in the main guide.
