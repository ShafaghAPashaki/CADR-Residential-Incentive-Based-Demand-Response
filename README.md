# CADR: capacity-aware residential incentive-based demand response

This repository contains the research code, frozen configurations, trained checkpoints and aggregate results for the manuscript *Capacity-Aware Residential Incentive-Based Demand Response through Appliance-Level Load Coordination*.

## The problem

A service provider (SP) offers hourly incentives to a group of residential end users (EUs) so that their aggregate demand stays below a local capacity target, while balancing its own economic value against the users' payments and dissatisfaction. CADR casts the SP's hourly incentive decision as a Double Deep Q-Network (DDQN) problem. Each household runs an appliance-level home energy management system (HEMS) that responds to the incentive by curtailing air conditioning or shifting EV charging, washing machine, dishwasher and dryer demand. The DDQN state uses one-step LSTM forecasts of household load and wholesale price, and it contains no realised future information.

The study uses three Pecan Street households (Dataport IDs 661, 3039 and 8565) and ERCOT wholesale prices for 2018. Days 2–167 are used for training, days 168–181 for validation, and July and August 2018 as held-out test months. The capacity target is 7 kW. It is a soft operational target, not an enforced network constraint, and residual exceedances, rebound, unserved energy and other side effects are reported.

## Methods in this repository

| Method | What it is | Code |
|---|---|---|
| **CADR** | The proposed method: a capacity-aware SP DDQN (43-element state, 125 joint incentive actions) coordinating appliance-level HEMS responses | `main.py`, `test.py`, `env/`, `agent/` |
| **EBDR** | Elasticity-based benchmark: a capacity-blind, curtailment-only DDQN using price-elasticity responses and the same forecast inputs | `benchmark/` |
| **CBSP** | Matched-HEMS ablation: the same capacity-aware HEMS, but the SP learns an economic-only objective and does not observe the two capacity-need features | `experiments/*matched_hems_ablation*.py` |

Sensitivity experiments vary the capacity target (6 and 8 kW), the number of incentive levels (10 instead of 5), and the reward parameters η and ρ.

## Repository layout

| Path | Contents |
|---|---|
| `main.py`, `test.py` | CADR training and evaluation entry points |
| `env/`, `agent/`, `utils/` | CADR environment and HEMS, DDQN agent, data loaders and frozen-run safeguards |
| `config_seed_{0,1,2}.yaml` | Frozen CADR configurations, identical except for the seed |
| `benchmark/` | EBDR implementation, configurations and checks (see `benchmark/README.md`) |
| `experiments/` | Matched-HEMS ablation, capacity-target and incentive-discretisation sensitivity, and reward-parameter sensitivity |
| `configs/` | Configurations for the ρ sensitivity runs |
| `predict_*_lstm.py`, `build_forecast_outputs.py`, `build_ddqn_forecast_inputs.py` | LSTM forecasting pipeline |
| `forecast_regeneration/` | Exact forecasting and cross-fitting sources used for the published forecasts, with regeneration instructions |
| `diagnostics/` | Value-of-information, forecast-confusion, λc-retention, sensitivity and inference-time diagnostics |
| `q_action_diagnostic.py`, `reward_counterfactual_audit.py`, `summarize_results.py` | Q-value audit and three-seed summary used by the standard evaluation run |
| `calibrate_lambda_c.py` | λc calibration diagnostic (never changes the frozen configurations) |
| `figures/` | Figure 5 (appliance schedule) reproduction |
| `tools/` | Aggregate-metric export, approximate evaluation with regenerated forecasts, manifest refresh |
| `tests/` | Regression tests for causality, forecasting, settlement and physical constraints |
| `results/` | Checkpoints, validation curves, evaluation summaries, aggregate metrics and provenance (see `results/README.md`) |
| `data/` | Location for the licensed input files, which are not distributed (see `data/README.md`) |
| `docs/` | Training, evaluation protocol, forecasting, experiments and development history |
| `*.ps1` | Windows PowerShell drivers for the full training and evaluation runs |

## Setup

Python 3.12 with the pinned packages in `requirements.txt` (PyTorch 2.13.0, `holidays==0.100` and others):

```
python -m venv .venv
.venv/bin/pip install -r requirements.txt          # Windows: .venv\Scripts\pip install -r requirements.txt
```

Training and evaluation stop unless the genuine `holidays==0.100` package is installed.

## Input data

The Pecan Street household data and the ERCOT price series are **not** included and must be obtained under the providers' terms. `data/README.md` gives the expected file names, columns, units and the SHA-256 hashes of the files used in the study. The DDQN forecast-state file `results/forecast/ddqn_state_forecasts.csv` is derived from the household data and is also not distributed. Regenerate it with `forecast_regeneration/REGENERATE.md`.

## Configuration

All CADR settings (households, data splits, DDQN hyperparameters, reward coefficients, λc, the capacity target and the forecast inputs) are in `config_seed_0.yaml`, `config_seed_1.yaml` and `config_seed_2.yaml`. EBDR uses `benchmark/config_benchmark_seed_*.yaml`. The configuration actually used by each trained model is stored next to its checkpoint as `results/**/training/config.yaml`.

## Quick start

All commands are run from the repository root.

**1. Inspect the published results (no data needed).** Seed-level monthly values behind the manuscript tables are in `results/aggregate_metrics/per_seed_monthly_metrics.csv`. `TABLE_MAP.md` gives the data file and script for every table and figure.

**2. Verify the frozen configurations (no data needed).**

```
python verify.py
```

**3. Run the full checks (licensed data and the forecast-state file needed).**

```
python -m benchmark.verify_benchmark
python tests/test_state_causality.py              # each tests/test_*.py is a standalone script
```

**4. Evaluate the released checkpoints.** Each command writes to a new folder; existing output folders are never overwritten.

```
python test.py --model_path results/seed_0/training/ddqn_best.pth --config_path config_seed_0.yaml \
  --output_dir eval/cadr_s0_july --test_start 182 --test_end 212 --target_plot_days 208
python -m benchmark.test_benchmark --model_path results/benchmark/seed_0/training/ddqn_best.pth \
  --config_path benchmark/config_benchmark_seed_0.yaml --period july --output_dir eval/ebdr_s0_july
python experiments/evaluate_matched_hems_ablation.py --seed 0 --month july --root <new folder containing a copy of seed_0/training/>
python experiments/evaluate_capacity_incentive_sensitivity.py --run lev10   # cap6 | cap8 | lev10; move the stored july/ folder aside first
```

The month windows are July (days 182–212, plot day 208) and August (days 213–243, plot day 220). `test_all_seeds.ps1` and `benchmark/test_benchmark_seeds.ps1` evaluate all seeds and months.

**5. Export the table values** from your evaluation folders:

```
python tools/export_aggregate_metrics.py --cadr-root <...> --cbsp-root <...> --ebdr-root <...> --sensitivity-root <...> --out <new file>.csv
```

## What can be reproduced, and how

| Level | Needs | Result |
|---|---|---|
| Inspect results | Nothing | Aggregate metrics, validation curves, sensitivity summaries and checkpoints are in `results/` |
| Static verification | Python environment | `verify.py` checks the frozen configurations |
| Exact re-evaluation | Licensed data **and** the original forecast-state file (SHA-256 `b86e09ba…`) | Reproduces the evaluation workbooks exactly |
| Approximate re-evaluation | Licensed data and a regenerated forecast-state file | `tools/evaluate_with_regenerated_forecasts.py`; results are labelled approximate |
| Retraining | Licensed data, the original forecast-state file, considerable CPU time | See `docs/TRAINING.md` |

`REPRODUCIBILITY.md` describes each level and what has been verified. The public repository alone cannot reproduce the manuscript numbers, because the household data are licensed.

## Further documentation

- `REPRODUCIBILITY.md`: exact versus approximate reproduction and the verification record
- `TABLE_MAP.md`: manuscript tables and figures mapped to files and scripts
- `docs/EXPERIMENTS.md`: the ablation, sensitivity and diagnostic experiments, with commands
- `docs/TRAINING.md`: training workflow and the retained λc
- `docs/EVALUATION_PROTOCOL.md`: the frozen evaluation and checkpoint-selection protocol
- `docs/FORECASTING.md`: LSTM forecasting and how forecasts enter the DDQN state
- `forecast_regeneration/REGENERATE.md`: regenerating the forecast inputs from licensed data
- `benchmark/README.md`: the EBDR benchmark
- `docs/DEVELOPMENT_HISTORY.md`: implementation history before the validated release
- `CHANGELOG.md`: release summary

## Licence

The code is released under the MIT licence (`LICENSE`). The licence does not cover third-party data. Checkpoints are loaded with `torch.load(weights_only=False)`, so only load checkpoint files from trusted sources.
