# CADR: capacity-aware residential incentive-based demand response

This repository contains the code, configurations, selected trained checkpoints and aggregate results for **Capacity-Aware Residential Incentive-Based Demand Response through Appliance-Level Load Coordination (CADR)**, the elasticity-based benchmark (EBDR), the matched-HEMS ablation (CBSP), the LSTM forecasting pipeline and the sensitivity experiments described in the accompanying manuscript.

**Licensed household data are not included.** The Pecan Street data, and household-level series derived from them, must be obtained under the provider's licence; see `data/README.md`. Read `REPRODUCIBILITY.md` before attempting a rerun. It states what can be reproduced exactly, what only approximately, and how.

## Layout

| Path | Contents |
|---|---|
| `env/`, `agent/`, `main.py`, `test.py` | CADR environment, DDQN agent, training and evaluation |
| `benchmark/` | EBDR benchmark (see `benchmark/README.md`) |
| `r1c3_r2c2_ablation.py`, `run_r1c3_r2c2_ablation.py`, `eval_r1c3_r2c2_ablation.py` | CBSP matched-HEMS ablation |
| `run_r2_sensitivity.py`, `eval_r2_july.py`, `r2_common.py` | Capacity-target (6/8 kW) and ten-level incentive sensitivity runs |
| `run_sensitivity.py`, `configs/` | Reward-parameter (η, ρ) sensitivity runs |
| `predict_*_lstm.py`, `build_forecast_outputs.py`, `build_ddqn_forecast_inputs.py`, `FORECASTING.md` | Forecasting pipeline |
| `forecast_regeneration/` | Exact original forecasting and cross-fitting sources, and step-by-step regeneration (`REGENERATE.md`) |
| `results/` | Selected checkpoints, validation curves, aggregate results and provenance |
| `results/aggregate_metrics/per_seed_monthly_metrics.csv` | Seed-level monthly values behind the manuscript tables |
| `TABLE_MAP.md` | Data file and script for every manuscript table and figure |
| `tests/`, `verify.py`, `benchmark/verify_benchmark.py` | Regression tests and configuration checks |
| `tools/` | Aggregate-metric export and approximate evaluation with regenerated forecasts |

## Setup

Python 3.12 and the pinned packages:

```
python -m venv .venv
.venv/bin/pip install -r requirements.txt            # Windows: .venv\Scripts\pip ...
```

Place the licensed input files in `data/` (see `data/README.md`). They are git-ignored.

## Checks

`verify.py` checks the frozen configurations and needs only the code. Everything else below needs the licensed data and the forecast-state file `results/forecast/ddqn_state_forecasts.csv`. That file is not distributed; regenerate it with `forecast_regeneration/REGENERATE.md`.

```
python verify.py
python -m benchmark.verify_benchmark
python tests/test_state_causality.py     # and the other tests/test_*.py, each run as a script
```

The tests are standalone scripts, not pytest test functions.

## Evaluating the released checkpoints

**With the original forecast-state file** (SHA-256 `b86e09ba…`), run the scripts directly:

```
python test.py --model_path results/seed_0/training/ddqn_best.pth --config_path config_seed_0.yaml \
  --output_dir eval/cadr_s0_july --test_start 182 --test_end 212 --target_plot_days 208
python -m benchmark.test_benchmark --model_path results/benchmark/seed_0/training/ddqn_best.pth \
  --config_path benchmark/config_benchmark_seed_0.yaml --period july --output_dir eval/ebdr_s0_july
python eval_r1c3_r2c2_ablation.py --seed 0 --month july --root <fresh folder containing seed_0/training/{ddqn_best.pth,config.yaml}>
python eval_r2_july.py --run lev10        # cap6 | cap8 | lev10; move the stored july/ folder aside first
```

The Windows PowerShell drivers `test_all_seeds.ps1` and `benchmark/test_benchmark_seeds.ps1` run all seeds. The evaluation drivers refuse to overwrite existing output folders, so evaluate into fresh folders and keep the released results unchanged.

**With a regenerated forecast-state file**, prefix the CADR/CBSP/sensitivity commands with `python tools/evaluate_with_regenerated_forecasts.py`, for example `python tools/evaluate_with_regenerated_forecasts.py test.py --model_path ...`. Results obtained this way are an approximate regeneration and are labelled as such in the output folder.

Then export the table values:

```
python tools/export_aggregate_metrics.py --cadr-root <...> --cbsp-root <...> --ebdr-root <...> --sensitivity-root <...>
```

## Training

Training is resource-intensive and not needed to check the reported results. Use `run_final_retraining.ps1` or `main.py` (CADR), `benchmark/run_benchmark_seeds.ps1` (EBDR), `run_r1c3_r2c2_ablation.py` (CBSP), `run_r2_sensitivity.py` and `run_sensitivity.py`. The selection rule and protocol are in `FINAL_TRAINING_README.md` and `FINAL_EVALUATION_PROTOCOL.md`. November outputs are retained for completeness only; November is not part of the revised manuscript.

## Interpretation

The 7-kW capacity target is a soft operational target, not an enforced network constraint. Residual exceedances, rebound, unserved energy and other side effects are reported in the results.

## Licence and data rights

The original code is released under the MIT licence (`LICENSE`). The licence does not cover third-party data. Checkpoints are loaded with `torch.load(weights_only=False)`, so only load checkpoint files from trusted sources.
