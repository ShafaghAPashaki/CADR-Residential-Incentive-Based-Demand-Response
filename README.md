# Capacity-Aware Residential Incentive-Based Demand Response through Appliance-Level Load Coordination (CADR)

Code and selected reproducibility artifacts for the manuscript **“Capacity-Aware Residential Incentive-Based Demand Response through Appliance-Level Load Coordination.”**

## Overview

This repository contains the DDQN implementation of the proposed capacity-aware residential incentive-based demand-response framework (CADR), together with the elasticity-based capacity-blind benchmark (EBDR), three-seed configurations, validation-selected checkpoints, and summary outputs used in the manuscript.

The repository is intended to document the computational implementation reported in the paper. Third-party residential load data are **not redistributed**; see [`data/README.md`](data/README.md).

## Repository structure

```text
agent/                         DDQN agent
benchmark/                     Capacity-blind elasticity-based DDQN benchmark
configs/rho_sensitivity/       rho sensitivity configurations
data/                          Data-access and expected-format documentation
env/                           CADR environment
results/                       Selected checkpoints and consolidated outputs
utils/                         Data/configuration/replay-buffer utilities
config_seed_0.yaml             CADR seed 0 configuration
config_seed_1.yaml             CADR seed 1 configuration
config_seed_2.yaml             CADR seed 2 configuration
main.py                        CADR training entry point
test.py                        CADR evaluation entry point
verify.py                      Static/numerical project checks
summarize_results.py           Cross-seed result aggregation
requirements.txt               Python dependencies
```

## Experimental protocol

### CADR

- Seeds: `0`, `1`, `2`
- Training: days `2–167` of 2018
- Validation: days `168–181`
- Test periods:
  - July: days `182–212`; representative day `208`
  - August: days `213–243`; representative day `220`
  - November robustness check: days `305–334`; representative day `312`
- Episodes: `2500`
- Capacity target: `7 kW`
- Validation is used for checkpoint selection; test data are not used to select a seed or checkpoint.

The three canonical CADR configurations differ only in `general.seed`.

### EBDR benchmark

The benchmark under `benchmark/` is an independent elasticity-based, capacity-blind, curtailment-only DDQN implementation. The 7 kW target is excluded from benchmark state construction, reward calculation, training, validation and action selection, and is used only for post-hoc comparison/reporting.

See [`benchmark/README.md`](benchmark/README.md) and [`benchmark/BENCHMARK_IMPLEMENTATION_NOTES.md`](benchmark/BENCHMARK_IMPLEMENTATION_NOTES.md) for the detailed implementation boundary.

## Environment

Python 3.12 is recommended (the publication package was verified with Python 3.12.3). Create a clean virtual environment and install the pinned dependencies:

```bash
python -m venv .venv
```

On Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

If PowerShell execution policy prevents activation, activation is optional; use the virtual-environment interpreter directly:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Then place the required data files in `data/` as described in [`data/README.md`](data/README.md).

## Verification

From the project root, after installing the dependencies:

```powershell
.\.venv\Scripts\python.exe verify.py
.\.venv\Scripts\python.exe -m benchmark.verify_benchmark
```

Both commands should report `PASS` before evaluation or retraining.

## Quick reproducibility check

With the required data files present, the included validation-selected checkpoints can be evaluated without retraining. For example, a one-day CADR check for Seed 0 is:

```powershell
.\.venv\Scripts\python.exe test.py --model_path "results/seed_0/training/ddqn_best.pth" --config_path "config_seed_0.yaml" --test_start 220 --test_end 220 --target_plot_day 220 --output_dir "local_validation/cadr_seed0_day220"
```

A corresponding one-day EBDR benchmark check is:

```powershell
.\.venv\Scripts\python.exe -m benchmark.test_benchmark --model_path "results/benchmark/seed_0/training/ddqn_best.pth" --config_path "benchmark/config_benchmark_seed_0.yaml" --test_start 220 --test_end 220 --plot_days 220 --output_dir "local_validation/ebdr_seed0_day220"
```

These commands use the archived checkpoints and do not retrain either model.

## Training

The three CADR seeds should be trained sequentially:

```powershell
.\run_all_seeds.ps1 -Python ".\.venv\Scripts\python.exe"
```

Training output for each seed is written below:

```text
results/seed_0/training
results/seed_1/training
results/seed_2/training
```

The validation-selected checkpoint is `ddqn_best.pth`.

## Evaluation

Evaluate the trained CADR agents with:

```powershell
.\test_all_seeds.ps1 -Python ".\.venv\Scripts\python.exe"
```

The consolidated multi-seed results used for reporting are provided in:

```text
results/summary.xlsx
results/method_comparison.xlsx
```

## Benchmark execution

Verify the benchmark before running it:

```powershell
python -m benchmark.verify_benchmark
```

Train and evaluate all three benchmark seeds using the scripts in `benchmark/`.
Selected benchmark checkpoints and the consolidated benchmark summary are included in `results/benchmark/`.

## Sensitivity analyses

The repository includes configuration/provenance material and consolidated outputs for the sensitivity analyses used in the study. Large intermediate training artifacts are intentionally omitted from the publication repository.

## Reproducibility notes

- All three primary seeds use the same source snapshot and fixed household-coefficient matrix.
- Cross-seed reporting uses the mean and sample standard deviation.
- Residual capacity exceedances are retained and the 7 kW level is treated as an operational target rather than a hard network constraint.
- The included `.pth` files are the validation-selected checkpoints used for the corresponding seed configurations.
- Large intermediate checkpoints, temporary diagnostics, caches and third-party data are excluded from the public repository.

## Data and code availability

The source code, configurations, selected trained checkpoints and consolidated result files are provided in this repository. Residential load data from Pecan Street Dataport are not redistributed and must be obtained separately under the provider's access terms. The ERCOT price series must likewise be obtained/prepared from the source cited in the manuscript.

## Citation

If you use this repository, please cite the associated manuscript. A DOI-specific citation can be added here after publication or after archiving a repository release.
