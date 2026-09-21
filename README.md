# Capacity-Aware Residential Incentive-Based Demand Response through Appliance-Level Load Coordination (CADR)

Code and reproducibility files for the associated manuscript.

This repository contains the CADR DDQN implementation, the capacity-blind EBDR benchmark, three-seed configurations, validation-selected checkpoints, and consolidated results.

## Setup

Python 3.12 is recommended.

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Place the required input files in `data/` as described in [`data/README.md`](data/README.md).

## Verification

```powershell
.\.venv\Scripts\python.exe verify.py
.\.venv\Scripts\python.exe -m benchmark.verify_benchmark
```

Both commands should report `PASS`.

## Quick evaluation

CADR, Seed 0:

```powershell
.\.venv\Scripts\python.exe test.py --model_path "results/seed_0/training/ddqn_best.pth" --config_path "config_seed_0.yaml" --test_start 220 --test_end 220 --target_plot_day 220 --output_dir "local_validation/cadr_seed0_day220"
```

EBDR, Seed 0:

```powershell
.\.venv\Scripts\python.exe -m benchmark.test_benchmark --model_path "results/benchmark/seed_0/training/ddqn_best.pth" --config_path "benchmark/config_benchmark_seed_0.yaml" --test_start 220 --test_end 220 --plot_days 220 --output_dir "local_validation/ebdr_seed0_day220"
```

## Full runs

CADR:

```powershell
.\run_all_seeds.ps1 -Python ".\.venv\Scripts\python.exe"
.\test_all_seeds.ps1 -Python ".\.venv\Scripts\python.exe"
```

EBDR:

```powershell
.\benchmark\run_benchmark_seeds.ps1 -Python ".\.venv\Scripts\python.exe"
.\benchmark\test_benchmark_seeds.ps1 -Python ".\.venv\Scripts\python.exe"
```

Main consolidated outputs are in:

```text
results/summary.xlsx
results/benchmark/summary.xlsx
results/method_comparison.xlsx
```

## Data

Pecan Street residential data and the ERCOT price data are not redistributed. See [`data/README.md`](data/README.md).
