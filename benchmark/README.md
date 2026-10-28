# Elasticity-based DDQN benchmark

## Scientific identity

This folder contains an independent **elasticity-based, capacity-blind,
curtailment-only DDQN benchmark**.

The benchmark does not read the 7 kW capacity threshold during training,
validation, checkpoint selection, state construction, reward calculation,
response calculation, or action selection. The threshold appears only under
`external_reference` in the YAML files and is read only by the test/reporting
pipeline after a trajectory has been generated.

## Active response and reward

For household `n` and hour `h`:

```text
DeltaE[n,h] = clip(
    baseline[n,h] * elasticity[h] * raw_action[n,h],
    0,
    max_reduction_fraction * baseline[n,h]
)
```

```text
R_SP = price[h] * sum(DeltaE[:,h]) - sum(lambda[:,h] * DeltaE[:,h])

R_EU[n] = rho * lambda[n,h] * DeltaE[n,h]
          - (1-rho) * (0.5 * mu[n] * DeltaE[n,h]^2 + kappa * DeltaE[n,h])

R_total = alpha * R_SP + (1-alpha) * sum(R_EU)
```

No capacity, tracking, no-need, offer-regularisation, appliance, or proposed
reward-normalisation term is present.

## Frozen conventions

- Houses: `661, 3039, 8565`
- Train: days `2–167`
- Validation: days `168–181`
- Test:
  - July `182–212`, plot day `208`
  - August `213–243`, plot day `220`
  - November `305–334`, plot day `312`
- Seeds: `0, 1, 2`
- Episodes: `2500`
- Network: `12 -> 128 -> 64 -> 125`
- Causal daily-price safeguard: incentive bounds retain the flat daily
  structure but use the previous day's minimum **strictly positive** price. Real hourly prices, including negative prices,
  remain in the SP reward.
- Raw action zero retains the archived nominal-rate convention
  (`lambda=lambda_min`) but produces zero response and zero payment.
- Rebound: `N/A`, because the benchmark curtails and never shifts energy.

## Files

- `env_benchmark.py`: active capacity-blind environment
- `main_benchmark.py`: independent training and validation-only checkpoint selection
- `test_benchmark.py`: full-month test, audit, Excel reports and plots
- `summarize_benchmark.py`: three-seed mean/SD and signal stability
- `compare_methods.py`: commensurable proposed-vs-benchmark comparison only
- `verify_benchmark.py`: boundary, hash and numerical verification
- `BENCHMARK_IMPLEMENTATION_NOTES.md`: exact archived-to-active change record
- `config_benchmark_seed_0.yaml`, `_1.yaml`, `_2.yaml`: differ only in seed
- `run_benchmark_seeds.ps1`: sequential fail-stop training
- `test_benchmark_seeds.ps1`: three seeds × three periods, then summaries

## Running the benchmark

First follow the data, environment and forecast preparation in the [main guide](../README.md). A regenerated forecast can differ from the one used for the paper, so its checkpoint evaluation is approximate; unlike the CADR wrapper, EBDR does not write a forecast-hash warning.

From the repository root, verify the implementation:

```powershell
& $py -m benchmark.verify_benchmark
```

Evaluate a published checkpoint into a new folder. For example, Seed 1 in August:

```powershell
& $py -m benchmark.test_benchmark --model_path results/benchmark/seed_1/training/ddqn_best.pth --config_path benchmark/config_benchmark_seed_1.yaml --period august --output_dir repro_runs/ebdr/seed_1/august
```

Change both seed numbers to `0`, `1` or `2`, and choose `july`, `august` or `november` for the period and output folder. Use a new output folder for each run. July and August are the manuscript test periods; November is an additional retained period.

The benchmark training driver uses a fixed output under `results/benchmark/` and refuses an existing training folder. To retrain, work in a separate copy of the repository without the benchmark training outputs in that copy, then run:

```powershell
powershell -ExecutionPolicy Bypass -File .\benchmark\run_benchmark_seeds.ps1 -Python $py
```

The `test_benchmark_seeds.ps1` wrapper now writes all evaluation runs to `repro_runs/ebdr/seed_<seed>/<period>/` and summary workbooks under `repro_runs/` (ignored by Git). It refuses existing output folders unless `-Overwrite` is explicitly supplied; the summary tools always refuse overwriting an existing workbook, including when `-Overwrite` is used for evaluations. Select a fresh root or remove only known disposable generated runs, never any archive below `results/`.

Fresh EBDR evaluations retain `Hourly_Aggregate` in `results.xlsx` for cross-seed action-consistency diagnostics, but place detailed household-hour records in the ignored `audit.xlsx` (`Household_Ledger`). Archived EBDR `results.xlsx` workbooks contain only the six historical report sheets. To compare regenerated results against the frozen archive, compare those six sheets by name and cell value; do **not** compare binary workbook hashes or assume the number of worksheets is unchanged. The summariser needs `Hourly_Aggregate` from **fresh evaluations** and cannot reconstruct `Signal_Stability` from the intentionally redacted archived workbooks.

For example, after running all nine benchmark evaluations, generate the summary and optional proposed-vs-benchmark comparison without touching `results/`:

```powershell
& $py -m benchmark.summarize_benchmark --results_root repro_runs/ebdr --output repro_runs/ebdr/summary.xlsx
& $py -m benchmark.compare_methods --proposed_summary results/summary.xlsx --benchmark_summary repro_runs/ebdr/summary.xlsx --output repro_runs/method_comparison.xlsx
```


## Comparison boundary

The comparison workbook includes physical/economic outcomes such as peak,
PAR, curtailment, external capacity alignment, payments and net wholesale
impact after payment. It deliberately excludes raw rewards, Q-values, raw
discomfort and raw household utility because the definitions differ between
methods.

## Status

This directory contains the active benchmark implementation and its reproducibility checks. Run `python -m benchmark.verify_benchmark` before retraining after any source/configuration change.

- The benchmark DDQN state uses the same frozen one-step household-load and
  wholesale-price forecasts as CADR: cross-fitted forecasts during training
  and fixed full-model forecasts from validation onward. Physical curtailment
  and reward accounting continue to use realised demand and price.
