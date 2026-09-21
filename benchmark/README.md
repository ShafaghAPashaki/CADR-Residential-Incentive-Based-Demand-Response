# Elasticity-based capacity-blind DDQN benchmark

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
- Negative-price safeguard: incentive bounds use the smallest **strictly
  positive** price in the day. Real hourly prices, including negative prices,
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
- `verify_benchmark.py`: pre-training boundary, hash and numerical audits
- `BENCHMARK_IMPLEMENTATION_NOTES.md`: exact archived-to-active change record
- `config_benchmark_seed_0.yaml`, `_1.yaml`, `_2.yaml`: differ only in seed
- `run_benchmark_seeds.ps1`: sequential fail-stop training
- `test_benchmark_seeds.ps1`: three seeds × three periods, then summaries
- `results/`: historical archived benchmark outputs; active runs never write here

## Output structure

```text
results/
  benchmark/
    seed_0/
      training/
      july/
      august/
      november/
    seed_1/
    seed_2/
    summary.xlsx
  method_comparison.xlsx
```

## Required execution order

Run commands from the project root.

### 1. Verify before any training

```powershell
& $PY -m benchmark.verify_benchmark
```

Expected:

```text
BENCHMARK VERIFY: PASS
```

### 2. Optional local smoke test

The smoke output is isolated under `results/benchmark_smoke` and must never be
used in the paper.

```powershell
& $PY -m benchmark.main_benchmark `
  --config benchmark/config_benchmark_seed_0.yaml `
  --smoke-test
```

### 3. Official three-seed training

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\benchmark\run_benchmark_seeds.ps1 -Python $PY
```

### 4. Official three-seed, three-period testing

```powershell
.\benchmark\test_benchmark_seeds.ps1 -Python $PY
```

On first official test, do not pass `-Overwrite`.

## Comparison boundary

The comparison workbook includes physical/economic outcomes such as peak,
PAR, curtailment, external capacity alignment, payments and net wholesale
impact after payment. It deliberately excludes raw rewards, Q-values, raw
discomfort and raw household utility because the definitions differ between
methods.

## Status

This directory contains the benchmark implementation corresponding to the reported manuscript comparison. Validation-selected checkpoints and consolidated benchmark outputs are provided under `results/benchmark/` in the repository root.
