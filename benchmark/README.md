# EBDR benchmark

This folder contains the elasticity-based, capacity-blind DDQN benchmark used in the manuscript.

The 7 kW capacity target is not used in the benchmark state, reward, training, validation, or action selection. It is used only for post-hoc evaluation.

## Verify

```powershell
.\.venv\Scripts\python.exe -m benchmark.verify_benchmark
```

## Train

```powershell
.\benchmark\run_benchmark_seeds.ps1 -Python ".\.venv\Scripts\python.exe"
```

## Evaluate

```powershell
.\benchmark\test_benchmark_seeds.ps1 -Python ".\.venv\Scripts\python.exe"
```

Validation-selected checkpoints and consolidated outputs are under `results/benchmark/`.
