# CADR: capacity-aware residential incentive-based demand response

Research code, frozen configurations, trained checkpoints and aggregate results for *Capacity-Aware Residential Incentive-Based Demand Response through Appliance-Level Load Coordination*. The proposed CADR method uses a Double Deep Q-Network (DDQN) to coordinate residential appliance responses. The repository also includes the elasticity-based EBDR benchmark and the matched-HEMS CBSP ablation.

## Start here

Choose the route that fits your purpose; each builds on the previous one.

| Route | What you can do | Inputs needed |
|---|---|---|
| **1. Inspect** | Read archived results, configurations, checkpoint hashes and table mappings. No model is run. | None |
| **2. Evaluate** | Run a published CADR checkpoint for any released seed and test period. No DDQN retraining. | Authorised demand and price data, plus the original or a regenerated forecast-state file |
| **3. Train** | Regenerate the forecasts from authorised inputs, train one or more CADR seeds, and evaluate the new checkpoints. | Authorised raw or prepared demand and price data |

The public repository does not contain the licensed household records or the household-derived forecast-state file. Regenerated forecasts can differ from those used in the paper; in turn, retraining can select a different checkpoint and produce different results. See [Reproducibility](REPRODUCIBILITY.md) for what was tested.

## 1. Inspect the release

From the repository root, run:

```text
python tools/inspect_release.py
```

The inspector uses only the Python standard library. `PASS` confirms that key released files exist and prints the available archive contents and selected hashes; it does not run a model. Browse [results/README.md](results/README.md), `results/aggregate_metrics/per_seed_monthly_metrics.csv`, and [TABLE_MAP.md](TABLE_MAP.md) for the published outputs and manuscript mappings.

## Setup for evaluation and training

Install Python 3.12 and the pinned packages in [requirements.txt](requirements.txt). For example, in Windows PowerShell from the repository root:

```powershell
py -3.12 -m venv .venv
$py = (Resolve-Path .\.venv\Scripts\python.exe).Path
& $py -m pip install -r requirements.txt
```

Activation is not required. If you already have a suitable environment, set `$py` to its full `python.exe` path instead. On macOS/Linux, create an environment with `python3.12 -m venv .venv`, install using `.venv/bin/python -m pip install -r requirements.txt`, and replace `& $py` in the examples with `.venv/bin/python`. The examples below use PowerShell.

## Prepare data and forecasts

1. Obtain authorised Pecan Street Dataport and ERCOT price data. [data/README.md](data/README.md) gives the source links, download names, schemas, units and preparation commands. The prepared files are `data/load_hourly_2018.csv` and `data/ercot_hourly_price.csv`. Keep licensed and household-derived files local.
2. If you do not have the original authorised forecast-state file, regenerate it once for all three CADR seeds. With the two prepared input files in `data/`, run:

   ```powershell
   powershell -ExecutionPolicy Bypass -File .\run_reproduction_seed0_july.ps1 -Python $py -ReusePreparedData -ForecastOnly
   ```

   Despite the historical filename, `-ForecastOnly` prepares the common forecasts and does **not** train a DDQN. Wait for the cross-fit gate to report `READY`. On macOS/Linux, or to run the stages individually, follow [forecast_regeneration/REGENERATE.md](forecast_regeneration/REGENERATE.md). Do not rerun preparation into existing outputs; use a fresh checkout or inspect the existing outputs and provenance first.

An authorised original forecast-state file can instead be placed at `results/forecast/ddqn_state_forecasts.csv` if its use and transfer are permitted by the data provider. Its recorded SHA-256 begins `b86e09ba`.

## 2. Evaluate published CADR checkpoints

The launcher defaults to seeds **0, 1 and 2**, and the paper's July and August test periods. It uses the installed forecast-state file and writes new results under `repro_runs/eval/published/`.

```powershell
& $py tools/run_cadr.py evaluate
```

To choose any combination, use `--seeds` and `--months`. For example, this evaluates seeds 1 and 2 in August and November:

```powershell
& $py tools/run_cadr.py evaluate --seeds 1 2 --months august november
```

| Period | 2018 day-of-year range | Detailed plot day | Status |
|---|---:|---:|---|
| `july` | 182–212 | 208 | Manuscript test period |
| `august` | 213–243 | 220 | Manuscript test period |
| `november` | 305–334 | 312 | Additional retained robustness period |

Use `--months july august november` for all three periods, and `--dry-run` to preview commands without loading data or running a model. Each result is stored as `repro_runs/eval/published/seed_<seed>/<month>/`. If the active forecast hash differs from the checkpoint's recorded hash, `APPROXIMATE_FORECAST_INPUTS.json` identifies the run as an approximate evaluation. Other checkpoint checks remain active.

For another 2018 period with available inputs, use inclusive day-of-year numbers. For example, September is days 244–273:

```powershell
& $py tools/run_cadr.py evaluate --seeds 0 1 2 --days 244 273 --plot-day 250
```

The forecast-state pipeline covers days 2–365; day 1 is not available. Custom periods are exploratory and are not the published July/August comparison. Days 2–167 were used for training and days 168–181 for validation, so evaluating those days is not a held-out test. The output folder is named `days_<start>_<end>`.

## 3. Train and evaluate new CADR checkpoints

After preparing the data and forecast-state file, train all three seeds with regenerated forecasts:

```powershell
& $py tools/run_cadr.py train --regenerated-forecasts
```

Each seed uses its own `config_seed_<seed>.yaml`, 2,500 episodes, and validation-based checkpoint selection. Training can take substantial time. To train a subset, add, for example, `--seeds 1 2`. If using the original forecast-state file, omit `--regenerated-forecasts` to retain the original strict forecast-hash check.

New checkpoints and their saved configurations are written to `repro_runs/train/seed_<seed>/training/`. Evaluate those checkpoints, using the **same active forecast-state file** used for training:

```powershell
& $py tools/run_cadr.py evaluate --source trained
```

The same `--seeds`, `--months` and custom `--days` options apply. To run only the trained seeds 1 and 2 for all three retained periods:

```powershell
& $py tools/run_cadr.py evaluate --source trained --seeds 1 2 --months july august november
```

The launcher refuses to reuse an existing output directory. Choose another `--train-root` for training or `--output-root` for evaluation when rerunning; pass the same `--train-root` when evaluating a newly trained checkpoint from a custom root. It does not overwrite published results.

## Reading and comparing outputs

Each CADR evaluation folder contains `results.xlsx` with monthly tables, `audit.xlsx` with detailed results, and a `day_<number>/figures/` folder for the selected plot day. Compare a published or newly trained run with `results/seed_<seed>/<month>/results.xlsx`, matching the **same seed and month**. The aggregate CSV and [TABLE_MAP.md](TABLE_MAP.md) identify the metrics behind the paper's tables. Outputs and plots alone do not establish that every value matches; compare the corresponding numbers.

Windows can reject very long paths when saving figures, especially in network folders. If this happens, move the checkout to a shorter local path or set a shorter, new `--output-root`.

For EBDR, see [benchmark/README.md](benchmark/README.md); for CBSP and sensitivity experiments, see [docs/EXPERIMENTS.md](docs/EXPERIMENTS.md). These are separate methods/experiments, not options of the CADR launcher. The evaluation and validation-selection rules are in [docs/EVALUATION_PROTOCOL.md](docs/EVALUATION_PROTOCOL.md).

## Licence

Code is released under the [MIT licence](LICENSE); it does not grant rights to third-party data. Only load checkpoints from trusted sources.
