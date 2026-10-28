# Regenerating the DDQN forecast-state inputs

This folder holds the forecasting sources exactly as they were used on 5 October 2026 to produce the forecasts behind the released checkpoints. Their SHA-256 hashes match those recorded in `reference/lstm_*_manifest_2026-10-05.json` and in the cross-fit manifest (`../results/forecast/crossfit/crossfit_manifest.json`).

| File | SHA-256 (prefix) | Role |
|---|---|---|
| `predict_load_lstm.py` | `e8781487` | Full-model household-load LSTMs |
| `predict_price_lstm.py` | `7f4a7863` | Full-model wholesale-price LSTM |
| `config_seed_0.yaml` | `ba3588b6` | Configuration snapshot from that date. Only its `general`, `training` and `forecast` sections are used here. The DDQN uses the root `config_seed_0.yaml`. |
| `crossfit_lstm.py` | `da0aecdb` | Blocked five-fold cross-fitting for training days 2–167 |
| `verify_crossfit.py` | `532925f8` | Independent cross-fit verification and gate |
| `crossfit_config.yaml` | `744ee8bc` | Cross-fit settings |

**Why a separate folder.** The root `predict_*_lstm.py` files are later edits of these scripts: identifiers were renamed (`ARTICLE_*` became `FORECAST_*`), messages and logging changed, and the holiday-package check became stricter. On the same machine, the root versions and these versions produce byte-identical forecasts. However, `crossfit_lstm.py` imports the original identifiers, so it cannot run against the root scripts. Keeping the original files together preserves every integrity check unchanged.

## Steps

Use Python 3.12 with `../requirements.txt` installed. The Windows driver in the [main guide](../README.md) performs these stages; the commands below expose them individually. Start from a fresh forecast-regeneration directory to avoid replacing existing outputs. The project root must be on `PYTHONPATH` so `utils/` can be imported.

```powershell
Push-Location forecast_regeneration
$env:PYTHONPATH = ".."
$env:PYTHONHASHSEED = "0"
New-Item -ItemType Directory data | Out-Null
Copy-Item ..\data\load_hourly_2018.csv data\
Copy-Item ..\data\ercot_hourly_price.csv data\
& $py predict_load_lstm.py --config config_seed_0.yaml
& $py predict_price_lstm.py --config config_seed_0.yaml
& $py crossfit_lstm.py --config config_seed_0.yaml
& $py verify_crossfit.py --config config_seed_0.yaml
# Continue only if the gate says READY.
& $py install_outputs.py
Pop-Location
```

`$py` is the full interpreter path set in the main guide. On Linux or macOS, start from the repository root and run:

```bash
CADR_PYTHON="$(pwd)/.venv/bin/python"
cd forecast_regeneration
export PYTHONPATH=.. PYTHONHASHSEED=0
mkdir data
cp ../data/load_hourly_2018.csv ../data/ercot_hourly_price.csv data/
"$CADR_PYTHON" predict_load_lstm.py --config config_seed_0.yaml
"$CADR_PYTHON" predict_price_lstm.py --config config_seed_0.yaml
"$CADR_PYTHON" crossfit_lstm.py --config config_seed_0.yaml
"$CADR_PYTHON" verify_crossfit.py --config config_seed_0.yaml
# Continue only if the gate says READY.
"$CADR_PYTHON" install_outputs.py
cd ..
```

Everything this folder writes (`data/`, `results/`, `crossfit_results/`) is listed in `.gitignore`. These files contain licensed or household-derived data and must never be committed. The installer refuses to overwrite existing installed forecasts and writes its new provenance to `repro_runs/forecast/`; the archived provenance and gate remain unchanged.

## What to expect

`install_outputs.py` compares each stage with the published run:

| Stage | Recorded SHA-256 (prefix) | Observed in an independent rerun |
|---|---|---|
| Load data | `c183ce46` | Historical file hash |
| Price data | `56fdd5b4` | Historical file hash |
| Full-model load forecasts | `e2a149ac` | `e49ff932`, **not** byte-identical: maximum difference 8.8e-6 kW, same best epochs |
| Full-model price forecasts | `c7446d11` | `17e16b26`, **not** byte-identical: maximum difference 1.2e-4 cent/kWh |
| Cross-fit training-day forecasts | `7ffbed0c` | `7ffbed0c`, **byte-identical** |
| DDQN forecast-state file | `b86e09ba` | `575b0ab9`, not byte-identical. Training-day rows are identical. Validation and test rows differ by at most 6.5e-6 kW (aggregate load) and 1.2e-4 cent/kWh. |

The table describes that particular rerun, not a cross-platform guarantee. A later Windows reader-style run produced a different cross-fit hash; retraining the DDQN with its regenerated forecasts also produced different results. Differences can arise from input preparation and execution environments; their precise cause in that run has not been established. See [REPRODUCIBILITY.md](../REPRODUCIBILITY.md).

- **If the forecast-state hash matches**, published CADR checkpoints can be evaluated directly.
- **If it differs**, use the published-checkpoint route in `../tools/run_cadr.py`; it invokes the approximate-evaluation wrapper and records the changed hash. The other checks remain active.

See [README_CROSSFIT.md](README_CROSSFIT.md) for the cross-fit file overview and [CROSSFIT_METHODOLOGY.md](CROSSFIT_METHODOLOGY.md) for the recorded methodology.
