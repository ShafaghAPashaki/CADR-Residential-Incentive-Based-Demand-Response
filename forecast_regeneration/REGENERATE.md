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

Use Python 3.12 with `../requirements.txt` installed. Run every command from inside this folder, with the project root on `PYTHONPATH` so that `utils/` can be imported.

```powershell
cd forecast_regeneration
$env:PYTHONPATH = ".."; $env:PYTHONHASHSEED = "0"
mkdir data; copy ..\data\load_hourly_2018.csv data\; copy ..\data\ercot_hourly_price.csv data\
python predict_load_lstm.py --config config_seed_0.yaml
python predict_price_lstm.py --config config_seed_0.yaml
python crossfit_lstm.py --config config_seed_0.yaml
python verify_crossfit.py --config config_seed_0.yaml      # must end with "Gate: READY"
python install_outputs.py                                   # copies outputs, builds the state file, compares hashes
```

On Linux or macOS, use `export PYTHONPATH=.. PYTHONHASHSEED=0` and `cp` instead.

Everything this folder writes (`data/`, `results/`, `crossfit_results/`) is listed in `.gitignore`. These files contain licensed or household-derived data and must never be committed.

## What to expect

`install_outputs.py` compares each stage with the published run:

| Stage | Recorded SHA-256 (prefix) | Observed in an independent rerun (Linux, Python 3.13, torch 2.13.0) |
|---|---|---|
| Load data | `c183ce46` | Must match. A different Dataport export cannot reproduce the forecasts. |
| Price data | `56fdd5b4` | Must match |
| Full-model load forecasts | `e2a149ac` | `e49ff932`, **not** byte-identical: maximum difference 8.8e-6 kW, same best epochs |
| Full-model price forecasts | `c7446d11` | `17e16b26`, **not** byte-identical: maximum difference 1.2e-4 cent/kWh |
| Cross-fit training-day forecasts | `7ffbed0c` | `7ffbed0c`, **byte-identical** |
| DDQN forecast-state file | `b86e09ba` | `575b0ab9`, not byte-identical. Training-day rows are identical. Validation and test rows differ by at most 6.5e-6 kW (aggregate load) and 1.2e-4 cent/kWh. |

The full-model forecasts were produced on the authors' Windows workstation. Floating-point results of LSTM training differ slightly across CPU and BLAS platforms, so other machines are expected to reproduce them only approximately. Cross-fitting ran with deterministic CPU settings and has been reproduced byte-for-byte on a second platform.

- **If every stage prints EXACT**, evaluate the released checkpoints with `test.py` directly.
- **Otherwise**, use `../tools/evaluate_with_regenerated_forecasts.py`, which relaxes only the forecast-state hash check, and report the results as an approximate regeneration. See `../REPRODUCIBILITY.md` for the measured effect.

`README_CROSSFIT.md` and `CROSSFIT_METHODOLOGY.md` are the original cross-fitting documentation. Their only change is that a local Windows path was replaced by a placeholder.
