# CADR LSTM cross-fitting package

Generates leakage-controlled **out-of-fold (OOF)** forecasts for the DDQN
training days (2–167) for households 661, 3039, 8565 and the wholesale price.
It does not touch the DDQN, the HEMS, the manuscript, Table A.9, Figure A.6,
or any existing full-model output. See `CROSSFIT_METHODOLOGY.md` for the
design and its justification.

## Files

| File | Purpose |
|---|---|
| `crossfit_lstm.py` | Static checks, 5-fold cross-fit of all four targets, determinism re-fit, combined DDQN file, audit and manifest |
| `verify_crossfit.py` | Independent re-verification from the saved files (no torch) and the final gate |
| `crossfit_config.yaml` | Cross-fit settings (folds, buffer, seeds, CPU/determinism). Kept separate so `config_seed_0.yaml` stays byte-identical |
| `run_lstm_crossfit.ps1` | One-command Windows runner |
| `CROSSFIT_METHODOLOGY.md` | Methodology note and suggested manuscript sentence |

No changes to `requirements.txt`, `config_seed_*.yaml`, `predict_load_lstm.py`
or `predict_price_lstm.py` are needed or made.

## Where to put the files

Copy the four code/config files into the **project root**, next to
`predict_load_lstm.py`, `predict_price_lstm.py`, `config_seed_0.yaml`, `utils/`
and `data/`. The existing full-model outputs must be present in
`results/forecast/` (load, price, publication).

The cross-fit imports the feature construction, LSTM class and scalers from
your own `predict_load_lstm.py` / `predict_price_lstm.py`, so there is a single
definition of the Appendix-A model. At start-up it checks that those two
scripts and `config_seed_0.yaml` have exactly the SHA-256 recorded in
`results/forecast/*/lstm_*_manifest.json`. **If you have edited either script
or the config since training the full models, the run stops.** Restore the
versions that produced the full models.

## Commands

Static checks only (no training, about 1–2 minutes):

```powershell
.\run_lstm_crossfit.ps1 `
  -Python ".\.venv\Scripts\python.exe" `
  -Config ".\config_seed_0.yaml" -CheckOnly
```

Full run:

```powershell
.\run_lstm_crossfit.ps1 `
  -Python ".\.venv\Scripts\python.exe" `
  -Config ".\config_seed_0.yaml"
```

If `crossfit_results\` already exists and is not empty, the run stops; add
`-Overwrite` to replace it deliberately. Only `config_seed_0.yaml` needs to be
used: the forecasting section is identical in all three seed configs and the
OOF forecasts are shared across DDQN seeds.

If PowerShell blocks the script, run once in that window:
`Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass`.

## Outputs

```text
crossfit_results/
  household_661/  household_3039/  household_8565/  price/
      oof_predictions.csv          timestamp, doy, fold_id, padding_free, actual, forecast, model_seed
      fold_0/ ... fold_4/
          model.pt  scaler.npz  training_history.csv  fold_meta.json
  combined/
      ddqn_crossfit_forecasts_days_002_167.csv   <- the DDQN-facing file
      column_map.json
  audit/
      crossfit_gate.txt / crossfit_gate.json     <- READY | BLOCKED_FOR_CROSSFIT_FAILURE
      crossfit_oof_metrics.csv                   OOF MAE/RMSE/WAPE + gap vs validation
      leakage_report.json  determinism_report.json
      full_model_integrity.json                  proof results/forecast/ was not changed
      crossfit_manifest.json                     design, folds, seeds, epochs, versions, SHA-256 of everything
      crossfit_internal_checks.json
```

`fold_meta.json` records the held-out days and timestamps, the excluded range
including buffers, the exact fitting target ranges after exclusions, sample
counts before and after each exclusion, the fixed epoch count, all seeds and
the model/scaler hashes.

### DDQN-facing file

`combined/ddqn_crossfit_forecasts_days_002_167.csv`, one row per hour of days 2–167:

| Column | Meaning |
|---|---|
| `timestamp`, `doy`, `fold_id` | Hour, day of year, fold that produced it |
| `forecast_load_eu1_kw` / `eu2` / `eu3` | OOF forecasts for households 661 / 3039 / 8565 |
| `forecast_aggregate_load_kw` | Exact sum of the three household forecasts |
| `forecast_price_cent_per_kwh` | OOF price forecast, = $/MWh ÷ 10 |
| `inputs_padding_free_all_targets` | False for the earliest hours whose inputs were zero-padded (forecast, but excluded from metrics) |
| `audit_*` | Actual values and the raw $/MWh price forecast, for auditing only. **Never use as DDQN inputs.** |

Days 168 onward are not in this file; the DDQN uses the existing full-model
forecasts there.

## Gate

`verify_crossfit.py` writes `READY` only if: all four targets completed;
every leakage check passed for every fold; coverage is exactly days 2–167 with
no duplicates or gaps; the startup-padding counts match the full-model
convention; the aggregate identity and unit conversion hold; fold 0 is
reproduced on re-fit; every artifact hash matches; and no full-model file has
changed. The train-versus-validation gap is reported transparently in
`crossfit_oof_metrics.csv` and in the gate file but does **not** gate, and no
further tuning is performed whatever its value.

## Expected successful output

The final lines printed by the runner should look like this (the figures are
from a test run on the same data and full-model artifacts; the run is
deterministic on CPU, so yours should match closely, though not necessarily to
the last digit if your PyTorch build differs):

```text
CROSSFIT COMPLETE

EU1 (household 661):
  OOF MAE  = 0.4627 kW
  OOF RMSE = 0.7975 kW
  OOF WAPE = 59.99 %
  gap vs validation = -0.604 (threshold 0.20; informational)
...
Price:
  OOF MAE  = 9.9948 $/MWh  (= 0.9995 cent/kWh)
...
Coverage:
  expected hours = 3984
  produced hours = 3984
  duplicates = 0
  gaps = 0

Leakage checks: PASS
Aggregate identity: PASS
Price-unit conversion: PASS
Determinism: PASS
Hashes: PASS
Full-model files unchanged: PASS

Gate: READY
```

## What to send back for independent verification

Zip and send the whole `crossfit_results\audit\` folder plus:

* `crossfit_results\combined\ddqn_crossfit_forecasts_days_002_167.csv` and `column_map.json`
* the four `crossfit_results\<target>\oof_predictions.csv` files
* the twenty `fold_*\fold_meta.json` files
* the console output (copy-paste or `... | Tee-Object crossfit_console.log`)

Model weights and scalers are not needed for verification; their hashes are in
the manifest.

## Runtime

Training runs on the CPU for reproducibility (20 fold fits plus 4 determinism
re-fits, 342 epochs in total). On a 2-core test machine the full run took about
11 minutes (about 1.5 minutes of static and leakage checks, then about 10
minutes of training). A typical 4–8-core laptop should be faster. Memory use is
under 1 GB. Nothing is written outside `crossfit_results\`.
