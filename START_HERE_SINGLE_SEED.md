# Independent CADR reproduction: one seed, July 2018

**Status:** This is a proposed fresh-data workflow. The repository's prior frozen-input validation was tested, but this end-to-end route has **not yet been run from a clean checkout with newly licensed Dataport data**. Do not report it as verified until that test passes.

The acceptance target is **CADR only**, Seed 0, 2,500 DDQN episodes, validation-selected checkpoint, **1–31 July 2018** (days 182–212). No EBDR, other seeds or sensitivity runs are needed for this test.

## 1. Install (Windows PowerShell; from repository root)

Use Python 3.12 and the exact versions in `requirements.txt`:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe verify.py
```

`verify.py` is a static/integration check, **not** a full data-free test suite; some additional tests require licensed inputs.

## 2. Acquire and validate authorised data

Obtain 2018 Pecan Street Dataport hourly household channels and the price data specified by the manuscript **under your own authorised access**. Place:

- `data/load_hourly_2018.csv`: hourly rows with `dataid,time,air,car,clotheswasher,dishwasher,dry,non-shiftable,total`; for houses 661, 3039, 8565. `time` format: `dd/mm/yyyy HH:MM`; power-channel values are hourly kWh (equivalent to hourly mean kW).
- `data/ercot_hourly_price.csv`: `timestamp,Price`, with `Price` in USD/MWh.

Read `data/README.md` for unit, completeness, timezone and total-summation constraints. **Raw 15-minute exports can now be prepared using a validated conversion procedure.** Compare your downloaded header to `data/examples/raw_15min_schema.csv`, and follow `data/README.md` for the 15-minute appliance rules, timezone and explicit assumptions. Source licence conditions still apply. Do not commit either data file.

## 2a. Produce the hourly study inputs from your licensed downloads

If you have already obtained the correct preprocessed CSVs, validate them with the helper scripts, using separate source/output paths. Otherwise download the 15-minute Austin export in the documented 79-column schema and run:

```powershell
New-Item -ItemType Directory -Force downloads | Out-Null
# Save the authorised raw export as downloads/15minute_data_austin.csv.
& .\.venv\Scripts\python.exe data/prepare_load.py --input downloads/15minute_data_austin.csv --output data/load_hourly_2018.csv
# Save the already selected 2018 hourly ERCOT USD/MWh series in downloads/ercot_hourly_2018.csv.
& .\.venv\Scripts\python.exe data/prepare_price.py --input downloads/ercot_hourly_2018.csv --output data/ercot_hourly_price.csv
& .\.venv\Scripts\python.exe -m unittest tests.test_data_preparation -v
```

The raw source file must have offset-bearing timestamps and the required appliance channels. The load pipeline reproduced every archived 2018 load value to floating-point precision in a local audit, but the resulting CSV may not be **byte-identical** to the historical input; the exact-price-zone selection remains external and is not automated. The tests use fabricated example records only.

## 3. Regenerate the full-model and cross-fitted forecasts

The original forecasting sources are in `forecast_regeneration/` because its cross-fitting source requires the original API and file-hash checks. Run these commands from the repository root:

```powershell
$py = (Resolve-Path .\.venv\Scripts\python.exe).Path
Push-Location forecast_regeneration
$env:PYTHONPATH = ".."
$env:PYTHONHASHSEED = "0"
New-Item -ItemType Directory -Force data | Out-Null
Copy-Item ..\data\load_hourly_2018.csv data\load_hourly_2018.csv
Copy-Item ..\data\ercot_hourly_price.csv data\ercot_hourly_price.csv
& $py predict_load_lstm.py --config config_seed_0.yaml
& $py predict_price_lstm.py --config config_seed_0.yaml
& $py crossfit_lstm.py --config config_seed_0.yaml
& $py verify_crossfit.py --config config_seed_0.yaml
# Proceed only if the preceding command reports Gate: READY.
& $py install_outputs.py
Pop-Location
```

`install_outputs.py` creates `results/forecast/ddqn_state_forecasts.csv` and compares its hashes against the historical reference. The **forecast file and stage outputs are untracked**, but the builder also rewrites the tracked `results/forecast/ddqn_state_forecasts_provenance.json` with the active forecast hash. Preserve that generated provenance locally, **do not commit it**, and do not mistake it for the historical reference.

If any prerequisite fails, stop and investigate. Do not bypass leakage, cross-fit, unit, coverage or holiday gates.

## 4. Train one new DDQN checkpoint (2,500 episodes)

From repository root (after `Pop-Location`):

```powershell
& .\.venv\Scripts\python.exe main.py --config_path config_seed_0.yaml --allow_regenerated_forecasts --results_root repro_runs\july_seed0
```

This explicit opt-in relaxes **only the equality comparison to the original historical forecast-file SHA**. It does **not** change the retained lambda_c, frozen environment hash, reward/HEMS logic, DDQN hyperparameters or validation-based selection. It does not permit training with missing forecasts. Training records the active forecast SHA, the original SHA, and the reproduction mode in its new provenance JSON.

The selected checkpoint is `repro_runs/july_seed0/seed_0/training/ddqn_best.pth`. Never use the published `results/seed_0/training/` as a destination for independent training.

## 5. Evaluate the *new* selected checkpoint on July

```powershell
& .\.venv\Scripts\python.exe test.py --model_path repro_runs\july_seed0\seed_0\training\ddqn_best.pth --config_path repro_runs\july_seed0\seed_0\training\config.yaml --output_dir repro_runs\july_seed0\evaluation_july --test_start 182 --test_end 212 --target_plot_days 208
```

Because this checkpoint is trained with the *active* regenerated forecast file, its saved forecast SHA must match the active file: **do not use** `tools/evaluate_with_regenerated_forecasts.py` for this newly trained checkpoint. That wrapper is intended for applying *historical* checkpoints to changed forecasts.

Compare the resulting July Seed-0 metrics to `results/aggregate_metrics/per_seed_monthly_metrics.csv`; follow `TABLE_MAP.md` for definitions. Differences are possible because the original input exports and cross-platform LSTM results might differ. Record input hashes, forecast hashes, output hashes, checkpoint-selected episode and deviations. Do not claim exact reproduction without matching original inputs and numerical outputs.

## Known limitations and acceptance gate

- Raw Dataport data and private, household-derived forecast-state inputs cannot be bundled publicly without permission.
- The historical `lambda_c` calibration was replayed privately, not recalibrated; see `docs/LAMBDA_C_HISTORICAL_CALIBRATION.md`.
- Original full-model forecasts may differ slightly across platforms; a new training run may therefore select a different checkpoint or produce different results.
- **End-to-end status remains UNVERIFIED until this full sequence has actually completed on a clean download with authorised data.**
