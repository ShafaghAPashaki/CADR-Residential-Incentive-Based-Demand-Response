# Forecasting

The forecasting scripts produce one-hour-ahead LSTM forecasts of household demand and wholesale electricity price, and merge them into the DDQN forecast-state table. They need the licensed input data (`data/README.md`). For the sources and steps used to regenerate the published forecast inputs, see `forecast_regeneration/REGENERATE.md`. The regenerated inputs are not guaranteed to be byte-identical; see `REPRODUCIBILITY.md`.

## Model specification

- Independent household-load LSTMs for households 661, 3039, and 8565.
- One wholesale-price LSTM.
- 14 features per recurrent step: month, day of week, hour, holiday, weekend, and lags h-1, h-2, h-3, h-24, h-25, h-26, h-48, h-49, h-50.
- 24-hour recurrent window and one-hour forecast horizon.
- Two LSTM layers, 64 hidden units, dropout 0.2.
- Adam optimiser and MSE training loss.
- Forecast seed 42.

## Data splits

- Training: days 2-167 of 2018.
- Validation: days 168-181.
- Held-out test: July and August 2018.
- Comparison-figure window: 23-29 July 2018.

## Run

The legacy root driver below writes reports into `results/forecast/`. Run it in a separate working copy if you need to preserve the archived results. For the reader-facing reproduction route, use the forecast preparation in the main `README.md` instead.

```powershell
.\run_lstm_forecasts.ps1 `
  -Python "C:\path\to\python.exe" `
  -Config ".\config_seed_0.yaml"
```

Static validation only:

```powershell
.\run_lstm_forecasts.ps1 `
  -Python "C:\path\to\python.exe" `
  -Config ".\config_seed_0.yaml" `
  -CheckOnly
```

## Output structure

The tree below describes a complete local forecast run; the public release contains only selected outputs and does not distribute licensed data or household-derived forecasts.

```text
results/forecast/
├── README_OUTPUTS.md
├── forecast_comparison.png
├── forecast_comparison.pdf
├── forecast_metrics.csv
├── forecast_metrics.tex
├── ddqn_state_integration_gate.txt
├── load/
│   ├── models and scalers
│   ├── training histories
│   ├── prediction CSVs
│   ├── internal diagnostic files
│   └── standalone load figures
└── price/
    ├── model and scaler
    ├── training history
    ├── prediction CSV
    ├── internal diagnostic files
    └── standalone price figure
```

The final forecast metric table reports MAE only. Wholesale-price outputs use cent/kWh and household-load outputs use kW. Internal train-validation diagnostics remain in the `load/` and `price/` directories for reproducibility and cross-fit gating.

## Reproducibility checks

Before training, the code validates the locked forecast specification, gap-free hourly indexing, input dimensions, sample counts, and absence of target/future leakage. Startup-padded samples are available for operational forecasting but excluded from fitting, scaling, and reported metrics.

If the train-validation diagnostic gate is triggered, training-day forecasts must use blocked cross-fitting before controller integration.

## DDQN decision-state integration

The controller consumes a forecast-only table at `results/forecast/ddqn_state_forecasts.csv`. Training days 2-167 use the frozen out-of-fold/cross-fitted forecasts. Days 168 onward use the fixed full-model forecasts. The input table contains no realised/audit target columns.

In the 43-element DDQN state, one-step forecasts replace the realised whole-house pre-action load and wholesale-price inputs. The aggregate pre-action load, capacity-need flag, and required-reduction fields are derived from the forecasted household loads after applying only commitments already known before the current action. Current appliance-flexibility variables and all commitment variables remain current/known observations.

Physical appliance response, reward, settlement, capacity evaluation, and the hourly incentive bound continue to use realised quantities. The diagnostic `oracle_actual` state mode exists only to verify that the integration maps exactly to the pre-integration environment when forecasts are replaced by realised values; production configs use `forecast`.

Negative household-load forecasts are floored at 0 kW only when forming the controller state. The wholesale-price forecast retains the existing state transform `log1p(clip(price, 0, 80))`; this does not alter the realised price used by settlement or incentive bounds.

The cross-fit package is stored under `results/forecast/crossfit/`, and its gate must be `READY`. `build_ddqn_forecast_inputs.py` rebuilds the merged controller-input table and provenance file. The cross-fitting code, together with the exact forecasting sources used for the published forecasts, is in `forecast_regeneration/`; follow `forecast_regeneration/REGENERATE.md` to regenerate the controller-input table from licensed data.

