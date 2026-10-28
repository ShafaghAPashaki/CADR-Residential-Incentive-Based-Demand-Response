# Cross-fitted training forecasts

The DDQN uses out-of-fold forecasts for the training days of 2018. Cross-fitting generates predictions for the three households and wholesale price without fitting the forecasting model on the held-out day block. Days 168 onward use the separately trained full-model forecasts.

## Code and settings

| File | Role |
|---|---|
| `predict_load_lstm.py`, `predict_price_lstm.py` | Forecasting models and feature construction used by cross-fitting |
| `config_seed_0.yaml` | Historical forecast configuration, shared by the DDQN seeds |
| `crossfit_config.yaml` | Fold, buffer, seed and determinism settings |
| `crossfit_lstm.py` | Trains the fold models and generates training-day forecasts |
| `verify_crossfit.py` | Checks the generated files and writes the verification gate |
| `install_outputs.py` | Installs the generated forecasts and builds the DDQN forecast-state file |
| `run_lstm_crossfit.ps1` | Optional Windows driver for the cross-fit and verification stages |

Keep these files in `forecast_regeneration/`; do not copy them into the project root. The root forecasting scripts are later versions with a different cross-fit interface. Use the complete preparation and execution steps in [REGENERATE.md](REGENERATE.md). The five-block design, 73-hour buffer and other methodological details are in [CROSSFIT_METHODOLOGY.md](CROSSFIT_METHODOLOGY.md).

## Outputs and validation

The pipeline writes local `crossfit_results/` folders for the four forecast targets, a combined training-day CSV, and an `audit/` folder with verification reports. After the verification gate reports `READY`, `install_outputs.py` assembles `results/forecast/ddqn_state_forecasts.csv` for evaluation or training. These household-derived outputs and input data are not part of the public release and must remain local under the applicable data licence.

`READY` means the cross-fit verification gate passed; it does not mean the forecast file is byte-identical to the original. The installer reports hash comparisons, and [REPRODUCIBILITY.md](../REPRODUCIBILITY.md) explains the observed differences.
