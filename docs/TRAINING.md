# Training

The step-by-step CADR workflow, including all three seeds, is in [README.md](../README.md). Training is not needed to inspect results or evaluate a published checkpoint. Each CADR seed runs for 2,500 episodes and selects its checkpoint by validation reward; never select a model using test-period results.

After preparing the authorised inputs and common forecasts, use `tools/run_cadr.py train`. With regenerated forecasts, add `--regenerated-forecasts`; with the original file, omit the option to keep the original strict hash check. Outputs and run provenance are written to a new `repro_runs/train/seed_<seed>/training/` directory. The launcher can select seeds with `--seeds` and a different destination with `--train-root`.

The CADR environment version is `forecast_state_v1`. All three CADR configurations retain λc = 0.2216381800923157. The historical training-only calibration and its public-data boundary are documented in [LAMBDA_C_HISTORICAL_CALIBRATION.md](LAMBDA_C_HISTORICAL_CALIBRATION.md). `calibrate_lambda_c.py` is an optional diagnostic; it must not replace the frozen setting.

For EBDR training see [benchmark/README.md](../benchmark/README.md); for ablation and sensitivity experiments see [EXPERIMENTS.md](EXPERIMENTS.md). The fixed evaluation protocol is in [EVALUATION_PROTOCOL.md](EVALUATION_PROTOCOL.md), and the limits on exact reproduction are in [REPRODUCIBILITY.md](../REPRODUCIBILITY.md).
