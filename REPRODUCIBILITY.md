# Reproducibility

The public release contains source code, configurations, selected checkpoints, validation records, and seed-level aggregate results. Licensed household data, household-level forecasts and hourly audits are not distributed. Obtain the data as described in [data/README.md](data/README.md). [README.md](README.md) gives the steps for inspection, checkpoint evaluation and new training.

The historical `results/forecast/README_OUTPUTS.md` describes a complete local forecast run, not a promise that every listed file is included in this public release.

## What has been checked

- **Published results:** The archived July and August CADR, EBDR and CBSP checkpoint evaluations for all three seeds were re-run with the original authorised inputs and forecast-state file; reported numeric workbook cells matched their archived counterparts. These are checkpoint evaluations, not fresh training.
- **A reader-style run on Windows:** The load and price data were prepared from licensed raw exports, the LSTM/cross-fit pipeline completed its gate, and the published CADR Seed-0 checkpoint was evaluated for July. The regenerated forecast-state hash differed from the historical hash, while the five resulting `results.xlsx` sheets matched the stored workbook in that run.
- **CADR training is an independent rerun, not guaranteed bitwise reproduction.** One author-environment Seed-0 training with the original forecast-state file produced the published weights and July results; this is a historical observation specific to that environment, not a portable guarantee. In an independent Linux/Python 3.12.3, 2-CPU audit using pinned dependencies and the original forecast-state hash, the validation curves first differed after episode 850 and selected episode 2050 instead of 2400. July daily-peak reduction was 21.65% instead of 23.35%. A separate author rerun using regenerated forecasts selected another checkpoint and obtained 21.12%. CPU threading, floating-point operations and software/platform builds can change training trajectories and checkpoint selection. The author run's exact thread count and PyTorch build are not archived here and cannot be asserted. Matching weights has **not** been established for all seeds or other platforms. Independent training results must be reported as new runs, never substituted into the frozen manuscript results.

The reader-style run used a downloaded repository, an existing Python environment, and manual stages. It does not verify the single-command runner from a new environment or exact end-to-end reproduction for every seed. Recorded inputs, hashes and software versions should accompany any new reproduction.

## Forecast inputs and evaluation

The original DDQN forecast-state SHA-256 is `b86e09baae30d3093ecb4cf8efb71dd273104739fc690120807edda8df88cd17`. The forecast file is household-derived and is not included. Where permitted by the data provider, authorised researchers may obtain the original file separately; the released pipeline can instead regenerate it from authorised data.

`tools/evaluate_with_regenerated_forecasts.py` relaxes only the released checkpoint's forecast-state hash check. If the hash differs, it writes `APPROXIMATE_FORECAST_INPUTS.json`; all other checkpoint/environment checks still apply. The general CADR launcher uses this wrapper for published checkpoints and the strict `test.py` path for newly trained checkpoints. Always use the same forecast-state file for new training and its evaluation.

The forecast pipeline retains the published sources, blocked five-fold training-day forecasts and verification gate. A previous Linux rerun reproduced the historical cross-fit CSV byte-for-byte; the later Windows reader-style rerun did not. Forecast differences can affect subsequent DDQN training, and exact results from newly obtained inputs or another environment are not guaranteed. [forecast_regeneration/REGENERATE.md](forecast_regeneration/REGENERATE.md) documents the stage hashes and procedure.

## Sensitive source data and plots

The public release tree intentionally excludes household-level hourly load and appliance-scheduling plots, household-profile benchmark plots, the forecast-comparison image and PDF, household-level forecast states, and detailed audits. Those outputs can be regenerated locally by authorised users and must not be committed or redistributed without the applicable permission. Aggregate three-household figures and metrics remain. **Deleting tracked figures from the current tree does not purge earlier Git commits or external mirrors**; the public Git history must not be described as a cleansed data release.

## Training and provenance

The released CADR configurations use the retained historical λc and validation-based checkpoint selection. Do not select checkpoints on the test periods. An author-specific Seed-0 weight match and an independent Linux Seed-0 mismatch are both documented above. Independently matching retraining for seeds 1 and 2 has not been established. `results/checkpoint_selection.csv` lists the archived selections. The historical λc provenance and source-data boundary are documented in [docs/LAMBDA_C_HISTORICAL_CALIBRATION.md](docs/LAMBDA_C_HISTORICAL_CALIBRATION.md).

The original, immutable calibration audit is restored under `results/calibration/lambda_c_retention_audit.json` to match the SHA-256 stored in the published runs. The privately recovered historical-calibration evidence is described separately in `results/calibration/lambda_c_historical_recovery_metadata.json` and `docs/LAMBDA_C_HISTORICAL_CALIBRATION.md`; no licensed scenario observations are released. Historical recorded source-code SHA-256 values need not match post-release script revisions; see [Provenance compatibility](docs/PROVENANCE_COMPATIBILITY.md).

The LSTM manifests in `forecast_regeneration/reference/` match the archived forecast-regeneration scripts. Some later manifests and CADR/EBDR run provenance record earlier or later versions of entry-point scripts; stored checkpoints, configuration and evaluation outputs remain available for comparison. See [docs/DEVELOPMENT_HISTORY.md](docs/DEVELOPMENT_HISTORY.md) for the historical layout and [TABLE_MAP.md](TABLE_MAP.md) for manuscript mappings.
