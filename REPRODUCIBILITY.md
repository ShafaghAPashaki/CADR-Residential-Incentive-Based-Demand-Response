# Reproducibility: what can be reproduced, and how

## 1. What is distributed

| Distributed | Not distributed (licensed data, or derived from it at household level) |
|---|---|
| All source code, configurations and tests | Pecan Street household load data |
| Selected checkpoints (CADR, EBDR, CBSP; capacity and incentive-level sensitivity runs) | Hourly evaluation workbooks (`audit.xlsx`, EBDR hourly sheets) |
| Seed-level monthly aggregate metrics (`results/aggregate_metrics/`) | Full-model LSTM forecast series |
| Validation curves, sensitivity summaries, forecast metrics, provenance and manifests | The DDQN forecast-state file `results/forecast/ddqn_state_forecasts.csv` |
| The original forecasting and cross-fitting sources (`forecast_regeneration/`) | ERCOT price series (obtain from the source cited in the paper) |

`TABLE_MAP.md` maps every manuscript table and figure to its data file and script.

## 2. Exact reproduction with the original inputs

All 18 primary evaluations were independently re-run from the released checkpoints with the current code: CADR, EBDR and CBSP, three seeds each, in July and August. The re-runs used the original licensed inputs and the original forecast-state file (SHA-256 `b86e09ba…`). Every numeric cell of every evaluation workbook was identical to the stored outputs.

The same holds for:
- the capacity runs at 6 and 8 kW and the ten-level incentive run (July, Seed 0), now in `results/capacity_incentive_sensitivity/`;
- the checkpoints after the local machine paths were removed from their metadata (seed 0, July, all three methods). The tensors themselves are unchanged.

## 3. Regenerating the forecast inputs from licensed data

`test.py` refuses to evaluate a checkpoint unless the forecast-state file is byte-identical to the one used in training. This check is deliberate and has been kept.

`forecast_regeneration/REGENERATE.md` gives the full pipeline:

1. full-model LSTMs;
2. blocked five-fold cross-fitting;
3. verification gate;
4. building the state file;
5. a stage-by-stage hash comparison.

In an independent rerun on a different platform:

| Stage | Result |
|---|---|
| Cross-fit training-day forecasts | Byte-identical (`7ffbed0c…`) |
| Full-model forecasts (originally produced on the authors' Windows workstation) | Not byte-identical. Maximum differences: 8.8e-6 kW for load and 1.2e-4 cent/kWh for price. Best epochs were unchanged. |
| Resulting state file | Not byte-identical (`575b0ab9…`). Training-day rows are identical; validation and test rows differ by no more than those amounts. |

The forecasting scripts at the project root are later, non-functional edits of the originals. On the same machine they produce byte-identical forecasts to the originals.

## 4. Approximate evaluation with regenerated forecasts

`tools/evaluate_with_regenerated_forecasts.py` runs the unmodified evaluation code and relaxes exactly one check: the forecast-state hash. All other checkpoint and environment checks still apply. When the hashes differ, the output folder receives `APPROXIMATE_FORECAST_INPUTS.json`, and the results must be reported as an approximate regeneration.

With the regenerated state file described above, the following evaluations gave evaluation workbooks numerically identical to the published ones (0 differing numeric cells):
- CADR: all three seeds, July and August;
- CBSP: seed 0, July;
- EBDR: seed 0, July.

EBDR's evaluation script does not check the forecast hash.

This is empirical evidence for these runs only. It is not a guarantee for other platforms, because larger forecast differences could change a greedy action.

With a regenerated state file, `verify.py`, `python -m benchmark.verify_benchmark` and 16 of the 17 test scripts in `tests/` pass, as does `benchmark/tests/test_fair_forecast_state.py`. The exception is `tests/test_retained_lambda_preflight.py`, which stops with "frozen forecast SHA changed". This is intended: it guards final retraining, which requires the original forecast-state file.

## 5. What remains unverifiable without the authors

- **The exact `b86e09ba…` state file** can be rebuilt only if the full-model forecasts are reproduced byte-for-byte. That appears to need the authors' original Windows CPU environment. Researchers who hold a Dataport licence may request the original file from the authors, subject to Pecan Street's terms.
- **The full-model LSTM manifests** in `results/forecast/{load,price}/` record script and configuration hashes (`4161ade2…`, `bc62a62f…`, `3915eae1…`) from a later re-run on the authors' workstation. That re-run produced the same prediction hashes, but those intermediate script versions are not archived. The original 5-October manifests, whose hashes match the scripts in `forecast_regeneration/`, are in `forecast_regeneration/reference/`.
- **Training provenance.** The CADR and EBDR `run_provenance.json` files record older hashes for the `main.py`, `test.py` and `benchmark/test_benchmark.py` entry points.
  - *Evaluation:* the current versions reproduce every stored evaluation exactly.
  - *Training:* retraining Seed 0 with the current `main.py` gave bit-identical network weights. Seeds 1 and 2 have not been retrained.
- **The 619.97 s training time and the inference timings** were measured on the authors' workstation. Only the inference measurements are archived (in the private project).

## 6. Verification of the current repository layout

The files were later reorganised (experiment drivers into `experiments/`, diagnostics into `diagnostics/`, and two result folders renamed; see `docs/FILE_RENAME_MAP.csv`). Checkpoints, configurations, results and the core code (`env/`, `agent/`, `utils/`, `main.py`, `test.py`, `benchmark/`, `tests/`, `forecast_regeneration/`) are byte-identical to the previous release. After the reorganisation, with the licensed inputs and the original forecast-state file, the following were run again:

- `verify.py`, `python -m benchmark.verify_benchmark`, all 17 `tests/test_*.py` scripts and `benchmark/tests/test_fair_forecast_state.py`: all pass.
- `experiments/check_matched_hems_ablation.py` and `diagnostics/verify_retained_lambda.py`: pass.
- Re-evaluation of CBSP Seed 0 July (through `tools/evaluate_with_regenerated_forecasts.py`, which correctly stayed inactive with the original file), of the `cap6`, `cap8` and `lev10` July runs, and of CADR Seed 0 July: every evaluation workbook was identical to the stored one (0 differing cells).

Training was not repeated after the reorganisation.
