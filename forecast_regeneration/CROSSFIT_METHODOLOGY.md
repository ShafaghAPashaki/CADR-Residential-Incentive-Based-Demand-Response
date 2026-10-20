# Cross-fitting of training-day forecasts — methodology note

## Purpose

The DDQN is trained on days 2–167 of 2018. If the forecasts it observes on those
days come from LSTMs fitted on the same days, they are in-sample and more
accurate than the forecasts the policy will see at validation/test time. The
policy would then learn to trust forecasts more than is warranted.

The pre-registered train-versus-validation WAPE gate (relative gap > 0.20)
flagged this for household 8565. Cross-fitting is applied to **all four**
forecasting targets (households 661, 3039, 8565 and the wholesale price) so
that every training-day forecast supplied to the DDQN is out-of-fold. This
stage is not intended to improve forecast accuracy and does not alter any
validation or test forecast.

## Design

| Element | Choice |
|---|---|
| Period | DDQN/LSTM training days 2–167 (3,984 h) |
| Folds | 5 contiguous day blocks: days 2–35 (34 d), 36–68, 69–101, 102–134, 135–167 (33 d each) |
| Held-out prediction | Each block is forecast by a model not fitted on that block |
| Exclusion buffer | 73 h on both sides of the held-out block |
| Fitting set | Training-period samples whose target lies outside the block and its buffer, excluding startup-padded samples |
| Model | Unchanged Appendix-A LSTM: 14 inputs per step, 24-step window, 1-h horizon, 2 × 64, dropout 0.2, Adam/MSE, same learning rate, weight decay and batch size |
| Scalers | Fitted on each fold's accepted fitting samples only |
| Epochs | Fixed per target = best validation epoch of the corresponding full model (661: 22, 3039: 13, 8565: 3, price: 19). No search; days 168–181 are not consulted by the folds |
| Seeds | 42 + fold index (0–4) for Python, NumPy and PyTorch; CPU, deterministic algorithms |

### Why 73 hours

A sample with target hour *t* uses observations from *t − 73 h* to *t − 1 h*
(24 recurrent steps, each carrying lags up to 50 h: 23 + 50 = 73). Excluding
fitting targets within 73 h after a held-out block guarantees that no fitting
sample carries a held-out observation in its inputs. The buffer before the
block is not strictly required (those samples' inputs precede the block), but
it is applied on both sides for symmetry and auditability.

### Why blocked folds rather than a rolling origin

The forecasts produced here are used only to train the controller offline, on
historical days whose realised outcomes the simulator already replays. No
validation or test data enters any fold, and all validation/test forecasts
remain strictly causal (fitted on days 2–167 only). Relative to the in-sample
forecasts they replace — which used the same later months and the target hour
itself — out-of-fold forecasts strictly remove information. A rolling-origin
scheme would leave January without any forecast (no earlier data exist) and
would fit the earliest folds on a few weeks of winter data, producing
training-day errors much larger than test-time errors. Cross-validation is valid
for autoregressive forecasters with lagged inputs (Bergmeir, Hyndman & Koo,
2018, *Computational Statistics & Data Analysis* 120:70–83).

### Startup padding

The load series starts at 06:00 on 1 January and the price series at 00:00, so
the earliest training hours have input windows reaching before the start of the
data. As in the full models, those inputs are zero-padded. These samples are
excluded from fitting, scaling and metrics, but are still forecast (by the
fold-0 model) so the DDQN has complete coverage. This affects 55 load hours
(up to 06:00 on 4 January) and 49 price hours (up to 00:00 on 4 January); they
are marked `inputs_padding_free_all_targets = False` in the combined file.

## What is unchanged

Validation forecasts (days 168–181), July and August test forecasts, Table A.9
and Figure A.6 all continue to come from the existing full models. The run
hashes every file under `results/forecast/` before and after training and
verifies that none has changed. The DDQN will use:

* days 2–167 → `crossfit_results/combined/ddqn_crossfit_forecasts_days_002_167.csv`
* days 168 onward → existing full-model forecasts

## Checks performed

* **Provenance:** `predict_load_lstm.py`, `predict_price_lstm.py` and the config
  must have the SHA-256 recorded in the full-model manifests; otherwise the run stops.
* **Leakage (every fold, every target):** held-out and buffer targets absent from
  the fitting set; no fitting input window overlaps the held-out block; an empirical
  test perturbs every held-out observation by +10⁴, rebuilds all inputs, and requires
  the fitting inputs to be bit-identical; fitting targets and inputs end before day 168.
* **Coverage:** chronological, no duplicates, exact hourly spacing, exactly days 2–167.
* **Aggregate identity:** aggregate = EU1 + EU2 + EU3 at every hour (≤ 1e-9 kW).
* **Units:** price cent/kWh = $/MWh ÷ 10 (≤ 1e-9).
* **Determinism:** fold 0 of every target is retrained from scratch and must
  reproduce its predictions (≤ 1e-6).
* **Hashes:** SHA-256 of every output artifact, re-verified independently.

## Suggested manuscript sentence

> Forecasts supplied during DDQN training (days 2–167) were generated
> out-of-fold using five contiguous blocks with 73-h exclusion buffers, so that
> no training-day forecast was produced by a model fitted on that day's data;
> all fitting used training-period data only, and validation and test forecasts
> were produced by the models fitted on the full training period.
