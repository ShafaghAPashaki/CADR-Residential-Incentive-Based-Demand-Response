# Change log

## public_release_v4 (repository packaging only; no change to models, configurations or results)

- Moved the original cross-fitting files into `forecast_regeneration/` together with the exact forecasting sources and configuration snapshot used on 5 October 2026 (`predict_load_lstm.py` e8781487, `predict_price_lstm.py` 7f4a7863, `config_seed_0.yaml` ba3588b6) and their original manifests. Added `install_outputs.py` and `REGENERATE.md`.
- Added `tools/evaluate_with_regenerated_forecasts.py` (relaxes only the forecast-state hash check and labels results as approximate) and `tools/export_aggregate_metrics.py`.
- Added `results/aggregate_metrics/per_seed_monthly_metrics.csv` and `TABLE_MAP.md`.
- Replaced `PUBLIC_REPRODUCIBILITY_LIMITATIONS.md` and `FORECAST_INPUT_REGENERATION_NOTE.md` with `REPRODUCIBILITY.md`; rewrote `README.md`; extended `data/README.md` and `.gitignore`.

## causal_state_v1

- Made the TS-NI state feature causal: it no longer reads realised appliance observations after the current decision hour.
- Preserved the existing physical HEMS, settlement, and reward logic in this state-only update.
- Added direct state-causality and state-feature equivalence regression tests.
- Added the active environment version to Q-action diagnostic metadata.

The remaining future-hour scheduling logic is intentionally unchanged in this version and is handled by subsequent causal-environment updates.

## causal_capacity_v1

- Replaced realised future-load capacity screening in EV and TS-NI destination
  scheduling with a causal 24-hour persistence reference from the immediately
  preceding day.
- Applied only commitments already created in the current episode when
  estimating future capacity usage.
- Removed direct consultation of future realised appliance demand when
  screening TS-NI destination slots; only already committed shifted-in load is
  used for double-booking protection.
- Added future-capacity causality and TS-NI overlap regression tests.
- The current declared TS-NI programme profile remains available to the
  household scheduling routine; future price handling is unchanged in this
  version.

## causal_capacity_v2

- Replaced the one-day persistence capacity reference with the mean of the
  same wall-clock hour over up to the preceding seven days, selected using
  training/validation data only.
- Added causal TS-NI conflict deferral when an already committed shifted-in
  programme would overlap a newly declared programme of the same appliance.
  Conflict deferrals are tracked separately and are not incentive-settled.
- Added a per-hour EV shifted-energy commitment cap from the configured device
  rating. Coincidence with future original EV requests remains an uncertainty
  diagnostic rather than a source of look-ahead.
- Added direct-state, destination-causality, TS-NI overlap, EV commitment-cap,
  and exhaustive whole-step future-load regression tests.
- Future-price handling is unchanged in this version.

## causal_capacity_v3

- Added dissatisfaction accounting for forced TS-NI conflict deferrals and
  explicit unmet-energy accounting when no feasible same-day slot remains.
- Unserved TS-NI and EV energy now always carries positive next-day-equivalent delay dissatisfaction, so late-day unmet energy cannot be cost-free.
- Enforced the configured EV charger rating at execution time. Newly observed
  EV demand that conflicts with earlier commitments is deferred causally, with
  delay dissatisfaction and explicit unmet-energy accounting at the deadline.
- Extended EV scheduling headroom to include both incentive-shifted and
  physical-deferral commitments.
- Added physical execution and energy-balance regression checks.
- Updated the end-to-end causality test so legitimate next-hour conflict
  resolution is not misclassified as current-hour look-ahead.

## causal_price_v1

- Removed future realised wholesale-price observations from TS-NI request
  admissibility. The locked incentive is now checked only against the price
  available at the current request hour.
- Preserved locked-rate settlement at later verification hours even when the
  realised wholesale price is lower. The service provider absorbs any negative
  settlement margin through the existing economic accounting.
- Added explicit diagnostics for above-price locked-rate settlement energy,
  event counts, and negative settlement margin.
- No DDQN retraining is included in this candidate.

- Frozen after independent cross-check confirmed zero future-load and
  future-price sensitivity in the whole-step look-ahead sweep.
- The request-hour incentive bound remains as a safeguard even though the
  configured action mapping satisfies it by construction.
## forecast_state_v1

- Integrated one-step household-load and wholesale-price forecasts into the DDQN decision state without changing its 43-element dimension.
- Routed days 2-167 to frozen cross-fitted/out-of-fold forecasts and days 168 onward to the fixed full-model forecasts.
- Forecasted whole-house load now drives the household pre-action load, aggregate pre-action load, capacity-need flag, and required-reduction state fields.
- Current appliance-flexibility and commitment fields remain observed/known quantities.
- Realised load and price remain authoritative for HEMS response, physical execution, reward, settlement, capacity evaluation, and the request-hour incentive bound.
- Added forecast-source provenance, an `oracle_actual` mapping diagnostic, and forecast-state regression tests.
- No DDQN retraining is included in this candidate.


## forecast_state_v1 freeze

- Forecast-state integration frozen after independent source reconstruction,
  timestamp alignment, oracle-equivalence and state-field isolation checks.
- The loader now verifies fixed full-model provenance for every complete day
  from validation onward, including July and November.

## benchmark_fair_forecast_v1

- Preserved the Appendix-D flat daily EBDR incentive structure while removing
  same-day price look-ahead: the daily reference is now the previous day's
  minimum strictly positive wholesale price.
- Routed the same frozen DDQN forecast table used by CADR into the five EBDR
  state entries for three household loads, aggregate load and wholesale price.
- Training uses cross-fitted forecasts; every day from validation onward uses
  the frozen full-model forecasts.
- EBDR physical response and reward/economic accounting remain based on
  realised baseline demand and realised wholesale price.
- No EBDR DDQN retraining has been performed.

## benchmark_fair_forecast_v1 freeze

- Frozen after independent exhaustive future-price testing, perfect-forecast
  equivalence across all days 2-243, field-isolation checks, reward-equivalence
  checks, and confirmation that the benchmark remains capacity-blind.
- Appendix-D incentive timing is preserved as a flat daily mapping using the
  previous day's minimum strictly positive wholesale price.
- The benchmark state uses the same frozen forecast provenance and routing as
  CADR.

## final_preflight_retained_lambda_v1 candidate

- Retain original λc=0.2216381800923157 instead of sample-size-sensitive
  train-only recalibration; training now validates a documented retention audit.
- Removed automatic λc recalibration from the final launcher; its calibration
  utility is diagnostics-only and refuses config writes.
- Added separate conflict-induced rebound hours/excess to final policy and
  method-specific test outputs while preserving original incentive-shift rebound.
- Recorded archived-forecast versus state-clipped-forecast distinction in the
  fixed confusion report metadata and protocol.
- Preserved strict genuine holidays==0.100 dependency and training-provenance
  capture; no production retraining has been executed.
