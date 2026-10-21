# Changelog

## Initial Research Release — CADR

This is the first public release of the code, frozen configurations, trained checkpoints and aggregate results that accompany the manuscript.

**Methods**
- CADR: a capacity-aware service-provider DDQN with appliance-level HEMS coordination of three households, with a soft 7-kW capacity target.
- EBDR: an elasticity-based, capacity-blind benchmark DDQN trained under the same data splits, seeds and forecast inputs.
- CBSP: a matched-HEMS ablation that keeps the capacity-aware HEMS but trains the service provider on an economic-only objective without the capacity-need state features.

**Methodological safeguards**
- The decision state is causal. Realised future load and price are not used for state construction, appliance scheduling or request admissibility.
- One-step LSTM forecasts of household load and wholesale price enter the DDQN state. Training days use blocked five-fold cross-fitted forecasts; validation and test days use full-model forecasts.
- EBDR uses the same forecast inputs as CADR and a causal previous-day price reference.
- Settlement, reward and physical response use realised quantities. This includes unserved-energy accounting, conflict deferral and EV charger limits.
- λc is retained at its original calibrated value, and checkpoints are selected on validation reward only.

**Experiments**
- Three seeds per method, evaluated on July and August 2018.
- Capacity-target sensitivity (6 and 8 kW), incentive-discretisation sensitivity (10 levels) and reward-parameter sensitivity (η, ρ).
- Inference-only value-of-information and forecast-confusion diagnostics.

**Reproducibility**
- Checkpoints and their `run_provenance.json` files record the environment version, configuration, code and forecast-input hashes. Evaluation refuses a checkpoint whose recorded metadata do not match the active configuration and forecast inputs.
- `forecast_regeneration/` regenerates the forecast inputs from licensed data. `tools/evaluate_with_regenerated_forecasts.py` evaluates approximately with regenerated forecasts and labels the results as approximate.
- `results/aggregate_metrics/per_seed_monthly_metrics.csv` and `TABLE_MAP.md` link every manuscript table to its source.
- Licensed household data and household-level derived series are not distributed.

The implementation history before this release is preserved in `docs/DEVELOPMENT_HISTORY.md`.
