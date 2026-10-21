# Evaluation protocol and checkpoint selection

*This protocol was fixed before any July or August test results were produced. It is reproduced unchanged except for this title and note. References to later experiments describe the plan at that time; those experiments are documented in `docs/EXPERIMENTS.md`.*

This protocol must not be revised in response to July/August test performance.
Use **validation mean reward only**, separately per method and seed, to retain
best checkpoints. Final periods: July 182–212, August 213–243, and November
305–334 (low-demand robustness). Three seeds: 0, 1, 2. Normalised state
forecast source: crossfit in training; frozen full model from day 168 onward.

## Primary outcomes

- Baseline/controlled peak demand (kW); peak reduction (%).
- Baseline/controlled PAR and PAR reduction (%).
- Hours above 7 kW and reduction in violation hours.
- Aggregate cumulative excess above 7 kW (kWh) and reduction.
- EU utility, SP economic value, total incentive payment and discomfort.

## Required physical and economic diagnostics (CADR)

- Rebound-induced violation **hours**: shifted-in load present, baseline <= 7,
  realised post-DR >7, and post-DR minus shifted-in <=7 (as in `test.py`).
- Rebound-induced **excess energy**: sum of (post-DR - 7) kWh in those hours.
- **Separate** conflict-induced rebound violation hours and excess (kWh),
  using `device_conflict_deferred_in.sum(axis=(1,2))` at each hour and the same
  causal-attribution counterfactual (`post-DR − conflict arrival <= Lcap`).
  The primary incentive-shift rebound definition above is unchanged.
  The two categories may overlap; never sum them as disjoint counts.
- Unserved energy = `device_unmet.sum()` (kWh), always shown next to capacity metrics.
- Forced conflict deferral out/in energy, `device_conflict_deferred_out/in.sum()`.
- Conflict dissatisfaction = `device_conflict_discomfort.sum()`.
- Above-price locked settlement: affected device-hours, energy, and negative
  settlement margin. This is a diagnostic; never subtract again from reward.
- EV realised charger rating violations; TS-NI double-booking; energy balance.

## Forecast error and inference-only value of information

- Fixed 7-kW confusion counts: TP, FP, FN, TN for training, validation, July,
  August, and November. This descriptive report uses the **unfloored** archived
  aggregate forecast; the agent state floors negative household forecasts
  at zero before summing. The two quantities are intentionally distinguished
  in the output metadata. No model selection uses either confusion count.
- Trained policies are evaluated **twice**, once with production forecast state
  and once with the state set to `oracle_actual`, with identical checkpoint and
  real physical demand/price dynamics. The latter is an inference-only
  value-of-information diagnostic. It does not tune or reselect checkpoints.
- Report performance difference without implying oracle necessarily improves
  a policy trained on forecast errors.

## Limits and reproducibility

- No July/August model choice, calibration, or checkpoint selection.
- EBDR remains capacity-blind in its state, actions, reward and training; use
  7 kW only as an external reporting reference.
- The 6/8 kW CADR sensitivity and the capacity-blind CADR ablation are later
  separate experiments, all using identical frozen forecast inputs. Recompute
  capacity-dependent economic normalisation and δ = 0.1033/Lcap. Keep the
  original calibrated λc fixed across 6, 7 and 8 kW rather than rerunning a
  scenario-selection-sensitive calibration.
- All results labelled with exact environment, holiday backend, forecast hash,
  configuration SHA-256, code hashes, and original-λc retention-audit hash.
