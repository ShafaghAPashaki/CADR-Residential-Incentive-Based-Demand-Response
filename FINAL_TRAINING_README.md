# Final training preflight — fixed original λc

Scientific environments are frozen: CADR `forecast_state_v1`; EBDR
`benchmark_elasticity_env_v3_fair_forecast`. Their forecast table is shared.
All three CADR configs retain **λc = 0.2216381800923157** with source label
`train_p95_scenario_action_matching_original_retained`.

## Why the original λc is retained

The historical train-only Q95 stakeholder/tracking calibration yielded
0.138830910593618 / 0.626385357142857 = 0.2216381800923157. Its twelve
original scenario IDs were not archived. Independent re-evaluation using a
new, deterministic training-only scenario sample found virtually no difference
between legacy and revised causal dynamics at comparable sample sizes, but
estimates varied considerably when the **selection sample itself** changed.
Substituting such a result would introduce sample-selection arbitrariness.
Therefore the original λc is retained and the supplied independent review
numbers are recorded (as reviewer-reported, not re-executed) in
`results/calibration/lambda_c_retention_audit.json`.

`calibrate_lambda_c.py` remains an optional diagnostic only; it writes its
records in `results/calibration_diagnostics` and refuses `--write-config`.
**Do not use it to change the three production seed configs.**

## Local Windows workflow

1. Activate the real project's Python environment. Install pinned dependencies
   from `requirements.txt` if required. The final run requires genuine
   `holidays==0.100`, verified with an actual US calendar self-test; no stub.
2. Run `python verify_retained_lambda.py`. This verifies all three 2500-episode
   CADR seed configs, the original λc retention record, the unchanged scientific
   environment hash, and the frozen forecast hash. No training is performed.
3. Launch `powershell -ExecutionPolicy Bypass -File .\run_final_retraining.ps1`.
   The launcher fails closed on invalid dependencies/retention, refreshes source
   manifests and saves frozen forecast-confusion reports, then trains CADR
   seeds 0/1/2 and EBDR seeds 0/1/2 with validation-selected checkpoints.
4. Retain each checkpoint and its `run_provenance.json`. Do not adjust the
   hyperparameters or select based on July or August test performance.
5. Run method-specific testing for July, August and November, then
   `.\run_inference_only_oracle_checks.ps1` for inference-only value of information.

Final evaluation definitions are predeclared in `FINAL_EVALUATION_PROTOCOL.md`.
The primary incentive-shift rebound definition is unchanged; physical-conflict
rebound has a separate metric. A single hour may satisfy both categories;
the categories must **not** be summed as mutually exclusive events.

The capacity-blind CADR ablation and the 6/8-kW sensitivity experiments come
later, without test-set tuning. Keep original λc across 7/6/8-kW variants;
recompute automatic economic scale and δ = 0.1033/Lcap. Alternative-capacity
run guards/configs will be prepared separately.

This candidate omits all old seed checkpoint folders to prevent run collisions.
If your execution machine does not have the genuine pinned dependencies,
**do not launch**; the guard must stop rather than silently substituting data.
