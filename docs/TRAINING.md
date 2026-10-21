# Training

Retraining is not needed to inspect or re-evaluate the published results. It needs the licensed input data, the original forecast-state file (SHA-256 `b86e09ba…`) and considerable CPU time: 2,500 episodes per run, with validation every 50 episodes.

## Frozen settings

The CADR environment version is `forecast_state_v1` and the EBDR environment version is `benchmark_elasticity_env_v3_fair_forecast`. Both methods use the same forecast-state table. All three CADR configurations retain **λc = 0.2216381800923157** with the source label `train_p95_scenario_action_matching_original_retained`.

## Why the original λc is retained

The historical training-only Q95 stakeholder/tracking calibration gave 0.138830910593618 / 0.626385357142857 = 0.2216381800923157. The twelve original scenario IDs used for that calibration were not archived. An independent re-evaluation with a new, deterministic training-only scenario sample found virtually no difference between the earlier and the revised causal dynamics at comparable sample sizes. However, the estimates varied considerably when the selection sample itself changed, so substituting a new estimate would introduce sample-selection arbitrariness. The original λc is therefore retained. The independent review figures are recorded, as reported and not re-executed, in `results/calibration/lambda_c_retention_audit.json`.

`calibrate_lambda_c.py` is an optional diagnostic only. It writes to `results/calibration_diagnostics/` and refuses `--write-config`. Do not use it to change the three production configurations.

## Workflow (Windows PowerShell; the Python commands are platform-independent)

1. Install the pinned dependencies from `requirements.txt`. Training requires the genuine `holidays==0.100` package, which is checked with a US-calendar self-test.
2. Run `python diagnostics/verify_retained_lambda.py`. It checks the three 2,500-episode CADR configurations, the λc retention record, the hash of the frozen scientific environment and the hash of the frozen forecast file. No training is performed.
3. Run `powershell -ExecutionPolicy Bypass -File .\run_final_retraining.ps1`. The launcher stops on invalid dependencies or retention records, refreshes the source manifests (`tools/refresh_source_manifests.py`), saves the forecast-confusion report, and then trains CADR seeds 0–2 and EBDR seeds 0–2 with validation-selected checkpoints. A single CADR seed can be trained with `python main.py --config_path config_seed_0.yaml`.
4. Keep each checkpoint with its `run_provenance.json`. Do not change hyperparameters, and do not select checkpoints using July or August test performance.
5. Evaluate July, August and November (`test_all_seeds.ps1`, `benchmark/test_benchmark_seeds.ps1`), then run `run_inference_only_oracle_checks.ps1` for the inference-only value-of-information comparison.

An independent retraining of CADR Seed 0 with the released `main.py` reproduced the stored network weights bit for bit. Seeds 1 and 2 have not been retrained independently. The retained checkpoints and their selected episodes are listed in `results/checkpoint_selection.csv`.

The evaluation definitions are fixed in `docs/EVALUATION_PROTOCOL.md`. The primary incentive-shift rebound definition is separate from the physical-conflict rebound metric. A single hour can belong to both categories, so the two must not be summed as mutually exclusive counts.

## Other training runs

The matched-HEMS ablation, the capacity-target and incentive-discretisation sensitivity runs, and the reward-parameter sensitivity runs have their own drivers in `experiments/`, described in `docs/EXPERIMENTS.md`. They keep the original λc. The 6- and 8-kW runs recompute the capacity-dependent economic normalisation from the configuration, and they replace only the 7-kW equality check of the final-run gate with a check against their predeclared capacity.
