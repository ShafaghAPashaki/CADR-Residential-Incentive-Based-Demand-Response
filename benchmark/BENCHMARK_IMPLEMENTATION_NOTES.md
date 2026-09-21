# Benchmark implementation notes for independent audit

## Scope

The archived elasticity-based DDQN benchmark was converted into an isolated,
reproducible three-seed pipeline without adding capacity awareness to its
policy. The proposed method's `main.py`, `env/env.py`, configs, checkpoints and
results were not modified.

## Scientific formulation retained

The active benchmark remains an aggregate, curtailment-only response model.
For household `n` and hour `h`:

```text
DeltaE[n,h] = clip(
    baseline[n,h] * elasticity[h] * raw_action[n,h],
    0,
    max_reduction_fraction * baseline[n,h]
)
```

```text
R_SP[h] = price[h] * sum_n DeltaE[n,h]
          - sum_n lambda[n,h] * DeltaE[n,h]

R_EU[n,h] = rho * lambda[n,h] * DeltaE[n,h]
            - (1-rho) * (0.5 * mu[n] * DeltaE[n,h]^2
                         + kappa * DeltaE[n,h])

R_total[h] = alpha * R_SP[h] + (1-alpha) * sum_n R_EU[n,h]
```

No capacity, tracking, no-need, offer penalty, appliance feasibility,
load-shifting, or proposed-method reward-normalisation term was introduced.

## Executable changes from the archive

1. **Independent training entry point**
   - Added `main_benchmark.py`.
   - Uses the shared `DDQNAgent` but a benchmark-specific environment and
     metadata boundary.
   - Best checkpoint is selected only by deterministic mean validation reward
     across every day 168–181; lower validation-reward SD breaks a numerical
     tie and the earliest episode is retained in a complete tie.

2. **Three independent seeds**
   - Added configs for seeds 0, 1 and 2.
   - They differ only in `general.seed`.
   - Train days 2–167, 2500 episodes, and the shared DDQN hyperparameters are
     frozen.

3. **State cleanup**
   - Removed three archived current-reduction inputs that were always zero when
     the current action was selected.
   - The active state has 12 features: three household baselines, clipped-log
     current price, elasticity, normalized hour, aggregate baseline, holiday,
     weekend, and on/mid/off-peak flags.
   - No capacity-derived feature is present.

4. **Response wiring**
   - Implemented the algebraically simplified archived response directly.
   - `max_reduction_fraction` is now read from the config rather than silently
     replaced by a hard-coded 0.3.

5. **Negative-price correction**
   - Real hourly prices, including negative values, remain in `R_SP`.
   - Day-level incentive bounds use the smallest strictly positive price in the
     day, preventing one negative hour from inverting the bound gap and
     disabling response for the whole day.
   - A hypothetical day with no positive price has zero nominal bounds and zero
     physical response.
   - This is a dynamics change relative to the buggy archived implementation;
     all official benchmark seeds therefore require fresh training.

6. **Zero-action convention retained**
   - Raw action zero maps to nominal `lambda_min` as in the archive.
   - It produces zero response and zero payment.
   - Intervention metrics use `raw_action > 0`; plots mask the inactive nominal
     floor while the audit ledger preserves it.

7. **Full evaluation pipeline**
   - July 182–212, plot day 208.
   - August 213–243, plot day 220.
   - November 305–334, plot day 312.
   - Each period produces hourly, daily, monthly, household, economics,
     external-capacity and reconstruction-audit sheets plus plots.

8. **Capacity boundary**
   - The 7 kW value exists only in the config's `external_reference` section.
   - `env_benchmark.py` deliberately discards that section.
   - Capacity is used only after trajectories are generated to describe
     violation, excess, useful DR and targeting outcomes, and to draw an
     explicitly labelled external reference line.

9. **Three-seed and cross-method reporting**
   - Added mean ± sample SD and pairwise action-signal stability.
   - Added a comparison utility restricted to physically commensurable
     outcomes.
   - Raw rewards, Q-values, benchmark-specific discomfort and raw household
     utility are excluded from cross-method comparison.
   - Rebound is `N/A` for the curtailment-only benchmark.

10. **Reproducibility and safety**
    - Added checkpoint provenance validation, config signatures, source hashes,
      fail-stop PowerShell runners, non-empty-output protection and an isolated
      `results/benchmark_smoke` path.

## Local checks completed before handoff

- `python -m benchmark.verify_benchmark`: PASS
- Two-episode smoke training for seed 0: completed under
  `results/benchmark_smoke` only
- One-day checkpoint load/test/audit: completed; reconstruction errors were
  zero to floating-point precision
- Synthetic plumbing check of three-seed summarisation and cross-method workbook:
  completed; synthetic outputs were removed and are not part of the package

These local checks establish executable plumbing only. The reported manuscript
results were generated with the frozen three-seed benchmark configuration included
in this repository.
