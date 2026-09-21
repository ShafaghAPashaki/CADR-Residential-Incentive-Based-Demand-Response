# Benchmark implementation notes

The EBDR benchmark is an independent, elasticity-based, curtailment-only DDQN implementation.

Key implementation points:

- Seeds: 0, 1, and 2.
- Training: days 2–167; validation: days 168–181.
- Test periods: July, August, and November.
- Capacity information is excluded from training and policy selection.
- The active state has 12 features and no capacity-derived variable.
- Negative wholesale prices remain in the economic reward; incentive bounds use the smallest strictly positive daily price.
- Checkpoints are selected using validation performance only.
- Cross-method comparison uses commensurable physical and economic metrics.

Run `python -m benchmark.verify_benchmark` to verify the benchmark configuration and implementation boundary.
