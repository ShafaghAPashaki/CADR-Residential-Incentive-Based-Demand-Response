# Independent Reproduction Acceptance Status — 28 October 2026

## Scientific immutability
The repository's existing environment, published checkpoints, published
aggregated results, three frozen seed configurations, retained historical
lambda_c and the original forecast SHA gate remain unchanged in this branch.
Independent retraining uses an explicitly labelled opt-in to accept regenerated
forecast-state bytes and writes to a separate `repro_runs/` directory.

## Evidence available

| Check | Result | Boundary |
|---|---|---|
| Original Dataport-to-hourly numerical reconciliation | PASS | 218,835 household-hour rows / 25 households, all consumption columns within 1e-9 of the retained hourly file; source licensed and not redistributed |
| ERCOT raw EnergyOnline LZ_AEN selection | PASS | 8,757 available hourly observations numerically equal to the corresponding historical series |
| Known three missing 2018 price hours | PASS | Historical forward-fill method produces 8,760 matching 2018 values |
| Source-free load/price regression tests | PASS | 11 tests in a local Python 3.13 environment; this is **not** a scientific DDQN training test |
| Full 2019 boundary reproduction | PENDING | Original 2019-01-01 00:00 value must be independently sourced, not fabricated |
| 2018 input byte-identical after raw preprocessing | NOT GUARANTEED | Numeric equivalence does not establish CSV serialization or input SHA identity |
| Source-code compilation and full scientific unit test suite | NOT VERIFIED on required environment | Different package versions in test container |
| Full LSTM/cross-fit from a clean fresh checkout | NOT VERIFIED | Requires specified Python 3.12 dependencies and licensed full inputs |
| DDQN Seed 0 / 2,500 episodes + July evaluation | NOT VERIFIED | No valid result reported or fabricated |
| PowerShell orchestration on Windows | NOT VERIFIED | No PowerShell executable in local validation container |
| Anonymous reviewer-release URL | NOT CREATED | Author-identifiable GitHub URL is not acceptable for blind review |

## Runtime blocker encountered

The available local runner has Python 3.13, NumPy 2.3.5,
Pandas 2.2.3, torch 2.10.0+cpu, and **does not have the required
`holidays==0.100`** installed. No external network resolution is available
from that runner. The frozen scientific code deliberately rejects substituting
another holidays version, so the full training run must not be reported as
verified in this environment. There is no valid shortcut using a stub.

## Acceptance command

After installing the exact requirements on a Windows machine with Python 3.12,
place the correctly licensed 2018 exports in `downloads/`, then run:

```powershell
powershell -ExecutionPolicy Bypass -File .\run_reproduction_seed0_july.ps1
```

The one-command runner performs input preparation, synthetic source-free tests,
full-model load and price LSTM training, cross-fitting, independent cross-fit
verification, state-input assembly, Seed-0 DDQN training, and July evaluation.
A report should record tool versions, raw/processed SHA-256 hashes, selected
training checkpoint and its validation-selected episode, all 31 July days
of results, and deviations from **Seed 0** rather than the three-seed mean.

**Do not claim end-to-end success** unless the command completes on a clean
checkout with the authorised 2018 data and outputs are inspected. Follow
`START_HERE_SINGLE_SEED.md` and `data/README.md` for manual recovery steps.
