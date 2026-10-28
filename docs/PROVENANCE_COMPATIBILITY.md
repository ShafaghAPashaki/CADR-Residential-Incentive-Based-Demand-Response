# Historical provenance compatibility

Published `results/**/training/run_provenance.json` files are frozen records of the files used in **original training**; later code revisions must not be backdated or made to appear byte-identical.

- The original `results/calibration/lambda_c_retention_audit.json` is restored byte-for-byte from tag-era commit `644f2a7`. Its historical SHA-256 is `325d3344b881c57e37e41e55fd2fa74e019652babde56fbb8dd8316483d5028a`, matching the published CADR run provenance.
- The later private discovery of original calibration labels and observations is documented in `results/calibration/lambda_c_historical_recovery_metadata.json`. It is **not** embedded into the original audit file and does not change any calibration value.
- Published runs recorded `utils/final_run_protocol.py` SHA-256 `cf9764d881d28bb3b21c09189d4d1e8c735902d1c43c279caaa4bf8f0dde5cae`. The current public implementation includes later safety checks and metadata adjustments, so its source SHA-256 is **expected to differ**. For a current hash run `python -c "import hashlib,pathlib; p=pathlib.Path('utils/final_run_protocol.py'); print(hashlib.sha256(p.read_bytes()).hexdigest())"`; do not alter archived provenance to claim that this new file was used for old training.
- Checkpoint parameters, validation-selection records, published result workbooks and aggregate metrics have not been recomputed by these publication changes. Evaluation with authorised original forecast state is independently rechecked; new training may choose different checkpoints across hardware or software.
- Public repository commit history can still contain old source versions and plots; publishing a cleaned **current tree** is not a Git history rewrite. Do not treat the main GitHub history as anonymised or fully purged.

See [Historical calibration](LAMBDA_C_HISTORICAL_CALIBRATION.md), [Reproducibility](../REPRODUCIBILITY.md), and [Data acquisition and rights](../data/README.md).
