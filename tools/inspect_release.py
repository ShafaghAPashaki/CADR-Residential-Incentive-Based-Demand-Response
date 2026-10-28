"""Data-free inspector for the public CADR reproducibility package.

Usage from repository root:
    python tools/inspect_release.py

Uses only the Python standard library; does not execute models, touch
licensed data, or modify the repository. Checks presence of essential
published records and summarizes the archived Seed-0 monthly rows.
"""
from __future__ import annotations
import csv
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REQUIRED = (
    "README.md", "REPRODUCIBILITY.md", "TABLE_MAP.md", "LICENSE",
    "config_seed_0.yaml", "config_seed_1.yaml", "config_seed_2.yaml",
    "results/aggregate_metrics/per_seed_monthly_metrics.csv",
    "results/seed_0/training/ddqn_best.pth",
    "results/seed_0/training/config.yaml",
    "results/forecast/ddqn_state_forecasts_provenance.json",
    "results/calibration/lambda_c_retention_audit.json",
)

def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for part in iter(lambda: f.read(1 << 20), b""):
            h.update(part)
    return h.hexdigest()

def main() -> int:
    errors = []
    for name in REQUIRED:
        if not (ROOT / name).is_file():
            errors.append(f"Required public file missing: {name}")
    if errors:
        for x in errors:
            print("ERROR:", x, file=sys.stderr)
        return 1
    with (ROOT / "results/aggregate_metrics/per_seed_monthly_metrics.csv").open(
        newline="", encoding="utf-8-sig"
    ) as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            print("ERROR: monthly metrics table has no header", file=sys.stderr)
            return 1
        rows = list(reader)
    if not rows:
        print("ERROR: monthly metrics table is empty", file=sys.stderr)
        return 1
    print("CADR DATA-FREE RELEASE INSPECTION: PASS")
    print("These are archived research outputs, NOT a fresh model evaluation.")
    print(f"Archived per-seed/month rows: {len(rows)}")
    print("Archive columns:", ", ".join(reader.fieldnames))
    for name in (
        "config_seed_0.yaml",
        "results/seed_0/training/ddqn_best.pth",
        "results/aggregate_metrics/per_seed_monthly_metrics.csv",
    ):
        print(f"sha256 {name}: {sha256(ROOT / name)}")
    print("See TABLE_MAP.md for full manuscript table/figure mappings.")
    print("For exact or approximate checkpoint evaluation, supply licensed inputs.")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
