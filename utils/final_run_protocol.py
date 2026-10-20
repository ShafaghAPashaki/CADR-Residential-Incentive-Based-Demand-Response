"""Pre-run dependency, calibration and provenance gates for frozen DDQN runs.

No network, fallback calendar, synthetic data, or test-set tuning is permitted.
"""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CALIBRATION_FILE = PROJECT_ROOT / "results/calibration/lambda_c_retention_audit.json"
LAMBDA_SOURCE = "train_p95_scenario_action_matching_original_retained"
RETAINED_LAMBDA_C = 0.2216381800923157


def sha256_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def require_real_holidays() -> dict[str, Any]:
    """Fail closed if real, pinned python-holidays is not installed."""
    try:
        import holidays
        version = importlib.metadata.version("holidays")
    except (ImportError, importlib.metadata.PackageNotFoundError) as exc:
        raise RuntimeError(
            "FINAL RUN BLOCKED: install the genuine `holidays==0.100` package "
            "from requirements.txt; no temporary stub or calendar fallback "
            "is allowed for final training or calibration."
        ) from exc
    required = "0.100"
    if version != required:
        raise RuntimeError(
            f"FINAL RUN BLOCKED: holidays version {version!r}; required {required!r}."
        )
    if not hasattr(holidays, "US") or not hasattr(holidays, "__file__"):
        raise RuntimeError("FINAL RUN BLOCKED: invalid holidays module.")
    import datetime
    cal = holidays.US(years=[2018])
    if datetime.date(2018, 7, 4) not in cal:
        raise RuntimeError("FINAL RUN BLOCKED: holidays.US calendar self-test failed.")
    return {
        "holiday_backend": "python-holidays / holidays.US(years=[2018])",
        "holiday_version": version,
        "holiday_module_path": str(holidays.__file__),
    }


def config_sha256(config: dict[str, Any]) -> str:
    canonical = yaml.safe_dump(config, sort_keys=True).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def enforce_lambda_calibration(config: dict[str, Any]) -> dict[str, Any]:
    """Fail closed unless the original, independently reviewed λc is retained.

    The historic calibration inputs are documented but the original twelve
    scenario identifiers were not archived. A new quantile-sampling estimate
    would not recreate that selection and must never silently replace λc.
    """
    if not CALIBRATION_FILE.is_file():
        raise RuntimeError(
            "FINAL CADR TRAINING BLOCKED: missing frozen original-λc "
            "retention audit; do not auto-recalibrate."
        )
    with CALIBRATION_FILE.open(encoding="utf-8") as handle:
        record = json.load(handle)
    if record.get("status") != "ORIGINAL_RETAINED":
        raise RuntimeError("FINAL CADR TRAINING BLOCKED: retention audit status differs.")
    if record.get("lambda_source") != LAMBDA_SOURCE:
        raise RuntimeError("FINAL CADR TRAINING BLOCKED: retention source differs.")
    settings = config["environment"]["symmetric_tracking"]
    if settings.get("lambda_source") != LAMBDA_SOURCE:
        raise RuntimeError("FINAL CADR TRAINING BLOCKED: lambda_source mismatch.")
    if abs(float(settings["lambda_c"]) - RETAINED_LAMBDA_C) > 1e-12:
        raise RuntimeError("FINAL CADR TRAINING BLOCKED: λc changed from frozen value.")
    if abs(float(record.get("retained_lambda_c", -1)) - RETAINED_LAMBDA_C) > 1e-12:
        raise RuntimeError("FINAL CADR TRAINING BLOCKED: retention-audit λc mismatch.")
    if float(config["environment"]["capacity_threshold"]) != 7.0:
        raise RuntimeError("FINAL CADR TRAINING BLOCKED: 7-kW final-run gate; "
                           "6/8-kW studies require a separately predeclared run protocol.")
    if float(record.get("capacity_reference_kw", -1)) != 7.0:
        raise RuntimeError("FINAL CADR TRAINING BLOCKED: retention audit capacity mismatch.")
    if record.get("environment_version") != "forecast_state_v1":
        raise RuntimeError("FINAL CADR TRAINING BLOCKED: retention audit environment mismatch.")
    forecast_path = PROJECT_ROOT / config["forecast"]["ddqn_state_input_file"]
    if sha256_file(forecast_path) != record.get("training_forecast_file_sha256"):
        raise RuntimeError("FINAL CADR TRAINING BLOCKED: frozen forecast SHA changed.")
    if sha256_file(PROJECT_ROOT / "env/env.py") != record.get(
        "frozen_cadr_environment_sha256"
    ):
        raise RuntimeError("FINAL CADR TRAINING BLOCKED: frozen scientific environment changed.")
    return record


def collect_run_provenance(config: dict[str, Any], environment_version: str) -> dict[str, Any]:
    holidays_meta = require_real_holidays()
    forecast_path = PROJECT_ROOT / config["forecast"]["ddqn_state_input_file"]
    files = [
        "env/env.py", "benchmark/env_benchmark.py", "main.py",
        "benchmark/main_benchmark.py", "agent/agent_ddqn.py",
        "utils/load_demand.py", "utils/load_price.py",
        "utils/final_run_protocol.py", "calibrate_lambda_c.py",
        "test.py", "benchmark/test_benchmark.py",
    ]
    code_hashes = {
        name: sha256_file(PROJECT_ROOT / name)
        for name in files if (PROJECT_ROOT / name).exists()
    }
    return {
        **holidays_meta,
        "environment_version": environment_version,
        "forecast_file_sha256": sha256_file(forecast_path),
        "configuration_sha256": config_sha256(config),
        "code_hashes_sha256": hashlib.sha256(
            json.dumps(code_hashes, sort_keys=True).encode("utf-8")
        ).hexdigest(),
        "code_hashes": code_hashes,
        "calibration_file_sha256": (
            sha256_file(CALIBRATION_FILE) if CALIBRATION_FILE.exists() else None
        ),
    }
