"""Validate the ETA sensitivity package structure and comparable run metadata."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import yaml

BASE = Path(__file__).resolve().parents[1]
EXPECTED = {
    "eta_0p00000": {"eta": 0.0, "enabled": False},
    "eta_0p01429": {"eta": 0.01429, "enabled": True},
    "eta_0p03571": {"eta": 0.03571, "enabled": True},
    "eta_0p07143": {"eta": 0.07143, "enabled": True},
}
REQUIRED_TRAINING_FILES = (
    "config.yaml",
    "ddqn_best.pth",
    "ddqn_final.pth",
    "training_metrics.npy",
    "validation_metrics.csv",
    "validation_summary.json",
    "episode_reward.png",
    "training_loss.png",
    "epsilon_decay.png",
    "evaluation_curve.png",
    "train_vs_validation.png",
    "validation_cumulative_excess.png",
    "validation_violation_steps.png",
)


def load_yaml(path: Path) -> dict:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise TypeError(f"Invalid YAML object: {path}")
    return data


def close(a: float, b: float, tol: float = 1e-10) -> bool:
    return abs(float(a) - float(b)) <= tol


def normalized_config(cfg: dict) -> dict:
    clone = json.loads(json.dumps(cfg))
    env = clone["environment"]
    env.pop("reward_mode", None)
    env.pop("no_need_offer_regularization", None)
    return clone


def main() -> None:
    failures: list[str] = []
    references: list[dict] = []

    for slug, expected in EXPECTED.items():
        config_path = BASE / "configs" / f"{slug}.yaml"
        training_dir = BASE / "runs" / slug / "training"
        diagnostic_path = BASE / "runs" / slug / "diagnostics" / "q_action_diagnostic.xlsx"

        if not config_path.is_file():
            failures.append(f"Missing canonical config: {config_path}")
            continue
        if not training_dir.is_dir():
            failures.append(f"Missing training directory: {training_dir}")
            continue
        if not diagnostic_path.is_file():
            failures.append(f"Missing Q diagnostic: {diagnostic_path}")

        cfg = load_yaml(config_path)
        snapshot = load_yaml(training_dir / "config.yaml")
        if cfg != snapshot:
            failures.append(f"Canonical config differs from run snapshot: {slug}")

        reg = cfg["environment"]["no_need_offer_regularization"]
        if not close(reg["coefficient"], expected["eta"]):
            failures.append(f"ETA mismatch for {slug}: {reg['coefficient']}")
        if bool(reg["enabled"]) != bool(expected["enabled"]):
            failures.append(f"ETA enabled flag mismatch for {slug}: {reg['enabled']}")
        if int(cfg["general"]["seed"]) != 0:
            failures.append(f"Unexpected seed for {slug}: {cfg['general']['seed']}")

        for name in REQUIRED_TRAINING_FILES:
            if not (training_dir / name).is_file():
                failures.append(f"Missing {slug}/training/{name}")

        checkpoints = [training_dir / f"ddqn_episode_{episode}.pth" for episode in (500, 1000, 1500, 2000, 2500)]
        for checkpoint in checkpoints:
            if not checkpoint.is_file():
                failures.append(f"Missing checkpoint: {checkpoint}")

        summary = json.loads((training_dir / "validation_summary.json").read_text(encoding="utf-8"))
        if int(summary["run_seed"]) != 0:
            failures.append(f"Validation summary seed mismatch for {slug}")
        if int(summary["best_metrics"]["validation_days"]) != 14:
            failures.append(f"Validation-day mismatch for {slug}")

        references.append(normalized_config(cfg))

    if references and any(item != references[0] for item in references[1:]):
        failures.append("The canonical configs differ in fields other than ETA activation/coefficient and reward mode.")

    manifest = BASE / "checksums.sha256"
    if not manifest.is_file():
        failures.append("Missing checksums.sha256")
    else:
        for line in manifest.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            expected_hash, relative = line.split("  ", 1)
            path = BASE / relative
            if not path.is_file():
                failures.append(f"Checksum target missing: {relative}")
                continue
            actual = hashlib.sha256(path.read_bytes()).hexdigest()
            if actual != expected_hash:
                failures.append(f"Checksum mismatch: {relative}")

    if failures:
        print("ETA SENSITIVITY VERIFY: FAIL")
        for failure in failures:
            print(f"- {failure}")
        raise SystemExit(1)

    print("ETA SENSITIVITY VERIFY: PASS")
    print("Validated ETA values: 0, 0.01429, 0.03571, 0.07143")
    print("All four runs use seed 0 and the same non-ETA configuration.")


if __name__ == "__main__":
    main()
