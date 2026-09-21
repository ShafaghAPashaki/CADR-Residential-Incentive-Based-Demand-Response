#!/usr/bin/env python3
"""
Sequential rho-sensitivity runner for Paper 1-1.

Purpose
-------
Train rho = {0.1, 0.3, 0.5, 0.7, 0.9} sequentially using the final Seed-0
configuration while preserving the existing official Seed-0 training results.

Only environment.rho is changed across sensitivity runs.
The base YAML and the official results/seed_0/training directory are restored
unchanged after the sweep.

Run from the project root:
    python run_rho_sensitivity.py --base-config config_seed_0.yaml
"""

import argparse
import copy
import json
import shutil
import subprocess
import sys
from pathlib import Path

import yaml


RHO_VALUES = [0.1, 0.3, 0.5, 0.7, 0.9]


def rho_tag(rho: float) -> str:
    return f"rho_{rho:.1f}".replace(".", "p")


def validate_base_config(cfg: dict) -> None:
    general = cfg.get("general", {})
    env = cfg.get("environment", {})

    if int(general.get("seed", -1)) != 0:
        raise ValueError("Base config must use seed 0 for the rho sensitivity study.")

    if env.get("reward_mode") != "weighted_stakeholder_tracking_no_need":
        raise ValueError(
            "Base config must use reward_mode=weighted_stakeholder_tracking_no_need."
        )

    nn = env.get("no_need_offer_regularization", {}) or {}
    if not bool(nn.get("enabled", False)):
        raise ValueError("No-need regularisation must remain enabled.")
    if abs(float(nn.get("coefficient", -1.0)) - 0.07143) > 1e-12:
        raise ValueError("Expected final eta_NN = 0.07143.")

    if env.get("fixed_household_coefficients") is None:
        raise ValueError(
            "Base config must contain environment.fixed_household_coefficients."
        )

    ws = env.get("weighted_stakeholder", {}) or {}
    if abs(float(ws.get("sp_weight", -1.0)) - 0.65) > 1e-12:
        raise ValueError("Expected final SP stakeholder weight = 0.65.")


def safe_restore_official_training(
    official_training: Path,
    backup_training: Path,
    result_root: Path,
    current_tag: str | None,
) -> None:
    """
    Restore the official Seed-0 training directory.

    If a failed sensitivity run left a partial results/seed_0/training folder,
    move that partial folder aside first so it can be inspected later.
    """
    if not backup_training.exists():
        return

    if official_training.exists():
        failed_root = result_root / "_failed_partial"
        failed_root.mkdir(parents=True, exist_ok=True)
        tag = current_tag or "unknown"
        failed_target = failed_root / f"{tag}_training"

        suffix = 1
        while failed_target.exists():
            failed_target = failed_root / f"{tag}_training_{suffix}"
            suffix += 1

        shutil.move(str(official_training), str(failed_target))
        print(f"\nPartial sensitivity output preserved at: {failed_target}")

    shutil.move(str(backup_training), str(official_training))
    print(f"Official Seed-0 training results restored: {official_training}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base-config",
        default="config_seed_0.yaml",
        help="Final Seed-0 YAML used as the immutable sensitivity template.",
    )
    parser.add_argument(
        "--main",
        default="main.py",
        help="Training entry point.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing rho-sensitivity run folders.",
    )
    args = parser.parse_args()

    project_root = Path.cwd()
    base_config_path = (project_root / args.base_config).resolve()
    main_path = (project_root / args.main).resolve()

    if not base_config_path.exists():
        raise FileNotFoundError(base_config_path)
    if not main_path.exists():
        raise FileNotFoundError(main_path)

    with base_config_path.open("r", encoding="utf-8") as f:
        base_cfg = yaml.safe_load(f)

    if not isinstance(base_cfg, dict):
        raise TypeError("Base YAML must contain a dictionary.")

    validate_base_config(base_cfg)

    config_dir = project_root / "configs" / "rho_sensitivity"
    result_root = project_root / "results" / "rho_sensitivity"
    log_dir = result_root / "logs"

    official_training = project_root / "results" / "seed_0" / "training"
    backup_training = (
        project_root / "results" / "seed_0" / "training__rho_sensitivity_original"
    )

    config_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    # Recover automatically from an earlier interrupted sensitivity attempt.
    if backup_training.exists():
        if official_training.exists():
            raise RuntimeError(
                "Both the official training directory and the rho-sensitivity "
                "backup exist. Nothing was changed. Please inspect:\n"
                f"  {official_training}\n"
                f"  {backup_training}"
            )
        print("Recovering official Seed-0 training results from an earlier backup...")
        shutil.move(str(backup_training), str(official_training))

    if not official_training.exists():
        raise FileNotFoundError(
            "Official Seed-0 training directory was not found:\n"
            f"  {official_training}\n"
            "The sensitivity runner intentionally refuses to proceed because "
            "the final Seed-0 training results should be preserved."
        )

    manifest = {
        "base_config": str(base_config_path),
        "rho_values": RHO_VALUES,
        "seed": 0,
        "official_training_preserved": str(official_training),
        "runs": [],
    }

    # main.py always writes Seed 0 to results/seed_0/training and refuses to
    # overwrite an existing directory. Move the official result aside once,
    # run the full sweep, then restore it in the finally block.
    print(f"Safeguarding official Seed-0 training results:\n  {official_training}")
    shutil.move(str(official_training), str(backup_training))

    current_tag = None

    try:
        for rho in RHO_VALUES:
            current_tag = rho_tag(rho)
            run_root = result_root / current_tag
            target_training = run_root / "training"
            config_path = config_dir / f"config_{current_tag}.yaml"
            log_path = log_dir / f"{current_tag}.log"

            if run_root.exists():
                if not args.overwrite:
                    raise FileExistsError(
                        f"{run_root} already exists. "
                        "Use --overwrite only if you intentionally want to replace it."
                    )
                shutil.rmtree(run_root)

            if official_training.exists():
                raise RuntimeError(
                    "Unexpected results/seed_0/training directory exists before "
                    f"starting {current_tag}. Aborting to avoid mixing outputs."
                )

            cfg = copy.deepcopy(base_cfg)
            cfg["environment"]["rho"] = float(rho)

            # Controlled sensitivity: everything except rho remains unchanged.
            sp_weight = float(cfg["environment"]["weighted_stakeholder"]["sp_weight"])
            eu_weight = 1.0 - sp_weight
            effective_fraction = (sp_weight - eu_weight * rho) / sp_weight
            minimum_fraction = float(
                cfg["environment"]["weighted_stakeholder"][
                    "minimum_effective_payment_cost_fraction"
                ]
            )
            if effective_fraction + 1e-12 < minimum_fraction:
                raise ValueError(
                    f"{current_tag}: effective payment-cost fraction "
                    f"{effective_fraction:.6f} is below configured minimum "
                    f"{minimum_fraction:.6f}."
                )

            with config_path.open("w", encoding="utf-8") as f:
                yaml.safe_dump(cfg, f, sort_keys=False)

            print("\n" + "=" * 78)
            print(f"STARTING {current_tag}")
            print(f"rho = {rho}")
            print(f"Config: {config_path}")
            print("=" * 78)

            cmd = [
                sys.executable,
                str(main_path),
                "--config_path",
                str(config_path),
            ]

            with log_path.open("w", encoding="utf-8") as log_file:
                process = subprocess.Popen(
                    cmd,
                    cwd=project_root,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,
                )

                assert process.stdout is not None
                for line in process.stdout:
                    print(line, end="")
                    log_file.write(line)
                    log_file.flush()

                return_code = process.wait()

            if return_code != 0:
                raise RuntimeError(
                    f"{current_tag} failed with return code {return_code}. "
                    f"See {log_path}"
                )

            if not official_training.exists():
                raise RuntimeError(
                    f"{current_tag} completed but main.py did not create:\n"
                    f"  {official_training}"
                )

            if not any(official_training.iterdir()):
                raise RuntimeError(
                    f"{current_tag} produced an empty training directory."
                )

            run_root.mkdir(parents=True, exist_ok=False)
            shutil.move(str(official_training), str(target_training))

            manifest["runs"].append(
                {
                    "rho": rho,
                    "tag": current_tag,
                    "config": str(config_path),
                    "training_results": str(target_training),
                    "log": str(log_path),
                    "effective_payment_cost_fraction": effective_fraction,
                }
            )

            with (result_root / "manifest.json").open(
                "w", encoding="utf-8"
            ) as f:
                json.dump(manifest, f, indent=2)

            print(f"FINISHED {current_tag}")
            print(f"Saved to: {target_training}")

    finally:
        safe_restore_official_training(
            official_training=official_training,
            backup_training=backup_training,
            result_root=result_root,
            current_tag=current_tag,
        )

    print("\nAll rho-sensitivity runs completed successfully.")
    print(f"Results root: {result_root}")
    print(f"Manifest: {result_root / 'manifest.json'}")


if __name__ == "__main__":
    main()
