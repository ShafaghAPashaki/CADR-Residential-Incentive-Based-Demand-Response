from __future__ import annotations

import hashlib
import importlib
import json
import py_compile
import subprocess
import sys
from pathlib import Path

import numpy as np
import yaml

EXPECTED_MATRIX_HASH = "468b5b24603e49fa84ea6e71180d0e18283f556fb43d72be5733571322b5d0e0"
CONFIGS = [Path(f"config_seed_{seed}.yaml") for seed in (0, 1, 2)]
PYTHON_FILES = [
    Path("main.py"), Path("test.py"), Path("q_action_diagnostic.py"),
    Path("reward_counterfactual_audit.py"), Path("summarize_results.py"),
    Path("agent/agent_ddqn.py"), Path("env/env.py"), Path("utils/replay_buffer.py"),
    Path("utils/config_loader.py"), Path("utils/load_demand.py"), Path("utils/load_price.py"),
    Path("benchmark/env_benchmark.py"), Path("benchmark/test_benchmark.py"),
]

IMPORT_MODULES = [
    "main", "test", "q_action_diagnostic", "reward_counterfactual_audit",
    "summarize_results", "agent.agent_ddqn", "env.env",
    "utils.replay_buffer", "utils.config_loader", "utils.load_demand",
    "utils.load_price", "benchmark.env_benchmark", "benchmark.test_benchmark",
]


def canonical(config):
    clone = json.loads(json.dumps(config))
    clone["general"]["seed"] = None
    return json.dumps(clone, sort_keys=True, separators=(",", ":"))


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    failures = []
    for path in PYTHON_FILES + CONFIGS:
        if not path.is_file():
            failures.append(f"Missing: {path}")
    if failures:
        raise SystemExit("\n".join(failures))

    for path in PYTHON_FILES:
        py_compile.compile(str(path), doraise=True)

    for module_name in IMPORT_MODULES:
        try:
            importlib.import_module(module_name)
        except Exception as exc:
            failures.append(f"Import failed for {module_name}: {exc}")

    for script in ("q_action_diagnostic.py", "summarize_results.py"):
        completed = subprocess.run(
            [sys.executable, script, "--help"],
            capture_output=True,
            text=True,
        )
        if completed.returncode != 0:
            detail = (completed.stderr or completed.stdout).strip()
            failures.append(f"CLI smoke check failed for {script}: {detail}")

    qdiag_source = Path("q_action_diagnostic.py").read_text(encoding="utf-8")
    for collector in ("all_audit", "all_diffs", "all_marginals", "all_summaries"):
        if f"{collector}: list[pd.DataFrame] = []" not in qdiag_source:
            failures.append(f"Q diagnostic collector is not initialized: {collector}")

    configs = []
    for path in CONFIGS:
        with path.open("r", encoding="utf-8") as handle:
            configs.append(yaml.safe_load(handle))
    if [cfg["general"]["seed"] for cfg in configs] != [0, 1, 2]:
        failures.append("Seeds must be 0, 1 and 2.")
    if len({canonical(cfg) for cfg in configs}) != 1:
        failures.append("Configs differ in fields other than general.seed.")

    for seed, cfg in enumerate(configs):
        if "DQN" in cfg or "LEARNING_RATE_DQN" in json.dumps(cfg):
            failures.append(f"Seed {seed}: old DQN naming remains.")
        if "DDQN" not in cfg or "LEARNING_RATE_DDQN" not in cfg["DDQN"]:
            failures.append(f"Seed {seed}: DDQN configuration is incomplete.")
        matrix = np.asarray(cfg["environment"]["fixed_household_coefficients"], dtype=np.float64)
        matrix_hash = hashlib.sha256(np.ascontiguousarray(matrix).tobytes()).hexdigest()
        if matrix_hash != EXPECTED_MATRIX_HASH:
            failures.append(f"Seed {seed}: household matrix hash mismatch.")
        if cfg["environment"]["rho"] != 0.9:
            failures.append(f"Seed {seed}: rho must remain 0.9.")
        if cfg["training"]["train_ranges"] != [[2, 167]]:
            failures.append(f"Seed {seed}: training range mismatch.")
        if cfg["training"]["val_range"] != [168, 181]:
            failures.append(f"Seed {seed}: validation range mismatch.")
        if cfg["training"]["test_range"] != [213, 243]:
            failures.append(f"Seed {seed}: primary test range mismatch.")
        if cfg["testing"]["plot_days"] != [220]:
            failures.append(f"Seed {seed}: August plot day must be 220.")

    root = Path('.')
    forbidden = [root / '.venv', root / 'phd_project.egg-info', root / 'setup.py']
    for path in forbidden:
        if path.exists():
            failures.append(f"Remove generated packaging item: {path}")

    if failures:
        print("VERIFY: FAIL")
        for failure in failures:
            print(f"- {failure}")
        raise SystemExit(1)

    files = {}
    for path in sorted(p for p in root.rglob('*') if p.is_file()):
        relative = path.as_posix()
        if '__pycache__' in relative or relative.endswith('.pyc'):
            continue
        if 'benchmark/results' in relative or relative == 'source_hashes.json':
            continue
        files[relative] = sha256(path)
    Path('source_hashes.json').write_text(json.dumps(files, indent=2), encoding='utf-8')
    print("VERIFY: PASS")
    print("Three configs are identical except for seed.")
    print("DDQN naming, fixed household matrix, data splits, rho and plot day are frozen.")


if __name__ == '__main__':
    main()
