"""Pre-training verifier for the benchmark package."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import py_compile
import subprocess
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
BENCHMARK_DIR = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import yaml

from benchmark.env_benchmark import Environment
from agent.agent_ddqn import DDQNAgent
from utils.config_loader import load_config


CONFIGS = [BENCHMARK_DIR / f"config_benchmark_seed_{seed}.yaml" for seed in range(3)]
PROHIBITED_ENV_KEYS = Environment._PROPOSED_ONLY_ENV_KEYS
HASH_FILE = BENCHMARK_DIR / "benchmark_source_hashes.json"


def fail(message: str) -> None:
    raise AssertionError(message)


def canonical_without_seed(cfg: dict[str, Any]) -> dict[str, Any]:
    value = copy.deepcopy(cfg)
    value["general"]["seed"] = "<SEED>"
    return value


def check_configs() -> list[dict[str, Any]]:
    configs = [load_config(str(path)) for path in CONFIGS]
    for seed, cfg in enumerate(configs):
        if int(cfg["general"]["seed"]) != seed:
            fail(f"{CONFIGS[seed].name} does not contain seed {seed}.")
        leaked = sorted(set(cfg["environment"]).intersection(PROHIBITED_ENV_KEYS))
        if leaked:
            fail(f"Proposed/capacity fields leaked into benchmark environment: {leaked}")
        if "external_reference" not in cfg:
            fail("external_reference section is missing.")
        if float(cfg["external_reference"]["capacity_threshold_kw"]) != 7.0:
            fail("External capacity reference must remain 7.0 kW.")
        if cfg["external_reference"].get("usage") != "post_hoc_testing_and_plotting_only":
            fail("External capacity usage label is incorrect.")
        if cfg["DDQN"].get("hidden_layers") != [128, 64]:
            fail("Benchmark DDQN hidden layers must be [128, 64].")
        if cfg["training"]["train_ranges"] != [[2, 167]]:
            fail("Training range must be [[2, 167]].")
        if cfg["training"]["val_range"] != [168, 181]:
            fail("Validation range must be [168, 181].")
        expected_periods = {
            "july": {"range": [182, 212], "plot_days": [208]},
            "august": {"range": [213, 243], "plot_days": [220]},
            "november": {"range": [305, 334], "plot_days": [312]},
        }
        if cfg["testing"]["periods"] != expected_periods:
            fail("Testing periods/plot days differ from the frozen design.")

    base = canonical_without_seed(configs[0])
    for seed, cfg in enumerate(configs[1:], start=1):
        if canonical_without_seed(cfg) != base:
            fail(f"Seed config {seed} differs from seed 0 in more than general.seed.")
    return configs


def check_environment(cfg: dict[str, Any]) -> None:
    house_ids = cfg["environment"]["house_ids"]
    env = Environment(house_ids, cfg_override=cfg)
    state = env.reset(day=168, mode="val")
    if state.shape != (12,) or env.state_dim != 12:
        fail(f"Expected 12-state benchmark; got {state.shape}, state_dim={env.state_dim}.")
    if env.num_actions != 125 or env.all_actions.shape != (125, 3):
        fail("Expected 125 joint actions for three households.")
    if "external_reference" in env.cfg or "testing" in env.cfg:
        fail("Environment retained external capacity/test metadata internally.")
    agent = DDQNAgent(
        env.state_dim,
        env.num_actions,
        cfg_override=cfg,
        action_costs=np.mean(env.all_actions, axis=1),
    )
    if (
        agent.policy_net.fc1.in_features != 12
        or agent.policy_net.fc1.out_features != 128
        or agent.policy_net.fc2.in_features != 128
        or agent.policy_net.fc2.out_features != 64
        or agent.policy_net.fc3.in_features != 64
        or agent.policy_net.fc3.out_features != 125
    ):
        fail("Active DDQN architecture is not 12 -> 128 -> 64 -> 125.")
    if not np.allclose(env.all_actions[0], 0.0):
        fail("Action index 0 must be the all-zero raw action.")
    if not np.allclose(env.all_actions[-1], 1.0):
        fail("Final action must be the all-one raw action.")

    env.reset(day=168, mode="val")
    _, reward, _, _ = env.step(0)
    if not np.allclose(env.reductions[0], 0.0, atol=1e-12):
        fail("Zero action produced nonzero response.")
    if abs(float(env.incentive_payment[0])) > 1e-12:
        fail("Zero action produced nonzero payment.")
    if abs(float(reward)) > 1e-12:
        fail("Zero action should produce zero immediate reward.")
    if not np.allclose(env.incentives[0], env.lambda_min[0]):
        fail("Archived zero-action nominal-rate convention was not preserved.")

    # Confirm direct response is the exact simplified archived equation.
    env.reset(day=168, mode="val")
    action_index = 124
    raw = env.all_actions[action_index]
    baseline = env.baseline_per_house[0]
    elasticity = env.elasticities[0]
    expected = np.clip(
        baseline * elasticity * raw,
        0.0,
        env.max_reduction_fraction * baseline,
    )
    env.step(action_index)
    if not np.allclose(env.reductions[0], expected, atol=1e-12):
        fail("Direct elasticity response does not match the archived algebraic model.")

    # Confirm max_reduction_fraction is wired, not hard-coded to 0.3.
    cfg_fraction = copy.deepcopy(cfg)
    cfg_fraction["environment"]["max_reduction_fraction"] = 0.123
    fraction_env = Environment(house_ids, cfg_override=cfg_fraction)
    fraction_env.reset(day=168, mode="val")
    fraction_env.step(124)
    upper = 0.123 * fraction_env.baseline_per_house[0]
    if np.any(fraction_env.reductions[0] > upper + 1e-12):
        fail("Configured max_reduction_fraction is not active.")
    if not np.any(np.isclose(fraction_env.reductions[0], upper, atol=1e-12)):
        fail("max_reduction_fraction wiring test did not bind as expected.")

    # Day 203 contains negative prices. A mixed-price day must remain operable.
    env.reset(day=203, mode="test")
    if not np.any(env.prices < 0.0) or not np.any(env.prices > 0.0):
        fail("Day 203 no longer provides the expected mixed-price audit case.")
    if env.daily_reference_price <= 0.0:
        fail("Mixed-price day was incorrectly collapsed to a zero daily reference.")
    if env.lambda_max[0] <= env.lambda_min[0]:
        fail("Negative-price sign inversion remains in incentive bounds.")
    env.step(124)
    if float(env.reductions[0].sum()) <= 0.0:
        fail("Mixed-price day cannot produce response after the safeguard.")

    # Changing the external threshold must not alter environment dynamics/signature.
    cfg_external = copy.deepcopy(cfg)
    cfg_external["external_reference"]["capacity_threshold_kw"] = 999.0
    external_env = Environment(house_ids, cfg_override=cfg_external)
    external_env.reset(day=168, mode="val")
    env_base = Environment(house_ids, cfg_override=cfg)
    env_base.reset(day=168, mode="val")
    if env_base.get_config_signature() != external_env.get_config_signature():
        fail("External capacity reference leaked into training config signature.")
    env_base.step(97)
    external_env.step(97)
    arrays = [
        (env_base.reductions[0], external_env.reductions[0]),
        (env_base.incentives[0], external_env.incentives[0]),
        (env_base.rewards_total[:1], external_env.rewards_total[:1]),
    ]
    if any(not np.allclose(left, right, atol=1e-12) for left, right in arrays):
        fail("External capacity reference altered benchmark policy dynamics.")


def check_source_hashes() -> None:
    if not HASH_FILE.exists():
        fail(f"Missing source hash manifest: {HASH_FILE}")
    manifest = json.loads(HASH_FILE.read_text(encoding="utf-8"))
    active_files = {
        path.relative_to(BENCHMARK_DIR).as_posix()
        for path in BENCHMARK_DIR.iterdir()
        if path.is_file()
        and path.name != HASH_FILE.name
        and not path.name.endswith((".pyc", ".pyo"))
    }
    manifest_files = set(manifest)
    if manifest_files != active_files:
        missing = sorted(active_files - manifest_files)
        extra = sorted(manifest_files - active_files)
        fail(
            "Benchmark hash manifest does not exactly cover active top-level files: "
            f"missing={missing}, extra={extra}"
        )
    for relative, expected in manifest.items():
        path = BENCHMARK_DIR / relative
        if not path.exists():
            fail(f"Hashed benchmark source is missing: {path}")
        observed = hashlib.sha256(path.read_bytes()).hexdigest()
        if observed != expected:
            fail(f"Source hash mismatch for {relative}: {observed} != {expected}")


def check_compile_and_cli(python_executable: str) -> None:
    py_files = sorted(BENCHMARK_DIR.glob("*.py"))
    for path in py_files:
        py_compile.compile(str(path), doraise=True)
    modules = [
        "benchmark.main_benchmark",
        "benchmark.test_benchmark",
        "benchmark.summarize_benchmark",
        "benchmark.compare_methods",
    ]
    for module in modules:
        completed = subprocess.run(
            [python_executable, "-m", module, "--help"],
            cwd=PROJECT_ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )
        if completed.returncode != 0:
            fail(
                f"CLI help failed for {module}:\nSTDOUT:\n{completed.stdout}\n"
                f"STDERR:\n{completed.stderr}"
            )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--skip-hashes",
        action="store_true",
        help="Developer-only: skip source hash validation while rebuilding the package.",
    )
    args = parser.parse_args()

    configs = check_configs()
    check_environment(configs[0])
    check_compile_and_cli(sys.executable)
    if not args.skip_hashes:
        check_source_hashes()

    print("BENCHMARK VERIFY: PASS")
    print("Three configs are identical except for seed.")
    print("Benchmark remains capacity-blind during training and action selection.")
    print("July, August and November test periods are frozen.")
    print("Negative-price, zero-action and response-ledger audits passed.")


if __name__ == "__main__":
    main()
