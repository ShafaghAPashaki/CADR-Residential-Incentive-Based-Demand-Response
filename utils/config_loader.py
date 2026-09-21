from pathlib import Path

import yaml


def load_config(path=None):
    config_path = Path(path) if path else Path(__file__).resolve().parents[1] / "config_seed_0.yaml"
    with config_path.open("r", encoding="utf-8-sig") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise TypeError(f"Configuration is not a dictionary: {config_path}")
    return config
