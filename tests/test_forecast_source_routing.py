"""Regression test for DDQN forecast-source routing and file hygiene."""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from env.env import Environment  # noqa: E402


def main() -> None:
    cfg = yaml.safe_load((ROOT / "config_seed_0.yaml").read_text(encoding="utf-8"))
    env = Environment(cfg["environment"]["house_ids"], cfg_override=cfg)
    table = env.decision_state_forecasts.reset_index(drop=True)

    if any("actual" in column.lower() for column in table.columns):
        raise AssertionError("Actual/audit columns are present in state input table.")

    train = table[table["doy"].between(2, 167)]
    val = table[table["doy"].between(168, 181)]
    test = table[table["doy"].between(182, 243)]
    if len(train) != 166 * 24 or set(train["forecast_source"]) != {"crossfit"}:
        raise AssertionError("Training forecast routing is not exclusively cross-fitted.")
    if len(val) != 14 * 24 or set(val["forecast_source"]) != {"full_model"}:
        raise AssertionError("Validation forecast routing is not full-model.")
    if len(test) != 62 * 24 or set(test["forecast_source"]) != {"full_model"}:
        raise AssertionError("Held-out test forecast routing is not full-model.")

    print({
        "train_rows": len(train),
        "validation_rows": len(val),
        "july_august_rows": len(test),
        "input_sha256": env.forecast_state_input_sha256,
        "status": "PASS",
    })


if __name__ == "__main__":
    main()
