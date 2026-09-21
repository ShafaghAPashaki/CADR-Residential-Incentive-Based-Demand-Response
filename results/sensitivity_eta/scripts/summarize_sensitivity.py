"""Create a compact CSV summary from the four validation summaries."""
from __future__ import annotations

import csv
import json
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]
ARMS = (
    ("eta_0p00000", 0.0),
    ("eta_0p01429", 0.01429),
    ("eta_0p03571", 0.03571),
    ("eta_0p07143", 0.07143),
)
FIELDS = (
    "episode",
    "mean_reward",
    "std_reward",
    "violation_steps",
    "violation_percentage",
    "cumulative_excess",
    "maximum_overrun",
    "target_tracking_mae_kwh",
    "incentive_steps",
    "unnecessary_incentive_steps",
    "average_offered_incentive_rate",
    "incentive_payment",
    "raw_discomfort",
)


def main() -> None:
    rows = []
    for slug, eta in ARMS:
        path = BASE / "runs" / slug / "training" / "validation_summary.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        metrics = payload["best_metrics"]
        row = {"run": slug, "eta": eta, "seed": payload["run_seed"]}
        row.update({field: metrics[field] for field in FIELDS})
        row["eta_penalty_total"] = metrics["no_need_offer_penalty"]
        rows.append(row)

    output = BASE / "summary.csv"
    columns = ["run", "eta", "seed", *FIELDS, "eta_penalty_total"]
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
