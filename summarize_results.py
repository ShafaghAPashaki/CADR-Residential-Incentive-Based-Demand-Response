from __future__ import annotations

import argparse
import itertools
from pathlib import Path

import numpy as np
import pandas as pd

SEEDS = (0, 1, 2)
PERIODS = ("july", "august", "november")


def mean_sd(values):
    array = np.asarray(values, dtype=float)
    return float(array.mean()), float(array.std(ddof=1))


def load_audit(root, seed, period):
    path = root / f"seed_{seed}" / period / "audit.xlsx"
    if not path.is_file():
        raise FileNotFoundError(path)
    daily = pd.read_excel(path, sheet_name="Daily_Summary")
    hourly = pd.read_excel(path, sheet_name="Hourly_Audit")
    return daily, hourly


def seed_metrics(daily, hourly):
    baseline_peak = float(daily["peak_baseline"].mean())
    after_peak = float(daily["peak_after"].mean())
    baseline_par = float(daily["PAR_baseline"].mean())
    after_par = float(daily["PAR_after"].mean())
    n_hours = len(hourly)
    no_need = int(hourly["no_need_gate"].sum())
    incentive = (hourly[["incentive_house_661", "incentive_house_3039", "incentive_house_8565"]].sum(axis=1) > 1e-9)
    return {
        "peak_reduction_percent": 100.0 * (baseline_peak - after_peak) / max(baseline_peak, 1e-12),
        "PAR_reduction_percent": 100.0 * (baseline_par - after_par) / max(baseline_par, 1e-12),
        "remaining_violation_rate_percent": 100.0 * float(hourly["remaining_capacity_violation"].sum()) / n_hours,
        "cumulative_excess_kWh": float(daily["cumulative_excess"].sum()),
        "maximum_overrun_kW": float(daily["maximum_overrun"].max()),
        "tracking_MAE_kWh": float(hourly["control_over_error"].add(hourly["control_under_error"]).abs().mean()),
        "useful_DR_kWh": float(hourly["useful_settlement_DR_energy"].sum()),
        "excess_DR_kWh": float(hourly["excess_settlement_DR_energy"].sum()),
        "unnecessary_incentive_rate_percent": 100.0 * float(hourly["unnecessary_incentive"].sum()) / max(no_need, 1),
        "no_response_rate_percent": 100.0 * float(hourly["positive_incentive_no_response"].sum()) / max(int(incentive.sum()), 1),
        "maximum_action_hours": int(hourly["maximum_action"].sum()),
        "rebound_violation_hours": int(hourly["rebound_induced_violation"].sum()),
        "incentive_payment_cent": float(hourly["incentive_payment"].sum()),
        "capacity_relief_settlement_cent": float(hourly["SP_settlement_value"].sum()),
        "net_wholesale_procurement_impact_cent": float(hourly["net_wholesale_procurement_impact"].sum()),
        "weighted_customer_utility_cent": float(hourly["weighted_customer_utility"].sum()),
        "raw_discomfort": float(daily["raw_discomfort_sum"].sum()),
    }


def spearman(a, b):
    return float(pd.Series(a).corr(pd.Series(b), method="spearman"))


def signal_rows(period, hourly_by_seed):
    rows = []
    incentive_cols = ["incentive_house_661", "incentive_house_3039", "incentive_house_8565"]
    raw_cols = ["raw_action_house_661", "raw_action_house_3039", "raw_action_house_8565"]
    for left, right in itertools.combinations(SEEDS, 2):
        a = hourly_by_seed[left].sort_values(["day", "hour"]).reset_index(drop=True)
        b = hourly_by_seed[right].sort_values(["day", "hour"]).reset_index(drop=True)
        if not a[["day", "hour"]].equals(b[["day", "hour"]]):
            raise ValueError(f"Hourly keys differ for seeds {left} and {right}, period {period}")
        a_inc = a[incentive_cols].to_numpy(float)
        b_inc = b[incentive_cols].to_numpy(float)
        a_raw = a[raw_cols].to_numpy(float)
        b_raw = b[raw_cols].to_numpy(float)
        a_active = a_inc.sum(axis=1) > 1e-9
        b_active = b_inc.sum(axis=1) > 1e-9
        exact_joint = np.all(np.isclose(a_raw, b_raw, atol=1e-12), axis=1)
        rows.append({
            "period": period,
            "seed_pair": f"{left}-{right}",
            "zero_nonzero_agreement_percent": 100.0 * np.mean(a_active == b_active),
            "exact_joint_action_agreement_percent": 100.0 * np.mean(exact_joint),
            "incentive_spearman": spearman(a_inc.ravel(), b_inc.ravel()),
            "incentive_MAE_cent_per_kWh": float(np.mean(np.abs(a_inc - b_inc))),
            "raw_action_MAE": float(np.mean(np.abs(a_raw - b_raw))),
            "no_need_zero_agreement_percent": 100.0 * np.mean((~a_active[a["no_need_gate"].astype(bool)]) & (~b_active[a["no_need_gate"].astype(bool)])) if a["no_need_gate"].astype(bool).any() else np.nan,
            "overload_active_agreement_percent": 100.0 * np.mean(a_active[~a["no_need_gate"].astype(bool)] & b_active[~a["no_need_gate"].astype(bool)]) if (~a["no_need_gate"].astype(bool)).any() else np.nan,
        })
    return rows


def q_summary(root):
    rows = []
    for seed in SEEDS:
        path = root / f"seed_{seed}" / "q_diagnostic.xlsx"
        if not path.is_file():
            continue
        states = pd.read_excel(path, sheet_name="State_Summary")
        failures = pd.read_excel(path, sheet_name="Policy_Failure_Candidates")
        row = {"seed": seed, "audited_states": len(states), "failure_candidates": len(failures)}
        for column in ["selected_action_index", "zero_action_is_selected", "no_need_gate", "capacity_violation"]:
            if column in states.columns:
                if states[column].dtype == bool or set(states[column].dropna().unique()).issubset({0, 1, True, False}):
                    row[f"{column}_count"] = int(states[column].fillna(False).astype(bool).sum())
        rows.append(row)
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description="Create the final three-seed summary workbook.")
    parser.add_argument(
        "--results_root",
        default="results",
        help="Directory containing seed_0, seed_1 and seed_2 outputs.",
    )
    args = parser.parse_args()
    root = Path(args.results_root)
    per_seed_rows = []
    stability_rows = []
    for period in PERIODS:
        hourly_by_seed = {}
        for seed in SEEDS:
            daily, hourly = load_audit(root, seed, period)
            hourly_by_seed[seed] = hourly
            per_seed_rows.append({"period": period, "seed": seed, **seed_metrics(daily, hourly)})
        stability_rows.extend(signal_rows(period, hourly_by_seed))

    per_seed = pd.DataFrame(per_seed_rows)
    aggregate_rows = []
    metric_columns = [c for c in per_seed.columns if c not in {"period", "seed"}]
    for period, group in per_seed.groupby("period", sort=False):
        for metric in metric_columns:
            mean, sd = mean_sd(group[metric])
            aggregate_rows.append({"period": period, "metric": metric, "mean": mean, "sample_SD": sd})
    aggregate = pd.DataFrame(aggregate_rows)
    stability = pd.DataFrame(stability_rows)

    failure_cols = [
        "period", "seed", "remaining_violation_rate_percent", "cumulative_excess_kWh",
        "maximum_overrun_kW", "unnecessary_incentive_rate_percent",
        "no_response_rate_percent", "maximum_action_hours", "rebound_violation_hours",
    ]
    failures = per_seed[failure_cols].copy()
    q = q_summary(root)

    out = root / "summary.xlsx"
    with pd.ExcelWriter(out, engine="openpyxl") as writer:
        per_seed.to_excel(writer, sheet_name="Seed_Performance", index=False)
        aggregate.to_excel(writer, sheet_name="Mean_SD", index=False)
        stability.to_excel(writer, sheet_name="Signal_Stability", index=False)
        failures.to_excel(writer, sheet_name="Failure_Cases", index=False)
        q.to_excel(writer, sheet_name="Q_Diagnostic", index=False)
    print(f"Summary saved: {out.resolve()}")


if __name__ == "__main__":
    main()
