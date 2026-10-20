"""Independent post-run verification of the CADR LSTM cross-fit outputs.

Re-checks everything from the saved files only (no torch import, no retraining)
and writes the final gate:

    crossfit_results/audit/crossfit_gate.json
    crossfit_results/audit/crossfit_gate.txt      -> READY | BLOCKED_FOR_CROSSFIT_FAILURE

Usage
-----
    python verify_crossfit.py --config config_seed_0.yaml
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

TEST_DOY_RANGE = (182, 243)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def resolve(base: Path, p: str) -> Path:
    q = Path(p)
    return q if q.is_absolute() else (base / q).resolve()


class Checks:
    def __init__(self):
        self.results: dict[str, dict] = {}

    def add(self, group: str, name: str, ok: bool, detail=None):
        self.results.setdefault(group, {})[name] = {"pass": bool(ok), "detail": detail}

    def group_pass(self, group: str) -> bool:
        return all(v["pass"] for v in self.results.get(group, {}).values())

    def failures(self) -> list[str]:
        return [f"{g}: {n} ({v['detail']})" for g, items in self.results.items()
                for n, v in items.items() if not v["pass"]]


def expected_hours(year: int, d0: int, d1: int) -> pd.DatetimeIndex:
    start = pd.to_datetime(f"{year}-{d0}", format="%Y-%j")
    return pd.date_range(start, periods=(d1 - d0 + 1) * 24, freq="h")


def check_series_coverage(c: Checks, group: str, ts: pd.Series, expected: pd.DatetimeIndex):
    ts = pd.DatetimeIndex(pd.to_datetime(ts))
    c.add(group, "chronological", ts.is_monotonic_increasing)
    c.add(group, "no_duplicates", not ts.has_duplicates, int(ts.duplicated().sum()))
    steps = pd.Series(ts).diff().iloc[1:].to_numpy()  # resolution-independent
    n_bad = int((steps != np.timedelta64(1, "h")).sum()) if len(steps) else 0
    c.add(group, "hourly_spacing", n_bad == 0, n_bad)
    c.add(group, "exact_days_coverage", ts.equals(expected),
          {"expected": len(expected), "produced": len(ts),
           "missing": int(len(expected.difference(ts))), "extra": int(len(ts.difference(expected)))})


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config_seed_0.yaml")
    ap.add_argument("--crossfit-config", default="crossfit_config.yaml")
    args = ap.parse_args()

    project_dir = Path(__file__).resolve().parent
    config_path = Path(args.config).resolve()
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    xcfg = yaml.safe_load((project_dir / args.crossfit_config).read_text(encoding="utf-8"))["crossfit"]
    year = int(cfg["general"].get("year", 2018))
    d0, d1 = (int(v) for v in xcfg["train_days"])
    out_dir = resolve(config_path.parent, xcfg["output_dir"])
    full_dir = resolve(config_path.parent, xcfg["full_model_dir"])
    audit = out_dir / "audit"
    house_ids = [int(x) for x in cfg["environment"]["house_ids"]]
    names = [f"household_{h}" for h in house_ids] + ["price"]
    receptive_h = int(cfg["forecast"]["window_size"]) - 1 + max(int(v) for v in cfg["forecast"]["historical_lags"])
    exp = expected_hours(year, d0, d1)
    c = Checks()

    # ---------- run completeness ----------
    manifest_path = audit / "crossfit_manifest.json"
    internal_path = audit / "crossfit_internal_checks.json"
    if not manifest_path.exists() or not internal_path.exists():
        print("[STOP] Cross-fit outputs not found; run crossfit_lstm.py first.", file=sys.stderr)
        return 2
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    internal = json.loads(internal_path.read_text(encoding="utf-8"))
    c.add("completion", "all_four_targets_completed", internal.get("all_targets_completed") is True)
    for n in names:
        c.add("completion", f"{n}_oof_file_present", (out_dir / n / "oof_predictions.csv").exists())

    # ---------- per-target coverage, folds, leakage geometry ----------
    oof = {}
    for n in names:
        path = out_dir / n / "oof_predictions.csv"
        if not path.exists():
            continue
        df = pd.read_csv(path, parse_dates=["timestamp"])
        oof[n] = df
        check_series_coverage(c, f"coverage_{n}", df["timestamp"], exp)
        c.add(f"coverage_{n}", "no_missing_forecasts",
              not df.filter(like="forecast").isna().any().any())
        c.add(f"test_isolation_{n}", "oof_rows_only_days_2_167",
              bool(df["doy"].between(d0, d1).all()))

        metas = [json.loads((out_dir / n / f"fold_{k}" / "fold_meta.json").read_text(encoding="utf-8"))
                 for k in range(int(xcfg["n_folds"]))]
        blocks = []
        for m in metas:
            s, e = (pd.Timestamp(x) for x in m["held_out_range"])
            ex_s, ex_e = (pd.Timestamp(x) for x in m["excluded_range_with_buffer"])
            blocks.append((s, e))
            buf = pd.Timedelta(hours=int(m["buffer_hours"]))
            g = f"leakage_{n}_fold{m['fold_id']}"
            c.add(g, "buffer_ge_receptive_field", int(m["buffer_hours"]) >= receptive_h,
                  {"buffer": m["buffer_hours"], "receptive": receptive_h})
            c.add(g, "excluded_range_matches_block_plus_buffer", (ex_s == s - buf) and (ex_e == e + buf))
            ok_targets, ok_inputs, ok_iso = True, True, True
            first_val = pd.to_datetime(f"{year}-{int(cfg['training']['val_range'][0])}", format="%Y-%j")
            for a, b in m["fitting_target_ranges"]:
                a, b = pd.Timestamp(a), pd.Timestamp(b)
                if not (b < ex_s or a > ex_e):
                    ok_targets = False
                # inputs of targets in [a, b] span [a - receptive, b - 1h]
                if (b - pd.Timedelta(hours=1) >= s) and (a - pd.Timedelta(hours=receptive_h) <= e):
                    ok_inputs = False
                if b - pd.Timedelta(hours=1) >= first_val or a.dayofyear < d0:
                    ok_iso = False
            c.add(g, "fit_targets_outside_block_and_buffer", ok_targets)
            c.add(g, "fit_input_windows_exclude_held_out", ok_inputs)
            c.add(g, "fit_targets_and_inputs_before_validation_and_test", ok_iso)
            sel = df[df["fold_id"] == m["fold_id"]]
            c.add(g, "oof_rows_inside_own_block",
                  bool(((sel["timestamp"] >= s) & (sel["timestamp"] <= e)).all())
                  and len(sel) == m["sample_counts"]["held_out_forecast_hours"])
            cnt = m["sample_counts"]
            c.add(g, "sample_counts_consistent",
                  cnt["training_period_samples"] == cnt["excluded_held_out_block"] + cnt["excluded_buffer"]
                  + cnt["excluded_startup_padding"] + cnt["accepted_fitting_samples"], cnt)
        blocks.sort()
        tiling = (blocks[0][0] == exp[0] and blocks[-1][1] == exp[-1]
                  and all(blocks[i + 1][0] - blocks[i][1] == pd.Timedelta(hours=1)
                          for i in range(len(blocks) - 1)))
        c.add(f"folds_{n}", "blocks_tile_days_2_167_contiguously", tiling)
        epochs = {m["fixed_epochs"] for m in metas}
        c.add(f"folds_{n}", "same_fixed_epochs_all_folds", len(epochs) == 1, sorted(epochs))
        seeds = [m["seeds"]["torch_cpu"] for m in metas]
        c.add(f"folds_{n}", "seed_rule_42_plus_fold", seeds == [int(xcfg["seed_base"]) + k for k in range(len(metas))], seeds)

        # epoch provenance: re-derive best epoch from the full-model history
        hist = (full_dir / "load" / f"lstm_load_house_{n.split('_')[1]}_history.csv"
                if n != "price" else full_dir / "price" / "lstm_price_history.csv")
        h = pd.read_csv(hist)
        best = int(h["epoch"].iloc[int(np.argmin(h["validation_loss"].to_numpy()))])
        c.add(f"folds_{n}", "fixed_epochs_equal_full_model_best_epoch", epochs == {best},
              {"fixed": sorted(epochs), "full_model_best": best})

        # startup padding convention
        n_pad = int((~df["padding_free"].astype(bool)).sum())
        full_metrics = (pd.read_csv(full_dir / "load" / "lstm_load_metrics.csv")
                        if n != "price" else pd.read_csv(full_dir / "price" / "lstm_price_metrics.csv"))
        if n != "price":
            full_metrics = full_metrics[full_metrics["house_id"] == int(n.split("_")[1])]
        full_train_n = int(full_metrics.set_index("split").loc["train", "n_samples"])
        c.add(f"padding_{n}", "padded_hours_match_full_model_convention", n_pad == len(exp) - full_train_n,
              {"padded_oof_hours": n_pad, "full_model_train_n": full_train_n})
        c.add(f"padding_{n}", "padded_hours_only_at_series_start",
              bool(n_pad == 0 or (~df["padding_free"].astype(bool)).iloc[:n_pad].all()))

    # ---------- leakage + determinism reports from the training run ----------
    leak = json.loads((audit / "leakage_report.json").read_text(encoding="utf-8"))
    for n, folds in leak.items():
        for fk, res in folds.items():
            c.add(f"leakage_{n}_{fk.replace('_', '')}", "training_run_leakage_checks", res.get("all_pass") is True,
                  {k: v for k, v in res.items() if not v} or None)
    det = json.loads((audit / "determinism_report.json").read_text(encoding="utf-8"))
    c.add("determinism", "report_present_for_all_targets", set(det) == set(names), sorted(det))
    for n, r in det.items():
        c.add("determinism", f"{n}_fold0_reproduced", r.get("pass") is True, r.get("max_abs_diff"))

    # ---------- combined DDQN file ----------
    comb_path = out_dir / "combined" / "ddqn_crossfit_forecasts_days_002_167.csv"
    comb = pd.read_csv(comb_path, parse_dates=["timestamp"])
    check_series_coverage(c, "coverage_combined", comb["timestamp"], exp)
    eu = [f"forecast_load_eu{i}_kw" for i in (1, 2, 3)]
    agg_err = float(np.max(np.abs(comb["forecast_aggregate_load_kw"] - comb[eu].sum(axis=1))))
    c.add("aggregate_identity", "aggregate_equals_eu1_eu2_eu3", agg_err <= 1e-9, agg_err)
    unit_err = float(np.max(np.abs(comb["forecast_price_cent_per_kwh"] * 10.0 - comb["audit_forecast_price_usd_per_mwh"])))
    c.add("price_units", "cent_per_kwh_equals_usd_per_mwh_div_10", unit_err <= 1e-9, unit_err)
    if "price" in oof:
        p = oof["price"]
        e2 = float(np.max(np.abs(p["forecast_price_cent_per_kWh"] * 10.0 - p["forecast_price_USD_per_MWh"])))
        c.add("price_units", "per_target_file_conversion", e2 <= 1e-9, e2)
        e3 = float(np.max(np.abs(comb["forecast_price_cent_per_kwh"].to_numpy() - p["forecast_price_cent_per_kWh"].to_numpy())))
        c.add("price_units", "combined_matches_price_oof", e3 <= 1e-12, e3)
    for i, h in enumerate(house_ids, start=1):
        if f"household_{h}" in oof:
            o = oof[f"household_{h}"]
            e4 = float(np.max(np.abs(comb[f"forecast_load_eu{i}_kw"].to_numpy() - o["forecast_load_kw"].to_numpy())))
            c.add("aggregate_identity", f"eu{i}_matches_household_{h}_oof", e4 <= 1e-12, e4)
    c.add("coverage_combined", "no_missing_forecasts",
          not comb[[col for col in comb.columns if col.startswith("forecast_")]].isna().any().any())

    # ---------- hashes ----------
    bad, missing = [], []
    for rel, digest in manifest.get("artifact_sha256", {}).items():
        fp = out_dir / rel
        if not fp.exists():
            missing.append(rel)
        elif sha256_file(fp) != digest:
            bad.append(rel)
    c.add("hashes", "all_artifacts_present", not missing, missing or None)
    c.add("hashes", "all_artifact_hashes_match", not bad, bad or None)
    c.add("hashes", "manifest_lists_artifacts", len(manifest.get("artifact_sha256", {})) > 0,
          len(manifest.get("artifact_sha256", {})))
    for key, rel in [("crossfit_lstm.py", "crossfit_lstm.py"), ("crossfit_config.yaml", args.crossfit_config),
                     ("predict_load_lstm.py", "predict_load_lstm.py"),
                     ("predict_price_lstm.py", "predict_price_lstm.py")]:
        c.add("hashes", f"{key}_unchanged_since_run",
              manifest["sha256"].get(key) == sha256_file(project_dir / rel))

    # ---------- full models untouched ----------
    integ = json.loads((audit / "full_model_integrity.json").read_text(encoding="utf-8"))
    c.add("full_models_unchanged", "unchanged_during_run", integ.get("unchanged_during_run") is True)
    now = {str(pp.relative_to(full_dir)).replace("\\", "/"): sha256_file(pp)
           for pp in sorted(full_dir.rglob("*")) if pp.is_file()}
    c.add("full_models_unchanged", "unchanged_since_run", now == integ.get("hashes"))
    for sub in ("load", "price"):
        fm = json.loads((full_dir / sub / f"lstm_{sub}_manifest.json").read_text(encoding="utf-8"))["sha256"]
        c.add("full_models_unchanged", f"{sub}_predictions_match_full_manifest",
              sha256_file(full_dir / sub / f"lstm_{sub}_predictions_all.csv") == fm["predictions"])
        c.add("full_models_unchanged", f"{sub}_metrics_match_full_manifest",
              sha256_file(full_dir / sub / f"lstm_{sub}_metrics.csv") == fm["metrics"])

    # ---------- gate ----------
    failures = c.failures()
    gate = "READY" if not failures else "BLOCKED_FOR_CROSSFIT_FAILURE"
    metrics = pd.read_csv(audit / "crossfit_oof_metrics.csv")
    gate_obj = {
        "gate": gate,
        "failures": failures,
        "combined_file": str(comb_path.relative_to(out_dir)).replace("\\", "/"),
        "combined_file_sha256": sha256_file(comb_path),
        "train_validation_gap_informational": metrics[[
            "target", "OOF_WAPE_percent", "full_model_validation_WAPE_percent",
            "relative_gap_validation_vs_OOF", "preregistered_threshold",
            "gap_exceeds_threshold_informational"]].to_dict(orient="records"),
        "note": "The gap is reported transparently and does not gate; no further tuning is performed.",
        "checks": c.results,
    }
    (audit / "crossfit_gate.json").write_text(json.dumps(gate_obj, indent=2, default=str), encoding="utf-8")
    (audit / "crossfit_gate.txt").write_text(gate + "\n", encoding="utf-8")

    # ---------- console summary ----------
    label = {f"household_{h}": f"EU{i} (household {h})" for i, h in enumerate(house_ids, start=1)}
    label["price"] = "Price"
    print("\nCROSSFIT COMPLETE\n" if gate == "READY" else "\nCROSSFIT FINISHED WITH FAILURES\n")
    for _, r in metrics.iterrows():
        unit = "kW" if r["unit"] == "kW" else "$/MWh"
        extra = f"  (= {r['OOF_MAE'] / 10:.4f} cent/kWh)" if r["target"] == "price" else ""
        print(f"{label[r['target']]}:")
        print(f"  OOF MAE  = {r['OOF_MAE']:.4f} {unit}{extra}")
        print(f"  OOF RMSE = {r['OOF_RMSE']:.4f} {unit}")
        print(f"  OOF WAPE = {r['OOF_WAPE_percent']:.2f} %")
        print(f"  gap vs validation = {r['relative_gap_validation_vs_OOF']:+.3f} "
              f"(threshold {r['preregistered_threshold']:.2f}; informational)\n")
    cov = c.results.get("coverage_combined", {}).get("exact_days_coverage", {}).get("detail", {})
    dup = c.results.get("coverage_combined", {}).get("no_duplicates", {}).get("detail", "?")
    gaps = c.results.get("coverage_combined", {}).get("hourly_spacing", {}).get("detail", "?")
    print("Coverage:")
    print(f"  expected hours = {cov.get('expected')}")
    print(f"  produced hours = {cov.get('produced')}")
    print(f"  duplicates = {dup}")
    print(f"  gaps = {gaps}\n")
    leak_ok = all(c.group_pass(g) for g in c.results if g.startswith("leakage_") or g.startswith("test_isolation_"))
    status = lambda ok: "PASS" if ok else "FAIL"  # noqa: E731
    print(f"Leakage checks: {status(leak_ok)}")
    print(f"Aggregate identity: {status(c.group_pass('aggregate_identity'))}")
    print(f"Price-unit conversion: {status(c.group_pass('price_units'))}")
    print(f"Determinism: {status(c.group_pass('determinism'))}")
    print(f"Hashes: {status(c.group_pass('hashes'))}")
    print(f"Full-model files unchanged: {status(c.group_pass('full_models_unchanged'))}")
    print(f"\nGate: {gate}")
    if failures:
        print("\nFailures:")
        for f in failures:
            print(f"  - {f}")
    return 0 if gate == "READY" else 3


if __name__ == "__main__":
    sys.exit(main())
