"""Leakage-controlled out-of-fold LSTM forecasts for the CADR DDQN training days.

Scope
-----
Generates out-of-fold (OOF) one-step-ahead forecasts for DDQN training days
2-167 for the four Appendix-A forecasting targets (households 661, 3039, 8565
and the wholesale price). Validation (days 168-181) and test (July/August)
forecasts are NOT touched: they remain the outputs of the existing full models
under results/forecast/.

Design (frozen)
---------------
* 5 contiguous day blocks over days 2-167.
* For each block, a fresh model (same Appendix-A specification) is fitted on
  training-period samples whose target lies outside the block and outside a
  73-h buffer on both sides of it, and then forecasts every hour of the block.
* Scalers are fitted on each fold's accepted fitting samples only.
* Each fold is trained for a fixed number of epochs equal to the best epoch of
  the corresponding full model (read from its saved history). No search.
* Fold seed = 42 + fold index.

Single source of truth
----------------------
Feature construction, the LSTM class and the scaler functions are imported
from the project's own predict_load_lstm.py / predict_price_lstm.py. Their
SHA-256 hashes (and the config's) must equal those recorded in the full-model
manifests, otherwise the run stops: the folds are guaranteed to use exactly
the code that produced the full models.

Usage
-----
    python crossfit_lstm.py --config config_seed_0.yaml [--check-only] [--overwrite]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

TARGET_LOAD = "load"
TARGET_PRICE = "price"
TEST_DOY_RANGE = (182, 243)  # July + August 2018 (non-leap year)


class CrossfitError(RuntimeError):
    """Raised for any condition under which the cross-fit must not proceed."""


# --------------------------------------------------------------------------
# Small utilities
# --------------------------------------------------------------------------

def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def resolve(base: Path, p: str) -> Path:
    q = Path(p)
    return q if q.is_absolute() else (base / q).resolve()


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=str), encoding="utf-8")


def contiguous_ranges(ts: pd.DatetimeIndex) -> list[list[str]]:
    """Collapse an hourly timestamp set into [start, end] inclusive runs."""
    if len(ts) == 0:
        return []
    ts = pd.DatetimeIndex(sorted(ts))
    # Compare as timedeltas (independent of the datetime resolution pandas chose).
    steps = pd.Series(ts).diff().iloc[1:].to_numpy()
    breaks = np.where(steps != np.timedelta64(1, "h"))[0]
    starts = np.r_[0, breaks + 1]
    ends = np.r_[breaks, len(ts) - 1]
    return [[str(ts[s]), str(ts[e])] for s, e in zip(starts, ends)]


def wape(y: np.ndarray, f: np.ndarray) -> float:
    return float(np.sum(np.abs(y - f)) / np.sum(np.abs(y)) * 100.0)


def point_metrics(y: np.ndarray, f: np.ndarray) -> dict:
    e = y - f
    return {
        "MAE": float(np.mean(np.abs(e))),
        "RMSE": float(np.sqrt(np.mean(e ** 2))),
        "WAPE_percent": wape(y, f),
    }


# --------------------------------------------------------------------------
# Environment and provenance checks
# --------------------------------------------------------------------------

def load_project_modules(project_dir: Path):
    """Import the project's forecasting modules (single source of truth)."""
    sys.path.insert(0, str(project_dir))
    try:
        import predict_load_lstm as PL  # noqa: WPS433
        import predict_price_lstm as PP  # noqa: WPS433
    except ImportError as exc:
        raise CrossfitError(
            "Could not import predict_load_lstm.py / predict_price_lstm.py. "
            "Place the cross-fit files in the project root next to them."
        ) from exc
    required = ["build_lstm_samples", "LSTMForecaster", "fit_scaler", "apply_scaler",
                "validate_article_locked_config", "ARTICLE_INPUT_SIZE", "ARTICLE_LAGS"]
    for mod in (PL, PP):
        missing = [name for name in required if not hasattr(mod, name)]
        if missing:
            raise CrossfitError(f"{mod.__name__} is missing expected members: {missing}")
    return PL, PP


def check_provenance(project_dir: Path, config_path: Path, full_dir: Path) -> dict:
    """The scripts/config/data must be exactly those that produced the full models."""
    out = {}
    for target, script, sub in [(TARGET_LOAD, "predict_load_lstm.py", "load"),
                                (TARGET_PRICE, "predict_price_lstm.py", "price")]:
        man_path = full_dir / sub / f"lstm_{sub}_manifest.json"
        if not man_path.exists():
            raise CrossfitError(f"Full-model manifest not found: {man_path}")
        man = json.loads(man_path.read_text(encoding="utf-8"))
        rec = man.get("sha256", {})
        actual = {
            "script": sha256_file(project_dir / script),
            "config": sha256_file(config_path),
        }
        for key, value in actual.items():
            if rec.get(key) != value:
                raise CrossfitError(
                    f"{key} hash for the {target} model does not match the full-model manifest "
                    f"({man_path}).\n  manifest: {rec.get(key)}\n  current : {value}\n"
                    "The cross-fit must use exactly the code/config that produced the full models."
                )
        if man.get("holiday_backend") not in (None, "holidays.US"):
            raise CrossfitError(f"Full {target} model was not trained with holidays.US.")
        out[target] = {"manifest": str(man_path), "manifest_sha256": rec,
                       "holiday_backend": man.get("holiday_backend")}
    return out


def full_model_artifact_hashes(full_dir: Path) -> dict:
    """Hash every existing full-model file (used to prove they are untouched)."""
    return {str(p.relative_to(full_dir)).replace("\\", "/"): sha256_file(p)
            for p in sorted(full_dir.rglob("*")) if p.is_file()}


def configure_torch(xcfg: dict):
    import torch
    if str(xcfg.get("device", "cpu")).lower() != "cpu":
        raise CrossfitError("This cross-fit stage is specified for device: cpu (reproducibility).")
    torch.set_num_threads(int(xcfg.get("torch_num_threads", 4)))
    if bool(xcfg.get("deterministic_algorithms", True)):
        torch.use_deterministic_algorithms(True)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    return torch


def seed_everything(seed: int, torch) -> dict:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    cuda = torch.cuda.is_available()
    if cuda:
        torch.cuda.manual_seed_all(seed)
    return {"python_random": seed, "numpy": seed, "torch_cpu": seed,
            "torch_cuda": seed if cuda else "not_applicable (CPU run)"}


# --------------------------------------------------------------------------
# Configuration and inputs
# --------------------------------------------------------------------------

def read_best_epoch(history_path: Path, patience: int, max_epochs: int) -> dict:
    if not history_path.exists():
        raise CrossfitError(f"Full-model training history not found: {history_path}")
    h = pd.read_csv(history_path)
    for col in ("epoch", "validation_loss"):
        if col not in h.columns:
            raise CrossfitError(f"{history_path} has no '{col}' column.")
    if h["validation_loss"].isna().any() or len(h) == 0:
        raise CrossfitError(f"{history_path} has missing validation losses.")
    # The full-model loop keeps the FIRST strict minimum (val_loss < best_val).
    idx = int(np.argmin(h["validation_loss"].to_numpy()))
    best = int(h["epoch"].iloc[idx])
    last = int(h["epoch"].iloc[-1])
    consistent = (last == best + patience) or (last == max_epochs)
    if not consistent:
        raise CrossfitError(
            f"{history_path}: last epoch {last} is inconsistent with best epoch {best} "
            f"+ patience {patience} (or max epochs {max_epochs}); cannot trust the best epoch."
        )
    if best < 1:
        raise CrossfitError(f"{history_path}: invalid best epoch {best}.")
    return {"history_file": str(history_path), "best_epoch": best, "last_epoch": last,
            "best_validation_loss": float(h["validation_loss"].iloc[idx]),
            "history_sha256": sha256_file(history_path)}


def build_targets(cfg: dict, config_path: Path, full_dir: Path) -> list[dict]:
    from utils.load_demand import load_demand
    from utils.load_price import load_price

    f = cfg["forecast"]
    year = int(cfg["general"].get("year", 2018))
    patience, max_epochs = int(f["patience"]), int(f["epochs"])
    house_ids = [int(x) for x in cfg["environment"]["house_ids"]]
    if len(house_ids) != 3:
        raise CrossfitError(f"Expected exactly three households, got {house_ids}.")

    load_path = resolve(config_path.parent, cfg["general"]["load_file"])
    price_path = resolve(config_path.parent, cfg["general"]["price_file"])
    df = load_demand(str(load_path), house_ids=house_ids)

    targets = []
    for eu, hid in enumerate(house_ids, start=1):
        sub = df[df["dataid"] == hid].sort_values("dt")
        series = sub.set_index("dt")["total"].astype(float)
        series = series[series.index.year == year]
        targets.append({
            "name": f"household_{hid}", "kind": TARGET_LOAD, "house_id": hid, "eu": eu,
            "series": series, "unit": "kW", "data_path": load_path,
            "full_predictions": full_dir / "load" / "lstm_load_predictions_all.csv",
            "full_metrics": full_dir / "load" / "lstm_load_metrics.csv",
            "epoch": read_best_epoch(full_dir / "load" / f"lstm_load_house_{hid}_history.csv",
                                     patience, max_epochs),
        })
    internal = load_price(str(price_path))
    series_usd = internal["price"].astype(float) * float(f["cent_kwh_to_usd_mwh_multiplier"])
    series_usd = series_usd[series_usd.index.year == year]
    targets.append({
        "name": "price", "kind": TARGET_PRICE, "house_id": None, "eu": None,
        "series": series_usd, "unit": "USD_per_MWh", "data_path": price_path,
        "full_predictions": full_dir / "price" / "lstm_price_predictions_all.csv",
        "full_metrics": full_dir / "price" / "lstm_price_metrics.csv",
        "epoch": read_best_epoch(full_dir / "price" / "lstm_price_history.csv",
                                 patience, max_epochs),
    })
    return targets


def make_folds(train_days: tuple[int, int], n_folds: int, year: int) -> list[dict]:
    days = np.arange(int(train_days[0]), int(train_days[1]) + 1)
    folds = []
    for k, block in enumerate(np.array_split(days, n_folds)):
        d0, d1 = int(block[0]), int(block[-1])
        start = pd.to_datetime(f"{year}-{d0}", format="%Y-%j")
        end = pd.to_datetime(f"{year}-{d1}", format="%Y-%j") + pd.Timedelta(hours=23)
        folds.append({"fold_id": k, "first_day": d0, "last_day": d1, "n_days": int(len(block)),
                      "block_start": start, "block_end": end})
    return folds


# --------------------------------------------------------------------------
# Fold construction and leakage checks
# --------------------------------------------------------------------------

def fold_masks(ts: pd.DatetimeIndex, doy: np.ndarray, padding_free: np.ndarray,
               fold: dict, train_days, buffer_h: int) -> dict:
    in_train = (doy >= train_days[0]) & (doy <= train_days[1])
    in_block = (ts >= fold["block_start"]) & (ts <= fold["block_end"])
    buf = pd.Timedelta(hours=buffer_h)
    in_buffer = (((ts >= fold["block_start"] - buf) & (ts < fold["block_start"]))
                 | ((ts > fold["block_end"]) & (ts <= fold["block_end"] + buf)))
    fit = in_train & ~in_block & ~in_buffer & padding_free
    return {
        "in_train": in_train, "held_out": in_block & in_train, "buffer": in_buffer & in_train,
        "fit": fit,
        "counts": {
            "training_period_samples": int(in_train.sum()),
            "excluded_held_out_block": int((in_train & in_block).sum()),
            "excluded_buffer": int((in_train & in_buffer).sum()),
            "excluded_startup_padding": int((in_train & ~in_block & ~in_buffer & ~padding_free).sum()),
            "accepted_fitting_samples": int(fit.sum()),
            "held_out_forecast_hours": int((in_train & in_block).sum()),
            "held_out_padded_hours": int((in_train & in_block & ~padding_free).sum()),
        },
    }


def leakage_checks(PM, target: dict, cfg: dict, year: int, X: np.ndarray, ts: pd.DatetimeIndex,
                   doy: np.ndarray, masks: dict, fold: dict, receptive_h: int,
                   buffer_h: int) -> dict:
    """Structural and empirical leakage checks for one fold of one target."""
    res = {}
    fit_ts, held_ts = ts[masks["fit"]], ts[masks["held_out"]]
    buf_ts = ts[masks["buffer"]]
    res["held_out_targets_not_in_fit"] = bool(len(fit_ts.intersection(held_ts)) == 0)
    res["buffer_targets_not_in_fit"] = bool(len(fit_ts.intersection(buf_ts)) == 0)

    # Input span of a sample with target t is [t - receptive_h, t - 1h].
    lo = fit_ts - pd.Timedelta(hours=receptive_h)
    hi = fit_ts - pd.Timedelta(hours=1)
    overlap = (hi >= fold["block_start"]) & (lo <= fold["block_end"])
    res["fit_input_windows_exclude_held_out"] = bool(not overlap.any())
    res["buffer_covers_receptive_field"] = bool(buffer_h >= receptive_h)

    # Empirical check: perturb every held-out observation and rebuild the inputs.
    perturbed = target["series"].copy()
    block_mask = (perturbed.index >= fold["block_start"]) & (perturbed.index <= fold["block_end"])
    perturbed.loc[block_mask] = perturbed.loc[block_mask] + 1.0e4
    Xp, _, meta_p, _ = PM.build_lstm_samples(perturbed, cfg, year)
    if not pd.DatetimeIndex(pd.to_datetime(meta_p["timestamp"])).equals(ts):
        raise CrossfitError("Perturbed rebuild produced a different timestamp index.")
    res["fit_inputs_invariant_to_held_out_perturbation"] = bool(
        np.array_equal(X[masks["fit"]], Xp[masks["fit"]]))
    # Sanity: the perturbation must actually reach some samples (else the test is vacuous).
    res["perturbation_effective"] = bool(not np.array_equal(X, Xp))

    # Test/validation isolation of the fitting set (targets AND inputs).
    first_val = pd.to_datetime(f"{year}-{int(cfg['training']['val_range'][0])}", format="%Y-%j")
    res["fit_targets_within_train_days"] = bool(
        ((doy[masks["fit"]] >= 2) & (doy[masks["fit"]] <= int(cfg["training"]["train_ranges"][0][1]))).all())
    res["fit_inputs_before_validation"] = bool((hi < first_val).all()) if len(hi) else True
    res["no_test_period_in_fit"] = bool(
        not ((doy[masks["fit"]] >= TEST_DOY_RANGE[0]) & (doy[masks["fit"]] <= TEST_DOY_RANGE[1])).any())
    res["all_pass"] = bool(all(v for k, v in res.items()))
    return res


# --------------------------------------------------------------------------
# Training and prediction
# --------------------------------------------------------------------------

def train_fixed_epochs(PM, torch, Xn_fit, yn_fit, fcfg: dict, epochs: int, seed: int):
    from torch.utils.data import DataLoader, TensorDataset

    seeds = seed_everything(seed, torch)
    model = PM.LSTMForecaster(
        input_size=int(PM.ARTICLE_INPUT_SIZE), hidden_size=int(fcfg["hidden_size"]),
        num_layers=int(fcfg["num_layers"]), dropout=float(fcfg["dropout"]),
    )
    loader = DataLoader(
        TensorDataset(torch.as_tensor(Xn_fit, dtype=torch.float32),
                      torch.as_tensor(yn_fit, dtype=torch.float32)),
        batch_size=int(fcfg["batch_size"]), shuffle=True,
        generator=torch.Generator().manual_seed(seed), num_workers=0,
    )
    opt = torch.optim.Adam(model.parameters(), lr=float(fcfg["learning_rate"]),
                           weight_decay=float(fcfg["weight_decay"]))
    loss_fn = torch.nn.MSELoss()
    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        total, n = 0.0, 0
        for xb, yb in loader:
            opt.zero_grad(set_to_none=True)
            loss = loss_fn(model(xb), yb)
            loss.backward()
            opt.step()
            total += float(loss.item()) * xb.shape[0]
            n += xb.shape[0]
        history.append({"epoch": epoch, "train_loss": total / max(n, 1)})
    return model, history, seeds


def predict_block(torch, model, Xn: np.ndarray) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        out = model(torch.as_tensor(Xn, dtype=torch.float32)).cpu().numpy()
    # Promote to float64 before un-scaling so stored forecasts, the aggregate sum and
    # the unit conversion are computed in double precision (NumPy 2 keeps float32 otherwise).
    return out.reshape(-1).astype(np.float64)


# --------------------------------------------------------------------------
# Main pipeline
# --------------------------------------------------------------------------

def run(args) -> int:
    config_path = Path(args.config).resolve()
    project_dir = Path(__file__).resolve().parent
    os.chdir(project_dir)
    sys.path.insert(0, str(project_dir))

    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    xcfg = yaml.safe_load((project_dir / args.crossfit_config).read_text(encoding="utf-8"))["crossfit"]
    fcfg = cfg["forecast"]
    year = int(cfg["general"].get("year", 2018))
    full_dir = resolve(config_path.parent, xcfg["full_model_dir"])
    out_dir = resolve(config_path.parent, xcfg["output_dir"])

    print("== CADR LSTM cross-fitting ==")
    print(f"Config: {config_path}")

    # ---- static configuration checks ----
    PL, PP = load_project_modules(project_dir)
    PL.validate_article_locked_config(cfg)
    PP.validate_article_locked_config(cfg)
    try:
        import holidays  # noqa: F401
    except ImportError as exc:
        raise CrossfitError("Package 'holidays' is required (holidays.US, as in the full models).") from exc

    train_days = tuple(int(v) for v in xcfg["train_days"])
    cfg_train = cfg["training"]["train_ranges"]
    if len(cfg_train) != 1 or tuple(int(v) for v in cfg_train[0]) != train_days:
        raise CrossfitError(f"crossfit.train_days {train_days} != training.train_ranges {cfg_train}.")
    receptive_h = int(fcfg["window_size"]) - 1 + max(int(v) for v in fcfg["historical_lags"])
    buffer_h = int(xcfg["buffer_hours"])
    if buffer_h < receptive_h:
        raise CrossfitError(f"buffer_hours={buffer_h} is smaller than the receptive field {receptive_h} h.")
    gap_thr = float(xcfg["gap_threshold"])
    if abs(gap_thr - float(fcfg["train_validation_wape_relative_gap_threshold"])) > 1e-12:
        raise CrossfitError("crossfit.gap_threshold differs from the pre-registered forecast threshold.")
    print(f"[PASS] Appendix-A config locked; receptive field {receptive_h} h; buffer {buffer_h} h.")

    provenance = check_provenance(project_dir, config_path, full_dir)
    print("[PASS] Scripts and config match the full-model manifests (SHA-256).")
    full_hashes_before = full_model_artifact_hashes(full_dir)

    targets = build_targets(cfg, config_path, full_dir)
    folds = make_folds(train_days, int(xcfg["n_folds"]), year)
    for fd in folds:
        print(f"  fold {fd['fold_id']}: days {fd['first_day']:3d}-{fd['last_day']:3d} "
              f"({fd['n_days']} days) {fd['block_start']} -> {fd['block_end']}")
    for t in targets:
        print(f"  {t['name']:16s} fixed epochs = {t['epoch']['best_epoch']} "
              f"(full-model best; last={t['epoch']['last_epoch']})")

    # ---- build samples once per target; structural + leakage checks for every fold ----
    prepared, leakage = [], {}
    for t in targets:
        PM = PL if t["kind"] == TARGET_LOAD else PP
        X, y, meta, feat_names = PM.build_lstm_samples(t["series"], cfg, year)
        ts = pd.DatetimeIndex(pd.to_datetime(meta["timestamp"]))
        doy = np.asarray(meta["doy"], dtype=int)
        if "padding_free" not in meta:
            raise CrossfitError("build_lstm_samples() does not return the padding_free flag.")
        pf = np.asarray(meta["padding_free"], dtype=bool)
        if X.shape[1:] != (int(fcfg["window_size"]), int(PM.ARTICLE_INPUT_SIZE)):
            raise CrossfitError(f"{t['name']}: unexpected sample shape {X.shape}.")
        # The series must reproduce the actual values used by the full model.
        full_pred = pd.read_csv(t["full_predictions"], parse_dates=["timestamp"])
        if t["kind"] == TARGET_LOAD:
            full_pred = full_pred[full_pred["house_id"] == t["house_id"]]
            actual_col = "actual_load_kw"
        else:
            actual_col = "actual_price_USD_per_MWh"
        fp = full_pred.set_index("timestamp")[actual_col].reindex(ts)
        if fp.isna().any() or not np.allclose(fp.to_numpy(), y.reshape(-1), rtol=0, atol=1e-9):
            raise CrossfitError(f"{t['name']}: rebuilt targets differ from the full-model actuals.")
        expected_train_hours = (train_days[1] - train_days[0] + 1) * 24
        in_train = (doy >= train_days[0]) & (doy <= train_days[1])
        if int(in_train.sum()) != expected_train_hours:
            raise CrossfitError(f"{t['name']}: {int(in_train.sum())} training hours, expected {expected_train_hours}.")
        fold_info = []
        leakage[t["name"]] = {}
        for fd in folds:
            m = fold_masks(ts, doy, pf, fd, train_days, buffer_h)
            lk = leakage_checks(PM, t, cfg, year, X, ts, doy, m, fd, receptive_h, buffer_h)
            leakage[t["name"]][f"fold_{fd['fold_id']}"] = lk
            if not lk["all_pass"]:
                raise CrossfitError(f"Leakage check failed for {t['name']} fold {fd['fold_id']}: {lk}")
            fold_info.append(m)
        n_pad_train = int((in_train & ~pf).sum())
        print(f"[PASS] {t['name']}: samples {X.shape}, training hours {int(in_train.sum())}, "
              f"startup-padded training samples {n_pad_train}; leakage checks pass for all folds.")
        prepared.append({"target": t, "PM": PM, "X": X, "y": y, "ts": ts, "doy": doy, "pf": pf,
                         "folds": fold_info, "feature_names": feat_names,
                         "n_pad_train": n_pad_train})

    if args.check_only:
        print("[PASS] Static cross-fit checks complete. No training was run.")
        return 0

    # ---- output directory protection ----
    if out_dir.exists() and any(out_dir.iterdir()):
        if not args.overwrite:
            raise CrossfitError(f"{out_dir} already exists and is not empty. Use --overwrite to replace it.")
        import shutil
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    torch = configure_torch(xcfg)
    started = time.time()
    seed_base = int(xcfg["seed_base"])
    oof_frames, target_records = {}, {}

    for p in prepared:
        t, PM = p["target"], p["PM"]
        tdir = out_dir / t["name"]
        tdir.mkdir(parents=True, exist_ok=True)
        epochs = int(t["epoch"]["best_epoch"])
        rows, fold_records = [], []
        for fd, m in zip(folds, p["folds"]):
            k = fd["fold_id"]
            seed = seed_base + k
            scaler = PM.fit_scaler(p["X"][m["fit"]], p["y"][m["fit"]])
            Xn, yn = PM.apply_scaler(p["X"], p["y"], scaler)
            t0 = time.time()
            model, history, seeds = train_fixed_epochs(PM, torch, Xn[m["fit"]], yn[m["fit"]],
                                                       fcfg, epochs, seed)
            pred_n = predict_block(torch, model, Xn[m["held_out"]])
            pred = pred_n * float(scaler["y_std"]) + float(scaler["y_mean"])
            fdir = tdir / f"fold_{k}"
            fdir.mkdir(parents=True, exist_ok=True)
            torch.save({"state_dict": model.state_dict(), "target": t["name"], "fold_id": k,
                        "seed": seed, "epochs": epochs, "input_size": int(PM.ARTICLE_INPUT_SIZE),
                        "hidden_size": int(fcfg["hidden_size"]), "num_layers": int(fcfg["num_layers"]),
                        "dropout": float(fcfg["dropout"]), "feature_names": p["feature_names"]},
                       fdir / "model.pt")
            np.savez(fdir / "scaler.npz", X_mean=scaler["X_mean"], X_std=scaler["X_std"],
                     y_mean=np.asarray([scaler["y_mean"]]), y_std=np.asarray([scaler["y_std"]]))
            pd.DataFrame(history).to_csv(fdir / "training_history.csv", index=False)
            fit_ts = p["ts"][m["fit"]]
            rec = {
                "target": t["name"], "fold_id": k,
                "held_out_days": [fd["first_day"], fd["last_day"]],
                "held_out_range": [str(fd["block_start"]), str(fd["block_end"])],
                "buffer_hours": buffer_h,
                "excluded_range_with_buffer": [str(fd["block_start"] - pd.Timedelta(hours=buffer_h)),
                                               str(fd["block_end"] + pd.Timedelta(hours=buffer_h))],
                "fitting_target_ranges": contiguous_ranges(fit_ts),
                "sample_counts": m["counts"],
                "fixed_epochs": epochs, "seeds": seeds, "device": "cpu",
                "torch_num_threads": int(xcfg.get("torch_num_threads", 4)),
                "scaler_fit_on": "accepted fitting samples of this fold only",
                "final_train_loss": history[-1]["train_loss"],
                "fit_seconds": round(time.time() - t0, 2),
                "sha256": {"model": sha256_file(fdir / "model.pt"),
                           "scaler": sha256_file(fdir / "scaler.npz"),
                           "training_history": sha256_file(fdir / "training_history.csv")},
            }
            write_json(fdir / "fold_meta.json", rec)
            fold_records.append(rec)
            frame = pd.DataFrame({
                "timestamp": p["ts"][m["held_out"]], "doy": p["doy"][m["held_out"]],
                "fold_id": k, "padding_free": p["pf"][m["held_out"]],
                "actual": p["y"].reshape(-1)[m["held_out"]], "forecast": pred, "model_seed": seed,
            })
            rows.append(frame)
            print(f"  {t['name']:16s} fold {k}: fit={m['counts']['accepted_fitting_samples']:5d} "
                  f"held-out={m['counts']['held_out_forecast_hours']:4d} epochs={epochs:3d} "
                  f"seed={seed} ({rec['fit_seconds']:.1f}s)")

        oof = pd.concat(rows, ignore_index=True).sort_values("timestamp").reset_index(drop=True)
        if t["kind"] == TARGET_LOAD:
            oof = oof.rename(columns={"actual": "actual_load_kw", "forecast": "forecast_load_kw"})
            oof.insert(1, "house_id", t["house_id"])
        else:
            oof = oof.rename(columns={"actual": "actual_price_USD_per_MWh",
                                      "forecast": "forecast_price_USD_per_MWh"})
            oof["actual_price_cent_per_kWh"] = oof["actual_price_USD_per_MWh"] / 10.0
            oof["forecast_price_cent_per_kWh"] = oof["forecast_price_USD_per_MWh"] / 10.0
        oof_path = tdir / "oof_predictions.csv"
        oof.to_csv(oof_path, index=False)
        oof_frames[t["name"]] = oof
        target_records[t["name"]] = {"folds": fold_records, "oof_file": str(oof_path),
                                     "oof_sha256": sha256_file(oof_path),
                                     "fixed_epochs": t["epoch"],
                                     "startup_padded_training_samples": p["n_pad_train"]}

    # ---- determinism check: retrain fold 0 of every target ----
    determinism = {}
    if bool(xcfg.get("determinism_check", True)):
        tol = float(xcfg.get("determinism_tolerance", 1e-6))
        for p in prepared:
            t, PM = p["target"], p["PM"]
            m, fd = p["folds"][0], folds[0]
            scaler = PM.fit_scaler(p["X"][m["fit"]], p["y"][m["fit"]])
            Xn, yn = PM.apply_scaler(p["X"], p["y"], scaler)
            model, _, _ = train_fixed_epochs(PM, torch, Xn[m["fit"]], yn[m["fit"]], fcfg,
                                             int(t["epoch"]["best_epoch"]), seed_base + fd["fold_id"])
            again = (predict_block(torch, model, Xn[m["held_out"]]) * float(scaler["y_std"])
                     + float(scaler["y_mean"]))
            col = "forecast_load_kw" if t["kind"] == TARGET_LOAD else "forecast_price_USD_per_MWh"
            first = oof_frames[t["name"]].loc[oof_frames[t["name"]]["fold_id"] == 0, col].to_numpy()
            diff = float(np.max(np.abs(first - again)))
            determinism[t["name"]] = {"fold": 0, "max_abs_diff": diff, "tolerance": tol,
                                      "pass": bool(diff <= tol)}
            print(f"  determinism {t['name']:16s} fold 0 max|diff| = {diff:.3e}")

    # ---- combined DDQN-facing file ----
    comb_dir = out_dir / "combined"
    comb_dir.mkdir(parents=True, exist_ok=True)
    loads = [p["target"] for p in prepared if p["target"]["kind"] == TARGET_LOAD]
    base = oof_frames["price"][["timestamp", "doy", "fold_id"]].copy()
    combined = base.copy()
    pf_all = oof_frames["price"]["padding_free"].to_numpy().copy()
    for t in loads:
        o = oof_frames[t["name"]].set_index("timestamp").reindex(combined["timestamp"])
        if o["forecast_load_kw"].isna().any():
            raise CrossfitError(f"{t['name']}: OOF forecasts missing for some price timestamps.")
        combined[f"forecast_load_eu{t['eu']}_kw"] = o["forecast_load_kw"].to_numpy()
        pf_all &= o["padding_free"].to_numpy().astype(bool)
    combined["forecast_aggregate_load_kw"] = sum(combined[f"forecast_load_eu{t['eu']}_kw"] for t in loads)
    combined["forecast_price_cent_per_kwh"] = oof_frames["price"]["forecast_price_USD_per_MWh"].to_numpy() / 10.0
    combined["inputs_padding_free_all_targets"] = pf_all
    for t in loads:
        o = oof_frames[t["name"]].set_index("timestamp").reindex(combined["timestamp"])
        combined[f"audit_actual_load_eu{t['eu']}_kw"] = o["actual_load_kw"].to_numpy()
    combined["audit_actual_aggregate_load_kw"] = sum(combined[f"audit_actual_load_eu{t['eu']}_kw"] for t in loads)
    combined["audit_actual_price_cent_per_kwh"] = oof_frames["price"]["actual_price_USD_per_MWh"].to_numpy() / 10.0
    combined["audit_forecast_price_usd_per_mwh"] = oof_frames["price"]["forecast_price_USD_per_MWh"].to_numpy()
    comb_path = comb_dir / "ddqn_crossfit_forecasts_days_002_167.csv"
    combined.to_csv(comb_path, index=False)
    write_json(comb_dir / "column_map.json", {
        "file": comb_path.name,
        "rows": "one per hour, days 2-167 of 2018 (DDQN training days)",
        "eu_to_household": {f"eu{t['eu']}": t["house_id"] for t in loads},
        "forecast_columns": [c for c in combined.columns if c.startswith("forecast_")],
        "audit_columns_do_not_use_as_ddqn_inputs": [c for c in combined.columns if c.startswith("audit_")],
        "aggregate_definition": "forecast_aggregate_load_kw = eu1 + eu2 + eu3 (exact sum)",
        "price_unit": "cent/kWh = (USD/MWh) / 10",
        "startup_padding": "inputs_padding_free_all_targets is False for early hours whose input "
                           "window reached before the start of the series (zero-padded, as in the "
                           "full models); these hours are forecast but excluded from metrics.",
        "days_168_onward": "not in this file; use the existing full-model forecasts",
    })

    # ---- metrics and gap audit ----
    audit_dir = out_dir / "audit"
    audit_dir.mkdir(parents=True, exist_ok=True)
    metric_rows = []
    for p in prepared:
        t = p["target"]
        o = oof_frames[t["name"]]
        o = o[o["padding_free"]]
        if t["kind"] == TARGET_LOAD:
            y, fc = o["actual_load_kw"].to_numpy(), o["forecast_load_kw"].to_numpy()
            fm = pd.read_csv(t["full_metrics"])
            fm = fm[fm["house_id"] == t["house_id"]].set_index("split")
        else:
            y, fc = o["actual_price_USD_per_MWh"].to_numpy(), o["forecast_price_USD_per_MWh"].to_numpy()
            fm = pd.read_csv(t["full_metrics"]).set_index("split")
        pm = point_metrics(y, fc)
        val_w = float(fm.loc["validation", "WAPE_percent"])
        insample_w = float(fm.loc["train", "WAPE_percent"])
        gap = (val_w - pm["WAPE_percent"]) / max(abs(val_w), 1e-12)
        metric_rows.append({
            "target": t["name"], "unit": t["unit"], "n_padding_free_hours": int(len(o)),
            "OOF_MAE": pm["MAE"], "OOF_RMSE": pm["RMSE"], "OOF_WAPE_percent": pm["WAPE_percent"],
            "full_model_in_sample_train_WAPE_percent": insample_w,
            "full_model_validation_WAPE_percent": val_w,
            "relative_gap_validation_vs_OOF": gap, "preregistered_threshold": gap_thr,
            "gap_exceeds_threshold_informational": bool(gap > gap_thr),
        })
    metrics_df = pd.DataFrame(metric_rows)
    metrics_df.to_csv(audit_dir / "crossfit_oof_metrics.csv", index=False)
    write_json(audit_dir / "leakage_report.json", leakage)
    write_json(audit_dir / "determinism_report.json", determinism)

    full_hashes_after = full_model_artifact_hashes(full_dir)
    integrity = {
        "full_model_dir": str(full_dir),
        "files_checked": len(full_hashes_before),
        "unchanged_during_run": full_hashes_before == full_hashes_after,
        "hashes": full_hashes_after,
    }
    write_json(audit_dir / "full_model_integrity.json", integrity)

    import holidays as _hol
    import torch as _t
    manifest = {
        "stage": "CADR LSTM cross-fitting (DDQN training days only)",
        "created_utc": pd.Timestamp.now(tz="UTC").isoformat(),
        "runtime_seconds": round(time.time() - started, 1),
        "design": {
            "train_days": list(train_days), "n_folds": int(xcfg["n_folds"]),
            "buffer_hours": buffer_h, "receptive_field_hours": receptive_h,
            "seed_rule": f"seed = {seed_base} + fold_index",
            "epoch_rule": "fixed = best validation epoch of the corresponding full model",
            "scaler_rule": "fitted on each fold's accepted fitting samples only",
            "padding_rule": "padded samples excluded from fitting, scaling and metrics; still forecast",
            "validation_and_test_forecasts": "unchanged; taken from the existing full models",
        },
        "folds": [{k: (str(v) if isinstance(v, pd.Timestamp) else v) for k, v in fd.items()} for fd in folds],
        "targets": target_records,
        "provenance": provenance,
        "environment": {"python": platform.python_version(), "platform": platform.platform(),
                        "numpy": np.__version__, "pandas": pd.__version__, "torch": _t.__version__,
                        "holidays": getattr(_hol, "__version__", "unknown"),
                        "device": "cpu", "torch_num_threads": int(xcfg.get("torch_num_threads", 4)),
                        "deterministic_algorithms": bool(xcfg.get("deterministic_algorithms", True))},
        "sha256": {
            "crossfit_lstm.py": sha256_file(Path(__file__).resolve()),
            "verify_crossfit.py": sha256_file(project_dir / "verify_crossfit.py")
            if (project_dir / "verify_crossfit.py").exists() else None,
            "crossfit_config.yaml": sha256_file(project_dir / args.crossfit_config),
            "predict_load_lstm.py": sha256_file(project_dir / "predict_load_lstm.py"),
            "predict_price_lstm.py": sha256_file(project_dir / "predict_price_lstm.py"),
            "main_config": sha256_file(config_path),
            "load_data": sha256_file(prepared[0]["target"]["data_path"]),
            "price_data": sha256_file(prepared[-1]["target"]["data_path"]),
        },
    }
    artifacts = {}
    for pth in sorted(out_dir.rglob("*")):
        if pth.is_file() and pth.name != "crossfit_manifest.json":
            artifacts[str(pth.relative_to(out_dir)).replace("\\", "/")] = sha256_file(pth)
    manifest["artifact_sha256"] = artifacts
    write_json(audit_dir / "crossfit_manifest.json", manifest)

    internal = {
        "all_targets_completed": len(oof_frames) == 4,
        "leakage_all_pass": all(v["all_pass"] for tgt in leakage.values() for v in tgt.values()),
        "determinism_all_pass": all(v["pass"] for v in determinism.values()) if determinism else None,
        "full_models_unchanged": integrity["unchanged_during_run"],
    }
    write_json(audit_dir / "crossfit_internal_checks.json", internal)
    print(f"[DONE] Cross-fit training finished in {manifest['runtime_seconds']} s. "
          f"Run verify_crossfit.py for the final gate.")
    return 0


def main() -> None:
    ap = argparse.ArgumentParser(description="Out-of-fold LSTM forecasts for CADR DDQN training days.")
    ap.add_argument("--config", default="config_seed_0.yaml")
    ap.add_argument("--crossfit-config", default="crossfit_config.yaml")
    ap.add_argument("--check-only", action="store_true", help="Static/structural checks only; no training.")
    ap.add_argument("--overwrite", action="store_true", help="Replace an existing crossfit_results folder.")
    args = ap.parse_args()
    try:
        sys.exit(run(args))
    except CrossfitError as exc:
        print(f"\n[STOP] {exc}", file=sys.stderr)
        sys.exit(2)


if __name__ == "__main__":
    main()
