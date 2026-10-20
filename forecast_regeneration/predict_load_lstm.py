from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import warnings
from copy import deepcopy
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from utils.config_loader import load_config
from utils.load_demand import load_demand


ARTICLE_CALENDAR_FEATURES = [
    "month",
    "day_of_week",
    "hour",
    "is_holiday",
    "is_weekend",
]
ARTICLE_LAGS = [1, 2, 3, 24, 25, 26, 48, 49, 50]
ARTICLE_INPUT_SIZE = 14


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


class SequenceDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.as_tensor(X, dtype=torch.float32)
        self.y = torch.as_tensor(y, dtype=torch.float32)

    def __len__(self) -> int:
        return int(self.X.shape[0])

    def __getitem__(self, idx: int):
        return self.X[idx], self.y[idx]


class LSTMForecaster(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_layers: int, dropout: float):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])


def _resolve_from_config(config_path: Path, configured_path: str) -> Path:
    p = Path(configured_path)
    if p.is_absolute():
        return p
    return (config_path.parent / p).resolve()


def _forecast_cfg(cfg: dict) -> dict:
    if "forecast" not in cfg:
        raise KeyError("Missing top-level 'forecast' section in config.")
    return cfg["forecast"]


def validate_article_locked_config(cfg: dict) -> None:
    f = _forecast_cfg(cfg)
    if not bool(f.get("article_locked", True)):
        return

    checks = {
        "calendar_features": (list(f["calendar_features"]), ARTICLE_CALENDAR_FEATURES),
        "historical_lags": ([int(v) for v in f["historical_lags"]], ARTICLE_LAGS),
        "window_size": (int(f["window_size"]), 24),
        "horizon": (int(f["horizon"]), 1),
        "hidden_size": (int(f["hidden_size"]), 64),
        "num_layers": (int(f["num_layers"]), 2),
        "dropout": (float(f["dropout"]), 0.2),
        "optimizer": (str(f["optimizer"]).lower(), "adam"),
        "loss": (str(f["loss"]).lower(), "mse"),
    }
    bad = []
    for name, (actual, expected) in checks.items():
        if actual != expected:
            bad.append(f"{name}: got {actual!r}, expected {expected!r}")
    if bad:
        raise ValueError(
            "Forecast config is not aligned with Appendix A of the manuscript:\n- "
            + "\n- ".join(bad)
        )
    if len(ARTICLE_CALENDAR_FEATURES) + len(ARTICLE_LAGS) != ARTICLE_INPUT_SIZE:
        raise AssertionError("Internal article input-size definition is inconsistent.")


def _holiday_calendar(year: int, country: str):
    if country.upper() not in {"US", "USA"}:
        raise ValueError(f"Unsupported holiday country: {country}")
    try:
        import holidays
        return {pd.Timestamp(d).date() for d in holidays.US(years=[year]).keys()}, "holidays.US"
    except ImportError:
        warnings.warn(
            "Python package 'holidays' is unavailable; using pandas USFederalHolidayCalendar "
            "for this run. Install the project requirements before final training so the "
            "forecaster and DDQN environment use the same holiday calendar.",
            RuntimeWarning,
        )
        from pandas.tseries.holiday import USFederalHolidayCalendar
        cal = USFederalHolidayCalendar()
        dates = cal.holidays(
            start=pd.Timestamp(f"{year}-01-01"),
            end=pd.Timestamp(f"{year}-12-31"),
        )
        return {ts.date() for ts in dates}, "pandas_fallback"


def _assert_gap_free_hourly(series: pd.Series, label: str) -> None:
    s = series.sort_index()
    if s.index.has_duplicates:
        raise ValueError(f"Duplicate timestamps found in {label} series.")
    expected = pd.date_range(s.index.min(), s.index.max(), freq="h")
    if len(expected) != len(s.index) or not expected.equals(pd.DatetimeIndex(s.index)):
        missing = expected.difference(pd.DatetimeIndex(s.index))
        extra = pd.DatetimeIndex(s.index).difference(expected)
        raise ValueError(
            f"{label} series is not gap-free hourly: missing={len(missing)}, extra={len(extra)}"
        )


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def build_feature_frame(series: pd.Series, cfg: dict, year: int) -> pd.DataFrame:
    """Build the exact 14 inputs claimed in Appendix A for every target hour.

    For target hour h, the calendar variables refer to h and the historical
    variables are observations at h-1, h-2, h-3, h-24, h-25, h-26, h-48,
    h-49 and h-50.  The target value y_h itself is never present in X.
    """
    f = _forecast_cfg(cfg)
    lags = [int(v) for v in f["historical_lags"]]
    calendar, holiday_backend = _holiday_calendar(year, f.get("holiday_country", "US"))

    s = series.sort_index().astype(float)
    _assert_gap_free_hourly(s, "household load")

    out = pd.DataFrame(index=s.index)
    out["month"] = out.index.month.astype(float)
    out["day_of_week"] = (out.index.dayofweek + 1).astype(float)
    out["hour"] = out.index.hour.astype(float)
    out["is_holiday"] = [1.0 if ts.date() in calendar else 0.0 for ts in out.index]
    out["is_weekend"] = (out.index.dayofweek >= 5).astype(float)

    for lag in lags:
        out[f"lag_{lag}"] = s.shift(lag)

    feature_names = ARTICLE_CALENDAR_FEATURES + [f"lag_{lag}" for lag in lags]
    if feature_names != list(out.columns):
        raise AssertionError(f"Feature order mismatch: {list(out.columns)}")

    padding = str(f.get("lag_padding", "zero")).lower()
    if padding == "zero":
        out = out.fillna(0.0)
    elif padding == "ffill_then_zero":
        out = out.ffill().fillna(0.0)
    else:
        raise ValueError(f"Unsupported forecast.lag_padding: {padding}")
    return out


def build_lstm_samples(series: pd.Series, cfg: dict, year: int):
    """Build 24-step sequences of the exact 14-D Appendix-A feature vector.

    Each sequence ends at its target hour h.  The row for h contains only
    known calendar variables for h plus lagged observations through h-1, so
    the construction remains one-step-ahead and contains no y_h leakage.
    """
    f = _forecast_cfg(cfg)
    window = int(f["window_size"])
    horizon = int(f["horizon"])
    if horizon != 1:
        raise ValueError("This manuscript-aligned implementation requires horizon=1.")

    feature_frame = build_feature_frame(series, cfg, year)
    y_all = series.reindex(feature_frame.index).astype(float)
    features = feature_frame.to_numpy(dtype=np.float64)
    targets = y_all.to_numpy(dtype=np.float64)
    timestamps = feature_frame.index.to_numpy()

    sequence_padding = str(f.get("sequence_padding", "zero")).lower()
    if sequence_padding != "zero":
        raise ValueError("Only zero sequence padding is supported in the article-locked pipeline.")

    X, y, ts, doy, padding_free = [], [], [], [], []
    for i in range(len(feature_frame)):
        start = i - window + 1
        if start >= 0:
            seq = features[start : i + 1]
        else:
            pad = np.zeros((-start, features.shape[1]), dtype=np.float64)
            seq = np.vstack([pad, features[0 : i + 1]])
        if seq.shape != (window, ARTICLE_INPUT_SIZE):
            raise AssertionError(f"Unexpected sample shape {seq.shape} at index {i}.")
        X.append(seq)
        y.append([targets[i]])
        t = pd.Timestamp(timestamps[i])
        ts.append(t)
        doy.append(t.timetuple().tm_yday)
        padding_free.append(i >= (window - 1 + max(ARTICLE_LAGS)))

    return (
        np.asarray(X, dtype=np.float64),
        np.asarray(y, dtype=np.float64),
        {
            "timestamp": np.asarray(ts, dtype=object),
            "doy": np.asarray(doy, dtype=int),
            "padding_free": np.asarray(padding_free, dtype=bool),
        },
        list(feature_frame.columns),
    )


def assert_no_target_or_future_leakage(series: pd.Series, cfg: dict, year: int) -> None:
    """Verify that X_h is unchanged when y_h or a future value is perturbed."""
    base_X, _, meta, _ = build_lstm_samples(series, cfg, year)
    # Pick a timestamp well inside the series so all article lags exist.
    idx = max(100, len(series) // 2)
    idx = min(idx, len(series) - 3)
    target_ts = pd.Timestamp(series.index[idx])
    sample_idx = int(np.where(pd.to_datetime(meta["timestamp"]) == target_ts)[0][0])

    changed = series.copy()
    changed.loc[target_ts] = float(changed.loc[target_ts]) + 9999.0
    future_ts = pd.Timestamp(series.index[idx + 1])
    changed.loc[future_ts] = float(changed.loc[future_ts]) - 9999.0
    changed_X, _, changed_meta, _ = build_lstm_samples(changed, cfg, year)
    changed_sample_idx = int(np.where(pd.to_datetime(changed_meta["timestamp"]) == target_ts)[0][0])
    if not np.array_equal(base_X[sample_idx], changed_X[changed_sample_idx]):
        raise AssertionError(
            "Leakage check failed: the input for target hour h changed when y_h/future values were perturbed."
        )


def _in_ranges(days: np.ndarray, ranges) -> np.ndarray:
    mask = np.zeros(len(days), dtype=bool)
    for start, end in ranges:
        mask |= (days >= int(start)) & (days <= int(end))
    return mask


def split_masks(meta: dict, cfg: dict):
    days = np.asarray(meta["doy"], dtype=int)
    train_ranges = cfg["training"]["train_ranges"]
    val_range = cfg["training"]["val_range"]
    test_ranges = _forecast_cfg(cfg)["test_ranges"]
    return {
        "train": _in_ranges(days, train_ranges),
        "validation": _in_ranges(days, [val_range]),
        "test": _in_ranges(days, test_ranges),
    }


def fit_scaler(X_train: np.ndarray, y_train: np.ndarray) -> dict:
    if len(X_train) == 0:
        raise ValueError("No training samples were selected for the load forecaster.")
    X_mean = X_train.mean(axis=(0, 1), keepdims=True)
    X_std = X_train.std(axis=(0, 1), keepdims=True)
    X_std[X_std == 0.0] = 1.0
    y_mean = float(y_train.mean())
    y_std = float(y_train.std())
    if y_std == 0.0:
        y_std = 1.0
    return {"X_mean": X_mean, "X_std": X_std, "y_mean": y_mean, "y_std": y_std}


def apply_scaler(X: np.ndarray, y: np.ndarray, scaler: dict):
    Xn = (X - scaler["X_mean"]) / scaler["X_std"]
    yn = (y - scaler["y_mean"]) / scaler["y_std"]
    return Xn.astype(np.float32), yn.astype(np.float32)


def train_model(model, train_loader, val_loader, fcfg: dict, device):
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(fcfg["learning_rate"]),
        weight_decay=float(fcfg["weight_decay"]),
    )
    max_epochs = int(fcfg["epochs"])
    patience = int(fcfg["patience"])
    log_every = int(fcfg["log_every"])

    best_val = math.inf
    best_state = None
    stale = 0
    history = []
    for epoch in range(1, max_epochs + 1):
        model.train()
        train_sum = 0.0
        train_n = 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()
            train_sum += float(loss.item()) * xb.shape[0]
            train_n += xb.shape[0]
        train_loss = train_sum / max(train_n, 1)

        model.eval()
        val_sum = 0.0
        val_n = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                loss = criterion(model(xb), yb)
                val_sum += float(loss.item()) * xb.shape[0]
                val_n += xb.shape[0]
        val_loss = val_sum / max(val_n, 1)
        history.append((epoch, train_loss, val_loss))

        if epoch == 1 or epoch % log_every == 0:
            print(f"Epoch {epoch:4d} | train={train_loss:.6f} | val={val_loss:.6f}")

        if val_loss < best_val:
            best_val = val_loss
            best_state = deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                print(f"Early stopping at epoch {epoch}; best validation MSE={best_val:.6f}")
                break

    if best_state is None:
        raise RuntimeError("Training did not produce a valid model state.")
    model.load_state_dict(best_state)
    return model, pd.DataFrame(history, columns=["epoch", "train_loss", "validation_loss"])


def predict(model, Xn: np.ndarray, yn: np.ndarray, batch_size: int, device) -> np.ndarray:
    loader = DataLoader(SequenceDataset(Xn, yn), batch_size=batch_size, shuffle=False)
    model.eval()
    chunks = []
    with torch.no_grad():
        for xb, _ in loader:
            chunks.append(model(xb.to(device)).cpu().numpy())
    return np.vstack(chunks) if chunks else np.empty((0, 1), dtype=np.float32)


def metrics(y_true: np.ndarray, y_pred: np.ndarray, zero_tol: float = 1e-9) -> dict:
    yt = np.asarray(y_true, dtype=float).reshape(-1)
    yp = np.asarray(y_pred, dtype=float).reshape(-1)
    err = yt - yp
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err**2)))
    nz = np.abs(yt) > zero_tol
    mape = float(np.mean(np.abs(err[nz] / yt[nz])) * 100.0) if nz.any() else math.nan
    denom = float(np.sum(np.abs(yt)))
    wape = float(np.sum(np.abs(err)) / denom * 100.0) if denom > zero_tol else math.nan
    ss_tot = float(np.sum((yt - yt.mean()) ** 2))
    r2 = float(1.0 - np.sum(err**2) / ss_tot) if ss_tot > 0 else math.nan
    return {"MAE_kW": mae, "RMSE_kW": rmse, "MAPE_percent": mape, "WAPE_percent": wape, "R2": r2}


def persistence_predictions(series: pd.Series, timestamps) -> np.ndarray:
    return np.asarray([series.get(pd.Timestamp(t) - pd.Timedelta(hours=1), np.nan) for t in timestamps], dtype=float)


def main() -> None:
    parser = argparse.ArgumentParser(description="Manuscript-aligned household-load LSTM forecasters.")
    parser.add_argument("--config", default="config_seed_0.yaml")
    parser.add_argument("--check-only", action="store_true", help="Validate structure and sample construction without training.")
    args = parser.parse_args()

    config_path = Path(args.config).resolve()
    cfg = load_config(str(config_path))
    validate_article_locked_config(cfg)
    fcfg = _forecast_cfg(cfg)
    seed = int(fcfg.get("seed", 42))
    set_seed(seed)

    data_path = _resolve_from_config(config_path, cfg["general"]["load_file"])
    df = load_demand(str(data_path), house_ids=cfg["environment"]["house_ids"])
    year = int(cfg["general"].get("year", 2018))

    print("[PASS] Appendix-A locked configuration validated.")
    print(f"Feature order: {ARTICLE_CALENDAR_FEATURES + [f'lag_{x}' for x in ARTICLE_LAGS]}")
    print(f"Window/horizon: {fcfg['window_size']}/{fcfg['horizon']} | LSTM: {fcfg['num_layers']} x {fcfg['hidden_size']} | dropout={fcfg['dropout']}")

    samples = {}
    for hid in [int(x) for x in cfg["environment"]["house_ids"]]:
        sub = df[df["dataid"] == hid].sort_values("dt")
        series = sub.set_index("dt")["total"].astype(float)
        series = series[series.index.year == year]
        X, y, meta, feature_names = build_lstm_samples(series, cfg, year)
        masks = split_masks(meta, cfg)
        samples[hid] = (series, X, y, meta, masks, feature_names)
        print(
            f"House {hid}: X={X.shape}, train={int(masks['train'].sum())}, "
            f"val={int(masks['validation'].sum())}, test={int(masks['test'].sum())}"
        )
        if X.shape[1:] != (24, 14):
            raise AssertionError(f"House {hid} does not produce 24x14 LSTM input samples.")
        assert_no_target_or_future_leakage(series, cfg, year)

    print("[PASS] No-target/future-leakage construction test passed for all household models.")
    if args.check_only:
        print("[PASS] Static load-LSTM check complete. No training was run.")
        return

    output_dir = _resolve_from_config(config_path, fcfg["output_dir"]) / "load"
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Training device:", device)

    all_rows = []
    metric_rows = []
    batch_size = int(fcfg["batch_size"])
    zero_tol = float(fcfg.get("mape_zero_tolerance", 1e-9))

    for hid, (series, X, y, meta, masks, feature_names) in samples.items():
        set_seed(seed + hid)
        train_mask = masks["train"]
        val_mask = masks["validation"]
        padding_free = np.asarray(meta["padding_free"], dtype=bool)
        fit_train_mask = train_mask & padding_free
        dropped = int(train_mask.sum() - fit_train_mask.sum())
        if dropped:
            print(f"House {hid}: excluding {dropped} padded training samples from fitting/scaling.")
        scaler = fit_scaler(X[fit_train_mask], y[fit_train_mask])
        Xn, yn = apply_scaler(X, y, scaler)

        train_loader = DataLoader(
            SequenceDataset(Xn[fit_train_mask], yn[fit_train_mask]),
            batch_size=batch_size,
            shuffle=True,
            generator=torch.Generator().manual_seed(seed + hid),
        )
        val_loader = DataLoader(
            SequenceDataset(Xn[val_mask], yn[val_mask]),
            batch_size=batch_size,
            shuffle=False,
        )

        model = LSTMForecaster(
            input_size=ARTICLE_INPUT_SIZE,
            hidden_size=int(fcfg["hidden_size"]),
            num_layers=int(fcfg["num_layers"]),
            dropout=float(fcfg["dropout"]),
        ).to(device)
        model, history = train_model(model, train_loader, val_loader, fcfg, device)

        torch.save(
            {
                "state_dict": model.state_dict(),
                "house_id": hid,
                "input_size": ARTICLE_INPUT_SIZE,
                "hidden_size": int(fcfg["hidden_size"]),
                "num_layers": int(fcfg["num_layers"]),
                "dropout": float(fcfg["dropout"]),
                "window_size": int(fcfg["window_size"]),
                "horizon": int(fcfg["horizon"]),
                "feature_names": feature_names,
            },
            output_dir / f"lstm_load_house_{hid}.pt",
        )
        np.savez(
            output_dir / f"lstm_load_house_{hid}_scaler.npz",
            X_mean=scaler["X_mean"],
            X_std=scaler["X_std"],
            y_mean=np.asarray([scaler["y_mean"]]),
            y_std=np.asarray([scaler["y_std"]]),
            feature_names=np.asarray(feature_names, dtype=str),
        )
        history.to_csv(output_dir / f"lstm_load_house_{hid}_history.csv", index=False)

        pred_n = predict(model, Xn, yn, batch_size, device)
        pred = pred_n * scaler["y_std"] + scaler["y_mean"]
        pred = pred.reshape(-1)
        true = y.reshape(-1)
        timestamps = pd.to_datetime(meta["timestamp"])
        split_label = np.full(len(true), "other", dtype=object)
        for name, mask in masks.items():
            split_label[mask] = name

        frame = pd.DataFrame(
            {
                "timestamp": timestamps,
                "doy": meta["doy"],
                "house_id": hid,
                "split": split_label,
                "padding_free": padding_free,
                "actual_load_kw": true,
                "forecast_load_kw": pred,
            }
        )
        all_rows.append(frame)

        for split_name, mask in masks.items():
            metric_mask = mask & padding_free
            if not metric_mask.any():
                continue
            row = {
                "house_id": hid,
                "split": split_name,
                "n_samples": int(metric_mask.sum()),
                **metrics(true[metric_mask], pred[metric_mask], zero_tol),
            }
            p = persistence_predictions(series, timestamps[metric_mask])
            valid = np.isfinite(p)
            if valid.any():
                pm = metrics(true[metric_mask][valid], p[valid], zero_tol)
                row.update({f"persistence_{k}": v for k, v in pm.items()})
            metric_rows.append(row)

    predictions = pd.concat(all_rows, ignore_index=True).sort_values(["timestamp", "house_id"])
    predictions.to_csv(output_dir / "lstm_load_predictions_all.csv", index=False)
    metrics_df = pd.DataFrame(metric_rows)
    metrics_path = output_dir / "lstm_load_metrics.csv"
    metrics_df.to_csv(metrics_path, index=False)

    gap_threshold = float(fcfg.get("train_validation_wape_relative_gap_threshold", 0.20))
    gap_rows = []
    for hid in sorted(metrics_df["house_id"].unique()):
        m = metrics_df[metrics_df["house_id"] == hid].set_index("split")
        train_wape = float(m.loc["train", "WAPE_percent"])
        val_wape = float(m.loc["validation", "WAPE_percent"])
        relative_gap = (val_wape - train_wape) / max(abs(val_wape), 1e-12)
        gap_rows.append({
            "house_id": int(hid),
            "train_WAPE_percent": train_wape,
            "validation_WAPE_percent": val_wape,
            "relative_gap": relative_gap,
            "threshold": gap_threshold,
            "requires_blocked_crossfit": bool(relative_gap > gap_threshold),
        })
    gap_df = pd.DataFrame(gap_rows)
    gap_df.to_csv(output_dir / "lstm_load_train_validation_gap_audit.csv", index=False)
    if gap_df["requires_blocked_crossfit"].any():
        print("[GATE] Load training-day forecasts require blocked cross-fitting before DDQN integration.")
    else:
        print("[PASS] Load train-vs-validation WAPE gap is within the pre-specified threshold.")

    # Aggregate forecast used by CADR: sum of the three independently forecast households.
    pivot_pred = predictions.pivot(index="timestamp", columns="house_id", values="forecast_load_kw")
    pivot_true = predictions.pivot(index="timestamp", columns="house_id", values="actual_load_kw")
    agg = pd.DataFrame(index=pivot_pred.index)
    agg["actual_aggregate_load_kw"] = pivot_true.sum(axis=1)
    agg["forecast_aggregate_load_kw"] = pivot_pred.sum(axis=1)
    agg.reset_index().to_csv(output_dir / "lstm_aggregate_load_predictions.csv", index=False)

    start = pd.Timestamp(fcfg["report_plot_start"])
    end = pd.Timestamp(fcfg["report_plot_end_exclusive"])
    week = predictions[(predictions["timestamp"] >= start) & (predictions["timestamp"] < end)]
    for hid in sorted(week["house_id"].unique()):
        sub = week[week["house_id"] == hid].sort_values("timestamp")
        plt.figure(figsize=(14, 4.5))
        plt.plot(sub["timestamp"], sub["actual_load_kw"], label="Observed")
        plt.plot(sub["timestamp"], sub["forecast_load_kw"], linestyle="--", label="Forecast")
        plt.xlabel("Time")
        plt.ylabel("Load (kW)")
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_dir / f"lstm_load_house_{hid}_23_29_july.png", dpi=180)
        plt.close()

    manifest = {
        "article_alignment": "Appendix A locked",
        "seed": seed,
        "feature_names": ARTICLE_CALENDAR_FEATURES + [f"lag_{x}" for x in ARTICLE_LAGS],
        "window_size": 24,
        "horizon": 1,
        "independent_household_models": [int(x) for x in cfg["environment"]["house_ids"]],
        "aggregate_definition": "sum of household forecasts",
        "output_predictions": "lstm_load_predictions_all.csv",
        "padding_policy": "forecast all hours; exclude padded samples from fitting, scaling, and metrics",
        "train_validation_wape_relative_gap_threshold": gap_threshold,
        "blocked_crossfit_required": bool(gap_df["requires_blocked_crossfit"].any()),
        "sha256": {
            "script": _sha256(Path(__file__).resolve()),
            "config": _sha256(config_path),
            "data": _sha256(data_path),
            "predictions": _sha256(output_dir / "lstm_load_predictions_all.csv"),
            "metrics": _sha256(metrics_path),
        },
    }
    (output_dir / "lstm_load_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print("[PASS] Load forecasting completed. Outputs:", output_dir)


if __name__ == "__main__":
    main()
