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

ALLOW_HOLIDAY_FALLBACK = False
LAST_HOLIDAY_BACKEND = None
from utils.load_price import load_price


FORECAST_CALENDAR_FEATURES = [
    "month",
    "day_of_week",
    "hour",
    "is_holiday",
    "is_weekend",
]
FORECAST_LAGS = [1, 2, 3, 24, 25, 26, 48, 49, 50]
FORECAST_INPUT_SIZE = 14


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


def validate_locked_forecast_config(cfg: dict) -> None:
    f = _forecast_cfg(cfg)
    if not bool(f.get("specification_locked", True)):
        return
    checks = {
        "calendar_features": (list(f["calendar_features"]), FORECAST_CALENDAR_FEATURES),
        "historical_lags": ([int(v) for v in f["historical_lags"]], FORECAST_LAGS),
        "window_size": (int(f["window_size"]), 24),
        "horizon": (int(f["horizon"]), 1),
        "hidden_size": (int(f["hidden_size"]), 64),
        "num_layers": (int(f["num_layers"]), 2),
        "dropout": (float(f["dropout"]), 0.2),
        "optimizer": (str(f["optimizer"]).lower(), "adam"),
        "loss": (str(f["loss"]).lower(), "mse"),
        "price_model_unit": (str(f["price_model_unit"]), "USD_per_MWh"),
        "ddqn_price_unit": (str(f["ddqn_price_unit"]), "cent_per_kWh"),
    }
    bad = [f"{k}: got {a!r}, expected {e!r}" for k, (a, e) in checks.items() if a != e]
    if bad:
        raise ValueError(
            "Forecast config does not match the locked forecasting specification:\n- "
            + "\n- ".join(bad)
        )


def _holiday_calendar(year: int, country: str):
    if country.upper() not in {"US", "USA"}:
        raise ValueError(f"Unsupported holiday country: {country}")
    try:
        import holidays
        return {pd.Timestamp(d).date() for d in holidays.US(years=[year]).keys()}, "holidays.US"
    except ImportError:
        if not ALLOW_HOLIDAY_FALLBACK:
            raise RuntimeError(
                "The 'holidays' package is required for full LSTM training so the "
                "forecaster uses the same US holiday calendar as the DDQN environment. "
                "Install the project requirements and rerun."
            )
        warnings.warn(
            "Python package 'holidays' is unavailable; using pandas USFederalHolidayCalendar "
            "for static check-only mode.",
            RuntimeWarning,
        )
        from pandas.tseries.holiday import USFederalHolidayCalendar
        cal = USFederalHolidayCalendar()
        dates = cal.holidays(
            start=pd.Timestamp(f"{year}-01-01"),
            end=pd.Timestamp(f"{year}-12-31"),
        )
        return {ts.date() for ts in dates}, "pandas_fallback_check_only"


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

def build_feature_frame(series_usd_mwh: pd.Series, cfg: dict, year: int) -> pd.DataFrame:
    f = _forecast_cfg(cfg)
    global LAST_HOLIDAY_BACKEND
    calendar, holiday_backend = _holiday_calendar(year, f.get("holiday_country", "US"))
    LAST_HOLIDAY_BACKEND = holiday_backend
    lags = [int(v) for v in f["historical_lags"]]
    s = series_usd_mwh.sort_index().astype(float)
    _assert_gap_free_hourly(s, "wholesale price")

    out = pd.DataFrame(index=s.index)
    out["month"] = out.index.month.astype(float)
    out["day_of_week"] = (out.index.dayofweek + 1).astype(float)
    out["hour"] = out.index.hour.astype(float)
    out["is_holiday"] = [1.0 if ts.date() in calendar else 0.0 for ts in out.index]
    out["is_weekend"] = (out.index.dayofweek >= 5).astype(float)
    for lag in lags:
        out[f"lag_{lag}"] = s.shift(lag)

    feature_names = FORECAST_CALENDAR_FEATURES + [f"lag_{lag}" for lag in lags]
    if list(out.columns) != feature_names:
        raise AssertionError(f"Feature order mismatch: {list(out.columns)}")
    if str(f.get("lag_padding", "zero")).lower() != "zero":
        raise ValueError("Only zero lag padding is supported in the locked forecasting pipeline.")
    return out.fillna(0.0)


def build_lstm_samples(series_usd_mwh: pd.Series, cfg: dict, year: int):
    f = _forecast_cfg(cfg)
    window = int(f["window_size"])
    if int(f["horizon"]) != 1:
        raise ValueError("This forecasting configuration requires horizon=1.")
    feature_frame = build_feature_frame(series_usd_mwh, cfg, year)
    y_all = series_usd_mwh.reindex(feature_frame.index).astype(float)
    features = feature_frame.to_numpy(dtype=np.float64)
    targets = y_all.to_numpy(dtype=np.float64)

    X, y, ts, doy, padding_free = [], [], [], [], []
    for i, timestamp in enumerate(feature_frame.index):
        start = i - window + 1
        if start >= 0:
            seq = features[start : i + 1]
        else:
            seq = np.vstack([
                np.zeros((-start, FORECAST_INPUT_SIZE), dtype=np.float64),
                features[0 : i + 1],
            ])
        if seq.shape != (window, FORECAST_INPUT_SIZE):
            raise AssertionError(f"Unexpected sample shape: {seq.shape}")
        X.append(seq)
        y.append([targets[i]])
        t = pd.Timestamp(timestamp)
        ts.append(t)
        doy.append(t.timetuple().tm_yday)
        padding_free.append(i >= (window - 1 + max(FORECAST_LAGS)))
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
    return {
        "train": _in_ranges(days, cfg["training"]["train_ranges"]),
        "validation": _in_ranges(days, [cfg["training"]["val_range"]]),
        "test": _in_ranges(days, _forecast_cfg(cfg)["test_ranges"]),
    }


def fit_scaler(X_train, y_train):
    if len(X_train) == 0:
        raise ValueError("No price training samples selected.")
    X_mean = X_train.mean(axis=(0, 1), keepdims=True)
    X_std = X_train.std(axis=(0, 1), keepdims=True)
    X_std[X_std == 0.0] = 1.0
    y_mean = float(y_train.mean())
    y_std = float(y_train.std())
    if y_std == 0.0:
        y_std = 1.0
    return {"X_mean": X_mean, "X_std": X_std, "y_mean": y_mean, "y_std": y_std}


def apply_scaler(X, y, scaler):
    return (
        ((X - scaler["X_mean"]) / scaler["X_std"]).astype(np.float32),
        ((y - scaler["y_mean"]) / scaler["y_std"]).astype(np.float32),
    )


def train_model(model, train_loader, val_loader, fcfg: dict, device):
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(fcfg["learning_rate"]),
        weight_decay=float(fcfg["weight_decay"]),
    )
    best_val = math.inf
    best_state = None
    stale = 0
    history = []
    for epoch in range(1, int(fcfg["epochs"]) + 1):
        model.train()
        total = 0.0
        count = 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
            total += float(loss.item()) * xb.shape[0]
            count += xb.shape[0]
        train_loss = total / max(count, 1)

        model.eval()
        total = 0.0
        count = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                loss = criterion(model(xb), yb)
                total += float(loss.item()) * xb.shape[0]
                count += xb.shape[0]
        val_loss = total / max(count, 1)
        history.append((epoch, train_loss, val_loss))
        # Epoch-level logging is intentionally suppressed for clean reproducible runs.
        if val_loss < best_val:
            best_val = val_loss
            best_state = deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1
            if stale >= int(fcfg["patience"]):
                break
    if best_state is None:
        raise RuntimeError("Training did not produce a valid model state.")
    model.load_state_dict(best_state)
    return model, pd.DataFrame(history, columns=["epoch", "train_loss", "validation_loss"])


def predict(model, Xn, yn, batch_size, device):
    loader = DataLoader(SequenceDataset(Xn, yn), batch_size=batch_size, shuffle=False)
    chunks = []
    model.eval()
    with torch.no_grad():
        for xb, _ in loader:
            chunks.append(model(xb.to(device)).cpu().numpy())
    return np.vstack(chunks) if chunks else np.empty((0, 1), dtype=np.float32)


def metrics(y_true, y_pred, zero_tol=1e-9):
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
    return {"MAE_USD_per_MWh": mae, "RMSE_USD_per_MWh": rmse, "MAPE_percent": mape, "WAPE_percent": wape, "R2": r2}


def persistence_predictions(series: pd.Series, timestamps):
    return np.asarray([series.get(pd.Timestamp(t) - pd.Timedelta(hours=1), np.nan) for t in timestamps], dtype=float)


def main() -> None:
    parser = argparse.ArgumentParser(description="Wholesale-price LSTM forecaster.")
    parser.add_argument("--config", default="config_seed_0.yaml")
    parser.add_argument("--check-only", action="store_true", help="Validate structure and sample construction without training.")
    args = parser.parse_args()

    global ALLOW_HOLIDAY_FALLBACK
    ALLOW_HOLIDAY_FALLBACK = bool(args.check_only)

    config_path = Path(args.config).resolve()
    cfg = load_config(str(config_path))
    validate_locked_forecast_config(cfg)
    fcfg = _forecast_cfg(cfg)
    seed = int(fcfg.get("seed", 42))
    set_seed(seed)

    data_path = _resolve_from_config(config_path, cfg["general"]["price_file"])
    internal = load_price(str(data_path))
    # load_price() converts raw $/MWh to cent/kWh by dividing by 10.
    # The forecasting pipeline is trained/evaluated in $/MWh and only
    # the final forecast is converted before entering DDQN, so invert that
    # loader conversion here for forecasting.
    series_usd_mwh = internal["price"].astype(float) * float(fcfg["cent_kwh_to_usd_mwh_multiplier"])
    year = int(cfg["general"].get("year", 2018))
    series_usd_mwh = series_usd_mwh[series_usd_mwh.index.year == year]

    X, y, meta, feature_names = build_lstm_samples(series_usd_mwh, cfg, year)
    masks = split_masks(meta, cfg)
    if X.shape[1:] != (24, 14):
        raise AssertionError("Price LSTM does not produce 24x14 input samples.")
    assert_no_target_or_future_leakage(series_usd_mwh, cfg, year)

    if args.check_only:
        return

    padding_free = np.asarray(meta["padding_free"], dtype=bool)
    fit_train_mask = masks["train"] & padding_free
    dropped = int(masks["train"].sum() - fit_train_mask.sum())
    scaler = fit_scaler(X[fit_train_mask], y[fit_train_mask])
    Xn, yn = apply_scaler(X, y, scaler)
    batch_size = int(fcfg["batch_size"])
    train_loader = DataLoader(
        SequenceDataset(Xn[fit_train_mask], yn[fit_train_mask]),
        batch_size=batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
    )
    val_loader = DataLoader(
        SequenceDataset(Xn[masks["validation"]], yn[masks["validation"]]),
        batch_size=batch_size,
        shuffle=False,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LSTMForecaster(
        input_size=FORECAST_INPUT_SIZE,
        hidden_size=int(fcfg["hidden_size"]),
        num_layers=int(fcfg["num_layers"]),
        dropout=float(fcfg["dropout"]),
    ).to(device)
    model, history = train_model(model, train_loader, val_loader, fcfg, device)

    output_dir = _resolve_from_config(config_path, fcfg["output_dir"]) / "price"
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "input_size": FORECAST_INPUT_SIZE,
            "hidden_size": int(fcfg["hidden_size"]),
            "num_layers": int(fcfg["num_layers"]),
            "dropout": float(fcfg["dropout"]),
            "window_size": int(fcfg["window_size"]),
            "horizon": int(fcfg["horizon"]),
            "feature_names": feature_names,
            "training_unit": "USD_per_MWh",
            "ddqn_unit": "cent_per_kWh",
        },
        output_dir / "lstm_price.pt",
    )
    np.savez(
        output_dir / "lstm_price_scaler.npz",
        X_mean=scaler["X_mean"],
        X_std=scaler["X_std"],
        y_mean=np.asarray([scaler["y_mean"]]),
        y_std=np.asarray([scaler["y_std"]]),
        feature_names=np.asarray(feature_names, dtype=str),
    )
    history.to_csv(output_dir / "lstm_price_history.csv", index=False)
    best_epoch = int(history.loc[history["validation_loss"].idxmin(), "epoch"])
    print(f"  Wholesale price complete | best epoch {best_epoch}")

    pred_n = predict(model, Xn, yn, batch_size, device)
    pred_usd_mwh = (pred_n * scaler["y_std"] + scaler["y_mean"]).reshape(-1)
    true_usd_mwh = y.reshape(-1)
    divisor = float(fcfg["usd_mwh_to_cent_kwh_divisor"])
    timestamps = pd.to_datetime(meta["timestamp"])
    split_label = np.full(len(y), "other", dtype=object)
    for name, mask in masks.items():
        split_label[mask] = name

    results = pd.DataFrame(
        {
            "timestamp": timestamps,
            "doy": meta["doy"],
            "split": split_label,
            "padding_free": padding_free,
            "actual_price_USD_per_MWh": true_usd_mwh,
            "forecast_price_USD_per_MWh": pred_usd_mwh,
            "actual_price_cent_per_kWh": true_usd_mwh / divisor,
            "forecast_price_cent_per_kWh": pred_usd_mwh / divisor,
        }
    )
    results.to_csv(output_dir / "lstm_price_predictions_all.csv", index=False)

    rows = []
    zero_tol = float(fcfg.get("mape_zero_tolerance", 1e-9))
    for split_name, mask in masks.items():
        metric_mask = mask & padding_free
        if not metric_mask.any():
            continue
        row = {
            "split": split_name,
            "n_samples": int(metric_mask.sum()),
            **metrics(true_usd_mwh[metric_mask], pred_usd_mwh[metric_mask], zero_tol),
        }
        p = persistence_predictions(series_usd_mwh, timestamps[metric_mask])
        valid = np.isfinite(p)
        if valid.any():
            pm = metrics(true_usd_mwh[metric_mask][valid], p[valid], zero_tol)
            row.update({f"persistence_{k}": v for k, v in pm.items()})
        rows.append(row)
    metrics_df = pd.DataFrame(rows)
    metrics_path = output_dir / "lstm_price_metrics.csv"
    metrics_df.to_csv(metrics_path, index=False)

    gap_threshold = float(fcfg.get("train_validation_wape_relative_gap_threshold", 0.20))
    m = metrics_df.set_index("split")
    train_wape = float(m.loc["train", "WAPE_percent"])
    val_wape = float(m.loc["validation", "WAPE_percent"])
    relative_gap = (val_wape - train_wape) / max(abs(val_wape), 1e-12)
    needs_crossfit = bool(relative_gap > gap_threshold)
    pd.DataFrame([{
        "train_WAPE_percent": train_wape,
        "validation_WAPE_percent": val_wape,
        "relative_gap": relative_gap,
        "threshold": gap_threshold,
        "requires_blocked_crossfit": needs_crossfit,
    }]).to_csv(output_dir / "lstm_price_train_validation_gap_audit.csv", index=False)
    if needs_crossfit:
        print("[GATE] Price training-day forecasts require blocked cross-fitting before DDQN integration.")
    else:
        print("[PASS] Price train-vs-validation WAPE gap is within the pre-specified threshold.")

    manifest = {
        "configuration_status": "locked",
        "seed": seed,
        "holiday_backend": LAST_HOLIDAY_BACKEND,
        "feature_names": feature_names,
        "window_size": 24,
        "horizon": 1,
        "training_unit": "USD_per_MWh",
        "ddqn_unit": "cent_per_kWh",
        "conversion": "divide forecast in USD/MWh by 10 before DDQN",
        "output_predictions": "lstm_price_predictions_all.csv",
        "padding_policy": "forecast all hours; exclude padded samples from fitting, scaling, and metrics",
        "train_validation_wape_relative_gap_threshold": gap_threshold,
        "blocked_crossfit_required": needs_crossfit,
        "sha256": {
            "script": _sha256(Path(__file__).resolve()),
            "config": _sha256(config_path),
            "data": _sha256(data_path),
            "predictions": _sha256(output_dir / "lstm_price_predictions_all.csv"),
            "metrics": _sha256(metrics_path),
        },
    }
    (output_dir / "lstm_price_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
