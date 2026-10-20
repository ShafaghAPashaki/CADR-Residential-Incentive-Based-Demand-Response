from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml


def resolve(config_path: Path, p: str) -> Path:
    q = Path(p)
    return q if q.is_absolute() else (config_path.parent / q).resolve()


def mae(y_true, y_pred) -> float:
    yt = np.asarray(y_true, dtype=float)
    yp = np.asarray(y_pred, dtype=float)
    return float(np.mean(np.abs(yt - yp)))


def _assert_exact_week(
    frame: pd.DataFrame,
    start: pd.Timestamp,
    end: pd.Timestamp,
    label: str,
) -> pd.DataFrame:
    sub = frame[
        (frame["timestamp"] >= start) & (frame["timestamp"] < end)
    ].sort_values("timestamp").copy()

    if len(sub) != 168:
        raise AssertionError(f"{label}: expected 168 hourly samples, found {len(sub)}")
    if sub["timestamp"].iloc[0] != pd.Timestamp("2018-07-23 00:00:00"):
        raise AssertionError(f"{label}: incorrect first timestamp: {sub['timestamp'].iloc[0]}")
    if sub["timestamp"].iloc[-1] != pd.Timestamp("2018-07-29 23:00:00"):
        raise AssertionError(f"{label}: incorrect last timestamp: {sub['timestamp'].iloc[-1]}")

    diffs = sub["timestamp"].diff().dropna()
    if not (diffs == pd.Timedelta(hours=1)).all():
        raise AssertionError(f"{label}: plot window is not gap-free hourly data")

    return sub


def _style_panel(ax, xlabel: str, ylabel: str, title: str) -> None:
    ax.set_xlabel(xlabel, fontsize=15, fontweight="bold")
    ax.set_ylabel(ylabel, fontsize=15, fontweight="bold")
    ax.set_title(title, fontsize=15, fontweight="bold")
    ax.tick_params(axis="both", labelsize=12.5, width=1.5, length=5)

    for tick in ax.get_xticklabels() + ax.get_yticklabels():
        tick.set_fontweight("bold")

    legend = ax.legend(fontsize=11.5, loc="upper right", frameon=True)
    for text in legend.get_texts():
        text.set_fontweight("bold")

    for spine in ax.spines.values():
        spine.set_linewidth(1.3)


def _write_table_outputs(table: pd.DataFrame, out: Path) -> None:
    csv_path = out / "forecast_metrics.csv"
    tex_path = out / "forecast_metrics.tex"

    try:
        table.to_csv(csv_path, index=False)
    except PermissionError as exc:
        raise PermissionError(
            f"Cannot overwrite {csv_path}. Close it in Excel or any other program and rerun."
        ) from exc

    linebreak = r"\\"
    tex = [
        r"\begin{table}[H]",
        r"\centering",
        r"\caption{Accuracy of the LSTM forecasting models evaluated on the held-out July--August 2018 test period.}",
        r"\label{tab:forecast_accuracy}",
        r"\small",
        r"\begin{tabular}{lcc}",
        r"\hline",
        rf"\textbf{{Forecasting target}} & \textbf{{Unit}} & \textbf{{MAE}} {linebreak}",
        r"\hline",
    ]

    for _, row in table.iterrows():
        tex.append(
            f"{row['Forecasting target']} & {row['Unit']} & {row['MAE']:.2f} {linebreak}"
        )

    tex += [
        r"\hline",
        r"\end{tabular}",
        r"\vspace{2pt}",
        r"\begin{minipage}{0.92\columnwidth}",
        r"\footnotesize",
        (
            r"Errors are evaluated exclusively on the held-out July--August 2018 test period (1,488 h). "
            r"Wholesale-price errors are reported in cent/kWh and household-load errors in kW."
        ),
        r"\end{minipage}",
        r"\end{table}",
    ]

    try:
        tex_path.write_text("\n".join(tex), encoding="utf-8")
    except PermissionError as exc:
        raise PermissionError(
            f"Cannot overwrite {tex_path}. Close it in any editor and rerun."
        ) from exc


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config_seed_0.yaml")
    ap.add_argument("--check-output-access", action="store_true")
    args = ap.parse_args()

    cp = Path(args.config).resolve()
    cfg = yaml.safe_load(cp.read_text(encoding="utf-8"))
    f = cfg["forecast"]

    root = resolve(cp, f["output_dir"])
    ld = root / "load"
    pdx = root / "price"
    out = root
    out.mkdir(parents=True, exist_ok=True)

    if args.check_output_access:
        # Catch the exact Windows/Excel locking problem before a long training run.
        for target in [
            out / "forecast_metrics.csv",
            out / "forecast_metrics.tex",
            out / "forecast_comparison.png",
            out / "forecast_comparison.pdf",
        ]:
            if target.exists():
                try:
                    with target.open("ab"):
                        pass
                except PermissionError as exc:
                    raise PermissionError(
                        f"Output file is locked: {target}. Close it before training."
                    ) from exc
        return

    lp = pd.read_csv(ld / "lstm_load_predictions_all.csv", parse_dates=["timestamp"])
    pp = pd.read_csv(pdx / "lstm_price_predictions_all.csv", parse_dates=["timestamp"])

    ids = [int(x) for x in cfg["environment"]["house_ids"]]
    split = f.get("metrics_split", "test")

    # ------------------------------------------------------------------
    # forecast metrics table: MAE only, held-out July-August test period.
    # Price is final in cent/kWh.
    # ------------------------------------------------------------------
    rows = []
    ptest = pp[(pp["split"] == split) & (pp["padding_free"].astype(bool))]
    rows.append(
        {
            "Forecasting target": "Wholesale electricity price",
            "Unit": "cent/kWh",
            "MAE": mae(
                ptest["actual_price_cent_per_kWh"],
                ptest["forecast_price_cent_per_kWh"],
            ),
        }
    )

    for i, hid in enumerate(ids, start=1):
        s = lp[
            (lp["house_id"] == hid)
            & (lp["split"] == split)
            & (lp["padding_free"].astype(bool))
        ]
        rows.append(
            {
                "Forecasting target": f"EU {i} load",
                "Unit": "kW",
                "MAE": mae(s["actual_load_kw"], s["forecast_load_kw"]),
            }
        )

    table = pd.DataFrame(rows)
    _write_table_outputs(table, out)

    # ------------------------------------------------------------------
    # combined forecast figure: exactly 23-29 July 2018 (168 hourly observations).
    # ------------------------------------------------------------------
    start = pd.Timestamp(f["plot_start"])
    end = pd.Timestamp(f["plot_end_exclusive"])

    if (
        start != pd.Timestamp("2018-07-23 00:00:00")
        or end != pd.Timestamp("2018-07-30 00:00:00")
    ):
        raise AssertionError(
            f"Plot interval must be 23-29 July 2018; got [{start}, {end})."
        )

    load_weeks = {
        hid: _assert_exact_week(
            lp[lp["house_id"] == hid], start, end, f"household {hid}"
        )
        for hid in ids
    }
    price_week = _assert_exact_week(pp, start, end, "wholesale price")

    # Standalone project-style figures. These deliberately overwrite the
    # simple diagnostic figures created by the training scripts so that the
    # final files left in results/forecast/ are output-ready.
    for hid in ids:
        s = load_weeks[hid]
        hours = np.arange(1, len(s) + 1)
        fig, ax = plt.subplots(figsize=(14, 4.8))
        ax.plot(hours, s["actual_load_kw"], color="red", linewidth=2.6, label="Real")
        ax.plot(hours, s["forecast_load_kw"], color="blue", linestyle="--", linewidth=2.6, label="Forecast")
        ax.set_xlabel("Time (h)", fontsize=24, fontweight="bold")
        ax.set_ylabel("Load (kW)", fontsize=24, fontweight="bold")
        ax.tick_params(axis="both", labelsize=20, width=1.8, length=6)
        for tick in ax.get_xticklabels() + ax.get_yticklabels():
            tick.set_fontweight("bold")
        legend = ax.legend(fontsize=20, loc="upper right", frameon=True)
        for text in legend.get_texts():
            text.set_fontweight("bold")
        for spine in ax.spines.values():
            spine.set_linewidth(1.5)
        fig.tight_layout()
        fig.savefig(ld / f"lstm_load_house_{hid}_23_29_july.png", dpi=300, bbox_inches="tight")
        fig.savefig(ld / f"lstm_load_house_{hid}_23_29_july.pdf", bbox_inches="tight")
        plt.close(fig)

    hours = np.arange(1, len(price_week) + 1)
    fig, ax = plt.subplots(figsize=(14, 4.8))
    ax.plot(hours, price_week["actual_price_cent_per_kWh"], color="red", linewidth=2.6, label="Real")
    ax.plot(hours, price_week["forecast_price_cent_per_kWh"], color="blue", linestyle="--", linewidth=2.6, label="Forecast")
    ax.set_xlabel("Time (h)", fontsize=24, fontweight="bold")
    ax.set_ylabel("Price (cent/kWh)", fontsize=24, fontweight="bold")
    ax.tick_params(axis="both", labelsize=20, width=1.8, length=6)
    for tick in ax.get_xticklabels() + ax.get_yticklabels():
        tick.set_fontweight("bold")
    legend = ax.legend(fontsize=20, loc="upper right", frameon=True)
    for text in legend.get_texts():
        text.set_fontweight("bold")
    for spine in ax.spines.values():
        spine.set_linewidth(1.5)
    fig.tight_layout()
    fig.savefig(pdx / "lstm_price_23_29_july.png", dpi=300, bbox_inches="tight")
    fig.savefig(pdx / "lstm_price_23_29_july.pdf", bbox_inches="tight")
    plt.close(fig)

    fig, axs = plt.subplots(2, 2, figsize=(16, 9))
    axs = axs.ravel()

    for i, hid in enumerate(ids):
        s = load_weeks[hid]
        hours = np.arange(1, len(s) + 1)
        axs[i].plot(
            hours,
            s["actual_load_kw"],
            color="red",
            linewidth=2.2,
            label="Real",
        )
        axs[i].plot(
            hours,
            s["forecast_load_kw"],
            color="blue",
            linestyle="--",
            linewidth=2.2,
            label="Forecast",
        )
        _style_panel(
            axs[i],
            "Time (h)",
            "Load (kW)",
            f"({chr(97 + i)}) EU {i + 1}",
        )

    hours = np.arange(1, len(price_week) + 1)
    axs[3].plot(
        hours,
        price_week["actual_price_cent_per_kWh"],
        color="red",
        linewidth=2.2,
        label="Real",
    )
    axs[3].plot(
        hours,
        price_week["forecast_price_cent_per_kWh"],
        color="blue",
        linestyle="--",
        linewidth=2.2,
        label="Forecast",
    )
    _style_panel(
        axs[3],
        "Time (h)",
        "Price (cent/kWh)",
        "(d) Wholesale electricity price",
    )

    fig.tight_layout()
    try:
        fig.savefig(out / "forecast_comparison.png", dpi=300, bbox_inches="tight")
        fig.savefig(out / "forecast_comparison.pdf", bbox_inches="tight")
    except PermissionError as exc:
        raise PermissionError(
            "combined forecast figure is open/locked. Close the PNG/PDF viewer and rerun."
        ) from exc
    finally:
        plt.close(fig)

    # Keep the internal train-validation gate logic intact.
    flags = []
    for gf in [
        ld / "lstm_load_train_validation_gap_audit.csv",
        pdx / "lstm_price_train_validation_gap_audit.csv",
    ]:
        g = pd.read_csv(gf)
        flags.extend(g["requires_blocked_crossfit"].astype(bool).tolist())

    gate = (
        "BLOCKED_FOR_CROSSFIT"
        if any(flags)
        else "READY_FOR_DDQN_INTEGRATION"
    )
    (out / "integration_gate.txt").write_text(
        gate + "\n", encoding="utf-8"
    )

    (root / "README_OUTPUTS.md").write_text(
        "# Forecast outputs\n\n"
        "The root forecast directory contains the combined forecast figure, "
        "forecast metric files, and the integration-gate file.\n\n"
        "`load/` contains household models, scalers, histories, predictions, "
        "diagnostics, and standalone load figures.\n\n"
        "`price/` contains the price model, scaler, history, predictions, "
        "diagnostics, and the standalone price figure.\n\n"
        "The final metric table reports MAE only.\n",
        encoding="utf-8",
    )

    print("\nForecast MAE")
    print(table.to_string(index=False))
    print(f"\nIntegration gate: {gate}")


if __name__ == "__main__":
    main()
