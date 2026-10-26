"""Create the monitoring charts from the pipeline logs.

Run after the pipeline:  python -m src.visualize
"""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np
import pandas as pd

from . import config
from .data_generator import generate_batch

PHASE_COLORS = {"stable": "#e5f4e4", "data_drift": "#f9eeda", "concept_drift": "#fbebeb"}
PHASE_LABELS = {"stable": "Stable", "data_drift": "Data drift", "concept_drift": "Concept drift"}
BLUE, GREY, RED = "#2563eb", "#9ca3af", "#dc2626"


def _shade_phases(ax, df):
    for phase, g in df.groupby("phase", sort=False):
        ax.axvspan(g["week"].min() - 0.5, g["week"].max() + 0.5,
                   color=PHASE_COLORS[phase], zorder=0)
        ax.text((g["week"].min() + g["week"].max()) / 2, 1.0, PHASE_LABELS[phase],
                transform=ax.get_xaxis_transform(), ha="center", va="bottom",
                fontsize=10, color="#374151")


def performance_chart(df):
    fig, ax = plt.subplots(figsize=(10, 5))
    _shade_phases(ax, df)
    ax.plot(df["week"], df["auc_static"], "o--", color=GREY, lw=2, label="Static model (never retrained)")
    ax.plot(df["week"], df["auc_mlops"], "o-", color=BLUE, lw=2.5, label="DriftGuard (auto-retraining)")
    for _, r in df[df["action"] == "retrained_promoted"].iterrows():
        ax.annotate(f"retrain -> v{int(r['new_version'])}", xy=(r["week"], r["auc_mlops"]),
                    xytext=(r["week"] - 2.4, r["auc_mlops"] + (0.08 if r["auc_mlops"] < 0.65 else -0.09)),
                    color=RED, fontsize=9,
                    arrowprops=dict(arrowstyle="->", color=RED))
    ax.set_xlabel("Production week")
    ax.set_ylabel("ROC-AUC on that week's data")
    ax.set_xticks(df["week"])
    ax.set_ylim(0.5, 0.95)
    ax.grid(axis="y", alpha=0.3)
    ax.legend(loc="lower left")
    ax.set_title("Model performance: static vs. DriftGuard", pad=22, fontsize=13, weight="bold")
    fig.tight_layout()
    fig.savefig(config.FIG_DIR / "01_performance_over_time.png", dpi=150)
    plt.close(fig)


def psi_heatmap(df):
    psi = df[[f"psi_{f}" for f in config.FEATURES]].T.values
    fig, ax = plt.subplots(figsize=(10, 4.2))
    im = ax.imshow(psi, cmap="YlOrRd", vmin=0, vmax=0.6, aspect="auto")
    ax.set_yticks(range(len(config.FEATURES)), config.FEATURES)
    ax.set_xticks(range(len(df)), [f"W{w}" for w in df["week"]])
    for i in range(psi.shape[0]):
        for j in range(psi.shape[1]):
            v = psi[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=8,
                    color="white" if v > 0.35 else "black")
    cb = fig.colorbar(im, ax=ax)
    cb.set_label("PSI (>0.25 = critical)")
    ax.set_title("Population Stability Index per feature and week\n"
                 "(compared with the reference data of the current production model)",
                 fontsize=12, weight="bold")
    fig.tight_layout()
    fig.savefig(config.FIG_DIR / "02_psi_heatmap.png", dpi=150)
    plt.close(fig)


def distribution_shift():
    ref = generate_batch(config.REFERENCE_SIZE, 0)
    cur = generate_batch(config.BATCH_SIZE, 8)
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8))
    for ax, f in zip(axes, ["income", "credit_score", "debt_to_income"]):
        lo, hi = np.percentile(pd.concat([ref[f], cur[f]]), [1, 99])
        bins = np.linspace(lo, hi, 40)
        ax.hist(ref[f], bins=bins, density=True, alpha=0.55, color=BLUE, label="Training (week 0)")
        ax.hist(cur[f], bins=bins, density=True, alpha=0.55, color=RED, label="Production (week 8)")
        ax.set_title(f, weight="bold")
        ax.set_yticks([])
        if f == "income":
            ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x, _: f"{x/1000:.0f}k"))
    axes[0].legend(fontsize=8)
    fig.suptitle("Data drift: the same features look different in production", weight="bold")
    fig.tight_layout()
    fig.savefig(config.FIG_DIR / "03_distribution_shift.png", dpi=150)
    plt.close(fig)


def drift_share_chart(df):
    fig, ax = plt.subplots(figsize=(10, 3.8))
    _shade_phases(ax, df)
    colors = [RED if d else BLUE for d in df["data_drift"]]
    ax.bar(df["week"], df["drift_share"] * 100, color=colors, width=0.6)
    ax.axhline(config.DRIFT_SHARE_THRESHOLD * 100, color="black", ls="--", lw=1)
    ax.text(0.6, config.DRIFT_SHARE_THRESHOLD * 100 + 2, "retraining threshold (30%)", fontsize=9)
    perf = df[df["performance_drift"]]
    ax.scatter(perf["week"], [5] * len(perf), marker="v", s=120, color="#7c3aed", zorder=3,
               label="performance-drift alert (AUC drop)")
    ax.set_xticks(df["week"])
    ax.set_ylim(0, 100)
    ax.set_xlabel("Production week")
    ax.set_ylabel("% features with critical drift")
    ax.legend(loc="upper left")
    ax.set_title("Drift alerts per week", pad=22, fontsize=13, weight="bold")
    fig.tight_layout()
    fig.savefig(config.FIG_DIR / "04_drift_alerts.png", dpi=150)
    plt.close(fig)


def text_screenshot(src, dst, title):
    """Render a text file (console output) as a terminal-style PNG screenshot."""
    lines = src.read_text().splitlines()
    h = 0.19 * len(lines) + 0.8
    fig = plt.figure(figsize=(11, h))
    fig.patch.set_facecolor("#1e1e1e")
    fig.text(0.01, 1 - 0.25 / h, f"$ {title}", color="#4ade80", family="monospace", fontsize=9, va="top")
    for i, line in enumerate(lines):
        color = "#f87171" if ("ALERT" in line or "FAIL" in line) else \
                "#facc15" if "RETRAIN" in line else \
                "#4ade80" if (" OK" in line or "PROMOTED" in line) else "#e5e7eb"
        fig.text(0.01, 1 - (0.5 + 0.19 * (i + 1)) / h, line, color=color,
                 family="monospace", fontsize=8.5, va="top")
    fig.savefig(dst, dpi=150, facecolor=fig.get_facecolor())
    plt.close(fig)


def main():
    config.FIG_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(config.REPORT_DIR / "monitoring_log.csv")
    performance_chart(df)
    psi_heatmap(df)
    distribution_shift()
    drift_share_chart(df)
    out = config.REPORT_DIR / "pipeline_output.txt"
    if out.exists():
        text_screenshot(out, config.FIG_DIR / "05_pipeline_console.png", "python -m src.pipeline")
    tests = config.REPORT_DIR / "test_output.txt"
    if tests.exists():
        text_screenshot(tests, config.FIG_DIR / "06_unit_tests.png", "python -m unittest -v")
    summary = json.loads((config.REPORT_DIR / "summary.json").read_text())
    print(f"Charts written to {config.FIG_DIR} | mean AUC static={summary['mean_auc_static']} "
          f"mlops={summary['mean_auc_mlops']}")


if __name__ == "__main__":
    main()
