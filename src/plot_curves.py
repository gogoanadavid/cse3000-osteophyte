"""Plot learning curves from collected result tables."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def _import_pyplot():
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        raise RuntimeError("matplotlib is required for plotting; install or load it before running plot_curves") from exc
    return plt


def _plot_metric(df: pd.DataFrame, metric: str, ylabel: str, out: Path, lower_better: bool = False) -> None:
    plt = _import_pyplot()
    fig, ax = plt.subplots(figsize=(7.5, 5))
    group_cols = ["mode", "strategy"]
    for (mode, strategy), g in df.groupby(group_cols):
        g = g.copy()
        g[metric] = pd.to_numeric(g[metric], errors="coerce")
        g["budget_fraction"] = pd.to_numeric(g["budget_fraction"], errors="coerce")
        summary = g.groupby("budget_fraction")[metric].agg(["mean", "sem"]).reset_index().sort_values("budget_fraction")
        ax.plot(summary["budget_fraction"], summary["mean"], marker="o", label=f"{mode}/{strategy}")
        if len(g["seed"].dropna().unique()) > 1:
            sem = summary["sem"].fillna(0.0)
            ax.fill_between(summary["budget_fraction"], summary["mean"] - sem, summary["mean"] + sem, alpha=0.15)
    ax.set_xlabel("Fraction of graded training annotations")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8)
    if lower_better:
        ax.set_title(f"{ylabel} (lower is better)")
    fig.tight_layout()
    fig.savefig(out, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", required=True)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()
    df = pd.read_csv(args.results)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if df.empty:
        raise ValueError(f"No rows found in {args.results}")

    _plot_metric(df, "quality_mean", "Quality = 1 - BMAE/3", out_dir / "main_learning_curve_quality.png")
    _plot_metric(df, "bmae_mean", "Balanced ordinal MAE", out_dir / "main_learning_curve_bmae.png", lower_better=True)

    plt = _import_pyplot()
    fig, ax = plt.subplots(figsize=(7.5, 5))
    for metric in ["auc_ge1_mean", "auc_ge2_mean", "auc_ge3_mean"]:
        g = df.groupby("budget_fraction")[metric].mean(numeric_only=True).reset_index().sort_values("budget_fraction")
        ax.plot(g["budget_fraction"], g[metric], marker="o", label=metric.replace("_mean", ""))
    ax.set_xlabel("Fraction of graded training annotations")
    ax.set_ylabel("AUROC")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "boundary_auc_curves.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.5, 5))
    for metric in ["ap_ge2_mean", "ap_ge3_mean"]:
        g = df.groupby("budget_fraction")[metric].mean(numeric_only=True).reset_index().sort_values("budget_fraction")
        ax.plot(g["budget_fraction"], g[metric], marker="o", label=metric.replace("_mean", ""))
    ax.set_xlabel("Fraction of graded training annotations")
    ax.set_ylabel("AUPRC")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "boundary_ap_curves.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.5, 5))
    for metric in ["recall_grade0_mean", "recall_grade1_mean", "recall_grade2_mean", "recall_grade3_mean"]:
        g = df.groupby("budget_fraction")[metric].mean(numeric_only=True).reset_index().sort_values("budget_fraction")
        ax.plot(g["budget_fraction"], g[metric], marker="o", label=metric.replace("_mean", ""))
    ax.set_xlabel("Fraction of graded training annotations")
    ax.set_ylabel("Recall")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "per_grade_recall_curves.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.5, 5))
    for strategy, g in df.groupby("strategy"):
        summary = g.groupby("budget_fraction")["quality_mean"].mean(numeric_only=True).reset_index().sort_values("budget_fraction")
        ax.plot(summary["budget_fraction"], summary["quality_mean"], marker="o", label=strategy)
    ax.set_xlabel("Fraction of graded training annotations")
    ax.set_ylabel("Quality = 1 - BMAE/3")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "sampling_strategy_comparison.png", dpi=180)
    plt.close(fig)
    print(f"Wrote figures to {out_dir}")


if __name__ == "__main__":
    main()
