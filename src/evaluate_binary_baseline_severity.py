"""Evaluate severity signal from a binary-only baseline using p_ge1 only."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .config import load_config
from .metrics import (
    adjacent_auc,
    average_precision,
    balanced_ordinal_mae,
    binary_auc,
    quality_from_bmae,
    spearman_corr,
)


def _finite_mean(values: list[float]) -> float:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    return float(arr.mean()) if arr.size else float("nan")


def _metric_row(y: np.ndarray, score: np.ndarray, location: str) -> dict[str, Any]:
    valid = y >= 0
    y = y[valid].astype(int)
    score = score[valid].astype(float)
    row: dict[str, Any] = {
        "location": location,
        "n": int(len(y)),
        "bmae": balanced_ordinal_mae(y, score),
        "quality": quality_from_bmae(balanced_ordinal_mae(y, score)),
        "auc_ge1": binary_auc((y >= 1).astype(int), score),
        "auc_ge2": binary_auc((y >= 2).astype(int), score),
        "auc_ge3": binary_auc((y >= 3).astype(int), score),
        "ap_ge1": average_precision((y >= 1).astype(int), score),
        "ap_ge2": average_precision((y >= 2).astype(int), score),
        "ap_ge3": average_precision((y >= 3).astype(int), score),
        "prevalence_ge2": float((y >= 2).mean()) if len(y) else float("nan"),
        "prevalence_ge3": float((y >= 3).mean()) if len(y) else float("nan"),
        "spearman_all": spearman_corr(y, score),
        "spearman_present": spearman_corr(y[y >= 1], score[y >= 1]) if np.any(y >= 1) else float("nan"),
        "adjacent_auc_1v2": adjacent_auc(y, score, 1, 2),
        "adjacent_auc_2v3": adjacent_auc(y, score, 2, 3),
    }
    for grade in range(4):
        m = y == grade
        row[f"n_grade{grade}"] = int(m.sum())
        row[f"mean_p_ge1_grade{grade}"] = float(score[m].mean()) if m.any() else float("nan")
        row[f"median_p_ge1_grade{grade}"] = float(np.median(score[m])) if m.any() else float("nan")
    return row


def evaluate_binary_baseline(
    data_config: dict[str, Any],
    index_csv: str | Path,
    predictions_csv: str | Path,
    split: str,
    seed: int,
) -> pd.DataFrame:
    """Return per-location and mean binary-baseline severity metrics.

    Only p_ge1 is used as the score for all severity-threshold analyses.
    p_ge2/p_ge3 columns from a binary-only checkpoint are intentionally ignored.
    """
    locations = list(data_config["locations"])
    index = pd.read_csv(index_csv)
    pred = pd.read_csv(predictions_csv)
    index = index[index["split"].astype(str) == str(split)].copy()
    keep_cols = ["h5_index"] + [f"grade_{loc}" for loc in locations]
    df = pred.merge(index[keep_cols], on="h5_index", how="inner")
    if df.empty:
        raise ValueError(f"No overlapping rows for split={split} between {index_csv} and {predictions_csv}")

    rows = []
    pooled_y = []
    pooled_score = []
    for loc in locations:
        y = df[f"grade_{loc}"].to_numpy(dtype=int)
        score = df[f"{loc}_p_ge1"].to_numpy(dtype=float)
        rows.append(_metric_row(y, score, loc))
        valid = y >= 0
        pooled_y.append(y[valid])
        pooled_score.append(score[valid])
    rows.append(_metric_row(np.concatenate(pooled_y), np.concatenate(pooled_score), "pooled"))

    mean_row: dict[str, Any] = {"location": "mean"}
    metric_keys = [k for k in rows[0] if k != "location"]
    location_rows = rows[: len(locations)]
    for key in metric_keys:
        if key.startswith("n") and key != "n":
            mean_row[key] = int(sum(int(r.get(key, 0)) for r in location_rows))
        else:
            mean_row[key] = _finite_mean([float(r.get(key, np.nan)) for r in location_rows])
    rows.append(mean_row)

    out = pd.DataFrame(rows)
    out.insert(0, "seed", seed)
    out.insert(1, "split", split)
    out.insert(2, "mode", "binary_only")
    out.insert(3, "strategy", "binary_baseline")
    out.insert(4, "budget_name", "0")
    out.insert(5, "budget_size", 0)
    return out


def _write_summary(out_path: Path, split: str) -> None:
    pattern = f"binary_baseline_severity_{split}_seed*.csv"
    files = sorted(out_path.parent.glob(pattern))
    if not files:
        return
    df = pd.concat([pd.read_csv(path) for path in files], ignore_index=True)
    mean = df[df["location"] == "mean"].copy()
    if mean.empty:
        return
    numeric_cols = mean.select_dtypes(include=[np.number]).columns.tolist()
    summary = mean.groupby(["split", "mode", "strategy", "budget_name", "budget_size"], dropna=False)[numeric_cols].mean().reset_index()
    summary.to_csv(out_path.parent / f"binary_baseline_severity_{split}_summary.csv", index=False)


def _plot_score_by_grade(metrics_df: pd.DataFrame, data_config: dict[str, Any], index_csv: str | Path, predictions_csv: str | Path, split: str, out_dir: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return
    locations = list(data_config["locations"])
    index = pd.read_csv(index_csv)
    pred = pd.read_csv(predictions_csv)
    index = index[index["split"].astype(str) == str(split)].copy()
    df = pred.merge(index[["h5_index"] + [f"grade_{loc}" for loc in locations]], on="h5_index", how="inner")
    fig, axes = plt.subplots(1, len(locations), figsize=(3.2 * len(locations), 3.2), sharey=True)
    if len(locations) == 1:
        axes = [axes]
    for ax, loc in zip(axes, locations):
        data = []
        labels = []
        for grade in range(4):
            m = df[f"grade_{loc}"].to_numpy(dtype=int) == grade
            values = df.loc[m, f"{loc}_p_ge1"].to_numpy(dtype=float)
            if values.size:
                data.append(values)
                labels.append(str(grade))
        ax.boxplot(data, labels=labels, showfliers=False)
        ax.set_title(loc)
        ax.set_xlabel("True grade")
        ax.grid(True, alpha=0.25)
    axes[0].set_ylabel("Binary baseline p_ge1")
    fig.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / f"binary_baseline_score_by_grade_{split}.png", dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-config", required=True)
    parser.add_argument("--index", required=True)
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--split", required=True, choices=["train", "val", "test"])
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--plot", action="store_true")
    args = parser.parse_args()

    data_config = load_config(args.data_config)
    df = evaluate_binary_baseline(data_config, args.index, args.predictions, args.split, args.seed)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    _write_summary(out, args.split)
    if args.plot:
        _plot_score_by_grade(df, data_config, args.index, args.predictions, args.split, Path("outputs/figures"))
    print(f"Wrote binary baseline severity metrics to {out}")


if __name__ == "__main__":
    main()
