#!/usr/bin/env python3
"""Plot and summarize the first binary baseline run."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


DEFAULT_RUN_DIR = Path("/scratch/dgogoana/osteophytes_project/runs/01_binary_baseline/20260519_235102")
LOCATIONS = (
    "osteo_acet_inf",
    "osteo_acet_sup",
    "osteo_fem_inf",
    "osteo_fem_sup",
)
HUMAN_NAMES = {
    "osteo_acet_inf": "Inferior acetabular",
    "osteo_acet_sup": "Superior acetabular",
    "osteo_fem_inf": "Inferior femoral",
    "osteo_fem_sup": "Superior femoral",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--output-dir", type=Path, help="Directory for figures and summary tables.")
    return parser.parse_args()


def as_float(value: Any) -> float:
    if value is None:
        return math.nan
    return float(value)


def load_history(path: Path) -> list[dict[str, Any]]:
    history = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(history, list) or not history:
        raise ValueError(f"Expected a non-empty metrics history list: {path}")
    return history


def best_epoch_record(history: list[dict[str, Any]]) -> dict[str, Any]:
    finite_records = [
        record for record in history if math.isfinite(as_float(record.get("val", {}).get("mean_auc")))
    ]
    if finite_records:
        return max(finite_records, key=lambda record: as_float(record["val"]["mean_auc"]))
    return history[-1]


def require_prediction_columns(predictions: Any) -> None:
    required = {"subject", "visit", "side", "split"}
    for location in LOCATIONS:
        required.add(location)
        required.add(f"prob_{location}")
        required.add(f"{location}_binary")
    missing = [column for column in required if column not in predictions.columns]
    if missing:
        raise ValueError(f"Missing required val_predictions.csv columns: {', '.join(missing)}")


def configure_matplotlib(plt: Any) -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 160,
            "savefig.dpi": 200,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "legend.frameon": False,
        }
    )


def plot_training_curves(history: list[dict[str, Any]], output_path: Path, plt: Any) -> None:
    epochs = [int(record["epoch"]) for record in history]
    train_loss = [as_float(record["train_loss"]) for record in history]
    val_loss = [as_float(record["val"]["loss"]) for record in history]

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(epochs, train_loss, marker="o", linewidth=2, label="Train loss")
    ax.plot(epochs, val_loss, marker="o", linewidth=2, label="Validation loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Binary baseline training curves")
    ax.set_xticks(epochs)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def plot_validation_summary(history: list[dict[str, Any]], output_path: Path, plt: Any) -> None:
    epochs = [int(record["epoch"]) for record in history]
    mean_auc = [as_float(record["val"]["mean_auc"]) for record in history]
    mean_spearman = [as_float(record["val"]["mean_spearman"]) for record in history]

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(epochs, mean_auc, marker="o", linewidth=2, label="Mean AUROC")
    ax.plot(epochs, mean_spearman, marker="o", linewidth=2, label="Mean Spearman")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Metric value")
    ax.set_title("Validation summary over epochs")
    ax.set_xticks(epochs)
    ax.set_ylim(0.0, 1.0)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def plot_metric_bars(
    values: dict[str, float],
    output_path: Path,
    title: str,
    ylabel: str,
    ylim: tuple[float, float],
    plt: Any,
) -> None:
    labels = [HUMAN_NAMES[location] for location in LOCATIONS]
    metric_values = [values[location] for location in LOCATIONS]
    colors = ["#4C78A8", "#F58518", "#54A24B", "#B279A2"]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    bars = ax.bar(labels, metric_values, color=colors)
    ax.set_ylabel(ylabel)
    ax.set_ylim(*ylim)
    ax.set_title(title)
    ax.tick_params(axis="x", rotation=20)

    for bar, value in zip(bars, metric_values, strict=True):
        if math.isfinite(value):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                value + 0.01,
                f"{value:.3f}",
                ha="center",
                va="bottom",
                fontsize=9,
            )
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def probability_groups(predictions: Any, location: str, pd: Any) -> list[Any]:
    grades = pd.to_numeric(predictions[location], errors="coerce")
    probabilities = pd.to_numeric(predictions[f"prob_{location}"], errors="coerce")
    groups = []
    for grade in range(4):
        values = probabilities[(grades == grade) & probabilities.notna()].to_numpy()
        groups.append(values)
    return groups


def draw_probability_boxplot(
    ax: Any,
    groups: list[Any],
    title: str,
    np: Any,
) -> None:
    positions = []
    data = []
    for grade, values in enumerate(groups):
        if len(values) > 0:
            positions.append(grade)
            data.append(values)
    if data:
        box = ax.boxplot(
            data,
            positions=positions,
            widths=0.55,
            patch_artist=True,
            showfliers=False,
            medianprops={"color": "black", "linewidth": 1.4},
        )
        for patch in box["boxes"]:
            patch.set_facecolor("#9ecae9")
            patch.set_edgecolor("#2b5c7e")
    for grade, values in enumerate(groups):
        if len(values) > 0:
            mean_value = float(np.mean(values))
            ax.scatter([grade], [mean_value], color="#d62728", s=24, zorder=3)
            ax.text(grade, 0.04, f"n={len(values)}", ha="center", va="bottom", fontsize=8)
        else:
            ax.text(grade, 0.5, "n=0", ha="center", va="center", fontsize=8, color="gray")
    ax.set_title(title)
    ax.set_xlabel("OARSI grade")
    ax.set_ylabel("Predicted binary probability")
    ax.set_xticks([0, 1, 2, 3])
    ax.set_ylim(0.0, 1.0)


def plot_probability_grid(predictions: Any, output_path: Path, plt: Any, np: Any, pd: Any) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(9, 7), sharex=True, sharey=True)
    for ax, location in zip(axes.ravel(), LOCATIONS, strict=True):
        groups = probability_groups(predictions, location, pd)
        draw_probability_boxplot(ax, groups, HUMAN_NAMES[location], np)
    fig.suptitle("Predicted binary probability by true OARSI grade", y=0.995)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def plot_probability_single(
    predictions: Any,
    location: str,
    output_path: Path,
    plt: Any,
    np: Any,
    pd: Any,
) -> None:
    fig, ax = plt.subplots(figsize=(5.5, 4))
    groups = probability_groups(predictions, location, pd)
    draw_probability_boxplot(ax, groups, HUMAN_NAMES[location], np)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def make_best_epoch_summary(best_record: dict[str, Any], pd: Any) -> Any:
    rows = []
    val = best_record["val"]
    for location in LOCATIONS:
        rows.append(
            {
                "location": location,
                "human_location": HUMAN_NAMES[location],
                "auroc": as_float(val["auc"].get(location)),
                "spearman": as_float(val["spearman"].get(location)),
            }
        )
    return pd.DataFrame(rows)


def make_probability_summary(predictions: Any, pd: Any, np: Any) -> Any:
    rows = []
    for location in LOCATIONS:
        grades = pd.to_numeric(predictions[location], errors="coerce")
        probabilities = pd.to_numeric(predictions[f"prob_{location}"], errors="coerce")
        for grade in range(4):
            values = probabilities[(grades == grade) & probabilities.notna()].to_numpy(dtype=float)
            if len(values) == 0:
                stats = {
                    "n": 0,
                    "mean_probability": math.nan,
                    "median_probability": math.nan,
                    "std_probability": math.nan,
                    "q25_probability": math.nan,
                    "q75_probability": math.nan,
                }
            else:
                stats = {
                    "n": int(len(values)),
                    "mean_probability": float(np.mean(values)),
                    "median_probability": float(np.median(values)),
                    "std_probability": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
                    "q25_probability": float(np.percentile(values, 25)),
                    "q75_probability": float(np.percentile(values, 75)),
                }
            rows.append(
                {
                    "location": location,
                    "human_location": HUMAN_NAMES[location],
                    "grade": grade,
                    **stats,
                }
            )
    return pd.DataFrame(rows)


def format_float(value: float) -> str:
    return "NaN" if not math.isfinite(value) else f"{value:.6f}"


def markdown_metrics_table(summary: Any) -> str:
    lines = [
        "| location | human_location | auroc | spearman |",
        "|---|---|---:|---:|",
    ]
    for row in summary.itertuples(index=False):
        lines.append(
            f"| {row.location} | {row.human_location} | "
            f"{format_float(float(row.auroc))} | {format_float(float(row.spearman))} |"
        )
    return "\n".join(lines)


def write_report(
    output_path: Path,
    run_dir: Path,
    best_record: dict[str, Any],
    best_summary: Any,
) -> None:
    val = best_record["val"]
    best_epoch = int(best_record["epoch"])
    text = f"""# Binary Baseline Interpretation Notes

Run directory:

```text
{run_dir}
```

Best epoch by mean validation AUROC: **{best_epoch}**

- Mean AUROC: **{format_float(as_float(val["mean_auc"]))}**
- Mean Spearman: **{format_float(as_float(val["mean_spearman"]))}**

## Per-Location Validation Metrics

{markdown_metrics_table(best_summary)}

## Interpretation

AUROC measures binary presence/absence discrimination for each osteophyte
location. In this baseline, the model is trained only to distinguish grade 0
from grades 1/2/3 collapsed into osteophyte-present.

Spearman correlation measures whether the model's binary probabilities increase
with true OARSI severity grades 0/1/2/3. Graded labels are used only for this
validation analysis, not for training.

These results are validation-set evidence from the first binary-supervised
baseline run. They should be interpreted as an initial baseline, not as final
test-set performance or evidence of generalization.
"""
    output_path.write_text(text, encoding="utf-8")


def main() -> None:
    args = parse_args()

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd

    configure_matplotlib(plt)

    run_dir = args.run_dir
    output_dir = args.output_dir or (run_dir / "figures")
    output_dir.mkdir(parents=True, exist_ok=True)

    metrics_path = run_dir / "metrics_history.json"
    predictions_path = run_dir / "val_predictions.csv"
    history = load_history(metrics_path)
    predictions = pd.read_csv(predictions_path)
    require_prediction_columns(predictions)
    best_record = best_epoch_record(history)
    best_epoch = int(best_record["epoch"])

    saved_paths: list[Path] = []

    path = output_dir / "training_curves.png"
    plot_training_curves(history, path, plt)
    saved_paths.append(path)

    path = output_dir / "validation_summary_over_epochs.png"
    plot_validation_summary(history, path, plt)
    saved_paths.append(path)

    best_auc = {location: as_float(best_record["val"]["auc"].get(location)) for location in LOCATIONS}
    path = output_dir / "best_epoch_auroc_by_location.png"
    plot_metric_bars(
        best_auc,
        path,
        title=f"Validation AUROC by location (best epoch {best_epoch})",
        ylabel="AUROC",
        ylim=(0.5, 1.0),
        plt=plt,
    )
    saved_paths.append(path)

    best_spearman = {
        location: as_float(best_record["val"]["spearman"].get(location)) for location in LOCATIONS
    }
    path = output_dir / "best_epoch_spearman_by_location.png"
    plot_metric_bars(
        best_spearman,
        path,
        title=f"Validation Spearman by location (best epoch {best_epoch})",
        ylabel="Spearman correlation",
        ylim=(0.0, 1.0),
        plt=plt,
    )
    saved_paths.append(path)

    path = output_dir / "probability_by_oarsi_grade_all_locations.png"
    plot_probability_grid(predictions, path, plt, np, pd)
    saved_paths.append(path)

    for location in LOCATIONS:
        path = output_dir / f"probability_by_grade_{location}.png"
        plot_probability_single(predictions, location, path, plt, np, pd)
        saved_paths.append(path)

    best_summary = make_best_epoch_summary(best_record, pd)
    path = output_dir / "best_epoch_metrics_summary.csv"
    best_summary.to_csv(path, index=False)
    saved_paths.append(path)

    probability_summary = make_probability_summary(predictions, pd, np)
    path = output_dir / "probability_by_grade_summary.csv"
    probability_summary.to_csv(path, index=False)
    saved_paths.append(path)

    path = output_dir / "baseline_interpretation_notes.md"
    write_report(path, run_dir, best_record, best_summary)
    saved_paths.append(path)

    print("Saved outputs:")
    for saved_path in saved_paths:
        print(f"  {saved_path}")


if __name__ == "__main__":
    main()
