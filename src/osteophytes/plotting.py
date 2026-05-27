"""Small matplotlib helpers for experiment summaries."""

from __future__ import annotations

from pathlib import Path
from typing import Any


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


def plot_budget_curve(
    summary: Any,
    y_column: str,
    output_path: str | Path,
    ylabel: str,
    title: str,
    plt: Any,
) -> None:
    fig, ax = plt.subplots(figsize=(7, 4.2))
    grouped = summary.sort_values("strong_fraction").groupby(
        ["model_head", "strong_sampling_strategy"],
        dropna=False,
    )
    for (model_head, strategy), group in grouped:
        label = f"{model_head} / {strategy}"
        ax.plot(
            group["strong_fraction"],
            group[y_column],
            marker="o",
            linewidth=2,
            label=label,
        )
    ax.set_xlabel("Fraction of complete graded training annotations")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def plot_high_grade_representation(summary: Any, output_path: str | Path, plt: Any) -> None:
    grade3_columns = [column for column in summary.columns if column.startswith("grade3_count_")]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for column in grade3_columns:
        ax.plot(
            summary.sort_values("strong_fraction")["strong_fraction"],
            summary.sort_values("strong_fraction")[column],
            marker="o",
            linewidth=2,
            label=column.removeprefix("grade3_count_"),
        )
    ax.set_xlabel("Fraction of complete graded training annotations")
    ax.set_ylabel("Strong-set grade 3 count")
    ax.set_title("High-grade representation by annotation budget")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
