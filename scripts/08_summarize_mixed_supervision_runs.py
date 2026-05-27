#!/usr/bin/env python3
"""Summarize mixed-supervision runs into annotation-budget tables and plots."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

PROJECT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from osteophytes.labels import LOCATION_NAMES
from osteophytes.plotting import configure_matplotlib, plot_budget_curve, plot_high_grade_representation


DEFAULT_RUNS_ROOT = Path("/scratch/dgogoana/osteophytes_project/runs/02_mixed_supervision")
DEFAULT_OUTPUT_DIR = DEFAULT_RUNS_ROOT / "summary"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-root", type=Path, default=DEFAULT_RUNS_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def find_run_dirs(runs_root: Path) -> list[Path]:
    run_dirs = []
    for config_path in runs_root.rglob("config.json"):
        run_dir = config_path.parent
        if (run_dir / "metrics_history.json").exists():
            run_dirs.append(run_dir)
    return sorted(run_dirs)


def best_epoch_record(history: list[dict[str, Any]], selection_metric: str) -> dict[str, Any]:
    if not history:
        raise ValueError("Empty metrics history")
    direction = "lower" if selection_metric == "mean_mae" else "higher"
    finite_records = [
        record
        for record in history
        if math.isfinite(float(record.get("selection_value", math.nan)))
    ]
    if finite_records:
        if direction == "lower":
            return min(finite_records, key=lambda record: float(record["selection_value"]))
        return max(finite_records, key=lambda record: float(record["selection_value"]))
    return history[-1]


def grade3_counts(run_dir: Path) -> dict[str, int]:
    counts = {f"grade3_count_{location.removeprefix('osteo_')}": 0 for location in LOCATION_NAMES}
    dist_path = run_dir / "strong_grade_distribution_by_location.csv"
    if not dist_path.exists():
        return counts
    import pandas as pd

    dist = pd.read_csv(dist_path)
    for location in LOCATION_NAMES:
        column = f"grade3_count_{location.removeprefix('osteo_')}"
        rows = dist[(dist["location"] == location) & (dist["grade"] == 3)]
        if len(rows):
            counts[column] = int(rows.iloc[0]["count"])
    return counts


def summarize_run(run_dir: Path) -> dict[str, Any]:
    config = load_json(run_dir / "config.json")
    history = load_json(run_dir / "metrics_history.json")
    selection_metric = str(config.get("selection_metric", history[-1].get("selection_metric", "mean_spearman")))
    best = best_epoch_record(history, selection_metric)
    val = best["val"]
    mean = val.get("mean", {})
    return {
        "experiment_name": config.get("experiment_name", run_dir.parent.name),
        "timestamp": run_dir.name,
        "run_dir": str(run_dir),
        "weak_label_mode": config.get("weak_label_mode"),
        "strong_fraction": float(config.get("effective_strong_fraction", config.get("strong_fraction", math.nan))),
        "strong_count": int(config.get("strong_count", config.get("num_strong_samples", 0))),
        "weak_count": int(config.get("weak_count", 0)),
        "strong_sampling_strategy": config.get("strong_sampling_strategy"),
        "model_head": config.get("model_head"),
        "init_from_binary_checkpoint": config.get("init_from_binary_checkpoint"),
        "loss_balance_mode": config.get("loss_balance_mode"),
        "ordinal_class_weighting": config.get("ordinal_class_weighting"),
        "seed": config.get("seed"),
        "best_epoch": int(best.get("epoch", -1)),
        "mean_spearman": float(mean.get("spearman", math.nan)),
        "mean_mae_expected": float(mean.get("mae_expected", math.nan)),
        "mean_qwk": float(mean.get("qwk", math.nan)),
        "mean_auroc": float(mean.get("binary_auroc", math.nan)),
        **grade3_counts(run_dir),
    }


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    import pandas as pd
    import matplotlib.pyplot as plt

    configure_matplotlib(plt)
    rows = [summarize_run(run_dir) for run_dir in find_run_dirs(args.runs_root)]
    if not rows:
        raise ValueError(f"No mixed-supervision runs found under {args.runs_root}")

    summary = pd.DataFrame(rows).sort_values(["strong_fraction", "experiment_name", "timestamp"])
    output_csv = args.output_dir / "annotation_budget_summary.csv"
    summary.to_csv(output_csv, index=False)

    plot_budget_curve(
        summary,
        "mean_spearman",
        args.output_dir / "annotation_budget_curve_spearman.png",
        "Mean Spearman",
        "Annotation budget curve: Spearman",
        plt,
    )
    plot_budget_curve(
        summary,
        "mean_mae_expected",
        args.output_dir / "annotation_budget_curve_mae.png",
        "Mean MAE(expected)",
        "Annotation budget curve: expected-grade MAE",
        plt,
    )
    plot_budget_curve(
        summary,
        "mean_qwk",
        args.output_dir / "annotation_budget_curve_qwk.png",
        "Mean QWK",
        "Annotation budget curve: quadratic weighted kappa",
        plt,
    )
    plot_high_grade_representation(
        summary,
        args.output_dir / "high_grade_representation_by_budget.png",
        plt,
    )
    print(f"Saved summary: {output_csv}")
    print(f"Saved plots to: {args.output_dir}")


if __name__ == "__main__":
    main()
