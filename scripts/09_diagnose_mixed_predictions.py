#!/usr/bin/env python3
"""Diagnose whether a mixed-supervision run is learning severity signal."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

PROJECT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


REQUIRED_COLUMNS = {
    "sample_id",
    "location",
    "true_grade",
    "pred_grade",
    "expected_grade",
    "p_present",
    "p_gt_0",
    "p_gt_1",
    "p_gt_2",
    "p_grade_0",
    "p_grade_1",
    "p_grade_2",
    "p_grade_3",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    return parser.parse_args()


def require_columns(frame: Any) -> None:
    missing = sorted(REQUIRED_COLUMNS - set(frame.columns))
    if missing:
        raise ValueError(f"Missing required prediction columns: {', '.join(missing)}")


def confusion_rows(frame: Any) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for location, group in frame.groupby("location"):
        for true_grade in range(4):
            for pred_grade in range(4):
                count = int(((group["true_grade"] == true_grade) & (group["pred_grade"] == pred_grade)).sum())
                rows.append(
                    {
                        "location": location,
                        "true_grade": true_grade,
                        "pred_grade": pred_grade,
                        "count": count,
                    }
                )
    return rows


def diagnostic_summary(frame: Any) -> list[dict[str, Any]]:
    p_sum = frame[["p_grade_0", "p_grade_1", "p_grade_2", "p_grade_3"]].sum(axis=1)
    monotonic_violations = (
        (frame["p_gt_0"] + 1e-8 < frame["p_gt_1"])
        | (frame["p_gt_1"] + 1e-8 < frame["p_gt_2"])
    )
    return [
        {"metric": "validation_rows", "value": int(len(frame))},
        {"metric": "unique_samples", "value": int(frame["sample_id"].nunique())},
        {"metric": "grade_probability_sum_mean", "value": float(p_sum.mean())},
        {"metric": "grade_probability_sum_max_abs_error", "value": float((p_sum - 1.0).abs().max())},
        {"metric": "threshold_monotonicity_violations", "value": int(monotonic_violations.sum())},
        {"metric": "min_grade_probability", "value": float(frame[["p_grade_0", "p_grade_1", "p_grade_2", "p_grade_3"]].min().min())},
        {"metric": "max_grade_probability", "value": float(frame[["p_grade_0", "p_grade_1", "p_grade_2", "p_grade_3"]].max().max())},
    ]


def main() -> None:
    args = parse_args()
    import pandas as pd

    prediction_path = args.run_dir / "val_predictions.csv"
    if not prediction_path.exists():
        raise FileNotFoundError(f"Prediction CSV not found: {prediction_path}")
    predictions = pd.read_csv(prediction_path)
    require_columns(predictions)

    expected_by_grade = (
        predictions.groupby(["location", "true_grade"], dropna=False)
        .agg(
            n=("expected_grade", "size"),
            expected_grade_mean=("expected_grade", "mean"),
            expected_grade_median=("expected_grade", "median"),
            p_present_mean=("p_present", "mean"),
            p_present_median=("p_present", "median"),
        )
        .reset_index()
    )
    pred_distribution = (
        predictions.groupby(["location", "pred_grade"], dropna=False)
        .size()
        .reset_index(name="count")
    )
    true_distribution = (
        predictions.groupby(["location", "true_grade"], dropna=False)
        .size()
        .reset_index(name="count")
    )
    summary = pd.DataFrame(diagnostic_summary(predictions))
    confusion = pd.DataFrame(confusion_rows(predictions))

    summary.to_csv(args.run_dir / "diagnostic_summary.csv", index=False)
    expected_by_grade.to_csv(args.run_dir / "expected_grade_by_true_grade.csv", index=False)
    pred_distribution.to_csv(args.run_dir / "predicted_grade_distribution.csv", index=False)
    true_distribution.to_csv(args.run_dir / "true_grade_distribution.csv", index=False)
    confusion.to_csv(args.run_dir / "confusion_matrices.csv", index=False)

    print("Diagnostic summary")
    print(summary.to_string(index=False))
    print("\nExpected grade and p_present by true grade")
    print(expected_by_grade.to_string(index=False))
    print("\nPredicted grade distribution")
    print(pred_distribution.to_string(index=False))
    print("\nTrue grade distribution")
    print(true_distribution.to_string(index=False))
    print("\nConfusion matrices saved to confusion_matrices.csv")


if __name__ == "__main__":
    main()
