#!/usr/bin/env python3
"""Build a reproducible dataset index for the H5-backed usable dataset."""

from __future__ import annotations

import argparse
import json
import sys
from numbers import Real
from pathlib import Path
from typing import Any

PROJECT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from osteophytes.config import load_config
from osteophytes.labels import OSTEOPHYTE_LABEL_COLUMNS, grade_to_binary, is_complete_graded_annotation


CONFIG_PATH = PROJECT_DIR / "configs" / "delftblue.yaml"
REQUIRED_COLUMNS = ("cohort", "subject", "visit", "side", *OSTEOPHYTE_LABEL_COLUMNS)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv-path", type=Path, help="Path to the label CSV.")
    parser.add_argument("--split-path", type=Path, help="Path to the two-column subject/split file.")
    parser.add_argument("--h5-path", type=Path, help="Path to the H5 image file.")
    parser.add_argument("--output-path", type=Path, help="Output dataset index CSV path.")
    return parser.parse_args()


def require_path(name: str, path: Path | None) -> Path:
    if path is None:
        raise ValueError(f"Missing {name}. Pass --{name.replace('_', '-')} or set it in configs/delftblue.yaml.")
    return path


def path_component(value: Any) -> str:
    if pd.isna(value):
        raise ValueError("Cannot build an H5 path from a missing subject, visit, or side value.")
    if isinstance(value, Real) and not isinstance(value, bool) and float(value).is_integer():
        return str(int(value))
    return str(value).strip()


def binary_label(value: Any) -> int | None:
    if pd.isna(value):
        return None
    if isinstance(value, str) and not value.strip():
        return None
    return grade_to_binary(value)


def h5_image_path(row: pd.Series) -> str:
    subject = path_component(row["subject"])
    visit = path_component(row["visit"])
    side = path_component(row["side"])
    return f"scans/{subject}/{visit}/{side}/image"


def value_counts_dict(series: pd.Series) -> dict[str, int]:
    counts = series.value_counts(dropna=False)
    return {str(key): int(value) for key, value in counts.items()}


def binary_counts_dict(series: pd.Series) -> dict[str, int]:
    binary = series.map(binary_label)
    counts = binary.value_counts(dropna=False)
    return {str(key): int(value) for key, value in counts.items()}


def verify_required_columns(df: pd.DataFrame) -> None:
    missing = [column for column in REQUIRED_COLUMNS if column not in df.columns]
    if missing:
        raise ValueError(f"Missing required label CSV columns: {', '.join(missing)}")


def read_split_file(split_path: Path) -> pd.DataFrame:
    splits = pd.read_csv(split_path, header=None, names=["subject", "split"], dtype="string")
    if splits["split"].isna().all():
        splits = pd.read_csv(
            split_path,
            header=None,
            names=["subject", "split"],
            sep=r"\s+",
            dtype="string",
        )
    splits["subject"] = splits["subject"].map(path_component)
    splits["split"] = splits["split"].astype("string").str.strip()
    if splits["split"].isna().any():
        raise ValueError(f"Expected a two-column split file with no header: {split_path}")
    return splits


def add_binary_label_columns(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    for column in OSTEOPHYTE_LABEL_COLUMNS:
        result[f"{column}_binary"] = result[column].map(binary_label)
    result["complete_graded"] = result.apply(is_complete_graded_annotation, axis=1)
    return result


def print_summary(summary: dict[str, Any]) -> None:
    print("Dataset index summary")
    print(f"  original CSV rows: {summary['original_csv_rows']}")
    print(f"  rows with split: {summary['rows_with_split']}")
    print(f"  rows missing split: {summary['rows_missing_split']}")
    print(f"  rows with H5 image: {summary['rows_with_h5_image']}")
    print(f"  rows missing H5 image: {summary['rows_missing_h5_image']}")

    print("\nUsable rows by split")
    for split, count in summary["usable_rows_by_split"].items():
        print(f"  {split}: {count}")

    complete = summary["complete_graded_rows_overall"]
    print("\nComplete graded rows")
    print(f"  overall: {complete['count']} ({complete['percentage']:.2f}%)")
    for split, values in summary["complete_graded_rows_by_split"].items():
        print(f"  {split}: {values['count']} ({values['percentage']:.2f}%)")

    print("\nGrade distributions on usable rows")
    for column, counts in summary["grade_distributions"].items():
        print(f"  {column}: {counts}")

    print("\nBinary distributions on usable rows")
    for column, counts in summary["binary_distributions"].items():
        print(f"  {column}: {counts}")


def main() -> None:
    args = parse_args()
    config = load_config(CONFIG_PATH)

    global pd
    import pandas as pd

    csv_path = args.csv_path or config.csv_path
    split_path = args.split_path or config.split_path
    h5_path = args.h5_path or config.h5_path
    output_path = args.output_path or (config.scratch_root / "audits" / "dataset_index.csv")

    csv_path = require_path("csv_path", csv_path)
    split_path = require_path("split_path", split_path)
    h5_path = require_path("h5_path", h5_path)

    labels = pd.read_csv(csv_path)
    verify_required_columns(labels)
    labels["subject"] = labels["subject"].map(path_component)

    splits = read_split_file(split_path)
    merged = labels.merge(splits, on="subject", how="left")
    with_split = merged[merged["split"].notna()].copy()
    rows_missing_split = int(len(merged) - len(with_split))

    import h5py

    h5_paths: list[str | None] = []
    has_image: list[bool] = []
    with h5py.File(h5_path, "r") as h5_file:
        for _, row in with_split.iterrows():
            internal_path = h5_image_path(row)
            h5_paths.append(internal_path)
            has_image.append(internal_path in h5_file)

    with_split["h5_internal_path"] = h5_paths
    with_split["has_h5_image"] = has_image
    usable = with_split[with_split["has_h5_image"]].copy()
    rows_missing_h5_image = int(len(with_split) - len(usable))

    usable = add_binary_label_columns(usable)
    output_columns = [
        "cohort",
        "subject",
        "visit",
        "side",
        "split",
        "h5_internal_path",
        *OSTEOPHYTE_LABEL_COLUMNS,
        *(f"{column}_binary" for column in OSTEOPHYTE_LABEL_COLUMNS),
        "complete_graded",
    ]
    dataset_index = usable[output_columns].copy()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    dataset_index.to_csv(output_path, index=False)

    usable_rows_by_split = value_counts_dict(dataset_index["split"])
    complete_count = int(dataset_index["complete_graded"].sum())
    complete_percentage = (complete_count / len(dataset_index) * 100.0) if len(dataset_index) else 0.0
    complete_by_split: dict[str, dict[str, float | int]] = {}
    for split, split_df in dataset_index.groupby("split", dropna=False):
        split_complete_count = int(split_df["complete_graded"].sum())
        split_total = int(len(split_df))
        complete_by_split[str(split)] = {
            "count": split_complete_count,
            "percentage": (split_complete_count / split_total * 100.0) if split_total else 0.0,
        }

    grade_distributions = {
        column: value_counts_dict(dataset_index[column]) for column in OSTEOPHYTE_LABEL_COLUMNS
    }
    binary_distributions = {
        column: binary_counts_dict(dataset_index[column]) for column in OSTEOPHYTE_LABEL_COLUMNS
    }
    summary: dict[str, Any] = {
        "csv_path": str(csv_path),
        "split_path": str(split_path),
        "h5_path": str(h5_path),
        "output_path": str(output_path),
        "original_csv_rows": int(len(labels)),
        "rows_with_split": int(len(with_split)),
        "rows_missing_split": rows_missing_split,
        "rows_with_h5_image": int(len(dataset_index)),
        "rows_missing_h5_image": rows_missing_h5_image,
        "usable_rows_by_split": usable_rows_by_split,
        "complete_graded_rows_overall": {
            "count": complete_count,
            "percentage": complete_percentage,
        },
        "complete_graded_rows_by_split": complete_by_split,
        "grade_distributions": grade_distributions,
        "binary_distributions": binary_distributions,
    }

    summary_path = output_path.with_name("dataset_index_summary.json")
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print_summary(summary)
    print(f"\nSaved dataset index: {output_path}")
    print(f"Saved JSON summary: {summary_path}")


if __name__ == "__main__":
    main()
