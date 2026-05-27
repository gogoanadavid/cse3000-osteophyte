#!/usr/bin/env python3
"""Audit osteophyte label columns and scaffold future split checks."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from osteophytes.config import load_config
from osteophytes.labels import OSTEOPHYTE_LABEL_COLUMNS, grade_to_binary, is_complete_graded_annotation


CONFIG_PATH = PROJECT_DIR / "configs" / "delftblue.yaml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv-path", required=True, type=Path, help="Path to the label CSV.")
    parser.add_argument("--split-path", type=Path, help="Optional split/list file to inspect later.")
    return parser.parse_args()


def value_counts_dict(series: pd.Series) -> dict[str, int]:
    counts = series.value_counts(dropna=False).sort_index()
    return {str(key): int(value) for key, value in counts.items()}


def binary_counts(series: pd.Series) -> dict[str, int]:
    binary = series.map(grade_to_binary)
    counts = binary.value_counts(dropna=False).sort_index()
    return {str(key): int(value) for key, value in counts.items()}


def main() -> None:
    args = parse_args()
    config = load_config(CONFIG_PATH)

    df = pd.read_csv(args.csv_path)
    print(f"Loaded CSV: {args.csv_path}")
    print(f"Rows: {len(df)}")
    print("Available columns:")
    for column in df.columns:
        print(f"  {column}")

    missing_columns = [column for column in OSTEOPHYTE_LABEL_COLUMNS if column not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing osteophyte label columns: {', '.join(missing_columns)}")

    grade_distributions: dict[str, dict[str, int]] = {}
    binary_distributions: dict[str, dict[str, int]] = {}
    for column in OSTEOPHYTE_LABEL_COLUMNS:
        grade_distributions[column] = value_counts_dict(df[column])
        binary_distributions[column] = binary_counts(df[column])

        print(f"\n{column}")
        print(f"  grade distribution: {grade_distributions[column]}")
        print(f"  binary distribution: {binary_distributions[column]}")

    complete_mask = df.apply(is_complete_graded_annotation, axis=1)
    complete_count = int(complete_mask.sum())
    complete_percentage = (complete_count / len(df) * 100.0) if len(df) else 0.0
    print("\nComplete graded annotations")
    print(f"  count: {complete_count}")
    print(f"  percentage: {complete_percentage:.2f}%")

    # TODO: Identify the subject/exam/image ID columns once the CSV schema is confirmed.
    # TODO: Validate split membership and leakage once the split file format is known.
    if args.split_path:
        print(f"\nSplit path provided for future validation: {args.split_path}")

    summary: dict[str, Any] = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "csv_path": str(args.csv_path),
        "split_path": str(args.split_path) if args.split_path else None,
        "row_count": int(len(df)),
        "columns": list(df.columns),
        "grade_distributions": grade_distributions,
        "binary_distributions": binary_distributions,
        "complete_graded_annotations": {
            "count": complete_count,
            "percentage": complete_percentage,
        },
        "todos": [
            "Identify exact subject/exam/image ID columns.",
            "Validate split membership and leakage once split format is known.",
        ],
    }

    audit_dir = config.scratch_root / "audits"
    audit_dir.mkdir(parents=True, exist_ok=True)
    output_path = audit_dir / f"label_audit_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.json"
    output_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nSaved audit summary: {output_path}")


if __name__ == "__main__":
    main()
