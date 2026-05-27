"""Collect metrics JSON files into one flat CSV."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pandas as pd

from .utils import load_json


OUTPUT_COLUMNS = [
    "seed",
    "strategy",
    "budget_name",
    "budget_size",
    "budget_fraction",
    "mode",
    "split",
    "bmae_mean",
    "quality_mean",
    "qwk_mean",
    "spearman_mean",
    "macro_f1_mean",
    "auc_ge1_mean",
    "auc_ge2_mean",
    "auc_ge3_mean",
    "ap_ge2_mean",
    "ap_ge3_mean",
    "recall_grade0_mean",
    "recall_grade1_mean",
    "recall_grade2_mean",
    "recall_grade3_mean",
    "severe_miss_rate_mean",
]


def collect(
    metrics_root: str | Path,
    checkpoints_root: str | Path | None = None,
    binary_baseline_root: str | Path | None = None,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for path in Path(metrics_root).rglob("metrics_summary.json"):
        row = load_json(path)
        row["_source"] = str(path)
        rows.append(row)
    binary_root = Path(binary_baseline_root) if binary_baseline_root else Path(metrics_root)
    for path in binary_root.rglob("binary_baseline_severity_*_seed*.csv"):
        df_binary = pd.read_csv(path)
        mean = df_binary[df_binary["location"] == "mean"]
        if mean.empty:
            continue
        row = mean.iloc[0].to_dict()
        row.update(
            {
                "strategy": "binary_baseline",
                "mode": "binary_only",
                "budget_name": "0",
                "budget_size": 0,
                "bmae_mean": row.get("bmae"),
                "quality_mean": row.get("quality"),
                "spearman_mean": row.get("spearman_all"),
                "auc_ge1_mean": row.get("auc_ge1"),
                "auc_ge2_mean": row.get("auc_ge2"),
                "auc_ge3_mean": row.get("auc_ge3"),
                "ap_ge2_mean": row.get("ap_ge2"),
                "ap_ge3_mean": row.get("ap_ge3"),
                "_source": str(path),
            }
        )
        rows.append(row)
    if not rows:
        return pd.DataFrame(columns=OUTPUT_COLUMNS)
    df = pd.DataFrame(rows)
    if "budget_size" in df.columns:
        max_full = df.loc[df["budget_name"].astype(str) == "full", "budget_size"]
        denom = float(max_full.max()) if len(max_full) and pd.notna(max_full.max()) else float(df["budget_size"].max())
        df["budget_fraction"] = df["budget_size"].astype(float) / denom if denom > 0 else 0.0
    else:
        df["budget_fraction"] = pd.NA
    for col in OUTPUT_COLUMNS:
        if col not in df.columns:
            df[col] = pd.NA
    return df[OUTPUT_COLUMNS + [c for c in df.columns if c not in OUTPUT_COLUMNS]]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics-root", required=True)
    parser.add_argument("--checkpoints-root", default=None)
    parser.add_argument("--binary-baseline-root", default=None)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    df = collect(args.metrics_root, args.checkpoints_root, args.binary_baseline_root)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"Wrote {len(df)} rows to {out}")


if __name__ == "__main__":
    main()
