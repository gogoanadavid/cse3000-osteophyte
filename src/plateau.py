"""Plateau analysis for graded-budget learning curves."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .utils import save_json


def _as_bool(value: str) -> bool:
    return str(value).lower() in {"1", "true", "yes", "y"}


def analyze_plateau(
    results_csv: str | Path,
    full_budget_name: str,
    primary_metric: str,
    higher_is_better: bool,
    noninferiority_margin_quality: float,
    doubling_improvement_margin: float,
) -> dict[str, Any]:
    df = pd.read_csv(results_csv)
    out: dict[str, Any] = {"groups": []}
    for (strategy, mode), g in df.groupby(["strategy", "mode"]):
        g = g.copy()
        g[primary_metric] = pd.to_numeric(g[primary_metric], errors="coerce")
        summary = g.groupby(["budget_name", "budget_size"], dropna=False)[primary_metric].mean().reset_index()
        full = summary[summary["budget_name"].astype(str) == str(full_budget_name)]
        if full.empty:
            continue
        full_metric = float(full[primary_metric].iloc[0])
        candidates = []
        for _, row in summary.iterrows():
            name = str(row["budget_name"])
            size = int(row["budget_size"]) if pd.notna(row["budget_size"]) else -1
            metric = float(row[primary_metric])
            if not np.isfinite(metric) or size < 0:
                continue
            within = (full_metric - metric) <= noninferiority_margin_quality if higher_is_better else (metric - full_metric) <= noninferiority_margin_quality
            doubled = summary[summary["budget_size"] >= size * 2].sort_values("budget_size").head(1)
            doubling_ok = True
            if size > 0 and not doubled.empty:
                next_metric = float(doubled[primary_metric].iloc[0])
                improvement = next_metric - metric if higher_is_better else metric - next_metric
                doubling_ok = improvement < doubling_improvement_margin
            if within and doubling_ok:
                candidates.append({"budget_name": name, "budget_size": size, "metric": metric})
        plateau = sorted(candidates, key=lambda x: x["budget_size"])[0] if candidates else None
        out["groups"].append(
            {
                "strategy": strategy,
                "mode": mode,
                "full_metric": full_metric,
                "plateau": plateau,
                "candidates": candidates,
            }
        )
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", required=True)
    parser.add_argument("--full-budget-name", default="full")
    parser.add_argument("--primary-metric", default="quality_mean")
    parser.add_argument("--higher-is-better", default="true")
    parser.add_argument("--noninferiority-margin-quality", type=float, default=0.01)
    parser.add_argument("--doubling-improvement-margin", type=float, default=0.005)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    result = analyze_plateau(
        args.results,
        args.full_budget_name,
        args.primary_metric,
        _as_bool(args.higher_is_better),
        args.noninferiority_margin_quality,
        args.doubling_improvement_margin,
    )
    out = Path(args.out)
    save_json(result, out)
    lines = []
    for group in result["groups"]:
        plateau = group["plateau"]
        if plateau:
            lines.append(
                f"{group['mode']}/{group['strategy']}: plateau at {plateau['budget_name']} "
                f"({plateau['metric']:.4f}; full {group['full_metric']:.4f})"
            )
        else:
            lines.append(f"{group['mode']}/{group['strategy']}: no plateau found")
    text_path = out.with_suffix(".txt")
    text_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
