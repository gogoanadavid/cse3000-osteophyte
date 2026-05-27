"""Create nested graded annotation budget subsets."""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .utils import save_json, timestamp

DEFAULT_BUDGETS = ["0", "64", "128", "256", "512", "1024", "2048", "4096", "8192", "full"]


def _locations(df: pd.DataFrame) -> list[str]:
    return [c.removeprefix("grade_") for c in df.columns if c.startswith("grade_")]


def _budget_sizes(budgets: list[str], full_size: int) -> list[tuple[str, int]]:
    out = []
    for name in budgets:
        if str(name).lower() == "full":
            out.append(("full", full_size))
        else:
            out.append((str(name), min(int(name), full_size)))
    out = sorted(out, key=lambda x: x[1])
    return out


def _shuffle(values: list[int], rng: np.random.Generator) -> list[int]:
    arr = np.asarray(values, dtype=int)
    rng.shuffle(arr)
    return arr.tolist()


def _append_until(selected: list[int], pool: list[int], target: int, used: set[int]) -> None:
    for item in pool:
        if len(selected) >= target:
            return
        if item not in used:
            selected.append(item)
            used.add(item)


def _incremental_two_pool(
    sizes: list[tuple[str, int]],
    pos_pool: list[int],
    neg_pool: list[int],
    pos_fraction: float,
) -> dict[str, list[int]]:
    selected: list[int] = []
    used: set[int] = set()
    out: dict[str, list[int]] = {}
    for name, size in sizes:
        if size == len(pos_pool) + len(neg_pool):
            merged = selected + [x for x in pos_pool + neg_pool if x not in used]
            out[name] = merged[:size]
            selected = out[name].copy()
            used = set(selected)
            continue
        desired_pos = min(len(pos_pool), int(round(pos_fraction * size)))
        current_pos = sum(1 for x in selected if x in set(pos_pool))
        _append_until(selected, pos_pool, len(selected) + max(0, desired_pos - current_pos), used)
        _append_until(selected, neg_pool, size, used)
        _append_until(selected, pos_pool, size, used)
        out[name] = selected[:size].copy()
    return out


def make_budget_selection(
    index_csv: str | Path,
    strategy: str,
    seed: int,
    budgets: list[str] | None = None,
    scores_csv: str | Path | None = None,
    score_column: str = "severity_proxy",
) -> tuple[dict[str, list[int]], dict[str, Any]]:
    df = pd.read_csv(index_csv)
    train = df[(df["split"].astype(str) == "train") & (df["has_complete_grades"] == 1)].copy()
    if train.empty:
        raise ValueError("No train samples with complete grades are available for budget creation")
    rng = np.random.default_rng(seed)
    budgets = budgets or DEFAULT_BUDGETS
    sizes = _budget_sizes([str(b) for b in budgets], len(train))
    h5_indices = train["h5_index"].astype(int).tolist()

    if strategy == "random":
        order = _shuffle(h5_indices, rng)
        selections = {name: order[:size] for name, size in sizes}
    elif strategy == "binary_positive_enriched":
        pos = _shuffle(train.loc[train["any_binary_positive"] == 1, "h5_index"].astype(int).tolist(), rng)
        neg = _shuffle(train.loc[train["any_binary_positive"] == 0, "h5_index"].astype(int).tolist(), rng)
        selections = _incremental_two_pool(sizes, pos, neg, pos_fraction=0.88)
    elif strategy == "score_stratified":
        if scores_csv is None:
            raise ValueError("score_stratified requires --scores")
        scores = pd.read_csv(scores_csv)[["h5_index", score_column]].copy()
        merged = train.merge(scores, on="h5_index", how="left")
        if merged[score_column].isna().any():
            missing = int(merged[score_column].isna().sum())
            raise ValueError(f"Scores missing for {missing} complete train samples")
        pos_df = merged[merged["any_binary_positive"] == 1].copy()
        neg = _shuffle(merged.loc[merged["any_binary_positive"] == 0, "h5_index"].astype(int).tolist(), rng)
        if len(pos_df) > 0:
            pos_df["bin"] = pd.qcut(pos_df[score_column].rank(method="first"), q=min(5, len(pos_df)), labels=False)
        bins = []
        for _, bdf in pos_df.groupby("bin"):
            bins.append(_shuffle(bdf["h5_index"].astype(int).tolist(), rng))
        pos_order = []
        while any(bins):
            for pool in bins:
                if pool:
                    pos_order.append(pool.pop(0))
        selections = _incremental_two_pool(sizes, pos_order, neg, pos_fraction=0.85)
    elif strategy == "oracle_grade_stratified":
        grade_pools: dict[int, list[int]] = {}
        for grade in [3, 2, 1, 0]:
            grade_pools[grade] = _shuffle(train.loc[train["max_grade"] == grade, "h5_index"].astype(int).tolist(), rng)
        shares = {3: 0.25, 2: 0.35, 1: 0.25, 0: 0.15}
        selected: list[int] = []
        used: set[int] = set()
        selections = {}
        for name, size in sizes:
            if size == len(train):
                merged = selected + [x for g in [3, 2, 1, 0] for x in grade_pools[g] if x not in used]
                selections[name] = merged[:size]
                selected = selections[name].copy()
                used = set(selected)
                continue
            for grade in [3, 2, 1, 0]:
                desired = min(len(grade_pools[grade]), int(round(shares[grade] * size)))
                current = sum(1 for x in selected if x in set(grade_pools[grade]))
                _append_until(selected, grade_pools[grade], len(selected) + max(0, desired - current), used)
            for grade in [3, 2, 1, 0]:
                _append_until(selected, grade_pools[grade], size, used)
            selections[name] = selected[:size].copy()
    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    manifest = {
        "seed": int(seed),
        "strategy": strategy,
        "oracle_only": strategy == "oracle_grade_stratified",
        "timestamp": timestamp(),
        "requested_budgets": [str(b) for b in budgets],
        "actual_budget_sizes": {name: len(values) for name, values in selections.items()},
        "nested": {},
        "counts": {},
    }
    prev_name = None
    prev_set: set[int] = set()
    locations = _locations(train)
    for name, values in selections.items():
        cur = set(values)
        manifest["nested"][name] = bool(prev_name is None or prev_set.issubset(cur))
        prev_name = name
        prev_set = cur
        sdf = train[train["h5_index"].isin(cur)]
        counts: dict[str, Any] = {
            "n": int(len(sdf)),
            "any_binary_positive": int(sdf["any_binary_positive"].sum()) if len(sdf) else 0,
            "max_grade": {str(k): int(v) for k, v in sdf["max_grade"].value_counts().sort_index().items()},
            "locations": {},
        }
        for loc in locations:
            counts["locations"][loc] = {str(k): int(v) for k, v in sdf[f"grade_{loc}"].value_counts().sort_index().items()}
        manifest["counts"][name] = counts
    return selections, manifest


def write_budget_files(selections: dict[str, list[int]], manifest: dict[str, Any], out_dir: str | Path) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, values in selections.items():
        pd.DataFrame({"h5_index": values, "selected_for_grading": 1}).to_csv(out_dir / f"budget_{name}.csv", index=False)
    save_json(manifest, out_dir / "manifest.json")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", required=True)
    parser.add_argument("--strategy", required=True, choices=["random", "binary_positive_enriched", "score_stratified", "oracle_grade_stratified"])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--budgets", default=",".join(DEFAULT_BUDGETS))
    parser.add_argument("--scores", default=None)
    parser.add_argument("--score-column", default="severity_proxy")
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()
    budgets = [x.strip() for x in args.budgets.split(",") if x.strip()]
    selections, manifest = make_budget_selection(
        args.index,
        strategy=args.strategy,
        seed=args.seed,
        budgets=budgets,
        scores_csv=args.scores,
        score_column=args.score_column,
    )
    write_budget_files(selections, manifest, args.out_dir)
    print(f"Wrote {len(selections)} nested budget files to {args.out_dir}")


if __name__ == "__main__":
    main()
