"""Verify nested budget CSVs and summarize composition."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pandas as pd

from .utils import save_json


def _budget_sort_key(path: Path) -> tuple[int, str]:
    name = path.stem.removeprefix("budget_")
    if name == "full":
        return (10**18, name)
    return (int(name), name)


def verify_budget_root(index_csv: str | Path, budget_root: str | Path) -> dict[str, Any]:
    index = pd.read_csv(index_csv)
    train = index[index["split"].astype(str) == "train"].copy()
    root = Path(budget_root)
    files = sorted(root.glob("budget_*.csv"), key=_budget_sort_key)
    if not files:
        raise FileNotFoundError(f"No budget_*.csv files found under {root}")
    locations = [c.removeprefix("grade_") for c in train.columns if c.startswith("grade_")]
    previous: set[int] = set()
    result: dict[str, Any] = {"budget_root": str(root), "budgets": {}, "all_nested": True}
    for path in files:
        name = path.stem.removeprefix("budget_")
        budget = pd.read_csv(path)
        selected = {int(x) for x in budget["h5_index"].tolist()}
        nested = previous.issubset(selected)
        result["all_nested"] = bool(result["all_nested"] and nested)
        sdf = train[train["h5_index"].astype(int).isin(selected)]
        info: dict[str, Any] = {
            "file": str(path),
            "n": int(len(selected)),
            "unique_n": int(len(selected)),
            "nested_with_previous": bool(nested),
            "missing_from_train": int(len(selected - set(train["h5_index"].astype(int).tolist()))),
            "any_binary_positive": int(sdf["any_binary_positive"].sum()) if len(sdf) else 0,
            "max_grade": {str(k): int(v) for k, v in sdf["max_grade"].value_counts().sort_index().items()},
            "grade_ge1": int((sdf["max_grade"] >= 1).sum()),
            "grade_ge2": int((sdf["max_grade"] >= 2).sum()),
            "grade_ge3": int((sdf["max_grade"] >= 3).sum()),
            "locations": {},
        }
        for loc in locations:
            info["locations"][loc] = {
                "grade_counts": {str(k): int(v) for k, v in sdf[f"grade_{loc}"].value_counts().sort_index().items()},
                "ge1": int((sdf[f"grade_{loc}"] >= 1).sum()),
                "ge2": int((sdf[f"grade_{loc}"] >= 2).sum()),
                "ge3": int((sdf[f"grade_{loc}"] >= 3).sum()),
            }
        result["budgets"][name] = info
        previous = selected
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", required=True)
    parser.add_argument("--budget-root", required=True)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()
    result = verify_budget_root(args.index, args.budget_root)
    out = Path(args.out) if args.out else Path(args.budget_root) / "verification.json"
    save_json(result, out)
    print(f"all_nested={result['all_nested']}")
    for name, info in result["budgets"].items():
        print(
            f"{name}: n={info['n']} nested={info['nested_with_previous']} "
            f"ge1={info['grade_ge1']} ge2={info['grade_ge2']} ge3={info['grade_ge3']}"
        )


if __name__ == "__main__":
    main()
