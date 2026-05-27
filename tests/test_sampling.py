from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.sampling import make_budget_selection


def _index_csv(tmp_path: Path) -> Path:
    rows = []
    for i in range(40):
        grade = i % 4
        rows.append(
            {
                "h5_index": i,
                "split": "train",
                "subject_id": f"s{i}",
                "visit_id": "v0",
                "side": "L",
                "binary_sup_acet": int(grade > 0),
                "binary_inf_acet": int(grade > 0),
                "binary_sup_fem": int(grade > 0),
                "binary_inf_fem": int(grade > 0),
                "grade_sup_acet": grade,
                "grade_inf_acet": grade,
                "grade_sup_fem": grade,
                "grade_inf_fem": grade,
                "has_complete_grades": 1,
                "max_grade": grade,
                "any_binary_positive": int(grade > 0),
            }
        )
    path = tmp_path / "index.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def test_random_budgets_are_nested(tmp_path: Path) -> None:
    index = _index_csv(tmp_path)
    selections, manifest = make_budget_selection(index, "random", seed=0, budgets=["0", "4", "8", "full"])
    assert set(selections["0"]).issubset(selections["4"])
    assert set(selections["4"]).issubset(selections["8"])
    assert set(selections["8"]).issubset(selections["full"])
    assert all(manifest["nested"].values())


def test_oracle_manifest_is_labeled(tmp_path: Path) -> None:
    index = _index_csv(tmp_path)
    _, manifest = make_budget_selection(index, "oracle_grade_stratified", seed=0, budgets=["4", "full"])
    assert manifest["strategy"] == "oracle_grade_stratified"
    assert manifest["oracle_only"] is True


if __name__ == "__main__":
    import tempfile

    with tempfile.TemporaryDirectory() as d:
        test_random_budgets_are_nested(Path(d))
        test_oracle_manifest_is_labeled(Path(d))
