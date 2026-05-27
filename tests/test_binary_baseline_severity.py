from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.evaluate_binary_baseline_severity import evaluate_binary_baseline


def test_binary_baseline_uses_p_ge1_only(tmp_path: Path) -> None:
    locations = ["sup_acet", "inf_acet", "sup_fem", "inf_fem"]
    index_rows = []
    pred_rows = []
    for i, grade in enumerate([0, 1, 2, 3]):
        index_row = {"h5_index": i, "split": "val"}
        pred_row = {"h5_index": i}
        for loc in locations:
            index_row[f"grade_{loc}"] = grade
            pred_row[f"{loc}_p_ge1"] = float(grade) / 3.0
            pred_row[f"{loc}_p_ge2"] = 1.0 - float(grade) / 3.0
            pred_row[f"{loc}_p_ge3"] = 1.0 - float(grade) / 3.0
        index_rows.append(index_row)
        pred_rows.append(pred_row)
    index = tmp_path / "index.csv"
    pred = tmp_path / "pred.csv"
    pd.DataFrame(index_rows).to_csv(index, index=False)
    pd.DataFrame(pred_rows).to_csv(pred, index=False)
    df = evaluate_binary_baseline({"locations": locations}, index, pred, "val", seed=0)
    mean = df[df["location"] == "mean"].iloc[0]
    assert mean["auc_ge2"] == 1.0
    assert mean["adjacent_auc_1v2"] == 1.0
    assert mean["adjacent_auc_2v3"] == 1.0


if __name__ == "__main__":
    import tempfile

    with tempfile.TemporaryDirectory() as d:
        test_binary_baseline_uses_p_ge1_only(Path(d))
