from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

PROJECT_DIR = Path(__file__).resolve().parents[1]


def load_script(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def fake_predictions(pd):
    rows = []
    for location in ("osteo_acet_inf", "osteo_acet_sup"):
        for grade in range(4):
            rows.append(
                {
                    "sample_id": f"s{grade}|00|L",
                    "subject": f"s{grade}",
                    "visit": "00",
                    "side": "L",
                    "split": "val",
                    "location": location,
                    "true_grade": grade,
                    "true_binary": int(grade > 0),
                    "pred_grade": grade,
                    "pred_grade_threshold": grade,
                    "expected_grade": float(grade),
                    "p_gt_0": 0.9 if grade > 0 else 0.1,
                    "p_gt_1": 0.8 if grade > 1 else 0.0,
                    "p_gt_2": 0.7 if grade > 2 else 0.0,
                    "p_grade_0": 1.0 if grade == 0 else 0.0,
                    "p_grade_1": 1.0 if grade == 1 else 0.0,
                    "p_grade_2": 1.0 if grade == 2 else 0.0,
                    "p_grade_3": 1.0 if grade == 3 else 0.0,
                    "p_present": 0.9 if grade > 0 else 0.1,
                }
            )
    return pd.DataFrame(rows)


def test_diagnostics_script_loads_fake_prediction_csv(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    pd = pytest.importorskip("pandas")
    module = load_script(PROJECT_DIR / "scripts" / "09_diagnose_mixed_predictions.py", "diagnose")
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    fake_predictions(pd).to_csv(run_dir / "val_predictions.csv", index=False)

    monkeypatch.setattr(sys, "argv", ["diagnose", "--run-dir", str(run_dir)])
    module.main()

    assert (run_dir / "diagnostic_summary.csv").exists()
    assert (run_dir / "expected_grade_by_true_grade.csv").exists()
    assert (run_dir / "predicted_grade_distribution.csv").exists()


def test_summary_script_aggregates_fake_run_folders(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    pd = pytest.importorskip("pandas")
    pytest.importorskip("matplotlib")
    module = load_script(PROJECT_DIR / "scripts" / "08_summarize_mixed_supervision_runs.py", "summary")

    run_dir = tmp_path / "runs" / "location_binary_mixed_025_random_threshold_independent" / "20260101_000000"
    run_dir.mkdir(parents=True)
    (run_dir / "config.json").write_text(
        json.dumps(
            {
                "experiment_name": "fake",
                "weak_label_mode": "location_binary",
                "effective_strong_fraction": 0.25,
                "strong_count": 10,
                "weak_count": 30,
                "strong_sampling_strategy": "random",
                "model_head": "threshold_independent",
                "init_from_binary_checkpoint": None,
                "loss_balance_mode": "proportional",
                "ordinal_class_weighting": "none",
                "seed": 123,
                "selection_metric": "mean_spearman",
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "metrics_history.json").write_text(
        json.dumps(
            [
                {
                    "epoch": 1,
                    "selection_value": 0.5,
                    "val": {
                        "mean": {
                            "spearman": 0.5,
                            "mae_expected": 0.7,
                            "qwk": 0.2,
                            "binary_auroc": 0.8,
                        }
                    },
                }
            ]
        ),
        encoding="utf-8",
    )
    pd.DataFrame(
        [
            {"location": "osteo_acet_inf", "grade": 3, "count": 1},
            {"location": "osteo_acet_sup", "grade": 3, "count": 2},
            {"location": "osteo_fem_inf", "grade": 3, "count": 3},
            {"location": "osteo_fem_sup", "grade": 3, "count": 4},
        ]
    ).to_csv(run_dir / "strong_grade_distribution_by_location.csv", index=False)

    output_dir = tmp_path / "summary"
    monkeypatch.setattr(sys, "argv", ["summary", "--runs-root", str(tmp_path / "runs"), "--output-dir", str(output_dir)])
    module.main()

    summary_csv = output_dir / "annotation_budget_summary.csv"
    assert summary_csv.exists()
    summary = pd.read_csv(summary_csv)
    assert summary.iloc[0]["mean_spearman"] == pytest.approx(0.5)
