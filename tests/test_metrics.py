from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


def test_metric_helpers_and_selection_direction() -> None:
    pytest.importorskip("numpy")
    pytest.importorskip("torch")

    from osteophytes.metrics import binary_auroc, quadratic_weighted_kappa, spearman_correlation
    from osteophytes.training import metric_is_better, select_best_metric

    assert binary_auroc([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9]) == pytest.approx(1.0)
    assert spearman_correlation([0, 1, 2, 3], [0.1, 0.2, 0.3, 0.4]) == pytest.approx(1.0)
    assert quadratic_weighted_kappa([0, 1, 2, 3], [0, 1, 2, 3]) == pytest.approx(1.0)

    val_metrics = {"mean": {"spearman": 0.5, "mae_expected": 0.4, "qwk": 0.3, "binary_auroc": 0.8}}
    assert select_best_metric(val_metrics, "mean_spearman") == (0.5, "higher")
    assert select_best_metric(val_metrics, "mean_mae") == (0.4, "lower")
    assert metric_is_better(0.6, 0.5, "higher")
    assert metric_is_better(0.3, 0.4, "lower")
