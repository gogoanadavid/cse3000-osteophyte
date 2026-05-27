from __future__ import annotations

import numpy as np

from src.metrics import (
    average_precision,
    balanced_ordinal_mae,
    binary_auc,
    macro_f1,
    off_by_one_accuracy,
    quality_from_bmae,
    quadratic_weighted_kappa,
    severe_miss_rate,
    spearman_corr,
)


def test_balanced_ordinal_mae_and_quality() -> None:
    y = np.array([0, 1, 2, 3])
    pred = np.array([0.0, 1.0, 2.5, 2.0])
    bmae = balanced_ordinal_mae(y, pred)
    assert np.isclose(bmae, 0.375)
    assert np.isclose(quality_from_bmae(bmae), 0.875)


def test_binary_auc_and_ap() -> None:
    y = np.array([0, 0, 1, 1])
    score = np.array([0.1, 0.2, 0.8, 0.9])
    assert np.isclose(binary_auc(y, score), 1.0)
    assert np.isclose(average_precision(y, score), 1.0)


def test_other_metrics_basic() -> None:
    y = np.array([0, 1, 2, 3])
    p = np.array([0, 1, 2, 1])
    assert quadratic_weighted_kappa(y, p) <= 1.0
    assert spearman_corr(y, p) > 0.0
    assert macro_f1(y, p) >= 0.0
    assert off_by_one_accuracy(y, p) == 0.75
    assert severe_miss_rate(y, p) == 1.0


if __name__ == "__main__":
    test_balanced_ordinal_mae_and_quality()
    test_binary_auc_and_ap()
    test_other_metrics_basic()
