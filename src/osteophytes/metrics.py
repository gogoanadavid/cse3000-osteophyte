"""Robust metrics for binary and ordinal osteophyte experiments."""

from __future__ import annotations

import math
from collections.abc import Iterable
from typing import Any

import numpy as np

from osteophytes.labels import LOCATION_NAMES, NUM_GRADES


def _to_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def finite_nanmean(values: Iterable[float]) -> float:
    finite_values = [float(value) for value in values if math.isfinite(float(value))]
    if not finite_values:
        return math.nan
    return float(sum(finite_values) / len(finite_values))


def nanmean_metric(metric_dict: dict[str, float]) -> float:
    return finite_nanmean(metric_dict.values())


def average_ranks(values: Any) -> np.ndarray:
    values = np.asarray(values)
    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    ranks = np.empty(values.size, dtype=float)
    start = 0
    while start < values.size:
        end = start + 1
        while end < values.size and sorted_values[end] == sorted_values[start]:
            end += 1
        ranks[order[start:end]] = (start + end - 1) / 2.0 + 1.0
        start = end
    return ranks


def spearman_correlation(y_true: Any, scores: Any) -> float:
    y_true = np.asarray(y_true, dtype=float)
    scores = np.asarray(scores, dtype=float)
    valid = np.isfinite(y_true) & np.isfinite(scores)
    y_true = y_true[valid]
    scores = scores[valid]
    if y_true.size < 3 or np.unique(y_true).size < 2 or np.unique(scores).size < 2:
        return math.nan
    true_ranks = average_ranks(y_true)
    score_ranks = average_ranks(scores)
    true_centered = true_ranks - true_ranks.mean()
    score_centered = score_ranks - score_ranks.mean()
    denominator = np.sqrt((true_centered**2).sum() * (score_centered**2).sum())
    if denominator <= 0:
        return math.nan
    return float((true_centered * score_centered).sum() / denominator)


def binary_auroc(y_true: Any, y_score: Any) -> float:
    y_true = np.asarray(y_true, dtype=int)
    y_score = np.asarray(y_score, dtype=float)
    valid = np.isfinite(y_score)
    y_true = y_true[valid]
    y_score = y_score[valid]
    positives = y_true == 1
    negatives = y_true == 0
    n_pos = int(positives.sum())
    n_neg = int(negatives.sum())
    if n_pos == 0 or n_neg == 0:
        return math.nan
    ranks = average_ranks(y_score)
    pos_rank_sum = float(ranks[positives].sum())
    return float((pos_rank_sum - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def binary_auprc(y_true: Any, y_score: Any) -> float:
    y_true = np.asarray(y_true, dtype=int)
    y_score = np.asarray(y_score, dtype=float)
    valid = np.isfinite(y_score)
    y_true = y_true[valid]
    y_score = y_score[valid]
    n_pos = int((y_true == 1).sum())
    if n_pos == 0 or n_pos == y_true.size:
        return math.nan
    order = np.argsort(-y_score, kind="mergesort")
    sorted_true = y_true[order]
    tp = np.cumsum(sorted_true == 1)
    fp = np.cumsum(sorted_true == 0)
    precision = tp / np.maximum(tp + fp, 1)
    return float(precision[sorted_true == 1].sum() / n_pos)


def quadratic_weighted_kappa(y_true: Any, y_pred: Any, num_grades: int = NUM_GRADES) -> float:
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    if y_true.size == 0 or np.unique(y_true).size < 2 or np.unique(y_pred).size < 2:
        return math.nan
    observed = np.zeros((num_grades, num_grades), dtype=float)
    for true_grade, pred_grade in zip(y_true, y_pred, strict=False):
        if 0 <= true_grade < num_grades and 0 <= pred_grade < num_grades:
            observed[true_grade, pred_grade] += 1.0
    total = observed.sum()
    if total <= 0:
        return math.nan
    true_hist = observed.sum(axis=1)
    pred_hist = observed.sum(axis=0)
    expected = np.outer(true_hist, pred_hist) / total
    weights = np.zeros_like(observed)
    denom = float((num_grades - 1) ** 2)
    for i in range(num_grades):
        for j in range(num_grades):
            weights[i, j] = ((i - j) ** 2) / denom
    expected_weighted = float((weights * expected).sum())
    if expected_weighted <= 0:
        return math.nan
    return float(1.0 - ((weights * observed).sum() / expected_weighted))


def macro_f1_score(y_true: Any, y_pred: Any, classes: tuple[int, ...] = (0, 1, 2, 3)) -> float:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    f1_values: list[float] = []
    for class_id in classes:
        true_positive = int(((y_true == class_id) & (y_pred == class_id)).sum())
        false_positive = int(((y_true != class_id) & (y_pred == class_id)).sum())
        false_negative = int(((y_true == class_id) & (y_pred != class_id)).sum())
        denominator = 2 * true_positive + false_positive + false_negative
        f1_values.append((2 * true_positive / denominator) if denominator else 0.0)
    return float(sum(f1_values) / len(f1_values))


def confusion_matrix_4class(y_true: Any, y_pred: Any) -> list[list[int]]:
    matrix = [[0 for _ in range(NUM_GRADES)] for _ in range(NUM_GRADES)]
    for true_grade, pred_grade in zip(y_true, y_pred, strict=False):
        true_i = int(true_grade)
        pred_i = int(pred_grade)
        if 0 <= true_i < NUM_GRADES and 0 <= pred_i < NUM_GRADES:
            matrix[true_i][pred_i] += 1
    return matrix


def per_grade_recall(y_true: Any, y_pred: Any) -> dict[str, float]:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    recalls: dict[str, float] = {}
    for grade in range(NUM_GRADES):
        mask = y_true == grade
        recalls[str(grade)] = float((y_pred[mask] == grade).mean()) if mask.any() else math.nan
    return recalls


def masked_binary_loss(logits: Any, labels: Any, mask: Any) -> Any:
    """Backward-compatible torch loss used by the original baseline script."""
    import torch.nn.functional as F

    losses = F.binary_cross_entropy_with_logits(logits, labels, reduction="none")
    mask = mask.to(dtype=losses.dtype)
    valid_count = mask.sum()
    if float(valid_count.detach().cpu()) <= 0.0:
        raise ValueError("Cannot compute masked binary loss with zero valid entries.")
    return (losses * mask).sum() / valid_count


def compute_binary_auc_per_location(
    y_true: Any,
    y_prob: Any,
    mask: Any,
    location_names: Iterable[str] = LOCATION_NAMES,
) -> dict[str, float]:
    y_true_np = _to_numpy(y_true)
    y_prob_np = _to_numpy(y_prob)
    mask_np = _to_numpy(mask)

    metrics: dict[str, float] = {}
    for index, location in enumerate(location_names):
        valid = mask_np[:, index] == 1
        valid &= np.isfinite(y_true_np[:, index])
        valid &= np.isfinite(y_prob_np[:, index])
        metrics[location] = binary_auroc(y_true_np[valid, index], y_prob_np[valid, index])
    return metrics


def compute_binary_auprc_per_location(
    y_true: Any,
    y_prob: Any,
    mask: Any,
    location_names: Iterable[str] = LOCATION_NAMES,
) -> dict[str, float]:
    y_true_np = _to_numpy(y_true)
    y_prob_np = _to_numpy(y_prob)
    mask_np = _to_numpy(mask)

    metrics: dict[str, float] = {}
    for index, location in enumerate(location_names):
        valid = mask_np[:, index] == 1
        valid &= np.isfinite(y_true_np[:, index])
        valid &= np.isfinite(y_prob_np[:, index])
        metrics[location] = binary_auprc(y_true_np[valid, index], y_prob_np[valid, index])
    return metrics


def compute_spearman_per_location(
    grades: Any,
    scores: Any,
    graded_mask: Any,
    location_names: Iterable[str] = LOCATION_NAMES,
) -> dict[str, float]:
    grades_np = _to_numpy(grades)
    scores_np = _to_numpy(scores)
    mask_np = _to_numpy(graded_mask)

    metrics: dict[str, float] = {}
    for index, location in enumerate(location_names):
        valid = mask_np[:, index] == 1
        valid &= np.isfinite(grades_np[:, index])
        valid &= np.isfinite(scores_np[:, index])
        metrics[location] = spearman_correlation(grades_np[valid, index], scores_np[valid, index])
    return metrics


def compute_metrics_by_location(
    true_grades: Any,
    pred_grades: Any,
    expected_grades: Any,
    p_present: Any,
    graded_mask: Any,
    location_names: tuple[str, ...] = LOCATION_NAMES,
) -> dict[str, Any]:
    true_np = _to_numpy(true_grades)
    pred_np = _to_numpy(pred_grades)
    expected_np = _to_numpy(expected_grades)
    p_present_np = _to_numpy(p_present)
    mask_np = _to_numpy(graded_mask)

    per_location: dict[str, Any] = {}
    for location_index, location in enumerate(location_names):
        valid = mask_np[:, location_index] == 1
        valid &= np.isfinite(true_np[:, location_index])
        valid &= np.isfinite(expected_np[:, location_index])
        y_true = true_np[valid, location_index].astype(int)
        y_pred = pred_np[valid, location_index].astype(int)
        y_expected = expected_np[valid, location_index].astype(float)
        y_present = (y_true > 0).astype(int)
        location_p_present = p_present_np[valid, location_index].astype(float)

        if y_true.size == 0:
            per_location[location] = {
                "spearman": math.nan,
                "mae_hard": math.nan,
                "mae_expected": math.nan,
                "qwk": math.nan,
                "binary_auroc": math.nan,
                "binary_auprc": math.nan,
                "accuracy": math.nan,
                "macro_f1": math.nan,
                "confusion_matrix": [[0 for _ in range(NUM_GRADES)] for _ in range(NUM_GRADES)],
                "per_grade_recall": {str(grade): math.nan for grade in range(NUM_GRADES)},
                "n": 0,
            }
            continue

        per_location[location] = {
            "spearman": spearman_correlation(y_true, y_expected),
            "mae_hard": float(np.abs(y_true - y_pred).mean()),
            "mae_expected": float(np.abs(y_true - y_expected).mean()),
            "qwk": quadratic_weighted_kappa(y_true, y_pred),
            "binary_auroc": binary_auroc(y_present, location_p_present),
            "binary_auprc": binary_auprc(y_present, location_p_present),
            "accuracy": float((y_true == y_pred).mean()),
            "macro_f1": macro_f1_score(y_true, y_pred),
            "confusion_matrix": confusion_matrix_4class(y_true, y_pred),
            "per_grade_recall": per_grade_recall(y_true, y_pred),
            "n": int(y_true.size),
        }

    mean_metrics = {
        "spearman": finite_nanmean(metrics["spearman"] for metrics in per_location.values()),
        "mae_hard": finite_nanmean(metrics["mae_hard"] for metrics in per_location.values()),
        "mae_expected": finite_nanmean(metrics["mae_expected"] for metrics in per_location.values()),
        "qwk": finite_nanmean(metrics["qwk"] for metrics in per_location.values()),
        "binary_auroc": finite_nanmean(metrics["binary_auroc"] for metrics in per_location.values()),
        "binary_auprc": finite_nanmean(metrics["binary_auprc"] for metrics in per_location.values()),
        "accuracy": finite_nanmean(metrics["accuracy"] for metrics in per_location.values()),
        "macro_f1": finite_nanmean(metrics["macro_f1"] for metrics in per_location.values()),
    }
    return {"per_location": per_location, "mean": mean_metrics}
