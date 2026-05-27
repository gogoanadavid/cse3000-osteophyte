"""NumPy metric implementations without sklearn."""

from __future__ import annotations

import math
from typing import Any

import numpy as np


def _valid(y_true: np.ndarray, score: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    y = np.asarray(y_true)
    s = np.asarray(score, dtype=float)
    mask = (y >= 0) & np.isfinite(s)
    return y[mask], s[mask]


def _rankdata(a: np.ndarray) -> np.ndarray:
    a = np.asarray(a, dtype=float)
    order = np.argsort(a, kind="mergesort")
    ranks = np.empty(len(a), dtype=float)
    i = 0
    while i < len(a):
        j = i + 1
        while j < len(a) and a[order[j]] == a[order[i]]:
            j += 1
        ranks[order[i:j]] = 0.5 * (i + j - 1) + 1.0
        i = j
    return ranks


def balanced_ordinal_mae(y_true: np.ndarray, y_expected: np.ndarray, num_classes: int = 4) -> float:
    y, pred = _valid(y_true, y_expected)
    if y.size == 0:
        return float("nan")
    vals = []
    for grade in range(num_classes):
        m = y == grade
        if m.any():
            vals.append(float(np.abs(pred[m] - grade).mean()))
    return float(np.mean(vals)) if vals else float("nan")


def quality_from_bmae(bmae: float) -> float:
    return float(1.0 - bmae / 3.0) if np.isfinite(bmae) else float("nan")


def quadratic_weighted_kappa(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int = 4) -> float:
    y, p = _valid(y_true, y_pred)
    if y.size == 0:
        return float("nan")
    y = y.astype(int)
    p = np.clip(np.rint(p), 0, num_classes - 1).astype(int)
    conf = np.zeros((num_classes, num_classes), dtype=float)
    for a, b in zip(y, p):
        conf[a, b] += 1.0
    hist_true = conf.sum(axis=1)
    hist_pred = conf.sum(axis=0)
    expected = np.outer(hist_true, hist_pred) / max(conf.sum(), 1.0)
    weights = np.zeros_like(conf)
    for i in range(num_classes):
        for j in range(num_classes):
            weights[i, j] = ((i - j) ** 2) / ((num_classes - 1) ** 2)
    observed = float((weights * conf).sum())
    expected_weighted = float((weights * expected).sum())
    if expected_weighted == 0:
        return 1.0 if observed == 0 else float("nan")
    return float(1.0 - observed / expected_weighted)


def binary_auc(y_true: np.ndarray, score: np.ndarray) -> float:
    y, s = _valid(y_true, score)
    y = y.astype(int)
    if y.size == 0 or len(np.unique(y)) < 2:
        return float("nan")
    ranks = _rankdata(s)
    n_pos = int((y == 1).sum())
    n_neg = int((y == 0).sum())
    rank_sum_pos = float(ranks[y == 1].sum())
    return float((rank_sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def average_precision(y_true: np.ndarray, score: np.ndarray) -> float:
    y, s = _valid(y_true, score)
    y = y.astype(int)
    n_pos = int((y == 1).sum())
    if y.size == 0 or n_pos == 0:
        return float("nan")
    order = np.argsort(-s, kind="mergesort")
    y_sorted = y[order]
    tp = np.cumsum(y_sorted == 1)
    precision = tp / (np.arange(len(y_sorted)) + 1)
    return float((precision * (y_sorted == 1)).sum() / n_pos)


def spearman_corr(y_true: np.ndarray, score: np.ndarray) -> float:
    y, s = _valid(y_true, score)
    if y.size < 2:
        return float("nan")
    ry = _rankdata(y.astype(float))
    rs = _rankdata(s)
    if np.std(ry) == 0 or np.std(rs) == 0:
        return float("nan")
    return float(np.corrcoef(ry, rs)[0, 1])


def macro_f1(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int = 4) -> float:
    y, p = _valid(y_true, y_pred)
    if y.size == 0:
        return float("nan")
    y = y.astype(int)
    p = np.clip(np.rint(p), 0, num_classes - 1).astype(int)
    f1s = []
    for cls in range(num_classes):
        tp = int(((y == cls) & (p == cls)).sum())
        fp = int(((y != cls) & (p == cls)).sum())
        fn = int(((y == cls) & (p != cls)).sum())
        if tp + fp + fn == 0:
            continue
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1s.append(2 * precision * recall / (precision + recall) if precision + recall else 0.0)
    return float(np.mean(f1s)) if f1s else float("nan")


def per_grade_recall(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int = 4) -> dict[int, float]:
    y, p = _valid(y_true, y_pred)
    p = np.clip(np.rint(p), 0, num_classes - 1).astype(int)
    out: dict[int, float] = {}
    for cls in range(num_classes):
        m = y == cls
        out[cls] = float((p[m] == cls).mean()) if m.any() else float("nan")
    return out


def off_by_one_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y, p = _valid(y_true, y_pred)
    if y.size == 0:
        return float("nan")
    p = np.rint(p).astype(int)
    return float((np.abs(y.astype(int) - p) <= 1).mean())


def severe_miss_rate(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y, p = _valid(y_true, y_pred)
    m = y == 3
    if not m.any():
        return float("nan")
    p = np.rint(p[m]).astype(int)
    return float((p <= 1).mean())


def adjacent_auc(y_true: np.ndarray, score: np.ndarray, low_grade: int, high_grade: int) -> float:
    y = np.asarray(y_true)
    s = np.asarray(score, dtype=float)
    mask = ((y == low_grade) | (y == high_grade)) & np.isfinite(s)
    if not mask.any():
        return float("nan")
    binary = (y[mask] == high_grade).astype(int)
    return binary_auc(binary, s[mask])


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int = 4) -> list[list[int]]:
    y, p = _valid(y_true, y_pred)
    mat = np.zeros((num_classes, num_classes), dtype=int)
    for a, b in zip(y.astype(int), np.clip(np.rint(p), 0, num_classes - 1).astype(int)):
        mat[a, b] += 1
    return mat.tolist()


def finite_mean_dict(rows: list[dict[str, Any]], keys: list[str]) -> dict[str, float]:
    out: dict[str, float] = {}
    for key in keys:
        values = np.asarray([row.get(key, np.nan) for row in rows], dtype=float)
        values = values[np.isfinite(values)]
        out[key] = float(values.mean()) if values.size else float("nan")
    return out
