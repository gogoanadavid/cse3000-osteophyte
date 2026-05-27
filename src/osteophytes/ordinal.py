"""Ordinal encoding and probability utilities for OARSI grades."""

from __future__ import annotations

from typing import Any

from osteophytes.labels import NUM_GRADES, NUM_LOCATIONS, NUM_THRESHOLDS


def reshape_threshold_logits(logits: Any) -> Any:
    """Return logits shaped [batch, 4 locations, 3 thresholds]."""
    if logits.ndim == 3 and logits.shape[1] == NUM_LOCATIONS and logits.shape[2] == NUM_THRESHOLDS:
        return logits
    if logits.ndim == 2 and logits.shape[1] == NUM_LOCATIONS * NUM_THRESHOLDS:
        return logits.view(logits.shape[0], NUM_LOCATIONS, NUM_THRESHOLDS)
    raise ValueError(
        "Expected threshold logits shaped [batch, 12] or [batch, 4, 3], "
        f"got {tuple(logits.shape)}"
    )


def encode_ordinal_thresholds(grades: Any, mask: Any | None = None) -> tuple[Any, Any]:
    """Encode grades as threshold targets for grade > 0, grade > 1, grade > 2."""
    import torch

    thresholds = torch.arange(NUM_THRESHOLDS, device=grades.device).view(1, 1, NUM_THRESHOLDS)
    targets = (grades.long().unsqueeze(2) > thresholds).float()
    if mask is None:
        threshold_mask = torch.ones_like(targets, dtype=torch.bool)
    else:
        threshold_mask = mask.bool().unsqueeze(2).expand_as(targets)
    return targets, threshold_mask


def logits_to_threshold_probabilities(logits: Any) -> Any:
    import torch

    return torch.sigmoid(reshape_threshold_logits(logits))


def enforce_monotonic_thresholds(p_gt: Any) -> Any:
    """Project threshold probabilities into p_gt_0 >= p_gt_1 >= p_gt_2."""
    import torch

    p_gt = p_gt.clamp(0.0, 1.0)
    p_gt_0 = p_gt[:, :, 0]
    p_gt_1 = torch.minimum(p_gt[:, :, 1], p_gt_0)
    p_gt_2 = torch.minimum(p_gt[:, :, 2], p_gt_1)
    return torch.stack((p_gt_0, p_gt_1, p_gt_2), dim=2)


def thresholds_to_grade_probabilities(p_gt: Any) -> Any:
    """Convert monotonic threshold probabilities to four grade probabilities."""
    import torch

    p_gt = enforce_monotonic_thresholds(p_gt)
    p0 = 1.0 - p_gt[:, :, 0]
    p1 = p_gt[:, :, 0] - p_gt[:, :, 1]
    p2 = p_gt[:, :, 1] - p_gt[:, :, 2]
    p3 = p_gt[:, :, 2]
    probs = torch.stack((p0, p1, p2, p3), dim=2).clamp_min(0.0)
    total = probs.sum(dim=2, keepdim=True).clamp_min(1e-12)
    return probs / total


def expected_grade_from_thresholds(p_gt: Any) -> Any:
    """Expected grade for threshold-coded ordinal probabilities."""
    return enforce_monotonic_thresholds(p_gt).sum(dim=2)


def expected_grade_from_grade_probs(p_grade: Any) -> Any:
    import torch

    weights = torch.arange(NUM_GRADES, device=p_grade.device, dtype=p_grade.dtype).view(1, 1, NUM_GRADES)
    return (p_grade * weights).sum(dim=2)


def hard_grade_from_grade_probs(p_grade: Any) -> Any:
    return p_grade.argmax(dim=2)


def hard_grade_from_thresholds(p_gt: Any, threshold: float = 0.5) -> Any:
    return (enforce_monotonic_thresholds(p_gt) > threshold).sum(dim=2)
