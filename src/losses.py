"""Losses and visible-label class weights for partial ordinal supervision."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn.functional as F


def ordinal_targets_from_grades(grades: torch.Tensor) -> torch.Tensor:
    """Convert grades 0..3 to cumulative targets [y>=1,y>=2,y>=3]."""
    thresholds = torch.arange(1, 4, device=grades.device).view(*([1] * grades.ndim), 3)
    return (grades.unsqueeze(-1) >= thresholds).to(torch.float32)


def partial_ordinal_targets_and_mask(
    binary: torch.Tensor,
    grades: torch.Tensor,
    binary_threshold_only: bool = False,
    binary_source_weight: float = 0.75,
    absent_high_threshold_weight: float = 0.35,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build targets, visible mask, and source weights under mixed supervision."""
    targets = torch.zeros((*binary.shape, 3), device=binary.device, dtype=torch.float32)
    mask = torch.zeros_like(targets)
    weights = torch.ones_like(targets)

    if binary_threshold_only:
        known_binary = binary >= 0
        targets[..., 0] = binary.clamp_min(0).to(torch.float32)
        mask[..., 0] = known_binary.to(torch.float32)
        weights[..., 0] = 1.0
        return targets, mask, weights

    known_grade = grades >= 0
    if known_grade.any():
        targets = torch.where(known_grade.unsqueeze(-1), ordinal_targets_from_grades(grades.clamp_min(0)), targets)
        mask = torch.where(known_grade.unsqueeze(-1), torch.ones_like(mask), mask)
        weights = torch.where(known_grade.unsqueeze(-1), torch.ones_like(weights), weights)

    unknown_grade = ~known_grade
    known_binary = binary >= 0
    binary_only = unknown_grade & known_binary
    if binary_only.any():
        bin_float = binary.clamp_min(0).to(torch.float32)
        positive = binary_only & (binary == 1)
        negative = binary_only & (binary == 0)
        targets[..., 0] = torch.where(binary_only, bin_float, targets[..., 0])
        mask[..., 0] = torch.where(binary_only, torch.ones_like(mask[..., 0]), mask[..., 0])
        weights[..., 0] = torch.where(binary_only, torch.full_like(weights[..., 0], binary_source_weight), weights[..., 0])

        mask[..., 1] = torch.where(negative, torch.ones_like(mask[..., 1]), mask[..., 1])
        mask[..., 2] = torch.where(negative, torch.ones_like(mask[..., 2]), mask[..., 2])
        weights[..., 1] = torch.where(
            negative,
            torch.full_like(weights[..., 1], binary_source_weight * absent_high_threshold_weight),
            weights[..., 1],
        )
        weights[..., 2] = torch.where(
            negative,
            torch.full_like(weights[..., 2], binary_source_weight * absent_high_threshold_weight),
            weights[..., 2],
        )
        targets[..., 1] = torch.where(positive | negative, targets[..., 1], targets[..., 1])
        targets[..., 2] = torch.where(positive | negative, targets[..., 2], targets[..., 2])
    return targets, mask, weights


def masked_partial_ordinal_loss(
    logits: torch.Tensor,
    binary: torch.Tensor,
    grades: torch.Tensor,
    pos_weight: torch.Tensor | None = None,
    threshold_weight: torch.Tensor | None = None,
    binary_source_weight: float = 0.75,
    absent_high_threshold_weight: float = 0.35,
    focal_gamma: float | None = None,
    binary_threshold_only: bool = False,
) -> torch.Tensor:
    """BCE-with-logits over only visible ordinal labels."""
    targets, mask, source_weights = partial_ordinal_targets_and_mask(
        binary,
        grades,
        binary_threshold_only=binary_threshold_only,
        binary_source_weight=binary_source_weight,
        absent_high_threshold_weight=absent_high_threshold_weight,
    )
    if pos_weight is not None:
        pos_weight = pos_weight.to(device=logits.device, dtype=logits.dtype)
    loss = F.binary_cross_entropy_with_logits(logits, targets.to(logits.dtype), pos_weight=pos_weight, reduction="none")
    if focal_gamma is not None:
        probs = torch.sigmoid(logits)
        pt = targets * probs + (1.0 - targets) * (1.0 - probs)
        loss = loss * (1.0 - pt).clamp_min(1e-6).pow(float(focal_gamma))
    weights = mask.to(logits.dtype) * source_weights.to(logits.dtype)
    if threshold_weight is not None:
        tw = threshold_weight.to(device=logits.device, dtype=logits.dtype).view(1, 1, 3)
        weights = weights * tw
    denom = weights.sum().clamp_min(1.0)
    return (loss * weights).sum() / denom


def effective_pos_weight(pos: int | float, neg: int | float, cap: float = 30.0) -> float:
    """Capped neg/pos class weight with safe behavior for rare positives."""
    pos = float(pos)
    neg = float(neg)
    if pos <= 0 and neg <= 0:
        return 1.0
    if pos <= 0:
        return float(cap)
    return float(min(cap, max(1.0, neg / pos)))


def compute_visible_pos_weights(
    binary: np.ndarray | torch.Tensor,
    grades: np.ndarray | torch.Tensor,
    cap: float = 30.0,
    binary_threshold_only: bool = False,
    binary_source_weight: float = 0.75,
    absent_high_threshold_weight: float = 0.35,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Compute [location, threshold] pos_weight using only visible labels."""
    binary_t = torch.tensor(np.asarray(binary).copy(), dtype=torch.long)
    grades_t = torch.tensor(np.asarray(grades).copy(), dtype=torch.long)
    targets, mask, _ = partial_ordinal_targets_and_mask(
        binary_t,
        grades_t,
        binary_threshold_only=binary_threshold_only,
        binary_source_weight=binary_source_weight,
        absent_high_threshold_weight=absent_high_threshold_weight,
    )
    weights = torch.ones((binary_t.shape[1], 3), dtype=torch.float32)
    counts: dict[str, Any] = {"cap": float(cap), "locations": []}
    for loc in range(binary_t.shape[1]):
        loc_counts = {"location_index": loc, "thresholds": []}
        for th in range(3):
            visible = mask[:, loc, th] > 0
            pos = int(((targets[:, loc, th] > 0.5) & visible).sum().item())
            neg = int(((targets[:, loc, th] <= 0.5) & visible).sum().item())
            w = effective_pos_weight(pos, neg, cap=cap)
            weights[loc, th] = w
            loc_counts["thresholds"].append({"threshold": th, "pos": pos, "neg": neg, "pos_weight": w})
        counts["locations"].append(loc_counts)
    return weights, counts
