"""Loss functions for binary, ordinal, and mixed-supervision training."""

from __future__ import annotations

import math
from typing import Any

from osteophytes.ordinal import encode_ordinal_thresholds, reshape_threshold_logits


def _to_bool_tensor(value: Any, device: Any):
    import torch

    if hasattr(value, "to"):
        return value.to(device).bool()
    return torch.tensor(value, dtype=torch.bool, device=device)


def _zero_like(logits: Any) -> Any:
    return logits.sum() * 0.0


def _float_or_nan(value: Any, count: int) -> float:
    if count <= 0:
        return math.nan
    return float(value.detach().cpu())


def _as_threshold_weights(threshold_weights: str | tuple[float, ...] | list[float] | None, device: Any):
    import torch

    if threshold_weights is None:
        return None
    if isinstance(threshold_weights, str):
        values = [float(part.strip()) for part in threshold_weights.split(",") if part.strip()]
    else:
        values = [float(value) for value in threshold_weights]
    if len(values) != 3:
        raise ValueError("threshold_weights must contain exactly three values")
    return torch.tensor(values, dtype=torch.float32, device=device).view(1, 1, 3)


def _focal_factor_from_logits(logits: Any, targets: Any, gamma: float) -> Any:
    import torch

    probs = torch.sigmoid(logits)
    p_t = torch.where(targets > 0.5, probs, 1.0 - probs)
    return (1.0 - p_t).clamp_min(0.0).pow(gamma)


def _ordinal_weight_tensor(
    targets: Any,
    mask: Any,
    mode: str,
    threshold_weights: str | tuple[float, ...] | list[float] | None,
    max_loss_weight: float,
) -> Any:
    import torch

    weights = torch.ones_like(targets)
    manual = _as_threshold_weights(threshold_weights, targets.device)
    if manual is not None:
        weights = weights * manual.to(dtype=weights.dtype)
    if mode == "none" or mode == "manual_threshold":
        return weights.clamp(max=max_loss_weight)
    if mode not in {"inverse_frequency", "effective_number"}:
        raise ValueError(f"Unsupported ordinal class weighting: {mode}")

    valid_targets = targets[mask]
    if valid_targets.numel() == 0:
        return weights
    for threshold_index in range(targets.shape[2]):
        valid = mask[:, :, threshold_index]
        if int(valid.sum().detach().cpu()) == 0:
            continue
        threshold_targets = targets[:, :, threshold_index][valid]
        pos = float((threshold_targets > 0.5).sum().detach().cpu())
        neg = float((threshold_targets <= 0.5).sum().detach().cpu())
        if pos <= 0 or neg <= 0:
            pos_weight = 1.0
        elif mode == "inverse_frequency":
            pos_weight = neg / pos
        else:
            beta = 0.999
            effective_pos = (1.0 - beta**pos) / (1.0 - beta)
            effective_neg = (1.0 - beta**neg) / (1.0 - beta)
            pos_weight = effective_neg / max(effective_pos, 1e-12)
        pos_weight = min(max(pos_weight, 1.0 / max_loss_weight), max_loss_weight)
        weights[:, :, threshold_index] = torch.where(
            targets[:, :, threshold_index] > 0.5,
            weights[:, :, threshold_index] * pos_weight,
            weights[:, :, threshold_index],
        )
    return weights.clamp(max=max_loss_weight)


def masked_bce_with_logits(
    logits: Any,
    labels: Any,
    mask: Any,
    weight: Any | None = None,
    use_focal_loss: bool = False,
    focal_gamma: float = 2.0,
) -> Any:
    import torch.nn.functional as F

    mask = mask.bool()
    valid_count = int(mask.sum().detach().cpu())
    if valid_count <= 0:
        return _zero_like(logits)
    losses = F.binary_cross_entropy_with_logits(logits, labels.float(), reduction="none")
    if weight is not None:
        losses = losses * weight.to(dtype=losses.dtype)
    if use_focal_loss:
        losses = losses * _focal_factor_from_logits(logits, labels.float(), focal_gamma)
    return losses[mask].mean()


def ordinal_threshold_loss(
    logits: Any,
    grades: Any,
    graded_mask: Any,
    ordinal_class_weighting: str = "none",
    threshold_weights: str | tuple[float, ...] | list[float] | None = None,
    max_loss_weight: float = 5.0,
    use_focal_loss: bool = False,
    focal_gamma: float = 2.0,
) -> tuple[Any, int]:
    threshold_logits = reshape_threshold_logits(logits)
    targets, threshold_mask = encode_ordinal_thresholds(grades.long(), graded_mask.bool())
    count = int(threshold_mask.sum().detach().cpu())
    if count <= 0:
        return _zero_like(threshold_logits), 0
    weights = _ordinal_weight_tensor(
        targets,
        threshold_mask,
        ordinal_class_weighting,
        threshold_weights,
        max_loss_weight,
    )
    return (
        masked_bce_with_logits(
            threshold_logits,
            targets,
            threshold_mask,
            weight=weights,
            use_focal_loss=use_focal_loss,
            focal_gamma=focal_gamma,
        ),
        count,
    )


def binary_location_loss(
    location_logits: Any,
    binary_labels: Any,
    binary_mask: Any,
    use_focal_loss: bool = False,
    focal_gamma: float = 2.0,
) -> tuple[Any, int]:
    mask = binary_mask.bool()
    count = int(mask.sum().detach().cpu())
    if count <= 0:
        return _zero_like(location_logits), 0
    return (
        masked_bce_with_logits(
            location_logits,
            binary_labels.float(),
            mask,
            use_focal_loss=use_focal_loss,
            focal_gamma=focal_gamma,
        ),
        count,
    )


def binary_image_noisy_or_loss(
    location_logits: Any,
    binary_labels: Any,
    binary_mask: Any,
    sample_mask: Any,
    use_focal_loss: bool = False,
    focal_gamma: float = 2.0,
) -> tuple[Any, int]:
    import torch
    import torch.nn.functional as F

    valid_locations = binary_mask.bool()
    valid_samples = sample_mask.bool() & valid_locations.any(dim=1)
    count = int(valid_samples.sum().detach().cpu())
    if count <= 0:
        return _zero_like(location_logits), 0
    p_location = torch.sigmoid(location_logits)
    one_minus_p = torch.where(valid_locations, 1.0 - p_location, torch.ones_like(p_location))
    p_any = (1.0 - one_minus_p.prod(dim=1)).clamp(1e-6, 1.0 - 1e-6)
    y_any = ((binary_labels > 0.5) & valid_locations).any(dim=1).float()
    losses = F.binary_cross_entropy(p_any[valid_samples], y_any[valid_samples], reduction="none")
    if use_focal_loss:
        p_t = torch.where(y_any[valid_samples] > 0.5, p_any[valid_samples], 1.0 - p_any[valid_samples])
        losses = losses * (1.0 - p_t).pow(focal_gamma)
    return losses.mean(), count


def _ordinal_logits(outputs: Any) -> Any:
    if isinstance(outputs, dict):
        return outputs["ordinal_logits"]
    return outputs


def _binary_logits(outputs: Any) -> Any:
    if isinstance(outputs, dict) and "binary_logits" in outputs:
        return outputs["binary_logits"]
    return reshape_threshold_logits(_ordinal_logits(outputs))[:, :, 0]


def _batch_flags(batch: dict[str, Any], device: Any) -> tuple[Any, Any]:
    is_strong = _to_bool_tensor(batch["is_strong"], device).view(-1)
    if "is_weak" in batch:
        is_weak = _to_bool_tensor(batch["is_weak"], device).view(-1)
    else:
        is_weak = ~is_strong
    return is_strong, is_weak


def _balance_scales(strong_samples: int, weak_samples: int, mode: str) -> tuple[float, float]:
    if mode == "equal" or mode == "manual":
        return 1.0, 1.0
    if mode != "proportional":
        raise ValueError(f"Unsupported loss balance mode: {mode}")
    total = strong_samples + weak_samples
    if total <= 0:
        return 1.0, 1.0
    strong_scale = strong_samples / total if strong_samples > 0 else 0.0
    weak_scale = weak_samples / total if weak_samples > 0 else 0.0
    return strong_scale, weak_scale


def dual_head_loss(
    outputs: dict[str, Any],
    batch: dict[str, Any],
    weak_label_mode: str,
    strong_trains_binary_head: bool = True,
    weak_trains_ordinal_presence: bool = False,
    weak_loss_weight: float = 1.0,
    ordinal_loss_weight: float = 1.0,
    strong_binary_loss_weight: float = 1.0,
    loss_balance_mode: str = "proportional",
    ordinal_class_weighting: str = "none",
    threshold_weights: str | tuple[float, ...] | list[float] | None = None,
    max_loss_weight: float = 5.0,
    use_focal_loss: bool = False,
    focal_gamma: float = 2.0,
) -> tuple[Any, dict[str, Any]]:
    return mixed_supervision_loss(
        outputs,
        batch,
        weak_label_mode=weak_label_mode,
        strong_trains_binary_head=strong_trains_binary_head,
        weak_trains_ordinal_presence=weak_trains_ordinal_presence,
        weak_loss_weight=weak_loss_weight,
        ordinal_loss_weight=ordinal_loss_weight,
        strong_binary_loss_weight=strong_binary_loss_weight,
        loss_balance_mode=loss_balance_mode,
        ordinal_class_weighting=ordinal_class_weighting,
        threshold_weights=threshold_weights,
        max_loss_weight=max_loss_weight,
        use_focal_loss=use_focal_loss,
        focal_gamma=focal_gamma,
    )


def mixed_supervision_loss(
    outputs: Any,
    batch: dict[str, Any],
    weak_label_mode: str,
    strong_trains_binary_head: bool = True,
    weak_trains_ordinal_presence: bool = False,
    weak_loss_weight: float = 1.0,
    ordinal_loss_weight: float = 1.0,
    strong_binary_loss_weight: float = 1.0,
    loss_balance_mode: str = "proportional",
    ordinal_class_weighting: str = "none",
    threshold_weights: str | tuple[float, ...] | list[float] | None = None,
    max_loss_weight: float = 5.0,
    use_focal_loss: bool = False,
    focal_gamma: float = 2.0,
) -> tuple[Any, dict[str, Any]]:
    ordinal_logits = reshape_threshold_logits(_ordinal_logits(outputs))
    binary_logits = _binary_logits(outputs)
    device = ordinal_logits.device

    grades = batch["graded_labels"].to(device).long()
    graded_mask = batch["graded_mask"].to(device).bool()
    binary_labels = batch["binary_labels"].to(device).float()
    binary_mask = batch["binary_mask"].to(device).bool()
    is_strong, is_weak = _batch_flags(batch, device)

    strong_location_mask = is_strong.view(-1, 1) & graded_mask
    strong_samples = int(is_strong.sum().detach().cpu())
    weak_samples = int(is_weak.sum().detach().cpu())

    ordinal_loss, ordinal_count = ordinal_threshold_loss(
        ordinal_logits,
        grades,
        strong_location_mask,
        ordinal_class_weighting=ordinal_class_weighting,
        threshold_weights=threshold_weights,
        max_loss_weight=max_loss_weight,
        use_focal_loss=use_focal_loss,
        focal_gamma=focal_gamma,
    )

    weak_mask = is_weak.view(-1, 1) & binary_mask
    if weak_label_mode == "location_binary":
        weak_loss, weak_count = binary_location_loss(
            binary_logits,
            binary_labels,
            weak_mask,
            use_focal_loss=use_focal_loss,
            focal_gamma=focal_gamma,
        )
    elif weak_label_mode == "image_binary":
        weak_loss, weak_count = binary_image_noisy_or_loss(
            binary_logits,
            binary_labels,
            binary_mask,
            is_weak,
            use_focal_loss=use_focal_loss,
            focal_gamma=focal_gamma,
        )
    else:
        raise ValueError(f"Unsupported weak label mode: {weak_label_mode}")

    if weak_trains_ordinal_presence and isinstance(outputs, dict):
        ordinal_presence_loss, ordinal_presence_count = binary_location_loss(
            ordinal_logits[:, :, 0],
            binary_labels,
            weak_mask,
            use_focal_loss=use_focal_loss,
            focal_gamma=focal_gamma,
        )
        if ordinal_presence_count > 0:
            weak_loss = weak_loss + ordinal_presence_loss
            weak_count += ordinal_presence_count

    strong_binary_mask = is_strong.view(-1, 1) & binary_mask
    # Non-dual ordinal models have no separate binary head; their grade > 0
    # threshold is already trained by the strong ordinal loss.
    if strong_trains_binary_head and isinstance(outputs, dict):
        strong_binary_loss, strong_binary_count = binary_location_loss(
            binary_logits,
            binary_labels,
            strong_binary_mask,
            use_focal_loss=use_focal_loss,
            focal_gamma=focal_gamma,
        )
    else:
        strong_binary_loss = _zero_like(binary_logits)
        strong_binary_count = 0

    strong_scale, weak_scale = _balance_scales(strong_samples, weak_samples, loss_balance_mode)
    total = _zero_like(ordinal_logits)
    if ordinal_count > 0:
        total = total + ordinal_loss_weight * strong_scale * ordinal_loss
    if weak_count > 0:
        total = total + weak_loss_weight * weak_scale * weak_loss
    if strong_binary_count > 0:
        total = total + strong_binary_loss_weight * strong_scale * strong_binary_loss

    has_loss = ordinal_count > 0 or weak_count > 0 or strong_binary_count > 0
    return total, {
        "loss": _float_or_nan(total, 1) if has_loss else math.nan,
        "ordinal_loss": _float_or_nan(ordinal_loss, ordinal_count),
        "weak_loss": _float_or_nan(weak_loss, weak_count),
        "strong_binary_loss": _float_or_nan(strong_binary_loss, strong_binary_count),
        "ordinal_count": ordinal_count,
        "weak_count": weak_count,
        "strong_binary_count": strong_binary_count,
        "strong_samples": strong_samples,
        "weak_samples": weak_samples,
        "strong_scale": strong_scale,
        "weak_scale": weak_scale,
        "has_loss": has_loss,
    }
