"""Training utilities shared by baseline and mixed-supervision scripts."""

from __future__ import annotations

from datetime import datetime
import json
import math
from pathlib import Path
import random
from typing import Any

from osteophytes.evaluation import evaluate_binary_model, evaluate_ordinal_model
from osteophytes.losses import masked_bce_with_logits, mixed_supervision_loss
from osteophytes.models import backbone_parameters, head_parameters


SELECTION_METRICS = ("mean_spearman", "mean_mae", "mean_qwk", "mean_auroc")


def set_seed(seed: int, torch: Any, np: Any) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def choose_device(device_arg: str, torch: Any) -> Any:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def make_run_dir(output_root: str | Path, *parts: str) -> Path:
    base = Path(output_root)
    if parts:
        base = base.joinpath(*parts)
    run_dir = base / datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def _json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and math.isnan(value):
        return None
    return value


def to_jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [to_jsonable(item) for item in value]
    if isinstance(value, tuple):
        return [to_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and math.isnan(value):
        return None
    return value


def save_json(path: str | Path, data: Any) -> None:
    Path(path).write_text(json.dumps(to_jsonable(data), indent=2, default=_json_safe), encoding="utf-8")


def save_checkpoint(path: str | Path, model: Any, optimizer: Any, epoch: int, metrics: dict[str, Any], config: Any) -> None:
    import torch

    config_dict = vars(config) if hasattr(config, "__dict__") else dict(config)
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
            "metrics": metrics,
            "config": config_dict,
        },
        Path(path),
    )


def load_checkpoint(path: str | Path, model: Any, optimizer: Any | None = None, map_location: str = "cpu") -> dict[str, Any]:
    import torch

    checkpoint = torch.load(Path(path), map_location=map_location)
    model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer is not None and checkpoint.get("optimizer_state_dict") is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    return checkpoint


def select_best_metric(val_metrics: dict[str, Any], selection_metric: str) -> tuple[float, str]:
    if selection_metric == "mean_spearman":
        if "mean" in val_metrics:
            return float(val_metrics["mean"]["spearman"]), "higher"
        return float(val_metrics["mean_spearman"]), "higher"
    if selection_metric == "mean_mae":
        if "mean" in val_metrics:
            return float(val_metrics["mean"]["mae_expected"]), "lower"
        raise ValueError("mean_mae is not available for binary baseline evaluation")
    if selection_metric == "mean_qwk":
        if "mean" in val_metrics:
            return float(val_metrics["mean"]["qwk"]), "higher"
        raise ValueError("mean_qwk is not available for binary baseline evaluation")
    if selection_metric == "mean_auroc":
        if "mean" in val_metrics:
            return float(val_metrics["mean"]["binary_auroc"]), "higher"
        return float(val_metrics["mean_auc"]), "higher"
    raise ValueError(f"Unsupported selection metric: {selection_metric}")


def metric_is_better(candidate: float, best: float | None, direction: str) -> bool:
    if not math.isfinite(candidate):
        return best is None
    if best is None or not math.isfinite(best):
        return True
    if direction == "lower":
        return candidate < best
    return candidate > best


def make_optimizer(
    model: Any,
    torch: Any,
    lr: float,
    weight_decay: float,
    backbone_lr: float | None = None,
    head_lr: float | None = None,
) -> Any:
    if backbone_lr is None and head_lr is None:
        return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    return torch.optim.AdamW(
        [
            {"params": list(backbone_parameters(model)), "lr": backbone_lr or lr},
            {"params": list(head_parameters(model)), "lr": head_lr or lr},
        ],
        lr=lr,
        weight_decay=weight_decay,
    )


def train_one_epoch(
    model: Any,
    dataloader: Any,
    optimizer: Any,
    device: Any,
    task: str,
    loss_options: dict[str, Any] | None = None,
) -> dict[str, float]:
    model.train()
    options = loss_options or {}
    total_loss = 0.0
    total_units = 0
    component_totals = {
        "weak_loss": 0.0,
        "ordinal_loss": 0.0,
        "strong_binary_loss": 0.0,
        "binary_loss": 0.0,
    }
    component_counts = {
        "weak_count": 0,
        "ordinal_count": 0,
        "strong_binary_count": 0,
        "binary_count": 0,
    }
    trained_batches = 0

    for batch in dataloader:
        images = batch["image"].to(device)
        optimizer.zero_grad(set_to_none=True)

        if task == "binary":
            labels = batch["binary_labels"].to(device)
            mask = batch["binary_mask"].to(device)
            logits = model(images)
            loss = masked_bce_with_logits(logits, labels, mask)
            valid_count = int(mask.sum().detach().cpu())
            parts = {"binary_loss": float(loss.detach().cpu()), "binary_count": valid_count, "has_loss": valid_count > 0}
        elif task == "mixed":
            outputs = model(images)
            loss, parts = mixed_supervision_loss(outputs, batch, **options)
        else:
            raise ValueError(f"Unsupported training task: {task}")

        if not parts["has_loss"]:
            continue
        loss.backward()
        optimizer.step()
        trained_batches += 1

        batch_units = int(batch["image"].shape[0])
        total_loss += float(loss.detach().cpu()) * batch_units
        total_units += batch_units
        for count_key in component_counts:
            count = int(parts.get(count_key, 0) or 0)
            component_counts[count_key] += count
            loss_key = count_key.replace("_count", "_loss")
            value = float(parts.get(loss_key, math.nan))
            if count > 0 and math.isfinite(value):
                component_totals[loss_key] += value * count

    if trained_batches == 0:
        raise ValueError("Training epoch had zero valid labels.")

    metrics: dict[str, float] = {
        "loss": total_loss / max(total_units, 1),
        "trained_batches": float(trained_batches),
    }
    for count_key, count in component_counts.items():
        loss_key = count_key.replace("_count", "_loss")
        metrics[count_key] = float(count)
        metrics[loss_key] = component_totals[loss_key] / count if count > 0 else math.nan
    return metrics


def evaluate(model: Any, dataloader: Any, device: Any, task: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if task == "binary":
        return evaluate_binary_model(model, dataloader, device)
    if task == "mixed":
        return evaluate_ordinal_model(model, dataloader, device)
    raise ValueError(f"Unsupported evaluation task: {task}")
