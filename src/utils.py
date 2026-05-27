"""Shared runtime utilities."""

from __future__ import annotations

import csv
import json
import logging
import math
import os
import random
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch


def set_seed(seed: int) -> None:
    """Set Python, NumPy, and PyTorch seeds."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def get_device() -> torch.device:
    """Return CUDA when available, otherwise CPU."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def safe_mkdir(path: str | Path) -> Path:
    """Create a directory if needed and return it as a Path."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def timestamp() -> str:
    """Return a compact UTC-like local timestamp."""
    return time.strftime("%Y%m%d_%H%M%S")


def load_json(path: str | Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(obj: Any, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, sort_keys=True)
        f.write("\n")


def count_parameters(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def git_commit_hash() -> str:
    try:
        out = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        return out
    except Exception:
        return "unknown"


def setup_file_logger(out_dir: str | Path, name: str = "run") -> logging.Logger:
    """Create a logger that writes to stdout and out_dir/log.txt."""
    safe_mkdir(out_dir)
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s %(levelname)s: %(message)s")
    stream = logging.StreamHandler(sys.stdout)
    stream.setFormatter(fmt)
    file_handler = logging.FileHandler(Path(out_dir) / "log.txt", mode="a", encoding="utf-8")
    file_handler.setFormatter(fmt)
    logger.addHandler(stream)
    logger.addHandler(file_handler)
    return logger


def build_warmup_cosine_scheduler(
    optimizer: torch.optim.Optimizer,
    epochs: int,
    steps_per_epoch: int,
    warmup_epochs: int,
) -> torch.optim.lr_scheduler.LambdaLR:
    """Create a per-step warmup cosine scheduler."""
    total_steps = max(1, epochs * max(1, steps_per_epoch))
    warmup_steps = max(0, warmup_epochs * max(1, steps_per_epoch))

    def lr_lambda(step: int) -> float:
        if warmup_steps > 0 and step < warmup_steps:
            return max(1e-8, float(step + 1) / float(warmup_steps))
        denom = max(1, total_steps - warmup_steps)
        progress = min(1.0, max(0.0, (step - warmup_steps) / denom))
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def move_batch_to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    """Move tensor values in a batch dictionary to device."""
    out: dict[str, Any] = {}
    for key, value in batch.items():
        out[key] = value.to(device, non_blocking=True) if torch.is_tensor(value) else value
    return out


def save_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None,
    epoch: int,
    best_metric: float,
    config: dict[str, Any],
    seed: int,
    locations: list[str],
    extra: dict[str, Any] | None = None,
) -> None:
    """Save a training checkpoint with reproducibility metadata."""
    payload: dict[str, Any] = {
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict() if optimizer is not None else None,
        "scheduler_state": scheduler.state_dict() if scheduler is not None else None,
        "epoch": epoch,
        "best_metric": best_metric,
        "config": config,
        "seed": seed,
        "locations": locations,
        "git_commit": git_commit_hash(),
    }
    if extra:
        payload.update(extra)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def load_checkpoint(
    path: str | Path,
    model: torch.nn.Module | None = None,
    map_location: str | torch.device = "cpu",
    strict: bool = True,
) -> dict[str, Any]:
    """Load a checkpoint and optionally load model weights."""
    checkpoint = torch.load(Path(path), map_location=map_location)
    if model is not None:
        result = model.load_state_dict(checkpoint["model_state"], strict=strict)
        if not strict:
            missing = list(result.missing_keys)
            unexpected = list(result.unexpected_keys)
            print(f"Loaded checkpoint with strict=False. Missing={missing}; unexpected={unexpected}")
    return checkpoint


class CSVLogger:
    """Append dictionaries to a CSV file, writing the header on first use."""

    def __init__(self, path: str | Path, fieldnames: Iterable[str]):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.fieldnames = list(fieldnames)
        self._exists = self.path.exists() and self.path.stat().st_size > 0

    def write(self, row: dict[str, Any]) -> None:
        with self.path.open("a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=self.fieldnames, extrasaction="ignore")
            if not self._exists:
                writer.writeheader()
                self._exists = True
            writer.writerow(row)


def choose_amp_dtype(name: str | None) -> torch.dtype | None:
    """Resolve an AMP dtype string."""
    if name is None:
        return None
    lowered = str(name).lower()
    if lowered in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if lowered in {"fp16", "float16"}:
        return torch.float16
    if lowered in {"none", "false", "off"}:
        return None
    raise ValueError(f"Unknown amp dtype: {name}")


def finite_mean(values: Iterable[float]) -> float:
    arr = np.asarray(list(values), dtype=float)
    arr = arr[np.isfinite(arr)]
    return float(arr.mean()) if arr.size else float("nan")


def describe_command_line(args: Any) -> dict[str, Any]:
    return {"argv": sys.argv, "args": vars(args), "git_commit": git_commit_hash()}


def worker_init_fn(worker_id: int) -> None:
    seed = torch.initial_seed() % 2**32
    np.random.seed(seed + worker_id)
    random.seed(seed + worker_id)
