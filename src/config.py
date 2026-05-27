"""JSON configuration helpers."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a JSON config file with a useful error message."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file does not exist: {path}")
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_config(config: dict[str, Any], path: str | Path) -> None:
    """Save a JSON config file with stable formatting."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, sort_keys=True)
        f.write("\n")


def apply_common_overrides(config: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    """Return a config copy with CLI overrides for common training fields."""
    out = copy.deepcopy(config)
    mapping = {
        "epochs": "epochs",
        "batch_size": "batch_size",
        "batch_size_all": "batch_size_all",
        "batch_size_graded": "batch_size_graded",
        "num_workers": "num_workers",
        "lr": "lr",
        "mode": "mode",
    }
    for arg_name, key in mapping.items():
        value = getattr(args, arg_name, None)
        if value is not None:
            out[key] = value
    return out


def add_common_training_overrides(parser: argparse.ArgumentParser) -> None:
    """Attach optional CLI overrides used by smoke tests and Slurm scripts."""
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--batch-size-all", type=int, default=None)
    parser.add_argument("--batch-size-graded", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--mode", choices=["mixed", "graded_only", "binary_only_curve"], default=None)
