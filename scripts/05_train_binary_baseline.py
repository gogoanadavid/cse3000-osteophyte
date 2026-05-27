#!/usr/bin/env python3
"""Train the location-specific binary osteophyte baseline."""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Any

PROJECT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from osteophytes.labels import LOCATION_NAMES
from osteophytes.training import (
    choose_device,
    make_optimizer,
    make_run_dir,
    metric_is_better,
    save_checkpoint,
    save_json,
    select_best_metric,
    set_seed,
    train_one_epoch,
    evaluate,
)


DEFAULT_INDEX_PATH = Path("/scratch/dgogoana/osteophytes_project/audits/dataset_index.csv")
DEFAULT_H5_PATH = Path(
    "/scratch/dgogoana/osteophytes_project/data/"
    "all-for-hip-prediction-20260420-0.4mm-224x224.h5"
)
DEFAULT_WEIGHTS_PATH = Path("/scratch/dgogoana/osteophytes_project/pretrained/resnet18-f37072fd.pth")
DEFAULT_OUTPUT_ROOT = Path("/scratch/dgogoana/osteophytes_project/runs/01_binary_baseline")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index-path", type=Path, default=DEFAULT_INDEX_PATH)
    parser.add_argument("--h5-path", type=Path, default=DEFAULT_H5_PATH)
    parser.add_argument("--weights-path", type=Path, default=DEFAULT_WEIGHTS_PATH)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--selection-metric", choices=("mean_auroc", "mean_spearman"), default="mean_auroc")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--backbone-lr", type=float)
    parser.add_argument("--head-lr", type=float)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-train-samples", type=int)
    parser.add_argument("--max-val-samples", type=int)
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def apply_smoke_overrides(args: argparse.Namespace) -> None:
    if args.smoke:
        args.epochs = 1
        args.max_train_samples = 64
        args.max_val_samples = 64
        args.batch_size = 8


def validate_args(args: argparse.Namespace) -> None:
    if args.epochs < 1:
        raise ValueError("--epochs must be at least 1")
    if args.batch_size < 1:
        raise ValueError("--batch-size must be at least 1")
    if args.num_workers < 0:
        raise ValueError("--num-workers must be non-negative")
    if args.max_train_samples is not None and args.max_train_samples < 1:
        raise ValueError("--max-train-samples must be at least 1 when provided")
    if args.max_val_samples is not None and args.max_val_samples < 1:
        raise ValueError("--max-val-samples must be at least 1 when provided")


def save_predictions_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    import pandas as pd

    columns = [
        "sample_id",
        "subject",
        "visit",
        "side",
        "split",
        *(f"prob_{location}" for location in LOCATION_NAMES),
        *(f"{location}_binary" for location in LOCATION_NAMES),
        *LOCATION_NAMES,
    ]
    pd.DataFrame(rows, columns=columns).to_csv(path, index=False)


def format_metric(value: float) -> str:
    return f"{value:.6f}" if math.isfinite(value) else "NaN"


def print_epoch_summary(
    epoch: int,
    train_metrics: dict[str, float],
    val_metrics: dict[str, Any],
    selection_metric: str,
    best_value: float | None,
) -> None:
    print(f"\nEpoch {epoch}")
    print(f"  train total loss: {format_metric(train_metrics['loss'])}")
    print(f"  train binary loss: {format_metric(train_metrics['binary_loss'])}")
    print(f"  val binary loss: {format_metric(val_metrics['loss'])}")
    print(f"  mean AUROC: {format_metric(val_metrics['mean_auc'])}")
    print(f"  mean AUPRC: {format_metric(val_metrics['mean_auprc'])}")
    print(f"  mean Spearman: {format_metric(val_metrics['mean_spearman'])}")
    best_text = "None" if best_value is None else format_metric(best_value)
    print(f"  selection metric: {selection_metric}; best so far: {best_text}")


def main() -> None:
    args = parse_args()
    apply_smoke_overrides(args)
    validate_args(args)

    import numpy as np
    import torch
    from torch.utils.data import DataLoader

    from osteophytes.dataset import HipOsteophyteDataset
    from osteophytes.models import create_model

    set_seed(args.seed, torch, np)
    device = choose_device(args.device, torch)
    weights_path = None if args.no_pretrained else args.weights_path
    run_dir = make_run_dir(args.output_root)

    config = vars(args).copy()
    config["resolved_device"] = str(device)
    config["effective_weights_path"] = str(weights_path) if weights_path is not None else None
    config["model_head"] = "binary"
    config["location_names"] = LOCATION_NAMES
    save_json(run_dir / "config.json", config)

    train_dataset = HipOsteophyteDataset(
        index_path=args.index_path,
        h5_path=args.h5_path,
        split="train",
        max_samples=args.max_train_samples,
    )
    val_dataset = HipOsteophyteDataset(
        index_path=args.index_path,
        h5_path=args.h5_path,
        split="val",
        max_samples=args.max_val_samples,
    )
    if len(train_dataset) == 0:
        raise ValueError("Train dataset is empty.")
    if len(val_dataset) == 0:
        raise ValueError("Validation dataset is empty.")

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    model = create_model("binary", weights_path=weights_path).to(device)
    optimizer = make_optimizer(
        model,
        torch,
        lr=args.lr,
        weight_decay=args.weight_decay,
        backbone_lr=args.backbone_lr,
        head_lr=args.head_lr,
    )

    print(f"Run directory: {run_dir}")
    print(f"Device: {device}")
    print(f"Selection metric: {args.selection_metric}")
    print(f"Train rows: {len(train_dataset)}")
    print(f"Val rows: {len(val_dataset)}")

    history: list[dict[str, Any]] = []
    best_metric_value: float | None = None
    best_prediction_rows: list[dict[str, Any]] | None = None

    try:
        for epoch in range(1, args.epochs + 1):
            train_metrics = train_one_epoch(model, train_loader, optimizer, device, task="binary")
            val_metrics, prediction_rows = evaluate(model, val_loader, device, task="binary")
            current_value, direction = select_best_metric(val_metrics, args.selection_metric)
            epoch_record = {
                "epoch": epoch,
                "train": train_metrics,
                "train_loss": train_metrics["loss"],
                "val": val_metrics,
                "selection_metric": args.selection_metric,
                "selection_value": current_value,
            }
            history.append(epoch_record)
            print_epoch_summary(epoch, train_metrics, val_metrics, args.selection_metric, best_metric_value)

            save_checkpoint(run_dir / "last_model.pt", model, optimizer, epoch, epoch_record, args)
            if metric_is_better(current_value, best_metric_value, direction):
                best_metric_value = current_value
                best_prediction_rows = prediction_rows
                save_checkpoint(run_dir / "best_model.pt", model, optimizer, epoch, epoch_record, args)
                print(
                    "  saved new best_model.pt with "
                    f"{args.selection_metric} {format_metric(current_value)}"
                )
            save_json(run_dir / "metrics_history.json", history)

        if best_prediction_rows is None:
            best_prediction_rows = prediction_rows
        save_predictions_csv(run_dir / "val_predictions.csv", best_prediction_rows)
        save_json(run_dir / "metrics_history.json", history)
        print(f"\nSaved outputs to: {run_dir}")
    finally:
        train_dataset.close()
        val_dataset.close()


if __name__ == "__main__":
    main()
