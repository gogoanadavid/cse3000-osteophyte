#!/usr/bin/env python3
"""Train mixed-supervision ordinal osteophyte severity models."""

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
from osteophytes.models import initialize_backbone_from_binary_checkpoint, set_backbone_trainable
from osteophytes.supervision import (
    STRONG_SAMPLING_STRATEGIES,
    SUPERVISION_MODES,
    MixedSupervisionDataset,
    assert_startup_sanity,
    build_supervision_split,
    save_supervision_artifacts,
)
from osteophytes.training import (
    SELECTION_METRICS,
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
DEFAULT_OUTPUT_ROOT = Path("/scratch/dgogoana/osteophytes_project/runs/02_mixed_supervision")


def str2bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    lowered = value.strip().lower()
    if lowered in {"true", "1", "yes", "y"}:
        return True
    if lowered in {"false", "0", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected true/false, got {value!r}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index-path", type=Path, default=DEFAULT_INDEX_PATH)
    parser.add_argument("--h5-path", type=Path, default=DEFAULT_H5_PATH)
    parser.add_argument("--weights-path", type=Path, default=DEFAULT_WEIGHTS_PATH)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--supervision-mode", choices=SUPERVISION_MODES, default="mixed")
    parser.add_argument("--strong-fraction", type=float, default=0.05)
    parser.add_argument("--weak-label-mode", choices=("location_binary", "image_binary"), default="location_binary")
    parser.add_argument("--strong-sampling-strategy", choices=STRONG_SAMPLING_STRATEGIES, default="random")
    parser.add_argument("--model-head", choices=("threshold_independent", "coral", "dual_head"), default="threshold_independent")
    parser.add_argument("--dual-ordinal-head", choices=("threshold_independent", "coral"), default="threshold_independent")
    parser.add_argument("--init-from-binary-checkpoint", type=Path)
    parser.add_argument("--binary-checkpoint", type=Path)
    parser.add_argument("--loss-balance-mode", choices=("equal", "proportional", "manual"), default="proportional")
    parser.add_argument("--weak-loss-weight", type=float, default=1.0)
    parser.add_argument("--ordinal-loss-weight", type=float, default=1.0)
    parser.add_argument("--strong-binary-loss-weight", type=float, default=1.0)
    parser.add_argument(
        "--ordinal-class-weighting",
        choices=("none", "inverse_frequency", "effective_number", "manual_threshold"),
        default="none",
    )
    parser.add_argument("--threshold-weights", default=None)
    parser.add_argument("--max-loss-weight", type=float, default=5.0)
    parser.add_argument("--use-focal-loss", type=str2bool, default=False)
    parser.add_argument("--focal-gamma", type=float, default=2.0)
    parser.add_argument("--strong-trains-binary-head", type=str2bool, default=True)
    parser.add_argument("--weak-trains-ordinal-presence", type=str2bool, default=False)
    parser.add_argument("--ordinal-include-weak", type=str2bool, default=False)
    parser.add_argument("--freeze-backbone-epochs", type=int, default=0)
    parser.add_argument("--backbone-lr", type=float)
    parser.add_argument("--head-lr", type=float)
    parser.add_argument("--selection-metric", choices=SELECTION_METRICS, default="mean_spearman")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--early-stopping-patience", type=int)
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
    if not 0.0 <= args.strong_fraction <= 1.0:
        raise ValueError("--strong-fraction must be in [0, 1]")
    if args.weak_loss_weight < 0 or args.ordinal_loss_weight < 0 or args.strong_binary_loss_weight < 0:
        raise ValueError("Loss weights must be non-negative")
    if args.max_loss_weight <= 0:
        raise ValueError("--max-loss-weight must be positive")
    if args.epochs < 1:
        raise ValueError("--epochs must be at least 1")
    if args.batch_size < 1:
        raise ValueError("--batch-size must be at least 1")
    if args.freeze_backbone_epochs < 0:
        raise ValueError("--freeze-backbone-epochs must be non-negative")
    if args.num_workers < 0:
        raise ValueError("--num-workers must be non-negative")
    if args.max_train_samples is not None and args.max_train_samples < 1:
        raise ValueError("--max-train-samples must be at least 1 when provided")
    if args.max_val_samples is not None and args.max_val_samples < 1:
        raise ValueError("--max-val-samples must be at least 1 when provided")


def fraction_run_name(args: argparse.Namespace, effective_fraction: float) -> str:
    percent = int(math.floor(effective_fraction * 100.0 + 0.5))
    return (
        f"{args.weak_label_mode}_{args.supervision_mode}_{percent:03d}_"
        f"{args.strong_sampling_strategy}_{args.model_head}"
    )


def save_predictions_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    import pandas as pd

    columns = [
        "sample_id",
        "subject",
        "visit",
        "side",
        "split",
        "location",
        "true_grade",
        "true_binary",
        "pred_grade",
        "pred_grade_threshold",
        "expected_grade",
        "p_gt_0",
        "p_gt_1",
        "p_gt_2",
        "p_grade_0",
        "p_grade_1",
        "p_grade_2",
        "p_grade_3",
        "p_present",
        "binary_head_p_present",
        "ordinal_p_present",
    ]
    frame = pd.DataFrame(rows)
    for column in columns:
        if column not in frame.columns:
            frame[column] = math.nan
    frame[columns].to_csv(path, index=False)


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
    print(f"  train weak loss: {format_metric(train_metrics['weak_loss'])}")
    print(f"  train ordinal loss: {format_metric(train_metrics['ordinal_loss'])}")
    print(f"  train strong binary loss: {format_metric(train_metrics['strong_binary_loss'])}")
    print(f"  val ordinal loss: {format_metric(val_metrics['loss'])}")
    print(
        "  validation mean: "
        f"Spearman {format_metric(val_metrics['mean']['spearman'])}, "
        f"MAE(expected) {format_metric(val_metrics['mean']['mae_expected'])}, "
        f"QWK {format_metric(val_metrics['mean']['qwk'])}, "
        f"AUROC {format_metric(val_metrics['mean']['binary_auroc'])}"
    )
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
    binary_checkpoint_path = args.binary_checkpoint or args.init_from_binary_checkpoint

    base_train_dataset = HipOsteophyteDataset(
        index_path=args.index_path,
        h5_path=args.h5_path,
        split="train",
        max_samples=args.max_train_samples,
    )
    base_val_dataset = HipOsteophyteDataset(
        index_path=args.index_path,
        h5_path=args.h5_path,
        split="val",
        max_samples=args.max_val_samples,
    )
    if len(base_train_dataset) == 0:
        raise ValueError("Train dataset is empty.")
    if len(base_val_dataset) == 0:
        raise ValueError("Validation dataset is empty.")

    split = build_supervision_split(
        base_train_dataset.index,
        supervision_mode=args.supervision_mode,
        strong_fraction=args.strong_fraction,
        seed=args.seed,
        strategy=args.strong_sampling_strategy,
        ordinal_include_weak=args.ordinal_include_weak,
    )
    if args.strong_fraction > 0 and args.supervision_mode == "mixed" and split.strong_count == 0:
        raise RuntimeError("Nonzero strong_fraction selected zero strong training samples.")

    run_name = fraction_run_name(args, split.effective_strong_fraction)
    run_dir = make_run_dir(args.output_root, run_name)
    train_dataset = MixedSupervisionDataset(
        base_train_dataset,
        split.strong_sample_ids,
        weak_sample_ids=split.weak_sample_ids,
    )
    val_dataset = MixedSupervisionDataset(base_val_dataset, set(), weak_sample_ids=set())
    save_supervision_artifacts(run_dir, split, base_train_dataset.index, args.seed)
    sanity = assert_startup_sanity(split, base_train_dataset, train_dataset, args.strong_fraction)

    config = vars(args).copy()
    config.update(
        {
            "resolved_device": str(device),
            "effective_weights_path": str(weights_path) if weights_path is not None else None,
            "binary_checkpoint": str(args.binary_checkpoint) if args.binary_checkpoint is not None else None,
            "init_from_binary_checkpoint": (
                str(args.init_from_binary_checkpoint)
                if args.init_from_binary_checkpoint is not None
                else None
            ),
            "effective_strong_fraction": split.effective_strong_fraction,
            "experiment_name": run_name,
            "location_names": LOCATION_NAMES,
            "ordinal_thresholds": ("grade_gt_0", "grade_gt_1", "grade_gt_2"),
            "weak_loss_weight": args.weak_loss_weight,
            "strong_count": split.strong_count,
            "weak_count": split.weak_count,
            "num_train_samples": len(train_dataset),
            "num_val_samples": len(val_dataset),
            "startup_sanity": sanity,
        }
    )
    save_json(run_dir / "config.json", config)

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

    model = create_model(
        args.model_head,
        weights_path=weights_path,
        dual_ordinal_head=args.dual_ordinal_head,
    ).to(device)
    if binary_checkpoint_path is not None:
        init_info = initialize_backbone_from_binary_checkpoint(model, binary_checkpoint_path)
        save_json(run_dir / "binary_checkpoint_init.json", init_info)

    if args.freeze_backbone_epochs > 0:
        set_backbone_trainable(model, False)
    optimizer = make_optimizer(
        model,
        torch,
        lr=args.lr,
        weight_decay=args.weight_decay,
        backbone_lr=args.backbone_lr,
        head_lr=args.head_lr,
    )

    loss_options = {
        "weak_label_mode": args.weak_label_mode,
        "strong_trains_binary_head": args.strong_trains_binary_head,
        "weak_trains_ordinal_presence": args.weak_trains_ordinal_presence,
        "weak_loss_weight": args.weak_loss_weight,
        "ordinal_loss_weight": args.ordinal_loss_weight,
        "strong_binary_loss_weight": args.strong_binary_loss_weight,
        "loss_balance_mode": args.loss_balance_mode,
        "ordinal_class_weighting": args.ordinal_class_weighting,
        "threshold_weights": args.threshold_weights,
        "max_loss_weight": args.max_loss_weight,
        "use_focal_loss": args.use_focal_loss,
        "focal_gamma": args.focal_gamma,
    }

    print(f"Run directory: {run_dir}")
    print(f"Experiment: {run_name}")
    print(f"Device: {device}")
    print(f"Selection metric: {args.selection_metric}")
    print(f"Model head: {args.model_head}")
    print(f"Weak label mode: {args.weak_label_mode}")
    print(f"Strong sampling strategy: {args.strong_sampling_strategy}")
    print(f"Train rows: {len(train_dataset)}")
    print(f"Val rows: {len(val_dataset)}")

    history: list[dict[str, Any]] = []
    best_metric_value: float | None = None
    best_prediction_rows: list[dict[str, Any]] | None = None
    epochs_since_improvement = 0

    try:
        for epoch_index in range(args.epochs):
            epoch = epoch_index + 1
            if args.binary_checkpoint is not None:
                if epoch_index == 0:
                    set_backbone_trainable(model, False)
                    print("Backbone frozen for epoch 1.")
                elif epoch_index == 2:
                    set_backbone_trainable(model, True)
                    optimizer = make_optimizer(
                        model,
                        torch,
                        lr=args.lr,
                        weight_decay=args.weight_decay,
                        backbone_lr=args.backbone_lr,
                        head_lr=args.head_lr,
                    )
                    print("Backbone unfrozen from epoch 3 onwards. Optimizer reset.")
            elif args.freeze_backbone_epochs > 0 and epoch == args.freeze_backbone_epochs + 1:
                set_backbone_trainable(model, True)
                print("Backbone unfrozen.")

            train_metrics = train_one_epoch(
                model,
                train_loader,
                optimizer,
                device,
                task="mixed",
                loss_options=loss_options,
            )
            val_metrics, prediction_rows = evaluate(model, val_loader, device, task="mixed")
            current_value, direction = select_best_metric(val_metrics, args.selection_metric)
            epoch_record = {
                "epoch": epoch,
                "train": train_metrics,
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
                epochs_since_improvement = 0
                save_checkpoint(run_dir / "best_model.pt", model, optimizer, epoch, epoch_record, args)
                print(
                    "  saved new best_model.pt with "
                    f"{args.selection_metric} {format_metric(current_value)}"
                )
            else:
                epochs_since_improvement += 1

            save_json(run_dir / "metrics_history.json", history)
            if (
                args.early_stopping_patience is not None
                and epochs_since_improvement >= args.early_stopping_patience
            ):
                print(f"Early stopping after {epochs_since_improvement} epochs without improvement.")
                break

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
