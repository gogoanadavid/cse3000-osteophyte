"""Mixed-supervision ordinal fine-tuning for graded annotation budgets."""

from __future__ import annotations

import argparse
import itertools
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

from .augment import make_eval_transform, make_train_transform
from .config import add_common_training_overrides, apply_common_overrides, load_config, save_config
from .data import HipH5Dataset
from .losses import compute_visible_pos_weights, masked_partial_ordinal_loss
from .metrics import (
    adjacent_auc,
    average_precision,
    balanced_ordinal_mae,
    binary_auc,
    macro_f1,
    off_by_one_accuracy,
    per_grade_recall,
    quality_from_bmae,
    quadratic_weighted_kappa,
    severe_miss_rate,
    spearman_corr,
)
from .model import OsteophyteOrdinalNet
from .predict import predict_dataframe
from .utils import (
    CSVLogger,
    build_warmup_cosine_scheduler,
    choose_amp_dtype,
    count_parameters,
    describe_command_line,
    get_device,
    load_checkpoint,
    move_batch_to_device,
    safe_mkdir,
    save_checkpoint,
    save_json,
    set_seed,
    setup_file_logger,
    worker_init_fn,
)


def _read_budget(path: str | Path | None) -> set[int]:
    if path is None:
        return set()
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Budget file does not exist: {p}")
    df = pd.read_csv(p)
    return {int(x) for x in df["h5_index"].tolist()}


def _make_sampler(df: pd.DataFrame, num_samples: int) -> WeightedRandomSampler:
    weights = []
    for grade in df["max_grade"].astype(int).tolist():
        w = 1.0
        if grade >= 1:
            w += 1.0
        if grade >= 2:
            w += 3.0
        if grade >= 3:
            w += 6.0
        weights.append(w)
    return WeightedRandomSampler(torch.as_tensor(weights, dtype=torch.double), num_samples=num_samples, replacement=True)


def _concat_training_batches(a: dict[str, Any], b: dict[str, Any] | None) -> dict[str, Any]:
    if b is None:
        return a
    return {
        "image": torch.cat([a["image"], b["image"]], dim=0),
        "binary": torch.cat([a["binary"], b["binary"]], dim=0),
        "grades": torch.cat([a["grades"], b["grades"]], dim=0),
    }


def _set_backbone_trainable(model: OsteophyteOrdinalNet, trainable: bool) -> None:
    for module in [model.stem, model.layer1, model.layer2, model.layer3, model.layer4]:
        for param in module.parameters():
            param.requires_grad = trainable


@torch.no_grad()
def evaluate_ordinal(
    model: OsteophyteOrdinalNet,
    loader: DataLoader,
    device: torch.device,
    locations: list[str],
    pos_weight: torch.Tensor,
    threshold_weight: torch.Tensor | None,
    config: dict[str, Any],
) -> dict[str, float]:
    model.eval()
    losses = []
    y_true_batches = []
    expected_batches = []
    hard_batches = []
    threshold_batches = []
    for batch in loader:
        batch = move_batch_to_device(batch, device)
        logits = model(batch["image"])
        loss = masked_partial_ordinal_loss(
            logits,
            batch["binary"],
            batch["grades"],
            pos_weight=pos_weight,
            threshold_weight=threshold_weight,
            binary_source_weight=float(config.get("binary_source_weight", 0.75)),
            absent_high_threshold_weight=float(config.get("absent_high_threshold_weight", 0.35)),
            focal_gamma=config.get("focal_gamma"),
        )
        losses.append(float(loss.detach().cpu()))
        pred = model.predict_from_logits(logits)
        y_true_batches.append(batch["grades"].detach().cpu().numpy())
        expected_batches.append(pred["expected_grade"].detach().cpu().numpy())
        hard_batches.append(pred["hard_grade"].detach().cpu().numpy())
        threshold_batches.append(pred["threshold_probs"].detach().cpu().numpy())

    y_true = np.concatenate(y_true_batches, axis=0)
    expected = np.concatenate(expected_batches, axis=0)
    hard = np.concatenate(hard_batches, axis=0)
    threshold = np.concatenate(threshold_batches, axis=0)
    per_loc = []
    for loc in range(len(locations)):
        y = y_true[:, loc]
        m = y >= 0
        recalls = per_grade_recall(y[m], hard[m, loc])
        bmae = balanced_ordinal_mae(y[m], expected[m, loc])
        per_loc.append(
            {
                "bmae": bmae,
                "quality": quality_from_bmae(bmae),
                "qwk": quadratic_weighted_kappa(y[m], hard[m, loc]),
                "spearman": spearman_corr(y[m], expected[m, loc]),
                "macro_f1": macro_f1(y[m], hard[m, loc]),
                "auc_ge1": binary_auc((y[m] >= 1).astype(int), threshold[m, loc, 0]),
                "auc_ge2": binary_auc((y[m] >= 2).astype(int), threshold[m, loc, 1]),
                "auc_ge3": binary_auc((y[m] >= 3).astype(int), threshold[m, loc, 2]),
                "ap_ge2": average_precision((y[m] >= 2).astype(int), threshold[m, loc, 1]),
                "ap_ge3": average_precision((y[m] >= 3).astype(int), threshold[m, loc, 2]),
                "recall_grade0": recalls[0],
                "recall_grade1": recalls[1],
                "recall_grade2": recalls[2],
                "recall_grade3": recalls[3],
                "off_by_one": off_by_one_accuracy(y[m], hard[m, loc]),
                "severe_miss_rate": severe_miss_rate(y[m], hard[m, loc]),
                "adjacent_auc_1v2": adjacent_auc(y[m], threshold[m, loc, 1], 1, 2),
                "adjacent_auc_2v3": adjacent_auc(y[m], threshold[m, loc, 2], 2, 3),
            }
        )
    keys = list(per_loc[0].keys())
    result = {"loss": float(np.mean(losses)) if losses else float("nan")}
    for key in keys:
        vals = np.asarray([row[key] for row in per_loc], dtype=float)
        result[f"{key}_mean"] = float(np.nanmean(vals)) if np.isfinite(vals).any() else float("nan")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-config", required=True)
    parser.add_argument("--train-config", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--budget-file", default=None)
    parser.add_argument("--budget-name", required=True)
    parser.add_argument("--strategy", required=True)
    parser.add_argument("--binary-checkpoint", default=None)
    parser.add_argument("--out-dir", required=True)
    add_common_training_overrides(parser)
    args = parser.parse_args()

    data_config = load_config(args.data_config)
    train_config = apply_common_overrides(load_config(args.train_config), args)
    mode = train_config.get("mode", "mixed")
    out_dir = safe_mkdir(args.out_dir)
    logger = setup_file_logger(out_dir, "train_ordinal")
    save_config(data_config, out_dir / "data_config.json")
    save_config(train_config, out_dir / "train_config.json")
    save_json(describe_command_line(args), out_dir / "command.json")

    set_seed(args.seed)
    device = get_device()
    locations = list(data_config["locations"])
    selected = _read_budget(args.budget_file)
    budget_size = len(selected)
    logger.info("mode=%s strategy=%s budget=%s selected=%d", mode, args.strategy, args.budget_name, budget_size)

    visible = selected if mode in {"mixed", "graded_only"} else set()
    train_index = pd.read_csv(data_config.get("index_csv", "outputs/index.csv"))
    train_rows = train_index[train_index["split"].astype(str) == "train"].copy()
    if mode == "graded_only":
        if not selected:
            raise ValueError("graded_only mode requires a non-empty budget")
        train_rows = train_rows[train_rows["h5_index"].astype(int).isin(selected)].copy()
    binary_arr = train_rows[[f"binary_{loc}" for loc in locations]].to_numpy(dtype=int, copy=True)
    grade_arr = train_rows[[f"grade_{loc}" for loc in locations]].to_numpy(dtype=int, copy=True)
    if mode == "mixed":
        hide_mask = ~train_rows["h5_index"].astype(int).isin(selected).to_numpy()
        grade_arr[hide_mask, :] = -1
    elif mode == "graded_only":
        pass
    else:
        grade_arr[:, :] = -1
    pos_weight, counts = compute_visible_pos_weights(
        binary_arr,
        grade_arr,
        cap=float(train_config.get("pos_weight_cap", 30.0)),
        binary_threshold_only=False,
        binary_source_weight=float(train_config.get("binary_source_weight", 0.75)),
        absent_high_threshold_weight=float(train_config.get("absent_high_threshold_weight", 0.35)),
    )
    save_json(counts, out_dir / "visible_label_counts_pos_weights.json")

    common_ds_kwargs = {
        "data_config": data_config,
        "transform": make_train_transform(train_config),
        "percentile_clip": train_config.get("percentile_clip"),
        "visible_grade_indices": visible,
    }
    if mode == "graded_only":
        train_ds = HipH5Dataset(filter_h5_indices=selected, split="train", **common_ds_kwargs)
    else:
        train_ds = HipH5Dataset(split="train", **common_ds_kwargs)
    num_workers = int(train_config.get("num_workers", 4))
    all_loader = DataLoader(
        train_ds,
        batch_size=int(train_config.get("batch_size_all", 96)),
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        worker_init_fn=worker_init_fn if num_workers > 0 else None,
    )

    graded_loader = None
    if mode == "mixed" and selected and bool(train_config.get("oversample_graded", True)):
        graded_ds = HipH5Dataset(
            data_config,
            split="train",
            transform=make_train_transform(train_config),
            percentile_clip=train_config.get("percentile_clip"),
            visible_grade_indices=selected,
            filter_h5_indices=selected,
        )
        num_samples = max(len(all_loader) * int(train_config.get("batch_size_graded", 32)), len(graded_ds))
        sampler = _make_sampler(graded_ds.df, num_samples=num_samples)
        graded_loader = DataLoader(
            graded_ds,
            batch_size=int(train_config.get("batch_size_graded", 32)),
            sampler=sampler,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
            worker_init_fn=worker_init_fn if num_workers > 0 else None,
        )

    val_ds = HipH5Dataset(
        data_config,
        split="val",
        transform=make_eval_transform(train_config),
        percentile_clip=train_config.get("percentile_clip"),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=int(train_config.get("batch_size_all", 96)),
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        worker_init_fn=worker_init_fn if num_workers > 0 else None,
    )

    model = OsteophyteOrdinalNet(num_locations=len(locations)).to(device)
    if args.binary_checkpoint:
        logger.info("Loading binary checkpoint with compatible strict=False initialization: %s", args.binary_checkpoint)
        load_checkpoint(args.binary_checkpoint, model=model, map_location=device, strict=False)
    logger.info("Model trainable parameters: %d", count_parameters(model))
    threshold_weight = torch.tensor(train_config.get("threshold_weight", [1.0, 1.25, 1.5]), dtype=torch.float32, device=device)
    pos_weight = pos_weight.to(device)

    fields = [
        "epoch",
        "train_loss",
        "val_loss",
        "val_bmae_mean",
        "val_quality_mean",
        "val_qwk_mean",
        "val_spearman_mean",
        "val_macro_f1_mean",
        "lr_backbone",
        "lr_head",
        "mode",
        "strategy",
        "budget_name",
        "budget_size",
        "seed",
    ]
    csv_logger = CSVLogger(out_dir / "metrics.csv", fields)

    if budget_size == 0 and mode != "graded_only":
        logger.info("Budget 0: saving binary-checkpoint baseline without ordinal fine-tuning")
        val = evaluate_ordinal(model, val_loader, device, locations, pos_weight, threshold_weight, train_config)
        csv_logger.write(
            {
                "epoch": 0,
                "train_loss": float("nan"),
                "val_loss": val["loss"],
                "val_bmae_mean": val["bmae_mean"],
                "val_quality_mean": val["quality_mean"],
                "val_qwk_mean": val["qwk_mean"],
                "val_spearman_mean": val["spearman_mean"],
                "val_macro_f1_mean": val["macro_f1_mean"],
                "lr_backbone": 0.0,
                "lr_head": 0.0,
                "mode": mode,
                "strategy": args.strategy,
                "budget_name": args.budget_name,
                "budget_size": budget_size,
                "seed": args.seed,
            }
        )
        save_checkpoint(
            out_dir / "best.pt",
            model,
            None,
            None,
            0,
            val["bmae_mean"],
            {"data": data_config, "train": train_config, "mode": mode},
            args.seed,
            locations,
            extra={
                "strategy": args.strategy,
                "budget_name": args.budget_name,
                "budget_size": budget_size,
                "mode": mode,
                "split": "val",
                "binary_only_curve": True,
            },
        )
        save_checkpoint(
            out_dir / "last.pt",
            model,
            None,
            None,
            0,
            val["bmae_mean"],
            {"data": data_config, "train": train_config, "mode": mode},
            args.seed,
            locations,
            extra={
                "strategy": args.strategy,
                "budget_name": args.budget_name,
                "budget_size": budget_size,
                "mode": mode,
                "split": "val",
                "binary_only_curve": True,
            },
        )
        val_pred = predict_dataframe(model, val_loader, device, locations)
        val_pred.to_csv(out_dir / "val_predictions.csv", index=False)
        return

    backbone_params = list(itertools.chain(model.stem.parameters(), model.layer1.parameters(), model.layer2.parameters(), model.layer3.parameters(), model.layer4.parameters()))
    backbone_ids = {id(p) for p in backbone_params}
    head_params = [p for p in model.parameters() if id(p) not in backbone_ids]
    optimizer = torch.optim.AdamW(
        [
            {"params": backbone_params, "lr": float(train_config.get("backbone_lr", 1e-4))},
            {"params": head_params, "lr": float(train_config.get("head_lr", 3e-4))},
        ],
        weight_decay=float(train_config.get("weight_decay", 1e-4)),
    )
    scheduler = build_warmup_cosine_scheduler(
        optimizer,
        epochs=int(train_config["epochs"]),
        steps_per_epoch=max(1, len(all_loader)),
        warmup_epochs=int(train_config.get("warmup_epochs", 0)),
    )
    amp_dtype = choose_amp_dtype(train_config.get("amp_dtype"))
    use_amp = device.type == "cuda" and amp_dtype is not None
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp and amp_dtype == torch.float16)
    best_metric = float("inf")
    bad_epochs = 0
    start = time.time()
    freeze_epochs = int(train_config.get("freeze_backbone_epochs_for_small_budgets", 0))
    freeze_threshold = int(train_config.get("freeze_backbone_budget_threshold", 0))

    for epoch in range(1, int(train_config["epochs"]) + 1):
        freeze = budget_size <= freeze_threshold and epoch <= freeze_epochs and mode != "binary_only_curve"
        _set_backbone_trainable(model, not freeze)
        model.train()
        train_losses = []
        graded_iter = iter(graded_loader) if graded_loader is not None else None
        for batch in all_loader:
            if graded_iter is not None:
                try:
                    graded_batch = next(graded_iter)
                except StopIteration:
                    graded_iter = iter(graded_loader)
                    graded_batch = next(graded_iter)
                batch = _concat_training_batches(batch, graded_batch)
            batch = move_batch_to_device(batch, device)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type="cuda", dtype=amp_dtype, enabled=use_amp):
                logits = model(batch["image"])
                loss = masked_partial_ordinal_loss(
                    logits,
                    batch["binary"],
                    batch["grades"],
                    pos_weight=pos_weight,
                    threshold_weight=threshold_weight,
                    binary_source_weight=float(train_config.get("binary_source_weight", 0.75)),
                    absent_high_threshold_weight=float(train_config.get("absent_high_threshold_weight", 0.35)),
                    focal_gamma=train_config.get("focal_gamma"),
                )
            scaler.scale(loss).backward()
            if float(train_config.get("grad_clip_norm", 0.0)) > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(train_config["grad_clip_norm"]))
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            train_losses.append(float(loss.detach().cpu()))

        val = evaluate_ordinal(model, val_loader, device, locations, pos_weight, threshold_weight, train_config)
        csv_logger.write(
            {
                "epoch": epoch,
                "train_loss": float(np.mean(train_losses)) if train_losses else float("nan"),
                "val_loss": val["loss"],
                "val_bmae_mean": val["bmae_mean"],
                "val_quality_mean": val["quality_mean"],
                "val_qwk_mean": val["qwk_mean"],
                "val_spearman_mean": val["spearman_mean"],
                "val_macro_f1_mean": val["macro_f1_mean"],
                "lr_backbone": float(optimizer.param_groups[0]["lr"]),
                "lr_head": float(optimizer.param_groups[1]["lr"]),
                "mode": mode,
                "strategy": args.strategy,
                "budget_name": args.budget_name,
                "budget_size": budget_size,
                "seed": args.seed,
            }
        )
        logger.info(
            "epoch=%d train_loss=%.4f val_bmae=%.4f quality=%.4f freeze_backbone=%s",
            epoch,
            float(np.mean(train_losses)) if train_losses else float("nan"),
            val["bmae_mean"],
            val["quality_mean"],
            freeze,
        )
        metric = val["bmae_mean"]
        if np.isfinite(metric) and metric < best_metric:
            best_metric = float(metric)
            bad_epochs = 0
            save_checkpoint(
                out_dir / "best.pt",
                model,
                optimizer,
                scheduler,
                epoch,
                best_metric,
                {"data": data_config, "train": train_config, "mode": mode},
                args.seed,
                locations,
                extra={
                    "strategy": args.strategy,
                    "budget_name": args.budget_name,
                    "budget_size": budget_size,
                    "mode": mode,
                    "split": "val",
                },
            )
        else:
            bad_epochs += 1
        save_checkpoint(
            out_dir / "last.pt",
            model,
            optimizer,
            scheduler,
            epoch,
            best_metric,
            {"data": data_config, "train": train_config, "mode": mode},
            args.seed,
            locations,
            extra={
                "strategy": args.strategy,
                "budget_name": args.budget_name,
                "budget_size": budget_size,
                "mode": mode,
                "split": "val",
            },
        )
        if bad_epochs >= int(train_config.get("early_stopping_patience", 10)):
            logger.info("Early stopping after %d bad epochs", bad_epochs)
            break

    checkpoint = torch.load(out_dir / "best.pt", map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    val_pred = predict_dataframe(model, val_loader, device, locations)
    val_pred.to_csv(out_dir / "val_predictions.csv", index=False)
    logger.info("Training duration %.1f seconds", time.time() - start)


if __name__ == "__main__":
    main()
