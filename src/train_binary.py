"""Binary-only pretraining using the y>=1 threshold for all locations."""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from .augment import make_eval_transform, make_train_transform
from .config import add_common_training_overrides, apply_common_overrides, load_config, save_config
from .data import HipH5Dataset
from .losses import compute_visible_pos_weights, masked_partial_ordinal_loss
from .metrics import average_precision, binary_auc, spearman_corr
from .model import OsteophyteOrdinalNet
from .predict import predict_dataframe
from .utils import (
    CSVLogger,
    build_warmup_cosine_scheduler,
    choose_amp_dtype,
    count_parameters,
    describe_command_line,
    get_device,
    move_batch_to_device,
    safe_mkdir,
    save_checkpoint,
    save_json,
    set_seed,
    setup_file_logger,
    worker_init_fn,
)


@torch.no_grad()
def evaluate_binary(
    model: OsteophyteOrdinalNet,
    loader: DataLoader,
    device: torch.device,
    pos_weight: torch.Tensor,
) -> dict[str, float]:
    model.eval()
    losses: list[float] = []
    all_binary: list[np.ndarray] = []
    all_grade: list[np.ndarray] = []
    all_p: list[np.ndarray] = []
    for batch in loader:
        batch = move_batch_to_device(batch, device)
        logits = model(batch["image"])
        loss = masked_partial_ordinal_loss(
            logits,
            batch["binary"],
            batch["grades"],
            pos_weight=pos_weight,
            binary_threshold_only=True,
        )
        losses.append(float(loss.detach().cpu()))
        pred = model.predict_from_logits(logits)
        all_p.append(pred["threshold_probs"][..., 0].detach().cpu().numpy())
        all_binary.append(batch["binary"].detach().cpu().numpy())
        all_grade.append(batch["grades"].detach().cpu().numpy())
    binary = np.concatenate(all_binary, axis=0)
    grades = np.concatenate(all_grade, axis=0)
    probs = np.concatenate(all_p, axis=0)
    loc_aucs = []
    loc_aps = []
    loc_spear = []
    for loc in range(binary.shape[1]):
        m = binary[:, loc] >= 0
        loc_aucs.append(binary_auc(binary[m, loc], probs[m, loc]))
        loc_aps.append(average_precision(binary[m, loc], probs[m, loc]))
        gm = grades[:, loc] >= 0
        loc_spear.append(spearman_corr(grades[gm, loc], probs[gm, loc]))
    pred_binary = (probs >= 0.5).astype(int)
    bal_accs = []
    for loc in range(binary.shape[1]):
        y = binary[:, loc]
        p = pred_binary[:, loc]
        m = y >= 0
        vals = []
        for cls in [0, 1]:
            cm = m & (y == cls)
            if cm.any():
                vals.append(float((p[cm] == cls).mean()))
        bal_accs.append(float(np.mean(vals)) if vals else float("nan"))
    return {
        "loss": float(np.mean(losses)) if losses else float("nan"),
        "mean_auroc": float(np.nanmean(loc_aucs)),
        "mean_ap": float(np.nanmean(loc_aps)),
        "balanced_accuracy": float(np.nanmean(bal_accs)),
        "spearman_vs_grade": float(np.nanmean(loc_spear)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-config", required=True)
    parser.add_argument("--train-config", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--out-dir", required=True)
    add_common_training_overrides(parser)
    args = parser.parse_args()

    data_config = load_config(args.data_config)
    train_config = apply_common_overrides(load_config(args.train_config), args)
    out_dir = safe_mkdir(args.out_dir)
    logger = setup_file_logger(out_dir, "train_binary")
    save_config(data_config, out_dir / "data_config.json")
    save_config(train_config, out_dir / "train_config.json")
    save_json(describe_command_line(args), out_dir / "command.json")

    set_seed(args.seed)
    device = get_device()
    locations = list(data_config["locations"])
    logger.info("Using device=%s", device)

    train_ds = HipH5Dataset(
        data_config,
        split="train",
        transform=make_train_transform(train_config),
        percentile_clip=train_config.get("percentile_clip"),
    )
    val_ds = HipH5Dataset(
        data_config,
        split="val",
        transform=make_eval_transform(train_config),
        percentile_clip=train_config.get("percentile_clip"),
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=int(train_config["batch_size"]),
        shuffle=True,
        num_workers=int(train_config.get("num_workers", 4)),
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
        worker_init_fn=worker_init_fn if int(train_config.get("num_workers", 4)) > 0 else None,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=int(train_config["batch_size"]),
        shuffle=False,
        num_workers=int(train_config.get("num_workers", 4)),
        pin_memory=torch.cuda.is_available(),
        worker_init_fn=worker_init_fn if int(train_config.get("num_workers", 4)) > 0 else None,
    )
    binary = train_ds.df[[f"binary_{loc}" for loc in locations]].to_numpy(dtype=int)
    grades = np.full_like(binary, -1)
    pos_weight, counts = compute_visible_pos_weights(
        binary,
        grades,
        cap=float(train_config.get("pos_weight_cap", 30.0)),
        binary_threshold_only=True,
    )
    save_json(counts, out_dir / "visible_label_counts_pos_weights.json")

    model = OsteophyteOrdinalNet(num_locations=len(locations)).to(device)
    logger.info("Model trainable parameters: %d", count_parameters(model))
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(train_config["lr"]), weight_decay=float(train_config["weight_decay"]))
    scheduler = build_warmup_cosine_scheduler(
        optimizer,
        epochs=int(train_config["epochs"]),
        steps_per_epoch=len(train_loader),
        warmup_epochs=int(train_config.get("warmup_epochs", 0)),
    )
    amp_dtype = choose_amp_dtype(train_config.get("amp_dtype"))
    use_amp = device.type == "cuda" and amp_dtype is not None
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp and amp_dtype == torch.float16)

    fields = ["epoch", "train_loss", "val_loss", "val_mean_auroc", "val_mean_ap", "val_balanced_accuracy", "val_spearman_vs_grade", "lr"]
    csv_logger = CSVLogger(out_dir / "metrics.csv", fields)
    best_metric = -float("inf")
    bad_epochs = 0
    start = time.time()
    pos_weight = pos_weight.to(device)
    for epoch in range(1, int(train_config["epochs"]) + 1):
        model.train()
        train_losses: list[float] = []
        for batch in train_loader:
            batch = move_batch_to_device(batch, device)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type="cuda", dtype=amp_dtype, enabled=use_amp):
                logits = model(batch["image"])
                loss = masked_partial_ordinal_loss(
                    logits,
                    batch["binary"],
                    batch["grades"],
                    pos_weight=pos_weight,
                    binary_threshold_only=True,
                )
            scaler.scale(loss).backward()
            if float(train_config.get("grad_clip_norm", 0.0)) > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(train_config["grad_clip_norm"]))
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            train_losses.append(float(loss.detach().cpu()))

        val = evaluate_binary(model, val_loader, device, pos_weight)
        metric = val["mean_auroc"] if np.isfinite(val["mean_auroc"]) else -val["loss"]
        lr = float(optimizer.param_groups[0]["lr"])
        csv_logger.write(
            {
                "epoch": epoch,
                "train_loss": float(np.mean(train_losses)),
                "val_loss": val["loss"],
                "val_mean_auroc": val["mean_auroc"],
                "val_mean_ap": val["mean_ap"],
                "val_balanced_accuracy": val["balanced_accuracy"],
                "val_spearman_vs_grade": val["spearman_vs_grade"],
                "lr": lr,
            }
        )
        logger.info("epoch=%d train_loss=%.4f val_loss=%.4f val_auc=%.4f", epoch, np.mean(train_losses), val["loss"], val["mean_auroc"])
        if metric > best_metric:
            best_metric = float(metric)
            bad_epochs = 0
            save_checkpoint(
                out_dir / "best.pt",
                model,
                optimizer,
                scheduler,
                epoch,
                best_metric,
                {"data": data_config, "train": train_config, "mode": "binary_pretrain"},
                args.seed,
                locations,
                extra={"strategy": "binary_pretrain", "budget_name": "0", "budget_size": 0, "split": "val"},
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
            {"data": data_config, "train": train_config, "mode": "binary_pretrain"},
            args.seed,
            locations,
            extra={"strategy": "binary_pretrain", "budget_name": "0", "budget_size": 0, "split": "val"},
        )
        if bad_epochs >= int(train_config.get("early_stopping_patience", 10)):
            logger.info("Early stopping after %d bad epochs", bad_epochs)
            break

    load_path = out_dir / "best.pt"
    checkpoint = torch.load(load_path, map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    val_pred = predict_dataframe(model, val_loader, device, locations)
    val_pred.to_csv(out_dir / "val_predictions.csv", index=False)
    logger.info("Training duration %.1f seconds", time.time() - start)


if __name__ == "__main__":
    main()
