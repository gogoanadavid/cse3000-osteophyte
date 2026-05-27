"""Evaluate checkpoints on validation or test grades."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from .augment import make_eval_transform
from .config import load_config
from .data import HipH5Dataset
from .metrics import (
    adjacent_auc,
    average_precision,
    balanced_ordinal_mae,
    binary_auc,
    confusion_matrix,
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
from .utils import get_device, load_checkpoint, save_json, worker_init_fn


def _metric_rows(pred: pd.DataFrame, locations: list[str]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    conf: dict[str, Any] = {}
    for loc in locations:
        y = pred[f"grade_{loc}"].to_numpy(dtype=int)
        m = y >= 0
        expected = pred[f"{loc}_expected_grade"].to_numpy(dtype=float)
        hard = pred[f"{loc}_hard_grade"].to_numpy(dtype=int)
        p1 = pred[f"{loc}_p_ge1"].to_numpy(dtype=float)
        p2 = pred[f"{loc}_p_ge2"].to_numpy(dtype=float)
        p3 = pred[f"{loc}_p_ge3"].to_numpy(dtype=float)
        recalls = per_grade_recall(y[m], hard[m])
        bmae = balanced_ordinal_mae(y[m], expected[m])
        row = {
            "location": loc,
            "n_graded": int(m.sum()),
            "bmae": bmae,
            "quality": quality_from_bmae(bmae),
            "qwk": quadratic_weighted_kappa(y[m], hard[m]),
            "spearman": spearman_corr(y[m], expected[m]),
            "macro_f1": macro_f1(y[m], hard[m]),
            "auc_ge1": binary_auc((y[m] >= 1).astype(int), p1[m]),
            "auc_ge2": binary_auc((y[m] >= 2).astype(int), p2[m]),
            "auc_ge3": binary_auc((y[m] >= 3).astype(int), p3[m]),
            "ap_ge1": average_precision((y[m] >= 1).astype(int), p1[m]),
            "ap_ge2": average_precision((y[m] >= 2).astype(int), p2[m]),
            "ap_ge3": average_precision((y[m] >= 3).astype(int), p3[m]),
            "adjacent_auc_1v2": adjacent_auc(y[m], p2[m], 1, 2),
            "adjacent_auc_2v3": adjacent_auc(y[m], p3[m], 2, 3),
            "off_by_one_accuracy": off_by_one_accuracy(y[m], hard[m]),
            "severe_miss_rate": severe_miss_rate(y[m], hard[m]),
            "recall_grade0": recalls[0],
            "recall_grade1": recalls[1],
            "recall_grade2": recalls[2],
            "recall_grade3": recalls[3],
        }
        rows.append(row)
        conf[loc] = confusion_matrix(y[m], hard[m])

    pooled_rows = []
    for loc in locations:
        tmp = pd.DataFrame(
            {
                "grade": pred[f"grade_{loc}"],
                "expected": pred[f"{loc}_expected_grade"],
                "hard": pred[f"{loc}_hard_grade"],
                "p1": pred[f"{loc}_p_ge1"],
                "p2": pred[f"{loc}_p_ge2"],
                "p3": pred[f"{loc}_p_ge3"],
            }
        )
        pooled_rows.append(tmp)
    pooled = pd.concat(pooled_rows, ignore_index=True)
    y = pooled["grade"].to_numpy(dtype=int)
    m = y >= 0
    hard = pooled["hard"].to_numpy(dtype=int)
    expected = pooled["expected"].to_numpy(dtype=float)
    recalls = per_grade_recall(y[m], hard[m])
    bmae = balanced_ordinal_mae(y[m], expected[m])
    pooled_row = {
        "location": "pooled",
        "n_graded": int(m.sum()),
        "bmae": bmae,
        "quality": quality_from_bmae(bmae),
        "qwk": quadratic_weighted_kappa(y[m], hard[m]),
        "spearman": spearman_corr(y[m], expected[m]),
        "macro_f1": macro_f1(y[m], hard[m]),
        "auc_ge1": binary_auc((y[m] >= 1).astype(int), pooled["p1"].to_numpy(dtype=float)[m]),
        "auc_ge2": binary_auc((y[m] >= 2).astype(int), pooled["p2"].to_numpy(dtype=float)[m]),
        "auc_ge3": binary_auc((y[m] >= 3).astype(int), pooled["p3"].to_numpy(dtype=float)[m]),
        "ap_ge1": average_precision((y[m] >= 1).astype(int), pooled["p1"].to_numpy(dtype=float)[m]),
        "ap_ge2": average_precision((y[m] >= 2).astype(int), pooled["p2"].to_numpy(dtype=float)[m]),
        "ap_ge3": average_precision((y[m] >= 3).astype(int), pooled["p3"].to_numpy(dtype=float)[m]),
        "adjacent_auc_1v2": adjacent_auc(y[m], pooled["p2"].to_numpy(dtype=float)[m], 1, 2),
        "adjacent_auc_2v3": adjacent_auc(y[m], pooled["p3"].to_numpy(dtype=float)[m], 2, 3),
        "off_by_one_accuracy": off_by_one_accuracy(y[m], hard[m]),
        "severe_miss_rate": severe_miss_rate(y[m], hard[m]),
        "recall_grade0": recalls[0],
        "recall_grade1": recalls[1],
        "recall_grade2": recalls[2],
        "recall_grade3": recalls[3],
    }
    rows.append(pooled_row)
    conf["pooled"] = confusion_matrix(y[m], hard[m])
    return rows, conf


def _summary(rows: list[dict[str, Any]], metadata: dict[str, Any]) -> dict[str, Any]:
    loc_rows = [r for r in rows if r["location"] not in {"pooled"}]
    keys = [
        "bmae",
        "quality",
        "qwk",
        "spearman",
        "macro_f1",
        "auc_ge1",
        "auc_ge2",
        "auc_ge3",
        "ap_ge1",
        "ap_ge2",
        "ap_ge3",
        "recall_grade0",
        "recall_grade1",
        "recall_grade2",
        "recall_grade3",
        "off_by_one_accuracy",
        "severe_miss_rate",
        "adjacent_auc_1v2",
        "adjacent_auc_2v3",
    ]
    out = dict(metadata)
    for key in keys:
        vals = np.asarray([row[key] for row in loc_rows], dtype=float)
        out[f"{key}_mean"] = float(np.nanmean(vals)) if np.isfinite(vals).any() else float("nan")
    pooled = next((r for r in rows if r["location"] == "pooled"), None)
    if pooled:
        for key in keys:
            out[f"{key}_pooled"] = pooled[key]
    return out


def _bootstrap(pred: pd.DataFrame, locations: list[str], metadata: dict[str, Any], n: int, seed: int) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    subjects = np.asarray(sorted(pred["subject_id"].astype(str).unique()))
    by_subject = {sid: pred[pred["subject_id"].astype(str) == sid] for sid in subjects}
    values: dict[str, list[float]] = {}
    for _ in range(n):
        sampled = rng.choice(subjects, size=len(subjects), replace=True)
        sample_df = pd.concat([by_subject[sid] for sid in sampled], ignore_index=True)
        rows, _ = _metric_rows(sample_df, locations)
        summary = _summary(rows, metadata)
        for key, value in summary.items():
            if isinstance(value, (float, int)) and np.isfinite(value):
                values.setdefault(key, []).append(float(value))
    ci: dict[str, Any] = {}
    for key, vals in values.items():
        arr = np.asarray(vals, dtype=float)
        if arr.size:
            ci[key] = {"low": float(np.percentile(arr, 2.5)), "high": float(np.percentile(arr, 97.5)), "n": int(arr.size)}
    return ci


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", required=True, choices=["train", "val", "test"])
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--bootstrap", type=int, default=0)
    parser.add_argument("--bootstrap-seed", type=int, default=123)
    args = parser.parse_args()

    data_config = load_config(args.data_config)
    locations = list(data_config["locations"])
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    dataset = HipH5Dataset(data_config, split=args.split, transform=make_eval_transform({}), percentile_clip=data_config.get("percentile_clip"))
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        worker_init_fn=worker_init_fn if args.num_workers > 0 else None,
    )
    device = get_device()
    model = OsteophyteOrdinalNet(num_locations=len(locations)).to(device)
    checkpoint = load_checkpoint(args.checkpoint, model=model, map_location=device, strict=True)
    pred = predict_dataframe(model, loader, device, locations)
    index = pd.read_csv(data_config.get("index_csv", "outputs/index.csv"))
    keep_cols = ["h5_index"] + [f"grade_{loc}" for loc in locations] + [f"binary_{loc}" for loc in locations]
    pred = pred.merge(index[keep_cols], on="h5_index", how="left")
    pred.to_csv(out_dir / "predictions.csv", index=False)

    metadata = {
        "seed": checkpoint.get("seed"),
        "strategy": checkpoint.get("strategy", "unknown"),
        "budget_name": str(checkpoint.get("budget_name", "unknown")),
        "budget_size": checkpoint.get("budget_size"),
        "mode": checkpoint.get("mode", checkpoint.get("config", {}).get("mode", "unknown")),
        "split": args.split,
        "checkpoint": str(args.checkpoint),
    }
    rows, conf = _metric_rows(pred, locations)
    pd.DataFrame(rows).to_csv(out_dir / "metrics_location.csv", index=False)
    summary = _summary(rows, metadata)
    if args.bootstrap > 0:
        summary["bootstrap_ci"] = _bootstrap(pred, locations, metadata, args.bootstrap, args.bootstrap_seed)
        summary["bootstrap_n"] = int(args.bootstrap)
        summary["bootstrap_seed"] = int(args.bootstrap_seed)
    save_json(summary, out_dir / "metrics_summary.json")
    save_json(conf, out_dir / "confusion_matrices.json")
    print(f"Wrote evaluation outputs to {out_dir}")


if __name__ == "__main__":
    main()
