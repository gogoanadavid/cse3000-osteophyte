"""Evaluation and prediction-row helpers for osteophyte experiments."""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from osteophytes.labels import LOCATION_NAMES
from osteophytes.losses import masked_bce_with_logits, ordinal_threshold_loss
from osteophytes.metrics import (
    compute_binary_auprc_per_location,
    compute_binary_auc_per_location,
    compute_metrics_by_location,
    compute_spearman_per_location,
    nanmean_metric,
)
from osteophytes.ordinal import (
    enforce_monotonic_thresholds,
    expected_grade_from_thresholds,
    hard_grade_from_grade_probs,
    hard_grade_from_thresholds,
    logits_to_threshold_probabilities,
    reshape_threshold_logits,
    thresholds_to_grade_probabilities,
)


def _output_ordinal_logits(outputs: Any) -> Any:
    if isinstance(outputs, dict):
        return outputs["ordinal_logits"]
    return outputs


def _output_binary_logits(outputs: Any) -> Any | None:
    if isinstance(outputs, dict):
        return outputs.get("binary_logits")
    return None


def binary_prediction_rows_from_batch(batch: dict[str, Any], probabilities: Any) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    probs = probabilities.detach().cpu()
    binary_labels = batch["binary_labels"].detach().cpu()
    binary_mask = batch["binary_mask"].detach().cpu()
    graded_labels = batch["graded_labels"].detach().cpu()
    graded_mask = batch["graded_mask"].detach().cpu()
    batch_size = probs.shape[0]

    for row_index in range(batch_size):
        row: dict[str, Any] = {
            "sample_id": str(batch["sample_id"][row_index]),
            "subject": str(batch["subject"][row_index]),
            "visit": str(batch["visit"][row_index]),
            "side": str(batch["side"][row_index]),
            "split": str(batch["split"][row_index]),
        }
        for location_index, location in enumerate(LOCATION_NAMES):
            row[f"prob_{location}"] = float(probs[row_index, location_index])
        for location_index, location in enumerate(LOCATION_NAMES):
            row[f"{location}_binary"] = (
                float(binary_labels[row_index, location_index])
                if float(binary_mask[row_index, location_index]) == 1.0
                else math.nan
            )
        for location_index, location in enumerate(LOCATION_NAMES):
            row[location] = (
                float(graded_labels[row_index, location_index])
                if float(graded_mask[row_index, location_index]) == 1.0
                else math.nan
            )
        rows.append(row)
    return rows


def ordinal_prediction_rows_from_batch(
    batch: dict[str, Any],
    threshold_probs: Any,
    grade_probs: Any,
    pred_grades: Any,
    pred_grades_threshold: Any,
    expected_grades: Any,
    binary_head_p_present: Any | None = None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    threshold_probs_cpu = threshold_probs.detach().cpu()
    grade_probs_cpu = grade_probs.detach().cpu()
    pred_grades_cpu = pred_grades.detach().cpu()
    pred_grades_threshold_cpu = pred_grades_threshold.detach().cpu()
    expected_grades_cpu = expected_grades.detach().cpu()
    binary_head_cpu = None if binary_head_p_present is None else binary_head_p_present.detach().cpu()
    true_grades = batch["graded_labels"].detach().cpu()
    binary_labels = batch["binary_labels"].detach().cpu()
    graded_mask = batch["graded_mask"].detach().cpu()
    sample_ids = batch["sample_id"]

    for row_index in range(threshold_probs_cpu.shape[0]):
        for location_index, location in enumerate(LOCATION_NAMES):
            if float(graded_mask[row_index, location_index]) != 1.0:
                continue
            true_grade = int(true_grades[row_index, location_index])
            row = {
                "sample_id": str(sample_ids[row_index]),
                "subject": str(batch["subject"][row_index]),
                "visit": str(batch["visit"][row_index]),
                "side": str(batch["side"][row_index]),
                "split": str(batch["split"][row_index]),
                "location": location,
                "true_grade": true_grade,
                "true_binary": int(binary_labels[row_index, location_index]),
                "pred_grade": int(pred_grades_cpu[row_index, location_index]),
                "pred_grade_threshold": int(pred_grades_threshold_cpu[row_index, location_index]),
                "expected_grade": float(expected_grades_cpu[row_index, location_index]),
                "p_gt_0": float(threshold_probs_cpu[row_index, location_index, 0]),
                "p_gt_1": float(threshold_probs_cpu[row_index, location_index, 1]),
                "p_gt_2": float(threshold_probs_cpu[row_index, location_index, 2]),
                "p_grade_0": float(grade_probs_cpu[row_index, location_index, 0]),
                "p_grade_1": float(grade_probs_cpu[row_index, location_index, 1]),
                "p_grade_2": float(grade_probs_cpu[row_index, location_index, 2]),
                "p_grade_3": float(grade_probs_cpu[row_index, location_index, 3]),
                "p_present": float(threshold_probs_cpu[row_index, location_index, 0]),
            }
            if binary_head_cpu is not None:
                row["binary_head_p_present"] = float(binary_head_cpu[row_index, location_index])
                row["ordinal_p_present"] = row["p_present"]
            rows.append(row)
    return rows


def _probability_by_grade(
    grades: np.ndarray,
    probabilities: np.ndarray,
    graded_mask: np.ndarray,
) -> dict[str, dict[str, dict[str, float | int]]]:
    summary: dict[str, dict[str, dict[str, float | int]]] = {}
    for location_index, location in enumerate(LOCATION_NAMES):
        location_summary: dict[str, dict[str, float | int]] = {}
        for grade in range(4):
            valid = graded_mask[:, location_index] == 1
            valid &= grades[:, location_index] == grade
            valid &= np.isfinite(probabilities[:, location_index])
            values = probabilities[valid, location_index]
            location_summary[str(grade)] = {
                "n": int(values.size),
                "mean_probability": float(values.mean()) if values.size else math.nan,
                "median_probability": float(np.median(values)) if values.size else math.nan,
            }
        summary[location] = location_summary
    return summary


def evaluate_binary_model(model: Any, dataloader: Any, device: Any) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    import torch

    model.eval()
    total_loss = 0.0
    total_valid = 0.0
    all_probs: list[Any] = []
    all_binary_labels: list[Any] = []
    all_binary_mask: list[Any] = []
    all_graded_labels: list[Any] = []
    all_graded_mask: list[Any] = []
    prediction_rows: list[dict[str, Any]] = []

    with torch.no_grad():
        for batch in dataloader:
            images = batch["image"].to(device)
            labels = batch["binary_labels"].to(device)
            mask = batch["binary_mask"].to(device)
            logits = model(images)
            loss = masked_bce_with_logits(logits, labels, mask)
            probabilities = torch.sigmoid(logits)

            valid_count = float(mask.sum().detach().cpu())
            total_loss += float(loss.detach().cpu()) * valid_count
            total_valid += valid_count

            all_probs.append(probabilities.detach().cpu())
            all_binary_labels.append(batch["binary_labels"].detach().cpu())
            all_binary_mask.append(batch["binary_mask"].detach().cpu())
            all_graded_labels.append(batch["graded_labels"].detach().cpu())
            all_graded_mask.append(batch["graded_mask"].detach().cpu())
            prediction_rows.extend(binary_prediction_rows_from_batch(batch, probabilities))

    if total_valid <= 0:
        raise ValueError("Validation epoch had zero valid binary labels.")

    probs_np = torch.cat(all_probs).numpy()
    binary_labels_np = torch.cat(all_binary_labels).numpy()
    binary_mask_np = torch.cat(all_binary_mask).numpy()
    graded_labels_np = torch.cat(all_graded_labels).numpy()
    graded_mask_np = torch.cat(all_graded_mask).numpy()

    auroc = compute_binary_auc_per_location(binary_labels_np, probs_np, binary_mask_np)
    auprc = compute_binary_auprc_per_location(binary_labels_np, probs_np, binary_mask_np)
    spearman = compute_spearman_per_location(graded_labels_np, probs_np, graded_mask_np)
    metrics = {
        "loss": total_loss / total_valid,
        "auc": auroc,
        "auprc": auprc,
        "mean_auc": nanmean_metric(auroc),
        "mean_auprc": nanmean_metric(auprc),
        "spearman": spearman,
        "mean_spearman": nanmean_metric(spearman),
        "probability_by_grade": _probability_by_grade(graded_labels_np, probs_np, graded_mask_np),
    }
    return metrics, prediction_rows


def evaluate_ordinal_model(model: Any, dataloader: Any, device: Any) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    import torch

    model.eval()
    total_loss = 0.0
    total_count = 0
    all_true_grades: list[Any] = []
    all_pred_grades: list[Any] = []
    all_expected_grades: list[Any] = []
    all_p_present: list[Any] = []
    all_graded_mask: list[Any] = []
    prediction_rows: list[dict[str, Any]] = []

    with torch.no_grad():
        for batch in dataloader:
            images = batch["image"].to(device)
            grades = batch["graded_labels"].to(device).long()
            graded_mask = batch["graded_mask"].to(device).bool()
            outputs = model(images)
            ordinal_logits = reshape_threshold_logits(_output_ordinal_logits(outputs))
            threshold_probs = enforce_monotonic_thresholds(logits_to_threshold_probabilities(ordinal_logits))
            grade_probs = thresholds_to_grade_probabilities(threshold_probs)
            pred_grades = hard_grade_from_grade_probs(grade_probs)
            pred_grades_threshold = hard_grade_from_thresholds(threshold_probs)
            expected_grades = expected_grade_from_thresholds(threshold_probs)
            binary_logits = _output_binary_logits(outputs)
            binary_head_p_present = None if binary_logits is None else torch.sigmoid(binary_logits)

            loss, valid_count = ordinal_threshold_loss(ordinal_logits, grades, graded_mask)
            if valid_count > 0:
                total_loss += float(loss.detach().cpu()) * valid_count
                total_count += valid_count

            all_true_grades.append(batch["graded_labels"].detach().cpu())
            all_pred_grades.append(pred_grades.detach().cpu())
            all_expected_grades.append(expected_grades.detach().cpu())
            all_p_present.append(threshold_probs[:, :, 0].detach().cpu())
            all_graded_mask.append(batch["graded_mask"].detach().cpu())
            prediction_rows.extend(
                ordinal_prediction_rows_from_batch(
                    batch,
                    threshold_probs,
                    grade_probs,
                    pred_grades,
                    pred_grades_threshold,
                    expected_grades,
                    binary_head_p_present=binary_head_p_present,
                )
            )

    if total_count == 0:
        raise ValueError("Validation epoch had zero valid graded labels.")

    true_grades_np = torch.cat(all_true_grades).numpy()
    pred_grades_np = torch.cat(all_pred_grades).numpy()
    expected_grades_np = torch.cat(all_expected_grades).numpy()
    p_present_np = torch.cat(all_p_present).numpy()
    graded_mask_np = torch.cat(all_graded_mask).numpy()

    metrics = compute_metrics_by_location(
        true_grades_np,
        pred_grades_np,
        expected_grades_np,
        p_present_np,
        graded_mask_np,
    )
    metrics["loss"] = total_loss / total_count
    return metrics, prediction_rows
