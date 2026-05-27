#!/usr/bin/env python3
"""Smoke-test one PyTorch DataLoader batch from the H5-backed dataset index."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

PROJECT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


DEFAULT_INDEX_PATH = Path("/scratch/dgogoana/osteophytes_project/audits/dataset_index.csv")
DEFAULT_H5_PATH = Path(
    "/scratch/dgogoana/osteophytes_project/data/"
    "all-for-hip-prediction-20260420-0.4mm-224x224.h5"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index-path", type=Path, default=DEFAULT_INDEX_PATH)
    parser.add_argument("--h5-path", type=Path, default=DEFAULT_H5_PATH)
    parser.add_argument("--split", default="train")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-samples", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    return parser.parse_args()


def first_values(value: Any, n: int = 3) -> Any:
    if isinstance(value, list):
        return value[:n]
    return value[:n]


def main() -> None:
    args = parse_args()
    if args.batch_size < 1:
        raise ValueError("--batch-size must be at least 1")
    if args.max_samples is not None and args.max_samples < 1:
        raise ValueError("--max-samples must be at least 1")
    if args.num_workers < 0:
        raise ValueError("--num-workers must be non-negative")

    import torch
    from torch.utils.data import DataLoader

    from osteophytes.dataset import HipOsteophyteDataset

    dataset = HipOsteophyteDataset(
        index_path=args.index_path,
        h5_path=args.h5_path,
        split=args.split,
        max_samples=args.max_samples,
    )
    if len(dataset) == 0:
        raise ValueError(f"No rows available for split={args.split!r} with max_samples={args.max_samples!r}")
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    batch = next(iter(dataloader))
    images = batch["image"]
    binary_labels = batch["binary_labels"]
    binary_mask = batch["binary_mask"]
    graded_labels = batch["graded_labels"]
    graded_mask = batch["graded_mask"]

    print(f"dataset length: {len(dataset)}")
    print(f"batch keys: {list(batch.keys())}")
    print(f"image shape: {tuple(images.shape)}")
    print(f"image dtype: {images.dtype}")
    print(f"image min/max: {float(images.min()):.6f} / {float(images.max()):.6f}")
    print(f"binary_labels shape: {tuple(binary_labels.shape)}")
    print(f"binary_labels first samples:\n{binary_labels[:3]}")
    print(f"binary_mask shape: {tuple(binary_mask.shape)}")
    print(f"binary_mask first samples:\n{binary_mask[:3]}")
    print(f"graded_labels shape: {tuple(graded_labels.shape)}")
    print(f"graded_labels first samples:\n{graded_labels[:3]}")
    print(f"graded_mask shape: {tuple(graded_mask.shape)}")
    print(f"graded_mask first samples:\n{graded_mask[:3]}")
    print(f"subjects: {first_values(batch['subject'])}")
    print(f"visits: {first_values(batch['visit'])}")
    print(f"sides: {first_values(batch['side'])}")
    print(f"splits: {first_values(batch['split'])}")

    actual_batch_size = min(args.batch_size, len(dataset))
    assert tuple(images.shape) == (actual_batch_size, 1, 224, 224)
    assert tuple(binary_labels.shape) == (actual_batch_size, 4)
    assert tuple(binary_mask.shape) == (actual_batch_size, 4)
    assert tuple(graded_labels.shape) == (actual_batch_size, 4)
    assert tuple(graded_mask.shape) == (actual_batch_size, 4)
    assert torch.isfinite(images).all()
    assert float(images.min()) >= -1e-6
    assert float(images.max()) <= 1.0 + 1e-6

    dataset.close()
    print("Smoke dataloader test passed.")


if __name__ == "__main__":
    main()
