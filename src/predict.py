"""Generate per-sample ordinal predictions from a checkpoint."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from torch.utils.data import DataLoader

from .augment import make_eval_transform
from .config import load_config
from .data import HipH5Dataset
from .model import OsteophyteOrdinalNet
from .utils import get_device, load_checkpoint, move_batch_to_device, worker_init_fn


@torch.no_grad()
def predict_dataframe(
    model: OsteophyteOrdinalNet,
    loader: DataLoader,
    device: torch.device,
    locations: list[str],
) -> pd.DataFrame:
    """Run model inference and return a flat prediction table."""
    model.eval()
    rows: list[dict[str, Any]] = []
    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        logits = model(images)
        pred = model.predict_from_logits(logits)
        threshold = pred["threshold_probs"].detach().cpu()
        expected = pred["expected_grade"].detach().cpu()
        hard = pred["hard_grade"].detach().cpu()
        batch_size = images.shape[0]
        for i in range(batch_size):
            row = {
                "h5_index": int(batch["h5_index"][i]),
                "subject_id": batch["subject_id"][i],
                "visit_id": batch["visit_id"][i],
                "side": batch["side"][i],
                "split": batch["split"][i],
            }
            severity_proxy = 0.0
            max_presence = 0.0
            for loc_idx, loc in enumerate(locations):
                p_ge1 = float(threshold[i, loc_idx, 0])
                row[f"{loc}_p_ge1"] = p_ge1
                row[f"{loc}_p_ge2"] = float(threshold[i, loc_idx, 1])
                row[f"{loc}_p_ge3"] = float(threshold[i, loc_idx, 2])
                row[f"{loc}_expected_grade"] = float(expected[i, loc_idx])
                row[f"{loc}_hard_grade"] = int(hard[i, loc_idx])
                severity_proxy += p_ge1
                max_presence = max(max_presence, p_ge1)
            row["severity_proxy"] = severity_proxy
            row["max_presence_proxy"] = max_presence
            rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", required=True, choices=["train", "val", "test"])
    parser.add_argument("--out", required=True)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=4)
    args = parser.parse_args()

    data_config = load_config(args.data_config)
    locations = list(data_config["locations"])
    dataset = HipH5Dataset(
        data_config,
        split=args.split,
        transform=make_eval_transform({}),
        percentile_clip=data_config.get("percentile_clip"),
    )
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
    load_checkpoint(args.checkpoint, model=model, map_location=device, strict=True)
    df = predict_dataframe(model, loader, device, locations)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"Wrote predictions to {out}")


if __name__ == "__main__":
    main()
