"""Create attention overlay examples without OpenCV."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .augment import make_eval_transform
from .config import load_config
from .data import HipH5Dataset
from .model import OsteophyteOrdinalNet
from .utils import get_device, load_checkpoint, worker_init_fn


def _plt():
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        raise RuntimeError("matplotlib is required for attention plots") from exc
    return plt


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--num-examples", type=int, default=12)
    args = parser.parse_args()

    data_config = load_config(args.data_config)
    locations = list(data_config["locations"])
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    dataset = HipH5Dataset(data_config, split=args.split, transform=make_eval_transform({}), percentile_clip=data_config.get("percentile_clip"))
    index = dataset.df.copy()
    if "max_grade" in index:
        index = index.sort_values("max_grade", ascending=False)
    selected = set(index["h5_index"].head(args.num_examples).astype(int).tolist())
    dataset = HipH5Dataset(
        data_config,
        split=args.split,
        transform=make_eval_transform({}),
        percentile_clip=data_config.get("percentile_clip"),
        filter_h5_indices=selected,
    )
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0, worker_init_fn=worker_init_fn)
    device = get_device()
    model = OsteophyteOrdinalNet(num_locations=len(locations)).to(device)
    load_checkpoint(args.checkpoint, model=model, map_location=device, strict=True)
    model.eval()
    plt = _plt()
    count = 0
    for batch in loader:
        image = batch["image"].to(device)
        logits, attention = model(image, return_attention=True)
        pred = model.predict_from_logits(logits)
        att = attention.detach()
        att = F.interpolate(att, size=image.shape[-2:], mode="bilinear", align_corners=False).cpu().numpy()[0]
        img = image.cpu().numpy()[0, 0]
        grades = batch["grades"].numpy()[0]
        hard = pred["hard_grade"].cpu().numpy()[0]
        fig, axes = plt.subplots(1, len(locations), figsize=(3.2 * len(locations), 3.2))
        if len(locations) == 1:
            axes = [axes]
        for loc_idx, loc in enumerate(locations):
            axes[loc_idx].imshow(img, cmap="gray")
            axes[loc_idx].imshow(att[loc_idx], cmap="magma", alpha=0.45)
            axes[loc_idx].axis("off")
            axes[loc_idx].set_title(f"{loc}\ntrue={grades[loc_idx]} pred={hard[loc_idx]}", fontsize=9)
        fig.tight_layout()
        fig.savefig(out_dir / f"attention_h5_{int(batch['h5_index'][0])}.png", dpi=160)
        plt.close(fig)
        count += 1
    print(f"Wrote {count} attention overlays to {out_dir}")


if __name__ == "__main__":
    main()
