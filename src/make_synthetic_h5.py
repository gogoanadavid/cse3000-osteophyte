"""Generate a tiny synthetic HDF5 dataset for pipeline smoke tests."""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np

from .utils import save_json


LOCATIONS = ["sup_acet", "inf_acet", "sup_fem", "inf_fem"]


def make_synthetic(out: str | Path, n: int = 128, seed: int = 0) -> None:
    rng = np.random.default_rng(seed)
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    images = rng.normal(0.08, 0.03, size=(n, 224, 224)).astype(np.float32)
    grades = np.zeros((n, 4), dtype=np.int64)
    centers = [(58, 70), (162, 70), (70, 154), (154, 154)]
    yy, xx = np.mgrid[:224, :224]
    for i in range(n):
        latent = rng.normal(size=4)
        if i % 17 == 0:
            latent += 2.3
        for loc, z in enumerate(latent):
            if z > 2.4:
                g = 3
            elif z > 1.2:
                g = 2
            elif z > 0.2:
                g = 1
            else:
                g = 0
            grades[i, loc] = g
            if g > 0:
                cy, cx = centers[loc]
                amp = 0.18 + 0.13 * g
                sigma = 9 + 2 * g
                blob = amp * np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * sigma * sigma))
                images[i] += blob.astype(np.float32)
    images = np.clip(images, 0.0, 1.0)
    binary = (grades > 0).astype(np.int64)

    split = np.array(["train"] * n, dtype="S10")
    split[int(0.65 * n) : int(0.82 * n)] = b"val"
    split[int(0.82 * n) :] = b"test"
    side = np.where(np.arange(n) % 2 == 0, b"L", b"R").astype("S1")
    subject_id = np.array([f"subj_{i:04d}".encode("utf-8") for i in range(n)])
    visit_id = np.array([b"v00" for _ in range(n)])

    visible_grades = grades.copy()
    train_idx = np.where(split == b"train")[0]
    complete_rows = rng.random(len(train_idx)) < 0.55
    partial_mask = rng.random((len(train_idx), 4)) < 0.65
    partial_mask[complete_rows, :] = False
    visible_grades[train_idx[:, None], np.arange(4)] = np.where(
        partial_mask,
        -1,
        visible_grades[train_idx[:, None], np.arange(4)],
    )

    with h5py.File(out, "w") as h5:
        h5.create_dataset("images", data=(images * 255).astype(np.uint8), compression="gzip")
        h5.create_dataset("split", data=split)
        h5.create_dataset("subject_id", data=subject_id)
        h5.create_dataset("visit_id", data=visit_id)
        h5.create_dataset("side", data=side)
        for loc_idx, loc in enumerate(LOCATIONS):
            h5.create_dataset(f"binary_{loc}", data=binary[:, loc_idx])
            h5.create_dataset(f"grade_{loc}", data=visible_grades[:, loc_idx])

    config = {
        "h5_path": str(out),
        "image_key": "images",
        "split_key": "split",
        "subject_id_key": "subject_id",
        "visit_id_key": "visit_id",
        "side_key": "side",
        "left_side_values": ["L", "left", 0],
        "right_side_values": ["R", "right", 1],
        "canonicalize_side": True,
        "flip_side_values": ["L", "left", 0],
        "binary_label_keys": {loc: f"binary_{loc}" for loc in LOCATIONS},
        "grade_label_keys": {loc: f"grade_{loc}" for loc in LOCATIONS},
        "split_values": {"train": ["train", 0], "val": ["val", "valid", "validation", 1], "test": ["test", 2]},
        "index_csv": "outputs/synthetic/index.csv",
        "train_mean_std_json": "outputs/synthetic/train_mean_std.json",
        "num_locations": 4,
        "locations": LOCATIONS,
    }
    save_json(config, "configs/data_synthetic.json")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--n", type=int, default=128)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    make_synthetic(args.out, args.n, args.seed)
    print(f"Wrote synthetic HDF5 to {args.out} and config to configs/data_synthetic.json")


if __name__ == "__main__":
    main()
