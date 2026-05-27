"""HDF5-backed hip crop dataset with explicit missing-label handling."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

try:
    import hdf5plugin  # noqa: F401
except Exception:
    hdf5plugin = None  # type: ignore

import h5py
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .utils import load_json


def _decode(value: Any) -> Any:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if hasattr(value, "item"):
        try:
            item = value.item()
            if isinstance(item, bytes):
                return item.decode("utf-8")
            return item
        except Exception:
            return value
    return value


def _value_matches(value: Any, options: Iterable[Any]) -> bool:
    value_decoded = _decode(value)
    value_text = str(value_decoded).lower()
    for opt in options:
        opt_decoded = _decode(opt)
        if value_decoded == opt_decoded:
            return True
        if value_text == str(opt_decoded).lower():
            return True
    return False


class HipH5Dataset(Dataset):
    """Lazy HDF5 dataset for 224x224 grayscale hip crops.

    The HDF5 file is opened independently inside each worker process. Training
    code can pass ``visible_grade_indices`` to expose only selected graded
    annotations and avoid hidden-grade leakage.
    """

    def __init__(
        self,
        data_config: dict[str, Any],
        index_csv: str | Path | None = None,
        split: str | None = None,
        transform: Any | None = None,
        percentile_clip: list[float] | tuple[float, float] | None = None,
        visible_grade_indices: set[int] | None = None,
        filter_h5_indices: set[int] | None = None,
        use_mean_std: bool = True,
    ) -> None:
        self.data_config = data_config
        self.h5_path = Path(data_config["h5_path"])
        self.image_key = data_config.get("image_key", "")
        self.schema = data_config.get("schema", "flat_h5")
        self.locations = list(data_config.get("locations", ["sup_acet", "inf_acet", "sup_fem", "inf_fem"]))
        self.binary_cols = [f"binary_{loc}" for loc in self.locations]
        self.grade_cols = [f"grade_{loc}" for loc in self.locations]
        self.transform = transform
        self.percentile_clip = percentile_clip
        self.visible_grade_indices = visible_grade_indices
        self._h5: h5py.File | None = None

        index_path = Path(index_csv or data_config.get("index_csv", "outputs/index.csv"))
        if not index_path.exists():
            raise FileNotFoundError(f"Index CSV does not exist: {index_path}")
        df = pd.read_csv(index_path)
        if split is not None:
            df = df[df["split"].astype(str) == str(split)].copy()
        if filter_h5_indices is not None:
            df = df[df["h5_index"].astype(int).isin({int(x) for x in filter_h5_indices})].copy()
        self.df = df.reset_index(drop=True)
        if self.df.empty:
            raise ValueError(f"No rows found for split={split}, filter_h5_indices={filter_h5_indices is not None}")

        self.mean = 0.0
        self.std = 1.0
        mean_std_path = data_config.get("train_mean_std_json")
        if use_mean_std and mean_std_path and Path(mean_std_path).exists():
            stats = load_json(mean_std_path)
            self.mean = float(stats.get("mean", 0.0))
            self.std = max(float(stats.get("std", 1.0)), 1e-6)

    def __len__(self) -> int:
        return len(self.df)

    def _ensure_h5(self) -> h5py.File:
        if self._h5 is None:
            self._h5 = h5py.File(self.h5_path, "r")
            if self.schema == "flat_h5" and self.image_key not in self._h5:
                self._h5.close()
                self._h5 = None
                raise KeyError(f"Image key '{self.image_key}' not found in {self.h5_path}")
        return self._h5

    def close(self) -> None:
        if self._h5 is not None:
            self._h5.close()
            self._h5 = None

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

    def _load_image(self, row: pd.Series) -> np.ndarray:
        h5 = self._ensure_h5()
        if "h5_path_key" in row and isinstance(row["h5_path_key"], str) and row["h5_path_key"]:
            image = np.asarray(h5[row["h5_path_key"]])
        else:
            image = np.asarray(h5[self.image_key][int(row["h5_index"])])
        image = np.squeeze(image)
        if image.ndim != 2:
            raise ValueError(f"Expected a 2D grayscale image after squeeze, got shape {image.shape}")
        image = image.astype(np.float32)
        if image.max(initial=0.0) > 10.0:
            image /= 255.0
        if self.percentile_clip is not None:
            lo, hi = np.percentile(image, [float(self.percentile_clip[0]), float(self.percentile_clip[1])])
            if hi > lo:
                image = np.clip(image, lo, hi)
                image = (image - lo) / max(hi - lo, 1e-6)
        return image

    def __getitem__(self, idx: int) -> dict[str, Any]:
        row = self.df.iloc[idx]
        h5_index = int(row["h5_index"])
        image = self._load_image(row)

        if self.data_config.get("canonicalize_side", True):
            side = row.get("side", "")
            if _value_matches(side, self.data_config.get("flip_side_values", [])):
                image = np.ascontiguousarray(image[:, ::-1])

        image = (image - self.mean) / self.std
        tensor = torch.from_numpy(image[None, :, :].astype(np.float32))
        if self.transform is not None:
            tensor = self.transform(tensor)

        binary = torch.tensor([int(row[col]) for col in self.binary_cols], dtype=torch.long)
        grades_raw = [int(row[col]) for col in self.grade_cols]
        if self.visible_grade_indices is not None and h5_index not in self.visible_grade_indices:
            grades_raw = [-1 for _ in grades_raw]
        grades = torch.tensor(grades_raw, dtype=torch.long)

        return {
            "image": tensor,
            "binary": binary,
            "grades": grades,
            "subject_id": _decode(row.get("subject_id", "")),
            "visit_id": _decode(row.get("visit_id", "")),
            "side": _decode(row.get("side", "")),
            "split": str(row.get("split", "")),
            "h5_index": h5_index,
        }
