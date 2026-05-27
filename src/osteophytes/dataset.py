"""H5-backed PyTorch dataset for hip osteophyte experiments."""

from __future__ import annotations

from pathlib import Path
from typing import Any

try:  # Needed before h5py reads Blosc2-compressed arrays on DelftBlue.
    import hdf5plugin  # noqa: F401
except ModuleNotFoundError:  # pragma: no cover - handled by runtime H5 checks.
    hdf5plugin = None  # type: ignore[assignment]

import h5py
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from osteophytes.labels import (
    BINARY_LABEL_COLUMNS,
    LOCATION_NAMES,
    SAMPLE_ID_COLUMNS,
    make_sample_id_from_row,
)


METADATA_COLUMNS = ("subject", "visit", "side", "split", "h5_internal_path")
EXPECTED_IMAGE_SHAPE = (224, 224)


def _is_missing(value: Any) -> bool:
    if pd.isna(value):
        return True
    return isinstance(value, str) and not value.strip()


def _optional_float(value: Any, allowed_values: set[float], column: str) -> tuple[float, float]:
    if _is_missing(value):
        return 0.0, 0.0
    numeric_value = float(value)
    if numeric_value not in allowed_values:
        raise ValueError(f"Invalid value {value!r} in label column {column!r}")
    return numeric_value, 1.0


def normalize_image(image: Any) -> np.ndarray:
    """Robustly normalize a single image array to float32 values in [0, 1]."""
    array = np.asarray(image)
    array = np.squeeze(array)
    if array.ndim != 2:
        raise ValueError(f"Expected a 2D image after squeeze, got shape {array.shape}")
    if array.shape != EXPECTED_IMAGE_SHAPE:
        raise ValueError(f"Expected image shape {EXPECTED_IMAGE_SHAPE}, got {array.shape}")

    array = array.astype(np.float32)
    finite = np.isfinite(array)
    if not finite.any():
        raise ValueError("Image contains no finite pixels.")

    low, high = np.percentile(array[finite], [1, 99])
    if high <= low:
        low = float(np.min(array[finite]))
        high = float(np.max(array[finite]))
    if high <= low:
        return np.zeros(array.shape, dtype=np.float32)

    array = np.clip((array - low) / (high - low), 0.0, 1.0)
    array[~finite] = 0.0
    return array.astype(np.float32)


def row_to_labels_and_masks(row: pd.Series) -> dict[str, torch.Tensor]:
    """Convert one dataset-index row into label and mask tensors."""
    binary_labels: list[float] = []
    binary_mask: list[float] = []
    graded_labels: list[float] = []
    graded_mask: list[float] = []

    for column in BINARY_LABEL_COLUMNS:
        label, mask = _optional_float(row[column], {0.0, 1.0}, column)
        binary_labels.append(label)
        binary_mask.append(mask)

    for column in LOCATION_NAMES:
        label, mask = _optional_float(row[column], {0.0, 1.0, 2.0, 3.0}, column)
        graded_labels.append(label)
        graded_mask.append(mask)

    return {
        "binary_labels": torch.tensor(binary_labels, dtype=torch.float32),
        "binary_mask": torch.tensor(binary_mask, dtype=torch.float32),
        "graded_labels": torch.tensor(graded_labels, dtype=torch.float32),
        "graded_mask": torch.tensor(graded_mask, dtype=torch.float32),
    }


class HipOsteophyteDataset(Dataset):
    """Lazy H5-backed dataset for hip X-ray osteophyte labels."""

    def __init__(
        self,
        index_path: str | Path,
        h5_path: str | Path,
        split: str | None = None,
        max_samples: int | None = None,
    ) -> None:
        self.index_path = Path(index_path)
        self.h5_path = Path(h5_path)
        self._h5: h5py.File | None = None

        index = pd.read_csv(self.index_path)
        self._verify_columns(index)
        if split is not None:
            index = index[index["split"] == split].copy()
        if max_samples is not None:
            if max_samples < 1:
                raise ValueError("max_samples must be at least 1 when provided.")
            index = index.head(max_samples).copy()

        self.index = index.reset_index(drop=True)

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        row = self.index.iloc[idx]
        internal_path = str(row["h5_internal_path"])
        h5_file = self._get_h5()
        if internal_path not in h5_file:
            raise KeyError(f"H5 image path not found: {internal_path}")

        image = normalize_image(h5_file[internal_path][()])
        labels = row_to_labels_and_masks(row)
        sample_id = make_sample_id_from_row(row)
        return {
            "image": torch.from_numpy(image).unsqueeze(0).float(),
            **labels,
            "subject": str(row["subject"]),
            "visit": str(row["visit"]),
            "side": str(row["side"]),
            "split": str(row["split"]),
            "h5_internal_path": internal_path,
            "sample_id": sample_id,
        }

    def _get_h5(self) -> h5py.File:
        if self._h5 is None:
            if not self.h5_path.exists():
                raise FileNotFoundError(f"H5 file not found: {self.h5_path}")
            self._h5 = h5py.File(self.h5_path, "r")
        return self._h5

    def _verify_columns(self, index: pd.DataFrame) -> None:
        required = {*METADATA_COLUMNS, *LOCATION_NAMES, *BINARY_LABEL_COLUMNS}
        missing = [column for column in required if column not in index.columns]
        if missing:
            raise ValueError(f"Missing required dataset index columns: {', '.join(missing)}")
        missing_id_cols = [column for column in SAMPLE_ID_COLUMNS if column not in index.columns]
        if missing_id_cols:
            raise ValueError(f"Missing sample-id columns: {', '.join(missing_id_cols)}")

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        state["_h5"] = None
        return state

    def close(self) -> None:
        if self._h5 is not None:
            self._h5.close()
            self._h5 = None

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass
