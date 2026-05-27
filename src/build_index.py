"""Build a flat metadata index from a configurable HDF5 schema."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

try:
    import hdf5plugin  # noqa: F401
except Exception:
    hdf5plugin = None  # type: ignore

import h5py
import numpy as np
import pandas as pd

from .config import load_config
from .utils import save_json, safe_mkdir


def _decode(value: Any) -> Any:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, np.generic):
        value = value.item()
        if isinstance(value, bytes):
            return value.decode("utf-8")
    return value


def _normalize_label(value: Any, missing_value: int = -1) -> int:
    value = _decode(value)
    if value is None:
        return missing_value
    if isinstance(value, str):
        value = value.strip()
        if value == "" or value.lower() in {"nan", "none", "missing", "na"}:
            return missing_value
    try:
        if np.isnan(value):  # type: ignore[arg-type]
            return missing_value
    except Exception:
        pass
    return int(value)


def _matches(value: Any, options: list[Any]) -> bool:
    value = _decode(value)
    text = str(value).lower()
    return any(value == _decode(opt) or text == str(_decode(opt)).lower() for opt in options)


def _read_1d(h5: h5py.File, key: str, n: int) -> np.ndarray:
    if key not in h5:
        raise KeyError(f"Required HDF5 key '{key}' does not exist")
    arr = h5[key]
    if arr.shape[0] != n:
        raise ValueError(f"Dataset '{key}' has first dimension {arr.shape[0]}, expected {n}")
    return arr[:]


def _resolve_split(raw: Any, split_values: dict[str, list[Any]]) -> str:
    for split, options in split_values.items():
        if _matches(raw, options):
            return split
    raise ValueError(f"Unknown split value {raw!r}; update split_values in data config")


def _compute_train_mean_std(
    h5: h5py.File,
    image_key: str,
    train_indices: list[int],
    percentile_clip: list[float] | None,
) -> dict[str, float]:
    images = h5[image_key]
    total = 0
    sum_x = 0.0
    sum_x2 = 0.0
    for idx in train_indices:
        image = np.asarray(images[int(idx)]).squeeze().astype(np.float32)
        if image.max(initial=0.0) > 10.0:
            image /= 255.0
        if percentile_clip is not None:
            lo, hi = np.percentile(image, [float(percentile_clip[0]), float(percentile_clip[1])])
            if hi > lo:
                image = np.clip(image, lo, hi)
                image = (image - lo) / max(hi - lo, 1e-6)
        total += image.size
        sum_x += float(image.sum())
        sum_x2 += float(np.square(image).sum())
    if total == 0:
        return {"mean": 0.0, "std": 1.0, "num_pixels": 0}
    mean = sum_x / total
    var = max(sum_x2 / total - mean * mean, 1e-12)
    return {"mean": float(mean), "std": float(np.sqrt(var)), "num_pixels": int(total)}


def build_index(data_config: dict[str, Any], out_csv: str | Path) -> tuple[pd.DataFrame, list[str]]:
    warnings: list[str] = []
    h5_path = Path(data_config["h5_path"])
    if not h5_path.exists():
        raise FileNotFoundError(f"HDF5 file does not exist: {h5_path}")

    locations = list(data_config["locations"])
    out_csv = Path(out_csv)
    with h5py.File(h5_path, "r") as h5:
        image_key = data_config["image_key"]
        if image_key not in h5:
            raise KeyError(f"Image key '{image_key}' does not exist in {h5_path}")
        n = int(h5[image_key].shape[0])
        split_raw = _read_1d(h5, data_config["split_key"], n)
        subject_raw = _read_1d(h5, data_config["subject_id_key"], n)
        visit_raw = _read_1d(h5, data_config["visit_id_key"], n)
        side_raw = _read_1d(h5, data_config["side_key"], n)

        binary_arrays = {loc: _read_1d(h5, data_config["binary_label_keys"][loc], n) for loc in locations}
        grade_arrays = {loc: _read_1d(h5, data_config["grade_label_keys"][loc], n) for loc in locations}

        rows: list[dict[str, Any]] = []
        for i in range(n):
            row: dict[str, Any] = {
                "h5_index": i,
                "split": _resolve_split(split_raw[i], data_config["split_values"]),
                "subject_id": _decode(subject_raw[i]),
                "visit_id": _decode(visit_raw[i]),
                "side": _decode(side_raw[i]),
            }
            max_grade = -1
            any_binary_positive = False
            complete = True
            for loc in locations:
                binary = _normalize_label(binary_arrays[loc][i])
                grade = _normalize_label(grade_arrays[loc][i])
                if binary not in {-1, 0, 1}:
                    raise ValueError(f"Invalid binary label {binary} for {loc} at h5_index={i}")
                if grade not in {-1, 0, 1, 2, 3}:
                    raise ValueError(f"Invalid grade label {grade} for {loc} at h5_index={i}")
                if grade == -1:
                    complete = False
                else:
                    max_grade = max(max_grade, grade)
                    expected_binary = 1 if grade > 0 else 0
                    if binary != -1 and binary != expected_binary:
                        warnings.append(
                            f"h5_index={i} loc={loc}: grade={grade} implies binary={expected_binary}, got {binary}"
                        )
                if binary == 1:
                    any_binary_positive = True
                row[f"binary_{loc}"] = binary
                row[f"grade_{loc}"] = grade
            row["has_complete_grades"] = int(complete)
            row["max_grade"] = int(max_grade if max_grade >= 0 else -1)
            row["any_binary_positive"] = int(any_binary_positive)
            rows.append(row)

        df = pd.DataFrame(rows)
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_csv, index=False)

        stats_path = Path(data_config.get("train_mean_std_json", "outputs/train_mean_std.json"))
        clip = data_config.get("percentile_clip")
        train_indices = df.loc[df["split"] == "train", "h5_index"].astype(int).tolist()
        stats = _compute_train_mean_std(h5, image_key, train_indices, clip)
        stats["image_key"] = image_key
        stats["num_train_samples"] = int(len(train_indices))
        save_json(stats, stats_path)

    warning_path = Path("outputs/logs/build_index_warnings.txt")
    safe_mkdir(warning_path.parent)
    warning_path.write_text("\n".join(warnings) + ("\n" if warnings else ""), encoding="utf-8")
    return df, warnings


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-config", required=True)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()
    config = load_config(args.data_config)
    out = args.out or config.get("index_csv", "outputs/index.csv")
    df, warnings = build_index(config, out)
    print(f"Wrote {len(df)} rows to {out}")
    print(f"Wrote {len(warnings)} warnings to outputs/logs/build_index_warnings.txt")


if __name__ == "__main__":
    main()
