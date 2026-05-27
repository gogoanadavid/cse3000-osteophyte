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


def _read_image_array(h5: h5py.File, image_key: str, row: pd.Series | dict[str, Any] | int) -> np.ndarray:
    if isinstance(row, int):
        image = np.asarray(h5[image_key][row])
    else:
        path_key = row.get("h5_path_key") if isinstance(row, dict) else row.get("h5_path_key", None)
        if path_key is not None and isinstance(path_key, str) and path_key:
            image = np.asarray(h5[path_key])
        else:
            image = np.asarray(h5[image_key][int(row["h5_index"])])
    return np.squeeze(image).astype(np.float32)


def _compute_train_mean_std(
    h5: h5py.File,
    image_key: str,
    train_rows: pd.DataFrame | list[int],
    percentile_clip: list[float] | None,
    max_samples: int | None = None,
    seed: int = 0,
) -> dict[str, float]:
    total = 0
    sum_x = 0.0
    sum_x2 = 0.0
    sampled_count = len(train_rows)
    if max_samples is not None and max_samples > 0 and len(train_rows) > max_samples:
        if isinstance(train_rows, pd.DataFrame):
            train_rows = train_rows.sample(n=max_samples, random_state=seed).sort_values("h5_index")
        else:
            rng = np.random.default_rng(seed)
            sampled = rng.choice(np.asarray(train_rows), size=max_samples, replace=False)
            train_rows = sorted(int(x) for x in sampled.tolist())
        sampled_count = max_samples
    iterator = train_rows.iterrows() if isinstance(train_rows, pd.DataFrame) else enumerate(train_rows)
    for _, row in iterator:
        image = _read_image_array(h5, image_key, row if isinstance(train_rows, pd.DataFrame) else int(row))
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
        return {"mean": 0.0, "std": 1.0, "num_pixels": 0, "num_samples_used": 0}
    mean = sum_x / total
    var = max(sum_x2 / total - mean * mean, 1e-12)
    return {
        "mean": float(mean),
        "std": float(np.sqrt(var)),
        "num_pixels": int(total),
        "num_samples_used": int(sampled_count),
        "mean_std_max_samples": int(max_samples) if max_samples is not None else None,
    }


def _visit_aliases(visit: Any) -> set[str]:
    text = str(_decode(visit))
    aliases = {text}
    if text.startswith("V"):
        aliases.add("T" + text[1:])
    if text.startswith("T"):
        aliases.add("V" + text[1:])
    return aliases


def _read_split_file(path: str | Path) -> dict[str, str]:
    split_map: dict[str, str] = {}
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = [p.strip() for p in line.replace("\t", ",").split(",") if p.strip()]
            if len(parts) < 2:
                raise ValueError(f"Could not parse split line: {line!r}")
            split_map[parts[0]] = parts[1]
    return split_map


def _build_label_lookup(labels_csv: str | Path, data_config: dict[str, Any]) -> dict[tuple[str, str, str], dict[str, int]]:
    labels = pd.read_csv(labels_csv)
    subject_col = data_config.get("label_subject_col", "subject")
    visit_col = data_config.get("label_visit_col", "visit")
    side_col = data_config.get("label_side_col", "side")
    required = {subject_col, visit_col, side_col, *data_config["grade_label_keys"].values()}
    missing = required - set(labels.columns)
    if missing:
        raise KeyError(f"Label CSV is missing columns: {sorted(missing)}")

    lookup: dict[tuple[str, str, str], dict[str, int]] = {}
    for _, row in labels.iterrows():
        subject = str(row[subject_col])
        side = str(row[side_col]).lower()
        values = {
            f"grade_{loc}": _normalize_label(row[col])
            for loc, col in data_config["grade_label_keys"].items()
        }
        for visit in _visit_aliases(row[visit_col]):
            lookup[(subject, visit, side)] = values
    return lookup


def _validate_and_finalize_row(row: dict[str, Any], locations: list[str], warnings: list[str]) -> dict[str, Any]:
    max_grade = -1
    any_binary_positive = False
    complete = True
    h5_index = row["h5_index"]
    for loc in locations:
        binary = _normalize_label(row[f"binary_{loc}"])
        grade = _normalize_label(row[f"grade_{loc}"])
        if binary not in {-1, 0, 1}:
            raise ValueError(f"Invalid binary label {binary} for {loc} at h5_index={h5_index}")
        if grade not in {-1, 0, 1, 2, 3}:
            raise ValueError(f"Invalid grade label {grade} for {loc} at h5_index={h5_index}")
        if grade == -1:
            complete = False
        else:
            max_grade = max(max_grade, grade)
            expected_binary = 1 if grade > 0 else 0
            if binary != -1 and binary != expected_binary:
                warnings.append(
                    f"h5_index={h5_index} loc={loc}: grade={grade} implies binary={expected_binary}, got {binary}"
                )
        if binary == 1:
            any_binary_positive = True
        row[f"binary_{loc}"] = binary
        row[f"grade_{loc}"] = grade
    row["has_complete_grades"] = int(complete)
    row["max_grade"] = int(max_grade if max_grade >= 0 else -1)
    row["any_binary_positive"] = int(any_binary_positive)
    return row


def _build_hierarchical_csv_index(
    h5: h5py.File,
    data_config: dict[str, Any],
    warnings: list[str],
) -> pd.DataFrame:
    scan_root = data_config.get("scan_root", "scans")
    if scan_root not in h5:
        raise KeyError(f"scan_root '{scan_root}' does not exist in HDF5")
    labels_csv = data_config.get("labels_csv")
    split_file = data_config.get("split_file")
    if not labels_csv:
        raise KeyError("hierarchical_csv schema requires labels_csv in the data config")
    if not split_file:
        raise KeyError("hierarchical_csv schema requires split_file in the data config")
    label_lookup = _build_label_lookup(labels_csv, data_config)
    split_map = _read_split_file(split_file)
    locations = list(data_config["locations"])

    rows: list[dict[str, Any]] = []
    idx = 0
    root = h5[scan_root]
    for subject in sorted(root.keys()):
        split = split_map.get(subject)
        if split is None:
            warnings.append(f"subject_id={subject}: missing from split file; skipping all scans")
            continue
        for visit in sorted(root[subject].keys()):
            for side in sorted(root[subject][visit].keys()):
                image_path = f"{scan_root}/{subject}/{visit}/{side}/image"
                if image_path not in h5:
                    warnings.append(f"{scan_root}/{subject}/{visit}/{side}: missing image dataset; skipping")
                    continue
                label_values = label_lookup.get((subject, visit, side.lower()))
                row: dict[str, Any] = {
                    "h5_index": idx,
                    "h5_path_key": image_path,
                    "split": split,
                    "subject_id": subject,
                    "visit_id": visit,
                    "side": side,
                }
                if label_values is None:
                    warnings.append(f"{image_path}: no matching label CSV row; labels set to -1")
                    for loc in locations:
                        row[f"grade_{loc}"] = -1
                        row[f"binary_{loc}"] = -1
                else:
                    for loc in locations:
                        grade = int(label_values[f"grade_{loc}"])
                        row[f"grade_{loc}"] = grade
                        row[f"binary_{loc}"] = -1 if grade == -1 else int(grade > 0)
                rows.append(_validate_and_finalize_row(row, locations, warnings))
                idx += 1
    if not rows:
        raise ValueError("No indexed HDF5 image rows were produced")
    return pd.DataFrame(rows)


def _build_flat_h5_index(h5: h5py.File, data_config: dict[str, Any], warnings: list[str]) -> pd.DataFrame:
    locations = list(data_config["locations"])
    image_key = data_config["image_key"]
    if image_key not in h5:
        raise KeyError(f"Image key '{image_key}' does not exist")
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
        for loc in locations:
            row[f"binary_{loc}"] = _normalize_label(binary_arrays[loc][i])
            row[f"grade_{loc}"] = _normalize_label(grade_arrays[loc][i])
        rows.append(_validate_and_finalize_row(row, locations, warnings))
    return pd.DataFrame(rows)


def build_index(data_config: dict[str, Any], out_csv: str | Path) -> tuple[pd.DataFrame, list[str]]:
    warnings: list[str] = []
    h5_path = Path(data_config["h5_path"])
    if not h5_path.exists():
        raise FileNotFoundError(f"HDF5 file does not exist: {h5_path}")

    out_csv = Path(out_csv)
    with h5py.File(h5_path, "r") as h5:
        schema = data_config.get("schema", "flat_h5")
        if schema == "hierarchical_csv":
            df = _build_hierarchical_csv_index(h5, data_config, warnings)
        elif schema == "flat_h5":
            df = _build_flat_h5_index(h5, data_config, warnings)
        else:
            raise ValueError(f"Unknown data schema: {schema}")
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_csv, index=False)

        stats_path = Path(data_config.get("train_mean_std_json", "outputs/train_mean_std.json"))
        clip = data_config.get("percentile_clip")
        train_rows = df[df["split"] == "train"].copy()
        stats = _compute_train_mean_std(
            h5,
            data_config.get("image_key", ""),
            train_rows,
            clip,
            max_samples=data_config.get("mean_std_max_samples"),
            seed=int(data_config.get("mean_std_seed", 0)),
        )
        stats["image_key"] = data_config.get("image_key", "")
        stats["schema"] = data_config.get("schema", "flat_h5")
        stats["num_train_samples"] = int(len(train_rows))
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
