#!/usr/bin/env python3
"""Create PNG visual audit grids from the H5-backed dataset index."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from osteophytes.config import load_config
from osteophytes.labels import OSTEOPHYTE_LABEL_COLUMNS


CONFIG_PATH = PROJECT_DIR / "configs" / "delftblue.yaml"
GRADES = (0, 1, 2, 3)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index-path", type=Path, help="Path to dataset_index.csv.")
    parser.add_argument("--h5-path", type=Path, help="Path to the H5 image file.")
    parser.add_argument("--output-dir", type=Path, help="Directory for visual audit PNGs.")
    parser.add_argument("--examples-per-grade", type=int, default=6, help="Number of examples per grade row.")
    parser.add_argument("--seed", type=int, default=123, help="Random seed for reproducible sampling.")
    parser.add_argument(
        "--locations",
        nargs="+",
        choices=OSTEOPHYTE_LABEL_COLUMNS,
        default=list(OSTEOPHYTE_LABEL_COLUMNS),
        help="Osteophyte label columns to audit.",
    )
    return parser.parse_args()


def require_path(name: str, path: Path | None) -> Path:
    if path is None:
        raise ValueError(f"Missing {name}. Pass --{name.replace('_', '-')} or set it in configs/delftblue.yaml.")
    return path


def verify_index_columns(index: Any) -> None:
    required = {"subject", "visit", "side", "split", "h5_internal_path", *OSTEOPHYTE_LABEL_COLUMNS}
    missing = [column for column in required if column not in index.columns]
    if missing:
        raise ValueError(f"Missing required dataset index columns: {', '.join(missing)}")


def normalize_to_uint8(image: Any, np: Any) -> Any:
    array = np.asarray(image)
    array = np.squeeze(array)
    if array.ndim == 3 and array.shape[-1] in {3, 4}:
        array = array[..., :3]
    elif array.ndim != 2:
        raise ValueError(f"Unsupported image shape: {array.shape}")

    array = array.astype("float32")
    finite = np.isfinite(array)
    if not finite.any():
        return np.zeros(array.shape, dtype="uint8")

    low, high = np.percentile(array[finite], [1, 99])
    if high <= low:
        low = float(np.min(array[finite]))
        high = float(np.max(array[finite]))
    if high <= low:
        return np.zeros(array.shape, dtype="uint8")

    array = np.clip((array - low) / (high - low), 0.0, 1.0)
    return (array * 255).astype("uint8")


def read_image_tile(h5_file: Any, internal_path: str, tile_size: int, np: Any, Image: Any) -> Any:
    if internal_path not in h5_file:
        raise KeyError(f"H5 image path not found: {internal_path}")
    image = normalize_to_uint8(h5_file[internal_path][()], np)
    pil_image = Image.fromarray(image)
    if pil_image.mode != "L":
        pil_image = pil_image.convert("L")
    return pil_image.resize((tile_size, tile_size))


def make_caption(row: Any) -> str:
    return f"{row.subject} | {row.visit} | {row.side} | {row.split}"


def draw_tile(tile: Any, caption: str, grade: int, Image: Any, ImageDraw: Any) -> Any:
    tile_size = tile.size[0]
    caption_height = 34
    canvas = Image.new("RGB", (tile_size, tile_size + caption_height), "white")
    canvas.paste(tile.convert("RGB"), (0, 0))
    draw = ImageDraw.Draw(canvas)
    draw.rectangle((0, 0, tile_size - 1, tile_size - 1), outline=(30, 30, 30), width=1)
    draw.text((4, tile_size + 3), f"g{grade} {caption}", fill=(0, 0, 0))
    return canvas


def blank_tile(message: str, tile_size: int, Image: Any, ImageDraw: Any) -> Any:
    caption_height = 34
    canvas = Image.new("RGB", (tile_size, tile_size + caption_height), (245, 245, 245))
    draw = ImageDraw.Draw(canvas)
    draw.rectangle((0, 0, tile_size - 1, tile_size - 1), outline=(180, 180, 180), width=1)
    draw.text((8, 8), message, fill=(80, 80, 80))
    return canvas


def save_location_grid(
    location: str,
    selected: dict[int, Any],
    h5_file: Any,
    output_path: Path,
    examples_per_grade: int,
    np: Any,
    Image: Any,
    ImageDraw: Any,
) -> list[dict[str, Any]]:
    tile_size = 224
    caption_height = 34
    label_width = 90
    title_height = 34
    gap = 8
    width = label_width + examples_per_grade * tile_size + (examples_per_grade + 1) * gap
    row_height = tile_size + caption_height
    height = title_height + len(GRADES) * row_height + (len(GRADES) + 1) * gap
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((8, 8), f"{location}: examples by grade", fill=(0, 0, 0))

    manifest_rows: list[dict[str, Any]] = []
    for row_index, grade in enumerate(GRADES):
        y = title_height + gap + row_index * (row_height + gap)
        draw.text((8, y + 8), f"grade {grade}", fill=(0, 0, 0))
        grade_rows = selected[grade]
        for col_index in range(examples_per_grade):
            x = label_width + gap + col_index * (tile_size + gap)
            if col_index >= len(grade_rows):
                tile = blank_tile("no sample", tile_size, Image, ImageDraw)
            else:
                row = grade_rows.iloc[col_index]
                image = read_image_tile(h5_file, row.h5_internal_path, tile_size, np, Image)
                tile = draw_tile(image, make_caption(row), grade, Image, ImageDraw)
                manifest_rows.append(
                    {
                        "location": location,
                        "grade": grade,
                        "subject": str(row.subject),
                        "visit": str(row.visit),
                        "side": str(row.side),
                        "split": str(row.split),
                        "h5_internal_path": str(row.h5_internal_path),
                    }
                )
            canvas.paste(tile, (x, y))

    canvas.save(output_path)
    return manifest_rows


def main() -> None:
    args = parse_args()
    config = load_config(CONFIG_PATH)
    if args.examples_per_grade < 1:
        raise ValueError("--examples-per-grade must be at least 1")

    global pd
    import hdf5plugin  # registers Blosc2 HDF5 filter for h5py
    import h5py
    import numpy as np
    import pandas as pd
    from PIL import Image, ImageDraw

    index_path = args.index_path or (config.scratch_root / "audits" / "dataset_index.csv")
    h5_path = args.h5_path or config.h5_path
    output_dir = args.output_dir or (config.scratch_root / "audits" / "visual_audit")

    index_path = require_path("index_path", index_path)
    h5_path = require_path("h5_path", h5_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    index = pd.read_csv(index_path)
    verify_index_columns(index)

    print(f"Loaded dataset index: {index_path}")
    print(f"Rows: {len(index)}")
    print(f"Output directory: {output_dir}")

    manifest: dict[str, Any] = {
        "index_path": str(index_path),
        "h5_path": str(h5_path),
        "examples_per_grade": args.examples_per_grade,
        "seed": args.seed,
        "locations": {},
    }

    with h5py.File(h5_path, "r") as h5_file:
        for location in args.locations:
            selected: dict[int, Any] = {}
            location_grades = pd.to_numeric(index[location], errors="coerce")
            print(f"\n{location}")
            for grade in GRADES:
                grade_rows = index[location_grades == grade]
                sample_count = min(args.examples_per_grade, len(grade_rows))
                if sample_count:
                    selected[grade] = grade_rows.sample(
                        n=sample_count,
                        random_state=args.seed + grade,
                    ).reset_index(drop=True)
                else:
                    selected[grade] = grade_rows.head(0).reset_index(drop=True)
                print(f"  grade {grade}: sampled {sample_count} of {len(grade_rows)}")

            output_path = output_dir / f"{location}_visual_audit.png"
            manifest_rows = save_location_grid(
                location=location,
                selected=selected,
                h5_file=h5_file,
                output_path=output_path,
                examples_per_grade=args.examples_per_grade,
                np=np,
                Image=Image,
                ImageDraw=ImageDraw,
            )
            manifest["locations"][location] = {
                "png_path": str(output_path),
                "samples": manifest_rows,
            }
            print(f"  saved: {output_path}")

    manifest_path = output_dir / "visual_audit_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"\nSaved manifest: {manifest_path}")


if __name__ == "__main__":
    main()
