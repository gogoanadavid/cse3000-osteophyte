"""Inspect HDF5 structure without loading image arrays into memory."""

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


def _format_value(value: Any) -> str:
    if isinstance(value, bytes):
        return repr(value.decode("utf-8", errors="replace"))
    if isinstance(value, np.ndarray):
        if value.dtype.kind == "S":
            return repr([x.decode("utf-8", errors="replace") for x in value.tolist()])
        return repr(value.tolist())
    return repr(value)


def inspect_h5(path: str | Path) -> list[str]:
    """Return text lines describing groups, datasets, shapes, and examples."""
    lines: list[str] = []
    path = Path(path)
    with h5py.File(path, "r") as h5:
        lines.append(f"HDF5 file: {path}")

        def visitor(name: str, obj: h5py.Group | h5py.Dataset) -> None:
            prefix = "/" + name
            if isinstance(obj, h5py.Group):
                lines.append(f"{prefix}/  group")
                return
            shape = obj.shape
            dtype = obj.dtype
            line = f"{prefix}  dataset shape={shape} dtype={dtype}"
            if shape == ():
                try:
                    line += f" example={_format_value(obj[()])}"
                except Exception as exc:
                    line += f" example=<unreadable {type(exc).__name__}: {exc}>"
            elif len(shape) == 1:
                n = min(int(shape[0]), 8)
                try:
                    line += f" examples={_format_value(obj[:n])}"
                except Exception as exc:
                    line += f" examples=<unreadable {type(exc).__name__}: {exc}>"
            lines.append(line)

        h5.visititems(visitor)
    return lines


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--h5", required=True)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    lines = inspect_h5(args.h5)
    text = "\n".join(lines) + "\n"
    print(text, end="")
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
