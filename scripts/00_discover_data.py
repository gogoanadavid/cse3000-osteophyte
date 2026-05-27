#!/usr/bin/env python3
"""Discover candidate metadata and split files on DelftBlue.

This script only inspects paths and filenames. It intentionally does not load
H5/HDF5 contents because the imaging data is confidential and large.
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from osteophytes.config import load_config
from osteophytes.data_index import find_candidate_files


CONFIG_PATH = PROJECT_DIR / "configs" / "delftblue.yaml"


def print_path_status(name: str, path: Path) -> None:
    print(f"{name}: {path}")
    print(f"  exists: {path.exists()}")


def print_candidates(label: str, paths: list[Path]) -> None:
    print(f"\n{label} ({len(paths)} found)")
    for path in paths:
        print(f"  {path}")


def main() -> None:
    config = load_config(CONFIG_PATH)

    print("Configured DelftBlue paths")
    print_path_status("project_root", config.project_root)
    print_path_status("source_data_root", config.source_data_root)
    print_path_status("scratch_root", config.scratch_root)

    print("\nScanning project_root for candidate files...")
    candidates = find_candidate_files(config.project_root)
    print_candidates("CSV files", candidates["csv"])
    print_candidates("H5/HDF5 files", candidates["hdf5"])
    print_candidates("Split/list files", candidates["splits"])


if __name__ == "__main__":
    main()
