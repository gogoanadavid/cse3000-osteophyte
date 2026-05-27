"""Helpers for finding candidate data files without loading image arrays."""

from __future__ import annotations

from pathlib import Path


CSV_EXTENSIONS = {".csv"}
HDF5_EXTENSIONS = {".h5", ".hdf5"}
SPLIT_LIST_EXTENSIONS = {".txt", ".tsv", ".json", ".jsonl", ".yaml", ".yml"}


def find_candidate_files(root: Path) -> dict[str, list[Path]]:
    """Return likely metadata, array, and split/list files under a root."""
    candidates = {"csv": [], "hdf5": [], "splits": []}
    if not root.exists():
        return candidates

    for path in root.rglob("*"):
        if not path.is_file():
            continue
        suffix = path.suffix.lower()
        if suffix in CSV_EXTENSIONS:
            candidates["csv"].append(path)
        elif suffix in HDF5_EXTENSIONS:
            candidates["hdf5"].append(path)
        elif suffix in SPLIT_LIST_EXTENSIONS or "split" in path.name.lower():
            candidates["splits"].append(path)

    return {key: sorted(paths) for key, paths in candidates.items()}
