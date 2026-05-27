"""Configuration loading helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DelftBlueConfig:
    project_root: Path
    source_data_root: Path
    scratch_root: Path
    csv_path: Path | None = None
    h5_path: Path | None = None
    split_path: Path | None = None


def _optional_path(values: dict[str, str], key: str) -> Path | None:
    value = values.get(key)
    if not value:
        return None
    return Path(value)


def _read_simple_yaml(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        key, sep, value = stripped.partition(":")
        if not sep:
            raise ValueError(f"Invalid config line in {path}: {line!r}")
        values[key.strip()] = value.strip()
    return values


def load_config(path: str | Path) -> DelftBlueConfig:
    """Load the DelftBlue path config from the minimal YAML file."""
    config_path = Path(path)
    values = _read_simple_yaml(config_path)
    required = ("project_root", "source_data_root", "scratch_root")
    missing = [key for key in required if key not in values]
    if missing:
        raise ValueError(f"Missing required config keys: {', '.join(missing)}")
    return DelftBlueConfig(
        project_root=Path(values["project_root"]),
        source_data_root=Path(values["source_data_root"]),
        scratch_root=Path(values["scratch_root"]),
        csv_path=_optional_path(values, "csv_path"),
        h5_path=_optional_path(values, "h5_path"),
        split_path=_optional_path(values, "split_path"),
    )
