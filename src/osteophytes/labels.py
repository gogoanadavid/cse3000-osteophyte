"""Label constants and small helpers for osteophyte experiments."""

from __future__ import annotations

from collections.abc import Mapping
from math import isnan
from numbers import Integral, Real
from typing import Any


LOCATION_NAMES: tuple[str, str, str, str] = (
    "osteo_acet_inf",
    "osteo_acet_sup",
    "osteo_fem_inf",
    "osteo_fem_sup",
)

LOCATION_DISPLAY_NAMES: dict[str, str] = {
    "osteo_acet_inf": "Inferior acetabular",
    "osteo_acet_sup": "Superior acetabular",
    "osteo_fem_inf": "Inferior femoral",
    "osteo_fem_sup": "Superior femoral",
}

BINARY_LABEL_COLUMNS: tuple[str, str, str, str] = (
    "osteo_acet_inf_binary",
    "osteo_acet_sup_binary",
    "osteo_fem_inf_binary",
    "osteo_fem_sup_binary",
)

NUM_LOCATIONS = 4
NUM_GRADES = 4
NUM_THRESHOLDS = 3

SAMPLE_ID_COLUMNS: tuple[str, str, str] = ("subject", "visit", "side")

# Backward-compatible name used by the initial audit scripts.
OSTEOPHYTE_LABEL_COLUMNS = LOCATION_NAMES

VALID_GRADES: set[int] = {0, 1, 2, 3}


def _is_missing(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return not value.strip()
    try:
        return bool(isnan(value))
    except TypeError:
        return False


def _parse_grade(value: Any) -> int | None:
    if _is_missing(value):
        return None
    if isinstance(value, bool):
        raise ValueError(f"Invalid osteophyte grade: {value!r}")
    if isinstance(value, Integral):
        grade = int(value)
    elif isinstance(value, Real) and float(value).is_integer():
        grade = int(value)
    elif isinstance(value, str) and value.strip() in {"0", "1", "2", "3"}:
        grade = int(value.strip())
    else:
        raise ValueError(f"Invalid osteophyte grade: {value!r}")

    if grade not in VALID_GRADES:
        raise ValueError(f"Invalid osteophyte grade: {value!r}")
    return grade


def coerce_grade(value: Any, column: str = "grade") -> int | None:
    """Return an integer grade, None for missing, or raise for invalid values."""
    try:
        return _parse_grade(value)
    except ValueError as exc:
        raise ValueError(f"Invalid value in {column!r}: {value!r}") from exc


def grade_to_binary(grade: Any) -> int | None:
    """Map an osteophyte grade to a binary absence/presence label."""
    parsed = _parse_grade(grade)
    if parsed is None:
        return None
    return 0 if parsed == 0 else 1


def is_complete_graded_annotation(row: Mapping[str, Any]) -> bool:
    """Return True when all osteophyte locations have valid grades."""
    for column in LOCATION_NAMES:
        try:
            parsed = _parse_grade(row[column])
        except (KeyError, ValueError):
            return False
        if parsed is None:
            return False
    return True


def _sample_value(container: Mapping[str, Any], column: str) -> Any:
    value = container[column]
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return value


def make_sample_id_from_row(row: Mapping[str, Any]) -> str:
    """Build the deterministic hip identifier used across all experiments."""
    return "|".join(str(_sample_value(row, column)) for column in SAMPLE_ID_COLUMNS)


def build_sample_id(sample: Mapping[str, Any]) -> str:
    """Build the deterministic hip identifier from a dataset sample."""
    return "|".join(str(_sample_value(sample, column)) for column in SAMPLE_ID_COLUMNS)
