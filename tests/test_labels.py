from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest

PROJECT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from osteophytes.labels import (
    LOCATION_NAMES,
    SAMPLE_ID_COLUMNS,
    build_sample_id,
    grade_to_binary,
    is_complete_graded_annotation,
    make_sample_id_from_row,
)


@pytest.mark.parametrize(
    ("grade", "expected"),
    [
        (0, 0),
        (1, 1),
        (2, 1),
        (3, 1),
        (0.0, 0),
        (3.0, 1),
        ("2", 1),
        (None, None),
        (math.nan, None),
    ],
)
def test_grade_to_binary_valid_and_missing(grade: object, expected: int | None) -> None:
    assert grade_to_binary(grade) == expected


@pytest.mark.parametrize("grade", [-1, 4, 1.5, "bad", True])
def test_grade_to_binary_invalid(grade: object) -> None:
    with pytest.raises(ValueError):
        grade_to_binary(grade)


def test_is_complete_graded_annotation_true_for_all_valid_grades() -> None:
    row = {
        "osteo_acet_inf": 0,
        "osteo_acet_sup": 1,
        "osteo_fem_inf": 2,
        "osteo_fem_sup": 3,
    }

    assert is_complete_graded_annotation(row)


def test_is_complete_graded_annotation_false_for_missing_or_invalid() -> None:
    missing = {column: 1 for column in LOCATION_NAMES}
    missing["osteo_fem_sup"] = None
    invalid = {column: 1 for column in LOCATION_NAMES}
    invalid["osteo_fem_sup"] = 4

    assert not is_complete_graded_annotation(missing)
    assert not is_complete_graded_annotation(invalid)


def test_sample_id_uses_subject_visit_side_only() -> None:
    row = {
        "subject": "s1",
        "visit": "00",
        "side": "L",
        "h5_internal_path": "scans/s1/00/L/image",
    }
    sample = {"subject": "s1", "visit": "00", "side": "L"}

    assert SAMPLE_ID_COLUMNS == ("subject", "visit", "side")
    assert make_sample_id_from_row(row) == "s1|00|L"
    assert build_sample_id(sample) == "s1|00|L"
    assert make_sample_id_from_row(row) == build_sample_id(sample)
    assert "|" in make_sample_id_from_row(row)
    assert "::" not in make_sample_id_from_row(row)
