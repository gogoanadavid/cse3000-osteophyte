from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


def test_ordinal_target_encoding_and_probabilities() -> None:
    torch = pytest.importorskip("torch")

    from osteophytes.ordinal import (
        encode_ordinal_thresholds,
        enforce_monotonic_thresholds,
        expected_grade_from_thresholds,
        thresholds_to_grade_probabilities,
    )

    grades = torch.tensor([[0, 1, 2, 3]])
    mask = torch.ones_like(grades, dtype=torch.bool)
    targets, threshold_mask = encode_ordinal_thresholds(grades, mask)
    expected = torch.tensor(
        [
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [1.0, 1.0, 0.0],
                [1.0, 1.0, 1.0],
            ]
        ]
    )

    assert torch.equal(targets, expected)
    assert threshold_mask.shape == targets.shape

    violating = torch.tensor([[[0.2, 0.8, 0.6]]]).repeat(1, 4, 1)
    corrected = enforce_monotonic_thresholds(violating)
    assert torch.all(corrected[:, :, 0] >= corrected[:, :, 1])
    assert torch.all(corrected[:, :, 1] >= corrected[:, :, 2])
    grade_probs = thresholds_to_grade_probabilities(corrected)
    assert torch.allclose(grade_probs.sum(dim=2), torch.ones(1, 4))
    assert torch.allclose(expected_grade_from_thresholds(corrected), corrected.sum(dim=2))
