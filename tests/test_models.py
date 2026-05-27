from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


def test_model_output_shapes_and_coral_monotonicity() -> None:
    torch = pytest.importorskip("torch")
    from osteophytes.models import (
        BinaryBaselineResNet18,
        CoralOrdinalResNet18,
        DualHeadResNet18,
        OrdinalThresholdResNet18,
    )

    x = torch.randn(2, 1, 224, 224)
    with torch.no_grad():
        binary = BinaryBaselineResNet18()(x)
        threshold = OrdinalThresholdResNet18()(x)
        coral_logits = CoralOrdinalResNet18()(x)
        dual = DualHeadResNet18()(x)

    assert binary.shape == (2, 4)
    assert threshold.shape == (2, 4, 3)
    assert coral_logits.shape == (2, 4, 3)
    assert dual["binary_logits"].shape == (2, 4)
    assert dual["ordinal_logits"].shape == (2, 4, 3)

    coral_probs = torch.sigmoid(coral_logits)
    assert torch.all(coral_probs[:, :, 0] >= coral_probs[:, :, 1])
    assert torch.all(coral_probs[:, :, 1] >= coral_probs[:, :, 2])
