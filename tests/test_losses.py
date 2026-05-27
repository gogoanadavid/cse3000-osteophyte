from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


def make_batch(torch, is_strong: list[bool]) -> dict[str, object]:
    grades = torch.tensor(
        [
            [0, 1, 2, 3],
            [1, 0, 3, 2],
            [2, 3, 0, 1],
        ],
        dtype=torch.long,
    )
    return {
        "graded_labels": grades,
        "graded_mask": torch.ones_like(grades, dtype=torch.float32),
        "binary_labels": (grades > 0).float(),
        "binary_mask": torch.ones_like(grades, dtype=torch.float32),
        "is_strong": torch.tensor(is_strong, dtype=torch.bool),
    }


def test_noisy_or_image_level_weak_loss() -> None:
    torch = pytest.importorskip("torch")
    from osteophytes.losses import binary_image_noisy_or_loss

    logits = torch.zeros(2, 4)
    binary_labels = torch.tensor([[0.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]])
    binary_mask = torch.ones_like(binary_labels)
    sample_mask = torch.tensor([True, True])

    loss, count = binary_image_noisy_or_loss(logits, binary_labels, binary_mask, sample_mask)
    assert count == 2
    assert torch.isfinite(loss)


@pytest.mark.parametrize(
    ("is_strong", "expected_ordinal", "expected_weak"),
    [
        ([True, True, True], 36, 0),
        ([False, False, False], 0, 12),
        ([True, False, True], 24, 4),
    ],
)
def test_mixed_loss_handles_batch_compositions(
    is_strong: list[bool],
    expected_ordinal: int,
    expected_weak: int,
) -> None:
    torch = pytest.importorskip("torch")
    from osteophytes.losses import mixed_supervision_loss

    logits = torch.randn(3, 4, 3, requires_grad=True)
    batch = make_batch(torch, is_strong)
    loss, parts = mixed_supervision_loss(logits, batch, weak_label_mode="location_binary")

    assert torch.isfinite(loss)
    assert parts["ordinal_count"] == expected_ordinal
    assert parts["weak_count"] == expected_weak
    loss.backward()
    assert logits.grad is not None


def test_loss_balance_proportional_reports_expected_scales() -> None:
    torch = pytest.importorskip("torch")
    from osteophytes.losses import mixed_supervision_loss

    logits = torch.randn(3, 4, 3, requires_grad=True)
    batch = make_batch(torch, [True, False, False])
    _, parts = mixed_supervision_loss(
        logits,
        batch,
        weak_label_mode="location_binary",
        loss_balance_mode="proportional",
    )

    assert parts["strong_scale"] == pytest.approx(1 / 3)
    assert parts["weak_scale"] == pytest.approx(2 / 3)
