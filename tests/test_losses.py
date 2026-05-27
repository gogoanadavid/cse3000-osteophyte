from __future__ import annotations

import torch

from src.losses import masked_partial_ordinal_loss, ordinal_targets_from_grades, partial_ordinal_targets_and_mask


def test_ordinal_targets_from_grades() -> None:
    grades = torch.tensor([0, 1, 2, 3])
    targets = ordinal_targets_from_grades(grades)
    expected = torch.tensor(
        [
            [0, 0, 0],
            [1, 0, 0],
            [1, 1, 0],
            [1, 1, 1],
        ],
        dtype=torch.float32,
    )
    assert torch.equal(targets, expected)


def test_binary_only_positive_masks_only_first_threshold() -> None:
    binary = torch.tensor([[1]])
    grades = torch.tensor([[-1]])
    targets, mask, _ = partial_ordinal_targets_and_mask(binary, grades)
    assert torch.equal(targets[0, 0], torch.tensor([1.0, 0.0, 0.0]))
    assert torch.equal(mask[0, 0], torch.tensor([1.0, 0.0, 0.0]))


def test_binary_only_negative_masks_all_thresholds_negative() -> None:
    binary = torch.tensor([[0]])
    grades = torch.tensor([[-1]])
    targets, mask, _ = partial_ordinal_targets_and_mask(binary, grades)
    assert torch.equal(targets[0, 0], torch.tensor([0.0, 0.0, 0.0]))
    assert torch.equal(mask[0, 0], torch.tensor([1.0, 1.0, 1.0]))


def test_missing_labels_produce_no_loss_contribution() -> None:
    logits = torch.zeros((1, 1, 3), requires_grad=True)
    binary = torch.tensor([[-1]])
    grades = torch.tensor([[-1]])
    targets, mask, _ = partial_ordinal_targets_and_mask(binary, grades)
    assert mask.sum().item() == 0
    loss = masked_partial_ordinal_loss(logits, binary, grades)
    assert loss.item() == 0.0


if __name__ == "__main__":
    test_ordinal_targets_from_grades()
    test_binary_only_positive_masks_only_first_threshold()
    test_binary_only_negative_masks_all_thresholds_negative()
    test_missing_labels_produce_no_loss_contribution()
