from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from src.losses import compute_visible_pos_weights, partial_ordinal_targets_and_mask
from src.train_ordinal import prepare_visible_training_arrays


LOCATIONS = ["sup_acet", "inf_acet", "sup_fem", "inf_fem"]


def _train_rows() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "h5_index": 1,
                "split": "train",
                **{f"binary_{loc}": 1 for loc in LOCATIONS},
                **{f"grade_{loc}": 2 for loc in LOCATIONS},
            },
            {
                "h5_index": 2,
                "split": "train",
                **{f"binary_{loc}": 1 for loc in LOCATIONS},
                **{f"grade_{loc}": 3 for loc in LOCATIONS},
            },
            {
                "h5_index": 3,
                "split": "train",
                **{f"binary_{loc}": 0 for loc in LOCATIONS},
                **{f"grade_{loc}": 0 for loc in LOCATIONS},
            },
        ]
    )


def test_mixed_mode_hides_non_budget_grades() -> None:
    rows, binary, grades = prepare_visible_training_arrays(_train_rows(), LOCATIONS, {1}, "mixed")
    assert rows["h5_index"].tolist() == [1, 2, 3]
    assert (grades[0] == 2).all()
    assert (grades[1:] == -1).all()
    assert binary.shape == (3, 4)


def test_mixed_masks_budget_and_non_budget_samples() -> None:
    binary = torch.tensor([[1], [1], [0], [-1]])
    grades = torch.tensor([[2], [-1], [-1], [-1]])
    targets, mask, _ = partial_ordinal_targets_and_mask(binary, grades)
    assert torch.equal(targets[0, 0], torch.tensor([1.0, 1.0, 0.0]))
    assert torch.equal(mask[0, 0], torch.tensor([1.0, 1.0, 1.0]))
    assert torch.equal(targets[1, 0], torch.tensor([1.0, 0.0, 0.0]))
    assert torch.equal(mask[1, 0], torch.tensor([1.0, 0.0, 0.0]))
    assert torch.equal(targets[2, 0], torch.tensor([0.0, 0.0, 0.0]))
    assert torch.equal(mask[2, 0], torch.tensor([1.0, 1.0, 1.0]))
    assert mask[3, 0].sum().item() == 0


def test_pos_weights_ignore_hidden_grade_changes() -> None:
    rows = _train_rows()
    _, binary, grades = prepare_visible_training_arrays(rows, LOCATIONS, {1}, "mixed")
    weights_a, _ = compute_visible_pos_weights(binary, grades)
    rows_changed = rows.copy()
    for loc in LOCATIONS:
        rows_changed.loc[rows_changed["h5_index"] != 1, f"grade_{loc}"] = [0, 1]
    _, binary_b, grades_b = prepare_visible_training_arrays(rows_changed, LOCATIONS, {1}, "mixed")
    weights_b, _ = compute_visible_pos_weights(binary_b, grades_b)
    assert np.allclose(weights_a.numpy(), weights_b.numpy())


if __name__ == "__main__":
    test_mixed_mode_hides_non_budget_grades()
    test_mixed_masks_budget_and_non_budget_samples()
    test_pos_weights_ignore_hidden_grade_changes()
