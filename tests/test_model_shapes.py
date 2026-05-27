from __future__ import annotations

import torch

from src.model import OsteophyteOrdinalNet


def test_model_shapes() -> None:
    model = OsteophyteOrdinalNet()
    x = torch.randn(2, 1, 224, 224)
    logits, attention = model(x, return_attention=True)
    assert logits.shape == (2, 4, 3)
    assert attention.shape[0] == 2
    assert attention.shape[1] == 4
    assert attention.shape[2] in {14, 28}
    pred = model.predict_from_logits(logits)
    assert pred["class_probs"].shape == (2, 4, 4)
    sums = pred["class_probs"].sum(dim=-1)
    assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)


if __name__ == "__main__":
    test_model_shapes()
