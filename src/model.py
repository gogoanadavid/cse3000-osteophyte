"""Custom CNN with location-specific attention and ordinal cutpoints."""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


def _group_norm(channels: int) -> nn.GroupNorm:
    groups = 8
    while channels % groups != 0 and groups > 1:
        groups //= 2
    return nn.GroupNorm(groups, channels)


class BasicBlock(nn.Module):
    """Small residual block using GroupNorm."""

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False)
        self.norm1 = _group_norm(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False)
        self.norm2 = _group_norm(out_channels)
        self.act = nn.SiLU(inplace=True)
        self.downsample = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False),
                _group_norm(out_channels),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.act(self.norm1(self.conv1(x)))
        out = self.norm2(self.conv2(out))
        if self.downsample is not None:
            identity = self.downsample(identity)
        return self.act(out + identity)


class OsteophyteOrdinalNet(nn.Module):
    """Predict cumulative OARSI thresholds for four hip locations."""

    def __init__(self, num_locations: int = 4, channels: int = 256) -> None:
        super().__init__()
        self.num_locations = num_locations
        self.backbone_channels = channels
        self.stem = nn.Sequential(
            nn.Conv2d(1, 32, 5, stride=2, padding=2, bias=False),
            _group_norm(32),
            nn.SiLU(inplace=True),
        )
        self.layer1 = nn.Sequential(BasicBlock(32, 64), BasicBlock(64, 64))
        self.layer2 = nn.Sequential(BasicBlock(64, 128, stride=2), BasicBlock(128, 128))
        self.layer3 = nn.Sequential(BasicBlock(128, 192, stride=2), BasicBlock(192, 192))
        self.layer4 = nn.Sequential(BasicBlock(192, channels, stride=2), BasicBlock(channels, channels))
        self.attention = nn.Sequential(
            nn.Conv2d(channels + 2, 128, 1),
            nn.SiLU(inplace=True),
            nn.Conv2d(128, num_locations, 1),
        )
        self.location_heads = nn.ModuleList(
            [
                nn.Sequential(
                    nn.LayerNorm(channels),
                    nn.Linear(channels, 128),
                    nn.SiLU(inplace=True),
                    nn.Linear(128, 1),
                )
                for _ in range(num_locations)
            ]
        )
        base = torch.full((num_locations, 1), -0.5)
        deltas = torch.full((num_locations, 2), 0.75)
        self.cutpoint_base = nn.Parameter(base)
        self.cutpoint_delta_raw = nn.Parameter(deltas)

    def backbone(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return self.layer4(x)

    @staticmethod
    def _coord_channels(batch: int, height: int, width: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        yy = torch.linspace(-1.0, 1.0, height, device=device, dtype=dtype).view(1, 1, height, 1)
        xx = torch.linspace(-1.0, 1.0, width, device=device, dtype=dtype).view(1, 1, 1, width)
        return torch.cat([xx.expand(batch, 1, height, width), yy.expand(batch, 1, height, width)], dim=1)

    def ordered_cutpoints(self) -> torch.Tensor:
        deltas = F.softplus(self.cutpoint_delta_raw) + 1e-4
        return torch.cat([self.cutpoint_base, self.cutpoint_base + torch.cumsum(deltas, dim=1)], dim=1)

    def forward(self, x: torch.Tensor, return_attention: bool = False):
        features = self.backbone(x)
        b, c, h, w = features.shape
        coords = self._coord_channels(b, h, w, features.device, features.dtype)
        attention_logits = self.attention(torch.cat([features, coords], dim=1))
        attention = torch.softmax(attention_logits.flatten(2), dim=-1).view(b, self.num_locations, h, w)
        pooled = torch.einsum("bchw,blhw->blc", features, attention)

        scores = []
        for loc in range(self.num_locations):
            scores.append(self.location_heads[loc](pooled[:, loc, :]).squeeze(-1))
        score = torch.stack(scores, dim=1)
        cutpoints = self.ordered_cutpoints()
        logits = score.unsqueeze(-1) - cutpoints.unsqueeze(0)
        if return_attention:
            return logits, attention
        return logits

    @staticmethod
    def predict_from_logits(logits: torch.Tensor) -> dict[str, torch.Tensor]:
        q = torch.sigmoid(logits).clamp(1e-6, 1.0 - 1e-6)
        p0 = 1.0 - q[..., 0]
        p1 = q[..., 0] - q[..., 1]
        p2 = q[..., 1] - q[..., 2]
        p3 = q[..., 2]
        class_probs = torch.stack([p0, p1, p2, p3], dim=-1).clamp_min(0.0)
        class_probs = class_probs / class_probs.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        expected = (class_probs * torch.arange(4, device=logits.device, dtype=logits.dtype)).sum(dim=-1)
        hard = torch.argmax(class_probs, dim=-1)
        return {
            "threshold_probs": q,
            "expected_grade": expected,
            "class_probs": class_probs,
            "hard_grade": hard,
        }

    @torch.no_grad()
    def predict(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        self.eval()
        return self.predict_from_logits(self.forward(x))
