"""Small no-torchvision augmentation pipeline for 1-channel crops."""

from __future__ import annotations

import math
from typing import Callable

import torch
import torch.nn.functional as F


class IdentityTransform:
    """Return the input unchanged."""

    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        return image


class TrainTransform:
    """Affine and intensity augmentations for [1, H, W] tensors."""

    def __init__(
        self,
        rotation_degrees: float = 7.0,
        translation_fraction: float = 0.05,
        scale_range: tuple[float, float] = (0.90, 1.10),
        brightness: float = 0.08,
        contrast: float = 0.10,
        noise_std: float = 0.015,
    ) -> None:
        self.rotation_degrees = rotation_degrees
        self.translation_fraction = translation_fraction
        self.scale_range = scale_range
        self.brightness = brightness
        self.contrast = contrast
        self.noise_std = noise_std

    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        if image.ndim != 3 or image.shape[0] != 1:
            raise ValueError(f"Expected image shape [1,H,W], got {tuple(image.shape)}")
        c, h, w = image.shape
        x = image.unsqueeze(0)

        angle = (torch.rand(()) * 2.0 - 1.0).item() * self.rotation_degrees
        angle_rad = math.radians(angle)
        scale = (self.scale_range[0] + torch.rand(()).item() * (self.scale_range[1] - self.scale_range[0]))
        tx = (torch.rand(()).item() * 2.0 - 1.0) * self.translation_fraction * 2.0
        ty = (torch.rand(()).item() * 2.0 - 1.0) * self.translation_fraction * 2.0

        cos_a = math.cos(angle_rad) / scale
        sin_a = math.sin(angle_rad) / scale
        theta = torch.tensor(
            [[[cos_a, -sin_a, tx], [sin_a, cos_a, ty]]],
            dtype=x.dtype,
            device=x.device,
        )
        grid = F.affine_grid(theta, size=(1, c, h, w), align_corners=False)
        x = F.grid_sample(x, grid, mode="bilinear", padding_mode="border", align_corners=False)

        if self.contrast > 0:
            factor = 1.0 + (torch.rand((), device=x.device) * 2.0 - 1.0) * self.contrast
            mean = x.mean(dim=(-1, -2), keepdim=True)
            x = (x - mean) * factor + mean
        if self.brightness > 0:
            offset = (torch.rand((), device=x.device) * 2.0 - 1.0) * self.brightness
            x = x + offset
        if self.noise_std > 0:
            x = x + torch.randn_like(x) * self.noise_std
        return x.squeeze(0).to(dtype=image.dtype)


def make_train_transform(config: dict) -> Callable[[torch.Tensor], torch.Tensor]:
    """Build the training transform; disabled when config['augment'] is false."""
    if not config.get("augment", True):
        return IdentityTransform()
    return TrainTransform(
        rotation_degrees=float(config.get("rotation_degrees", 7.0)),
        translation_fraction=float(config.get("translation_fraction", 0.05)),
        scale_range=tuple(config.get("scale_range", [0.90, 1.10])),
        brightness=float(config.get("brightness_jitter", 0.08)),
        contrast=float(config.get("contrast_jitter", 0.10)),
        noise_std=float(config.get("gaussian_noise_std", 0.015)),
    )


def make_eval_transform(config: dict) -> Callable[[torch.Tensor], torch.Tensor]:
    """Evaluation uses deterministic preprocessing only."""
    return IdentityTransform()
