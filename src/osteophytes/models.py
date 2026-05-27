"""Local ResNet18 model heads for osteophyte experiments.

This module intentionally avoids importing torchvision at runtime because the
DelftBlue environment used for this project has had torchvision compatibility
issues. The backbone matches ResNet18 naming closely enough to load official
ResNet18 checkpoints.
"""

from __future__ import annotations

from pathlib import Path, PosixPath, WindowsPath
from typing import Any

import torch
from torch import nn
import torch.nn.functional as F

from osteophytes.labels import NUM_LOCATIONS, NUM_THRESHOLDS


def conv3x3(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=1,
        bias=False,
    )


def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: nn.Module | None = None,
    ) -> None:
        super().__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        return out


class ResNet18Backbone(nn.Module):
    feature_dim = 512

    def __init__(self, in_channels: int = 1) -> None:
        super().__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv2d(
            in_channels,
            self.inplanes,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(64, 2)
        self.layer2 = self._make_layer(128, 2, stride=2)
        self.layer3 = self._make_layer(256, 2, stride=2)
        self.layer4 = self._make_layer(512, 2, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self._init_weights()

    def _make_layer(self, planes: int, blocks: int, stride: int = 1) -> nn.Sequential:
        downsample = None
        if stride != 1 or self.inplanes != planes:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes, stride),
                nn.BatchNorm2d(planes),
            )

        layers = [BasicBlock(self.inplanes, planes, stride, downsample)]
        self.inplanes = planes
        for _ in range(1, blocks):
            layers.append(BasicBlock(self.inplanes, planes))
        return nn.Sequential(*layers)

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(module, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        return torch.flatten(x, 1)


class ResNet(nn.Module):
    """Backward-compatible single-head ResNet18 used by earlier scripts."""

    def __init__(self, num_outputs: int = 4, in_channels: int = 1) -> None:
        super().__init__()
        self.backbone = ResNet18Backbone(in_channels=in_channels)
        self.fc = nn.Linear(self.backbone.feature_dim, num_outputs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(self.backbone(x))


class BinaryBaselineResNet18(nn.Module):
    """Binary baseline with one location-specific logit per osteophyte location."""

    def __init__(self, in_channels: int = 1) -> None:
        super().__init__()
        self.backbone = ResNet18Backbone(in_channels=in_channels)
        self.binary_head = nn.Linear(self.backbone.feature_dim, NUM_LOCATIONS)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.binary_head(self.backbone(x))


class OrdinalThresholdResNet18(nn.Module):
    """Threshold-based ordinal regression head with independent thresholds."""

    def __init__(self, in_channels: int = 1) -> None:
        super().__init__()
        self.backbone = ResNet18Backbone(in_channels=in_channels)
        self.ordinal_head = nn.Linear(self.backbone.feature_dim, NUM_LOCATIONS * NUM_THRESHOLDS)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.ordinal_head(self.backbone(x))
        return logits.view(x.shape[0], NUM_LOCATIONS, NUM_THRESHOLDS)


class CoralOrdinalHead(nn.Module):
    """CORAL-style rank-consistent ordinal head for all four locations."""

    def __init__(self, feature_dim: int) -> None:
        super().__init__()
        self.score = nn.Linear(feature_dim, NUM_LOCATIONS)
        self.raw_delta = nn.Parameter(torch.zeros(NUM_LOCATIONS, NUM_THRESHOLDS))

    def thresholds(self) -> torch.Tensor:
        return torch.cumsum(F.softplus(self.raw_delta), dim=1)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        severity_score = self.score(features)
        return severity_score.unsqueeze(2) - self.thresholds().unsqueeze(0)


class CoralOrdinalResNet18(nn.Module):
    """ResNet18 with a CORAL-style rank-consistent ordinal head."""

    def __init__(self, in_channels: int = 1) -> None:
        super().__init__()
        self.backbone = ResNet18Backbone(in_channels=in_channels)
        self.ordinal_head = CoralOrdinalHead(self.backbone.feature_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.ordinal_head(self.backbone(x))


class DualHeadResNet18(nn.Module):
    """Shared backbone with binary and ordinal heads."""

    def __init__(self, in_channels: int = 1, dual_ordinal_head: str = "threshold_independent") -> None:
        super().__init__()
        if dual_ordinal_head not in {"threshold_independent", "coral"}:
            raise ValueError(f"Unsupported dual ordinal head: {dual_ordinal_head}")
        self.dual_ordinal_head = dual_ordinal_head
        self.backbone = ResNet18Backbone(in_channels=in_channels)
        self.binary_head = nn.Linear(self.backbone.feature_dim, NUM_LOCATIONS)
        if dual_ordinal_head == "threshold_independent":
            self.ordinal_head = nn.Linear(self.backbone.feature_dim, NUM_LOCATIONS * NUM_THRESHOLDS)
        else:
            self.ordinal_head = CoralOrdinalHead(self.backbone.feature_dim)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        features = self.backbone(x)
        binary_logits = self.binary_head(features)
        ordinal_logits = self.ordinal_head(features)
        if self.dual_ordinal_head == "threshold_independent":
            ordinal_logits = ordinal_logits.view(x.shape[0], NUM_LOCATIONS, NUM_THRESHOLDS)
        return {"binary_logits": binary_logits, "ordinal_logits": ordinal_logits}


def _torch_load_weights(weights_path: Path) -> dict[str, Any]:
    try:
        return torch.load(weights_path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(weights_path, map_location="cpu")
    except Exception as exc:
        message = str(exc)
        if "Weights only load failed" not in message and "Unsupported global" not in message:
            raise
        try:
            torch.serialization.add_safe_globals([PosixPath, WindowsPath])
            return torch.load(weights_path, map_location="cpu", weights_only=True)
        except Exception:
            print(
                "Falling back to torch.load(weights_only=False) for a trusted "
                f"local project checkpoint: {weights_path}"
            )
            return torch.load(weights_path, map_location="cpu", weights_only=False)


def _checkpoint_state_dict(checkpoint: Any) -> dict[str, Any]:
    if isinstance(checkpoint, dict):
        for key in ("model_state_dict", "state_dict"):
            value = checkpoint.get(key)
            if isinstance(value, dict):
                return value
        return checkpoint
    raise ValueError("Unsupported checkpoint format")


def _adapt_conv1_if_needed(key: str, value: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    if (
        key.endswith("conv1.weight")
        and value.ndim == 4
        and value.shape[1] == 3
        and target.ndim == 4
        and target.shape[1] == 1
    ):
        return value.mean(dim=1, keepdim=True)
    return value


def load_pretrained_resnet18_backbone(backbone: ResNet18Backbone, weights_path: str | Path) -> dict[str, Any]:
    weights_path = Path(weights_path)
    if not weights_path.exists():
        raise FileNotFoundError(f"Pretrained weights not found: {weights_path}")
    checkpoint = _torch_load_weights(weights_path)
    state_dict = _checkpoint_state_dict(checkpoint)
    model_state = backbone.state_dict()
    loadable: dict[str, torch.Tensor] = {}
    skipped: list[str] = []

    for raw_key, raw_value in state_dict.items():
        key = str(raw_key).removeprefix("module.").removeprefix("backbone.")
        if key.startswith("fc.") or "head" in key:
            skipped.append(key)
            continue
        if not isinstance(raw_value, torch.Tensor):
            skipped.append(key)
            continue
        if key in model_state:
            value = _adapt_conv1_if_needed(key, raw_value, model_state[key])
            if model_state[key].shape == value.shape:
                loadable[key] = value
            else:
                skipped.append(key)
        else:
            skipped.append(key)

    missing, unexpected = backbone.load_state_dict(loadable, strict=False)
    info = {
        "loaded": sorted(loadable),
        "skipped": sorted(skipped),
        "missing": sorted(missing),
        "unexpected": sorted(unexpected),
    }
    print(
        "Loaded pretrained ResNet18 backbone "
        f"from {weights_path} ({len(info['loaded'])} loaded, "
        f"{len(info['skipped'])} skipped, {len(info['missing'])} missing)."
    )
    return info


def _model_backbone(model: nn.Module) -> ResNet18Backbone:
    backbone = getattr(model, "backbone", None)
    if not isinstance(backbone, ResNet18Backbone):
        raise ValueError("Model does not expose a ResNet18Backbone at .backbone")
    return backbone


def initialize_backbone_from_binary_checkpoint(
    model: nn.Module,
    checkpoint_path: str | Path,
) -> dict[str, list[str]]:
    """Load matching backbone tensors from a binary baseline checkpoint."""
    checkpoint_path = Path(checkpoint_path)
    checkpoint = _torch_load_weights(checkpoint_path)
    source_state = _checkpoint_state_dict(checkpoint)
    target_state = model.state_dict()
    loadable: dict[str, torch.Tensor] = {}
    skipped: list[str] = []

    for raw_key, raw_value in source_state.items():
        key = str(raw_key).removeprefix("module.")
        if not isinstance(raw_value, torch.Tensor):
            skipped.append(key)
            continue
        if key.startswith(("fc.", "binary_head.", "ordinal_head.")) or "head" in key:
            skipped.append(key)
            continue
        candidates = [key]
        if not key.startswith("backbone."):
            candidates.append(f"backbone.{key}")
        if key.startswith("backbone."):
            candidates.append(key.removeprefix("backbone."))
        matched = False
        for candidate in candidates:
            if candidate in target_state:
                value = _adapt_conv1_if_needed(candidate, raw_value, target_state[candidate])
                if target_state[candidate].shape == value.shape:
                    loadable[candidate] = value
                    matched = True
                    break
        if not matched:
            skipped.append(key)

    missing, unexpected = model.load_state_dict(loadable, strict=False)
    missing_backbone = sorted(key for key in missing if key.startswith("backbone."))
    info = {
        "loaded": sorted(loadable),
        "skipped": sorted(set(skipped)),
        "missing": missing_backbone,
        "unexpected": sorted(unexpected),
    }
    print(f"Initialized backbone from binary checkpoint: {checkpoint_path}")
    print(f"  loaded tensors: {len(info['loaded'])}")
    print(f"  skipped tensors: {len(info['skipped'])}")
    print(f"  missing backbone tensors: {len(info['missing'])}")
    return info


def resnet18(
    num_outputs: int = 4,
    in_channels: int = 1,
    weights_path: str | Path | None = None,
) -> ResNet:
    """Backward-compatible local ResNet18 with a single linear output head."""
    model = ResNet(num_outputs=num_outputs, in_channels=in_channels)
    if weights_path is None:
        print("Using randomly initialized ResNet18 weights.")
    else:
        load_pretrained_resnet18_backbone(model.backbone, weights_path)
    return model


def create_model(
    model_head: str,
    weights_path: str | Path | None = None,
    in_channels: int = 1,
    dual_ordinal_head: str = "threshold_independent",
) -> nn.Module:
    if model_head == "binary":
        model: nn.Module = BinaryBaselineResNet18(in_channels=in_channels)
    elif model_head == "threshold_independent":
        model = OrdinalThresholdResNet18(in_channels=in_channels)
    elif model_head == "coral":
        model = CoralOrdinalResNet18(in_channels=in_channels)
    elif model_head == "dual_head":
        model = DualHeadResNet18(in_channels=in_channels, dual_ordinal_head=dual_ordinal_head)
    else:
        raise ValueError(f"Unsupported model head: {model_head}")

    if weights_path is None:
        print("Using randomly initialized ResNet18 backbone.")
    else:
        load_pretrained_resnet18_backbone(_model_backbone(model), weights_path)
    return model


def backbone_parameters(model: nn.Module):
    return _model_backbone(model).parameters()


def head_parameters(model: nn.Module):
    backbone_param_ids = {id(param) for param in _model_backbone(model).parameters()}
    return [param for param in model.parameters() if id(param) not in backbone_param_ids]


def set_backbone_trainable(model: nn.Module, trainable: bool) -> None:
    for parameter in _model_backbone(model).parameters():
        parameter.requires_grad = trainable
