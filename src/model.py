"""CNN feature extractor (ResNet18) + MLP classification head.

Tensor flow for a 224x224 RGB input (batch size B):
    input            [B, 3, 224, 224]
    conv1+bn+relu    [B, 64, 112, 112]   7x7 stride-2 convolution
    maxpool          [B, 64, 56, 56]
    layer1           [B, 64, 56, 56]     2 BasicBlocks
    layer2           [B, 128, 28, 28]    2 BasicBlocks, stride 2
    layer3           [B, 256, 14, 14]    2 BasicBlocks, stride 2
    layer4           [B, 512, 7, 7]      2 BasicBlocks, stride 2  <- Grad-CAM target
    global avg pool  [B, 512, 1, 1] -> flatten -> [B, 512]  (feature vector)
    MLP head         Linear(512->256) -> ReLU -> Dropout -> Linear(256->num_classes)
    output logits    [B, num_classes]; softmax applied only at inference time.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path

import torch
import torch.nn as nn
from torchvision import models

from .utils import count_parameters

BACKBONE_STAGES = ("conv1", "bn1", "layer1", "layer2", "layer3", "layer4")


@dataclass
class ModelConfig:
    backbone: str = "resnet18"
    pretrained: bool = True
    num_classes: int | None = None
    img_size: int = 224
    hidden_dim: int = 256
    dropout: float = 0.3
    trainable_stages: list[str] = field(default_factory=lambda: ["layer3", "layer4"])

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "ModelConfig":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


class MLPHead(nn.Module):
    """Fully connected -> ReLU -> Dropout -> fully connected (one logit per class)."""

    def __init__(self, in_features: int, hidden_dim: int, num_classes: int, dropout: float) -> None:
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_dim)
        self.act = nn.ReLU(inplace=True)
        self.drop = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.drop(self.act(self.fc1(x))))


class TomatoLeafCNN(nn.Module):
    """ImageNet-pretrained ResNet18 convolutional trunk + global average pooling + MLP head."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        if config.backbone != "resnet18":
            raise ValueError("Only resnet18 is supported in this project")
        if not config.num_classes:
            raise ValueError("ModelConfig.num_classes must be set from the class mapping")
        weights = models.ResNet18_Weights.IMAGENET1K_V1 if config.pretrained else None
        backbone = models.resnet18(weights=weights)
        self.stem = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool)
        self.layer1, self.layer2, self.layer3, self.layer4 = (
            backbone.layer1,
            backbone.layer2,
            backbone.layer3,
            backbone.layer4,
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.feature_dim = backbone.fc.in_features  # 512 for ResNet18
        self.head = MLPHead(self.feature_dim, config.hidden_dim, config.num_classes, config.dropout)
        self._frozen_modules: list[nn.Module] = []
        self.set_trainable_stages(config.trainable_stages)

    # ------------------------------------------------------------------ freezing
    def _stage_modules(self) -> dict[str, nn.Module]:
        return {
            "conv1": self.stem[0],
            "bn1": self.stem[1],
            "layer1": self.layer1,
            "layer2": self.layer2,
            "layer3": self.layer3,
            "layer4": self.layer4,
        }

    def set_trainable_stages(self, trainable: list[str]) -> None:
        """Freeze every backbone stage not listed; the MLP head is always trainable."""
        unknown = set(trainable) - set(BACKBONE_STAGES)
        if unknown:
            raise ValueError(f"Unknown stages {unknown}; choose from {BACKBONE_STAGES}")
        self._frozen_modules = []
        for name, module in self._stage_modules().items():
            is_trainable = name in trainable
            for p in module.parameters():
                p.requires_grad = is_trainable
            if not is_trainable:
                self._frozen_modules.append(module)
        for p in self.head.parameters():
            p.requires_grad = True

    def train(self, mode: bool = True) -> "TomatoLeafCNN":
        """Keep frozen stages (and their BatchNorm running statistics) in eval mode."""
        super().train(mode)
        for m in self._frozen_modules:
            m.eval()
        return self

    # ------------------------------------------------------------------ forward
    @property
    def gradcam_target_layer(self) -> nn.Module:
        return self.layer4

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """Image batch -> final convolutional feature map [B, 512, 7, 7]."""
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return self.layer4(x)

    def pool_features(self, fmap: torch.Tensor) -> torch.Tensor:
        """Feature map [B, 512, 7, 7] -> feature vector [B, 512]."""
        return torch.flatten(self.pool(fmap), 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.pool_features(self.forward_features(x)))


def build_model(config: ModelConfig | dict) -> TomatoLeafCNN:
    if isinstance(config, dict):
        config = ModelConfig.from_dict(config)
    return TomatoLeafCNN(config)


def parameter_breakdown(model: TomatoLeafCNN) -> dict[str, dict[str, int]]:
    parts = {"stem": model.stem, "layer1": model.layer1, "layer2": model.layer2,
             "layer3": model.layer3, "layer4": model.layer4, "head": model.head}
    out = {name: count_parameters(m) for name, m in parts.items()}
    out["total"] = count_parameters(model)
    return out


@torch.no_grad()
def trace_shapes(model: TomatoLeafCNN, img_size: int | None = None) -> list[tuple[str, tuple[int, ...]]]:
    """Run one dummy image through the network and record the tensor shape after each stage."""
    img_size = img_size or model.config.img_size
    was_training = model.training
    model.eval()
    device = next(model.parameters()).device
    x = torch.zeros(1, 3, img_size, img_size, device=device)
    shapes = [("input", tuple(x.shape))]
    x = model.stem(x); shapes.append(("stem (conv1+bn+relu+maxpool)", tuple(x.shape)))
    x = model.layer1(x); shapes.append(("layer1", tuple(x.shape)))
    x = model.layer2(x); shapes.append(("layer2", tuple(x.shape)))
    x = model.layer3(x); shapes.append(("layer3", tuple(x.shape)))
    x = model.layer4(x); shapes.append(("layer4 (Grad-CAM target)", tuple(x.shape)))
    x = model.pool_features(x); shapes.append(("global avg pool + flatten", tuple(x.shape)))
    x = model.head.act(model.head.fc1(x)); shapes.append(("head.fc1 + ReLU (+Dropout)", tuple(x.shape)))
    x = model.head.fc2(x); shapes.append(("head.fc2 logits", tuple(x.shape)))
    model.train(was_training)
    return shapes


def model_summary(model: TomatoLeafCNN) -> str:
    lines = [f"Backbone: {model.config.backbone} (ImageNet pretrained={model.config.pretrained})",
             f"Trainable backbone stages: {model.config.trainable_stages}; frozen: "
             f"{[s for s in BACKBONE_STAGES if s not in model.config.trainable_stages]}",
             f"MLP head: Linear({model.feature_dim}->{model.config.hidden_dim}) -> ReLU -> "
             f"Dropout({model.config.dropout}) -> Linear({model.config.hidden_dim}->{model.config.num_classes})",
             "", "Tensor shapes (batch of 1):"]
    for name, shape in trace_shapes(model):
        lines.append(f"  {name:32s} {list(shape)}")
    lines.append("")
    lines.append("Parameters:")
    for name, c in parameter_breakdown(model).items():
        lines.append(f"  {name:8s} total={c['total']:>11,}  trainable={c['trainable']:>11,}  frozen={c['frozen']:>11,}")
    return "\n".join(lines)


def load_checkpoint(path: str | Path, device: str | torch.device = "cpu") -> tuple[TomatoLeafCNN, dict, dict]:
    """Load models/best_model.pth -> (model in eval mode on `device`, class_mapping, checkpoint metadata)."""
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    config = ModelConfig.from_dict(ckpt["config"])
    config.pretrained = False  # weights come from the checkpoint, not from torchvision downloads
    model = TomatoLeafCNN(config)
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model.to(device).eval()
    if ckpt.get("hyperparameters", {}).get("smoke"):
        import warnings
        warnings.warn(f"{path} was produced by a --smoke run and is not a trained model", stacklevel=2)
    meta = {k: v for k, v in ckpt.items() if k not in ("model_state_dict", "class_mapping")}
    return model, ckpt["class_mapping"], meta
