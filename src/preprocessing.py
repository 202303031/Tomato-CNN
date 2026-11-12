"""Image preprocessing: training augmentation, deterministic eval transforms, inference helpers."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

# ImageNet statistics, required because the ResNet18 backbone was pretrained on ImageNet.
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
IMG_SIZE = 224


def get_train_transforms(img_size: int = IMG_SIZE) -> transforms.Compose:
    """Augmentation applied to training images only."""
    return transforms.Compose(
        [
            transforms.RandomResizedCrop(img_size, scale=(0.75, 1.0), ratio=(0.9, 1.1)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.02),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


def get_eval_transforms(img_size: int = IMG_SIZE) -> transforms.Compose:
    """Deterministic transform for validation, test and inference."""
    return transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


def load_image(path: str | Path) -> Image.Image:
    """Open an image file and convert it to 3-channel RGB."""
    with Image.open(path) as im:
        return im.convert("RGB")


def preprocess_for_inference(image: Image.Image, img_size: int = IMG_SIZE) -> torch.Tensor:
    """PIL image -> normalized tensor of shape [1, 3, img_size, img_size]."""
    return get_eval_transforms(img_size)(image.convert("RGB")).unsqueeze(0)


def denormalize(tensor: torch.Tensor) -> np.ndarray:
    """Normalized CHW tensor -> HWC float image in [0, 1] for plotting."""
    t = tensor.detach().cpu().clone()
    if t.dim() == 4:
        t = t[0]
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    t = (t * std + mean).clamp(0, 1)
    return t.permute(1, 2, 0).numpy()
