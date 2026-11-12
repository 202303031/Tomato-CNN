"""Shared helpers: project paths, seeding, device selection and JSON I/O."""
from __future__ import annotations

import json
import os
import platform
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
SPLITS_DIR = DATA_DIR / "splits"
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"
PLOTS_DIR = RESULTS_DIR / "plots"

DEFAULT_SEED = 42


def ensure_dirs() -> None:
    for d in (RAW_DIR, PROCESSED_DIR, SPLITS_DIR, MODELS_DIR, RESULTS_DIR, PLOTS_DIR):
        d.mkdir(parents=True, exist_ok=True)


def set_seed(seed: int = DEFAULT_SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def seed_worker(worker_id: int) -> None:
    """DataLoader worker init so augmentation randomness is reproducible."""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def get_device(prefer: str | None = None) -> torch.device:
    """Pick Apple MPS when available, otherwise CUDA, otherwise CPU."""
    if prefer:
        return torch.device(prefer)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def environment_report(device: torch.device | None = None) -> dict[str, Any]:
    import torchvision

    device = device or get_device()
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "machine": platform.machine(),
        "torch": str(torch.__version__),
        "torchvision": str(torchvision.__version__),
        "mps_built": torch.backends.mps.is_built(),
        "mps_available": torch.backends.mps.is_available(),
        "cuda_available": torch.cuda.is_available(),
        "selected_device": str(device),
        "cpu_count": os.cpu_count(),
    }


def save_json(obj: Any, path: Path | str, indent: int = 2) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=indent, default=_json_default)


def load_json(path: Path | str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _json_default(o: Any) -> Any:
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    raise TypeError(f"Object of type {type(o).__name__} is not JSON serializable")


def relative_to_project(path: Path | str) -> str:
    """Store paths relative to the project root so the CSVs are portable."""
    p = Path(path).resolve()
    try:
        return p.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return p.as_posix()


def absolute_from_project(rel: str) -> Path:
    p = Path(rel)
    return p if p.is_absolute() else PROJECT_ROOT / p


def count_parameters(model: torch.nn.Module) -> dict[str, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total": total, "trainable": trainable, "frozen": total - trainable}
