"""Download the PlantVillage dataset from Kaggle with kagglehub and locate the class folders."""
from __future__ import annotations

import os
from pathlib import Path

from .utils import PROCESSED_DIR, RAW_DIR, ensure_dirs, load_json, relative_to_project, save_json

KAGGLE_DATASET = "emmarex/plantdisease"
KAGGLE_URL = "https://www.kaggle.com/datasets/emmarex/plantdisease"
TOMATO_PREFIX = "Tomato"


def download_dataset(dataset: str = KAGGLE_DATASET) -> Path:
    """Download (or reuse the cached copy of) the Kaggle dataset into data/raw."""
    import kagglehub

    ensure_dirs()
    cache = RAW_DIR / "kagglehub_cache"
    os.environ["KAGGLEHUB_CACHE"] = str(cache)
    path = Path(kagglehub.dataset_download(dataset))
    return path


def find_class_root(download_path: Path, prefix: str = TOMATO_PREFIX) -> Path:
    """Return the shallowest directory whose children include `prefix*` class folders.

    The Kaggle archive contains `PlantVillage/` and an identical nested copy
    `PlantVillage/PlantVillage/`; picking the shallowest match avoids indexing both.
    """
    candidates = []
    for dirpath, dirnames, _ in os.walk(download_path):
        if any(d.startswith(prefix) for d in dirnames):
            candidates.append(Path(dirpath))
    if not candidates:
        raise FileNotFoundError(f"No '{prefix}*' class folders found under {download_path}")
    return min(candidates, key=lambda p: len(p.parts))


def record_source(download_path: Path, class_root: Path) -> dict:
    info = {
        "kaggle_dataset": KAGGLE_DATASET,
        "kaggle_url": KAGGLE_URL,
        "download_tool": "kagglehub.dataset_download (public dataset, no credentials used)",
        "download_path": relative_to_project(download_path),
        "class_root": relative_to_project(class_root),
        "original_dataset": "PlantVillage (Hughes & Salathe, 2015), arXiv:1511.08060",
        "kaggle_uploader": "Tairu Oluwafemi Emmanuel",
        "kaggle_license_as_listed": "Unknown (as shown on the Kaggle dataset page, checked 2026-11-03)",
        "kaggle_acknowledgement": "Uploader states the dataset was taken from spMohanty's GitHub repo",
        "upstream_repository": "https://github.com/spMohanty/PlantVillage-Dataset",
        "usage_note": "Academic, non-commercial research/education; attribute PlantVillage (Hughes & Salathe) and the Kaggle listing.",
        "notes": [
            "Archive contains PlantVillage/ and an identical nested PlantVillage/PlantVillage/ copy; only the outer copy is indexed.",
            "Only folders starting with 'Tomato' are used; Pepper and Potato folders are excluded.",
        ],
    }
    out = PROCESSED_DIR / "dataset_source.json"
    if out.exists():
        info = {**load_json(out), **info}
    save_json(info, out)
    return info


def main() -> None:
    path = download_dataset()
    root = find_class_root(path)
    info = record_source(path, root)
    print("Download path :", path)
    print("Class root    :", root)
    print("Tomato folders:", sorted(d.name for d in root.iterdir() if d.is_dir() and d.name.startswith(TOMATO_PREFIX)))
    print("Source info saved to", PROCESSED_DIR / "dataset_source.json", "->", info["kaggle_url"])


if __name__ == "__main__":
    main()
