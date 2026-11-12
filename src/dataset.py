"""Dataset indexing, integrity checks, duplicate detection, stratified group splitting and the
PyTorch Dataset used for training.

Run `python -m src.dataset` to build data/processed/image_index.csv and data/splits/*.csv.
"""
from __future__ import annotations

import argparse
import hashlib
import re
from collections import defaultdict
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image, UnidentifiedImageError
from torch.utils.data import Dataset

from .download import TOMATO_PREFIX, download_dataset, find_class_root, record_source
from .utils import (
    DEFAULT_SEED,
    MODELS_DIR,
    PROCESSED_DIR,
    SPLITS_DIR,
    absolute_from_project,
    ensure_dirs,
    relative_to_project,
    save_json,
)

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
SPLIT_RATIOS = {"train": 0.70, "val": 0.15, "test": 0.15}
DHASH_HAMMING_THRESHOLD = 4  # out of 64 bits; <= this is treated as a near-duplicate


# --------------------------------------------------------------------------- classes
def discover_tomato_classes(class_root: Path, prefix: str = TOMATO_PREFIX) -> list[str]:
    """Class names are derived from the actual folder names, never hardcoded."""
    classes = sorted(d.name for d in class_root.iterdir() if d.is_dir() and d.name.startswith(prefix))
    if not classes:
        raise FileNotFoundError(f"No '{prefix}*' folders in {class_root}")
    return classes


def display_name(folder_name: str) -> str:
    """'Tomato__Tomato_YellowLeaf__Curl_Virus' -> 'Tomato Yellow Leaf Curl Virus'."""
    name = re.sub(r"^Tomato_+", "", folder_name)
    name = name.replace("_", " ")
    name = re.sub(r"(?<=[a-z])(?=[A-Z])", " ", name)
    name = re.sub(r"\s+", " ", name).strip()
    if name.lower() == "healthy":
        return "Healthy"
    return name


def build_class_mapping(classes: list[str]) -> dict:
    return {
        "classes": classes,
        "class_to_idx": {c: i for i, c in enumerate(classes)},
        "idx_to_class": {str(i): c for i, c in enumerate(classes)},
        "display_names": {c: display_name(c) for c in classes},
        "num_classes": len(classes),
    }


# --------------------------------------------------------------------------- per-image analysis
def _dhash(gray: Image.Image, hash_size: int = 8) -> int:
    small = gray.resize((hash_size + 1, hash_size), Image.Resampling.LANCZOS)
    arr = np.asarray(small, dtype=np.int16)
    bits = (arr[:, 1:] > arr[:, :-1]).flatten()
    value = 0
    for b in bits:
        value = (value << 1) | int(b)
    return value


def analyze_image(args: tuple[str, str]) -> dict:
    """Worker: validate one image and compute md5 + dHash. Returns a flat record."""
    rel_path, class_name = args
    path = absolute_from_project(rel_path)
    rec = {
        "path": rel_path,
        "class_name": class_name,
        "ok": False,
        "error": "",
        "width": -1,
        "height": -1,
        "mode": "",
        "format": "",
        "size_bytes": path.stat().st_size if path.exists() else -1,
        "md5": "",
        "dhash": "",
    }
    try:
        data = path.read_bytes()
        rec["md5"] = hashlib.md5(data).hexdigest()
        with Image.open(path) as im:
            im.verify()
        with Image.open(path) as im:
            rec["format"] = im.format or ""
            rec["mode"] = im.mode
            rec["width"], rec["height"] = im.size
            im.load()
            rec["dhash"] = f"{_dhash(im.convert('L')):016x}"
        rec["ok"] = True
    except (UnidentifiedImageError, OSError, SyntaxError, ValueError) as e:
        rec["error"] = f"{type(e).__name__}: {e}"
    return rec


def list_image_files(class_root: Path, classes: list[str]) -> list[tuple[str, str]]:
    items: list[tuple[str, str]] = []
    skipped: list[str] = []
    for c in classes:
        for p in sorted((class_root / c).iterdir()):
            if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS:
                items.append((relative_to_project(p), c))
            elif p.is_file():
                skipped.append(relative_to_project(p))
    if skipped:
        save_json({"non_image_files_skipped": skipped}, PROCESSED_DIR / "skipped_files.json")
    return items


def build_image_index(class_root: Path, classes: list[str], workers: int = 8) -> pd.DataFrame:
    items = list_image_files(class_root, classes)
    with Pool(processes=workers) as pool:
        records = pool.map(analyze_image, items, chunksize=64)
    df = pd.DataFrame.from_records(records)
    df["label"] = df["class_name"].map({c: i for i, c in enumerate(classes)})
    return df


# --------------------------------------------------------------------------- duplicates
class _UnionFind:
    def __init__(self, n: int) -> None:
        self.parent = np.arange(n)

    def find(self, i: int) -> int:
        while self.parent[i] != i:
            self.parent[i] = self.parent[self.parent[i]]
            i = self.parent[i]
        return i

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[max(ra, rb)] = min(ra, rb)


def assign_duplicate_groups(df: pd.DataFrame, hamming_threshold: int = DHASH_HAMMING_THRESHOLD) -> pd.DataFrame:
    """Add `group_id`, `exact_dup` and `near_dup` columns.

    Exact duplicates share an md5. Near-duplicates have dHash Hamming distance <= threshold.
    All members of a group are later forced into the same split to prevent leakage.
    """
    df = df.reset_index(drop=True).copy()
    n = len(df)
    uf = _UnionFind(n)

    md5_groups: dict[str, list[int]] = defaultdict(list)
    for i, h in enumerate(df["md5"].tolist()):
        if h:
            md5_groups[h].append(i)
    exact_dup = np.zeros(n, dtype=bool)
    for idxs in md5_groups.values():
        if len(idxs) > 1:
            exact_dup[idxs] = True
            for j in idxs[1:]:
                uf.union(idxs[0], j)

    valid = df["ok"].to_numpy() & (df["dhash"] != "").to_numpy()
    valid_idx = np.flatnonzero(valid)
    hashes = np.array([int(h, 16) for h in df.loc[valid_idx, "dhash"]], dtype=np.uint64)
    near_dup = np.zeros(n, dtype=bool)
    chunk = 512
    for start in range(0, len(hashes), chunk):
        block = hashes[start : start + chunk, None]
        dist = np.bitwise_count(np.bitwise_xor(block, hashes[None, :]))
        rows, cols = np.nonzero(dist <= hamming_threshold)
        for r, c in zip(rows, cols):
            gi, gj = int(valid_idx[start + r]), int(valid_idx[c])
            if gi < gj:
                near_dup[gi] = near_dup[gj] = True
                uf.union(gi, gj)

    roots = np.array([uf.find(i) for i in range(n)])
    _, group_id = np.unique(roots, return_inverse=True)
    df["group_id"] = group_id
    df["exact_dup"] = exact_dup
    df["near_dup"] = near_dup & ~exact_dup
    return df


def duplicate_summary(df: pd.DataFrame) -> dict:
    sizes = df.groupby("group_id").size()
    multi = sizes[sizes > 1]
    cross_class = (
        df[df["group_id"].isin(multi.index)].groupby("group_id")["class_name"].nunique()
    )
    return {
        "images": int(len(df)),
        "unique_groups": int(sizes.size),
        "groups_with_duplicates": int(multi.size),
        "images_in_duplicate_groups": int(multi.sum()),
        "exact_duplicate_images": int(df["exact_dup"].sum()),
        "near_duplicate_images": int(df["near_dup"].sum()),
        "largest_group_size": int(sizes.max()) if sizes.size else 0,
        "groups_spanning_multiple_classes": int((cross_class > 1).sum()) if len(cross_class) else 0,
        "dhash_hamming_threshold": DHASH_HAMMING_THRESHOLD,
    }


# --------------------------------------------------------------------------- splitting
def stratified_group_split(
    df: pd.DataFrame, ratios: dict[str, float] = SPLIT_RATIOS, seed: int = DEFAULT_SEED
) -> pd.DataFrame:
    """Assign every duplicate group to exactly one split while keeping class proportions.

    Groups are processed per class (by the group's majority class), largest first, and each
    group goes to the split that is currently furthest below its target share for that class.
    """
    assert abs(sum(ratios.values()) - 1.0) < 1e-6
    rng = np.random.default_rng(seed)
    df = df.copy()
    df["split"] = ""
    split_names = list(ratios.keys())

    group_class = df.groupby("group_id")["class_name"].agg(lambda s: s.value_counts().idxmax())
    group_size = df.groupby("group_id").size()

    for class_name in sorted(df["class_name"].unique()):
        groups = group_class[group_class == class_name].index.to_numpy().copy()
        rng.shuffle(groups)
        groups = sorted(groups, key=lambda g: -group_size[g])  # stable: big groups first
        total = int(group_size[groups].sum())
        counts = {s: 0 for s in split_names}
        for g in groups:
            deficits = {s: ratios[s] * total - counts[s] for s in split_names}
            target = max(deficits, key=deficits.get)
            df.loc[df["group_id"] == g, "split"] = target
            counts[target] += int(group_size[g])
    return df


def verify_splits(df: pd.DataFrame) -> dict:
    splits = {s: df[df["split"] == s] for s in SPLIT_RATIOS}
    paths = {s: set(d["path"]) for s, d in splits.items()}
    groups = {s: set(d["group_id"]) for s, d in splits.items()}
    md5s = {s: set(d["md5"]) for s, d in splits.items()}
    names = list(splits)
    overlaps = {}
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            overlaps[f"{a}-{b}"] = {
                "shared_paths": len(paths[a] & paths[b]),
                "shared_groups": len(groups[a] & groups[b]),
                "shared_md5": len(md5s[a] & md5s[b]),
            }
    per_class = (
        df.groupby(["class_name", "split"]).size().unstack(fill_value=0).reindex(columns=names)
    )
    per_class_frac = per_class.div(per_class.sum(axis=1), axis=0).round(4)
    return {
        "counts": {s: int(len(d)) for s, d in splits.items()},
        "fractions": {s: round(len(d) / len(df), 4) for s, d in splits.items()},
        "overlaps": overlaps,
        "leakage_free": all(v["shared_paths"] == 0 and v["shared_groups"] == 0 and v["shared_md5"] == 0 for v in overlaps.values()),
        "per_class_counts": per_class.to_dict(orient="index"),
        "per_class_fractions": per_class_frac.to_dict(orient="index"),
    }


# --------------------------------------------------------------------------- torch dataset
class TomatoLeafDataset(Dataset):
    """Reads (path, label) rows from a split CSV and applies a transform."""

    def __init__(self, split_csv: str | Path, transform=None) -> None:
        self.df = pd.read_csv(split_csv)
        self.paths = [absolute_from_project(p) for p in self.df["path"].tolist()]
        self.labels = self.df["label"].astype(int).tolist()
        self.transform = transform

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        with Image.open(self.paths[idx]) as im:
            image = im.convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, self.labels[idx]


def class_counts_from_csv(split_csv: str | Path, num_classes: int) -> torch.Tensor:
    labels = pd.read_csv(split_csv)["label"].astype(int)
    return torch.bincount(torch.tensor(labels.to_numpy()), minlength=num_classes)


# --------------------------------------------------------------------------- CLI
def prepare_dataset(seed: int = DEFAULT_SEED, workers: int = 8) -> dict:
    ensure_dirs()
    download_path = download_dataset()
    class_root = find_class_root(download_path)
    record_source(download_path, class_root)
    classes = discover_tomato_classes(class_root)
    mapping = build_class_mapping(classes)
    save_json(mapping, MODELS_DIR / "class_mapping.json")

    print(f"Indexing {len(classes)} tomato classes from {class_root} ...")
    df = build_image_index(class_root, classes, workers=workers)
    n_bad = int((~df["ok"]).sum())
    print(f"Indexed {len(df)} files, corrupt/unreadable: {n_bad}")
    df.to_csv(PROCESSED_DIR / "image_index.csv", index=False)

    clean = df[df["ok"]].reset_index(drop=True)
    clean = assign_duplicate_groups(clean)
    dup = duplicate_summary(clean)
    print("Duplicate summary:", dup)

    clean = stratified_group_split(clean, seed=seed)
    report = verify_splits(clean)
    report["seed"] = seed
    report["ratios"] = SPLIT_RATIOS
    report["duplicates"] = dup
    report["corrupt_images_removed"] = n_bad
    print("Split counts:", report["counts"], "leakage_free:", report["leakage_free"])
    if not report["leakage_free"]:
        raise RuntimeError("Split verification failed: overlap between subsets detected")

    clean.to_csv(PROCESSED_DIR / "image_index_clean.csv", index=False)
    cols = ["path", "class_name", "label", "group_id", "md5", "width", "height"]
    for s in SPLIT_RATIOS:
        clean.loc[clean["split"] == s, cols].reset_index(drop=True).to_csv(SPLITS_DIR / f"{s}.csv", index=False)
    save_json(report, SPLITS_DIR / "split_summary.json")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Download, index, deduplicate and split the tomato dataset")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    prepare_dataset(seed=args.seed, workers=args.workers)


if __name__ == "__main__":
    main()
