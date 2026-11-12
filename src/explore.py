"""Dataset exploration: class distribution, image dimensions, integrity, samples and split balance.

Run `python -m src.explore` after `python -m src.dataset`.
"""
from __future__ import annotations

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

from .utils import MODELS_DIR, PLOTS_DIR, PROCESSED_DIR, RESULTS_DIR, SPLITS_DIR, absolute_from_project, load_json, save_json

plt.rcParams.update({"figure.dpi": 110, "savefig.bbox": "tight"})


def plot_class_distribution(df: pd.DataFrame, display: dict[str, str]) -> None:
    counts = df["class_name"].value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar([display[c] for c in counts.index], counts.values, color="#4C8C4A")
    ax.bar_label(bars, fontsize=8)
    ax.set_ylabel("Images")
    ax.set_title("Tomato leaf images per class (after removing corrupt files)")
    ax.tick_params(axis="x", rotation=40)
    for lbl in ax.get_xticklabels():
        lbl.set_ha("right")
    fig.savefig(PLOTS_DIR / "class_distribution.png")
    plt.close(fig)


def plot_image_dimensions(df: pd.DataFrame) -> dict:
    dims = df.groupby(["width", "height"]).size().sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(7, 4))
    labels = [f"{w}x{h}" for (w, h) in dims.index[:10]]
    ax.bar(labels, dims.values[:10], color="#5B7DB1")
    ax.set_ylabel("Images")
    ax.set_title("Image resolutions (top 10)")
    ax.tick_params(axis="x", rotation=30)
    fig.savefig(PLOTS_DIR / "image_dimensions.png")
    plt.close(fig)
    return {f"{w}x{h}": int(n) for (w, h), n in dims.items()}


def plot_samples(df: pd.DataFrame, display: dict[str, str], per_class: int = 6, seed: int = 0) -> None:
    classes = sorted(df["class_name"].unique())
    rng = np.random.default_rng(seed)
    fig, axes = plt.subplots(len(classes), per_class, figsize=(per_class * 1.8, len(classes) * 1.9), squeeze=False)
    for r, c in enumerate(classes):
        paths = df.loc[df["class_name"] == c, "path"].to_numpy()
        picks = rng.choice(paths, size=min(per_class, len(paths)), replace=False)
        for k in range(per_class):
            ax = axes[r, k]
            ax.axis("off")
            if k < len(picks):
                with Image.open(absolute_from_project(picks[k])) as im:
                    ax.imshow(im.convert("RGB"))
            if k == 0:
                ax.set_title(display[c], fontsize=8, loc="left")
    fig.suptitle("Sample images per class", y=0.995)
    fig.subplots_adjust(top=0.97, hspace=0.35)
    fig.savefig(PLOTS_DIR / "sample_images.png")
    plt.close(fig)


def plot_split_distribution(clean: pd.DataFrame, display: dict[str, str]) -> None:
    table = clean.groupby(["class_name", "split"]).size().unstack(fill_value=0)[["train", "val", "test"]]
    fig, ax = plt.subplots(figsize=(10, 5))
    bottom = np.zeros(len(table))
    for split, color in zip(table.columns, ["#4C8C4A", "#E0A526", "#B14C4C"]):
        ax.bar([display[c] for c in table.index], table[split].values, bottom=bottom, label=split, color=color)
        bottom += table[split].values
    ax.set_ylabel("Images")
    ax.set_title("Stratified train / val / test split per class")
    ax.legend()
    ax.tick_params(axis="x", rotation=40)
    for lbl in ax.get_xticklabels():
        lbl.set_ha("right")
    fig.savefig(PLOTS_DIR / "split_distribution.png")
    plt.close(fig)


def plot_duplicate_examples(clean: pd.DataFrame, max_groups: int = 4) -> int:
    sizes = clean.groupby("group_id").size()
    groups = sizes[sizes > 1].sort_values(ascending=False).index[:max_groups]
    if len(groups) == 0:
        return 0
    fig, axes = plt.subplots(len(groups), 4, figsize=(8, 2.1 * len(groups)), squeeze=False)
    for r, g in enumerate(groups):
        rows = clean[clean["group_id"] == g].head(4)
        for k in range(4):
            ax = axes[r, k]
            ax.axis("off")
            if k < len(rows):
                row = rows.iloc[k]
                with Image.open(absolute_from_project(row["path"])) as im:
                    ax.imshow(im.convert("RGB"))
                ax.set_title(f"group {g} | {row['split']}\n{row['class_name'][:22]}", fontsize=6)
    fig.suptitle("Largest duplicate / near-duplicate groups (all members share one split)")
    fig.savefig(PLOTS_DIR / "duplicate_groups.png")
    plt.close(fig)
    return len(groups)


def main() -> None:
    index = pd.read_csv(PROCESSED_DIR / "image_index.csv")
    clean = pd.read_csv(PROCESSED_DIR / "image_index_clean.csv")
    mapping = load_json(MODELS_DIR / "class_mapping.json")
    display = mapping["display_names"]
    split_report = load_json(SPLITS_DIR / "split_summary.json")
    source = load_json(PROCESSED_DIR / "dataset_source.json")

    plot_class_distribution(clean, display)
    dims = plot_image_dimensions(clean)
    plot_samples(clean, display)
    plot_split_distribution(clean, display)
    plot_duplicate_examples(clean)

    counts = clean["class_name"].value_counts().sort_index()
    summary = {
        "source": source,
        "num_classes": mapping["num_classes"],
        "classes": mapping["classes"],
        "display_names": display,
        "total_files_indexed": int(len(index)),
        "corrupt_or_unreadable": int((~index["ok"]).sum()),
        "corrupt_files": index.loc[~index["ok"], ["path", "error"]].to_dict(orient="records"),
        "usable_images": int(len(clean)),
        "class_counts": {c: int(n) for c, n in counts.items()},
        "imbalance_ratio_max_over_min": round(float(counts.max() / counts.min()), 2),
        "image_modes": clean["mode"].value_counts().to_dict(),
        "image_formats": clean["format"].value_counts().to_dict(),
        "image_dimensions": dims,
        "file_size_bytes": {
            "min": int(clean["size_bytes"].min()),
            "median": int(clean["size_bytes"].median()),
            "max": int(clean["size_bytes"].max()),
        },
        "duplicates": split_report["duplicates"],
        "splits": {k: split_report[k] for k in ("counts", "fractions", "leakage_free", "seed", "ratios")},
    }
    save_json(summary, RESULTS_DIR / "dataset_summary.json")

    lines = ["| Class | Display name | Images | Share |", "|---|---|---:|---:|"]
    for c, n in counts.items():
        lines.append(f"| {c} | {display[c]} | {n} | {100 * n / len(clean):.1f}% |")
    (RESULTS_DIR / "dataset_summary.md").write_text(
        "# Dataset summary\n\n"
        f"Source: {source['kaggle_url']}\n\n"
        f"Usable images: {len(clean)} across {mapping['num_classes']} classes "
        f"(corrupt removed: {summary['corrupt_or_unreadable']}).\n\n" + "\n".join(lines) + "\n"
    )
    print(f"Usable images: {len(clean)}, classes: {mapping['num_classes']}, imbalance ratio: {summary['imbalance_ratio_max_over_min']}")
    print("Dimensions:", dims)
    print("Plots saved to", PLOTS_DIR)


if __name__ == "__main__":
    matplotlib.use("Agg")
    main()
