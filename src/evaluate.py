"""Evaluate the best checkpoint on the untouched test split and write all metrics/plots.

Run `python -m src.evaluate`.
"""
from __future__ import annotations

import argparse
import textwrap

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from PIL import Image
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
from torch.utils.data import DataLoader

from .dataset import TomatoLeafDataset
from .model import load_checkpoint
from .preprocessing import get_eval_transforms
from .utils import MODELS_DIR, PLOTS_DIR, RESULTS_DIR, SPLITS_DIR, absolute_from_project, get_device, save_json

plt.rcParams.update({"figure.dpi": 110, "savefig.bbox": "tight"})


@torch.no_grad()
def predict_split(model, split_csv, device, batch_size=64, num_workers=4) -> pd.DataFrame:
    ds = TomatoLeafDataset(split_csv, get_eval_transforms(model.config.img_size))
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    probs_all = []
    for x, _ in loader:
        probs_all.append(torch.softmax(model(x.to(device)), dim=1).cpu())
    probs = torch.cat(probs_all).numpy()
    out = ds.df[["path", "class_name", "label"]].copy()
    out["pred"] = probs.argmax(1)
    out["confidence"] = probs.max(1)
    out["correct"] = out["pred"] == out["label"]
    for i in range(probs.shape[1]):
        out[f"p{i}"] = probs[:, i]
    return out


def compute_metrics(df: pd.DataFrame, classes: list[str]) -> dict:
    y, p = df["label"].to_numpy(), df["pred"].to_numpy()
    labels = list(range(len(classes)))
    prec, rec, f1, support = precision_recall_fscore_support(y, p, labels=labels, zero_division=0)
    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(y, p, labels=labels, average="macro", zero_division=0)
    w_p, w_r, w_f1, _ = precision_recall_fscore_support(y, p, labels=labels, average="weighted", zero_division=0)
    cm = confusion_matrix(y, p, labels=labels)
    off = cm.copy(); np.fill_diagonal(off, 0)
    pairs = sorted(((int(off[i, j]), classes[i], classes[j]) for i in labels for j in labels if off[i, j] > 0), reverse=True)
    return {
        "test_images": int(len(df)),
        "accuracy": float(accuracy_score(y, p)),
        "macro_precision": float(macro_p), "macro_recall": float(macro_r), "macro_f1": float(macro_f1),
        "weighted_precision": float(w_p), "weighted_recall": float(w_r), "weighted_f1": float(w_f1),
        "per_class": {c: {"precision": float(prec[i]), "recall": float(rec[i]), "f1": float(f1[i]), "support": int(support[i])}
                      for i, c in enumerate(classes)},
        "confusion_matrix": cm.tolist(),
        "top_confusions": [{"count": n, "true": t, "pred": q} for n, t, q in pairs[:10]],
        "mean_confidence_correct": float(df.loc[df["correct"], "confidence"].mean()),
        "mean_confidence_incorrect": float(df.loc[~df["correct"], "confidence"].mean()) if (~df["correct"]).any() else None,
        "n_incorrect": int((~df["correct"]).sum()),
    }


def _wrap(name: str, width: int = 16) -> str:
    return textwrap.fill(name, width)


def plot_confusion_matrix(cm: np.ndarray, names: list[str], path) -> None:
    cmn = cm / cm.sum(axis=1, keepdims=True).clip(min=1)
    names = [_wrap(n) for n in names]
    fig, axes = plt.subplots(1, 2, figsize=(20, 8.5))
    fig.subplots_adjust(wspace=0.45)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Greens", xticklabels=names, yticklabels=names, ax=axes[0], cbar=False)
    axes[0].set_title("Confusion matrix (counts)"); axes[0].set_xlabel("Predicted"); axes[0].set_ylabel("True")
    sns.heatmap(cmn, annot=True, fmt=".2f", cmap="Blues", xticklabels=names, yticklabels=names, ax=axes[1], vmin=0, vmax=1)
    axes[1].set_title("Confusion matrix (row-normalised = recall)"); axes[1].set_xlabel("Predicted"); axes[1].set_ylabel("")
    for ax in axes:
        ax.tick_params(axis="x", rotation=45, labelsize=8)
        ax.tick_params(axis="y", labelsize=8)
        for lbl in ax.get_xticklabels():
            lbl.set_ha("right")
    fig.savefig(path)
    plt.close(fig)


def plot_per_class(metrics: dict, names: dict[str, str], path) -> None:
    classes = list(metrics["per_class"])
    x = np.arange(len(classes)); w = 0.27
    fig, ax = plt.subplots(figsize=(12, 5))
    for k, (key, color) in enumerate((("precision", "#4C8C4A"), ("recall", "#E0A526"), ("f1", "#5B7DB1"))):
        vals = [metrics["per_class"][c][key] for c in classes]
        ax.bar(x + (k - 1) * w, vals, w, label=key, color=color)
    ax.set_xticks(x); ax.set_xticklabels([names[c] for c in classes], rotation=40, ha="right")
    ax.set_ylim(0, 1.02); ax.set_ylabel("score"); ax.legend(loc="lower right")
    ax.set_title(f"Per-class test metrics (accuracy {metrics['accuracy']:.4f}, macro-F1 {metrics['macro_f1']:.4f})")
    ax.grid(axis="y", alpha=0.3)
    fig.savefig(path)
    plt.close(fig)


def plot_examples(df: pd.DataFrame, classes: list[str], names: dict[str, str], correct: bool, path, n: int = 12, seed: int = 0) -> int:
    sub = df[df["correct"] == correct]
    if sub.empty:
        return 0
    if correct:  # spread correct examples across classes
        sub = pd.concat([g.sample(min(2, len(g)), random_state=seed) for _, g in sub.groupby("class_name")])
    sub = sub.sample(min(n, len(sub)), random_state=seed) if correct else sub.sort_values("confidence", ascending=False).head(n)
    cols = 4; rows = int(np.ceil(len(sub) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.3, rows * 3.6))
    axes = np.atleast_1d(axes).flatten()
    for ax in axes:
        ax.axis("off")
    for ax, (_, r) in zip(axes, sub.iterrows()):
        with Image.open(absolute_from_project(r["path"])) as im:
            ax.imshow(im.convert("RGB"))
        ax.set_title(f"true: {_wrap(names[r['class_name']], 26)}\npred: {_wrap(names[classes[int(r['pred'])]], 26)} "
                     f"({r['confidence']:.1%})", fontsize=8, color="#2d6a2d" if correct else "#a03030")
    fig.suptitle(("Correctly" if correct else "Incorrectly") + " classified test images" +
                 ("" if correct else " (highest-confidence errors first)"), y=1.0)
    fig.savefig(path)
    plt.close(fig)
    return len(sub)


def evaluate(num_workers: int = 4, batch_size: int = 64) -> dict:
    device = get_device()
    model, mapping, meta = load_checkpoint(MODELS_DIR / "best_model.pth", device)
    classes, names = mapping["classes"], mapping["display_names"]
    print(f"Loaded checkpoint from epoch {meta['epoch']} (val macro-F1 {meta['val_metrics']['macro_f1']:.4f}) on {device}")

    # Checkpoint-loading check: recompute validation metrics and compare with what training recorded.
    val_df = predict_split(model, SPLITS_DIR / "val.csv", device, batch_size, num_workers)
    val_metrics = compute_metrics(val_df, classes)
    val_check = {"recorded_val_acc": meta["val_metrics"]["acc"], "recomputed_val_acc": val_metrics["accuracy"],
                 "recorded_val_macro_f1": meta["val_metrics"]["macro_f1"], "recomputed_val_macro_f1": val_metrics["macro_f1"]}
    print("Checkpoint check (val):", val_check)

    test_df = predict_split(model, SPLITS_DIR / "test.csv", device, batch_size, num_workers)
    metrics = compute_metrics(test_df, classes)
    metrics["checkpoint_epoch"] = meta["epoch"]
    metrics["validation_check"] = val_check
    metrics["device"] = str(device)
    test_df.to_csv(RESULTS_DIR / "test_predictions.csv", index=False)
    save_json(metrics, RESULTS_DIR / "test_metrics.json")

    report = classification_report(test_df["label"], test_df["pred"], labels=list(range(len(classes))),
                                   target_names=[names[c] for c in classes], digits=4, zero_division=0)
    header = (f"Test set evaluation of models/best_model.pth (epoch {meta['epoch']}), device={device}\n"
              f"images={len(test_df)}  accuracy={metrics['accuracy']:.4f}  macro-P={metrics['macro_precision']:.4f}  "
              f"macro-R={metrics['macro_recall']:.4f}  macro-F1={metrics['macro_f1']:.4f}  weighted-F1={metrics['weighted_f1']:.4f}\n\n")
    (RESULTS_DIR / "classification_report.txt").write_text(header + report)
    print(header + report)

    plot_confusion_matrix(np.array(metrics["confusion_matrix"]), [names[c] for c in classes], RESULTS_DIR / "confusion_matrix.png")
    plot_per_class(metrics, names, PLOTS_DIR / "per_class_metrics.png")
    plot_examples(test_df, classes, names, True, PLOTS_DIR / "correct_predictions.png")
    plot_examples(test_df, classes, names, False, PLOTS_DIR / "incorrect_predictions.png")
    print("Top confusions:", metrics["top_confusions"][:5])
    return metrics


def main() -> None:
    p = argparse.ArgumentParser(description="Evaluate best checkpoint on the test split")
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--batch-size", type=int, default=64)
    a = p.parse_args()
    evaluate(a.num_workers, a.batch_size)


if __name__ == "__main__":
    matplotlib.use("Agg")
    main()
