"""Inference on new images: load checkpoint + class mapping, predict, explain with Grad-CAM.

Run `python -m src.predict img1.jpg img2.jpg ...` or `python -m src.predict --from-test 2`.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image

from .disease_info import describe
from .gradcam import explain_image, gradcam_figure
from .model import load_checkpoint
from .preprocessing import load_image, preprocess_for_inference
from .utils import MODELS_DIR, RESULTS_DIR, SPLITS_DIR, absolute_from_project, get_device, relative_to_project, save_json


class Predictor:
    """Loads the trained model once and serves predictions + Grad-CAM explanations."""

    def __init__(self, checkpoint: str | Path = MODELS_DIR / "best_model.pth", device: str | torch.device | None = None) -> None:
        self.device = get_device(device) if isinstance(device, (str, type(None))) else device
        self.model, self.mapping, self.meta = load_checkpoint(checkpoint, self.device)
        self.classes: list[str] = self.mapping["classes"]
        self.display: dict[str, str] = self.mapping["display_names"]

    @torch.no_grad()
    def predict(self, image: Image.Image) -> dict:
        x = preprocess_for_inference(image, self.model.config.img_size).to(self.device)
        probs = torch.softmax(self.model(x), dim=1)[0].cpu().numpy()
        return self._prediction_from_probs(probs)

    def _prediction_from_probs(self, probs: np.ndarray) -> dict:
        idx = int(probs.argmax())
        return {
            "pred_idx": idx,
            "pred_class": self.classes[idx],
            "pred_display": self.display[self.classes[idx]],
            "confidence": float(probs[idx]),
            "probabilities": {self.display[c]: float(probs[i]) for i, c in enumerate(self.classes)},
            "probabilities_by_class": {c: float(probs[i]) for i, c in enumerate(self.classes)},
            "description": describe(self.display[self.classes[idx]]),
        }

    def resolve_class_index(self, name: str) -> int:
        """Accept either the dataset folder name or the human-readable display name."""
        if name in self.classes:
            return self.classes.index(name)
        matches = [i for i, c in enumerate(self.classes) if self.display[c] == name]
        if not matches:
            raise ValueError(f"Unknown class '{name}'. Known: {self.classes}")
        return matches[0]

    def explain(self, image: Image.Image, target_class: str | int | None = None) -> dict:
        if isinstance(target_class, str):
            target_class = self.resolve_class_index(target_class)
        res = explain_image(self.model, image, self.device, target_class)
        res["pred_class"] = self.classes[res["pred_idx"]]
        res["target_class"] = self.classes[res["target_idx"]]
        return res

    def predict_and_explain(self, image: Image.Image, target_class: str | int | None = None) -> tuple[dict, dict]:
        """One forward + one backward pass: prediction dict and Grad-CAM result for the same image."""
        res = self.explain(image, target_class)
        return self._prediction_from_probs(res["probs"]), res


def run_on_images(paths: list[Path], expected: list[str | None], out_dir: Path, predictor: Predictor | None = None) -> pd.DataFrame:
    predictor = predictor or Predictor()
    out_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for path, exp in zip(paths, expected):
        image = load_image(path)
        t0 = time.time()
        pred, res = predictor.predict_and_explain(image)
        elapsed = time.time() - t0
        fig_path = out_dir / f"{path.stem}_gradcam.png"
        gradcam_figure(image, res, predictor.classes, true_label=predictor.display.get(exp, exp) if exp else None, path=fig_path)
        top3 = sorted(pred["probabilities"].items(), key=lambda kv: -kv[1])[:3]
        records.append({
            "image": relative_to_project(path), "expected": exp, "predicted": pred["pred_class"],
            "correct": (exp == pred["pred_class"]) if exp else None, "confidence": round(pred["confidence"], 4),
            "top3": "; ".join(f"{k}={v:.3f}" for k, v in top3), "gradcam_figure": relative_to_project(fig_path),
            "cam_blank": res["sanity"]["blank"], "cam_uniform": res["sanity"]["uniform"], "seconds": round(elapsed, 3),
            "image_size": f"{image.size[0]}x{image.size[1]}",
        })
        print(f"{path.name:40s} expected={exp} predicted={pred['pred_class']} conf={pred['confidence']:.3f}")
    df = pd.DataFrame.from_records(records)
    df.to_csv(out_dir / "unseen_predictions.csv", index=False)
    summary = {
        "n_images": int(len(df)),
        "n_with_expected_label": int(df["expected"].notna().sum()),
        "n_correct": int(df["correct"].fillna(False).sum()),
        "accuracy_on_labelled": float(df.loc[df["expected"].notna(), "correct"].mean()) if df["expected"].notna().any() else None,
        "mean_seconds_per_image": float(df["seconds"].mean()),
        "device": str(predictor.device),
        "records": records,
    }
    save_json(summary, out_dir / "unseen_predictions.json")
    return df


def sample_test_images(n_per_class: int, seed: int) -> tuple[list[Path], list[str]]:
    """Held-out test images were never used for training, validation or model selection."""
    test = pd.read_csv(SPLITS_DIR / "test.csv")
    rng = np.random.default_rng(seed)
    paths, labels = [], []
    for c in sorted(test["class_name"].unique()):
        pool = test.loc[test["class_name"] == c, "path"].to_numpy()
        for p in rng.choice(pool, size=min(n_per_class, len(pool)), replace=False):
            paths.append(absolute_from_project(p)); labels.append(c)
    return paths, labels


def main() -> None:
    p = argparse.ArgumentParser(description="Predict disease class (+ Grad-CAM) for new images")
    p.add_argument("images", nargs="*", help="image paths")
    p.add_argument("--from-test", type=int, default=0, help="also sample N held-out test images per class")
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--out", type=str, default=str(RESULTS_DIR / "unseen"))
    a = p.parse_args()
    paths = [Path(i) for i in a.images]
    expected: list[str | None] = [None] * len(paths)
    if a.from_test:
        tp, tl = sample_test_images(a.from_test, a.seed)
        paths += tp; expected += tl
    if not paths:
        p.error("give image paths and/or --from-test N")
    df = run_on_images(paths, expected, Path(a.out))
    labelled = df[df["expected"].notna()]
    if len(labelled):
        print(f"\nLabelled unseen images: {int(labelled['correct'].sum())}/{len(labelled)} correct")


if __name__ == "__main__":
    main()
