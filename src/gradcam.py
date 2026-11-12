"""Grad-CAM (Selvaraju et al., 2017) for the ResNet18 trunk of TomatoLeafCNN.

The target layer is `model.layer4`, the last convolutional stage ([B, 512, 7, 7] for 224x224
inputs). Grad-CAM highlights regions that increased the score of the explained class; it is a
coarse class-discriminative attribution map, NOT a lesion segmentation.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from .model import TomatoLeafCNN, load_checkpoint
from .preprocessing import load_image, preprocess_for_inference
from .utils import MODELS_DIR, PLOTS_DIR, RESULTS_DIR, SPLITS_DIR, absolute_from_project, get_device, save_json

plt.rcParams.update({"figure.dpi": 110, "savefig.bbox": "tight"})


class GradCAM:
    """Hooks the target layer to capture activations A_k and gradients dy_c/dA_k."""

    def __init__(self, model: TomatoLeafCNN, target_layer: torch.nn.Module | None = None) -> None:
        self.model = model
        self.target_layer = target_layer or model.gradcam_target_layer
        self.activations: torch.Tensor | None = None
        self.gradients: torch.Tensor | None = None
        self._handle = self.target_layer.register_forward_hook(self._forward_hook)

    def _forward_hook(self, module, inputs, output: torch.Tensor) -> None:
        self.activations = output.detach()
        if output.requires_grad:
            output.register_hook(self._save_gradient)

    def _save_gradient(self, grad: torch.Tensor) -> None:
        self.gradients = grad.detach()

    def remove(self) -> None:
        self._handle.remove()

    def __enter__(self) -> "GradCAM":
        return self

    def __exit__(self, *exc) -> None:
        self.remove()

    def __call__(self, x: torch.Tensor, target_class: int | None = None) -> dict:
        """x: [1, 3, H, W] normalised tensor. Returns raw + normalised CAM, probabilities, indices."""
        self.model.eval()
        self.model.zero_grad(set_to_none=True)
        self.activations = self.gradients = None
        with torch.enable_grad():
            x = x.clone().requires_grad_(True)
            logits = self.model(x)
            probs = torch.softmax(logits, dim=1)[0].detach()
            pred_idx = int(probs.argmax())
            target = pred_idx if target_class is None else int(target_class)
            logits[0, target].backward()
        if self.activations is None or self.gradients is None:
            raise RuntimeError("Grad-CAM hooks did not capture activations/gradients")

        weights = self.gradients.mean(dim=(2, 3), keepdim=True)           # alpha_k^c : [1, C, 1, 1]
        cam = F.relu((weights * self.activations).sum(dim=1, keepdim=True))  # [1, 1, h, w]
        raw_cam = cam[0, 0].detach().cpu().numpy()
        cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)[0, 0]
        cam = cam.detach().cpu().numpy()
        cam_norm = normalize_cam(cam)
        return {
            "cam": cam_norm,
            "raw_cam": raw_cam,
            "probs": probs.cpu().numpy(),
            "pred_idx": pred_idx,
            "target_idx": target,
            "logits": logits[0].detach().cpu().numpy(),
        }


def normalize_cam(cam: np.ndarray) -> np.ndarray:
    cam = cam.astype(np.float32)
    cam -= cam.min()
    peak = cam.max()
    return cam / peak if peak > 1e-12 else np.zeros_like(cam)


def resize_cam(cam: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """Resize a [h, w] CAM in [0,1] to (width, height) of the original image."""
    t = torch.from_numpy(cam)[None, None]
    out = F.interpolate(t, size=(size[1], size[0]), mode="bilinear", align_corners=False)[0, 0]
    return out.clamp(0, 1).numpy()


def cam_to_heatmap(cam: np.ndarray, cmap: str = "jet") -> np.ndarray:
    """[H, W] in [0,1] -> uint8 RGB heatmap."""
    colored = plt.get_cmap(cmap)(cam)[..., :3]
    return (colored * 255).astype(np.uint8)


def overlay_heatmap(image: Image.Image, cam: np.ndarray, alpha: float = 0.45) -> Image.Image:
    """Blend the heatmap on the original-resolution RGB image."""
    img = np.asarray(image.convert("RGB"), dtype=np.float32)
    if cam.shape != img.shape[:2]:
        cam = resize_cam(cam, image.size)
    heat = cam_to_heatmap(cam).astype(np.float32)
    blended = (1 - alpha) * img + alpha * heat
    return Image.fromarray(blended.clip(0, 255).astype(np.uint8))


def cam_sanity_report(result: dict) -> dict:
    raw, cam = result["raw_cam"], result["cam"]
    return {
        "raw_max": float(raw.max()),
        "raw_nonzero_fraction": float((raw > 0).mean()),
        "norm_min": float(cam.min()),
        "norm_max": float(cam.max()),
        "norm_std": float(cam.std()),
        "blank": bool(raw.max() <= 1e-12),
        "uniform": bool(cam.std() < 0.02),
        "normalised_correctly": bool(abs(cam.min()) < 1e-6 and (abs(cam.max() - 1.0) < 1e-6 or raw.max() <= 1e-12)),
    }


def explain_image(model: TomatoLeafCNN, image: Image.Image, device: torch.device, target_class: int | None = None) -> dict:
    """Full pipeline for one PIL image: preprocess -> Grad-CAM -> heatmap at original resolution."""
    x = preprocess_for_inference(image, model.config.img_size).to(device)
    with GradCAM(model) as gc:
        res = gc(x, target_class)
    cam_full = resize_cam(res["cam"], image.size)
    res["cam_full"] = cam_full
    res["heatmap"] = Image.fromarray(cam_to_heatmap(cam_full))
    res["overlay"] = overlay_heatmap(image, cam_full)
    res["sanity"] = cam_sanity_report(res)
    return res


def gradcam_figure(image: Image.Image, res: dict, class_names: list[str], true_label: str | None = None,
                   path: Path | None = None):
    pred = class_names[res["pred_idx"]]
    target = class_names[res["target_idx"]]
    blank_note = "\n(all-zero map: no region supports this class)" if res["sanity"]["blank"] else ""
    fig, axes = plt.subplots(1, 3, figsize=(11, 4))
    axes[0].imshow(image); axes[0].set_title("Original" + (f"\ntrue: {true_label}" if true_label else ""), fontsize=9)
    axes[1].imshow(res["heatmap"]); axes[1].set_title(f"Grad-CAM heatmap\nexplained class: {target}{blank_note}", fontsize=9)
    axes[2].imshow(res["overlay"]); axes[2].set_title(f"Overlay\npred: {pred} ({res['probs'][res['pred_idx']]:.2%})", fontsize=9)
    for ax in axes:
        ax.axis("off")
    if path is not None:
        fig.savefig(path)
        plt.close(fig)
    return fig


def class_specificity_check(model: TomatoLeafCNN, image: Image.Image, device: torch.device, top: dict | None = None) -> dict:
    """Grad-CAM for the top-1 class vs the 2nd most probable class should differ."""
    x = preprocess_for_inference(image, model.config.img_size).to(device)
    with GradCAM(model) as gc:
        top = top or gc(x)
        order = np.argsort(top["probs"])[::-1]
        second = gc(x, int(order[1]))
    a, b = top["cam"].flatten(), second["cam"].flatten()
    corr = float(np.corrcoef(a, b)[0, 1]) if a.std() > 0 and b.std() > 0 else float("nan")
    return {"top1_idx": int(order[0]), "top2_idx": int(order[1]), "cam_correlation": corr,
            "mean_abs_difference": float(np.abs(a - b).mean())}


def batch_examples(n_per_class: int = 1, seed: int = 0, device: torch.device | None = None) -> dict:
    """Grad-CAM grid over held-out test images (one row per class) + sanity statistics."""
    import pandas as pd

    device = device or get_device()
    model, mapping, _ = load_checkpoint(MODELS_DIR / "best_model.pth", device)
    classes, display = mapping["classes"], mapping["display_names"]
    test = pd.read_csv(SPLITS_DIR / "test.csv")
    rng = np.random.default_rng(seed)
    rows = []
    for c in classes:
        paths = test.loc[test["class_name"] == c, "path"].to_numpy()
        rows.extend((c, p) for p in rng.choice(paths, size=min(n_per_class, len(paths)), replace=False))

    fig, axes = plt.subplots(len(rows), 3, figsize=(9, 2.6 * len(rows)), squeeze=False)
    sanity, specificity = [], []
    for r, (c, p) in enumerate(rows):
        image = load_image(absolute_from_project(p))
        res = explain_image(model, image, device)
        sanity.append({"path": p, "true": c, "pred": classes[res["pred_idx"]], **res["sanity"]})
        specificity.append({"path": p, **class_specificity_check(model, image, device, top=res)})
        axes[r, 0].imshow(image); axes[r, 0].set_title(f"true: {display[c]}", fontsize=8)
        axes[r, 1].imshow(res["heatmap"]); axes[r, 1].set_title("Grad-CAM (predicted class)", fontsize=8)
        axes[r, 2].imshow(res["overlay"])
        axes[r, 2].set_title(f"pred: {display[classes[res['pred_idx']]]} ({res['probs'][res['pred_idx']]:.1%})", fontsize=8)
        for ax in axes[r]:
            ax.axis("off")
    fig.suptitle("Grad-CAM on held-out test images (layer4, explained class = predicted class)", y=1.0)
    fig.subplots_adjust(top=0.97, hspace=0.3)
    fig.savefig(PLOTS_DIR / "gradcam_examples.png")
    plt.close(fig)

    report = {
        "target_layer": "layer4",
        "images": sanity,
        "class_specificity": specificity,
        "n_blank": int(sum(s["blank"] for s in sanity)),
        "n_uniform": int(sum(s["uniform"] for s in sanity)),
        "all_normalised": bool(all(s["normalised_correctly"] for s in sanity)),
        "mean_cam_correlation_top1_vs_top2": float(np.nanmean([s["cam_correlation"] for s in specificity])),
    }
    save_json(report, RESULTS_DIR / "gradcam_checks.json")
    return report


def main() -> None:
    p = argparse.ArgumentParser(description="Grad-CAM for one image, or a batch check over test images")
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument("--image", type=str, help="path to an image")
    mode.add_argument("--batch-check", action="store_true", help="Grad-CAM grid + sanity checks over test images")
    p.add_argument("--target-class", type=str, default=None, help="class folder name to explain (default: predicted)")
    p.add_argument("--out", type=str, default=None)
    args = p.parse_args()
    device = get_device()
    if args.batch_check:
        rep = batch_examples(device=device)
        print({k: v for k, v in rep.items() if k not in ("images", "class_specificity")})
        return
    model, mapping, _ = load_checkpoint(MODELS_DIR / "best_model.pth", device)
    classes = mapping["classes"]
    image = load_image(args.image)
    target = classes.index(args.target_class) if args.target_class else None
    res = explain_image(model, image, device, target)
    print("pred:", classes[res["pred_idx"]], "explained:", classes[res["target_idx"]], "sanity:", res["sanity"])
    out = Path(args.out) if args.out else PLOTS_DIR / f"gradcam_{Path(args.image).stem}.png"
    gradcam_figure(image, res, classes, path=out)
    print("saved", out)


if __name__ == "__main__":
    matplotlib.use("Agg")
    main()
