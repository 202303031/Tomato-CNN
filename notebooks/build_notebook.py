"""Generate notebooks/Tomato_Leaf_CNN_Analysis.ipynb from the project modules (run from project root)."""
from __future__ import annotations

import json
from pathlib import Path

import nbformat as nbf
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"


def load(name: str) -> dict:
    p = RESULTS / name
    if not p.exists():
        raise FileNotFoundError(f"{p} is missing: run evaluate / gradcam / training before building the notebook")
    return json.loads(p.read_text())


metrics = load("test_metrics.json")
train_summary = load("training_summary.json")
data_summary = load("dataset_summary.json")
gradcam_checks = load("gradcam_checks.json")
history = pd.read_csv(RESULTS / "training_history.csv")

cells: list = []
md = lambda s: cells.append(nbf.v4.new_markdown_cell(s.strip()))
code = lambda s: cells.append(nbf.v4.new_code_cell(s.strip()))

# 1 ---------------------------------------------------------------------------------------------
md(r"""
# Tomato Leaf Disease Detection with a CNN Feature Extractor and an MLP Head

**Pipeline**

```
INPUT IMAGE → PREPROCESSING (RGB, 224×224, ImageNet normalisation)
            → CNN FEATURE EXTRACTOR (ResNet18 trunk, ImageNet-pretrained, layer3+layer4 fine-tuned)
            → FEATURE VECTOR (512-d, global average pooling)
            → MLP HEAD (Linear 512→256, ReLU, Dropout 0.3, Linear 256→C)
            → SOFTMAX → DISEASE CLASS + CLASS PROBABILITIES
            → GRAD-CAM (layer4) for explainability
```

This notebook reproduces the full workflow: dataset download, inspection, leak-free stratified splitting,
model construction, training, test evaluation, Grad-CAM and inference on unseen images. Every cell calls the
project's `src/` modules so the notebook and the scripts share one implementation.

> **Runtime note.** Training takes ~10–15 minutes on an Apple M-series GPU (MPS). The training cell runs the
> full training loop only when `models/best_model.pth` does not exist (or when `FORCE_RETRAIN = True`);
> otherwise it loads the saved history so the notebook runs top-to-bottom in a few minutes.
""")

# 2 ---------------------------------------------------------------------------------------------
md("## 2. Imports and environment checks")
code(r"""
import json, os, sys, time
from pathlib import Path

PROJECT_ROOT = Path.cwd().resolve()
if not (PROJECT_ROOT / "src").exists():
    PROJECT_ROOT = PROJECT_ROOT.parent
os.chdir(PROJECT_ROOT)
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from IPython.display import Image as IPyImage, display

from src.utils import (MODELS_DIR, PLOTS_DIR, RESULTS_DIR, SPLITS_DIR, PROCESSED_DIR,
                       environment_report, get_device, load_json, set_seed, absolute_from_project)

set_seed(42)
DEVICE = get_device()
ENV = environment_report(DEVICE)
print(json.dumps(ENV, indent=2))
assert ENV["selected_device"] in ("mps", "cpu", "cuda")
""")

# 3 ---------------------------------------------------------------------------------------------
md("""
## 3. Dataset download

Source: **PlantVillage** on Kaggle (`emmarex/plantdisease`, https://www.kaggle.com/datasets/emmarex/plantdisease),
downloaded with `kagglehub`. The dataset is public, so no credentials were needed; a cached copy is reused on re-runs.
""")
code(r"""
from src.download import download_dataset, find_class_root, record_source

download_path = download_dataset()
class_root = find_class_root(download_path)
source_info = record_source(download_path, class_root)
print("Download path:", download_path)
print("Class root   :", class_root)
""")

# 4 ---------------------------------------------------------------------------------------------
md("""
## 4. Dataset extraction

`kagglehub` extracts the archive automatically. The archive contains `PlantVillage/` and an identical nested
`PlantVillage/PlantVillage/` copy; `find_class_root` picks the shallowest folder so images are indexed once.
Non-tomato folders (Pepper, Potato) are listed here for transparency and excluded from everything that follows.
""")
code(r"""
folders = sorted(d for d in os.listdir(class_root) if (class_root / d).is_dir() and d != "PlantVillage")
rows = [{"folder": d, "is_tomato": d.startswith("Tomato"),
         "files": sum(1 for _ in (class_root / d).iterdir())} for d in folders]
folder_df = pd.DataFrame(rows)
display(folder_df)
print("Tomato folders:", int(folder_df.is_tomato.sum()), "| excluded non-tomato folders:", int((~folder_df.is_tomato).sum()))
""")

# 5 ---------------------------------------------------------------------------------------------
md("""
## 5. Dataset inspection

`src.dataset.prepare_dataset` indexes every tomato image in parallel: it fully decodes each file (corrupt
detection), records resolution/mode/format, and computes an MD5 (exact duplicates) and a 64-bit dHash
(near-duplicates, Hamming distance ≤ 4). Class names are derived from the folder names.
""")
code(r"""
from src.dataset import prepare_dataset, discover_tomato_classes, display_name
from src import explore

if not (SPLITS_DIR / "test.csv").exists():
    prepare_dataset(seed=42, workers=8)

classes = discover_tomato_classes(class_root)
print(f"{len(classes)} tomato classes:")
for c in classes:
    print(f"  {c:48s} -> {display_name(c)}")

index = pd.read_csv(PROCESSED_DIR / "image_index.csv")
clean = pd.read_csv(PROCESSED_DIR / "image_index_clean.csv")
print(f"\nIndexed files: {len(index)} | corrupt/unreadable: {(~index.ok).sum()} | usable: {len(clean)}")
display(index.loc[~index.ok, ["path", "error"]] if (~index.ok).any() else "No corrupt images found.")

explore.main()  # regenerates class distribution, dimension, sample and split plots
summary = load_json(RESULTS_DIR / "dataset_summary.json")
counts = pd.Series(summary["class_counts"]).rename("images").to_frame()
counts["share_%"] = (100 * counts.images / counts.images.sum()).round(2)
display(counts)
print("Imbalance ratio (max/min):", summary["imbalance_ratio_max_over_min"])
print("Resolutions:", summary["image_dimensions"], "| modes:", summary["image_modes"], "| formats:", summary["image_formats"])
display(IPyImage(filename=str(PLOTS_DIR / "class_distribution.png")))
display(IPyImage(filename=str(PLOTS_DIR / "sample_images.png"), width=900))
""")

# 6 ---------------------------------------------------------------------------------------------
md("""
## 6. Data preprocessing

* Convert to RGB, resize to 224×224, normalise with ImageNet mean/std (required by the pretrained ResNet18).
* **Training only**: RandomResizedCrop (scale 0.75–1), horizontal + vertical flips, ±15° rotation, mild colour jitter.
* Validation / test / inference use the deterministic resize + normalise transform only.
""")
code(r"""
from src.preprocessing import get_train_transforms, get_eval_transforms, load_image, denormalize, IMAGENET_MEAN, IMAGENET_STD

print("Train transform:\n", get_train_transforms(), "\n\nEval transform:\n", get_eval_transforms())
sample_path = absolute_from_project(clean.iloc[0].path)
img = load_image(sample_path)
x_eval = get_eval_transforms()(img)
print("\nEval tensor shape:", tuple(x_eval.shape), "| value range: [%.2f, %.2f]" % (x_eval.min(), x_eval.max()))

fig, axes = plt.subplots(1, 7, figsize=(17, 2.8))
axes[0].imshow(img); axes[0].set_title("original"); axes[0].axis("off")
torch.manual_seed(0)
for ax in axes[1:]:
    ax.imshow(denormalize(get_train_transforms()(img))); ax.set_title("augmented"); ax.axis("off")
display(fig); plt.close(fig)
""")

# 7 ---------------------------------------------------------------------------------------------
md("""
## 7. Train / validation / test splitting (70 / 15 / 15)

Splitting is **stratified by class and grouped by duplicate cluster** (seed 42): all exact or near-duplicate
images of a leaf land in the same subset, so no image (or near copy) can appear in two subsets. The test set
is never used for training, augmentation selection, hyper-parameter tuning or early stopping.
""")
code(r"""
split_report = load_json(SPLITS_DIR / "split_summary.json")
print("Counts   :", split_report["counts"])
print("Fractions:", split_report["fractions"])
print("Overlaps :", split_report["overlaps"])
print("Leakage-free:", split_report["leakage_free"])
print("Duplicates:", split_report["duplicates"])
per_class = pd.DataFrame(split_report["per_class_counts"]).T
per_class_frac = pd.DataFrame(split_report["per_class_fractions"]).T
display(pd.concat({"count": per_class, "fraction": per_class_frac}, axis=1))

for s in ("train", "val", "test"):
    d = pd.read_csv(SPLITS_DIR / f"{s}.csv"); print(s, len(d), "rows,", d.label.nunique(), "classes")
assert split_report["leakage_free"]
display(IPyImage(filename=str(PLOTS_DIR / "split_distribution.png")))
""")

# 8 ---------------------------------------------------------------------------------------------
md("""
## 8. CNN architecture

**Why transfer learning with ResNet18?** ~16k images of 256×256 leaves is small for training a deep CNN from
scratch, and the hardware is a 16 GB MacBook (Apple MPS). ImageNet-pretrained filters give strong generic
edge/texture features, converge in a few epochs and generalise better than a from-scratch network of the same
size. ResNet18 (11.2M parameters) is the smallest standard residual network, so it trains quickly on MPS.

| Component | Layers | Pretrained? | Trained here? |
|---|---|---|---|
| Stem (`conv1`, `bn1`, ReLU, maxpool) | 7×7 conv s2 + BN + 3×3 maxpool | ImageNet | **frozen** (BN in eval mode) |
| `layer1`, `layer2` | 2 BasicBlocks each (64, 128 channels) | ImageNet | **frozen** |
| `layer3`, `layer4` | 2 BasicBlocks each (256, 512 channels) | ImageNet | **fine-tuned** (lr 1e-4) |
| Global average pooling | AdaptiveAvgPool2d(1) → flatten | – | – |
| MLP head | Linear 512→256, ReLU, Dropout 0.3, Linear 256→C | random init | **trained** (lr 1e-3) |

Every CNN stage contains convolutions, batch normalisation, ReLU and (via strides / maxpool) pooling. The
feature extractor output is a 512-d vector; the MLP head produces one logit per class and softmax is applied
only at inference. Loss: `CrossEntropyLoss` with inverse-sqrt class weights and label smoothing 0.05.
""")
code(r"""
from src.model import ModelConfig, build_model, model_summary, trace_shapes, parameter_breakdown

mapping = load_json(MODELS_DIR / "class_mapping.json")
config = ModelConfig(num_classes=mapping["num_classes"])
model = build_model(config)
print(model)
""")

# 9 ---------------------------------------------------------------------------------------------
md("## 9. Model summary: tensor shapes at every stage and parameter counts")
code(r"""
print(model_summary(model))
params = parameter_breakdown(model)
display(pd.DataFrame(params).T)
""")

# 10 --------------------------------------------------------------------------------------------
md("""
## 10. Training

AdamW (backbone lr 1e-4, head lr 1e-3, weight decay 1e-4), `ReduceLROnPlateau` on validation macro-F1,
early stopping (patience 4) on validation macro-F1, batch size 32, up to 15 epochs, seed 42. The best
checkpoint is selected on **validation** macro-F1 only. Training writes `models/best_model.pth`,
`models/model_config.json`, `results/training_history.csv` and `results/training_summary.json`.
""")
code(r"""
from src.train import train, parse_args

FORCE_RETRAIN = False
if FORCE_RETRAIN or not (MODELS_DIR / "best_model.pth").exists():
    training_summary = train(parse_args(["--epochs", "15", "--batch-size", "32", "--num-workers", "6", "--patience", "4"]))
else:
    training_summary = load_json(RESULTS_DIR / "training_summary.json")
    print("Using existing checkpoint; training summary from the saved run:")
print(json.dumps({k: training_summary[k] for k in ("device", "epochs_run", "best_epoch", "best_val_macro_f1",
                                                     "best_val_acc", "total_training_seconds", "mean_epoch_seconds")}, indent=2))
print("Hyper-parameters:", json.dumps({k: v for k, v in training_summary["hyperparameters"].items()
                                       if k not in ("class_weights", "train_class_counts")}, indent=2))
""")

# 11 --------------------------------------------------------------------------------------------
md("## 11. Training graphs")
code(r"""
history = pd.read_csv(RESULTS_DIR / "training_history.csv")
display(history.round(4))
display(IPyImage(filename=str(PLOTS_DIR / "training_curves.png")))
best = history.loc[history.epoch == training_summary["best_epoch"]].iloc[0]
print(f"Selected checkpoint = epoch {int(best.epoch)}: val acc {best.val_acc:.4f}, val macro-F1 {best.val_macro_f1:.4f}, "
      f"train acc {best.train_acc:.4f}; train-val loss gap {best.train_loss - best.val_loss:+.4f}")
worse = history[history.val_loss > history.train_loss].epoch.tolist()
print("Epochs where validation loss exceeded training loss:", worse or "none")
""")
_h = history
_best = int(train_summary["best_epoch"]); _n = int(train_summary["epochs_run"])
_cross = _h[_h.val_loss > _h.train_loss].epoch.astype(int).tolist()
_dips = _h.epoch[(_h.val_macro_f1.diff() < 0)].astype(int).tolist()
_max_dip = float((-_h.val_macro_f1.diff()).clip(lower=0).max() * 100)
_lr_changes = _h.epoch[_h.lr_backbone.diff() < 0].astype(int).tolist()
_last = _h.iloc[-1]
md(f"""
**Reading the curves.** Training ran for {_n} epochs ({train_summary['total_training_seconds']/60:.1f} min in total,
{train_summary['mean_epoch_seconds']:.0f} s per epoch on average; the first epoch is slower because of one-off kernel
compilation and DataLoader start-up). The validation loss lies *below* the training loss for most epochs
(it exceeds it only at epochs {_cross if _cross else 'none'}); this is expected because the training loss is computed
with augmentation, dropout, class weights and label smoothing active, whereas validation uses clean images with
dropout off. Validation macro-F1 climbs from {_h.val_macro_f1.iloc[0]:.4f} to {_h.val_macro_f1.max():.4f}; it dips
slightly at epochs {_dips} (largest dip {_max_dip:.2f} percentage points) but trends upward throughout.
`ReduceLROnPlateau` halved both learning rates after epochs {_lr_changes}. At the final epoch training accuracy
({_last.train_acc:.4f}) and validation accuracy ({_last.val_acc:.4f}) are essentially equal, so there is no sign of
over-fitting, and no sign of under-fitting either (both losses fall together and training accuracy approaches 1.0).
The checkpoint selected on validation macro-F1 is epoch **{_best}**{' (the last epoch: early stopping never triggered, so the run was capped by the epoch budget and a slightly longer schedule might give a marginal further gain)' if _best == _n else ''}.
""")


# 12 --------------------------------------------------------------------------------------------
md("""
## 12. Test evaluation (untouched test split)

`src.evaluate.evaluate` first re-computes the validation metrics from the loaded checkpoint and compares them with
the values recorded during training (a checkpoint-loading check), then evaluates the test split once.
""")
code(r"""
from src.evaluate import evaluate

test_metrics = evaluate(num_workers=4)
print(json.dumps({k: test_metrics[k] for k in ("test_images", "accuracy", "macro_precision", "macro_recall",
                                                "macro_f1", "weighted_f1", "n_incorrect")}, indent=2))
display(pd.DataFrame(test_metrics["per_class"]).T.round(4))
print(open(RESULTS_DIR / "classification_report.txt").read())
""")

# 13 --------------------------------------------------------------------------------------------
md("## 13. Confusion matrix, per-class chart, correct and incorrect examples")
code(r"""
display(IPyImage(filename=str(RESULTS_DIR / "confusion_matrix.png"), width=1100))
display(IPyImage(filename=str(PLOTS_DIR / "per_class_metrics.png"), width=900))
print("Most frequent confusions (true -> predicted):")
display(pd.DataFrame(test_metrics["top_confusions"]))
display(IPyImage(filename=str(PLOTS_DIR / "correct_predictions.png"), width=900))
display(IPyImage(filename=str(PLOTS_DIR / "incorrect_predictions.png"), width=900))
""")

# 14 --------------------------------------------------------------------------------------------
md("""
## 14. Grad-CAM (explainable AI)

Grad-CAM hooks `layer4` (the last convolutional stage, 512×7×7). For the explained class *c* it computes
α<sub>k</sub> = mean of ∂y<sub>c</sub>/∂A<sup>k</sup>, then CAM = ReLU(Σ<sub>k</sub> α<sub>k</sub> A<sup>k</sup>),
up-samples it bilinearly to the input size, min-max normalises it to [0, 1] and overlays a `jet` colour map on
the original-resolution image. The explained class defaults to the predicted class and can be overridden.

**Scientific caveat.** Grad-CAM shows *which regions increased the class score*. It is a coarse (7×7 → upsampled)
attribution, not a lesion segmentation: it does not delineate individual bacterial spots. Pixel-accurate lesion
localisation would need an annotated segmentation dataset and a separate segmentation/detection model.
""")
code(r"""
from src.gradcam import batch_examples, explain_image, gradcam_figure, class_specificity_check
from src.model import load_checkpoint

checks = batch_examples(n_per_class=1, seed=0, device=DEVICE)
print({k: v for k, v in checks.items() if k not in ("images", "class_specificity")})
display(pd.DataFrame(checks["images"])[["true", "pred", "raw_max", "norm_min", "norm_max", "norm_std", "blank", "uniform"]].round(3))
display(IPyImage(filename=str(PLOTS_DIR / "gradcam_examples.png"), width=750))
""")
code(r"""
# Target-class selection: explain the predicted class vs. a user-chosen class on the same image.
trained_model, mapping, meta = load_checkpoint(MODELS_DIR / "best_model.pth", DEVICE)
classes = mapping["classes"]
example = sorted((PROJECT_ROOT / "examples").glob("*Bacterial*"))[0]
image = load_image(example)
res_pred = explain_image(trained_model, image, DEVICE)
other = [i for i in range(len(classes)) if i != res_pred["pred_idx"]][0]
res_other = explain_image(trained_model, image, DEVICE, target_class=other)
print("Predicted:", classes[res_pred["pred_idx"]], "| sanity:", res_pred["sanity"])
print("Specificity (top-1 vs top-2 CAM correlation):", class_specificity_check(trained_model, image, DEVICE))
fig = gradcam_figure(image, res_pred, classes, true_label="Tomato_Bacterial_spot"); display(fig); plt.close(fig)
fig = gradcam_figure(image, res_other, classes, true_label="Tomato_Bacterial_spot"); display(fig); plt.close(fig)
""")

# 15 --------------------------------------------------------------------------------------------
md("""
## 15. Unseen image prediction

`src.predict.Predictor` loads the checkpoint and class mapping once and applies exactly the inference transform.
The images in `examples/` are held-out **test** images (never seen in training or validation). Results are also
recorded in `results/unseen_notebook/unseen_predictions.csv`.
""")
code(r"""
from src.predict import Predictor, run_on_images

predictor = Predictor(device=DEVICE)
example_paths = sorted((PROJECT_ROOT / "examples").glob("*.JPG"))
expected = [p.stem for p in example_paths]
unseen_df = run_on_images(example_paths, expected, RESULTS_DIR / "unseen_notebook", predictor)
display(unseen_df[["image", "expected", "predicted", "correct", "confidence", "top3"]])
print(f"Correct: {int(unseen_df.correct.sum())}/{len(unseen_df)}")
for fig_path in unseen_df.gradcam_figure.head(3):
    display(IPyImage(filename=str(PROJECT_ROOT / fig_path), width=800))
""")

# 16 --------------------------------------------------------------------------------------------
acc = metrics.get("accuracy", float("nan"))
mf1 = metrics.get("macro_f1", float("nan"))
md(f"""
## 16. Conclusion

* A ResNet18 feature extractor (ImageNet-pretrained, `layer3`/`layer4` fine-tuned) with a 512→256→C MLP head was
  trained on {data_summary.get('usable_images', '?')} PlantVillage tomato leaf images across
  {data_summary.get('num_classes', '?')} classes using Apple MPS.
* Held-out **test accuracy {acc:.4f}** and **macro-F1 {mf1:.4f}** on {metrics['test_images']} images
  ({metrics['n_incorrect']} errors; see Section 12 for the full table); the checkpoint was chosen on validation macro-F1 only.
* Grad-CAM sanity checks over one test image per class: {gradcam_checks['n_blank']} blank maps,
  {gradcam_checks['n_uniform']} uniform maps, all normalised = {gradcam_checks['all_normalised']}, mean top-1 vs top-2
  CAM correlation {gradcam_checks['mean_cam_correlation_top1_vs_top2']:.3f} (class-specific). The maps concentrate on
  leaf tissue and symptomatic regions, but they are attribution maps, not lesion segmentations.
* Limitations: PlantVillage images are lab-style single leaves on plain backgrounds, so field performance will be
  lower; the near-duplicate structure of the dataset makes benchmark numbers optimistic even with grouped
  splitting; the model only knows these {data_summary.get('num_classes', '?')} classes and will still output a
  class for unrelated images.
* See `PROJECT_REPORT.md` for the complete write-up.
""")

nb = nbf.v4.new_notebook(cells=cells)
nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3 (tomato-venv)", "language": "python"}
nb.metadata["language_info"] = {"name": "python"}
out = ROOT / "notebooks" / "Tomato_Leaf_CNN_Analysis.ipynb"
nbf.write(nb, out)
print("wrote", out, "with", len(cells), "cells")
