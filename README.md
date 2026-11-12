# Tomato Leaf Disease Detection

This PyTorch project classifies tomato leaf photos into 10 classes (9 disease or pest classes plus healthy) from the PlantVillage dataset. An ImageNet-pretrained ResNet18 convolutional trunk turns each image into a 512-dimensional feature vector, and a small MLP head turns that vector into class probabilities. Grad-CAM on the last convolutional stage shows which image regions influenced a prediction. The repository covers the whole workflow: a reproducible download, integrity and duplicate checks, a leakage-free stratified 70/15/15 split, training with early stopping, held-out test evaluation, command-line inference, a Jupyter notebook and a local Gradio web app. It was built for a MacBook with an Apple M5 chip and 16 GB of RAM, and the device is picked automatically in the order MPS, then CUDA, then CPU.

## Pipeline

```text
INPUT IMAGE  (any size, converted to RGB)
     |
     v
PREPROCESSING  resize to 224x224, ImageNet mean/std normalisation
     |         (training only: random resized crop, H/V flips, rotation, colour jitter)
     v
CNN FEATURE EXTRACTOR  ResNet18 trunk: stem -> layer1 -> layer2 -> layer3 -> layer4
     |                                                                  |
     |  global average pooling of the [512, 7, 7] layer4 feature map   +--> GRAD-CAM (layer4)
     v                                                                        class-specific heatmap
FEATURE VECTOR (512-d)                                                        + overlay on the input
     |
     v
MLP HEAD  Linear(512 -> 256) -> ReLU -> Dropout(0.3) -> Linear(256 -> 10)
     |
     v
SOFTMAX
     |
     v
CLASS + PROBABILITIES  (top-1 class and a probability for each of the 10 classes)
```

## Dataset

- **Source:** the Kaggle "PlantVillage Dataset" uploaded by Tairu Oluwafemi Emmanuel (`emmarex/plantdisease`, https://www.kaggle.com/datasets/emmarex/plantdisease). The uploader's acknowledgement says it was taken from spMohanty's GitHub repository [PlantVillage-Dataset](https://github.com/spMohanty/PlantVillage-Dataset), which redistributes the PlantVillage images of Hughes & Salathé (2015, arXiv:1511.08060).
- **Download:** `kagglehub` downloads it without credentials because the dataset is public. The archive is cached in `data/raw/kagglehub_cache/` and reused on later runs.
- **License:** the Kaggle page lists the license as "Unknown". This project uses the images only for academic, non-commercial education and research, and credits PlantVillage (Hughes & Salathé) and the Kaggle listing. Check the terms yourself before any other use.
- **Scope:** only the 10 folders whose names start with `Tomato` are used. The Pepper and Potato folders are left out. The archive also contains an identical nested copy, `PlantVillage/PlantVillage/`, which is ignored so that no image is indexed twice.

| Label | Folder name | Display name | Images | Share |
|---:|---|---|---:|---:|
| 0 | `Tomato_Bacterial_spot` | Bacterial spot | 2127 | 13.3% |
| 1 | `Tomato_Early_blight` | Early blight | 1000 | 6.2% |
| 2 | `Tomato_Late_blight` | Late blight | 1909 | 11.9% |
| 3 | `Tomato_Leaf_Mold` | Leaf Mold | 952 | 5.9% |
| 4 | `Tomato_Septoria_leaf_spot` | Septoria leaf spot | 1771 | 11.1% |
| 5 | `Tomato_Spider_mites_Two_spotted_spider_mite` | Spider mites Two spotted spider mite | 1676 | 10.5% |
| 6 | `Tomato__Target_Spot` | Target Spot | 1404 | 8.8% |
| 7 | `Tomato__Tomato_YellowLeaf__Curl_Virus` | Tomato Yellow Leaf Curl Virus | 3208 | 20.0% |
| 8 | `Tomato__Tomato_mosaic_virus` | Tomato mosaic virus | 373 | 2.3% |
| 9 | `Tomato_healthy` | Healthy | 1591 | 9.9% |
| | **Total usable images** | | **16011** | |

**Integrity:** 16011 files were indexed and every one was fully decoded. 0 files were corrupt or unreadable, so all 16011 images are usable. They are all 256x256 RGB JPEGs. The largest class has 8.6 times as many images as the smallest (`imbalance_ratio_max_over_min`). Training corrects for this mildly with class weights.

**Duplicates:** exact duplicates are images with the same MD5 hash. Near-duplicates are images whose 64-bit dHash values differ in at most 4 bits. Matching images are joined into groups (from `data/splits/split_summary.json`):

| Statistic | Value |
|---|---:|
| Unique groups (out of 16011 images) | 15977 |
| Groups containing more than one image | 34 |
| Images in those groups | 68 |
| Exact-duplicate images / near-duplicate images | 28 / 40 |
| Largest group size | 2 |
| Groups spanning more than one class | 2 |

**Splits (seed 42):** the 70/15/15 split is **stratified by class and grouped by duplicate cluster**. Each group goes to exactly one split, which is chosen from the group's majority class. The verification step found 0 shared paths, 0 shared groups and 0 shared MD5 hashes between every pair of splits (`leakage_free: true`). Per-class counts are in `data/splits/split_summary.json`.

| Split | Images | Fraction |
|---|---:|---:|
| train | 11208 | 0.7 |
| val | 2404 | 0.1501 |
| test | 2399 | 0.1498 |

The test split is never used for training, early stopping or checkpoint selection.

## Model

`src/model.py` defines `TomatoLeafCNN`, which is a ResNet18 trunk (torchvision `IMAGENET1K_V1` weights) followed by global average pooling and an MLP head. Parameter counts come from `results/model_summary.txt`:

| Stage | Initialisation | Training status | Parameters |
|---|---|---|---:|
| Stem (`conv1`, `bn1`, ReLU, maxpool) | ImageNet | frozen | 9,536 |
| `layer1` (64 channels) | ImageNet | frozen | 147,968 |
| `layer2` (128 channels) | ImageNet | frozen | 525,568 |
| `layer3` (256 channels) | ImageNet | fine-tuned | 2,099,712 |
| `layer4` (512 channels, Grad-CAM target) | ImageNet | fine-tuned | 8,393,728 |
| MLP head: Linear(512->256), ReLU, Dropout(0.3), Linear(256->10) | random | trained | 133,898 |
| **Total** | | trainable 10,627,338 / frozen 683,072 | **11,310,410** |

Frozen stages are kept in eval mode, so their BatchNorm running statistics stay fixed. Softmax is applied only at inference time.

**Training setup (`src/train.py`):**
- **Optimiser:** AdamW with separate learning rates for the backbone and the head.
- **Learning-rate schedule:** `ReduceLROnPlateau` (factor 0.5, patience 2) on validation macro-F1.
- **Loss:** `CrossEntropyLoss` with inverse-square-root class weights (normalised to mean 1) and label smoothing.
- **Stopping and checkpointing:** early stopping on validation macro-F1. The best epoch is saved to `models/best_model.pth`.

The command-line flags and their defaults:

| Flag | Default | Flag | Default |
|---|---|---|---|
| `--epochs` | 15 | `--backbone-lr` | 1e-4 |
| `--batch-size` | 32 (validation uses 2x) | `--head-lr` | 1e-3 |
| `--num-workers` | 6 | `--weight-decay` | 1e-4 |
| `--img-size` | 224 | `--label-smoothing` | 0.05 |
| `--hidden-dim` | 256 | `--patience` | 4 |
| `--dropout` | 0.3 | `--min-delta` | 1e-4 |
| `--trainable-stages` | `layer3 layer4` | `--seed` | 42 |
| `--device` | auto (MPS -> CUDA -> CPU) | `--smoke` | off (2 batches per epoch) |

## Project structure

```text
Tomato_Leaf_Disease_Detection/
├── app.py                  # Gradio web app
├── requirements.txt        # pinned dependencies
├── .claude/launch.json     # "gradio-app" launch config (.venv/bin/python app.py --port 7860)
├── src/
│   ├── __init__.py
│   ├── utils.py            # paths, seeding, device selection, JSON helpers
│   ├── download.py         # kagglehub download + class-folder discovery
│   ├── dataset.py          # indexing, integrity, duplicate groups, stratified group split, Dataset
│   ├── explore.py          # dataset statistics and plots
│   ├── preprocessing.py    # train / eval transforms, inference helpers
│   ├── model.py            # ResNet18 trunk + MLP head, freezing, checkpoint loading
│   ├── train.py            # training CLI
│   ├── evaluate.py         # test-set evaluation CLI
│   ├── gradcam.py          # Grad-CAM implementation + CLI
│   ├── predict.py          # Predictor class + CLI for new images
│   └── disease_info.py     # short class notes and the disclaimer shown in the app
├── notebooks/
│   ├── build_notebook.py   # generates the notebook below from the src/ modules + results/*.json
│   └── Tomato_Leaf_CNN_Analysis.ipynb  # executed end-to-end notebook (16 sections)
├── examples/               # 10 held-out test images (one per class) used as app examples
├── data/
│   ├── raw/kagglehub_cache/  # downloaded Kaggle archive
│   ├── processed/          # image_index.csv, image_index_clean.csv, dataset_source.json,
│   │                       # skipped_files.json
│   └── splits/             # train.csv, val.csv, test.csv, split_summary.json
├── models/                 # best_model.pth, model_config.json, class_mapping.json
├── PROJECT_REPORT.md       # academic report with the actual results
└── results/                # summaries, metrics, reports, training history, training_log.txt
    ├── plots/              # all figures
    ├── unseen/             # predictions + Grad-CAM on held-out images (src.predict)
    └── unseen_notebook/    # the same, produced by the notebook's Section 15
```

## Setup

```bash
cd Tomato_Leaf_Disease_Detection
uv venv --python 3.12 .venv
source .venv/bin/activate
uv pip install -r requirements.txt
```

You can also create the venv with `python -m venv .venv` (using Python 3.12) and install with `pip install -r requirements.txt`. The first `python -m src.dataset` needs internet access to download the Kaggle archive. The first training run also needs it, to download the torchvision ImageNet ResNet18 weights. Loading a saved checkpoint for inference downloads nothing.

## Reproduce

Run every command from the project root with the venv active.

1. **Download, index, deduplicate and split:** `python -m src.dataset` (`--seed`, `--workers`). This writes `data/processed/`, `models/class_mapping.json` and `data/splits/{train,val,test}.csv` + `split_summary.json`. It stops with an error if it finds any overlap between splits.
2. **Explore:** `python -m src.explore` writes `results/dataset_summary.{json,md}`. It also writes the class-distribution, image-dimension, sample-image, split-distribution and duplicate-group plots to `results/plots/`.
3. **Train:** `python -m src.train` (see the flags above). This writes `models/best_model.pth`, `models/model_config.json`, `results/model_summary.txt`, `results/training_history.csv`, `results/training_summary.json` and `results/plots/training_curves.png`. For a quick pipeline check, run `python -m src.train --smoke --epochs 1`, which uses 2 batches per epoch and writes every artefact with a `_smoke` suffix (`models/best_model_smoke.pth`, `results/training_history_smoke.csv`, ...), so it never overwrites a real run.
4. **Evaluate:** `python -m src.evaluate` (`--batch-size`, `--num-workers`). It reloads the checkpoint and recomputes the validation metrics to confirm the checkpoint loaded correctly. Then it evaluates the test split once.
5. **Grad-CAM:**
   - `python -m src.gradcam --batch-check` explains one test image per class. It writes `results/plots/gradcam_examples.png` and `results/gradcam_checks.json`.
   - `python -m src.gradcam --image <path> [--target-class <folder name>]` explains a single image and saves `results/plots/gradcam_<image name>.png`, or the file given with `--out`. `--target-class` takes a folder name such as `Tomato_Late_blight`. Without it, the predicted class is explained.
6. **Predict:** `python -m src.predict --from-test 2` samples 2 held-out test images per class. `python -m src.predict img1.jpg img2.jpg` runs on your own files. Either command writes a Grad-CAM figure per image, plus `unseen_predictions.csv` and `unseen_predictions.json`, to `results/unseen/` (change it with `--out`).
7. **App:** `python app.py`, then open http://127.0.0.1:7860 (`--port`, `--share`).
8. **Notebook:** `python notebooks/build_notebook.py`, then `jupyter nbconvert --to notebook --execute --inplace notebooks/Tomato_Leaf_CNN_Analysis.ipynb` (or open it in Jupyter). Build it after evaluation, because the builder copies metrics from `results/*.json` into the notebook text. The training cell reuses `models/best_model.pth` if it exists; set `FORCE_RETRAIN = True` to retrain.

**If MPS runs out of memory:**
- Lower `--batch-size` first (default 32).
- Then lower `--num-workers` (default 6).
- Only as a last resort, lower `--img-size` (default 224).

`src.evaluate` also accepts `--batch-size` and `--num-workers`.

## Results

Test results for the checkpoint with the best validation macro-F1, on the held-out test split (2399 images):

| Metric | Value |
|---|---|
| Test accuracy | 0.9946 |
| Test macro-F1 | 0.9944 |
| Test macro precision | 0.9943 |
| Test macro recall | 0.9945 |
| Best epoch (validation macro-F1) | 15 |
| Epochs run | 15 |
| Training time (minutes) | 16.8 |
| Device | mps |
| Grad-CAM checks (one test image per class) | 0 blank, 0 uniform maps; top-1 vs top-2 map correlation 0.056 |
| Unseen-image test (`src.predict --from-test 2`) | 20/20 held-out images correct |

Result artefacts:

- `results/confusion_matrix.png`: confusion matrix, as counts and row-normalised
- `results/classification_report.txt`: per-class precision, recall and F1
- `results/test_metrics.json`: all test metrics, top confusions and the checkpoint check
- `results/training_history.csv`: loss, accuracy, macro-F1 and learning rates per epoch
- `results/plots/training_curves.png`: loss, accuracy and macro-F1 curves
- `results/plots/per_class_metrics.png`: per-class precision, recall and F1 bars
- `results/plots/gradcam_examples.png`: Grad-CAM on one held-out test image per class
- `results/plots/correct_predictions.png`: correctly classified test examples
- `results/plots/incorrect_predictions.png`: misclassified test examples, highest confidence first
- `results/unseen/`: predictions and Grad-CAM figures from `src.predict`

## Gradio app

`python app.py` loads `models/best_model.pth` once and serves a local app at http://127.0.0.1:7860. The app needs a trained checkpoint.

- **Upload:** add a leaf image from a file or the clipboard, or pick one of the 10 example images in `examples/`. The examples are held-out test images, one per class, that were never used for training or validation.
- **Run prediction:** a button runs the classifier and Grad-CAM.
- **Grad-CAM target class dropdown:** keep "Predicted class (default)" or pick any of the 10 classes to see which regions support that class instead.
- **Clear / reset:** clears the image and all outputs and resets the dropdown.
- **Probabilities:** a label shows the predicted class and its confidence, and a bar chart shows the probability for every class.
- **Grad-CAM heatmap and overlay:** the heatmap shown on its own and blended over the original image.
- **Disease note and disclaimer:** a short note about the predicted class (from `src/disease_info.py`), the class Grad-CAM explained, and a disclaimer that this is not a diagnosis.

## Explainability caveat

Grad-CAM weights the 512 `layer4` feature maps by the averaged gradient of the explained class score. It keeps only the positive evidence, then upsamples the 7x7 map to the image size. The result shows **which regions increased the score of the explained class**. It is a coarse attribution map, **not a lesion segmentation**: it does not mark the boundaries of individual spots or measure how much of the leaf is affected. A warm region means only that the model relied on that region, not that the region is diseased.

## Limitations

- **Lab-style images:** PlantVillage shows single leaves photographed on plain backgrounds, and all images in this subset are 256x256. Field photos with several leaves, cluttered backgrounds or different lighting are likely to score lower than the test set suggests.
- **Only 10 classes:** other tomato diseases, nutrient deficiencies and leaves with more than one problem are not covered.
- **No out-of-distribution rejection:** softmax always picks one of the 10 classes, even for non-tomato or non-leaf images. A high confidence does not mean the input is valid.
- **Class imbalance:** Tomato mosaic virus has only 373 images (56 in the test split), so its per-class metrics are noisier.
- **Duplicate checks are limited:** they catch only exact copies and near-copies within the dHash threshold. Different photos of the same leaf can still land in different splits, which can make test scores optimistic.
- **Single run:** the reported numbers come from one training run. Variation across seeds is not measured.

## Disclaimer

This is an experimental machine-learning project for education and research. Its predictions and Grad-CAM maps are **not an agricultural or plant-pathology diagnosis**. Do not use them to decide on treatment, pesticide use or crop removal; consult an agronomist or plant pathologist instead.

## Attribution and references

- D. P. Hughes and M. Salathé (2015). *An open access repository of images on plant health to enable the development of mobile disease diagnostics.* arXiv:1511.08060.
- S. P. Mohanty, D. P. Hughes and M. Salathé (2016). *Using Deep Learning for Image-Based Plant Disease Detection.* Frontiers in Plant Science. doi:10.3389/fpls.2016.01419.
- K. He, X. Zhang, S. Ren and J. Sun (2016). *Deep Residual Learning for Image Recognition.* CVPR. arXiv:1512.03385.
- R. R. Selvaraju, M. Cogswell, A. Das, R. Vedantam, D. Parikh and D. Batra (2017). *Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization.* ICCV. arXiv:1610.02391.
- Kaggle dataset "PlantVillage Dataset" by Tairu Oluwafemi Emmanuel: https://www.kaggle.com/datasets/emmarex/plantdisease
- spMohanty, PlantVillage-Dataset (GitHub): https://github.com/spMohanty/PlantVillage-Dataset
