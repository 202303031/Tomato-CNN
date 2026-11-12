"""Gradio web app: upload a tomato leaf image -> disease prediction, class probabilities, Grad-CAM.

Run `python app.py` (loads models/best_model.pth once; no retraining).
"""
from __future__ import annotations

import argparse

import gradio as gr
import pandas as pd
from PIL import Image

from src.disease_info import DISCLAIMER
from src.predict import Predictor
from src.utils import PROJECT_ROOT

PREDICTED = "Predicted class (default)"
EXAMPLES_DIR = PROJECT_ROOT / "examples"

predictor = Predictor()
DISPLAY_NAMES = [predictor.display[c] for c in predictor.classes]
TARGET_CHOICES = [PREDICTED] + DISPLAY_NAMES


def analyze(image: Image.Image | None, target_choice: str):
    if image is None:
        raise gr.Error("Please upload a tomato leaf image first.")
    image = image.convert("RGB")
    target = None if target_choice == PREDICTED else target_choice
    pred, res = predictor.predict_and_explain(image, target)
    explained = predictor.display[res["target_class"]]
    blank_note = ("\n\n**Note:** the Grad-CAM map for this class is all zeros: no region of the image increases "
                  "the score of the selected class, so the heatmap is blank.") if res["sanity"]["blank"] else ""


    probs_df = pd.DataFrame({"class": DISPLAY_NAMES, "probability": [pred["probabilities"][n] for n in DISPLAY_NAMES]})
    info_md = (
        f"## Prediction: **{pred['pred_display']}**\n"
        f"Confidence: **{pred['confidence']:.1%}**\n\n"
        f"**About this class:** {pred['description']}\n\n"
        f"**Grad-CAM target:** {explained}"
        + ("" if target is None else f" (user-selected; the model predicted {pred['pred_display']})")
        + ". Warm regions increased the score of that class at the last convolutional layer (layer4). "
          "This is a coarse attribution map, not a lesion segmentation."
        + blank_note + f"\n\n---\n{DISCLAIMER}"
    )
    return pred["probabilities"], probs_df, res["heatmap"], res["overlay"], info_md


def build_demo() -> gr.Blocks:
    examples = sorted(str(p) for p in EXAMPLES_DIR.glob("*.JPG")) + sorted(str(p) for p in EXAMPLES_DIR.glob("*.jpg"))
    with gr.Blocks(title="Tomato Leaf Disease Detection") as demo:
        gr.Markdown(
            "# 🍅 Tomato Leaf Disease Detection\n"
            "ResNet18 CNN feature extractor + MLP classification head, trained on the PlantVillage tomato classes. "
            f"Supported classes: {len(DISPLAY_NAMES)}. Explanations via Grad-CAM on the final convolutional layer."
        )
        with gr.Row():
            with gr.Column(scale=1):
                image_in = gr.Image(type="pil", label="Upload a tomato leaf image", sources=["upload", "clipboard"], height=320)
                target = gr.Dropdown(choices=TARGET_CHOICES, value=PREDICTED, label="Grad-CAM target class")
                with gr.Row():
                    run_btn = gr.Button("Run prediction", variant="primary")
                    clear_btn = gr.ClearButton(value="Clear / reset")
                if examples:
                    gr.Examples(examples=examples, inputs=image_in, label="Held-out test images (never used in training)")
            with gr.Column(scale=1):
                label_out = gr.Label(num_top_classes=len(DISPLAY_NAMES), label="Predicted disease and confidence")
                bar_out = gr.BarPlot(x="class", y="probability", y_lim=[0, 1], x_label_angle=-40, height=320,
                                     title="Probability for every class", label="Class probabilities")
        with gr.Row():
            heat_out = gr.Image(label="Grad-CAM heatmap", interactive=False, height=300)
            overlay_out = gr.Image(label="Grad-CAM overlay", interactive=False, height=300)
        info_out = gr.Markdown()

        outputs = [label_out, bar_out, heat_out, overlay_out, info_out]
        # Grad-CAM attaches hooks to the single shared model, so requests must not run concurrently.
        run_btn.click(analyze, inputs=[image_in, target], outputs=outputs, concurrency_limit=1)
        clear_btn.add([image_in, *outputs])
        clear_btn.click(lambda: PREDICTED, outputs=target)
    return demo


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true")
    args = parser.parse_args()
    build_demo().launch(server_name="127.0.0.1", server_port=args.port, share=args.share, inbrowser=False)
