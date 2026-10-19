import os, glob, json
import cv2
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
import gradio as gr

# Automatically run on local CPU or local GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMG_SIZE = 64

# Load class names and ablation CSV
with open("class_names.json", "r") as f:
    class_names = json.load(f)

df_results = pd.read_csv("cnn_syllabus_ablation_results.csv") if os.path.exists("cnn_syllabus_ablation_results.csv") else pd.DataFrame()

transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# =====================================================================
# MODEL DEFINITIONS (Must match training architecture)
# =====================================================================
class RBFLayer(nn.Module):
    def __init__(self, in_features, num_centers=32):
        super().__init__()
        self.in_features = in_features
        self.num_centers = num_centers
        self.centers = nn.Parameter(torch.randn(num_centers, in_features) * 0.5)
        self.log_gamma = nn.Parameter(torch.zeros(num_centers))

    def forward(self, x):
        x_norm = F.normalize(x, p=2, dim=1)
        dist_sq = torch.cdist(x_norm, self.centers, p=2) ** 2
        gamma = torch.exp(self.log_gamma).unsqueeze(0)
        return torch.exp(-gamma * dist_sq)

class ConfigurableCNN(nn.Module):
    def __init__(self, num_classes=5, num_conv_layers=3, base_filters=32,
                 kernel_size=3, pooling_type="max", activation="relu", head_type="mlp"):
        super().__init__()
        act_map = {"relu": nn.ReLU(), "leaky_relu": nn.LeakyReLU(0.1), "tanh": nn.Tanh()}
        self.act = act_map[activation]
        
        layers = []
        in_ch = 3
        curr_filters = base_filters
        padding = kernel_size // 2
        
        for _ in range(num_conv_layers):
            if pooling_type == "strided":
                layers.append(nn.Conv2d(in_ch, curr_filters, kernel_size=kernel_size, stride=2, padding=padding))
                layers.append(nn.BatchNorm2d(curr_filters))
                layers.append(self.act)
            else:
                layers.append(nn.Conv2d(in_ch, curr_filters, kernel_size=kernel_size, stride=1, padding=padding))
                layers.append(nn.BatchNorm2d(curr_filters))
                layers.append(self.act)
                if pooling_type == "max":
                    layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
                elif pooling_type == "avg":
                    layers.append(nn.AvgPool2d(kernel_size=2, stride=2))
            in_ch = curr_filters
            curr_filters *= 2
            
        self.features = nn.Sequential(*layers)
        with torch.no_grad():
            dummy = torch.zeros(1, 3, IMG_SIZE, IMG_SIZE)
            feat_dim = self.features(dummy).view(1, -1).shape[1]
            
        if head_type == "perceptron":
            self.classifier = nn.Linear(feat_dim, num_classes)
        elif head_type == "mlp":
            self.classifier = nn.Sequential(
                nn.Linear(feat_dim, 128),
                self.act,
                nn.Dropout(0.3),
                nn.Linear(128, num_classes)
            )
        elif head_type == "rbf":
            self.classifier = nn.Sequential(
                nn.Linear(feat_dim, 64),
                RBFLayer(in_features=64, num_centers=32),
                nn.Linear(32, num_classes)
            )

    def forward(self, x):
        f = self.features(x)
        flat = torch.flatten(f, 1)
        return self.classifier(flat)

# Load all 3 saved .pth weights onto local CPU/GPU using map_location
SAVED_MODELS = {}
for head_type in ["perceptron", "mlp", "rbf"]:
    m = ConfigurableCNN(num_classes=len(class_names), num_conv_layers=3, base_filters=32,
                        kernel_size=3, pooling_type="max", activation="relu", head_type=head_type).to(device)
    weight_path = f"tomato_cnn_{head_type}.pth"
    if os.path.exists(weight_path):
        m.load_state_dict(torch.load(weight_path, map_location=device, weights_only=True))
    m.eval()
    SAVED_MODELS[head_type] = m

# =====================================================================
# VISUALIZATION & GRAD-CAM FUNCTIONS
# =====================================================================
def generate_gradcam_and_bbox(model, img_tensor, orig_pil):
    model.eval()
    activations, gradients = [], []
    target_layer = None
    for module in model.features.modules():
        if isinstance(module, nn.Conv2d):
            target_layer = module

    h1 = target_layer.register_forward_hook(lambda m, i, o: activations.append(o))
    h2 = target_layer.register_full_backward_hook(lambda m, gi, go: gradients.append(go[0]))

    input_t = img_tensor.unsqueeze(0).to(device)
    input_t.requires_grad = True
    logits = model(input_t)
    probs = F.softmax(logits, dim=1).detach().cpu().numpy()[0]
    pred_idx = int(logits.argmax(dim=1).item())

    model.zero_grad()
    logits[0, pred_idx].backward()
    h1.remove()
    h2.remove()

    grads = gradients[0].detach().cpu()[0]
    fmaps = activations[0].detach().cpu()[0]
    weights = torch.mean(grads, dim=(1, 2))

    cam = torch.zeros(fmaps.shape[1:], dtype=torch.float32)
    for i, w in enumerate(weights):
        cam += w * fmaps[i]

    cam = F.relu(cam).numpy()
    cam = cam - np.min(cam)
    if np.max(cam) > 0:
        cam = cam / np.max(cam)

    orig_np = np.array(orig_pil.resize((256, 256)))
    cam_resized = cv2.resize(cam, (256, 256))

    heatmap_color = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)
    overlay_img = cv2.addWeighted(orig_np, 0.55, heatmap_color, 0.45, 0)

    bbox_img = orig_np.copy()
    thresh_mask = np.uint8(cam_resized > 0.60) * 255
    contours, _ = cv2.findContours(thresh_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        if cv2.contourArea(cnt) > 150:
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(bbox_img, (x, y), (x + w, y + h), (255, 50, 50), 2)
            cv2.putText(bbox_img, f"{class_names[pred_idx][:14]}", (x, max(15, y - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 0), 1, cv2.LINE_AA)

    return overlay_img, bbox_img, probs, pred_idx

def visualize_pooling_and_weights(model, img_tensor):
    model.eval()
    first_conv = model.features[0]
    with torch.no_grad():
        x = img_tensor.unsqueeze(0).to(device)
        conv_out = F.relu(first_conv(x))
        max_pooled = F.max_pool2d(conv_out, kernel_size=2, stride=2)
        avg_pooled = F.avg_pool2d(conv_out, kernel_size=2, stride=2)
        weights = first_conv.weight.detach().cpu().numpy()

    fig, axes = plt.subplots(3, 4, figsize=(12, 8))
    for i in range(4):
        axes[0, i].imshow(conv_out[0, i].cpu().numpy(), cmap="viridis")
        axes[0, i].set_title(f"Filter #{i+1}: Conv Output (64x64)", fontsize=9)
        axes[0, i].axis("off")

        axes[1, i].imshow(max_pooled[0, i].cpu().numpy(), cmap="magma")
        axes[1, i].set_title(f"Filter #{i+1}: Max Pool (32x32)", fontsize=9)
        axes[1, i].axis("off")

        axes[2, i].imshow(avg_pooled[0, i].cpu().numpy(), cmap="plasma")
        axes[2, i].set_title(f"Filter #{i+1}: Avg Pool (32x32)", fontsize=9)
        axes[2, i].axis("off")

    plt.suptitle("Feature Extraction (Conv1) vs. Max Pooling vs. Average Pooling", fontweight="bold")
    plt.tight_layout()

    fig_w, axes_w = plt.subplots(2, 4, figsize=(10, 4.5))
    for i in range(8):
        r, c = i // 4, i % 4
        k_w = weights[i].transpose(1, 2, 0)
        k_w = (k_w - k_w.min()) / (k_w.max() - k_w.min() + 1e-8)
        axes_w[r, c].imshow(k_w)
        axes_w[r, c].set_title(f"Kernel #{i+1} Weights\nMean={weights[i].mean():.3f}", fontsize=8)
        axes_w[r, c].axis("off")
    plt.suptitle("Learned 3x3 RGB Convolutional Filter Weights", fontweight="bold")
    plt.tight_layout()
    return fig, fig_w

def visualize_rbf_activations(rbf_model, img_tensor):
    rbf_model.eval()
    with torch.no_grad():
        x = img_tensor.unsqueeze(0).to(device)
        feat = torch.flatten(rbf_model.features(x), 1)
        proj = rbf_model.classifier[0](feat)
        rbf_layer = rbf_model.classifier[1]
        x_norm = F.normalize(proj, p=2, dim=1)
        dists = torch.cdist(x_norm, rbf_layer.centers, p=2).cpu().numpy()[0]
        rbf_acts = rbf_layer(proj).cpu().numpy()[0]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    ax1.bar(range(1, len(dists)+1), dists, color="#e74c3c", alpha=0.85)
    ax1.set_title("Euclidean Distance ||x - c_k|| to 32 RBF Centers", fontsize=10, fontweight="bold")
    ax1.set_xlabel("RBF Prototype Center Index (k = 1..32)")
    ax1.set_ylabel("Euclidean Distance")

    ax2.bar(range(1, len(rbf_acts)+1), rbf_acts, color="#2ecc71", alpha=0.85)
    ax2.set_title("Gaussian RBF Node Activations exp(-gamma * ||x - c_k||^2)", fontsize=10, fontweight="bold")
    ax2.set_xlabel("RBF Structural Node Index (k = 1..32)")
    ax2.set_ylabel("Gaussian Activation Strength")
    plt.tight_layout()
    return fig

def run_full_pipeline_ui(input_pil, selected_head, bam_noise_pct):
    if input_pil is None:
        return None, None, {}, None, None, None, None, "Please upload or select a leaf image."

    img_tensor = transform(input_pil.convert("RGB"))
    model = SAVED_MODELS[selected_head]

    heatmap_img, bbox_img, probs, pred_idx = generate_gradcam_and_bbox(model, img_tensor, input_pil.convert("RGB"))
    prob_dict = {class_names[i]: float(probs[i]) for i in range(len(class_names))}

    fig_pool, fig_weights = visualize_pooling_and_weights(model, img_tensor)
    fig_rbf = visualize_rbf_activations(SAVED_MODELS["rbf"], img_tensor)

    gray_16 = np.array(input_pil.convert("L").resize((16, 16)))
    clean_bipolar = np.where(gray_16 > np.median(gray_16), 1, -1).flatten()
    W_bam = np.outer(clean_bipolar, np.array([1, -1, 1, -1, 1, -1, 1, -1]))
    noisy_bipolar = clean_bipolar.copy()
    n_flip = int((bam_noise_pct / 100.0) * 256)
    if n_flip > 0:
        flip_idx = np.random.choice(256, size=n_flip, replace=False)
        noisy_bipolar[flip_idx] *= -1
    y_rec = np.where((noisy_bipolar @ W_bam) >= 0, 1, -1)
    x_rec = np.where((y_rec @ W_bam.T) >= 0, 1, -1)

    fig_bam, ax_b = plt.subplots(1, 3, figsize=(8, 2.8))
    ax_b[0].imshow(clean_bipolar.reshape(16, 16), cmap="gray")
    ax_b[0].set_title("16x16 Bipolar Leaf")
    ax_b[1].imshow(noisy_bipolar.reshape(16, 16), cmap="gray")
    ax_b[1].set_title(f"Noisy Input ({bam_noise_pct}% Flips)")
    ax_b[2].imshow(x_rec.reshape(16, 16), cmap="gray")
    ax_b[2].set_title("BAM Reconstructed")
    for a in ax_b: a.axis("off")
    plt.suptitle("Bidirectional Associative Memory (BAM) Live Pattern Recall", fontweight="bold")
    plt.tight_layout()

    summary_md = f"""
    ### Diagnosis Result: **{class_names[pred_idx]}** ({probs[pred_idx]*100:.2f}% Confidence)
    * **Active Classification Head:** `{selected_head.upper()}` (Running 100% Offline on `{device}`)
    * **Optimal Hyperparameter Weights Found:** `3 Conv Layers (32 -> 64 -> 128 filters)` | `3x3 Kernel` | `Max Pooling (2x2, stride 2)` | `ReLU Activation` | `Learning Rate = 0.001 (Adam)` | `Batch Size = 32`
    """
    return bbox_img, heatmap_img, prob_dict, fig_pool, fig_weights, fig_rbf, fig_bam, summary_md

unseen_sample_paths = glob.glob(os.path.join("tomato_unseen", "*", "*.*"))[:10]

with gr.Blocks(theme=gr.themes.Soft(), title="Tomato CNN Offline Inspector") as demo:
    gr.Markdown(
        """
        # 🍅 Tomato Leaf Disease Detection & Neural Network Inspector (Offline Local Edition)
        Upload any unseen tomato leaf image or click a bundled sample below to run **Disease Detection (Bounding Box + Grad-CAM Heatmap)** and inspect **Learned Conv Weights**, **Max vs. Average Pooling**, **RBF Network Prototype Centers**, and **BAM Associative Memory**.
        """
    )
    with gr.Row():
        with gr.Column(scale=1):
            img_input = gr.Image(type="pil", label="Upload Unseen Tomato Leaf Image")
            head_selector = gr.Radio(choices=["mlp", "rbf", "perceptron"], value="mlp", label="Select Syllabus Classification Head")
            noise_slider = gr.Slider(0, 45, value=20, step=5, label="BAM Bit-Flip Noise (%)")
            run_btn = gr.Button("🔍 Run Full Detection & Syllabus Pipeline", variant="primary")
            if unseen_sample_paths:
                gr.Examples(examples=unseen_sample_paths, inputs=img_input, label="Bundled Unseen Test Images (Offline):")

        with gr.Column(scale=2):
            summary_box = gr.Markdown("### Upload or click an image and press Run.")
            with gr.Row():
                bbox_output = gr.Image(label="1. Infected Region Bounding Box Detection")
                heatmap_output = gr.Image(label="2. Grad-CAM Disease Activation Heatmap")
            prob_output = gr.Label(num_top_classes=5, label="3. Disease Class Confidence Scores")

    with gr.Tabs():
        with gr.TabItem("Max vs. Avg Pooling & Feature Maps"):
            pool_plot = gr.Plot(label="Comparison of Conv Output, Max Pooling, and Average Pooling")
        with gr.TabItem("Learned 3x3 Kernel Weights"):
            weights_plot = gr.Plot(label="Learned RGB Filter Weights from Conv Layer 1")
        with gr.TabItem("RBF Network Prototype Centers"):
            rbf_plot = gr.Plot(label="Euclidean Distance & Gaussian RBF Activations")
        with gr.TabItem("BAM Associative Memory Recall"):
            bam_plot = gr.Plot(label="Bidirectional Associative Memory Denoising")
        with gr.TabItem("Full Hyperparameter Ablation Table"):
            gr.Dataframe(value=df_results, label="Experimental Comparison Across All 8 Configurations")

    run_btn.click(
        fn=run_full_pipeline_ui,
        inputs=[img_input, head_selector, noise_slider],
        outputs=[bbox_output, heatmap_output, prob_output, pool_plot, weights_plot, fig_rbf if False else rbf_plot, bam_plot, summary_box]
    )

if __name__ == "__main__":
    # share=False and inbrowser=True ensures it runs 100% offline on localhost:7860
    demo.launch(server_name="127.0.0.1", server_port=7860, share=False, inbrowser=True)
