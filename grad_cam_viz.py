"""
grad_cam_viz.py - Grad-CAM Visualization untuk CIFAKE Detection
================================================================
Menggunakan library pytorch-grad-cam untuk menghasilkan heatmap
yang menunjukkan area citra mana yang paling berpengaruh terhadap
keputusan klasifikasi model.

Fitur:
- Grad-CAM heatmap untuk setiap model
- Visualisasi side-by-side: Original | Heatmap | Overlay
- Menyimpan contoh untuk citra REAL dan FAKE
"""

import os
import random

import torch
import numpy as np
from PIL import Image

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

import config
from models import get_model, get_target_layer
from data_loader import get_transforms

from torchvision import datasets


# ============================================================
# GRAD-CAM VISUALIZATION
# ============================================================

def generate_gradcam(model_name: str, num_samples: int = 5, seed: int = 42):
    """
    Generate Grad-CAM visualization untuk model tertentu.
    
    Args:
        model_name: Nama model
        num_samples: Jumlah contoh per kelas (FAKE & REAL)
        seed: Random seed untuk pemilihan sample
    
    Returns:
        List path ke gambar yang disimpan
    """
    print(f"{'='*60}")
    print(f"  GRAD-CAM Visualization: {model_name}")
    print(f"{'='*60}")
    
    device = config.DEVICE
    save_dir = os.path.join(config.RESULT_DIR, "gradcam", model_name)
    os.makedirs(save_dir, exist_ok=True)
    
    # ---- 1. Load Model ----
    model_path = os.path.join(config.MODEL_DIR, f"{model_name}_best.pt")
    if not os.path.exists(model_path):
        print(f"[ERROR] Model checkpoint tidak ditemukan: {model_path}")
        return []
    
    model = get_model(model_name)
    model.load_state_dict(
        torch.load(model_path, map_location=device, weights_only=True)
    )
    model.eval()
    print(f"[Model] Loaded: {model_path}")
    
    # ---- 2. Setup Grad-CAM ----
    target_layer = get_target_layer(model, model_name)
    
    # Wrap target_layer in a list
    cam = GradCAM(model=model, target_layers=[target_layer])
    
    # ---- 3. Load Test Images ----
    test_dataset = datasets.ImageFolder(config.TEST_DIR)
    eval_transform = get_transforms(model_name, is_training=False)
    
    # Pilih sample acak per kelas
    random.seed(seed)
    class_indices = {}
    for idx, (_, label) in enumerate(test_dataset.samples):
        if label not in class_indices:
            class_indices[label] = []
        class_indices[label].append(idx)
    
    saved_paths = []
    
    for class_idx, class_name in enumerate(config.CLASS_NAMES):
        print(f"\n  Kelas: {class_name}")
        
        indices = class_indices.get(class_idx, [])
        if not indices:
            print(f"    [WARNING] Tidak ada sampel untuk kelas {class_name}")
            continue
        
        sample_indices = random.sample(indices, min(num_samples, len(indices)))
        
        for i, data_idx in enumerate(sample_indices):
            img_path, label = test_dataset.samples[data_idx]
            
            # Load gambar asli
            original_img = Image.open(img_path).convert("RGB")
            
            # Resize untuk model (untuk visualisasi)
            if model_name == config.MODEL_LIGHTWEIGHT:
                target_size = config.IMG_SIZE_LIGHTWEIGHT
            else:
                target_size = config.IMG_SIZE_TRANSFER
            
            resized_img = original_img.resize(
                (target_size, target_size), Image.BILINEAR
            )
            
            # Numpy array untuk overlay (0-1 range)
            img_np = np.array(resized_img).astype(np.float32) / 255.0
            
            # Transform untuk model input
            input_tensor = eval_transform(original_img).unsqueeze(0).to(device)
            
            # ---- Generate Grad-CAM ----
            # Target: predicted class
            with torch.no_grad():
                output = model(input_tensor)
                pred_class = output.argmax(dim=1).item()
            
            targets = [ClassifierOutputTarget(pred_class)]
            
            # Generate CAM
            grayscale_cam = cam(input_tensor=input_tensor, targets=targets)
            grayscale_cam = grayscale_cam[0, :]  # (H, W)
            
            # Overlay heatmap pada gambar
            cam_image = show_cam_on_image(img_np, grayscale_cam, use_rgb=True)
            
            # ---- Visualisasi ----
            fig, axes = plt.subplots(1, 3, figsize=(12, 4))
            
            # Original
            axes[0].imshow(img_np)
            axes[0].set_title(f"Original\nTrue: {class_name}", fontsize=11)
            axes[0].axis("off")
            
            # Heatmap
            axes[1].imshow(grayscale_cam, cmap="jet")
            axes[1].set_title("Grad-CAM Heatmap", fontsize=11)
            axes[1].axis("off")
            
            # Overlay
            axes[2].imshow(cam_image)
            pred_name = config.CLASS_NAMES[pred_class]
            color = "green" if pred_class == label else "red"
            axes[2].set_title(
                f"Overlay\nPred: {pred_name}",
                fontsize=11, color=color
            )
            axes[2].axis("off")
            
            fig.suptitle(
                f"{model_name} - Grad-CAM ({class_name} #{i+1})",
                fontsize=13, fontweight="bold"
            )
            
            plt.tight_layout()
            
            fname = f"gradcam_{class_name}_{i+1}.png"
            fpath = os.path.join(save_dir, fname)
            fig.savefig(fpath, dpi=150, bbox_inches="tight")
            plt.close(fig)
            
            saved_paths.append(fpath)
            print(f"    [{i+1}/{num_samples}] True: {class_name}, "
                  f"Pred: {pred_name} → {fname}")
    
    print(f"\n[Save] {len(saved_paths)} gambar Grad-CAM disimpan ke: {save_dir}")
    
    # ---- Grid Visualization ----
    if saved_paths:
        create_gradcam_grid(model_name, save_dir, num_samples)
    
    return saved_paths


def create_gradcam_grid(model_name: str, save_dir: str, num_per_class: int):
    """
    Membuat grid visualization dari semua Grad-CAM samples.
    """
    images = []
    titles = []
    
    for class_name in config.CLASS_NAMES:
        for i in range(1, num_per_class + 1):
            fpath = os.path.join(save_dir, f"gradcam_{class_name}_{i}.png")
            if os.path.exists(fpath):
                img = Image.open(fpath)
                images.append(img)
                titles.append(f"{class_name} #{i}")
    
    if not images:
        return
    
    n_cols = min(num_per_class, 5)
    n_rows = len(config.CLASS_NAMES)
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
    if n_rows == 1:
        axes = [axes]
    
    idx = 0
    for row in range(n_rows):
        for col in range(n_cols):
            if idx < len(images):
                if n_cols == 1:
                    ax = axes[row]
                else:
                    ax = axes[row][col]
                ax.imshow(images[idx])
                ax.axis("off")
                idx += 1
    
    fig.suptitle(f"Grad-CAM Grid - {model_name}",
                 fontsize=16, fontweight="bold")
    plt.tight_layout()
    
    grid_path = os.path.join(config.RESULT_DIR, "plots",
                             f"{model_name}_gradcam_grid.png")
    fig.savefig(grid_path, dpi=100, bbox_inches="tight")
    plt.close(fig)
    
    print(f"[Plot] Grad-CAM grid: {grid_path}")


# ============================================================
# CLI ENTRY POINT
# ============================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Grad-CAM Visualization untuk CIFAKE Detection"
    )
    parser.add_argument(
        "--model", "-m",
        type=str,
        choices=config.ALL_MODELS,
        default=None,
        help=f"Model untuk Grad-CAM. Default: semua model."
    )
    parser.add_argument(
        "--num-samples", "-n",
        type=int, default=5,
        help="Jumlah contoh per kelas (default: 5)"
    )
    
    args = parser.parse_args()
    
    if args.model:
        models_to_viz = [args.model]
    else:
        models_to_viz = config.ALL_MODELS
    
    for model_name in models_to_viz:
        generate_gradcam(model_name, num_samples=args.num_samples)
