"""
evaluate_per_category.py - Akurasi per Kategori Objek CIFAKE
=============================================================
CIFAKE berisi 10 kategori objek CIFAR-10. Kategori tidak dipakai saat
training (label hanya REAL/FAKE), tetapi bisa dibaca dari nama file:
"xxxx.jpg" = kategori 1, "xxxx (k).jpg" = kategori k (k = 2..10).

Script ini menghitung akurasi deteksi REAL/FAKE per kategori untuk
setiap model pada test set, untuk melihat kategori mana yang paling sulit.

Output: results/per_category_accuracy.csv/.json + plot.
"""

import json
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

import config
from data_loader import get_dataloaders
from evaluate import evaluate_model
from models import get_model

CATEGORIES = ["airplane", "automobile", "bird", "cat", "deer",
              "dog", "frog", "horse", "ship", "truck"]
CATEGORIES_ID = ["pesawat", "mobil", "burung", "kucing", "rusa",
                 "anjing", "katak", "kuda", "kapal", "truk"]


def category_of(path: str) -> int:
    """Indeks kategori 0-9 dari nama file CIFAKE."""
    m = re.search(r"\((\d+)\)\.jpg$", os.path.basename(path))
    return int(m.group(1)) - 1 if m else 0


def per_category_accuracy():
    device = config.DEVICE
    rows = []

    for model_name in config.ALL_MODELS:
        model_path = os.path.join(config.MODEL_DIR, f"{model_name}_best.pt")
        if not os.path.exists(model_path):
            print(f"[SKIP] {model_path} tidak ditemukan")
            continue

        model = get_model(model_name)
        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
        model.eval()

        _, _, test_loader = get_dataloaders(model_name)   # shuffle=False, urutan = samples
        result = evaluate_model(model, test_loader, device)
        cats = np.array([category_of(p) for p, _ in test_loader.dataset.samples])
        y_true, y_pred = result["y_true"], result["y_pred"]

        for c in range(len(CATEGORIES)):
            mask = cats == c
            real = mask & (y_true == 1)
            fake = mask & (y_true == 0)
            rows.append({
                "Model": model_name,
                "Kategori": CATEGORIES_ID[c],
                "Category": CATEGORIES[c],
                "Accuracy": float((y_pred[mask] == y_true[mask]).mean()),
                "Recall_REAL": float((y_pred[real] == 1).mean()),
                "Recall_FAKE": float((y_pred[fake] == 0).mean()),
                "N": int(mask.sum()),
            })
        print(f"[OK] {model_name}")

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(config.RESULT_DIR, "per_category_accuracy.csv"), index=False)
    with open(os.path.join(config.RESULT_DIR, "per_category_accuracy.json"), "w") as f:
        json.dump(rows, f, indent=2)

    pivot = df.pivot(index="Kategori", columns="Model", values="Accuracy").reindex(CATEGORIES_ID)
    print("\nAkurasi per kategori (%):")
    print((pivot * 100).round(2).to_string())

    # Plot grouped bar
    fig, ax = plt.subplots(figsize=(14, 6))
    colors = {"LightweightCNN": "#2196F3", "ResNet50": "#4CAF50", "EfficientNetV2B0": "#FF9800"}
    width = 0.8 / len(pivot.columns)
    x = np.arange(len(pivot.index))
    for i, m in enumerate(pivot.columns):
        ax.bar(x + i * width - 0.4 + width / 2, pivot[m] * 100, width, label=m, color=colors.get(m))
    ax.set_xticks(x)
    ax.set_xticklabels(pivot.index)
    ax.set_ylabel("Accuracy (%)")
    ax.set_ylim(max(0, pivot.values.min() * 100 - 3), 100.5)
    ax.set_title("Akurasi Deteksi REAL/FAKE per Kategori Objek (test set)", fontweight="bold")
    ax.grid(axis="y", alpha=0.3)
    ax.legend()
    plt.tight_layout()
    path = os.path.join(config.RESULT_DIR, "plots", "per_category_accuracy.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[Plot] {path}")
    return df


if __name__ == "__main__":
    per_category_accuracy()
