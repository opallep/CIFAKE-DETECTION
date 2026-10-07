"""
test_compression_robustness.py - Uji Ketahanan Model terhadap Kompresi JPEG
=============================================================================
Script ini menguji ketiga model pada citra test yang sudah dikompresi
JPEG dengan berbagai level kualitas (100, 80, 60, 40) untuk mengukur
seberapa tahan setiap model terhadap degradasi kualitas citra.

Hasil berupa:
- Tabel akurasi per model per level kompresi
- Plot kurva accuracy vs JPEG quality
"""

import os
import json

import torch
import numpy as np
import pandas as pd
from tabulate import tabulate
from tqdm import tqdm

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import config
from models import get_model
from data_loader import get_compressed_test_loader
from evaluate import evaluate_model


def test_compression_robustness():
    """
    Menguji ketahanan ketiga model terhadap kompresi JPEG.
    
    Untuk setiap model dan setiap level JPEG quality:
    1. Kompres test images secara on-the-fly
    2. Evaluasi model pada citra terkompresi
    3. Kumpulkan metrik (akurasi, F1)
    """
    print("=" * 70)
    print("  UJI KETAHANAN MODEL TERHADAP KOMPRESI JPEG")
    print("=" * 70)
    print()
    
    device = config.DEVICE
    qualities = config.JPEG_QUALITIES
    
    # Struktur hasil: {model_name: {quality: metrics}}
    all_results = {}
    
    for model_name in config.ALL_MODELS:
        print(f"\n{'─'*50}")
        print(f"  Model: {model_name}")
        print(f"{'─'*50}")
        
        # Load model
        model_path = os.path.join(config.MODEL_DIR, f"{model_name}_best.pt")
        if not os.path.exists(model_path):
            print(f"  [SKIP] Checkpoint tidak ditemukan: {model_path}")
            continue
        
        model = get_model(model_name)
        model.load_state_dict(
            torch.load(model_path, map_location=device, weights_only=True)
        )
        model.eval()
        
        model_results = {}
        
        for quality in qualities:
            print(f"  JPEG Quality {quality}...", end=" ", flush=True)
            
            # Buat DataLoader dengan kompresi JPEG
            test_loader = get_compressed_test_loader(model_name, quality)
            
            # Evaluasi
            eval_result = evaluate_model(model, test_loader, device)
            metrics = eval_result["metrics"]
            
            model_results[quality] = {
                "accuracy": metrics["accuracy"],
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "f1_score": metrics["f1_score"],
                "auc_roc": metrics["auc_roc"],
            }
            
            print(f"Acc: {metrics['accuracy']:.4%} | F1: {metrics['f1_score']:.4%}")
        
        all_results[model_name] = model_results
    
    if not all_results:
        print("[ERROR] Tidak ada model yang bisa diuji.")
        return None
    
    # ---- Build Comparison Table ----
    print_compression_table(all_results, qualities)
    
    # ---- Plot Results ----
    plot_compression_curves(all_results, qualities)
    
    # ---- Save Results ----
    save_compression_results(all_results, qualities)
    
    return all_results


def print_compression_table(all_results: dict, qualities: list):
    """Print tabel ringkasan hasil uji kompresi."""
    print("\n" + "=" * 80)
    print("  TABEL KETAHANAN KOMPRESI JPEG")
    print("=" * 80)
    
    # Tabel Akurasi
    print("\n  ► Accuracy (%)")
    rows = []
    for model_name, results in all_results.items():
        row = {"Model": model_name}
        for q in qualities:
            row[f"Q={q}"] = f"{results[q]['accuracy'] * 100:.2f}"
        # Hitung penurunan dari Q=100 ke Q=40
        if 100 in results and 40 in results:
            drop = (results[100]["accuracy"] - results[40]["accuracy"]) * 100
            row["Drop (Q100→Q40)"] = f"{drop:+.2f}"
        rows.append(row)
    
    print(tabulate(rows, headers="keys", tablefmt="grid",
                   showindex=False, numalign="right"))
    
    # Tabel F1-Score
    print("\n  ► F1-Score (%)")
    rows = []
    for model_name, results in all_results.items():
        row = {"Model": model_name}
        for q in qualities:
            row[f"Q={q}"] = f"{results[q]['f1_score'] * 100:.2f}"
        if 100 in results and 40 in results:
            drop = (results[100]["f1_score"] - results[40]["f1_score"]) * 100
            row["Drop (Q100→Q40)"] = f"{drop:+.2f}"
        rows.append(row)
    
    print(tabulate(rows, headers="keys", tablefmt="grid",
                   showindex=False, numalign="right"))


def plot_compression_curves(all_results: dict, qualities: list):
    """Plot kurva accuracy & F1 vs JPEG quality."""
    save_dir = os.path.join(config.RESULT_DIR, "plots")
    os.makedirs(save_dir, exist_ok=True)
    
    colors = {"LightweightCNN": "#2196F3", "ResNet50": "#4CAF50",
              "EfficientNetV2B0": "#FF9800"}
    markers = {"LightweightCNN": "o", "ResNet50": "s", "EfficientNetV2B0": "D"}
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    for model_name, results in all_results.items():
        qs = sorted(results.keys(), reverse=True)
        accs = [results[q]["accuracy"] * 100 for q in qs]
        f1s = [results[q]["f1_score"] * 100 for q in qs]
        
        color = colors.get(model_name, "#666666")
        marker = markers.get(model_name, "^")
        
        axes[0].plot(qs, accs, f"-{marker}", color=color, label=model_name,
                     linewidth=2, markersize=8)
        axes[1].plot(qs, f1s, f"-{marker}", color=color, label=model_name,
                     linewidth=2, markersize=8)
    
    axes[0].set_xlabel("JPEG Quality", fontsize=12)
    axes[0].set_ylabel("Accuracy (%)", fontsize=12)
    axes[0].set_title("Accuracy vs JPEG Quality", fontsize=13, fontweight="bold")
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)
    axes[0].invert_xaxis()  # Q tinggi di kiri
    
    axes[1].set_xlabel("JPEG Quality", fontsize=12)
    axes[1].set_ylabel("F1-Score (%)", fontsize=12)
    axes[1].set_title("F1-Score vs JPEG Quality", fontsize=13, fontweight="bold")
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3)
    axes[1].invert_xaxis()
    
    plt.suptitle("Ketahanan Model terhadap Kompresi JPEG",
                 fontsize=15, fontweight="bold", y=1.02)
    plt.tight_layout()
    
    save_path = os.path.join(save_dir, "compression_robustness.png")
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    
    print(f"\n[Plot] Compression robustness: {save_path}")


def save_compression_results(all_results: dict, qualities: list):
    """Simpan hasil ke JSON dan CSV."""
    # JSON
    json_path = os.path.join(config.RESULT_DIR, "compression_robustness.json")
    
    # Convert keys to strings for JSON serialization
    serializable = {}
    for model_name, results in all_results.items():
        serializable[model_name] = {str(k): v for k, v in results.items()}
    
    with open(json_path, "w") as f:
        json.dump(serializable, f, indent=2)
    print(f"[Save] JSON: {json_path}")
    
    # CSV
    rows = []
    for model_name, results in all_results.items():
        for q in qualities:
            row = {
                "Model": model_name,
                "JPEG_Quality": q,
                "Accuracy": results[q]["accuracy"],
                "Precision": results[q]["precision"],
                "Recall": results[q]["recall"],
                "F1_Score": results[q]["f1_score"],
                "AUC_ROC": results[q]["auc_roc"],
            }
            rows.append(row)
    
    df = pd.DataFrame(rows)
    csv_path = os.path.join(config.RESULT_DIR, "compression_robustness.csv")
    df.to_csv(csv_path, index=False)
    print(f"[Save] CSV: {csv_path}")


# ============================================================
# CLI ENTRY POINT
# ============================================================

if __name__ == "__main__":
    test_compression_robustness()
