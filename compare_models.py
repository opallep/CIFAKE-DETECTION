"""
compare_models.py - Perbandingan Ketiga Model CIFAKE Detection
==============================================================
Script ini menjalankan ketiga model pada test set yang identik
dan menghasilkan tabel perbandingan akhir meliputi:
1. Akurasi & F1-Score
2. Waktu inferensi per citra
3. Jumlah parameter (total & trainable)
4. AUC-ROC

Hasil disimpan dalam format tabel (console + CSV).
"""

import os
import time
import json

import torch
import numpy as np
import pandas as pd
from tabulate import tabulate
from tqdm import tqdm
from torch.amp import autocast

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import config
from models import get_model, count_parameters
from data_loader import get_dataloaders
from evaluate import evaluate_model


def measure_inference_time(model, dataloader, device, num_warmup=10):
    """
    Mengukur rata-rata waktu inferensi per citra.
    
    Args:
        model: Model yang sudah di-load
        dataloader: DataLoader test set
        device: torch.device
        num_warmup: Jumlah batch warmup sebelum pengukuran
    
    Returns:
        Rata-rata waktu inferensi per citra dalam milidetik
    """
    model.eval()
    
    # Warmup GPU
    warmup_iter = iter(dataloader)
    for _ in range(min(num_warmup, len(dataloader))):
        images, _ = next(warmup_iter)
        images = images.to(device, non_blocking=True)
        with torch.no_grad():
            with autocast(device_type="cuda", enabled=(device.type == "cuda")):
                _ = model(images)
    
    if device.type == "cuda":
        torch.cuda.synchronize()
    
    # Pengukuran sebenarnya
    total_time = 0.0
    total_images = 0
    
    for images, _ in dataloader:
        images = images.to(device, non_blocking=True)
        batch_size = images.size(0)
        
        if device.type == "cuda":
            torch.cuda.synchronize()
        
        start = time.perf_counter()
        
        with torch.no_grad():
            with autocast(device_type="cuda", enabled=(device.type == "cuda")):
                _ = model(images)
        
        if device.type == "cuda":
            torch.cuda.synchronize()
        
        end = time.perf_counter()
        
        total_time += (end - start)
        total_images += batch_size
    
    avg_time_ms = (total_time / total_images) * 1000  # ms per image
    return avg_time_ms


def compare_all_models():
    """
    Membandingkan ketiga model secara lengkap.
    
    Returns:
        pandas.DataFrame dengan hasil perbandingan
    """
    print("=" * 70)
    print("  PERBANDINGAN MODEL CIFAKE DETECTION")
    print("=" * 70)
    print()
    
    device = config.DEVICE
    results = []
    
    for model_name in config.ALL_MODELS:
        print(f"{'─'*50}")
        print(f"  Evaluating: {model_name}")
        print(f"{'─'*50}")
        
        # Load model
        model_path = os.path.join(config.MODEL_DIR, f"{model_name}_best.pt")
        if not os.path.exists(model_path):
            print(f"  [SKIP] Model checkpoint tidak ditemukan: {model_path}")
            continue
        
        model = get_model(model_name)
        model.load_state_dict(
            torch.load(model_path, map_location=device, weights_only=True)
        )
        model.eval()
        
        # Data
        _, _, test_loader = get_dataloaders(model_name)
        
        # Evaluasi metrik
        eval_results = evaluate_model(model, test_loader, device)
        metrics = eval_results["metrics"]
        
        # Hitung parameter
        params = count_parameters(model)
        
        # Ukur waktu inferensi
        print("  Mengukur waktu inferensi...")
        avg_inference_ms = measure_inference_time(model, test_loader, device)
        
        # Kumpulkan hasil
        result = {
            "Model": model_name,
            "Accuracy (%)": round(metrics["accuracy"] * 100, 2),
            "Precision (%)": round(metrics["precision"] * 100, 2),
            "Recall (%)": round(metrics["recall"] * 100, 2),
            "F1-Score (%)": round(metrics["f1_score"] * 100, 2),
            "AUC-ROC": round(metrics["auc_roc"], 4),
            "Inference (ms/img)": round(avg_inference_ms, 3),
            "Total Params": params["total"],
            "Trainable Params": params["trainable"],
        }
        results.append(result)
        
        print(f"  Accuracy: {metrics['accuracy']:.4%} | "
              f"F1: {metrics['f1_score']:.4%} | "
              f"Inference: {avg_inference_ms:.3f} ms/img")
        print()
    
    if not results:
        print("[ERROR] Tidak ada model yang bisa dievaluasi.")
        return None
    
    # Buat DataFrame
    df = pd.DataFrame(results)
    
    # ---- Print Tabel Perbandingan ----
    print("\n" + "=" * 90)
    print("  TABEL PERBANDINGAN AKHIR")
    print("=" * 90)
    
    # Format parameter count untuk display
    display_df = df.copy()
    display_df["Total Params"] = display_df["Total Params"].apply(
        lambda x: f"{x:,}"
    )
    display_df["Trainable Params"] = display_df["Trainable Params"].apply(
        lambda x: f"{x:,}"
    )
    
    print(tabulate(
        display_df, headers="keys", tablefmt="grid",
        showindex=False, numalign="right"
    ))
    
    # ---- Save CSV ----
    csv_path = os.path.join(config.RESULT_DIR, "model_comparison.csv")
    df.to_csv(csv_path, index=False)
    print(f"\n[Save] Tabel perbandingan: {csv_path}")
    
    # ---- Save JSON ----
    json_path = os.path.join(config.RESULT_DIR, "model_comparison.json")
    df.to_json(json_path, orient="records", indent=2)
    print(f"[Save] JSON perbandingan: {json_path}")
    
    # ---- Comparison Bar Chart ----
    plot_comparison_chart(df)
    
    return df


def plot_comparison_chart(df: pd.DataFrame):
    """
    Plot bar chart perbandingan metrik antar model.
    """
    save_dir = os.path.join(config.RESULT_DIR, "plots")
    os.makedirs(save_dir, exist_ok=True)
    
    models = df["Model"].tolist()
    metrics_to_plot = ["Accuracy (%)", "F1-Score (%)", "AUC-ROC"]
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    colors = ["#2196F3", "#4CAF50", "#FF9800"]
    
    for idx, metric in enumerate(metrics_to_plot):
        values = df[metric].tolist()
        bars = axes[idx].bar(models, values, color=colors, edgecolor="white", width=0.6)
        axes[idx].set_title(metric, fontsize=13, fontweight="bold")
        axes[idx].set_ylabel(metric, fontsize=11)
        axes[idx].grid(axis="y", alpha=0.3)

        # Set y-axis range untuk visibility (sumbu dipotong agar selisih antar model terlihat)
        min_val = min(values)
        if "%" in metric:
            axes[idx].set_ylim(max(0, min_val - 5), 100.5)
        else:
            axes[idx].set_ylim(max(0, min_val - 0.01), 1.001)

        # Tampilkan nilai di atas bar (offset 1% dari rentang sumbu)
        y_lo, y_hi = axes[idx].get_ylim()
        for bar, val in zip(bars, values):
            axes[idx].text(
                bar.get_x() + bar.get_width() / 2, bar.get_height() + (y_hi - y_lo) * 0.01,
                f"{val}", ha="center", va="bottom", fontsize=10, fontweight="bold"
            )
    
    plt.suptitle("Perbandingan Model CIFAKE Detection", fontsize=15, fontweight="bold", y=1.02)
    plt.tight_layout()
    
    save_path = os.path.join(save_dir, "model_comparison_chart.png")
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    
    print(f"[Plot] Comparison chart: {save_path}")
    
    # ---- Inference Time Chart ----
    fig2, ax2 = plt.subplots(figsize=(8, 5))
    inf_times = df["Inference (ms/img)"].tolist()
    bars = ax2.bar(models, inf_times, color=colors, edgecolor="white", width=0.5)
    ax2.set_title("Waktu Inferensi per Citra", fontsize=13, fontweight="bold")
    ax2.set_ylabel("Waktu (ms/image)", fontsize=11)
    ax2.grid(axis="y", alpha=0.3)
    
    for bar, val in zip(bars, inf_times):
        ax2.text(
            bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
            f"{val:.3f}", ha="center", va="bottom", fontsize=10, fontweight="bold"
        )
    
    plt.tight_layout()
    save_path2 = os.path.join(save_dir, "inference_time_chart.png")
    fig2.savefig(save_path2, dpi=150, bbox_inches="tight")
    plt.close(fig2)
    
    print(f"[Plot] Inference time chart: {save_path2}")


# ============================================================
# CLI ENTRY POINT
# ============================================================

if __name__ == "__main__":
    compare_all_models()
