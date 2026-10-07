"""
evaluate.py - Evaluasi Model pada Test Set CIFAKE
===================================================
Modul ini menghitung metrik evaluasi lengkap:
1. Accuracy
2. Precision, Recall, F1-Score (per kelas & macro)
3. Confusion Matrix (+ visualisasi)
4. ROC Curve & AUC Score
5. Classification Report

Hasil evaluasi disimpan ke folder results/.
"""

import os
import json
import numpy as np

import torch
import torch.nn as nn
from torch.amp import autocast
from tqdm import tqdm

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    roc_auc_score,
)

import config
from models import get_model, count_parameters
from data_loader import get_dataloaders


# ============================================================
# CORE EVALUATION FUNCTION
# ============================================================

@torch.no_grad()
def evaluate_model(model, dataloader, device=None):
    """
    Evaluasi model pada dataloader yang diberikan.
    
    Args:
        model: nn.Module model yang sudah di-load
        dataloader: DataLoader untuk evaluasi
        device: torch.device (default: config.DEVICE)
    
    Returns:
        dict berisi:
        - y_true: ground truth labels
        - y_pred: predicted labels
        - y_probs: predicted probabilities (untuk ROC)
        - metrics: dict accuracy, precision, recall, f1
    """
    if device is None:
        device = config.DEVICE
    
    model.eval()
    
    all_labels = []
    all_preds = []
    all_probs = []
    
    pbar = tqdm(dataloader, desc="  Evaluating", leave=False, ncols=100)
    
    for images, labels in pbar:
        images = images.to(device, non_blocking=True)
        
        with autocast(device_type="cuda", enabled=(device.type == "cuda")):
            outputs = model(images)
        
        # Probabilitas via softmax
        probs = torch.softmax(outputs, dim=1).cpu().numpy()
        preds = outputs.argmax(dim=1).cpu().numpy()
        
        all_labels.extend(labels.numpy())
        all_preds.extend(preds)
        all_probs.extend(probs)
    
    y_true = np.array(all_labels)
    y_pred = np.array(all_preds)
    y_probs = np.array(all_probs)
    
    # Hitung metrik
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "recall": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "f1_score": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }
    
    # AUC-ROC (menggunakan probabilitas kelas positif = REAL)
    try:
        metrics["auc_roc"] = roc_auc_score(y_true, y_probs[:, 1])
    except ValueError:
        metrics["auc_roc"] = 0.0
    
    return {
        "y_true": y_true,
        "y_pred": y_pred,
        "y_probs": y_probs,
        "metrics": metrics,
    }


# ============================================================
# VISUALIZATION FUNCTIONS
# ============================================================

def plot_confusion_matrix(y_true, y_pred, model_name: str, save_dir: str = None):
    """
    Plot dan simpan confusion matrix.
    """
    if save_dir is None:
        save_dir = os.path.join(config.RESULT_DIR, "plots")
    os.makedirs(save_dir, exist_ok=True)
    
    cm = confusion_matrix(y_true, y_pred)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=config.CLASS_NAMES,
        yticklabels=config.CLASS_NAMES,
        ax=ax, annot_kws={"size": 14},
    )
    ax.set_xlabel("Predicted Label", fontsize=12)
    ax.set_ylabel("True Label", fontsize=12)
    ax.set_title(f"Confusion Matrix - {model_name}", fontsize=14, fontweight="bold")
    
    plt.tight_layout()
    save_path = os.path.join(save_dir, f"{model_name}_confusion_matrix.png")
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    
    print(f"  [Plot] Confusion matrix: {save_path}")
    return save_path


def plot_roc_curve(y_true, y_probs, model_name: str, save_dir: str = None):
    """
    Plot dan simpan ROC curve.
    """
    if save_dir is None:
        save_dir = os.path.join(config.RESULT_DIR, "plots")
    os.makedirs(save_dir, exist_ok=True)
    
    # Probabilitas kelas positif (REAL = 1)
    fpr, tpr, _ = roc_curve(y_true, y_probs[:, 1])
    auc = roc_auc_score(y_true, y_probs[:, 1])
    
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(fpr, tpr, color="#2196F3", lw=2, label=f"ROC Curve (AUC = {auc:.4f})")
    ax.plot([0, 1], [0, 1], color="gray", linestyle="--", lw=1, label="Random")
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title(f"ROC Curve - {model_name}", fontsize=14, fontweight="bold")
    ax.legend(loc="lower right", fontsize=11)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    save_path = os.path.join(save_dir, f"{model_name}_roc_curve.png")
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    
    print(f"  [Plot] ROC curve: {save_path}")
    return save_path


def plot_training_history(model_name: str, save_dir: str = None):
    """
    Plot training & validation loss/accuracy dari history JSON.
    """
    if save_dir is None:
        save_dir = os.path.join(config.RESULT_DIR, "plots")
    os.makedirs(save_dir, exist_ok=True)
    
    history_path = os.path.join(config.RESULT_DIR, f"{model_name}_history.json")
    if not os.path.exists(history_path):
        print(f"  [WARNING] History tidak ditemukan: {history_path}")
        return None
    
    with open(history_path, "r") as f:
        history = json.load(f)
    
    epochs = range(1, len(history["train_loss"]) + 1)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Loss Plot
    axes[0].plot(epochs, history["train_loss"], "b-o", markersize=4, label="Train Loss")
    axes[0].plot(epochs, history["val_loss"], "r-o", markersize=4, label="Val Loss")
    axes[0].set_xlabel("Epoch", fontsize=11)
    axes[0].set_ylabel("Loss", fontsize=11)
    axes[0].set_title(f"Loss - {model_name}", fontsize=13, fontweight="bold")
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)
    
    # Accuracy Plot
    train_acc_pct = [a * 100 for a in history["train_acc"]]
    val_acc_pct = [a * 100 for a in history["val_acc"]]
    axes[1].plot(epochs, train_acc_pct, "b-o", markersize=4, label="Train Acc")
    axes[1].plot(epochs, val_acc_pct, "r-o", markersize=4, label="Val Acc")
    axes[1].set_xlabel("Epoch", fontsize=11)
    axes[1].set_ylabel("Accuracy (%)", fontsize=11)
    axes[1].set_title(f"Accuracy - {model_name}", fontsize=13, fontweight="bold")
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    save_path = os.path.join(save_dir, f"{model_name}_training_curves.png")
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    
    print(f"  [Plot] Training curves: {save_path}")
    return save_path


# ============================================================
# FULL EVALUATION PIPELINE
# ============================================================

def full_evaluate(model_name: str):
    """
    Evaluasi lengkap satu model pada test set.
    
    Pipeline:
    1. Load model terbaik dari checkpoint
    2. Evaluasi pada test set
    3. Print classification report
    4. Plot confusion matrix & ROC curve
    5. Plot training history
    6. Simpan metrik ke JSON
    
    Args:
        model_name: Nama model
    
    Returns:
        dict metrik evaluasi
    """
    print("=" * 60)
    print(f"  EVALUATION: {model_name}")
    print("=" * 60)
    
    device = config.DEVICE
    
    # ---- 1. Load Model ----
    model_path = os.path.join(config.MODEL_DIR, f"{model_name}_best.pt")
    if not os.path.exists(model_path):
        print(f"[ERROR] Model checkpoint tidak ditemukan: {model_path}")
        print("[ERROR] Jalankan train.py terlebih dahulu.")
        return None
    
    model = get_model(model_name)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()
    print(f"[Model] Loaded from: {model_path}")
    
    # ---- 2. Load Test Data ----
    _, _, test_loader = get_dataloaders(model_name)
    
    # ---- 3. Evaluate ----
    results = evaluate_model(model, test_loader, device)
    metrics = results["metrics"]
    
    # ---- 4. Print Results ----
    print(f"\n  {'─'*40}")
    print(f"  Test Set Metrics - {model_name}")
    print(f"  {'─'*40}")
    print(f"  Accuracy  : {metrics['accuracy']:.4%}")
    print(f"  Precision : {metrics['precision']:.4%}")
    print(f"  Recall    : {metrics['recall']:.4%}")
    print(f"  F1-Score  : {metrics['f1_score']:.4%}")
    print(f"  AUC-ROC   : {metrics['auc_roc']:.4f}")
    print(f"  {'─'*40}")
    
    # Classification Report
    print(f"\n  Classification Report:")
    print(classification_report(
        results["y_true"], results["y_pred"],
        target_names=config.CLASS_NAMES, digits=4
    ))
    
    # ---- 5. Plot Visualizations ----
    plot_confusion_matrix(results["y_true"], results["y_pred"], model_name)
    plot_roc_curve(results["y_true"], results["y_probs"], model_name)
    plot_training_history(model_name)
    
    # ---- 6. Save Metrics ----
    metrics_path = os.path.join(config.RESULT_DIR, f"{model_name}_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"  [Save] Metrics: {metrics_path}")
    print()
    
    return metrics


# ============================================================
# CLI ENTRY POINT
# ============================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Evaluasi Model CIFAKE Detection pada Test Set"
    )
    parser.add_argument(
        "--model", "-m",
        type=str,
        choices=config.ALL_MODELS,
        default=None,
        help=f"Model yang akan dievaluasi. Pilihan: {config.ALL_MODELS}. "
             f"Jika tidak diisi, semua model akan dievaluasi."
    )
    
    args = parser.parse_args()
    
    if args.model:
        models_to_eval = [args.model]
    else:
        models_to_eval = config.ALL_MODELS
    
    all_metrics = {}
    for model_name in models_to_eval:
        metrics = full_evaluate(model_name)
        if metrics:
            all_metrics[model_name] = metrics
    
    # Print summary
    if len(all_metrics) > 1:
        print("\n" + "=" * 70)
        print("  RANGKUMAN EVALUASI SEMUA MODEL")
        print("=" * 70)
        print(f"  {'Model':20s} | {'Accuracy':>10} | {'F1-Score':>10} | {'AUC-ROC':>10}")
        print(f"  {'─'*20}-+-{'─'*10}-+-{'─'*10}-+-{'─'*10}")
        for name, m in all_metrics.items():
            print(f"  {name:20s} | {m['accuracy']:>9.4%} | "
                  f"{m['f1_score']:>9.4%} | {m['auc_roc']:>10.4f}")
        print("=" * 70)
