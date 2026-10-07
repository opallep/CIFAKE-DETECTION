"""
train.py - Training Loop Generik untuk CIFAKE Detection
========================================================
Modul ini menyediakan:
1. EarlyStopping callback
2. Fungsi training dan validasi per epoch
3. Fungsi utama train_model() yang mengorkestrasi keseluruhan
   proses training termasuk:
   - Optimizer & scheduler setup
   - Mixed precision training (AMP)
   - Checkpointing model terbaik
   - History logging (loss & accuracy per epoch)
   - Progress reporting ke console
"""

import os
import time
import json
import copy

import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from tqdm import tqdm

import config
from models import get_model, count_parameters, ResNet50Transfer, EfficientNetV2B0Transfer
from data_loader import get_dataloaders


# ============================================================
# EARLY STOPPING
# ============================================================

class EarlyStopping:
    """
    Early Stopping untuk menghentikan training saat validasi
    tidak membaik setelah sejumlah epoch (patience).
    
    Args:
        patience: Jumlah epoch tanpa improvement sebelum stop
        min_delta: Minimum perubahan yang dianggap improvement
        mode: 'min' untuk loss, 'max' untuk accuracy
    """
    
    def __init__(self, patience: int = 7, min_delta: float = 0.001, mode: str = "max"):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.should_stop = False
    
    def __call__(self, score: float) -> bool:
        if self.best_score is None:
            self.best_score = score
            return False
        
        if self.mode == "max":
            improved = score > self.best_score + self.min_delta
        else:
            improved = score < self.best_score - self.min_delta
        
        if improved:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        
        return self.should_stop


# ============================================================
# TRAINING & VALIDATION PER EPOCH
# ============================================================

def train_one_epoch(model, dataloader, criterion, optimizer, scaler, device):
    """
    Training untuk satu epoch.
    
    Returns:
        Tuple (avg_loss, accuracy)
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(dataloader, desc="  Train", leave=False, ncols=100)
    
    for images, labels in pbar:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        
        optimizer.zero_grad()
        
        # Mixed precision forward pass
        with autocast(device_type="cuda", enabled=(device.type == "cuda")):
            outputs = model(images)
            loss = criterion(outputs, labels)
        
        # Backward pass dengan gradient scaling
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        # Statistik
        running_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        pbar.set_postfix({
            "loss": f"{loss.item():.4f}",
            "acc": f"{100. * correct / total:.2f}%"
        })
    
    avg_loss = running_loss / total
    accuracy = correct / total
    
    return avg_loss, accuracy


@torch.no_grad()
def validate_one_epoch(model, dataloader, criterion, device):
    """
    Validasi untuk satu epoch.
    
    Returns:
        Tuple (avg_loss, accuracy)
    """
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(dataloader, desc="  Val  ", leave=False, ncols=100)
    
    for images, labels in pbar:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        
        with autocast(device_type="cuda", enabled=(device.type == "cuda")):
            outputs = model(images)
            loss = criterion(outputs, labels)
        
        running_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
    
    avg_loss = running_loss / total
    accuracy = correct / total
    
    return avg_loss, accuracy


# ============================================================
# MAIN TRAINING FUNCTION
# ============================================================

def train_model(model_name: str):
    """
    Fungsi utama untuk melatih satu model secara lengkap.
    
    Pipeline:
    1. Load data dengan transform yang sesuai
    2. Inisialisasi model, optimizer, scheduler
    3. Training loop dengan early stopping
    4. Simpan model terbaik dan training history
    
    Args:
        model_name: Nama model (config.MODEL_LIGHTWEIGHT, dsb.)
    
    Returns:
        dict berisi training history dan path model tersimpan
    """
    print("=" * 70)
    print(f"  TRAINING: {model_name}")
    print("=" * 70)
    
    device = config.DEVICE
    
    # ---- 1. Data Loading ----
    train_loader, val_loader, _ = get_dataloaders(model_name)
    
    # ---- 2. Model ----
    model = get_model(model_name)
    params_info = count_parameters(model)
    print(f"[Model] Total params    : {params_info['total']:,}")
    print(f"[Model] Trainable params: {params_info['trainable']:,}")
    print(f"[Model] Frozen params   : {params_info['frozen']:,}")
    print()
    
    # ---- 3. Loss Function ----
    criterion = nn.CrossEntropyLoss()
    
    # ---- 4. Optimizer ----
    if isinstance(model, (ResNet50Transfer, EfficientNetV2B0Transfer)):
        # Differential learning rate untuk transfer learning
        param_groups = model.get_param_groups()
        optimizer = torch.optim.Adam(
            param_groups,
            weight_decay=config.WEIGHT_DECAY
        )
        print(f"[Optimizer] Adam (differential LR: "
              f"pretrained={config.LEARNING_RATE_PRETRAINED}, "
              f"head={config.LEARNING_RATE_HEAD})")
    else:
        # Single learning rate untuk training from scratch
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=config.LEARNING_RATE,
            weight_decay=config.WEIGHT_DECAY
        )
        print(f"[Optimizer] Adam (LR={config.LEARNING_RATE})")
    
    # ---- 5. Scheduler ----
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        patience=config.SCHEDULER_PATIENCE,
        factor=config.SCHEDULER_FACTOR,
        verbose=True,
    )
    
    # ---- 6. Mixed Precision Scaler ----
    scaler = GradScaler("cuda", enabled=(device.type == "cuda"))
    
    # ---- 7. Early Stopping ----
    early_stopping = EarlyStopping(
        patience=config.EARLY_STOPPING_PATIENCE,
        mode="max"
    )
    
    # ---- 8. Training History ----
    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
        "lr": [],
        "epoch_time": [],
    }
    
    best_val_acc = 0.0
    best_model_state = None
    best_epoch = 0
    
    model_save_path = os.path.join(config.MODEL_DIR, f"{model_name}_best.pt")
    
    # ---- 9. Training Loop ----
    print()
    print(f"{'Epoch':>6} | {'Train Loss':>10} | {'Train Acc':>10} | "
          f"{'Val Loss':>10} | {'Val Acc':>10} | {'LR':>10} | {'Time':>8}")
    print("-" * 85)
    
    total_start = time.time()
    
    for epoch in range(1, config.NUM_EPOCHS + 1):
        epoch_start = time.time()
        
        # Training
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, scaler, device
        )
        
        # Validation
        val_loss, val_acc = validate_one_epoch(
            model, val_loader, criterion, device
        )
        
        epoch_time = time.time() - epoch_start
        
        # Current learning rate
        current_lr = optimizer.param_groups[0]["lr"]
        
        # Update scheduler
        scheduler.step(val_acc)
        
        # Save history
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["lr"].append(current_lr)
        history["epoch_time"].append(epoch_time)
        
        # Checkpoint model terbaik
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch
            marker = " ★"
        else:
            marker = ""
        
        # Print progress
        print(f"{epoch:>6} | {train_loss:>10.4f} | {train_acc:>9.2%} | "
              f"{val_loss:>10.4f} | {val_acc:>9.2%} | {current_lr:>10.6f} | "
              f"{epoch_time:>6.1f}s{marker}")
        
        # Early stopping check
        if early_stopping(val_acc):
            print(f"\n[EarlyStopping] Tidak ada improvement selama "
                  f"{config.EARLY_STOPPING_PATIENCE} epoch. Training dihentikan.")
            break
    
    total_time = time.time() - total_start
    
    # ---- 10. Save Best Model ----
    if best_model_state is not None:
        torch.save(best_model_state, model_save_path)
        print(f"\n[Save] Model terbaik (epoch {best_epoch}, "
              f"val_acc={best_val_acc:.4%}) disimpan ke: {model_save_path}")
    
    # ---- 11. Save Training History ----
    history_path = os.path.join(config.RESULT_DIR, f"{model_name}_history.json")
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)
    print(f"[Save] Training history disimpan ke: {history_path}")
    
    # ---- 12. Summary ----
    print(f"\n{'─'*40}")
    print(f"  Training Summary - {model_name}")
    print(f"{'─'*40}")
    print(f"  Best Epoch     : {best_epoch}")
    print(f"  Best Val Acc   : {best_val_acc:.4%}")
    print(f"  Total Time     : {total_time:.1f}s ({total_time/60:.1f} min)")
    print(f"  Epochs Trained : {len(history['train_loss'])}")
    print(f"{'─'*40}")
    print()
    
    return {
        "model_name": model_name,
        "best_epoch": best_epoch,
        "best_val_acc": best_val_acc,
        "total_time": total_time,
        "history": history,
        "model_path": model_save_path,
    }


# ============================================================
# CLI ENTRY POINT
# ============================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Training CIFAKE Detection Model"
    )
    parser.add_argument(
        "--model", "-m",
        type=str,
        choices=config.ALL_MODELS,
        default=None,
        help=f"Model yang akan di-train. Pilihan: {config.ALL_MODELS}. "
             f"Jika tidak diisi, semua model akan di-train."
    )
    
    args = parser.parse_args()
    
    # Set seed untuk reprodusibilitas
    torch.manual_seed(config.RANDOM_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.RANDOM_SEED)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    print("=" * 70)
    print("  CIFAKE Detection - Training Pipeline")
    print(f"  Device: {config.DEVICE}")
    if config.DEVICE.type == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
    print("=" * 70)
    print()
    
    if args.model:
        models_to_train = [args.model]
    else:
        models_to_train = config.ALL_MODELS
    
    results = []
    for model_name in models_to_train:
        result = train_model(model_name)
        results.append(result)
    
    # Print rangkuman semua model
    if len(results) > 1:
        print("\n" + "=" * 70)
        print("  RANGKUMAN TRAINING SEMUA MODEL")
        print("=" * 70)
        for r in results:
            print(f"  {r['model_name']:20s} | "
                  f"Best Val Acc: {r['best_val_acc']:.4%} | "
                  f"Best Epoch: {r['best_epoch']:2d} | "
                  f"Time: {r['total_time']:.0f}s")
        print("=" * 70)
