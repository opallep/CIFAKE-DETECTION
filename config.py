"""
config.py - Konfigurasi Global Project CIFAKE Detection
========================================================
Berisi semua hyperparameter, path, dan konstanta yang digunakan
di seluruh project. Sentralisasi konfigurasi memudahkan
eksperimen dan reprodusibilitas.
"""

import os
import torch

# ============================================================
# PATH CONFIGURATION
# ============================================================
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
CIFAKE_DIR = os.path.join(DATA_DIR, "archive")
TRAIN_DIR = os.path.join(CIFAKE_DIR, "train")
TEST_DIR = os.path.join(CIFAKE_DIR, "test")
MODEL_DIR = os.path.join(PROJECT_ROOT, "models")
RESULT_DIR = os.path.join(PROJECT_ROOT, "results")

# ============================================================
# DEVICE CONFIGURATION
# ============================================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ============================================================
# TRAINING HYPERPARAMETERS
# ============================================================
BATCH_SIZE = 128                   # Batch size untuk lightweight CNN (32x32, ringan di VRAM)
BATCH_SIZE_TRANSFER = 32           # Batch size untuk transfer learning (hemat VRAM)
NUM_WORKERS = 4                    # DataLoader worker threads
NUM_EPOCHS = 25                    # Maksimum epoch training
EARLY_STOPPING_PATIENCE = 7       # Patience untuk early stopping
LEARNING_RATE = 1e-3               # Learning rate untuk training from scratch
LEARNING_RATE_PRETRAINED = 1e-4    # Learning rate untuk fine-tuning pretrained layers
LEARNING_RATE_HEAD = 1e-3          # Learning rate untuk classification head baru
WEIGHT_DECAY = 1e-4                # L2 regularization
SCHEDULER_PATIENCE = 3             # Patience untuk ReduceLROnPlateau
SCHEDULER_FACTOR = 0.5             # Factor pengurangan LR

# ============================================================
# IMAGE CONFIGURATION
# ============================================================
IMG_SIZE_LIGHTWEIGHT = 32          # Ukuran asli CIFAKE
IMG_SIZE_TRANSFER = 224            # Ukuran standar untuk pretrained models

# ImageNet normalization statistics
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# ============================================================
# DATASET CONFIGURATION
# ============================================================
CLASS_NAMES = ["FAKE", "REAL"]     # Urutan alfabetis (ImageFolder default)
NUM_CLASSES = 2
VAL_SPLIT_RATIO = 0.2             # 20% data training untuk validasi
RANDOM_SEED = 42                  # Seed untuk reprodusibilitas

# ============================================================
# MODEL NAMES
# ============================================================
MODEL_LIGHTWEIGHT = "LightweightCNN"
MODEL_RESNET50 = "ResNet50"
MODEL_EFFICIENTNET = "EfficientNetV2B0"
ALL_MODELS = [MODEL_LIGHTWEIGHT, MODEL_RESNET50, MODEL_EFFICIENTNET]

# ============================================================
# JPEG COMPRESSION TEST
# ============================================================
JPEG_QUALITIES = [100, 80, 60, 40]

# ============================================================
# AUTO-CREATE DIRECTORIES
# ============================================================
for _dir in [MODEL_DIR, RESULT_DIR, os.path.join(RESULT_DIR, "plots"),
             os.path.join(RESULT_DIR, "gradcam")]:
    os.makedirs(_dir, exist_ok=True)

# ============================================================
# STATUS REPORT
# ============================================================
if __name__ == "__main__":
    print("=" * 60)
    print("CIFAKE Detection - Configuration")
    print("=" * 60)
    print(f"Device          : {DEVICE}")
    if DEVICE.type == "cuda":
        print(f"GPU             : {torch.cuda.get_device_name(0)}")
        print(f"VRAM            : {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")
    print(f"Project Root    : {PROJECT_ROOT}")
    print(f"Data Dir        : {CIFAKE_DIR}")
    print(f"Batch Size      : {BATCH_SIZE} (lightweight) / {BATCH_SIZE_TRANSFER} (transfer)")
    print(f"Max Epochs      : {NUM_EPOCHS}")
    print(f"Early Stop Pat. : {EARLY_STOPPING_PATIENCE}")
    print(f"Random Seed     : {RANDOM_SEED}")
    print("=" * 60)
