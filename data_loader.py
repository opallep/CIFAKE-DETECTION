"""
data_loader.py - Data Loading & Preprocessing untuk CIFAKE Dataset
===================================================================
Modul ini menangani:
1. Definisi transform/augmentasi untuk setiap tipe model
2. Loading dataset CIFAKE menggunakan ImageFolder
3. Pembagian data training menjadi train/validation
4. Pembuatan DataLoader untuk training, validasi, dan testing
5. Transform khusus untuk uji kompresi JPEG

Dataset CIFAKE:
- 100.000 citra training (50k REAL + 50k FAKE)
- 20.000 citra test (10k REAL + 10k FAKE)
- Resolusi asli: 32x32 RGB
- Struktur: train/REAL, train/FAKE, test/REAL, test/FAKE
"""

import os
import io
from PIL import Image

import torch
from torch.utils.data import DataLoader, random_split, Dataset
from torchvision import datasets, transforms

import config


# ============================================================
# TRANSFORM DEFINITIONS
# ============================================================

def get_transforms(model_name: str, is_training: bool = True):
    """
    Mengembalikan transform yang sesuai untuk model tertentu.
    
    - LightweightCNN: input 32x32, augmentasi standar saat training
    - ResNet50 / EfficientNetV2B0: input 224x224, augmentasi minimal
    
    Args:
        model_name: Nama model (sesuai config.ALL_MODELS)
        is_training: True untuk training set, False untuk val/test
    
    Returns:
        torchvision.transforms.Compose
    """
    if model_name == config.MODEL_LIGHTWEIGHT:
        # ---- Lightweight CNN: 32x32, augmentasi untuk training ----
        # Hanya augmentasi geometris tanpa interpolasi (flip + crop bergeser).
        # Rotasi (interpolasi bilinear) dan ColorJitter dihindari karena dapat
        # menghapus artefak frekuensi tinggi & statistik warna yang menjadi
        # petunjuk citra AI-generated.
        if is_training:
            return transforms.Compose([
                transforms.Resize((config.IMG_SIZE_LIGHTWEIGHT, config.IMG_SIZE_LIGHTWEIGHT)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomCrop(config.IMG_SIZE_LIGHTWEIGHT, padding=2, padding_mode="reflect"),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ])
        else:
            return transforms.Compose([
                transforms.Resize((config.IMG_SIZE_LIGHTWEIGHT, config.IMG_SIZE_LIGHTWEIGHT)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ])
    else:
        # ---- Transfer Learning: 224x224, normalisasi ImageNet ----
        if is_training:
            return transforms.Compose([
                transforms.Resize((config.IMG_SIZE_TRANSFER, config.IMG_SIZE_TRANSFER)),
                transforms.RandomHorizontalFlip(p=0.3),
                transforms.ToTensor(),
                transforms.Normalize(mean=config.IMAGENET_MEAN, std=config.IMAGENET_STD),
            ])
        else:
            return transforms.Compose([
                transforms.Resize((config.IMG_SIZE_TRANSFER, config.IMG_SIZE_TRANSFER)),
                transforms.ToTensor(),
                transforms.Normalize(mean=config.IMAGENET_MEAN, std=config.IMAGENET_STD),
            ])


# ============================================================
# DATASET LOADING
# ============================================================

def get_datasets(model_name: str):
    """
    Memuat dataset CIFAKE dan membagi training set menjadi train/val.
    
    Args:
        model_name: Nama model untuk menentukan transform
    
    Returns:
        Tuple (train_dataset, val_dataset, test_dataset)
    """
    train_transform = get_transforms(model_name, is_training=True)
    eval_transform = get_transforms(model_name, is_training=False)
    
    # Load full training set
    full_train = datasets.ImageFolder(
        root=config.TRAIN_DIR,
        transform=train_transform
    )
    
    # Load test set
    test_dataset = datasets.ImageFolder(
        root=config.TEST_DIR,
        transform=eval_transform
    )
    
    # Split training → train + validation
    total = len(full_train)
    val_size = int(total * config.VAL_SPLIT_RATIO)
    train_size = total - val_size
    
    generator = torch.Generator().manual_seed(config.RANDOM_SEED)
    train_dataset, val_dataset_raw = random_split(
        full_train, [train_size, val_size], generator=generator
    )
    
    # Wrap validation split agar menggunakan eval transform (tanpa augmentasi)
    val_dataset = TransformSubset(val_dataset_raw, eval_transform)
    
    print(f"[Data] Model: {model_name}")
    print(f"[Data] Training   : {len(train_dataset):,} citra")
    print(f"[Data] Validation : {len(val_dataset):,} citra")
    print(f"[Data] Test       : {len(test_dataset):,} citra")
    print(f"[Data] Classes    : {full_train.classes}")
    
    return train_dataset, val_dataset, test_dataset


class TransformSubset(Dataset):
    """
    Wrapper untuk menerapkan transform berbeda pada subset dataset.
    Diperlukan karena random_split mewarisi transform dari dataset induk,
    sementara kita ingin validation set tanpa augmentasi.
    """
    
    def __init__(self, subset, transform):
        self.subset = subset
        self.transform = transform
    
    def __len__(self):
        return len(self.subset)
    
    def __getitem__(self, idx):
        # Ambil data dari subset (sudah di-transform oleh parent)
        # Kita perlu akses gambar asli, bukan yang sudah di-transform
        original_dataset = self.subset.dataset
        original_idx = self.subset.indices[idx]
        
        # Ambil path gambar asli
        img_path, label = original_dataset.samples[original_idx]
        
        # Load ulang dan apply eval transform
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        
        return img, label


# ============================================================
# DATALOADER CREATION
# ============================================================

def get_dataloaders(model_name: str):
    """
    Membuat DataLoader untuk training, validation, dan test.
    
    Args:
        model_name: Nama model untuk menentukan transform dan batch size
    
    Returns:
        Tuple (train_loader, val_loader, test_loader)
    """
    train_dataset, val_dataset, test_dataset = get_datasets(model_name)
    
    # Tentukan batch size berdasarkan tipe model
    if model_name == config.MODEL_LIGHTWEIGHT:
        batch_size = config.BATCH_SIZE
    else:
        batch_size = config.BATCH_SIZE_TRANSFER
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
        drop_last=True,
        persistent_workers=config.NUM_WORKERS > 0,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
        persistent_workers=config.NUM_WORKERS > 0,
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
    )
    
    print(f"[Data] Batch Size : {batch_size}")
    print(f"[Data] Train Batches: {len(train_loader)}, Val Batches: {len(val_loader)}, "
          f"Test Batches: {len(test_loader)}")
    print()
    
    return train_loader, val_loader, test_loader


# ============================================================
# JPEG COMPRESSION DATASET (untuk uji ketahanan)
# ============================================================

class JPEGCompressedDataset(Dataset):
    """
    Dataset yang menerapkan kompresi JPEG dengan quality tertentu
    secara on-the-fly. Digunakan untuk menguji ketahanan model
    terhadap degradasi kualitas citra.
    
    Args:
        base_dataset_path: Path ke folder dataset (format ImageFolder)
        jpeg_quality: Level kualitas JPEG (1-100)
        transform: Transform yang diterapkan setelah kompresi
    """
    
    def __init__(self, base_dataset_path: str, jpeg_quality: int, transform):
        self.dataset = datasets.ImageFolder(base_dataset_path)
        self.jpeg_quality = jpeg_quality
        self.transform = transform
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        img_path, label = self.dataset.samples[idx]
        
        # Load gambar asli
        img = Image.open(img_path).convert("RGB")
        
        # Kompres dengan JPEG quality tertentu (in-memory)
        if self.jpeg_quality < 100:
            buffer = io.BytesIO()
            img.save(buffer, format="JPEG", quality=self.jpeg_quality)
            buffer.seek(0)
            img = Image.open(buffer).convert("RGB")
        
        # Apply transform
        if self.transform:
            img = self.transform(img)
        
        return img, label


def get_compressed_test_loader(model_name: str, jpeg_quality: int):
    """
    Membuat DataLoader untuk test set yang dikompresi JPEG.
    
    Args:
        model_name: Nama model untuk menentukan transform
        jpeg_quality: Level kualitas JPEG (1-100)
    
    Returns:
        DataLoader untuk test set terkompresi
    """
    eval_transform = get_transforms(model_name, is_training=False)
    
    if model_name == config.MODEL_LIGHTWEIGHT:
        batch_size = config.BATCH_SIZE
    else:
        batch_size = config.BATCH_SIZE_TRANSFER
    
    dataset = JPEGCompressedDataset(
        base_dataset_path=config.TEST_DIR,
        jpeg_quality=jpeg_quality,
        transform=eval_transform,
    )
    
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
    )
    
    return loader


# ============================================================
# TESTING
# ============================================================

if __name__ == "__main__":
    print("Testing data_loader.py...")
    print()
    
    if not os.path.isdir(config.TRAIN_DIR):
        print(f"[WARNING] Dataset belum tersedia di: {config.CIFAKE_DIR}")
        print("Download CIFAKE dari: https://www.kaggle.com/datasets/birdy654/cifake-real-and-ai-generated-synthetic-images")
        print(f"Ekstrak ke: {config.CIFAKE_DIR}")
        print("Struktur yang diharapkan:")
        print(f"  {config.CIFAKE_DIR}/train/REAL/")
        print(f"  {config.CIFAKE_DIR}/train/FAKE/")
        print(f"  {config.CIFAKE_DIR}/test/REAL/")
        print(f"  {config.CIFAKE_DIR}/test/FAKE/")
    else:
        for model_name in config.ALL_MODELS:
            print(f"{'='*50}")
            train_loader, val_loader, test_loader = get_dataloaders(model_name)
            
            # Test satu batch dari train_loader
            print("  [Train Loader]")
            images, labels = next(iter(train_loader))
            print(f"    Sample batch shape: {images.shape}")
            print(f"    Labels: {labels[:10]}")
            
            # Test satu batch dari val_loader
            print("  [Val Loader]")
            val_images, val_labels = next(iter(val_loader))
            print(f"    Sample batch shape: {val_images.shape}")
            print(f"    Labels: {val_labels[:10]}")
            
            # Test satu batch dari test_loader
            print("  [Test Loader]")
            test_images, test_labels = next(iter(test_loader))
            print(f"    Sample batch shape: {test_images.shape}")
            print(f"    Labels: {test_labels[:10]}")
            
            # Test compressed loader
            print("  [Compressed Test Loader (Q=60)]")
            comp_loader = get_compressed_test_loader(model_name, jpeg_quality=60)
            comp_images, comp_labels = next(iter(comp_loader))
            print(f"    Sample batch shape: {comp_images.shape}")
            print(f"    Labels: {comp_labels[:10]}")
            print()
