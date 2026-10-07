"""
run_all.py - Master Script untuk Pipeline Lengkap CIFAKE Detection
====================================================================
Menjalankan seluruh pipeline secara berurutan:
1. Verifikasi environment & dataset
2. Training ketiga model
3. Evaluasi pada test set
4. Perbandingan model
5. Uji ketahanan kompresi JPEG
6. Grad-CAM visualization

Penggunaan:
    python run_all.py              # Jalankan semua tahap
    python run_all.py --skip-train # Skip training, langsung evaluasi
"""

import os
import sys
import time
import argparse

import torch


def check_environment():
    """Verifikasi environment Python, PyTorch, dan GPU."""
    print("=" * 70)
    print("  CIFAKE Detection - Environment Check")
    print("=" * 70)
    
    print(f"  Python       : {sys.version.split()[0]}")
    print(f"  PyTorch      : {torch.__version__}")
    print(f"  CUDA         : {'Available' if torch.cuda.is_available() else 'NOT AVAILABLE'}")
    
    if torch.cuda.is_available():
        print(f"  GPU          : {torch.cuda.get_device_name(0)}")
        vram = torch.cuda.get_device_properties(0).total_mem / (1024**3)
        print(f"  VRAM         : {vram:.1f} GB")
    
    # Check required packages
    required = ["torchvision", "timm", "numpy", "pandas", "matplotlib",
                "seaborn", "sklearn", "tqdm", "pytorch_grad_cam", "tabulate"]
    
    missing = []
    for pkg in required:
        try:
            __import__(pkg)
        except ImportError:
            missing.append(pkg)
    
    if missing:
        print(f"\n  [WARNING] Package belum terinstall: {missing}")
        print(f"  Jalankan: pip install -r requirements.txt")
        return False
    
    print(f"  Dependencies : All installed ✓")
    print("=" * 70)
    return True


def check_dataset():
    """Verifikasi keberadaan dataset CIFAKE."""
    import config
    
    print("\n  Checking dataset...")
    
    if not os.path.isdir(config.TRAIN_DIR):
        print("\n[ERROR] Dataset belum ditemukan di data/cifake/. Silakan download CIFAKE dari Kaggle dan extract ke folder tersebut dengan struktur train/REAL, train/FAKE, test/REAL, test/FAKE")
        return False
    
    # Count images
    for split in ["train", "test"]:
        for cls in ["REAL", "FAKE"]:
            path = os.path.join(config.CIFAKE_DIR, split, cls)
            if os.path.isdir(path):
                count = len([f for f in os.listdir(path)
                            if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
                print(f"  {split}/{cls}: {count:,} images")
            else:
                print(f"  [WARNING] Folder tidak ditemukan: {path}")
                return False
    
    print("  Dataset OK ✓")
    return True


def main():
    parser = argparse.ArgumentParser(
        description="CIFAKE Detection - Full Pipeline"
    )
    parser.add_argument("--skip-train", action="store_true",
                        help="Skip training, langsung ke evaluasi")
    parser.add_argument("--skip-eval", action="store_true",
                        help="Skip evaluasi individual")
    parser.add_argument("--skip-compare", action="store_true",
                        help="Skip perbandingan model")
    parser.add_argument("--skip-compression", action="store_true",
                        help="Skip uji kompresi JPEG")
    parser.add_argument("--skip-gradcam", action="store_true",
                        help="Skip Grad-CAM visualization")
    parser.add_argument("--model", "-m", type=str, default=None,
                        help="Train/evaluate hanya model tertentu")
    
    args = parser.parse_args()
    
    total_start = time.time()
    
    # ---- 1. Environment Check ----
    if not check_environment():
        print("\n[ABORT] Environment check failed. Install dependencies dulu.")
        sys.exit(1)
    
    import config
    
    # ---- 2. Dataset Check ----
    if not check_dataset():
        print("\n[ABORT] Dataset tidak tersedia. Setup dataset dulu.")
        sys.exit(1)
    
    # Set reproducibility
    torch.manual_seed(config.RANDOM_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.RANDOM_SEED)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    models_to_process = [args.model] if args.model else config.ALL_MODELS
    
    # ---- 3. Training ----
    if not args.skip_train:
        print("\n\n" + "▓" * 70)
        print("  TAHAP 1: TRAINING")
        print("▓" * 70)
        
        from train import train_model
        
        for model_name in models_to_process:
            train_model(model_name)
    else:
        print("\n[SKIP] Training dilewati (--skip-train)")
    
    # ---- 4. Evaluation ----
    if not args.skip_eval:
        print("\n\n" + "▓" * 70)
        print("  TAHAP 2: EVALUASI INDIVIDUAL")
        print("▓" * 70)
        
        from evaluate import full_evaluate
        
        for model_name in models_to_process:
            full_evaluate(model_name)
    else:
        print("\n[SKIP] Evaluasi dilewati (--skip-eval)")
    
    # ---- 5. Comparison ----
    if not args.skip_compare:
        print("\n\n" + "▓" * 70)
        print("  TAHAP 3: PERBANDINGAN MODEL")
        print("▓" * 70)
        
        from compare_models import compare_all_models
        compare_all_models()
    else:
        print("\n[SKIP] Perbandingan dilewati (--skip-compare)")
    
    # ---- 6. Compression Test ----
    if not args.skip_compression:
        print("\n\n" + "▓" * 70)
        print("  TAHAP 4: UJI KETAHANAN KOMPRESI JPEG")
        print("▓" * 70)
        
        from test_compression_robustness import test_compression_robustness
        test_compression_robustness()
    else:
        print("\n[SKIP] Uji kompresi dilewati (--skip-compression)")
    
    # ---- 7. Grad-CAM ----
    if not args.skip_gradcam:
        print("\n\n" + "▓" * 70)
        print("  TAHAP 5: GRAD-CAM VISUALIZATION")
        print("▓" * 70)
        
        from grad_cam_viz import generate_gradcam
        
        for model_name in models_to_process:
            generate_gradcam(model_name, num_samples=5)
    else:
        print("\n[SKIP] Grad-CAM dilewati (--skip-gradcam)")
    
    # ---- Summary ----
    total_time = time.time() - total_start
    print("\n\n" + "█" * 70)
    print("  PIPELINE SELESAI")
    print("█" * 70)
    print(f"  Total waktu : {total_time:.1f}s ({total_time/60:.1f} min)")
    print(f"  Hasil       : {config.RESULT_DIR}")
    print(f"  Models      : {config.MODEL_DIR}")
    print(f"  Plots       : {os.path.join(config.RESULT_DIR, 'plots')}")
    print(f"  Grad-CAM    : {os.path.join(config.RESULT_DIR, 'gradcam')}")
    print("█" * 70)


if __name__ == "__main__":
    main()
