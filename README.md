# CIFAKE Detection - Deteksi Citra AI-Generated vs Citra Asli

> **Tugas Akhir / Skripsi**  
> *Rancang Bangun Aplikasi Deteksi Citra AI-Generated dan Citra Asli Berbasis Deep Learning*

## 📋 Deskripsi Project

Project ini membandingkan tiga arsitektur CNN untuk mengklasifikasi citra
AI-Generated (FAKE) dan citra asli (REAL) menggunakan dataset CIFAKE.

### Model yang Dibandingkan

| # | Model | Strategi | Input Size |
|---|-------|----------|-----------|
| 1 | **Lightweight CNN** | Training from scratch (8 conv layers, 32-64-128 filters, BatchNorm+Dropout) | 32×32 |
| 2 | **ResNet-50** | Transfer learning (ImageNet pretrained, fine-tune layer4+fc) | 224×224 |
| 3 | **EfficientNetV2-B0** | Transfer learning (ImageNet pretrained, fine-tune last 2 blocks+classifier) | 224×224 |

## 📁 Struktur Project

```
cifake-detection/
├── config.py                         # Konfigurasi global (hyperparameters, paths)
├── data_loader.py                    # Data loading, augmentasi, JPEG compression dataset
├── models.py                         # Definisi 3 arsitektur model
├── train.py                          # Training loop generik (EarlyStopping, AMP)
├── evaluate.py                       # Evaluasi (accuracy, F1, confusion matrix, ROC)
├── compare_models.py                 # Perbandingan ketiga model (tabel & chart)
├── test_compression_robustness.py    # Uji ketahanan terhadap kompresi JPEG
├── grad_cam_viz.py                   # Grad-CAM visualization
├── run_all.py                        # Master script (jalankan semua tahap)
├── requirements.txt                  # Dependencies
├── README.md                         # Dokumentasi ini
├── web/                              # Website CekCitra (HTML/CSS/JS + server Flask)
│   ├── index.html
│   ├── app.py                        # python web/app.py → http://127.0.0.1:5000
│   └── assets/
├── data/                             # (tidak di-commit, lihat .gitignore)
│   └── archive/                      # Dataset CIFAKE (download manual)
│       ├── train/
│       │   ├── REAL/                 # 50,000 citra asli
│       │   └── FAKE/                 # 50,000 citra AI-generated
│       └── test/
│           ├── REAL/                 # 10,000 citra asli
│           └── FAKE/                 # 10,000 citra AI-generated
├── models/                           # Bobot model terbaik (.pt)
├── results/                          # Hasil evaluasi, metrik, CSV
│   ├── plots/                        # Grafik training, confusion matrix, ROC
│   └── gradcam/                      # Heatmap Grad-CAM
└── notebooks/                        # Jupyter notebooks (opsional)
```

## 🚀 Quick Start

### 1. Setup Environment

```bash
# Buat virtual environment (opsional tapi recommended)
python -m venv venv
venv\Scripts\activate          # Windows

# Install dependencies
pip install -r requirements.txt
```

### 2. Download Dataset CIFAKE

Download dari Kaggle:  
https://www.kaggle.com/datasets/birdy654/cifake-real-and-ai-generated-synthetic-images

Ekstrak ke folder `data/archive/` sehingga strukturnya:
```
data/archive/train/REAL/
data/archive/train/FAKE/
data/archive/test/REAL/
data/archive/test/FAKE/
```

### 3. Jalankan Pipeline Lengkap

```bash
# Jalankan semua tahap (training → evaluasi → perbandingan → kompresi → Grad-CAM)
python run_all.py
```

### 4. Atau Jalankan Per Tahap

```bash
# Training satu model
python train.py --model LightweightCNN
python train.py --model ResNet50
python train.py --model EfficientNetV2B0

# Training semua model
python train.py

# Evaluasi
python evaluate.py
python evaluate.py --model ResNet50

# Perbandingan
python compare_models.py

# Uji kompresi JPEG
python test_compression_robustness.py

# Grad-CAM
python grad_cam_viz.py --model LightweightCNN --num-samples 5
```

## ⚙️ Konfigurasi

Semua hyperparameter tersentralisasi di `config.py`:

| Parameter | Default | Keterangan |
|-----------|---------|-----------|
| `NUM_EPOCHS` | 25 | Maksimum epoch |
| `EARLY_STOPPING_PATIENCE` | 7 | Epoch tanpa improvement sebelum stop |
| `BATCH_SIZE` | 64 | Batch size untuk Lightweight CNN |
| `BATCH_SIZE_TRANSFER` | 32 | Batch size untuk transfer learning |
| `LEARNING_RATE` | 1e-3 | LR untuk training from scratch |
| `LEARNING_RATE_PRETRAINED` | 1e-4 | LR untuk fine-tuning pretrained layers |
| `VAL_SPLIT_RATIO` | 0.2 | Proporsi data training untuk validasi |
| `RANDOM_SEED` | 42 | Seed untuk reprodusibilitas |

## 📊 Output yang Dihasilkan

Setelah pipeline selesai, folder `results/` berisi:

- **Training curves** - Plot loss & accuracy per epoch
- **Confusion matrices** - Per model
- **ROC curves** - Per model
- **Tabel perbandingan** - CSV dengan akurasi, F1, waktu inferensi, jumlah parameter
- **Compression robustness** - Akurasi pada JPEG quality 100/80/60/40
- **Grad-CAM heatmaps** - Visualisasi area fokus model

## 🖥️ System Requirements

- **Python** ≥ 3.10
- **GPU**: NVIDIA dengan CUDA support (direkomendasikan ≥ 6GB VRAM)
- **Tested on**: RTX 4060 Laptop (8GB VRAM)
- **Estimasi waktu training**: ~1-2 jam total (ketiga model)

## 📚 Referensi

- **Dataset CIFAKE**: Bird, J.J., & Lotfi, A. (2024). CIFAKE: Image Classification and Explainable Identification of AI-Generated Synthetic Images.
- **ResNet-50**: He, K., et al. (2016). Deep Residual Learning for Image Recognition.
- **EfficientNetV2**: Tan, M., & Le, Q.V. (2021). EfficientNetV2: Smaller Models and Faster Training.
- **Grad-CAM**: Selvaraju, R.R., et al. (2017). Grad-CAM: Visual Explanations from Deep Networks.
