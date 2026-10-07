# Laporan Sesi 01: Training Model 1 (LightweightCNN) dan Website CekCitra

**Tanggal:** 7 Oktober 2026
**Repo:** https://github.com/opallep/CIFAKE-DETECTION
**Status akhir sesi:** Tahap 1 dan 1b selesai. Tahap berikutnya adalah tahap 2 (training ResNet-50).

---

## 1. Tujuan

Tugas akhir ini membandingkan tiga model CNN untuk mendeteksi apakah sebuah gambar **asli (REAL)** atau **buatan AI (FAKE)** menggunakan dataset CIFAKE. Model dengan akurasi terbaik akan dipakai di website, tempat pengguna bisa mengunggah foto untuk diperiksa.

Tiga model yang dibandingkan (definisinya di `models.py`):

| # | Model | Jenis | Input |
|---|---|---|---|
| 1 | LightweightCNN | CNN 8 lapis konvolusi, dilatih dari nol | 32×32 |
| 2 | ResNet50 | Transfer learning dari ImageNet (layer4 + fc di-fine-tune) | 224×224 |
| 3 | EfficientNetV2B0 | Transfer learning dari ImageNet via `timm` (2 blok terakhir + head) | 224×224 |

Model dilatih **satu per satu**, tidak sekaligus.

## 2. Rencana tahapan dan status

| Tahap | Pekerjaan | Status |
|---|---|---|
| 1 | Training dan evaluasi LightweightCNN | ✅ Selesai |
| 1b | Membuat tampilan website | ✅ Selesai (desain v3) |
| 2 | Training dan evaluasi ResNet-50 | ⏳ Berikutnya |
| 3 | Training dan evaluasi EfficientNetV2-B0 | Belum |
| 4 | Membandingkan ketiga model (`compare_models.py`) lalu memilih yang terbaik | Belum |
| 5 | Menghubungkan website ke model terpilih (`/api/predict` di `web/app.py`) | Belum |
| 6 | Menguji website dengan foto asli dari luar dataset | Belum |

Alasan training didahulukan sebelum website: input website (32×32 atau 224×224), cara normalisasi, dan file model baru bisa dipastikan setelah model terbaik dipilih. Selain itu, hasil perbandingan model adalah inti Bab 4. Tampilan website tetap boleh dikerjakan dari awal karena tidak bergantung pada model yang dipilih.

## 3. Lingkungan

| Komponen | Versi |
|---|---|
| OS | Windows 11 Home |
| GPU | NVIDIA GeForce RTX 4060 Laptop, 8 GB VRAM |
| Python | 3.12.6 |
| PyTorch / torchvision | 2.6.0+cu124 / 0.21.0+cu124 (CUDA aktif) |
| Library lain | timm, scikit-learn, matplotlib, seaborn, tqdm (sudah terpasang) |
| Ditambahkan di sesi ini | Flask 3.1.3, Git 2.55 (dipasang via winget) |

## 4. Dataset

CIFAKE, disimpan di `data/archive/`. Folder ini tidak di-commit (lihat `.gitignore`).

| Split | REAL | FAKE | Total |
|---|---|---|---|
| `train/` | 50.000 | 50.000 | 100.000 |
| `test/` | 10.000 | 10.000 | 20.000 |

- Semua gambar berformat JPG berukuran **32×32 px**.
- Gambar FAKE dibuat dengan Stable Diffusion v1.4.
- Data `train/` dibagi lagi menjadi **80.000 untuk train dan 20.000 untuk validasi** dengan `random_split` dan seed 42. Ketiga model memakai pembagian yang sama persis, sehingga perbandingannya adil.
- Data `test/` **tidak dipakai sama sekali saat training** dan hanya digunakan untuk evaluasi akhir.
- Urutan kelas (ImageFolder): `0 = FAKE`, `1 = REAL`.

## 5. Perubahan kode di sesi ini

| File | Perubahan | Alasan |
|---|---|---|
| `config.py` | `CIFAKE_DIR` dari `data/cifake` menjadi `data/archive` | Menyesuaikan lokasi dataset yang sebenarnya |
| `config.py` | `BATCH_SIZE` 64 menjadi 128 | Gambar 32×32 ringan untuk VRAM, sehingga training lebih cepat dan stabil |
| `data_loader.py` | Augmentasi LightweightCNN: **RandomRotation dan ColorJitter dihapus**, diganti `RandomCrop(32, padding=2, reflect)`. `RandomHorizontalFlip` tetap dipakai | Rotasi (interpolasi bilinear) dan ColorJitter bisa menghapus artefak frekuensi tinggi dan statistik warna yang menjadi ciri gambar AI. Alasan ini bisa ditulis di bab metodologi |
| `data_loader.py` | `persistent_workers=True` untuk loader train dan val | Di Windows, worker tidak perlu dibuat ulang setiap epoch |
| `README.md` | Path dataset diperbarui dan struktur folder `web/` ditambahkan | Dokumentasi |
| `.gitignore` | Baru: mengabaikan `data/`, `models/*.pt`, `results/*_log.txt`, `__pycache__` | Dataset 105 MB dan bobot ResNet bisa lebih dari 90 MB (batas file GitHub 100 MB) |

## 6. Hyperparameter Model 1 (LightweightCNN)

| Parameter | Nilai |
|---|---|
| Parameter model | 583.138 (semuanya trainable) |
| Input | 32×32 RGB, normalisasi mean=std=0,5 (rentang −1 sampai 1) |
| Optimizer | Adam, learning rate 1e-3, weight decay 1e-4 |
| Batch size | 128 (625 batch per epoch) |
| Loss | CrossEntropyLoss |
| Scheduler | ReduceLROnPlateau (berdasarkan val_acc), patience 3, factor 0,5 |
| Early stopping | patience 7, min_delta 0,001 |
| Maksimal epoch | 25 |
| Lainnya | Mixed precision (AMP), cudnn deterministic, seed 42; checkpoint disimpan berdasarkan val_acc terbaik |

## 7. Hasil training Model 1

Perintah: `python train.py --model LightweightCNN`

| Epoch | Train Loss | Train Acc | Val Loss | Val Acc | LR | Waktu (detik) |
|---|---|---|---|---|---|---|
| 1 | 0.3312 | 85.77% | 0.3184 | 86.83% | 0.001 | 259 |
| 2 | 0.2443 | 90.23% | 0.2441 | 90.10% | 0.001 | 226 |
| 3 | 0.2219 | 91.30% | 0.3216 | 87.18% | 0.001 | 209 |
| 4 | 0.2065 | 91.96% | 0.2223 | 90.76% | 0.001 | 286 |
| 5 | 0.1975 | 92.26% | 0.2109 | 91.56% | 0.001 | 229 |
| 6 | 0.1883 | 92.65% | 0.2577 | 89.84% | 0.001 | 212 |
| 7 | 0.1815 | 93.03% | 0.1758 | 93.07% | 0.001 | 25 |
| 8 | 0.1780 | 93.19% | 0.2193 | 91.27% | 0.001 | 16 |
| 9 | 0.1749 | 93.30% | 0.1596 | 93.73% | 0.001 | 16 |
| 10 | 0.1712 | 93.49% | 0.1585 | 93.79% | 0.001 | 16 |
| 11 | 0.1684 | 93.52% | 0.2002 | 92.24% | 0.001 | 16 |
| 12 | 0.1627 | 93.76% | 0.1365 | 94.57% | 0.001 | 16 |
| 13 | 0.1613 | 93.75% | 0.1837 | 92.66% | 0.001 | 16 |
| 14 | 0.1578 | 94.00% | 0.1598 | 93.69% | 0.001 | 16 |
| 15 | 0.1508 | 94.20% | 0.1948 | 92.16% | 0.001 | 16 |
| 16 | 0.1504 | 94.28% | 0.1758 | 93.10% | 0.001 | 17 |
| **17** | **0.1361** | **94.87%** | **0.1320** | **94.80%** ★ | 0.0005 | 235 |
| 18 | 0.1320 | 95.02% | 0.1481 | 94.16% | 0.0005 | 233 |
| 19 | 0.1311 | 95.06% | 0.1784 | 92.68% | 0.0005 | 256 |
| 20 | 0.1303 | 95.07% | 0.1464 | 94.14% | 0.0005 | 247 |
| 21 | 0.1280 | 95.13% | 0.1587 | 93.64% | 0.0005 | 337 |
| 22 | 0.1190 | 95.47% | 0.1559 | 93.86% | 0.00025 | 304 |
| 23 | 0.1157 | 95.71% | 0.1336 | 94.79% | 0.00025 | 236 |
| 24 | 0.1158 | 95.69% | 0.1362 | 94.58% | 0.00025 | 250 |

**Ringkasan:**
- Training berhenti di epoch 24 karena early stopping (7 epoch tanpa peningkatan).
- Model terbaik ada di **epoch 17** dengan **val acc 94,80%**, disimpan di `models/LightweightCNN_best.pt` (2,3 MB).
- Total waktu training 61,5 menit.
- Learning rate turun otomatis dua kali (menjadi 5e-4, lalu 2,5e-4), dan setiap kali turun akurasi validasi membaik.
- Tidak ada overfitting: train acc dan val acc selalu berdekatan (selisih kurang dari 1–3%).
- Waktu per epoch berubah-ubah (16 detik sampai 5 menit). Penyebabnya pembacaan 100.000 file JPG kecil dari disk. Waktu jadi sekitar 16 detik ketika file sudah ada di cache memori Windows. GPU bukan penyebabnya.

## 8. Hasil evaluasi Model 1 pada test set (20.000 gambar)

Perintah: `python evaluate.py --model LightweightCNN`

| Metrik | Nilai |
|---|---|
| **Accuracy** | **94,75%** |
| Precision (macro) | 94,93% |
| Recall (macro) | 94,75% |
| F1-Score (macro) | 94,74% |
| **AUC-ROC** | **0,9921** |

**Classification report:**

| Kelas | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| FAKE | 0,9779 | 0,9157 | 0,9458 | 10.000 |
| REAL | 0,9207 | 0,9793 | 0,9491 | 10.000 |

**Confusion matrix** (baris = label sebenarnya, kolom = prediksi):

| | Prediksi FAKE | Prediksi REAL |
|---|---|---|
| **FAKE** | 9.157 | 843 |
| **REAL** | 207 | 9.793 |

**Analisis singkat:**
- Akurasi test (94,75%) hampir sama dengan akurasi validasi (94,80%), jadi model bisa menangani data baru dengan baik.
- AUC 0,992 berarti kemampuan model membedakan kedua kelas sangat baik.
- Kelemahannya: **843 gambar AI (8,4%) lolos dan dianggap asli**, sedangkan foto asli yang salah dianggap AI hanya 207 (2,1%). Model cenderung memilih kelas REAL. Ini bisa menjadi poin perbandingan dengan ResNet-50 dan EfficientNet di Bab 4.

**File hasil** (sudah di-commit):
- `results/LightweightCNN_metrics.json`
- `results/LightweightCNN_history.json`
- `results/plots/LightweightCNN_confusion_matrix.png`
- `results/plots/LightweightCNN_roc_curve.png`
- `results/plots/LightweightCNN_training_curves.png`

## 9. Website CekCitra

### Struktur

```
web/
├── index.html            # Halaman utama, HTML statis murni
├── app.py                # Server Flask
└── assets/
    ├── style.css
    ├── app.js
    └── samples/          # 12 contoh gambar dari test set (real_1..6, fake_1..6)
```

### Cara menjalankan

- **Tanpa server:** dobel-klik `web/index.html`. Halaman berjalan dalam **mode simulasi**.
- **Dengan server:** jalankan `python web/app.py`, lalu buka http://127.0.0.1:5000. Selama model belum dihubungkan, hasilnya tetap simulasi.

Hasil simulasi **bukan prediksi model** dan selalu diberi label "Simulasi, bukan prediksi model". Nilainya pseudo-acak tetapi deterministik per gambar, jadi gambar yang sama selalu memberi hasil simulasi yang sama.

### Riwayat desain

1. **v1:** halaman upload sederhana.
2. **v2:** desain lebih lengkap, tetapi dinilai pengguna **terlihat seperti "AI slop"** (hero besar, kartu berbayang, langkah bernomor 1–4, eyebrow huruf kapital).
3. **v3 (sekarang):** konsep "alat ukur", bukan landing page:
   - Halaman langsung dibuka pada alatnya, berupa satu bingkai bergaris tegas yang dibagi **Gambar | Hasil**.
   - Warna netral seperti kertas dan tinta. Hijau hanya untuk Asli, oranye untuk AI. Tanpa bayangan dan sudut tumpul.
   - Font: Archivo untuk judul dan kata hasil, Schibsted Grotesk untuk teks, Spline Sans Mono untuk angka.
   - **Animasi pemikselan** 160 → 96 → 64 → 48 → 32 px saat tombol Periksa ditekan, memperlihatkan pra-pemrosesan model.
   - Tombol tampilan **Asli / 32×32**.
   - Hasil ditampilkan sebagai kata besar **ASLI** atau **AI**, bar 40 segmen peluang Asli vs AI, dan tabel: peluang, keyakinan (tinggi ≥90%, sedang ≥70%, rendah), model, dan waktu inferensi.
   - **12 contoh data uji dengan label tersembunyi ("?").** Label terbuka setelah gambar diperiksa, lengkap dengan keterangan benar atau keliru. Cocok untuk demo saat sidang.
   - Bagian "Tentang model" berisi spesifikasi sebenarnya dan batasan model.
   - Upload bisa lewat drag-and-drop, pilih file, atau paste (Ctrl+V). Format JPG/PNG/WEBP, maksimal 10 MB. Pintasan keyboard: Enter untuk memeriksa, Esc untuk mengganti gambar. Mendukung mode terang dan gelap, tampilan HP, serta riwayat 8 pemeriksaan terakhir.

### API (kontrak untuk tahap 5)

- `GET /api/status` mengembalikan `{"model_ready": false, "model": null}`. Variabel `MODEL` dan `MODEL_NAME` di `web/app.py` diisi pada tahap 5.
- `POST /api/predict` (multipart, field `image`) **saat ini selalu mengembalikan 503**. Pada tahap 5, endpoint ini harus mengembalikan:

```json
{
  "label": "FAKE",
  "confidence": 0.973,
  "probabilities": {"FAKE": 0.973, "REAL": 0.027},
  "model": "LightweightCNN",
  "inference_ms": 12.4
}
```

Frontend otomatis beralih ke prediksi sungguhan begitu `/api/status` mengembalikan `model_ready: true`.

**Catatan untuk tahap 5:** pra-pemrosesan di server harus sama dengan transform evaluasi model terpilih (`get_transforms(model_name, is_training=False)` di `data_loader.py`). Gambar perlu di-`convert("RGB")` dan di-resize menjadi persegi (rasio dipaksa). Indeks keluaran: `0 = FAKE`, `1 = REAL`.

## 10. GitHub

- Repo: https://github.com/opallep/CIFAKE-DETECTION (branch `main`)
- Identitas git lokal (hanya untuk repo ini): `opallep` / `alonegtw499@gmail.com`
- Commit yang sudah di-push:
  - `bf282a3` Pipeline CIFAKE detection + website CekCitra
  - `d196144` Hasil evaluasi LightweightCNN pada test set (acc 94,75%, AUC 0,9921)
- Tidak di-commit: `data/`, `models/*.pt`, `results/*_log.txt`. **Simpan cadangan `models/LightweightCNN_best.pt` secara terpisah**, misalnya di Google Drive, karena file ini tidak ada di GitHub.

## 11. Masalah yang ditemui dan solusinya

| Masalah | Solusi |
|---|---|
| Path dataset di config salah (`data/cifake`) | Diganti menjadi `data/archive` |
| PowerShell 5.1 menganggap stderr (warning PyTorch atau progress bar tqdm) sebagai error, sehingga perintah lanjutan setelah `;if ($?)` tidak berjalan dan evaluasi tidak berjalan otomatis setelah training | Jalankan `evaluate.py` sebagai perintah terpisah |
| Warning `ReduceLROnPlateau(verbose=True)` deprecated di PyTorch 2.6 | Tidak berbahaya. Bisa dihapus dari `train.py` jika ingin log bersih |
| Pembacaan data lambat (sekitar 520 gambar/detik) | Akibat membaca 120 ribu file kecil. Jauh lebih cepat setelah file masuk cache. Opsional: cache dataset ke RAM (sekitar 300 MB untuk 32×32) |
| `git` tidak dikenali di terminal VS Code setelah Git dipasang | Restart VS Code, atau jalankan `$env:Path = "C:\Program Files\Git\cmd;" + $env:Path` |
| Push gagal karena belum login | Login GitHub lewat browser saat `git push` pertama. Kredensial tersimpan di Git Credential Manager |

## 12. Langkah berikutnya

### Tahap 2: ResNet-50
```
python train.py --model ResNet50
python evaluate.py --model ResNet50
```
- Input 224×224, batch 32, Adam dengan learning rate berbeda (pretrained 1e-4, head 1e-3).
- Perkiraan **10–20 menit per epoch** di RTX 4060. Laptop jangan di-sleep.
- File `models/ResNet50_best.pt` sekitar 90 MB dan **tidak boleh di-commit**.

### Tahap 3: EfficientNetV2-B0
```
python train.py --model EfficientNetV2B0
python evaluate.py --model EfficientNetV2B0
```
Pertama kali dijalankan, `timm` perlu internet untuk mengunduh bobot pretrained.

### Tahap 4: perbandingan
```
python compare_models.py
```
Opsional: `test_compression_robustness.py` (ketahanan terhadap kompresi JPEG) dan `grad_cam_viz.py` (visualisasi Grad-CAM) sebagai bahan tambahan Bab 4.

### Tahap 5 dan 6
Hubungkan model terbaik ke `web/app.py` (lihat bagian 9), lalu uji dengan foto asli dari luar dataset. Catat sebagai **batasan penelitian**: model dilatih dengan gambar 32×32 dari satu generator (Stable Diffusion v1.4), sehingga akurasi pada foto beresolusi tinggi atau dari generator lain kemungkinan lebih rendah.

## 13. Catatan untuk sesi Claude Code berikutnya

- Baca file ini dan `CLAUDE.md` terlebih dahulu.
- Pengguna berkomunikasi dalam **Bahasa Indonesia** dan ingin model dilatih **satu per satu**, dengan penjelasan langkah sebelum dieksekusi serta **update berkala** selama training.
- Pengguna tidak menyukai desain web yang terlihat generik atau "AI slop". Pertahankan arah desain v3.
- Setiap tahap selesai: commit dan push hasilnya (metrik, plot, history), kecuali file `.pt`.
