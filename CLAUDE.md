# CLAUDE.md

Tugas akhir: membandingkan 3 CNN (LightweightCNN, ResNet50, EfficientNetV2B0) untuk mendeteksi citra AI vs asli pada dataset CIFAKE, lalu memakai model terbaik di website CekCitra (`web/`).

Riwayat dan status lengkap: lihat `docs/LAPORAN_SESI_*.md` (terbaru = nomor terbesar).
Hasil eksperimen lengkap ketiga model: `docs/LAPORAN_PERBANDINGAN_MODEL.md`. Model terpilih: EfficientNetV2B0 (akurasi uji 98,52%).

## Cara kerja dengan pengguna
- Komunikasi dalam Bahasa Indonesia.
- Latih model satu per satu, jelaskan langkahnya sebelum dieksekusi, dan beri update berkala selama training.
- Setelah tiap tahap: commit dan push hasil (metrik, plot, history) ke `origin main`. Jangan commit `data/` atau `models/*.pt`.
- Desain website: pertahankan arah "alat ukur" yang sekarang; hindari tampilan generik.

## Perintah
- Training: `python train.py --model <LightweightCNN|ResNet50|EfficientNetV2B0>`
- Evaluasi: `python evaluate.py --model <nama>` (jalankan terpisah dari training; di PowerShell 5.1 stderr tqdm membuat `$?` false)
- Perbandingan: `python compare_models.py`
- Website: `python web/app.py` → http://127.0.0.1:5000

## Lingkungan
Windows 11, PowerShell 5.1, Python 3.12, PyTorch 2.6 + CUDA, RTX 4060 8 GB. Dataset di `data/archive/{train,test}/{REAL,FAKE}` (32×32 JPG). Git ada di `C:\Program Files\Git\cmd` (tambahkan ke PATH jika `git` tidak dikenali).
