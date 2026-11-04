# CLAUDE.md

Tugas akhir (Capstone Design): **Aplikasi Deteksi Citra AI Generated Berbasis Deeplearning Menggunakan Dataset Multi-Generator**.
Acuan utama: `D:\TUGAS AKHIR\CD 1-3\DIRECTORY TO ANTI GRAVITY\PROJECT_BRIEF_CD2.md` (menggantikan brief CD-1). Spesifikasi Bagian 6 brief = acceptance criteria kode.

## Status
- **Tahap sekarang (CD-2):** pipeline multi-generator di `src/` + verifikasi di `tests/verification/`. Dataset GenImage/ArtiFact **belum diunduh**; `configs/data.yaml` masih template.
- **Baseline lama (CIFAKE 32×32, 1 generator):** skrip di root (`train.py`, `evaluate.py`, `models.py`, ...), hasil di `results/` dan `docs/LAPORAN_PERBANDINGAN_MODEL.md`. Riwayat sesi: `docs/LAPORAN_SESI_*.md`. Jangan dihapus, tetapi CIFAKE bukan dataset utama lagi. Website `web/` masih memakai model CIFAKE.
- Keputusan default yang belum disetujui pembimbing: `docs/decisions.md`. Status verifikasi: `docs/verification_report.md`.

## Aturan kerja (brief Bagian 11)
- Komunikasi dalam Bahasa Indonesia; nama variabel dan kode dalam bahasa Inggris.
- Pakai istilah "permasalahan utama", bukan "pain point".
- **Jangan membuat angka hasil fiktif.** Semua angka dari eksekusi nyata. Hasil `tests/smoke` (data sintetis) tidak boleh dipakai sebagai hasil.
- Cegah kebocoran: split per generator, dedup lintas split, generator unseen tidak dipakai untuk tuning/pemilihan checkpoint.
- Praproses training = inferensi: selalu lewat `src/preprocess.py`.
- Setiap skrip mencantumkan kode spesifikasi (T1, H5, ...). Setelah pengujian jalankan `python -m tests.verification.build_report`.
- Jangan klaim "semua foto"/"semua generator AI"; tampilkan disclaimer H4 (`src.common.DISCLAIMER`).
- Label: `ai` = 1 (positif, termasuk foto manusia diedit AI), `real` = 0.
- Latih model satu per satu, jelaskan langkah sebelum eksekusi, beri update berkala selama training.
- Setelah tiap tahap commit + push ke `origin main`. Jangan commit `data/`, `models/*.pt`, `experiments/**/*.pt`.
- Desain website: pertahankan arah "alat ukur"; hindari tampilan generik.

## Alur CD-2
```
python -m src.data.build_manifest --config configs/data.yaml --data-root <folder dataset>   # T1, H7
python -m src.data.split_by_generator                                                      # T2 (dedup + split)
python -m src.data.harmonize                                                               # T3 -> data/manifests/manifest.csv
python -m tests.verification.verify_t1_h7_inventory
python -m tests.verification.verify_t2_leakage --rehash
python -m tests.verification.verify_t3_metadata_bias
python -m src.run_experiments --evaluate        # T4: 3 arsitektur x (base, se) x 3 seed + evaluasi per run
python -m src.summarize_experiments             # rekap + T4, T5, T6, H5, H6, E1, B2
python -m src.profile_models                    # E1, E4
python -m tests.verification.build_report       # docs/verification_report.md
python -m unittest tests.test_metrics           # rumus Bagian 7
```
Uji cepat tanpa dataset asli: `python -m tests.smoke.make_synthetic --out <folder>`, lalu jalankan alur di atas dengan `--config <folder>/data.yaml` atau `<folder>/train.yaml` dan env `CD2_REPORT_DIR=<folder>/reports` (agar laporan proyek tidak tertimpa).

## Lingkungan
Windows 11, PowerShell 5.1, Python 3.12, PyTorch 2.6 + CUDA, timm 1.0, RTX 4060 8 GB. Git di `C:\Program Files\Git\cmd` (tambahkan ke PATH jika `git` tidak dikenali). Di PowerShell 5.1 stderr tqdm membuat `$?` false; jalankan evaluasi terpisah dari training.
