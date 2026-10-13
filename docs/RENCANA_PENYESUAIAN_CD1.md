# Rencana Penyesuaian Kode terhadap Dokumen CD-1

**Proyek:** Rancang Bangun Aplikasi Deteksi Citra AI-Generated dan Citra Asli Berbasis Deep Learning  
**Acuan:** Dokumen CD-1 (FTE-CD-1, revisi 0)  
**Status:** Usulan, **belum ada perubahan kode** yang dilakukan

---

## 1. Kondisi website saat ini

| Link | Status |
|---|---|
| https://opallep.github.io/CIFAKE-DETECTION/ | Online dan bisa dibuka dari mana saja, tetapi **hasilnya masih simulasi** karena GitHub Pages tidak bisa menjalankan model Python |
| http://127.0.0.1:5000 (laptop) atau `http://<IP-laptop>:5000` (HP di Wi-Fi yang sama) | Memakai model EfficientNetV2-B0 sungguhan, tetapi hanya berjalan saat server dinyalakan dengan `python web/app.py --lan` |

Agar prediksi sungguhan dapat diakses masyarakat (Tujuan 3 CD-1), server Python perlu di-deploy ke hosting yang mendukung Python (poin H2).

---

## 2. Kesenjangan antara kode sekarang dan CD-1

| Aspek di CD-1 | Kondisi kode sekarang |
|---|---|
| Dataset gabungan **multi-generator** (Stable Diffusion, Midjourney, DALL·E, Flux, GAN) | Hanya CIFAKE (1 generator: Stable Diffusion v1.4) |
| **Berbagai jenis citra**: manusia, objek, pemandangan, foto yang diedit | Hanya 10 kategori hewan dan kendaraan |
| Foto asli **beresolusi tinggi** | Seluruh alur dikunci ke 32×32 (data loader, server, tampilan web) |
| **Tahan penyuntingan dan kompresi** | Augmentasi hanya flip dan crop; uji ketahanan hanya kompresi JPEG |
| **Generalisasi lintas generator** | Belum ada pengujian pada generator yang tidak dilihat saat training |
| **Visualisasi area yang memengaruhi keputusan** (Tujuan 3) | `grad_cam_viz.py` sudah ada, tetapi belum tampil di website |
| **Mekanisme pembaruan model modular** (Tujuan 4) | Belum ada; model hanya diganti lewat variabel lingkungan |
| **Efisiensi** (parameter, waktu inferensi, GFLOPs) | Parameter dan waktu inferensi sudah diukur; GFLOPs belum |

---

## 3. Daftar perubahan

### A. Menjaga agar yang sudah ada tidak terganggu (dikerjakan pertama)

| No | File | Perubahan |
|---|---|---|
| A1 | `config.py` | Tambah pilihan dataset aktif (`cifake` / `gabungan`). Hasil dan model disimpan per dataset: `results/cifake/…`, `results/gabungan/…`, `models/cifake/…`, dan seterusnya |
| A2 | `results/`, `models/` | Hasil CIFAKE yang sekarang dipindah ke subfolder `cifake/` dan tetap bisa dipakai sebagai **eksperimen awal (baseline)** di laporan |
| A3 | `web/app.py` | Website tetap memakai model CIFAKE sampai model dari dataset baru selesai dilatih, lalu tinggal diganti |

### B. Dataset gabungan

| No | File | Perubahan |
|---|---|---|
| B1 | baru: `prepare_dataset.py` | Menggabungkan beberapa sumber data ke `data/gabungan/`: menghapus duplikat, menyamakan format, dan **memperkecil citra besar sekali di awal** (misalnya sisi pendek 512 px) supaya training tidak lambat karena decoding JPEG besar |
| B2 | baru: `data/gabungan/metadata.csv` | Satu baris per citra: path, label REAL/FAKE, **generator**, **jenis citra** (manusia, objek, pemandangan, ilustrasi), status diedit/difilter, dan split. Dibutuhkan untuk analisis per generator dan per jenis citra |
| B3 | `prepare_dataset.py` | Split train/val/test **tetap** (disimpan, tidak diacak ulang setiap run) dan berimbang per generator. **1–2 generator disisihkan sepenuhnya** sebagai data uji "generator tak dikenal" untuk membuktikan celah (a) di CD-1 |
| B4 | `prepare_dataset.py` | Membuat subset **foto asli yang diberi filter, efek, dan kompresi gaya media sosial** untuk menguji tuduhan palsu (aspek hukum dan sosial) |

### C. Pemuatan data dan pra-pemrosesan

| No | File | Perubahan |
|---|---|---|
| C1 | `data_loader.py` | Ganti `ImageFolder` + `random_split` dengan Dataset yang membaca `metadata.csv`, sehingga info generator dan jenis citra ikut terbawa ke evaluasi |
| C2 | `data_loader.py`, `config.py` | **Hapus asumsi 32×32.** Semua model memakai input **224×224**: random crop saat training, center crop saat evaluasi. Crop dipilih karena resize/interpolasi terbukti memengaruhi deteksi [9] |
| C3 | `data_loader.py` | **Augmentasi ketahanan**: kompresi JPEG acak (Q 30–100), blur, noise, resize turun-naik, serta filter warna/kecerahan ringan, masing-masing dengan peluang tertentu. Sebelumnya ColorJitter sengaja dibuang untuk CIFAKE, tetapi CD-1 menuntut model tahan terhadap foto yang difilter, jadi augmentasi ini perlu kembali dalam dosis ringan |

### D. Model dan training

| No | File | Perubahan |
|---|---|---|
| D1 | `models.py` | LightweightCNN memakai input 224. Arsitektur 8 lapisan konvolusinya tetap (Global Average Pooling bisa menerima ukuran berapa pun), mungkin dengan satu pooling tambahan agar tidak terlalu berat. ResNet-50 dan EfficientNetV2-B0 tidak berubah |
| D2 | `config.py` | Batch size LightweightCNN diturunkan (misalnya 64); jumlah epoch dan patience disesuaikan dengan ukuran dataset baru |
| D3 | `train.py` | Menyimpan metadata run (dataset, generator yang dipakai, seed, tanggal) bersama checkpoint |

### E. Pengujian baru sesuai CD-1

| No | File | Pengujian |
|---|---|---|
| E1 | baru: `evaluate_per_source.py` (pengganti `evaluate_per_category.py`) | Akurasi **per generator** dan **per jenis citra** |
| E2 | baru / perluasan | **Uji lintas generator**: akurasi pada generator yang disisihkan di B3 |
| E3 | `test_robustness.py` (perluasan uji kompresi) | Kompresi JPEG, resize, blur, noise, crop, dan filter, termasuk kombinasi "gaya media sosial" (serangan seperti di [11]) |
| E4 | baru | **Tingkat tuduhan palsu** (REAL → FAKE) pada foto asli yang diedit secara wajar |
| E5 | `compare_models.py` | Tambah **GFLOPs** dan ukuran file model untuk aspek efisiensi |

### F. Grad-CAM di website (Tujuan 3)

| No | File | Perubahan |
|---|---|---|
| F1 | `web/app.py` | `/api/predict` juga mengembalikan **heatmap Grad-CAM** (gambar overlay) |
| F2 | `web/index.html`, `web/assets/app.js` | Heatmap ditampilkan di panel gambar dengan tombol tampilan **Asli / Area berpengaruh**, menggantikan tampilan "32×32" yang sekarang |
| F3 | `grad_cam_viz.py` | Disesuaikan dengan dataset baru untuk gambar di laporan |

### G. Mekanisme pembaruan model (Tujuan 4)

| No | File | Perubahan |
|---|---|---|
| G1 | baru: `models/registry.json` | Daftar versi model: nama, versi, tanggal, generator yang dicakup, metrik uji, dan penanda **model aktif** |
| G2 | `web/app.py` | Server membaca model aktif dari registry; model bisa dimuat ulang **tanpa mengubah frontend**. Versi model ditampilkan di website |
| G3 | baru: `update_model.py` | Fine-tuning model aktif dengan citra dari generator baru, lalu evaluasi. Model baru **hanya dipromosikan menjadi aktif kalau metriknya tidak turun** dibanding versi lama |

### H. Website, deploy, dan dokumentasi

| No | File | Perubahan |
|---|---|---|
| H1 | `web/` | Teks yang menyebut "CIFAKE / 32×32 / 10 kategori" diganti. Contoh gambar diganti dengan citra dari data uji baru (beberapa generator dan jenis citra) |
| H2 | deploy | Deploy server Python ke hosting gratis (misalnya Hugging Face Spaces) agar prediksi sungguhan bisa diakses publik |
| H3 | `README.md`, `run_all.py`, `CLAUDE.md`, `docs/` | Disesuaikan dengan dataset dan alur baru |

---

## 4. Keputusan yang dibutuhkan sebelum mulai

1. **Sumber dataset.** CD-1 mengutip **GenImage** [5] (multi-generator, antara lain Midjourney, berbagai versi Stable Diffusion, ADM, GLIDE, dan BigGAN). Ukuran totalnya ratusan GB, jadi kemungkinan hanya sebagian yang dipakai. Sumber data untuk generator terbaru (Flux, DALL·E 3, Midjourney v6) juga perlu ditentukan.
2. **Ukuran target dataset.** Sebagai gambaran, dengan RTX 4060 dan input 224, 100.000–150.000 citra membutuhkan sekitar 1–2 jam training per model. Ruang disk tidak menjadi kendala (tersedia sekitar 565 GB).
3. **Apakah CIFAKE tetap ikut** sebagai salah satu sumber data, atau hanya dipakai sebagai eksperimen awal.
4. **Generator yang disisihkan** untuk uji lintas generator (misalnya Midjourney atau Flux).
5. **Hosting deploy:** Hugging Face Spaces (gratis) atau server dari kampus.

---

## 5. Urutan pengerjaan

A → B → C → D (training ulang 3 model, satu per satu) → E → F → G → H.

Bagian A dan C–G dapat dikerjakan dan diuji lebih dulu dengan CIFAKE **sambil dataset baru dikumpulkan**, sehingga training dapat langsung dijalankan begitu dataset gabungan siap.
