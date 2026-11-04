# Catatan Keputusan Teknis (CD-2 → CD-3)

Keputusan default yang diambil saat implementasi karena butir brief masih **[TERBUKA]** atau **[USULAN]**
(brief Bagian 11 no. 9). Semua dapat diubah lewat `configs/*.yaml` setelah dibahas dengan pembimbing.

| ID | Keputusan | Alasan | Status |
|----|-----------|--------|--------|
| D-01 | CIFAKE tidak lagi menjadi dataset utama. Pipeline baru (`src/`) membaca `manifest.csv` multi-generator. Kode CIFAKE lama di root proyek tetap ada sebagai baseline. | Brief Bagian 8: CIFAKE hanya 1 generator 32×32 (permasalahan utama 1). | Diterapkan |
| D-02 | Extension SE = **satu blok SE pada peta fitur terakhir backbone, sebelum global pooling**, reduction ratio 16, sama untuk ketiga arsitektur. | Lokasi seragam membuat ablation membandingkan efek extension yang sama. **Catatan penting:** EfficientNetV2-B0 dan MobileNetV3 sudah memiliki SE internal di blok MBConv; konfigurasi "tanpa SE" berarti tanpa extension, bukan tanpa SE sama sekali. ResNet-50 tidak punya SE internal. Ini perlu disebut di CD-3. | Usulan, perlu persetujuan pembimbing |
| D-03 | Varian MobileNet = **MobileNetV3-Large** (`mobilenetv3_large_100`, timm). | Contoh di brief; pretrained ImageNet tersedia. Alternatif bila ingin varian tanpa SE internal: MobileNetV2 (ganti satu baris di `configs/train.yaml`). | Usulan |
| D-04 | Harmonisasi kedua kelas: sisi pendek 256 px (bicubic), center crop 256×256, encode ulang JPEG Q95 (tanpa subsampling chroma, tanpa EXIF). Training memakai random crop 224, evaluasi/server memakai center crop 224. | T3 dan T7: satu fungsi `src/preprocess.harmonize` dipakai offline dan oleh server. Encode ulang juga membuang EXIF (H1). | Usulan |
| D-05 | Pada uji T3, fitur kelulusan = format, lebar, tinggi, kualitas JPEG. Ukuran berkas dilaporkan terpisah sebagai informasi. | Setelah format, resolusi, dan kompresi seragam, ukuran berkas ditentukan isi citra (tekstur), sehingga ikut mengukur isi, bukan metadata. Bila pembimbing menganggap ukuran berkas termasuk "metadata", ubah `with_size` di skrip. | Usulan |
| D-06 | Deduplikasi sebelum split: sha256 identik atau dHash 64-bit dengan jarak ≤ `phash_max_distance` (default 0). Duplikat berlabel berbeda dibuang seluruhnya. | T2: tumpang tindih antar split = 0 dijamin sejak awal, bukan hanya diperiksa. | Diterapkan |
| D-07 | Split `split_manifest`: generator seen → train 80% / val 10% / test_seen 10% (stratified per generator, label, image_type); generator unseen → `test_unseen` saja. Foto asli dipasangkan ke generator lewat kolom `generator` agar akurasi per generator memuat kedua kelas. | T2, T5; checkpoint dan pemilihan konfigurasi hanya dari val (seen). | Diterapkan |
| D-08 | Pelatihan: AdamW, LR backbone 1e-4 / head+SE 1e-3, cosine schedule, 20 epoch, early stopping 5, AMP, seluruh backbone di-fine-tune. Augmentasi (flip, filter warna ringan, blur, JPEG Q60–100) diterapkan sama pada kedua kelas. | Dataset multi-generator jauh lebih besar dari CIFAKE; augmentasi JPEG ditujukan untuk T6. Hiperparameter belum di-tuning. | Usulan |
| D-09 | Target T5, T6, H5, H6 dinilai pada **rerata lintas seed** dari konfigurasi dengan rerata akurasi validasi tertinggi. | Pemilihan tidak boleh memakai test_unseen (T2). | Usulan |
| D-10 | Daftar generator di `configs/data.yaml` masih **template** (7 seen, 3 unseen). | Brief Bagian 12 no. 1 masih terbuka; perlu disesuaikan dengan hasil unduhan GenImage/ArtiFact. | Terbuka |

## Belum dikerjakan (tahap berikutnya)

- Server FastAPI + klien web baru (T7 jalur server, T8, T9, H1–H4, E2, E3, K1). `web/` lama masih memakai model CIFAKE.
- Skrip pelatihan ulang generator baru (K2) dan uji reproduksibilitas lingkungan bersih (K3).
- Pemetaan kelas ImageNet → `hewan`/`objek` untuk sumber GenImage (`image_type_rules`).
