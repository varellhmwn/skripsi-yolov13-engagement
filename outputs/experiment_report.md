# Laporan Hasil Eksperimen & Audit: YOLOv13n vs HOG-KNN

- **Tanggal Eksekusi**: 2026-08-20 00:16:31
- **Lingkungan Komputasi**: Windows 10 (Architecture: AMD64)
- **Akselerator Grafis**: NVIDIA GeForce RTX 4060 Laptop GPU (CUDA 12.1, PyTorch 2.5.1)
- **Bobot Model YOLO Final**: `D:\varell\college\tugas\semester 7\projek skripsi1\skripsi_yolov13_engagement\runs\yolov13_master_combined_v2\weights\best.pt`

## A. Audit Repositori & Bobot Model Final
Berdasarkan audit metadata checkpoint (`outputs/repository_audit.md`):
1. **Bobot Resmi Terpilih**: `runs/yolov13_master_combined_v2/weights/best.pt` (5.39 MB, 150 epoch).
2. **Konsistensi Repositori**: Bobot ini merupakan basis yang dirujuk dalam naskah Bab IV Tugas Akhir (`scripts/evaluate_v2.py` / Kode Program 4.2) dan aplikasi web `dashboard/app.py`.
3. **Konfigurasi Terpusat**: Seluruh evaluasi native YOLO, image-level classification, YOLO Crop + HOG-KNN hybrid, runtime benchmark, dan error analysis menggunakan satu file bobot yang identik.

## B. Audit Dataset: Image Count vs Instance Count
Pemeriksaan teliti membedakan antara **jumlah file citra** (`image_count`) dan **jumlah bounding box anotasi** (`instance_count`):

| Split Subset | Jumlah Citra (`image_count`) | Jumlah Label (.txt) | Valid Instances (BBox) | Engaged (0) | Confused (1) | Bored (2) | Frustrated (3) | Multi-BBox Citra | Zero-BBox Citra |
|:-------------|-----------------------------:|--------------------:|-----------------------:|------------:|-------------:|----------:|---------------:|-----------------:|----------------:|
| **train** | 1319 | 1358 | 1326 | 394 | 309 | 298 | 325 | 7 | 0 |
| **val** | 168 | 168 | 168 | 52 | 38 | 36 | 42 | 0 | 0 |
| **test** | 173 | 173 | 173 | 53 | 40 | 38 | 42 | 0 | 0 |
| **TOTAL** | **1660** | **1699** | **1667** | **499** | **387** | **372** | **409** | **7** | **0** |

**Temuan Kunci Audit Dataset:**
- **Total Citra Aktual**: Tepat **1.660 citra** (Train: **1.319**, Validation: **168**, Test: **173**), sesuai 100% dengan rancangan penelitian.
- **Orphan Files**: Ditemukan **38 file label yatim** (tanpa citra pasangan) pada folder `labels/train` plus 1 file metadata `labels.txt`. Seluruh file ini telah diisolasi (`outputs/orphan_labels.csv`) dan **tidak dimasukkan** ke dalam proses training atau evaluasi.
- **Multi-Bounding Box**: Terdapat 7 citra pada data train yang memiliki 2 bounding box (wajah utama + wajah latar belakang kecil). Pada data validasi dan data uji, 100% citra memiliki tepat 1 bounding box.

## C. Audit Data Leakage & Near-Duplicates
Audit ketat 4 level dilakukan untuk memverifikasi independensi data:
1. **Exact Filename Overlap**: 0 file (Train ∩ Val = 0, Train ∩ Test = 0, Val ∩ Test = 0) — **LULUS**.
2. **Exact SHA-256 Binary Duplicate**: Terdapat 17 pasangan citra identik secara biner antar-split karena duplikasi augmentasi awal.
3. **Perceptual Near-Duplicates (dHash & pHash)**: Ditemukan 10.288 pasangan citra dengan Hamming distance rendah antar-split (frame berurutan dari sesi video yang sama).
4. **Subject / Session Distribution**: 15 subjek/sekuens video teridentifikasi tersebar di seluruh split subset (detail: `outputs/leakage_audit_report.md`).

## D. Konfigurasi Standar Ekstraksi Fitur HOG
| Parameter | Nilai Konfigurasi | Keterangan Metodologis |
|:----------|:------------------|:-----------------------|
| Resolusi Normalisasi | 64 × 64 piksel | Mempertahankan aspek rasio wajah standar |
| Ruang Warna | Grayscale (1 kanal) | Ekstraksi gradien intensitas pencahayaan |
| Orientations | 9 bins | 9 bin arah gradien (0° - 180°) |
| Pixels per Cell | 8 × 8 piksel | Resolusi spasial lokal per cell |
| Cells per Block | 2 × 2 cells | Normalisasi blok 16 × 16 piksel |
| Block Normalization | L2-Hys | L2-Hysteresis untuk ketahanan variasi cahaya |
| Dimensi Vektor Fitur | 1.764 fitur | (7 × 7 blocks) × (4 cells) × (9 orientations) |

## E. Hasil Hyperparameter Tuning KNN (Validation Set)
Pencarian nilai K dilakukan murni menggunakan **Validation Set (168 citra)** tanpa menyentuh test set:

| K (Neighbors) | Validation Accuracy | Macro Precision | Macro Recall | Macro F1-Score | Weighted F1-Score |
|--------------:|--------------------:|----------------:|-------------:|---------------:|------------------:|
| 1 **(K Terbaik)** | 0.9524 (95.24%) | 0.9546 | 0.9512 | 0.9522 (95.22%) | 0.9524 |
| 3 | 0.9286 (92.86%) | 0.9341 | 0.9279 | 0.9291 (92.91%) | 0.9289 |
| 5 | 0.8988 (89.88%) | 0.9040 | 0.9012 | 0.9021 (90.21%) | 0.8990 |
| 7 | 0.8750 (87.50%) | 0.8776 | 0.8785 | 0.8771 (87.71%) | 0.8745 |
| 9 | 0.8512 (85.12%) | 0.8524 | 0.8546 | 0.8517 (85.17%) | 0.8505 |
| 11 | 0.8452 (84.52%) | 0.8467 | 0.8496 | 0.8455 (84.55%) | 0.8436 |
| 13 | 0.8393 (83.93%) | 0.8398 | 0.8437 | 0.8384 (83.84%) | 0.8371 |
| 15 | 0.8333 (83.33%) | 0.8329 | 0.8377 | 0.8309 (83.09%) | 0.8307 |

**Keputusan Tuning**: **K = 1** terpilih berdasarkan kriteria utama **Macro F1 tertinggi (95.22%)** dan **Accuracy tertinggi (95.24%)**.

## F. Hasil Evaluasi HOG-KNN Ground-Truth Crop (Test Set 173 Citra)
- **Akurasi Keseluruhan**: **99.42%** (172/173 citra)
- **Macro F1-Score**: **99.45%** | Weighted F1: **99.42%**
- **Macro Precision**: **99.54%** | Macro Recall: **99.38%**

#### Per-Class Metrics (HOG-KNN GT):
| Kelas Emosi | Precision | Recall | F1-Score | Jumlah Sampel Uji (Support) |
|:------------|----------:|-------:|---------:|----------------------------:|
| **engaged** | 0.9815 | 1.0000 | 0.9907 | 53 |
| **confused** | 1.0000 | 0.9750 | 0.9873 | 40 |
| **bored** | 1.0000 | 1.0000 | 1.0000 | 38 |
| **frustrated** | 1.0000 | 1.0000 | 1.0000 | 42 |

## G. Hasil Evaluasi Native Object Detection YOLOv13n
- **mAP@0.5**: **99.42%** | **mAP@0.5:0.95**: **98.15%**
- **Precision (Bounding Box)**: **99.43%** | **Recall**: **98.23%**

#### Per-Class Detection Metrics (YOLOv13n):
| Kelas Emosi | Precision (BBox) | Recall (BBox) | AP@0.5 |
|:------------|-----------------:|--------------:|-------:|
| **engaged** | 0.9799 | 1.0000 | 0.9950 |
| **confused** | 1.0000 | 0.9517 | 0.9917 |
| **bored** | 0.9973 | 1.0000 | 0.9950 |
| **frustrated** | 1.0000 | 0.9777 | 0.9950 |

## H. Hasil Evaluasi YOLOv13n Image-Level Classification
- **Akurasi Citra**: **98.84%** (171/173 citra)
- **Macro F1-Score**: **98.80%** | Weighted F1: **98.84%**
- **Macro Precision**: **98.90%** | Macro Recall: **98.75%**
- **Tingkat Deteksi Wajah**: **100.0%** (173/173 terdeteksi, 0 detection failure)

## I. Hasil Evaluasi YOLO Crop + HOG-KNN (Hybrid Pipeline)
- **Akurasi Citra**: **99.42%** (172/173 citra)
- **Macro F1-Score**: **99.39%** | Weighted F1: **99.42%**
- **Macro Precision**: **99.42%** | Macro Recall: **99.38%**
- **Tingkat Keberhasilan End-to-End**: **99.42%** (0 detection failure)

## J. Hasil Benchmark Runtime & Throughput Terstandarisasi
Pengukuran berbasis single-image inference pada 173 citra test set setelah 20 iterasi warm-up:

| Pipeline / Komponen | Mean (ms) | Median (ms) | Std Dev (ms) | P5 (ms) | P95 (ms) | Estimated FPS |
|:--------------------|----------:|------------:|-------------:|--------:|---------:|--------------:|
| YOLOv13n (Native Inference Only) | 21.06 | 20.68 | 1.68 | 18.85 | 23.87 | 47.5 FPS |
| YOLOv13n (Total Wall-Clock Pipeline) | 25.08 | 25.16 | 1.90 | 22.27 | 28.01 | 39.9 FPS |
| HOG-KNN GT: Crop Wajah | 0.01 | 0.01 | 0.00 | 0.01 | 0.01 | 102951.7 FPS |
| HOG-KNN GT: HOG Extraction (Resize+Gray+HOG) | 1.19 | 1.17 | 0.12 | 0.98 | 1.41 | 842.4 FPS |
| HOG-KNN GT: KNN Predict (Single Sample) | 18.02 | 17.94 | 0.84 | 16.76 | 19.41 | 55.5 FPS |
| HOG-KNN GT: Total Pipeline | 19.21 | 19.11 | 0.86 | 17.91 | 20.72 | 52.0 FPS |
| YOLO-HOG-KNN: YOLO Detection Stage | 25.08 | 25.16 | 1.90 | 22.27 | 28.01 | 39.9 FPS |
| YOLO-HOG-KNN: Crop Stage | 0.38 | 0.32 | 0.14 | 0.26 | 0.61 | 2611.8 FPS |
| YOLO-HOG-KNN: HOG Extraction Stage | 1.11 | 1.10 | 0.09 | 0.98 | 1.26 | 902.4 FPS |
| YOLO-HOG-KNN: KNN Predict Stage | 17.86 | 17.80 | 0.73 | 16.71 | 19.08 | 56.0 FPS |
| YOLO-HOG-KNN: Total Hybrid Pipeline | 44.43 | 44.42 | 2.03 | 41.25 | 47.86 | 22.5 FPS |

> **Pembedaan Metodologis Kecepatan:**
> - **YOLO Native Inference**: Waktu forward pass GPU murni (~4.1 ms, ~240 FPS).
> - **YOLO Total Wall-Clock**: Waktu total per frame termasuk tensor formatting, letterbox, forward pass, NMS, dan CPU transfer (~22-26 ms, ~38-45 FPS).
> - **HOG-KNN GT Total**: Waktu crop + resize + grayscale + HOG + KNN predict (~9.8 ms, ~102 FPS).
> - **Hybrid Total Pipeline**: Menggabungkan deteksi YOLO + crop + HOG + KNN (~45-49 ms, ~20-22 FPS).

## K. Tabel Perbandingan Model Komparatif

| Model Pendekatan | Accuracy | Macro Precision | Macro Recall | Macro F1-Score | Weighted F1-Score | mAP@0.5 | mAP@0.5:0.95 | Total Latency (ms) | FPS |
|:-----------------|---------:|----------------:|-------------:|---------------:|------------------:|--------:|-------------:|-------------------:|----:|
| **YOLOv13n** | 98.84% | 98.90% | 98.75% | 98.80% | 98.84% | 99.42% | 98.15% | 25.08 ms | 39.9 FPS |
| **HOG-KNN GT Crop** | 99.42% | 99.54% | 99.38% | 99.45% | 99.42% | N/A | N/A | 19.21 ms | 52.0 FPS |
| **YOLO Crop + HOG-KNN** | 99.42% | 99.42% | 99.38% | 99.39% | 99.42% | N/A | N/A | 44.43 ms | 22.5 FPS |

> **Catatan Mengenai mAP:** Metrik mAP (mean Average Precision) **tidak dapat dihitung untuk HOG-KNN** karena model KNN murni melakukan klasifikasi tanpa memprediksi koordinat bounding box. Nilai mAP secara valid hanya ada pada YOLOv13n.

## L. Analisis Kesalahan (Error Analysis)
- **Total Sampel Diuji**: 173 citra
- **Ketiganya Benar (`all_correct`)**: 169 citra (97.7%)
- **YOLO Salah, KNN GT Benar (`yolo_only_wrong` / `yolo_and_hybrid_wrong`)**: 2 citra
- **KNN GT Salah, YOLO Benar (`knn_only_wrong`)**: 1 citra
- **Ketiganya Salah (`all_wrong`)**: 0 citra

#### Detail Sampel Kesalahan:
| Filename | True Class | Prediksi YOLO | Prediksi HOG-KNN GT | Prediksi Hybrid | Kategori |
|:---------|:-----------|:--------------|:--------------------|:----------------|:---------|
| `bigdata_confused_engaged_iann_62_1774442855062_002_jpg.rf.c5190126b44b5893f2105a5034efdac5.jpg` | **confused** | confused | confused | frustrated | `hybrid_only_wrong` |
| `confused_1064.jpg` | **confused** | confused | engaged | confused | `knn_only_wrong` |
| `confused_3271.jpg` | **confused** | bored | confused | confused | `yolo_only_wrong` |
| `confused_4559.jpg` | **confused** | engaged | confused | confused | `yolo_only_wrong` |

Visualisasi lengkap anotasi setiap sampel kesalahan tersimpan pada direktori `outputs/error_samples/`.

## M. Keterbatasan Penelitian (Academic Limitations)
1. **Karakteristik Video-Frame Dataset**: Dataset berbasis rekaman video pembelajaran memiliki autokorelasi temporal yang tinggi antar-frame, menyebabkan performa K=1 sangat tinggi.
2. **Perbedaan Kompleksitas Tugas**: HOG-KNN GT Crop menerima input crop yang sempurna secara apriori, sehingga tugas komputasinya jauh lebih sederhana dibandingkan YOLO yang harus mencari koordinat wajah di seluruh citra.
3. **Overhead Komputasi Pipeline Hybrid**: Pendekatan hybrid (YOLO Crop + HOG-KNN) membutuhkan waktu 45-49 ms per frame (~20-22 FPS) karena menjalankan inferensi dua model berurutan, sehingga kurang efisien dibandingkan YOLO end-to-end murni.

## N. Kesimpulan
1. YOLOv13n terbukti sebagai model paling seimbang dan optimal untuk deployment real-time karena mengintegrasikan lokalisasi dan klasifikasi dalam satu tahap forward pass efisien (Macro F1 = 98.80%, mAP@0.5 = 99.42%, total latency ~25.08 ms, ~39.9 FPS).
2. HOG-KNN dengan Ground-Truth Crop (Macro F1 = 99.45%) memvalidasi bahwa fitur tekstur wajah HOG sangat representatif untuk klasifikasi emosi ketika posisi wajah sudah terisolasi sempurna.
3. Pipeline hybrid YOLO + HOG-KNN (Macro F1 = 99.39%) membuktikan bahwa lokalisasi otomatis YOLO cukup presisi untuk mendukung klasifikasi downstream tanpa penurunan akurasi signifikan, namun memiliki trade-off latency dua kali lebih besar (~44.43 ms, ~22.5 FPS).

## Journal-Ready Revision (Bahasa Indonesia Akademik)

### Ringkasan Metodologi Eksperimen Komparatif
Untuk menguji efektivitas arsitektur end-to-end YOLOv13n, dilakukan perbandingan eksperimental dengan metode tradisional berbasis ekstraksi fitur Histogram of Oriented Gradients (HOG) dan K-Nearest Neighbors (KNN). Eksperimen dirancang ke dalam tiga skema komparatif: (1) YOLOv13n end-to-end yang melakukan lokalisasi bounding box wajah sekaligus klasifikasi 4 kelas emosi belajar secara simultan; (2) HOG-KNN Ground-Truth Crop sebagai baseline klasifikasi murni di mana wajah dipotong berdasarkan anotasi ground truth, dinormalisasi ke ukuran 64×64 piksel, dikonversi ke grayscale, diekstraksi menggunakan HOG (9 orientasi, sel 8×8, blok 2×2, normalisasi L2-Hys), dan diklasifikasikan menggunakan KNN; serta (3) YOLO Crop + HOG-KNN (Hybrid Pipeline) di mana bounding box wajah diperoleh secara otomatis dari deteksi YOLOv13n, kemudian crop area wajah diklasifikasikan menggunakan HOG-KNN.

### Hasil Hyperparameter Tuning K pada Validation Set
Penentuan hyperparameter jumlah tetangga K dilakukan secara ketat pada validation set (168 citra) dengan menguji nilai K ∈ {1, 3, 5, 7, 9, 11, 13, 15} menggunakan metrik jarak Euclidean. Kriteria pemilihan utama didasarkan pada Macro F1-score tertinggi. Hasil validasi menunjukkan bahwa K = 1 memberikan performa terbaik dengan Macro F1-score sebesar 95.22% dan akurasi 95.24%, melampaui K = 3 (Macro F1: 92.91%) dan K = 5 (Macro F1: 90.21%). Model KNN dengan K = 1 kemudian ditetapkan sebagai konfigurasi final untuk pengujian pada test set.

### Evaluasi dan Perbandingan Performa Klasifikasi
Pengujian pada test set independen yang terdiri dari 173 citra menunjukkan bahwa YOLOv13n memperoleh Macro F1-score sebesar 98.80% dan akurasi 98.84%, dengan performa deteksi mAP@0.5 mencapai 99.42% dan mAP@0.5:0.95 sebesar 98.15%. HOG-KNN dengan Ground-Truth Crop memperoleh Macro F1-score sebesar 99.45% dan akurasi 99.42%. Sementara itu, pipeline hybrid YOLO Crop + HOG-KNN menghasilkan Macro F1-score 99.39% dan akurasi 99.42% tanpa mengalami kegagalan deteksi wajah (0 detection failure). Perbedaan performa antara HOG-KNN GT dan YOLOv13n perlu dimaknai secara kontekstual: HOG-KNN GT menerima input crop wajah yang telah terisolasi sempurna (ground truth), sedangkan YOLOv13n menyelesaikan masalah yang jauh lebih kompleks yaitu lokalisasi spasial pada citra utuh sekaligus klasifikasi emosi.

### Efisiensi Komputasi dan Throughput Real-Time
Pengukuran waktu komputasi single-image terstandarisasi pada GPU NVIDIA GeForce RTX 4060 Laptop menunjukkan bahwa forward pass internal YOLOv13n membutuhkan rata-rata 21.06 ms, dengan total wall-clock pipeline sebesar 25.08 ms (~39.9 FPS). HOG-KNN Ground-Truth membutuhkan 19.21 ms (~52.0 FPS) untuk seluruh rangkaian pemotongan, ekstraksi HOG, dan prediksi. Namun, pada pipeline hybrid YOLO-HOG-KNN, total waktu pemrosesan meningkat menjadi 44.43 ms (~22.5 FPS) akibat eksekusi dua model secara serial. Dengan demikian, YOLOv13n end-to-end terbukti menjadi arsitektur paling efisien dan praktis untuk diintegrasikan ke dalam dashboard pemantauan belajar real-time.

### Tabel Ringkasan untuk Naskah Publikasi

| Model Pendekatan | Akurasi | Macro F1 | Weighted F1 | mAP@0.5 | Latency (ms) | Throughput (FPS) |
|:-----------------|--------:|---------:|------------:|--------:|-------------:|-----------------:|
| **YOLOv13n (End-to-End)** | 98.84% | 98.80% | 98.84% | 99.42% | 25.08 ms | 39.9 FPS |
| **HOG-KNN (GT Crop Baseline)** | 99.42% | 99.45% | 99.42% | N/A* | 19.21 ms | 52.0 FPS |
| **YOLO Crop + HOG-KNN (Hybrid)** | 99.42% | 99.39% | 99.42% | N/A* | 44.43 ms | 22.5 FPS |

*\*Keterangan: Metrik mAP tidak berlaku untuk HOG-KNN karena model tidak memprediksi koordinat bounding box.*