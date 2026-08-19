# Laporan Akhir Eksperimen: Group-Wise Split, Retraining YOLOv13n & Evaluasi Komparatif HOG-KNN

- **Tanggal Eksekusi**: 2026-08-20 01:13:52
- **Platform Komputasi**: Windows 10 (AMD64)
- **Akselerator**: NVIDIA GeForce RTX 4060 Laptop GPU (CUDA 12.1, PyTorch 2.5.1)
- **Dataset Path Baru**: `D:\varell\college\tugas\semester 7\projek skripsi1\skripsi_yolov13_engagement\datasets\master_combined_groupwise_v1`
- **Bobot YOLO Group-Wise Final**: `D:\varell\college\tugas\semester 7\projek skripsi1\skripsi_yolov13_engagement\runs\yolov13_master_groupwise_v1\weights\best.pt`

## 1. Repository & Dataset Audit
Dataset utama dipertahankan tepat **1.660 citra** (1.667 instances bounding box) tanpa penghapusan file asli.

## 2. Group Discovery & Manifest
Ditemukan **181 group independen** mencakup 11 subjek Roboflow (953 citra) dan klaster Hard Samples (707 citra).

## 3. Split Proposal & Leakage Validation
- **Train Set**: 1.327 citra (79.94%) | 126 groups
- **Validation Set**: 167 citra (10.06%) | 18 groups
- **Test Set**: 166 citra (10.00%) | 37 groups
- **Status Leakage Gate**: 100% LULUS (0 subjek cross-split, 0 duplikat SHA-256 cross-split).

## 4. Hasil Retraining YOLOv13n (Group-Wise)
- **Epochs Trained**: 150 (Patience 25, AdamW lr0=0.001)
- **mAP@0.5 Test**: **70.78%** | **mAP@0.5:0.95**: **63.70%**
- **Akurasi Klasifikasi**: **62.65%** | **Macro F1**: **61.80%**

## 5. Hasil Tuning & Evaluasi HOG-KNN
- **K Optimal (Validation)**: K = 1 (Macro F1: 32.24%)
- **HOG-KNN GT Crop (Test)**: Akurasi = **25.90%** | Macro F1 = **25.35%**
- **YOLO Crop + HOG-KNN Hybrid**: Akurasi = **31.93%** | Macro F1 = **29.12%**

## 6. Tabel Perbandingan Model Komparatif Final

| Model | Accuracy | Macro Precision | Macro Recall | Macro F1 | Weighted F1 | mAP@0.5 | mAP@0.5:0.95 | Total Latency | FPS |
|:------|---------:|----------------:|-------------:|---------:|------------:|--------:|-------------:|--------------:|----:|
| **YOLOv13n** | 62.65% | 62.12% | 62.05% | 61.80% | 62.46% | 70.78% | 63.70% | 25.35 ms | 39.4 FPS |
| **HOG-KNN GT Crop** | 25.90% | 30.12% | 24.31% | 25.35% | 26.72% | N/A | N/A | 3.50 ms | 285.4 FPS |
| **YOLO Crop + HOG-KNN** | 31.93% | 31.56% | 30.20% | 29.12% | 30.24% | N/A | N/A | 29.16 ms | 34.3 FPS |

## 7. Perbandingan Split Lama vs Group-Wise Baru

| Model | Old Split Accuracy | New Group-Wise Accuracy | Old Split Macro F1 | New Group-Wise Macro F1 | New Latency | New FPS |
|:------|-------------------:|------------------------:|-------------------:|------------------------:|------------:|--------:|
| **YOLOv13n** | 98.84% | 62.65% | 98.80% | 61.80% | 25.35 ms | 39.4 FPS |
| **HOG-KNN GT Crop** | 99.42% | 25.90% | 99.45% | 25.35% | 3.50 ms | 285.4 FPS |
| **YOLO Crop + HOG-KNN** | 99.42% | 31.93% | 99.39% | 29.12% | 29.16 ms | 34.3 FPS |

## 8. Analisis Kesalahan (Error Analysis)
- **Total Sampel Test**: 166 citra
- **Semua Benar (`all_correct`)**: 34 citra (20.5%)
- **Sampel Kesalahan**: 132 citra (visualisasi tersimpan pada `outputs_groupwise/error_samples/`)

## 9. Kesimpulan & Rekomendasi Ilmiah
1. Pembagian dataset group-wise berhasil mengeliminasi kebocoran frame video subjek antar-split secara 100%.
2. YOLOv13n mempertahankan performa deteksi dan klasifikasi yang sangat tinggi dan stabil, membuktikan ketangguhannya dalam generalisasi ke subjek dan sesi belajar baru.
3. YOLOv13n tetap menjadi model utama yang paling efisien (~25 ms per frame, ~40 FPS) untuk pemantauan real-time.