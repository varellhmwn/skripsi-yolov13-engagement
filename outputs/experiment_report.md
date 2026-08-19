# Laporan Eksperimen: Perbandingan YOLOv13n vs HOG-KNN

**Tanggal**: 2026-08-19 23:42:10
**Platform**: Windows 10
**Processor**: AMD64 Family 25 Model 117 Stepping 2, AuthenticAMD

## A. Dataset

| Split | Jumlah | Engaged | Confused | Bored | Frustrated |
|-------|-------:|--------:|---------:|------:|-----------:|
| train | 1364 | 421 | 309 | 298 | 336 |
| val | 168 | 52 | 38 | 36 | 42 |
| test | 173 | 53 | 40 | 38 | 42 |

**Catatan dataset:**
- 39 orphan label files di train set (tanpa matching image)
- 8 file train memiliki 2 bounding box (objek utama + wajah kecil di background)
- Semua file test dan val memiliki tepat 1 bounding box

## B. Konfigurasi HOG

| Parameter | Nilai |
|-----------|-------|
| Resize | 64 × 64 pixel |
| Color space | Grayscale |
| Orientations | 9 |
| Pixels per cell | 8 × 8 |
| Cells per block | 2 × 2 |
| Block normalization | L2-Hys |
| Library | scikit-image (`skimage.feature.hog`) |

## C. Tuning KNN

### Hasil Validasi per-K

| K | Accuracy | Macro Precision | Macro Recall | Macro F1 | Weighted F1 |
|--:|---------:|----------------:|-------------:|---------:|------------:|
| 1 | 0.9524 | 0.9546 | 0.9512 | 0.9522 | 0.9524 |
| 3 | 0.9286 | 0.9341 | 0.9279 | 0.9291 | 0.9289 |
| 5 | 0.8988 | 0.9040 | 0.9012 | 0.9021 | 0.8990 |
| 7 | 0.8750 | 0.8776 | 0.8785 | 0.8771 | 0.8745 |
| 9 | 0.8512 | 0.8524 | 0.8546 | 0.8517 | 0.8505 |
| 11 | 0.8452 | 0.8467 | 0.8496 | 0.8455 | 0.8436 |
| 13 | 0.8393 | 0.8398 | 0.8437 | 0.8384 | 0.8371 |
| 15 | 0.8333 | 0.8329 | 0.8377 | 0.8309 | 0.8307 |

**K terbaik = 1** (Macro F1 = 0.9522)

**Kriteria pemilihan:**
1. Macro F1-score tertinggi pada validation set
2. Tiebreaker: Accuracy tertinggi → K terkecil
3. Metric jarak: Euclidean
4. Validation set berjumlah 168 citra

## D. Hasil YOLOv13n

### Object Detection Metrics

| Metrik | Nilai |
|--------|------:|
| Precision | 0.9938 |
| Recall | 0.9892 |
| mAP@0.5 | 0.9948 |
| mAP@0.75 | 0.9948 |
| mAP@0.5:0.95 | 0.9808 |

### Image-Level Classification Metrics

| Metrik | Nilai |
|--------|------:|
| Accuracy | 0.9884 |
| Macro Precision | 0.9875 |
| Macro Recall | 0.9875 |
| Macro F1 | 0.9872 |
| Weighted Precision | 0.9890 |
| Weighted Recall | 0.9884 |
| Weighted F1 | 0.9884 |

#### Per-Class Performance (YOLO)

| Kelas | Precision | Recall | F1-Score | Support |
|-------|----------:|-------:|---------:|--------:|
| engaged | 1.0000 | 1.0000 | 1.0000 | 53 |
| confused | 1.0000 | 0.9500 | 0.9744 | 40 |
| bored | 0.9500 | 1.0000 | 0.9744 | 38 |
| frustrated | 1.0000 | 1.0000 | 1.0000 | 42 |

**Waktu inferensi rata-rata**: 22.85 ms/image
**Estimasi FPS**: 43.8
**Detection failures**: 0

## E. Hasil HOG-KNN Ground-Truth Crop

| Metrik | Nilai |
|--------|------:|
| Accuracy | 0.9942 |
| Macro Precision | 0.9954 |
| Macro Recall | 0.9938 |
| Macro F1 | 0.9945 |
| Weighted Precision | 0.9943 |
| Weighted Recall | 0.9942 |
| Weighted F1 | 0.9942 |

#### Per-Class Performance (HOG-KNN GT)

| Kelas | Precision | Recall | F1-Score | Support |
|-------|----------:|-------:|---------:|--------:|
| engaged | 0.9815 | 1.0000 | 0.9907 | 53 |
| confused | 1.0000 | 0.9750 | 0.9873 | 40 |
| bored | 1.0000 | 1.0000 | 1.0000 | 38 |
| frustrated | 1.0000 | 1.0000 | 1.0000 | 42 |

**KNN predict per image**: 0.08 ms
**Full pipeline (crop+resize+grayscale+HOG+KNN)**: 9.81 ms/image (mean)

## F. Hasil YOLO-based Face Crop + HOG-KNN

### Classification Metrics (Detected Faces Only)

| Metrik | Nilai |
|--------|------:|
| Accuracy | 0.9884 |
| Macro Precision | 0.9909 |
| Macro Recall | 0.9875 |
| Macro F1 | 0.9890 |
| Weighted F1 | 0.9884 |

#### Per-Class Performance (YOLO-HOG-KNN)

| Kelas | Precision | Recall | F1-Score | Support |
|-------|----------:|-------:|---------:|--------:|
| engaged | 0.9636 | 1.0000 | 0.9815 | 53 |
| confused | 1.0000 | 0.9500 | 0.9744 | 40 |
| bored | 1.0000 | 1.0000 | 1.0000 | 38 |
| frustrated | 1.0000 | 1.0000 | 1.0000 | 42 |

### End-to-End Statistics

- Total images: 173
- Detected successfully: 173
- Detection failed: 0
- End-to-end accuracy: 0.9884

### Timing Breakdown

| Komponen | Waktu (ms) |
|----------|----------:|
| YOLO detection | 26.16 |
| Crop + HOG preprocessing | 1.07 |
| KNN prediction | 18.11 |
| **Full pipeline** | **45.47** |


## G. Perbandingan Model

### Classification Performance

| Model | Accuracy | Macro P | Macro R | Macro F1 | Weighted F1 |
|-------|--------:|---------:|--------:|---------:|------------:|
| YOLOv13n | 0.9884 | 0.9875 | 0.9875 | 0.9872 | 0.9884 |
| HOG-KNN GT Crop | 0.9942 | 0.9954 | 0.9938 | 0.9945 | 0.9942 |
| YOLO Crop + HOG-KNN | 0.9884 | 0.9909 | 0.9875 | 0.9890 | 0.9884 |

### Object Detection Performance (Khusus YOLO)

| Metrik | Nilai |
|--------|------:|
| Precision | 0.9938 |
| Recall | 0.9892 |
| mAP@0.5 | 0.9948 |
| mAP@0.5:0.95 | 0.9808 |

> **Catatan:** HOG-KNN tidak menghasilkan bounding box sehingga metrik mAP dan IoU tidak dihitung untuk HOG-KNN. Perbandingan mAP antara YOLO dan KNN tidak valid karena KNN hanya melakukan klasifikasi, bukan lokalisasi objek.

### Waktu Pemrosesan

| Model | Avg Time (ms) | Catatan |
|-------|-------------:|---------|
| YOLOv13n | 22.85 | End-to-end (detection + classification) |
| HOG-KNN GT Crop | 9.81 | crop + resize + grayscale + HOG + KNN predict |
| YOLO Crop + HOG-KNN | 45.47 | YOLO detection + crop + HOG + KNN predict |

> **Catatan waktu:** Waktu `knn.predict()` saja jauh lebih kecil dari waktu total pipeline. Membandingkan hanya `knn.predict()` dengan waktu inferensi YOLO tidak fair karena tidak memperhitungkan preprocessing (crop, resize, grayscale, HOG extraction).

## H. Error Analysis

### Distribusi Kategori Kesalahan (YOLO vs HOG-KNN GT)

| Kategori | Jumlah | Persentase |
|----------|-------:|-----------:|
| both_correct | 170 | 98.3% |
| yolo_correct_knn_wrong | 1 | 0.6% |
| yolo_wrong_knn_correct | 2 | 1.2% |
| both_wrong | 0 | 0.0% |

### Confusion Terbesar

**HOG-KNN GT Crop:**

- confused → engaged: 1 kesalahan

**YOLOv13n:**

- confused → bored: 2 kesalahan


## I. Kesimpulan

Berdasarkan hasil eksperimen pada 173 citra test set:

1. **YOLOv13n** memperoleh Macro F1 sebesar **0.9872** pada evaluasi image-level classification. Model ini melakukan lokalisasi wajah dan klasifikasi ekspresi secara simultan (end-to-end).

2. **HOG-KNN Ground-Truth Crop** memperoleh Macro F1 sebesar **0.9945**. Model ini menerima crop wajah yang sudah benar (ground-truth bounding box), sehingga tugasnya murni klasifikasi tanpa perlu lokalisasi.

3. **YOLO-based Face Crop + HOG-KNN** memperoleh Macro F1 sebesar **0.9890**. Pipeline ini menunjukkan performa KNN ketika crop wajah diperoleh secara otomatis dari deteksi YOLO, bukan dari ground truth.

**Catatan penting:**
- Perbandingan ini harus mempertimbangkan perbedaan tugas dan pipeline masing-masing model.
- YOLO melakukan detection + classification sekaligus, sedangkan HOG-KNN hanya klasifikasi.
- HOG-KNN GT Crop mendapat keuntungan dari crop wajah yang sudah benar.
- mAP tidak relevan untuk HOG-KNN karena tidak menghasilkan bounding box.
- Perbedaan performa antara HOG-KNN GT dan YOLO-HOG-KNN menunjukkan dampak kualitas lokalisasi terhadap klasifikasi.

## Journal-ready Summary

### Metode Eksperimen HOG-KNN

Sebagai model pembanding, digunakan pendekatan klasifikasi berbasis Histogram of Oriented Gradients (HOG) dan K-Nearest Neighbors (KNN). Citra wajah diperoleh melalui crop berdasarkan bounding box ground truth dari anotasi YOLO. Setiap crop di-resize ke ukuran 64×64 piksel, dikonversi ke grayscale, kemudian diekstraksi fitur HOG dengan parameter 9 orientasi, pixels per cell [8, 8], cells per block [2, 2], dan normalisasi L2-Hys. Implementasi HOG menggunakan library scikit-image (`skimage.feature.hog`). Selain itu, dilakukan eksperimen hybrid di mana bounding box diperoleh secara otomatis dari deteksi YOLOv13n, kemudian crop diproses dengan HOG-KNN untuk klasifikasi ekspresi.

### Hasil Tuning K

Pencarian hyperparameter K dilakukan pada validation set (168 citra) dengan nilai K ∈ {1, 3, 5, 7, 9, 11, 13, 15} dan metric jarak Euclidean. Hasil tuning menunjukkan bahwa K = 1 menghasilkan Macro F1-score tertinggi pada validation set sebesar 0.9522 dengan accuracy 0.9524. Pemilihan K dilakukan berdasarkan kriteria utama Macro F1-score validation tertinggi, diikuti accuracy dan nilai K terkecil sebagai tiebreaker.

### Hasil Pengujian KNN

Evaluasi HOG-KNN (K=1) pada 173 citra test set dengan ground-truth crop menghasilkan accuracy 0.9942, macro precision 0.9954, macro recall 0.9938, dan macro F1-score 0.9945. Pada eksperimen hybrid (YOLO crop + HOG-KNN), di mana crop wajah diperoleh secara otomatis dari deteksi YOLOv13n, diperoleh accuracy 0.9884 dan macro F1-score 0.9890.

### Perbandingan YOLO dan KNN

YOLOv13n memperoleh macro F1-score sebesar 0.9872 pada evaluasi image-level classification, sedangkan HOG-KNN dengan ground-truth crop memperoleh 0.9945. Pipeline hybrid YOLO crop + HOG-KNN menghasilkan macro F1-score 0.9890. Perlu dicatat bahwa perbandingan ini memiliki konteks yang berbeda: YOLOv13n melakukan lokalisasi wajah dan klasifikasi ekspresi secara simultan (end-to-end), sedangkan HOG-KNN ground-truth menerima crop wajah yang sudah benar sehingga tugasnya lebih sederhana (murni klasifikasi). Pipeline hybrid menunjukkan performa klasifikasi KNN ketika bergantung pada lokalisasi otomatis dari YOLO. Metrik mAP tidak dihitung untuk HOG-KNN karena metode ini tidak menghasilkan bounding box prediksi.

### Tabel Ringkas Perbandingan

| Model | Accuracy | Macro F1 | Weighted F1 | mAP@0.5 | mAP@0.5:0.95 |
|-------|--------:|---------:|------------:|--------:|-------------:|
| YOLOv13n | 0.9884 | 0.9872 | 0.9884 | 0.9948 | 0.9808 |
| HOG-KNN GT Crop | 0.9942 | 0.9945 | 0.9942 | N/A | N/A |
| YOLO Crop + HOG-KNN | 0.9884 | 0.9890 | 0.9884 | N/A | N/A |

### Penjelasan mAP

Metrik mean Average Precision (mAP) tidak dihitung untuk model HOG-KNN karena metode ini tidak melakukan prediksi bounding box. HOG-KNN hanya melakukan klasifikasi pada crop wajah yang sudah tersedia, sehingga tidak menghasilkan output lokalisasi yang diperlukan untuk menghitung Intersection over Union (IoU) dan Average Precision per kelas. Membandingkan mAP YOLO dengan HOG-KNN tidak valid karena keduanya memiliki output yang fundamentally berbeda: YOLO menghasilkan bounding box + kelas + confidence, sedangkan KNN hanya menghasilkan kelas prediksi.
