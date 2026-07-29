# Preprocessing Scripts

Skrip-skrip di folder ini digunakan untuk menyiapkan **Master Combined Dataset**
dari sumber data mentah. Jalankan secara berurutan sesuai nomor prefix.

## Urutan Eksekusi

```
Tahap 1: Filter & Remap Kelas dari Roboflow
    python 01_filter_roboflow_to_4class.py

Tahap 2: Gabungkan Roboflow + Hard Samples (Big-Data) → Master Dataset
    python 02_build_master_dataset.py

Tahap 3: Stratified Random Split (80/10/10, seed=42)
    python 03_stratified_resplit.py
```

## Detail Setiap Skrip

### 01_filter_roboflow_to_4class.py
- **Input**: Dataset Roboflow mentah (multi-kelas)
- **Proses**: 
  - Membaca `data.yaml` dari dataset Roboflow
  - Memetakan nama kelas ke 4 kelas standar (engaged, confused, bored, frustrated)
  - Meng-remap class ID pada file label YOLO
  - Memvalidasi format bounding box
- **Output**: Dataset Roboflow yang sudah di-filter ke 4 kelas (`roboflow_4class_yolo_finetuned/`)

### 02_build_master_dataset.py
- **Input**: 
  - `roboflow_4class_yolo_finetuned/` (953 gambar, dari Tahap 1)
  - `big-data-data.v1i.yolov8.zip` (745 gambar hard samples)
- **Proses**:
  - Menyalin dataset Roboflow (prefix: tanpa perubahan)
  - Mengekstrak ZIP hard samples dan meng-remap class ID:
    - Big-data original: 0=bored, 1=confused, 2=engaged, 3=frustrated
    - Target standar:    0=engaged, 1=confused, 2=bored, 3=frustrated
  - Menambahkan prefix `bigdata_` pada file hard samples
  - Membuat `data.yaml` final
- **Output**: `master_4class/` (1.698 gambar gabungan)

### 03_stratified_resplit.py
- **Input**: Dataset gabungan dari Tahap 2
- **Proses**:
  - Membaca seluruh gambar dan label
  - Melakukan validasi format YOLO
  - Melakukan Stratified Random Sampling berdasarkan kelas dominan
  - Split: 80% Train / 10% Validation / 10% Test
  - Random seed: 42 (untuk reprodusibilitas)
- **Output**: `master_combined_dataset/` (dataset final siap training)
