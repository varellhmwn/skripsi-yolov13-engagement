# Laporan Audit Repositori & Bobot Model YOLOv13n

## 1. Lokasi Dataset & Konfigurasi
- **Dataset Path**: `datasets/master_combined_dataset/`
- **Konfigurasi Data**: `datasets/master_combined_dataset/data.yaml`
- **Mapping Kelas**:
  - `0`: `engaged`
  - `1`: `confused`
  - `2`: `bored`
  - `3`: `frustrated`
- **Split Direktori**:
  - `images/train` & `labels/train`
  - `images/val` & `labels/val`
  - `images/test` & `labels/test`

---

## 2. Audit Seluruh File Weights (`best.pt`)

Berdasarkan inspeksi metadata PyTorch checkpoint (`torch.load`), berikut adalah seluruh checkpoint `best.pt` yang ditemukan di repositori:

| No | Path Bobot | Ukuran File | Epochs / LR | Deskripsi & Asal Usul | Penggunaan di Repositori |
|:---|:-----------|:------------|:------------|:----------------------|:-------------------------|
| 1 | `runs/yolov13_master_combined_v2/weights/best.pt` | 5,387,144 bytes | 150 epoch, lr0=0.001 | Pelatihan utama model YOLOv13n (150 epoch, AdamW) pada Master Combined Dataset | Digunakan di `dashboard/app.py`, `scripts/evaluate_v2.py` (Kode Program 4.2 Tugas Akhir), dan `scripts/evaluate.py` |
| 2 | `runs/yolov13_master_combined_v3/weights/best.pt` | 5,390,344 bytes | 100 epoch, lr0=0.0001 | Fine-tuning lanjutan dari V2 menggunakan hard samples (Active Learning) | Digunakan di `scripts/realtime_predict.py` |
| 3 | `runs/yolov13_master_combined_wtest_4_kelas/weights/best.pt` | 5,398,984 bytes | 150 epoch, lr0=0.001 | Pelatihan ulang dari nol (scratch) dengan konfigurasi `scripts/train.py` | Model retrained scratch 4-class |
| 4 | `runs/yolov13n_retrained_scratch/weights/best.pt` | 5,398,984 bytes | 150 epoch, lr0=0.001 | Salinan identik dari model retrained scratch | `pipeline_retrain_from_scratch/` |
| 5 | `runs/yolov13_master_combinedw/weights/best.pt` | 5,397,064 bytes | 100 epoch, lr0=0.001 | Eksperimen awal 100 epoch | Eksperimen lama |
| 6 | `yolov13n.pt` (Root) | 10,570,688 bytes | N/A | Base pretrained weights YOLOv13n COCO (80 kelas) | Bobot awal sebelum fine-tuning |

---

## 3. Evaluasi Komparatif Kandidat Model pada Test Set (173 Citra)

Hasil pengujian native YOLO pada test split yang sama (173 citra, `imgsz=640`, `conf=0.25`, `device=0`):

| Model Candidate | Precision | Recall | mAP@0.5 | mAP@0.5:0.95 | Engaged AP50 | Confused AP50 | Bored AP50 | Frustrated AP50 |
|:----------------|----------:|-------:|--------:|-------------:|-------------:|--------------:|-----------:|----------------:|
| **v2 (Master Final TA)** | **0.9938** | **0.9892** | **0.9948** | **0.9808** | 0.9950 | 0.9940 | 0.9950 | 0.9950 |
| **v3 (Realtime Fine-Tuned)** | 0.9916 | 0.9922 | 0.9948 | 0.9892 | 0.9950 | 0.9940 | 0.9950 | 0.9950 |
| **wtest_4_kelas (Scratch)** | **0.9938** | **0.9892** | **0.9948** | **0.9808** | 0.9950 | 0.9940 | 0.9950 | 0.9950 |

---

## 4. Penentuan Model Final untuk Eksperimen

**Model Final yang Dipilih**:
`runs/yolov13_master_combined_v2/weights/best.pt` (atau `runs/yolov13_master_combined_wtest_4_kelas/weights/best.pt` yang memiliki performa identik).

**Alasan Pemilihan**:
1. `yolov13_master_combined_v2` adalah bobot resmi yang dirujuk dalam naskah Tugas Akhir Bab IV (`scripts/evaluate_v2.py` / Kode Program 4.2) dan aplikasi web `dashboard/app.py`.
2. Menghasilkan performa deteksi mAP@0.5 sebesar **99.48%** dan mAP@0.5:0.95 sebesar **98.08%**.
3. Seluruh skrip eksperimen pembanding (YOLO evaluation, Image-level classification, YOLO Crop + HOG-KNN hybrid, runtime benchmark, dan error analysis) akan disatukan menggunakan konfigurasi terpusat di `experiments/config.py` yang mengarah ke bobot ini.
