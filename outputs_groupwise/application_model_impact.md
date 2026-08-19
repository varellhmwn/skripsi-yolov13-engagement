# Dampak Model Group-Wise terhadap Aplikasi Web Real-Time

## 1. Analisis Kompatibilitas Model
Aplikasi web dashboard (`dashboard/app.py`) dan skrip pemantauan real-time (`scripts/realtime_predict.py`) saat ini menggunakan bobot dari pelatihan awal:
- **Bobot Aplikasi Saat Ini**: `runs/yolov13_master_combined_v2/weights/best.pt`
- **Bobot Baru Hasil Group-Wise**: `runs/yolov13_master_groupwise_v1/weights/best.pt`

| Aspek Arsitektur | Model Lama (v2) | Model Baru Group-Wise (v1) | Status Kompatibilitas |
|:-----------------|:----------------|:---------------------------|:----------------------|
| **Arsitektur Model** | YOLOv13n | YOLOv13n | 100% Identik |
| **Input Image Size** | 640 × 640 piksel | 640 × 640 piksel | 100% Identik |
| **Jumlah Kelas (nc)** | 4 Kelas | 4 Kelas | 100% Identik |
| **Class Mapping** | `{0: engaged, 1: confused, 2: bored, 3: frustrated}` | `{0: engaged, 1: confused, 2: bored, 3: frustrated}` | 100% Identik |
| **Output Tensor** | Bounding boxes + Class probs | Bounding boxes + Class probs | 100% Identik |

## 2. Pengujian Aplikasi yang Perlu Diulang jika Bobot Diganti
Apabila bobot model pada `dashboard/app.py` dialihkan ke model group-wise baru, pengujian berikut **wajib diuji ulang**:
1. **Real-time Inference Speed & FPS**: Menguji stabilitas frame rate kamera web secara langsung.
2. **Post-processing EMA Smoothing**: Menguji parameter smoothing window pada deteksi ekspresi berkelanjutan.
3. **Pose & Head Orientation Scenarios**: Menguji respons model terhadap variasi sudut kepala mahasiswa.
4. **Lighting Variation Scenarios**: Menguji keandalan deteksi pada kondisi pencahayaan minim/berlebih.

## 3. Fitur yang TIDAK Perlu Diulang
Pengujian fitur *black-box software* seperti sistem autentikasi, navigasi modul pembelajaran, pencatatan histori belajar ke JSON, dan visualisasi grafik dashboard **tidak terpengaruh** karena antarmuka data tensor YOLO tidak berubah.