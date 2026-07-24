# 🎓 Skripsi — Deteksi Emosi Mahasiswa Real-Time dengan YOLOv13

> **Student Engagement Detection using YOLOv13 on DAiSEE-based Dataset**
> Tugas Akhir — Program Studi Informatika

## 📋 Deskripsi

Sistem deteksi emosi mahasiswa secara real-time menggunakan model **YOLOv13n** (Nano) yang dilatih pada dataset gabungan (*Master Combined Dataset*). Sistem ini mampu mendeteksi 4 emosi utama (**Engaged, Confused, Bored, Frustrated**) melalui ekspresi wajah, ditambah label **Neutral** yang dihasilkan dari mekanisme *Confidence Thresholding* saat model tidak cukup yakin.

Aplikasi utama berupa **Dashboard Web** berbasis Flask + SocketIO yang memungkinkan mahasiswa untuk:
1. Login dan memilih modul pembelajaran
2. Membaca materi sambil dideteksi emosinya secara real-time
3. Mengerjakan kuis
4. Melihat laporan skor belajar dan distribusi emosi

## 🏗️ Arsitektur Model

| Parameter | Nilai |
|---|---|
| **Model** | YOLOv13n (Nano) |
| **Framework** | Ultralytics YOLO |
| **Dataset** | Master Combined (1.698 gambar) |
| **Kelas** | 4 (engaged, confused, bored, frustrated) + neutral trick |
| **Epochs** | 100 |
| **Image Size** | 640×640 |
| **Optimizer** | AdamW (lr=0.001) |
| **Batch Size** | 16 |

## 📊 Hasil Performa (Epoch Terakhir)

| Metrik | Nilai |
|---|---|
| **Precision** | 97.40% |
| **Recall** | 97.57% |
| **mAP@50** | 99.00% |
| **mAP@50-95** | 97.45% |

## 📂 Struktur Folder

```
skripsi_yolov13_engagement/
├── README.md                    ← Anda di sini
├── requirements.txt             ← Dependensi Python
│
├── configs/
│   └── training_config.yaml     ← Konfigurasi training
│
├── scripts/
│   ├── train.py                 ← Script training model
│   ├── evaluate.py              ← Script evaluasi pada test set
│   └── realtime_predict.py      ← Real-time inference via webcam
│
├── runs/                        ← Hasil training (weights, grafik, CSV)
│   └── yolov13_master_combined/
│       ├── weights/best.pt      ← Model terbaik
│       ├── results.csv          ← Log training per epoch
│       ├── confusion_matrix.png
│       └── ...
│
├── docs/
│   ├── preprocessing.md         ← Dokumentasi preprocessing & augmentasi
│   └── architecture.md          ← Arsitektur sistem & alur kerja
│
├── datasets/
│   └── README.md                ← Info dataset (pointer ke datasets/)
│
└── dashboard/
    └── README.md                ← Instruksi menjalankan dashboard web
```

## 🚀 Cara Menjalankan

### 1. Install Dependensi
```bash
pip install -r requirements.txt
```

### 2. Training Model
```bash
python scripts/train.py
```

### 3. Evaluasi Model pada Test Set
```bash
python scripts/evaluate.py
```

### 4. Real-Time Inference (Webcam)
```bash
python scripts/realtime_predict.py
```

### 5. Dashboard Web
```bash
python ../dashboard/app.py
```
Buka browser ke **http://localhost:5000**

## 📖 Dokumentasi Tambahan

- [Preprocessing & Augmentasi](docs/preprocessing.md)
- [Arsitektur Sistem](docs/architecture.md)
- [Info Dataset](datasets/README.md)
- [Panduan Dashboard](dashboard/README.md)

## 🛠️ Teknologi

- **Python 3.10+**
- **YOLOv13** (Ultralytics)
- **OpenCV** — Pemrosesan citra & video
- **Flask + SocketIO** — Dashboard web real-time
- **PyTorch** — Backend deep learning
