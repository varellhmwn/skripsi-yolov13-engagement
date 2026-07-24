# Arsitektur Sistem

Dokumen ini menjelaskan arsitektur keseluruhan sistem deteksi emosi
mahasiswa real-time menggunakan YOLOv13.

## 1. Gambaran Umum

```
┌─────────────────────────────────────────────────────────────────┐
│                    DASHBOARD WEB (Browser)                       │
│  ┌──────────────┐  ┌──────────────┐  ┌────────────────────┐    │
│  │  Login Page   │→│ Pilih Modul  │→│  Sesi Pembelajaran  │    │
│  └──────────────┘  └──────────────┘  │  ┌──────┐ ┌──────┐ │    │
│                                       │  │Baca  │→│ Kuis │ │    │
│                                       │  │Materi│ │      │ │    │
│                                       │  └──────┘ └──┬───┘ │    │
│                                       └──────────────┼─────┘    │
│                                                      ↓          │
│                                              ┌──────────────┐   │
│                                              │  Laporan     │   │
│                                              │  Skor + Emosi│   │
│                                              └──────────────┘   │
├─────────────────────────────────────────────────────────────────┤
│                     WebSocket (Socket.IO)                        │
│              Frame kamera ↕ Hasil deteksi emosi                 │
├─────────────────────────────────────────────────────────────────┤
│                    FLASK SERVER (Python)                         │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────┐      │
│  │   OpenCV      │→│  YOLOv13n    │→│ Post-Processing   │      │
│  │ (Capture)     │  │ (Inference)  │  │ (Smoothing +     │      │
│  │               │  │              │  │  Neutral Trick)  │      │
│  └──────────────┘  └──────────────┘  └──────────────────┘      │
└─────────────────────────────────────────────────────────────────┘
```

## 2. Komponen Utama

### 2.1 Model YOLOv13n (Nano)

**YOLOv13** adalah arsitektur object detection terbaru dari keluarga YOLO.
Varian **Nano (n)** dipilih karena:
- Ukuran model kecil (~5.4 MB)
- Kecepatan inference tinggi (cocok untuk real-time)
- Performa tetap akurat untuk tugas 4-kelas

| Spesifikasi | Nilai |
|-------------|-------|
| Arsitektur | YOLOv13n |
| Input Size | 640×640 |
| Output | Bounding box + class probability |
| Kelas | 4 (engaged, confused, bored, frustrated) |
| Framework | PyTorch (Ultralytics) |

### 2.2 Neutral Trick (Confidence Thresholding)

Model YOLOv13 hanya dilatih pada 4 kelas emosi. Namun di dunia nyata,
wajah manusia sering kali berada dalam keadaan **datar/rileks** (baseline)
yang tidak termasuk dalam 4 kelas tersebut.

**Solusi**: Menerapkan mekanisme *Confidence Thresholding*:

```
┌──────────────────┐
│  Frame Wajah     │
└────────┬─────────┘
         ↓
┌──────────────────┐
│  YOLOv13 Predict │
│  → class_id      │
│  → confidence    │
└────────┬─────────┘
         ↓
┌──────────────────────────────┐
│  Sliding Window (30 frame)   │
│  → Vote Ratio (kelas dominan)│
│  → Avg Confidence            │
└────────┬─────────────────────┘
         ↓
    ┌────────────┐
    │ Vote ≥ 50% │──── YA ───→ ┌────────────────┐
    │    DAN      │             │ Confidence      │
    │ Conf ≥ 65% │             │ ≥ 65%?          │
    └────────────┘             └───────┬─────────┘
         │                         YA  │    TIDAK
       TIDAK                          ↓      ↓
         │                     ┌─────────┐ ┌─────────┐
         └────────────────────→│ NEUTRAL  │ │ NEUTRAL │
                               └─────────┘ └─────────┘
                               ┌─────────────────────┐
                         YA →  │ LABEL EMOSI ASLI    │
                               │ (engaged/confused/  │
                               │  bored/frustrated)  │
                               └─────────────────────┘
```

**Parameter Default**:
- `min_vote_ratio`: 0.50 (50% frame harus sepakat)
- `min_avg_confidence`: 0.65 (rata-rata confidence ≥ 65%)
- `window_size`: 30 frame (buffer temporal)

### 2.3 Sliding Window Smoothing

Untuk menghindari prediksi yang "berkedip-kedip" (*flickering*) antar frame,
sistem menggunakan *temporal smoothing*:

1. Setiap prediksi frame disimpan dalam buffer (deque) berukuran 30
2. Kelas yang paling sering muncul (*majority voting*) dipilih sebagai
   prediksi stabil
3. Prediksi baru akan diterima hanya setelah minimal 8 frame terkumpul
4. Jika buffer belum cukup, label default adalah "neutral"

### 2.4 Arsitektur Dashboard Web

Dashboard menggunakan arsitektur **Single Page Application (SPA)**
berbasis Flask + Socket.IO:

| Layer | Teknologi | Fungsi |
|-------|-----------|--------|
| **Frontend** | HTML/CSS/JavaScript | Antarmuka pengguna |
| **Transport** | Socket.IO (WebSocket) | Streaming frame real-time |
| **Backend** | Flask (Python) | Server + API |
| **AI Engine** | YOLOv13 + PyTorch | Inferensi model |
| **Vision** | OpenCV | Capture & pengolahan gambar |

**Alur Data Real-Time**:
1. OpenCV membaca frame dari webcam server
2. Frame dikirim ke YOLOv13 untuk inferensi
3. Hasil deteksi (emosi + confidence) diolah via Sliding Window
4. Frame + bounding box + data emosi dikirim ke browser via Socket.IO
5. JavaScript di browser menampilkan frame di canvas + update UI sidebar

## 3. Alur Kerja Pengguna (User Flow)

```
Login (Nama + NIM)
       ↓
Pilih Modul Pembelajaran
       ↓
Nyalakan Kamera ← [Kamera mulai mendeteksi emosi]
       ↓
Baca Materi (Timer 1 menit minimum)
       ↓
Kerjakan Kuis (5 soal pilihan ganda)
       ↓
Submit Jawaban
       ↓
Laporan Akhir:
  - Skor Belajar (0-100)
  - Jawaban Benar
  - Durasi Sesi
  - Emosi Dominan
  - Distribusi Emosi (bar chart)
```

## 4. Struktur File Utama

```
skripsi_yolov13_engagement/
├── scripts/
│   ├── train.py              ← Training model
│   ├── evaluate.py           ← Evaluasi pada test set
│   └── realtime_predict.py   ← Inference webcam (standalone)
│
├── runs/weights/best.pt      ← Model terlatih
│
└── (dashboard/ → ../dashboard/)
    ├── app.py                ← Flask server + Socket.IO
    ├── templates/
    │   ├── index.html        ← Halaman login
    │   └── dashboard.html    ← Halaman utama SPA
    ├── static/
    │   ├── style.css         ← Stylesheet
    │   └── app.js            ← Frontend logic
    └── modules.json          ← Data modul pembelajaran
```
