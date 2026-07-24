# Preprocessing & Data Augmentation

Dokumen ini menjelaskan tahapan preprocessing dan augmentasi data yang dilakukan
pada projek deteksi emosi mahasiswa menggunakan YOLOv13.

## 1. Sumber Dataset

Master Combined Dataset dibentuk dari penggabungan **2 sumber utama**:

| Sumber | Jumlah | Keterangan |
|--------|--------|------------|
| **Roboflow 4-Class Finetuned** | 953 gambar | Dataset DAiSEE yang sudah di-finetune dan diverifikasi |
| **Hard Samples (Big-Data)** | 745 gambar | Sampel tambahan dari skenario sulit yang dikumpulkan melalui real-time capture |
| **Total** | **1.698 gambar** | — |

### Pembagian Data (Stratified Split)

| Split | Engaged | Confused | Bored | Frustrated | Total |
|-------|---------|----------|-------|------------|-------|
| **Train** (80%) | 419 (30.9%) | 308 (22.7%) | 294 (21.7%) | 336 (24.8%) | 1.357 |
| **Val** (10%) | 52 (31.0%) | 38 (22.6%) | 36 (21.4%) | 42 (25.0%) | 168 |
| **Test** (10%) | 53 (30.6%) | 40 (23.1%) | 38 (22.0%) | 42 (24.3%) | 173 |

- **Random Seed**: 42 (untuk reprodusibilitas)
- **Metode**: Stratified random split (proporsi kelas dijaga konsisten)

---

## 2. Preprocessing Data

### 2.1 Normalisasi Bounding Box (Format YOLO)

Setiap anotasi wajah dikonversi ke format standar YOLO:
```
<class_id> <x_center> <y_center> <width> <height>
```
Dimana semua koordinat dinormalisasi ke rentang `[0.0, 1.0]` relatif terhadap
dimensi gambar asli.

### 2.2 Class Remapping

Dataset dari sumber berbeda memiliki urutan kelas yang berbeda.
Semua disamakan ke standar DAiSEE:

| ID | Kelas | Deskripsi |
|----|-------|-----------|
| 0 | engaged | Mahasiswa fokus dan terlibat aktif |
| 1 | confused | Mahasiswa tampak kebingungan |
| 2 | bored | Mahasiswa tampak bosan/mengantuk |
| 3 | frustrated | Mahasiswa tampak frustrasi/kesal |

Contoh remapping dari sumber Big-Data:
```
Big-Data → Master Combined
0 (bored)      → 2 (bored)
1 (confused)   → 1 (confused)
2 (engaged)    → 0 (engaged)
3 (frustrated) → 3 (frustrated)
```

### 2.3 Resizing Otomatis saat Training

Framework YOLO secara otomatis melakukan:
- **Resize** semua gambar ke **640×640 piksel** (letterbox padding)
- **Normalisasi piksel** dari rentang `[0, 255]` ke `[0.0, 1.0]`

### 2.4 Data Cleansing

- Validasi format YOLO (5 kolom per baris: class, x, y, w, h)
- Penghapusan file label kosong atau rusak
- Verifikasi pasangan gambar-label (setiap gambar harus punya label, dan sebaliknya)
- Pengecekan nilai koordinat di luar batas (x, y harus ∈ [0,1])

---

## 3. Data Augmentation

Augmentasi dilakukan secara **on-the-fly** selama training untuk meningkatkan
variasi data dan mencegah overfitting.

### 3.1 Augmentasi Warna (Color Space)

| Parameter | Nilai | Efek |
|-----------|-------|------|
| `hsv_h` | 0.015 | Pergeseran Hue ±1.5% |
| `hsv_s` | 0.7 | Perubahan Saturasi ±70% |
| `hsv_v` | 0.4 | Perubahan Brightness ±40% |

**Tujuan**: Membuat model tahan terhadap variasi pencahayaan ruangan
(terang, redup, lampu kuning/putih).

### 3.2 Augmentasi Geometri

| Parameter | Nilai | Efek |
|-----------|-------|------|
| `degrees` | 10.0 | Rotasi acak ±10° |
| `translate` | 0.1 | Translasi acak ±10% |
| `scale` | 0.5 | Perubahan skala ±50% |
| `fliplr` | 0.5 | Flip horizontal (probabilitas 50%) |

**Tujuan**: Membuat model tahan terhadap variasi posisi duduk, jarak ke kamera,
dan kemiringan kepala mahasiswa.

### 3.3 Augmentasi Komposit (Advanced)

| Parameter | Nilai | Efek |
|-----------|-------|------|
| `mosaic` | 1.0 | Menggabungkan 4 gambar menjadi 1 (aktif 100%) |
| `mixup` | 0.1 | Mencampur 2 gambar secara transparan (10%) |
| `copy_paste` | 0.1 | Copy-paste objek dari gambar lain (10%) |
| `erasing` | 0.4 | Menghapus area acak pada gambar (40%) |
| `auto_augment` | randaugment | Kebijakan augmentasi otomatis |

**Tujuan**: Meningkatkan generalisasi model secara signifikan dengan
memperkenalkan variasi yang lebih kompleks.

### 3.4 Close Mosaic

```yaml
close_mosaic: 10  # Matikan mosaic di 10 epoch terakhir
```
Mosaic dinonaktifkan di 10 epoch terakhir agar model belajar dari
gambar individual yang lebih realistis pada fase akhir training.

---

## 4. Preprocessing saat Inference (Real-Time)

Saat model dijalankan secara real-time, terdapat preprocessing tambahan:

1. **Frame Rescaling**: Frame video di-resize ke 640×640
2. **Face Area Filtering**: Hanya wajah dengan area ≥ 2% dari total frame
   yang diproses (menghindari deteksi noise)
3. **Temporal Smoothing**: Sliding window 30 frame terakhir untuk
   mengurangi flicker prediksi
4. **Confidence Thresholding (Neutral Trick)**: Prediksi dengan
   confidence < 65% atau vote ratio < 50% dikategorikan sebagai "neutral"
