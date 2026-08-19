# Laporan Eksperimen Perbandingan YOLOv13n vs HOG-KNN

Penelitian Tugas Akhir: **“Deteksi Emosi Belajar Mahasiswa pada Pembelajaran Pemrograman Menggunakan YOLOv13n”**

---

## 1. Dataset
- **Dataset Path**: `D:\varell\college\tugas\semester 7\projek skripsi1\skripsi_yolov13_engagement\datasets\master_combined_dataset`
- **Train Set**: 1319 citra (digunakan untuk pelatihan bobot YOLO dan ekstraksi fitur pelatihan HOG-KNN)
- **Validation Set**: 168 citra
- **Test Set**: 173 citra (dievaluasi pada data uji yang identik 100%)
- **Kelas Ekspresi (4 Kelas)**: `0 = engaged`, `1 = confused`, `2 = bored`, `3 = frustrated`

## 2. YOLOv13n (Model Utama Penelitian)
- **Model Weights**: `D:\varell\college\tugas\semester 7\projek skripsi1\skripsi_yolov13_engagement\runs\yolov13_master_combined_v2\weights\best.pt`
- **Metrik Native Object Detection**: Precision=99.43%, Recall=98.23%, F1-Score=98.83%, mAP@0.5=99.42%, mAP@0.75=99.42%, mAP@0.5:0.95=98.15%
- **Metrik Image-Level Classification**: Akurasi = **98.84%** (171/173 benar), Macro Precision = 98.90%, Macro Recall = 98.75%, Macro F1-Score = **98.80%**, Weighted F1-Score = 98.84%
- **Detection Failures**: 0 citra (tingkat deteksi wajah = 100.0%)
- **Waktu Pemrosesan**: Native Inference = **20.20 ms** | Total Pipeline Latency = **23.10 ms** (43.3 FPS)

## 3. HOG-KNN (Baseline Klasifikasi Tradisional)
- **Preprocessing**: Crop wajah dari Ground-Truth Bounding Box -> Resize 64×64 -> Grayscale
- **Konfigurasi HOG**: Orientations=9, Pixels per Cell=8×8, Cells per Block=2×2, Block Normalization=L2-Hys (Total 1.764 fitur per wajah)
- **Konfigurasi KNN**: K = 5, Metrik Jarak = Euclidean
- **Metrik Klasifikasi**: Akurasi = **93.06%** (161/173 benar), Macro Precision = 93.56%, Macro Recall = 92.58%, Macro F1-Score = **92.90%**, Weighted F1-Score = 93.01%
- **Waktu Pemrosesan**: KNN Predict Only = **17.94 ms** | Total HOG-KNN Pipeline = **19.14 ms** (52.2 FPS)

## 4. Tabel Perbandingan YOLOv13n vs HOG-KNN

### A. Perbandingan Metrik Klasifikasi
| Model | Accuracy | Macro Precision | Macro Recall | Macro F1 | Weighted F1 | Avg Processing Time |
|:---|---:|---:|---:|---:|---:|---:|
| **YOLOv13n** | **98.84%** | **98.90%** | **98.75%** | **98.80%** | **98.84%** | **23.10 ms** |
| **HOG-KNN K=5** | 93.06% | 93.56% | 92.58% | 92.90% | 93.01% | 19.14 ms |

### B. Metrik Detection-Specific YOLO
| Model | mAP@0.5 | mAP@0.75 | mAP@0.5:0.95 | Native Inference |
|:---|---:|---:|---:|---:|
| **YOLOv13n** | **99.42%** | **99.42%** | **98.15%** | **20.20 ms** |
| **HOG-KNN** | *N/A* | *N/A* | *N/A* | *N/A* |

> *Catatan: HOG-KNN tidak menghasilkan bounding box sehingga mAP dan IoU tidak dihitung. Metrik tersebut hanya digunakan pada YOLOv13n sebagai model object detection.*

## 5. Error Analysis
- **Both Correct**: 161 citra (93.06%)
- **Both Wrong**: 2 citra (1.16%)
- **YOLO Correct, KNN Wrong**: 10 citra (5.78%)
- **YOLO Wrong, KNN Correct**: 0 citra (0.00%)

## 6. Pembahasan
Tugas dan cakupan fungsionalitas kedua metode berbeda secara fundamental:
1. **YOLOv13n** menerima citra utuh lingkungan belajar dan secara simultan melakukan lokalisasi wajah (bounding box) sekaligus mengklasifikasikan kelas ekspresi emosi dalam arsitektur end-to-end terpadu.
2. **HOG-KNN** adalah classifier tradisional yang menerima crop wajah yang telah diisolasi dari bounding box ground truth, lalu mengekstraksi histogram orientasi gradien sebelum diklasifikasikan dengan algoritma tetangga terdekat.
Oleh karena itu, perbandingan akurasi, precision, recall, dan F1-score digunakan untuk membandingkan kapasitas diskriminasi pola ekspresi kedua pendekatan, sedangkan mAP secara eksklusif menjadi tolak ukur evaluasi lokalisasi spasial pada YOLOv13n.

## 7. Kesimpulan
Berdasarkan pengujian pada 173 citra uji yang identik, model utama **YOLOv13n** memperoleh akurasi **98.84%** dan Macro F1-Score **98.80%** serta mAP@0.5 sebesar **99.42%**, sementara baseline **HOG-KNN (K=5)** memperoleh akurasi **93.06%** dan Macro F1-Score **92.90%**.

---

## Journal-ready revision

### A. Metode HOG-KNN
Sebagai metode pembanding (baseline) berbasis pembelajaran mesin konvensional, diimplementasikan algoritma *Histogram of Oriented Gradients* yang dipadukan dengan *K-Nearest Neighbors* (HOG-KNN). Ekstraksi fitur tekstur wajah dilakukan pada area wajah yang dipotong berdasarkan anotasi *ground-truth* kemudian diubah ke skala abu-abu (*grayscale*) dan diubah ukurannya menjadi 64×64 piksel. Fitur HOG diekstraksi dengan konfigurasi 9 orientasi gradien, ukuran sel 8×8 piksel, dan ukuran blok 2×2 sel dengan normalisasi L2-Hys (menghasilkan vektor fitur berdimensi 1.764). Klasifikasi emosi dilakukan menggunakan pengklasifikasi KNN dengan parameter $K = 5$ dan metrik jarak Euclidean.

### B. Evaluasi YOLOv13n vs HOG-KNN
Evaluasi komparatif dilakukan pada kumpulan data uji (*test set*) yang identik sebanyak 173 citra. YOLOv13n dievaluasi secara *end-to-end* menerima citra utuh untuk mendeteksi lokasi wajah dan mengklasifikasikan emosi, sedangkan HOG-KNN dievaluasi pada potongan citra wajah *ground-truth* untuk mengukur kemampuan klasifikasi representasi tekstur tradisional.

### C. Hasil HOG-KNN
Pengujian baseline HOG-KNN ($K=5$) menghasilkan akurasi sebesar 93.06%, *macro precision* 93.56%, *macro recall* 92.58%, dan *macro F1-score* sebesar 92.90% dengan rata-rata total waktu pemrosesan 19.14 ms per citra.

### D. Tabel Perbandingan YOLOv13n vs HOG-KNN

| Metode Pendekatan | Akurasi | Macro Precision | Macro Recall | Macro F1-Score | Weighted F1-Score | Rata-rata Latensi (ms) |
|:---|---:|---:|---:|---:|---:|---:|
| **YOLOv13n (Model Utama)** | **98.84%** | **98.90%** | **98.75%** | **98.80%** | **98.84%** | **23.10 ms** |
| **HOG-KNN K=5 (Baseline Pembanding)** | 93.06% | 93.56% | 92.58% | 92.90% | 93.01% | 19.14 ms |

### E. Pembahasan
Perbedaan mendasar antara kedua pendekatan terletak pada skema pemrosesan data. YOLOv13n melakukan lokalisasi wajah dan klasifikasi ekspresi secara simultan dari citra utuh menggunakan representasi hierarki fitur konvolusional multi-skala. Sementara itu, HOG-KNN beroperasi hanya sebagai pengklasifikasi pada potongan wajah yang telah diketahui sebelumnya (*ideal ground-truth crop*). Dengan demikian, metrik klasifikasi (akurasi, presisi, *recall*, dan F1-*score*) mencerminkan kapabilitas klasifikasi kedua metode, sedangkan metrik mAP@0.5 ({map50*100:.2f}%) secara khusus menegaskan kemampuan lokalisasi spasial objek yang hanya dimiliki oleh model utama YOLOv13n.

### F. Kesimpulan Pembanding
Hasil eksperimen menunjukkan bahwa YOLOv13n memberikan kinerja deteksi dan klasifikasi emosi belajar mahasiswa yang sangat unggul dan tangguh secara menyeluruh, mengungguli metode pembanding konvensional HOG-KNN dengan tingkat throughput mencapai 43.3 FPS, sehingga sangat optimal untuk diintegrasikan pada sistem pemantauan pembelajaran pemrograman secara *real-time*.