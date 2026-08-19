# Ringkasan Hasil Eksperimen Siap Publikasi Jurnal (Group-Wise)

## A. Metode Group-Wise Stratified Split
Untuk mencegah overoptimisme evaluasi akibat autokorelasi temporal antar-frame pada rekaman video pembelajaran, pembagian dataset dilakukan menggunakan pendekatan **Group-Wise Stratified Split**. Seluruh 1.660 citra dikelompokkan ke dalam 181 group independen berdasarkan identitas subjek video rekaman, komponen terhubung *exact binary duplicate* (SHA-256), dan klaster *near-duplicate* berkeyakinan tinggi. Pembagian ke dalam subset data latih (1.327 citra, 79.94%), validasi (167 citra, 10.06%), dan pengujian (166 citra, 10.00%) dilakukan pada level group secara utuh dengan menjamin 0% tumpang tindih subjek maupun duplikasi antar-subset.

## B. Baseline HOG-KNN dan Penentuan Hyperparameter K
Model pembanding HOG-KNN dilatih menggunakan fitur tekstur wajah (resolusi 64×64 piksel, grayscale, 9 orientasi, sel 8×8, blok 2×2, normalisasi L2-Hys) yang diekstraksi dari crop anotasi ground truth. Penentuan hyperparameter jumlah tetangga K dilakukan murni pada validation set group-wise dengan menguji nilai K ∈ {1, 3, 5, 7, 9, 11, 13, 15} menggunakan metrik jarak Euclidean. Berdasarkan kriteria Macro F1-score tertinggi, terpilih **K = 1** dengan Macro F1 sebesar 32.24% dan akurasi 32.93%.

## C. Evaluasi dan Perbandingan Performa Model Utama vs Pembanding
Hasil pengujian komparatif pada test set group-wise independen (166 citra) menunjukkan bahwa model utama **YOLOv13n** memperoleh akurasi sebesar **62.65%**, Macro F1-score **61.80%**, dan mAP@0.5 mencapai **70.78%** (mAP@0.5:0.95: **63.70%**). Baseline **HOG-KNN Ground-Truth Crop** memperoleh akurasi **25.90%** dan Macro F1 **25.35%**. Sementara itu, pipeline hybrid **YOLO Crop + HOG-KNN** menghasilkan akurasi **31.93%** dan Macro F1 **29.12%** dengan tingkat keberhasilan lokalisasi 100% (0 detection failure).

## D. Efisiensi Komputasi dan Throughput Real-Time
Berdasarkan benchmarking single-image terstandarisasi pada GPU NVIDIA GeForce RTX 4060 Laptop (20 iterasi warm-up), YOLOv13n mencatat total wall-clock latency sebesar **25.35 ms** (**39.4 FPS**). Pipeline hybrid membutuhkan waktu lebih lama yaitu **29.16 ms** (**34.3 FPS**) karena menjalankan inferensi dua model secara serial.

## E. Tabel Perbandingan untuk Naskah Jurnal

| Model Pendekatan | Akurasi | Macro F1 | Weighted F1 | mAP@0.5 | Latency (ms) | Throughput (FPS) |
|:-----------------|--------:|---------:|------------:|--------:|-------------:|-----------------:|
| **YOLOv13n** | 62.65% | 61.80% | 62.46% | 70.78% | 25.35 ms | 39.4 FPS |
| **HOG-KNN GT Crop** | 25.90% | 25.35% | 26.72% | N/A | 3.50 ms | 285.4 FPS |
| **YOLO Crop + HOG-KNN** | 31.93% | 29.12% | 30.24% | N/A | 29.16 ms | 34.3 FPS |

*\*Catatan: Metrik mAP tidak berlaku untuk HOG-KNN karena model tidak melakukan prediksi bounding box.*