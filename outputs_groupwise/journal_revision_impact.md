# Pemetaan Revisi Naskah Jurnal (Journal Revision Impact)

| Bagian Naskah Jurnal | Pernyataan Eksisting (Old Split) | Hasil / Revisi yang Diperlukan (Group-Wise Split) | Alasan Metodologis |
|:---------------------|:---------------------------------|:---------------------------------------------------|:-------------------|
| **Judul** | *Tetap*: Deteksi Emosi Belajar Mahasiswa Menggunakan YOLOv13n | **Tidak Berubah** | YOLOv13n tetap model utama penelitian dan aplikasi |
| **Metode Split Dataset** | Random train/val/test split (1319 / 168 / 173) | Group-wise stratified split berbasis subjek/sekuens (1327 / 167 / 166) | Menghilangkan data leakage akibat korelasi frame video dari subjek yang sama |
| **Hyperparameter KNN** | Tuning K pada validation set | K = 1 terpilih berdasarkan Macro F1 validation | Penyesuaian hyperparameter pada dataset bebas leakage |
| **Hasil YOLOv13n** | Akurasi ~98.84%, Macro F1 ~98.80% | Akurasi 62.65%, Macro F1 61.80%, mAP@0.5 70.78% | Hasil evaluasi murni tanpa kebocoran subjek |
| **Hasil HOG-KNN GT** | Akurasi ~99.42%, Macro F1 ~99.45% | Akurasi 25.90%, Macro F1 25.35% | Baseline klasifikasi tekstur pada crop ideal |
| **Hasil Hybrid** | Akurasi ~99.42%, Macro F1 ~99.39% | Akurasi 31.93%, Macro F1 29.12% | Evaluasi lokalisasi otomatis + klasifikasi HOG-KNN |
| **Diskusi & Keterbatasan** | Belum mendiskusikan dependensi video frame | Menjelaskan evaluasi group-wise sebagai pengujian generalisasi subjek baru | Meningkatkan derajat objektivitas dan integritas ilmiah naskah |