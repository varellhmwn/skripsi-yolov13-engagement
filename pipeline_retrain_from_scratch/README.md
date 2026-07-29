# 🚀 Automated Retraining Pipeline from Scratch

Folder ini dirancang khusus untuk mengeksekusi **seluruh alur kerja pra-pemrosesan data (*preprocessing*) hingga pelatihan model (*training*) YOLOv13n dari nol (*from scratch*)** secara mandiri dan terdokumentasi utuh.

---

## 📁 Struktur Direktori

```
pipeline_retrain_from_scratch/
├── raw_datasets/                             ← Data mentah (ZIP asli)
│   ├── student_engagement_roboflow.zip       (953 gambar Roboflow)
│   └── big_data_hard_samples.zip             (745 gambar Hard Samples)
├── scripts/                                  ← Skrip pra-pemrosesan & training
│   ├── 01_extract_and_filter_roboflow.py     (Filter 4 kelas Roboflow)
│   ├── 02_build_master_dataset.py             (Gabung Roboflow + Hard Samples)
│   ├── 03_stratified_resplit_80_10_10.py      (Stratified Split 80:10:10, seed=42)
│   └── 04_train_yolov13.py                    (Pelatihan YOLOv13n 150 epoch)
├── run_full_pipeline.py                      ← Skrip eksekusi otomatis 1-Klik
└── README.md                                 ← Dokumentasi & panduan
```

---

## ⚡ Cara Eksekusi

### **Opsi A: Jalankan Seluruh Pipeline (Preprocessing + Training)**
Jalankan satu perintah ini di terminal:
```bash
python pipeline_retrain_from_scratch/run_full_pipeline.py
```

### **Opsi B: Jalankan Preprocessing Saja (Tanpa Training)**
Jika Anda hanya ingin mengevaluasi pra-pemrosesan data:
```bash
python pipeline_retrain_from_scratch/run_full_pipeline.py --skip-train
```

### **Opsi C: Jalankan Tahap Secara Manual Satu per Satu**
1. **Tahap 1 — Filter Roboflow**:
   ```bash
   python pipeline_retrain_from_scratch/scripts/01_extract_and_filter_roboflow.py
   ```
2. **Tahap 2 — Gabung Hard Samples**:
   ```bash
   python pipeline_retrain_from_scratch/scripts/02_build_master_dataset.py
   ```
3. **Tahap 3 — Stratified Split 80:10:10**:
   ```bash
   python pipeline_retrain_from_scratch/scripts/03_stratified_resplit_80_10_10.py
   ```
4. **Tahap 4 — Training YOLOv13n**:
   ```bash
   python pipeline_retrain_from_scratch/scripts/04_train_yolov13.py
   ```

---

## 📊 Hasil Dokumentasi yang Dihasilkan

Setelah dieksekusi, sistem akan menghasilkan log otentik yang dapat dilampirkan pada naskah skripsi Anda:
- `pipeline_execution_log.txt`: Catatan waktu dan statistik eksekusi lengkap.
- `datasets_processed/03_master_combined_80_10_10/data.yaml`: File acuan training.
- `runs/yolov13n_retrained_scratch/weights/best.pt`: Bobot model terbaik.
- `runs/yolov13n_retrained_scratch/results.csv`: Tabel metrik mAP, Precision, Recall per epoch.
