# README: Analisis Log Prediksi Real-Time YOLOv13n

## Cara Menjalankan

```bash
python scripts/analyze_realtime_log.py outputs/realtime_smoothed/realtime_prediction_log.csv
```

Seluruh file output akan disimpan di folder yang sama dengan file CSV input.

---

## Daftar Output

| File | Deskripsi |
|------|-----------|
| `realtime_analysis_summary.csv` | Ringkasan metrik dalam format tabel (metric, raw, postprocessed, change) |
| `realtime_analysis_summary.json` | Ringkasan metrik dalam format JSON untuk penggunaan ulang |
| `raw_segments.csv` | Detail setiap segmen pada prediksi mentah (raw_class_name) |
| `stable_segments.csv` | Detail setiap segmen pada prediksi stabil (stable_label) |
| `label_changes_comparison.png` | Grafik batang perbandingan jumlah perubahan label |
| `short_segments_comparison.png` | Grafik batang perbandingan segmen pendek (< 8 frame) |
| `fps_over_time.png` | Grafik garis FPS terhadap waktu dengan garis target 30 FPS |
| `raw_vs_stable_timeline.png` | Dua panel timeline emosi (raw vs stable) |

---

## Penjelasan Metrik

### Performa Real-Time (FPS)

| Metrik | Arti |
|--------|------|
| **Total frames** | Jumlah total frame yang dianalisis dari webcam |
| **Test duration** | Durasi pengujian berdasarkan timestamp terakhir dikurangi timestamp pertama |
| **Mean FPS** | Rata-rata *frames per second* setelah mengecualikan 10 frame *warm-up* |
| **Median FPS** | Nilai tengah dari distribusi FPS |
| **P5 - P95** | Persentil ke-5 dan ke-95; menunjukkan rentang FPS pada 90% data |
| **Frames >= 30 FPS** | Jumlah dan persentase frame yang mencapai target *real-time* 30 FPS |

### Analisis Post-Processing

| Metrik | Arti |
|--------|------|
| **Raw label changes** | Jumlah kali label berubah antar-frame pada prediksi mentah YOLOv13n |
| **Stable label changes** | Jumlah kali label berubah antar-frame setelah *temporal post-processing* (sliding window + majority voting) |
| **Reduction (%)** | Persentase penurunan perubahan label: `(raw - stable) / raw * 100` |

### Analisis Segmentasi

| Metrik | Arti |
|--------|------|
| **Segmen** | Rangkaian frame berturut-turut dengan label yang sama |
| **Segmen < 8 frame** | Segmen yang durasinya sangat pendek (kurang dari ~0,27 detik pada 30 FPS), sering kali merupakan *noise* |
| **Median panjang segmen** | Nilai tengah panjang segmen dalam frame |
| **Median durasi segmen** | Nilai tengah durasi segmen dalam detik, dihitung dari selisih `timestamp_sec` |

### Status Neutral

| Metrik | Arti |
|--------|------|
| **Neutral frames** | Jumlah frame di mana `stable_label = 'neutral'` |
| **Neutral percentage** | Persentase frame neutral terhadap seluruh data |
| **Neutral segments** | Jumlah segmen berturut-turut berstatus neutral |
| **Neutral median duration** | Median durasi segmen neutral (detik) |

> **Catatan penting:** `neutral` bukan merupakan salah satu dari 4 kelas emosi yang dilatih pada model YOLOv13n. Status ini muncul ketika rata-rata *confidence* model pada *sliding window* berada di bawah *threshold* (`min_avg_confidence`), yang menandakan bahwa model tidak cukup yakin terhadap ekspresi wajah yang terdeteksi.

---

## Interpretasi yang Benar

*Temporal post-processing* **tidak meningkatkan akurasi** model YOLOv13n.

Interpretasi yang tepat adalah:

> *"Temporal post-processing digunakan untuk mengurangi perubahan label berdurasi singkat dan meningkatkan kestabilan penyajian prediksi kepada pengguna."*

Stabilisasi ini penting untuk pengalaman pengguna (*user experience*) pada aplikasi *Learning Analytics Dashboard*, agar tampilan emosi di layar tidak berkedip-kedip (*flickering*) akibat *noise* sesaat.
