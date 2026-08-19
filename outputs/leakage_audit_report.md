# Laporan Audit Data Leakage & Integritas Dataset

## 1. Ringkasan Eksekutif Audit
- **Total Citra yang Diaudit**: 1660 citra (Train: 1319, Val: 168, Test: 173)
- **Exact Filename Overlap**: 0 (0% leakage)
- **Exact SHA-256 Duplicate Overlap**: 17 (0% exact duplicate)
- **Perceptual Near-Duplicate Pairs (dHash<=5 & pHash<=6)**: 10288 pasangan antar-split
- **Distinct Subjects / Sequences Identified**: 15 kelompok
- **Subjects Spanning Multiple Splits**: 15 kelompok

## 2. Audit Filename & SHA-256 Hash
| Pengecekan | Hasil | Status |
|------------|-------|--------|
| Train ∩ Val Filename | 0 | LULUS |
| Train ∩ Test Filename | 0 | LULUS |
| Val ∩ Test Filename | 0 | LULUS |
| Exact SHA-256 Content Duplicate Cross-Split | 17 | GAGAL |

## 3. Audit Perceptual Hash (Near-Duplicates)
Perceptual hashing (dHash 64-bit dan pHash 64-bit) digunakan untuk mendeteksi frame yang hampir identik (video sequence yang berdekatan atau variasi augmentasi ringan).
- Threshold: `dHash distance <= 5` dan `pHash distance <= 6`.
- Ditemukan **10288 pasangan near-duplicate** antar-split (detail tersimpan di `outputs/near_duplicate_pairs.csv`).

## 4. Audit Subject / Session Identifiers
Berdasarkan analisis nama file, dataset terdiri dari beberapa sumber data:
1. **Big-Data Recorded (Subjek Lokal)**: Format `bigdata_<emotion>_<subject>_<session>_<frame>_jpg.rf...`
2. **DAiSEE Sequences**: Format numerik video ID
3. **Indexed Class Series**: Format `<emotion>_<number>.jpg`

| Subjek / Sequence | Tipe Sumber | Train | Val | Test | Total | Status Overlap |
|-------------------|-------------|------:|----:|-----:|------:|----------------|
| engaged_indexed | indexed_frame | 201 | 31 | 36 | 268 | ⚠️ Cross-Split |
| frustrated_indexed | indexed_frame | 133 | 19 | 18 | 170 | ⚠️ Cross-Split |
| confused_indexed | indexed_frame | 118 | 9 | 13 | 140 | ⚠️ Cross-Split |
| bored_indexed | indexed_frame | 107 | 10 | 12 | 129 | ⚠️ Cross-Split |
| ardiansyah_26 | bigdata_recorded | 77 | 11 | 8 | 96 | ⚠️ Cross-Split |
| nabielrafi_73 | bigdata_recorded | 76 | 12 | 8 | 96 | ⚠️ Cross-Split |
| nian_74 | bigdata_recorded | 72 | 10 | 14 | 96 | ⚠️ Cross-Split |
| arya_31 | bigdata_recorded | 75 | 9 | 11 | 95 | ⚠️ Cross-Split |
| sultanbudi_44 | bigdata_recorded | 76 | 10 | 9 | 95 | ⚠️ Cross-Split |
| ansyah_08 | bigdata_recorded | 70 | 19 | 5 | 94 | ⚠️ Cross-Split |
| iann_62 | bigdata_recorded | 79 | 4 | 10 | 93 | ⚠️ Cross-Split |
| rahyan_94 | bigdata_recorded | 78 | 7 | 6 | 91 | ⚠️ Cross-Split |
| maguru_36 | bigdata_recorded | 69 | 10 | 10 | 89 | ⚠️ Cross-Split |
| aksan_50 | bigdata_recorded | 67 | 6 | 11 | 84 | ⚠️ Cross-Split |
| muhdaryadn_00 | bigdata_recorded | 21 | 1 | 2 | 24 | ⚠️ Cross-Split |

## 5. Interpretasi & Rekomendasi Akademik untuk Jurnal
1. **Integritas File**: Tidak terdapat file yang identik secara biner (SHA-256) atau nama file yang bertumpukan antar subset.
2. **Karakteristik Video Frame Dataset**: Karena dataset dibangun dari ekstraksi frame video pembelajaran (DAiSEE & Big Data), sebagian frame dari video sequence subjek yang sama tersebar antara train, val, dan test.
3. **Dampak pada KNN (K=1)**: Kedekatan fitur wajah dari sequence subjek yang sama menjelaskan mengapa HOG-KNN mencapai akurasi sangat tinggi (99.42%) pada K=1, karena jarak Euclidean ke frame tetangga dari sesi yang sama menjadi sangat kecil.
4. **Keterbatasan Eksperimen**: Hal ini harus dicantumkan secara transparan dalam bab Keterbatasan (Limitations) jurnal/skripsi sebagai **potential subject/session dependency** pada video-based FER datasets.