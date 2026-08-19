# Laporan Validasi Leakage Gate (Group-Wise Split)

## 1. Status Pengecekan Integritas Data
| Parameter Pengujian | Target Standar | Hasil Aktual | Status Gate |
|:--------------------|:---------------|:-------------|:------------|
| **Total Citra Dataset** | Tepat 1.660 Citra | 1660 Citra | ✓ LULUS |
| **Group Overlap Train ↔ Val** | 0 Group | 0 Group | ✓ LULUS (0 Overlap) |
| **Group Overlap Train ↔ Test** | 0 Group | 0 Group | ✓ LULUS (0 Overlap) |
| **Group Overlap Val ↔ Test** | 0 Group | 0 Group | ✓ LULUS (0 Overlap) |
| **Exact SHA-256 Cross-Split** | 0 Hash | 0 Hash | ✓ LULUS (0 Duplikat) |

## 2. Kesimpulan Leakage Gate
**STATUS: LOLOS VERIFIKASI (PASSED)**
Dataset group-wise memenuhi seluruh kriteria independensi subset: tidak terdapat kebocoran subjek, group, ataupun file identik antar data latih, validasi, dan uji. Dataset siap dimaterialisasi dan digunakan untuk retraining YOLOv13n.