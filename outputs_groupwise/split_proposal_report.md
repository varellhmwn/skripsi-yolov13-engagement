# Laporan Proposal Group-Wise Stratified Split (1.660 Citra)

## 1. Ringkasan Pembagian Dataset Group-Wise
| Split Subset | Jumlah Citra | Persentase | Jumlah Group | Engaged (0) | Confused (1) | Bored (2) | Frustrated (3) | Roboflow | Hard Samples |
|:-------------|-------------:|-----------:|-------------:|------------:|-------------:|----------:|---------------:|---------:|-------------:|
| **Train** | **1327** | 79.94% | 126 | 390 | 316 | 293 | 328 | 770 | 557 |
| **Validation** | **167** | 10.06% | 18 | 58 | 32 | 37 | 40 | 94 | 73 |
| **Test** | **166** | 10.00% | 37 | 49 | 38 | 38 | 41 | 89 | 77 |
| **TOTAL** | **1660** | **100.00%** | **181** | **497** | **386** | **368** | **409** | **953** | **707** |

## 2. Prinsip & Keunggulan Metodologis Group-Wise Split
1. **Zero Cross-Split Subject Overlap**: Seluruh frame dari 1 subjek/sekuens yang sama dimasukkan secara utuh ke dalam 1 subset.
2. **Zero Exact Duplicate Leakage**: Semua pasangan citra yang identik secara biner (SHA-256) dipaksa berada di subset yang sama.
3. **Integritas Total Dataset**: Total citra dipertahankan tepat **1.660 citra** (Train: 1.327, Val: 167, Test: 166).
4. **Keseimbangan Kelas**: Proporsi keempat kelas emosi tetap terjaga seimbang di semua subset.