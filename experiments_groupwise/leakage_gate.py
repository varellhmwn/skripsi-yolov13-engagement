"""
leakage_gate.py — Validasi Ketat Leakage Gate Sebelum Training YOLO
====================================================================
Memastikan secara mutlak:
  1. Group overlap train <-> val, train <-> test, val <-> test = 0
  2. Exact SHA-256 duplicate cross-split = 0
  3. Total citra tepat 1.660
Output:
  - outputs_groupwise/leakage_validation_report.md
"""

import sys
from pathlib import Path
from collections import defaultdict
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from experiments_groupwise.config import OUTPUT_GROUPWISE_DIR
from experiments_groupwise.split_proposal import generate_split_proposal


def validate_leakage_gate():
    print("=" * 65)
    print("  TAHAP 4: VALIDASI LEAKAGE GATE SEBELUM TRAINING YOLO")
    print("=" * 65)

    proposal_path = OUTPUT_GROUPWISE_DIR / 'split_proposal.csv'
    if not proposal_path.exists():
        df_proposal = generate_split_proposal()
    else:
        df_proposal = pd.read_csv(proposal_path)

    # 1. Total count check
    total_imgs = len(df_proposal)
    assert total_imgs == 1660, f"Total image count harus tepat 1660, didapat {total_imgs}"

    # 2. Group Overlap Check
    train_groups = set(df_proposal[df_proposal['new_split'] == 'train']['group_id'])
    val_groups = set(df_proposal[df_proposal['new_split'] == 'val']['group_id'])
    test_groups = set(df_proposal[df_proposal['new_split'] == 'test']['group_id'])

    tv_group_overlap = train_groups & val_groups
    tt_group_overlap = train_groups & test_groups
    vt_group_overlap = val_groups & test_groups

    # 3. Exact SHA-256 Duplicate Check
    sha_splits = defaultdict(set)
    for _, row in df_proposal.iterrows():
        sha_splits[row['sha256']].add(row['new_split'])

    cross_split_shas = [sha for sha, splits in sha_splits.items() if len(splits) > 1]

    # Gate verification
    gate_passed = True
    errors = []

    if len(tv_group_overlap) > 0:
        gate_passed = False
        errors.append(f"Group overlap Train & Val: {len(tv_group_overlap)} groups ({tv_group_overlap})")
    if len(tt_group_overlap) > 0:
        gate_passed = False
        errors.append(f"Group overlap Train & Test: {len(tt_group_overlap)} groups ({tt_group_overlap})")
    if len(vt_group_overlap) > 0:
        gate_passed = False
        errors.append(f"Group overlap Val & Test: {len(vt_group_overlap)} groups ({vt_group_overlap})")
    if len(cross_split_shas) > 0:
        gate_passed = False
        errors.append(f"Exact SHA-256 duplicate cross-split: {len(cross_split_shas)} hashes")

    # Generate Markdown Report
    rep_lines = [
        "# Laporan Validasi Leakage Gate (Group-Wise Split)",
        "\n## 1. Status Pengecekan Integritas Data",
        "| Parameter Pengujian | Target Standar | Hasil Aktual | Status Gate |",
        "|:--------------------|:---------------|:-------------|:------------|",
        f"| **Total Citra Dataset** | Tepat 1.660 Citra | {total_imgs} Citra | {'✓ LULUS' if total_imgs==1660 else '✗ GAGAL'} |",
        f"| **Group Overlap Train ↔ Val** | 0 Group | {len(tv_group_overlap)} Group | {'✓ LULUS (0 Overlap)' if len(tv_group_overlap)==0 else '✗ GAGAL'} |",
        f"| **Group Overlap Train ↔ Test** | 0 Group | {len(tt_group_overlap)} Group | {'✓ LULUS (0 Overlap)' if len(tt_group_overlap)==0 else '✗ GAGAL'} |",
        f"| **Group Overlap Val ↔ Test** | 0 Group | {len(vt_group_overlap)} Group | {'✓ LULUS (0 Overlap)' if len(vt_group_overlap)==0 else '✗ GAGAL'} |",
        f"| **Exact SHA-256 Cross-Split** | 0 Hash | {len(cross_split_shas)} Hash | {'✓ LULUS (0 Duplikat)' if len(cross_split_shas)==0 else '✗ GAGAL'} |",
        "\n## 2. Kesimpulan Leakage Gate",
    ]

    if gate_passed:
        rep_lines.append("**STATUS: LOLOS VERIFIKASI (PASSED)**")
        rep_lines.append("Dataset group-wise memenuhi seluruh kriteria independensi subset: tidak terdapat kebocoran subjek, group, ataupun file identik antar data latih, validasi, dan uji. Dataset siap dimaterialisasi dan digunakan untuk retraining YOLOv13n.")
    else:
        rep_lines.append("**STATUS: GAGAL VERIFIKASI (FAILED)**")
        for err in errors:
            rep_lines.append(f"- {err}")

    with open(OUTPUT_GROUPWISE_DIR / 'leakage_validation_report.md', 'w', encoding='utf-8') as f:
        f.write('\n'.join(rep_lines))

    print(f"  [SAVED] {OUTPUT_GROUPWISE_DIR / 'leakage_validation_report.md'}")

    if not gate_passed:
        raise RuntimeError(f"Leakage Gate Gagal: {errors}")

    print("  ✓ LEAKAGE GATE PASSED (Semua parameter integritas 100% LULUS)!\n")
    return gate_passed


if __name__ == '__main__':
    validate_leakage_gate()
