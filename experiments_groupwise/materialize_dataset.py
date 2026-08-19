"""
materialize_dataset.py — Materialisasi Direktori Dataset Group-Wise v1
======================================================================
Membangun struktur dataset fisik datasets/master_combined_groupwise_v1/
tanpa memodifikasi atau menghapus dataset lama.
"""

import sys
import shutil
from pathlib import Path
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from experiments_groupwise.config import (
    GROUPWISE_DATASET_DIR, GROUPWISE_DATA_YAML, OUTPUT_GROUPWISE_DIR,
    CLASS_LIST
)
from experiments_groupwise.leakage_gate import validate_leakage_gate


def materialize_groupwise_dataset():
    print("=" * 65)
    print("  TAHAP 5: MATERIALISASI STRUKTUR DATASET GROUP-WISE v1")
    print("=" * 65)

    validate_leakage_gate()

    proposal_path = OUTPUT_GROUPWISE_DIR / 'split_proposal.csv'
    df_proposal = pd.read_csv(proposal_path)

    # 1. Bersihkan direktori tujuan baru jika ada
    if GROUPWISE_DATASET_DIR.exists():
        print(f"  Membersihkan direktori groupwise lama: {GROUPWISE_DATASET_DIR}...")
        shutil.rmtree(GROUPWISE_DATASET_DIR)

    # 2. Buat folder-folder subset baru
    for split in ['train', 'val', 'test']:
        (GROUPWISE_DATASET_DIR / 'images' / split).mkdir(parents=True, exist_ok=True)
        (GROUPWISE_DATASET_DIR / 'labels' / split).mkdir(parents=True, exist_ok=True)

    # 3. Salin citra dan label ke lokasi baru
    print(f"  Menyalin 1.660 citra dan label ke direktori group-wise baru...")
    copied_counts = {'train': 0, 'val': 0, 'test': 0}

    for _, row in df_proposal.iterrows():
        new_s = row['new_split']
        src_img = Path(row['img_path'])
        src_lbl = Path(row['lbl_path'])

        dst_img = GROUPWISE_DATASET_DIR / 'images' / new_s / src_img.name
        dst_lbl = GROUPWISE_DATASET_DIR / 'labels' / new_s / src_lbl.name

        shutil.copy2(src_img, dst_img)
        shutil.copy2(src_lbl, dst_lbl)
        copied_counts[new_s] += 1

    # 4. Buat data.yaml baru
    # Gunakan absolute path agar aman di semua runner Ultralytics Windows
    data_yaml_dict = {
        'path': str(GROUPWISE_DATASET_DIR.resolve()),
        'train': 'images/train',
        'val': 'images/val',
        'test': 'images/test',
        'nc': len(CLASS_LIST),
        'names': CLASS_LIST
    }

    with open(GROUPWISE_DATA_YAML, 'w', encoding='utf-8') as f:
        yaml.dump(data_yaml_dict, f, default_flow_style=False, sort_keys=False)

    print(f"\n  [BERHASIL DIMATERIALISASI]:")
    print(f"  - Train : {copied_counts['train']} citra & label ({copied_counts['train']/1660*100:.2f}%)")
    print(f"  - Val   : {copied_counts['val']} citra & label ({copied_counts['val']/1660*100:.2f}%)")
    print(f"  - Test  : {copied_counts['test']} citra & label ({copied_counts['test']/1660*100:.2f}%)")
    print(f"  - Total : {sum(copied_counts.values())} citra")
    print(f"  - YAML  : {GROUPWISE_DATA_YAML}")
    print("=" * 65)


if __name__ == '__main__':
    materialize_groupwise_dataset()
