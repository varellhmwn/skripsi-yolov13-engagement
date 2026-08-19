"""
group_discovery.py — Identifikasi Group / Subject / Sequence / Duplicate Linkage
==================================================================================
Membangun pemetaan group terstruktur untuk seluruh 1.660 citra berdasarkan hirarki:
  1. Identitas subjek eksplisit pada Roboflow bigdata (11 subjek)
  2. Komponen terhubung Exact Duplicate (SHA-256 identik)
  3. Komponen terhubung High-Confidence Near-Duplicates (dHash <= 2, pHash <= 2)
  4. Sequence / discrete image fallback
Output:
  - outputs_groupwise/group_manifest.csv
  - outputs_groupwise/exact_duplicate_groups.csv
  - outputs_groupwise/near_duplicate_groups.csv
"""

import sys
import hashlib
import re
from pathlib import Path
from collections import defaultdict
import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from experiments_groupwise.config import (
    ORIGINAL_DATASET_DIR, OUTPUT_GROUPWISE_DIR, CLASS_NAMES, VALID_IMG_EXTS
)


def compute_sha256(filepath):
    h = hashlib.sha256()
    with open(filepath, 'rb') as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()


def compute_dhash(img_path, hash_size=8):
    img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return np.zeros(hash_size * hash_size, dtype=bool)
    resized = cv2.resize(img, (hash_size + 1, hash_size), interpolation=cv2.INTER_AREA)
    diff = resized[:, 1:] > resized[:, :-1]
    return diff.flatten()


def compute_phash(img_path, hash_size=8, highfreq_factor=4):
    img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return np.zeros(hash_size * hash_size, dtype=bool)
    img_size = hash_size * highfreq_factor
    resized = cv2.resize(img, (img_size, img_size), interpolation=cv2.INTER_AREA)
    dct = cv2.dct(np.float32(resized))
    dct_low = dct[:hash_size, :hash_size]
    med = np.median(dct_low)
    return (dct_low > med).flatten()


def discover_groups():
    print("=" * 65)
    print("  TAHAP 2: IDENTIFIKASI & PEMBENTUKAN GROUP (1.660 CITRA)")
    print("=" * 65)

    OUTPUT_GROUPWISE_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Muat seluruh 1.660 citra dan anotasi
    all_records = []
    splits = ['train', 'val', 'test']

    for split in splits:
        img_dir = ORIGINAL_DATASET_DIR / 'images' / split
        lbl_dir = ORIGINAL_DATASET_DIR / 'labels' / split

        for img_path in sorted(img_dir.iterdir()):
            if img_path.suffix.lower() not in VALID_IMG_EXTS:
                continue

            lbl_path = lbl_dir / f"{img_path.stem}.txt"
            if not lbl_path.exists():
                continue

            # Read primary class
            class_id = -1
            with open(lbl_path, 'r', encoding='utf-8') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        class_id = int(parts[0])
                        break

            source = 'roboflow' if 'bigdata' in img_path.name else 'hard_samples'
            all_records.append({
                'filename': img_path.name,
                'orig_split': split,
                'img_path': str(img_path),
                'lbl_path': str(lbl_path),
                'class_id': class_id,
                'class_name': CLASS_NAMES.get(class_id, 'unknown'),
                'source': source,
                'sha256': compute_sha256(img_path)
            })

    df = pd.DataFrame(all_records)
    print(f"  Total citra dimuat: {len(df)} citra (Roboflow: {(df['source']=='roboflow').sum()}, Hard Samples: {(df['source']=='hard_samples').sum()})")

    # 2. Perceptual Hashing
    print("  Menghitung perceptual hashes (dHash 64-bit & pHash 64-bit)...")
    df['dhash'] = [compute_dhash(p) for p in df['img_path']]
    df['phash'] = [compute_phash(p) for p in df['img_path']]

    # 3. Union-Find Data Structure untuk pembentukan group
    parent = list(range(len(df)))
    rule_applied = ['discrete_sample'] * len(df)
    confidence_level = ['medium'] * len(df)

    def find(i):
        if parent[i] == i:
            return i
        parent[i] = find(parent[i])
        return parent[i]

    def union(i, j, rule, conf='high'):
        root_i = find(i)
        root_j = find(j)
        if root_i != root_j:
            parent[root_i] = root_j
            rule_applied[root_j] = rule
            confidence_level[root_j] = conf

    # --- Rule 1: Explicit Subjek pada Roboflow Bigdata ---
    print("  [Rule 1] Mengelompokkan berdasarkan identitas subjek Roboflow...")
    subject_map = defaultdict(list)
    for idx, row in df.iterrows():
        fn = row['filename']
        if row['source'] == 'roboflow':
            m = re.search(r'bigdata_[a-z_]+_([a-z]+_\d+)_', fn)
            if not m:
                m = re.search(r'bigdata_[a-z_]+_([a-z]+)_', fn)
            subj = m.group(1) if m else 'unknown_roboflow'
            subject_map[f"roboflow_subject_{subj}"].append(idx)

    for subj_name, indices in subject_map.items():
        for k in range(1, len(indices)):
            union(indices[0], indices[k], 'filename_subject_session', 'high')

    # --- Rule 2: Exact SHA-256 Duplicates across Entire Dataset ---
    print("  [Rule 2] Menyatukan pasangan Exact SHA-256 Duplicates...")
    sha_map = defaultdict(list)
    for idx, row in df.iterrows():
        sha_map[row['sha256']].append(idx)

    exact_dup_rows = []
    dup_cluster_id = 0
    for sha, indices in sha_map.items():
        if len(indices) > 1:
            dup_cluster_id += 1
            for k in range(1, len(indices)):
                union(indices[0], indices[k], 'exact_duplicate_component', 'high')
            for i in indices:
                exact_dup_rows.append({
                    'exact_cluster_id': dup_cluster_id,
                    'sha256': sha,
                    'filename': df.loc[i, 'filename'],
                    'source': df.loc[i, 'source'],
                    'class_name': df.loc[i, 'class_name'],
                    'orig_split': df.loc[i, 'orig_split']
                })

    df_exact_dups = pd.DataFrame(exact_dup_rows)
    df_exact_dups.to_csv(OUTPUT_GROUPWISE_DIR / 'exact_duplicate_groups.csv', index=False)
    print(f"    - Ditemukan {dup_cluster_id} cluster exact duplicate ({len(exact_dup_rows)} citra total).")

    # --- Rule 3: High-Confidence Near-Duplicates for Hard Samples ---
    print("  [Rule 3] Menyatukan High-Confidence Near-Duplicates pada Hard Samples...")
    hard_indices = [idx for idx, r in df.iterrows() if r['source'] == 'hard_samples']
    near_dup_rows = []

    for i in range(len(hard_indices)):
        idx_i = hard_indices[i]
        for j in range(i + 1, len(hard_indices)):
            idx_j = hard_indices[j]
            if df.loc[idx_i, 'class_id'] == df.loc[idx_j, 'class_id']:
                dh_dist = int(np.count_nonzero(df.loc[idx_i, 'dhash'] != df.loc[idx_j, 'dhash']))
                ph_dist = int(np.count_nonzero(df.loc[idx_i, 'phash'] != df.loc[idx_j, 'phash']))

                if dh_dist <= 2 and ph_dist <= 2:
                    union(idx_i, idx_j, 'near_duplicate_component', 'high')
                    near_dup_rows.append({
                        'file_a': df.loc[idx_i, 'filename'],
                        'file_b': df.loc[idx_j, 'filename'],
                        'class_name': df.loc[idx_i, 'class_name'],
                        'dhash_dist': dh_dist,
                        'phash_dist': ph_dist,
                        'confidence': 'high_confidence'
                    })

    df_near_dups = pd.DataFrame(near_dup_rows)
    df_near_dups.to_csv(OUTPUT_GROUPWISE_DIR / 'near_duplicate_groups.csv', index=False)
    print(f"    - Ditemukan {len(near_dup_rows)} relasi high-confidence near duplicate.")

    # 4. Generate Final Group Mapping
    group_roots = [find(i) for i in range(len(df))]
    # Re-index groups to clean 1..N
    unique_roots = sorted(list(set(group_roots)))
    root_to_gid = {r: f"GRP_{idx+1:03d}" for idx, r in enumerate(unique_roots)}

    df['group_id'] = [root_to_gid[r] for r in group_roots]
    df['group_rule'] = [rule_applied[find(i)] for i in range(len(df))]
    df['group_confidence'] = [confidence_level[find(i)] for i in range(len(df))]

    manifest_cols = ['filename', 'source', 'class_name', 'class_id', 'group_id', 'group_rule', 'group_confidence', 'sha256', 'orig_split', 'img_path', 'lbl_path']
    df_manifest = df[manifest_cols]
    df_manifest.to_csv(OUTPUT_GROUPWISE_DIR / 'group_manifest.csv', index=False)

    print(f"\n  HASIL PEMBENTUKAN GROUP:")
    print(f"  - Total Group Terbentuk : {len(unique_roots)} groups")
    print(f"  - Roboflow Subjek Group : {len(subject_map)} groups (953 citra)")
    print(f"  - Hard Samples Groups   : {len(unique_roots) - len(subject_map)} groups (707 citra)")
    print(f"  [SAVED] {OUTPUT_GROUPWISE_DIR / 'group_manifest.csv'}")
    print(f"  [SAVED] {OUTPUT_GROUPWISE_DIR / 'exact_duplicate_groups.csv'}")
    print(f"  [SAVED] {OUTPUT_GROUPWISE_DIR / 'near_duplicate_groups.csv'}")

    return df_manifest


if __name__ == '__main__':
    discover_groups()
