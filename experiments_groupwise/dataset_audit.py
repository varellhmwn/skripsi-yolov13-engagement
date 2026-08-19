"""
dataset_audit.py — Audit Komprehensif Dataset Master Sebelum Group-Wise Split
=============================================================================
Memisahkan secara eksplisit antara:
  - image_count (jumlah file citra)
  - instance_count (jumlah bounding box / anotasi emosi)
Memverifikasi integritas 1.660 citra dan mengidentifikasi orphan labels.
"""

import sys
import json
from pathlib import Path
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from experiments_groupwise.config import (
    ORIGINAL_DATASET_DIR, OUTPUT_GROUPWISE_DIR, CLASS_NAMES, VALID_IMG_EXTS
)


def audit_original_dataset():
    print("=" * 65)
    print("  TAHAP 1: AUDIT DATASET SEBELUM GROUP-WISE SPLIT")
    print("=" * 65)

    OUTPUT_GROUPWISE_DIR.mkdir(parents=True, exist_ok=True)

    splits = ['train', 'val', 'test']
    audit_summary = []
    orphan_images = []
    orphan_labels = []
    detailed_data = {}

    all_valid_images = []

    for split in splits:
        img_dir = ORIGINAL_DATASET_DIR / 'images' / split
        lbl_dir = ORIGINAL_DATASET_DIR / 'labels' / split

        img_files = {f.stem: f for f in img_dir.iterdir() if f.suffix.lower() in VALID_IMG_EXTS} if img_dir.exists() else {}
        lbl_files = {f.stem: f for f in lbl_dir.iterdir() if f.suffix.lower() == '.txt'} if lbl_dir.exists() else {}

        clean_lbl_files = {stem: p for stem, p in lbl_files.items() if stem != 'labels' and p.name != 'labels.txt'}

        # Orphan check
        for stem, img_p in img_files.items():
            if stem not in clean_lbl_files:
                orphan_images.append({'split': split, 'filename': img_p.name, 'path': str(img_p)})
            else:
                all_valid_images.append({'orig_split': split, 'stem': stem, 'img_path': img_p, 'lbl_path': clean_lbl_files[stem]})

        for stem, lbl_p in clean_lbl_files.items():
            if stem not in img_files:
                orphan_labels.append({'split': split, 'filename': lbl_p.name, 'path': str(lbl_p)})

        # Instance counting
        class_counts = {0: 0, 1: 0, 2: 0, 3: 0}
        images_with_multi_bbox = []
        images_with_zero_bbox = []
        total_bboxes = 0

        for stem, img_p in img_files.items():
            lbl_p = clean_lbl_files.get(stem)
            if not lbl_p:
                images_with_zero_bbox.append(img_p.name)
                continue

            bboxes_in_img = 0
            with open(lbl_p, 'r', encoding='utf-8') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        try:
                            cid = int(parts[0])
                            if cid in class_counts:
                                class_counts[cid] += 1
                                bboxes_in_img += 1
                                total_bboxes += 1
                        except ValueError:
                            pass

            if bboxes_in_img == 0:
                images_with_zero_bbox.append(img_p.name)
            elif bboxes_in_img > 1:
                images_with_multi_bbox.append({'filename': img_p.name, 'bbox_count': bboxes_in_img})

        split_stat = {
            'split': split,
            'image_count': len(img_files),
            'label_file_count_total': len(lbl_files),
            'orphan_images_count': len([o for o in orphan_images if o['split'] == split]),
            'orphan_labels_count': len([o for o in orphan_labels if o['split'] == split]),
            'instance_count_in_valid_images': total_bboxes,
            'instance_count_per_class': {CLASS_NAMES[c]: class_counts[c] for c in sorted(class_counts.keys())},
            'images_with_single_bbox': len(img_files) - len(images_with_multi_bbox) - len(images_with_zero_bbox),
            'images_with_multi_bbox_count': len(images_with_multi_bbox),
            'images_with_zero_bbox_count': len(images_with_zero_bbox)
        }
        detailed_data[split] = split_stat

        audit_summary.append({
            'Split': split,
            'Image Count': len(img_files),
            'Label Files': len(lbl_files),
            'Orphan Images': split_stat['orphan_images_count'],
            'Orphan Labels': split_stat['orphan_labels_count'],
            'Valid Instances': total_bboxes,
            'Engaged (0)': class_counts[0],
            'Confused (1)': class_counts[1],
            'Bored (2)': class_counts[2],
            'Frustrated (3)': class_counts[3],
            'Multi-BBox': len(images_with_multi_bbox),
            'Zero-BBox': len(images_with_zero_bbox)
        })

    # Total Row
    audit_summary.append({
        'Split': 'TOTAL',
        'Image Count': sum(d['Image Count'] for d in audit_summary),
        'Label Files': sum(d['Label Files'] for d in audit_summary),
        'Orphan Images': len(orphan_images),
        'Orphan Labels': len(orphan_labels),
        'Valid Instances': sum(d['Valid Instances'] for d in audit_summary),
        'Engaged (0)': sum(d['Engaged (0)'] for d in audit_summary),
        'Confused (1)': sum(d['Confused (1)'] for d in audit_summary),
        'Bored (2)': sum(d['Bored (2)'] for d in audit_summary),
        'Frustrated (3)': sum(d['Frustrated (3)'] for d in audit_summary),
        'Multi-BBox': sum(d['Multi-BBox'] for d in audit_summary),
        'Zero-BBox': sum(d['Zero-BBox'] for d in audit_summary)
    })

    df_summary = pd.DataFrame(audit_summary)
    df_summary.to_csv(OUTPUT_GROUPWISE_DIR / 'dataset_audit.csv', index=False)

    with open(OUTPUT_GROUPWISE_DIR / 'dataset_audit.json', 'w', encoding='utf-8') as f:
        json.dump(detailed_data, f, indent=2)

    print("\n" + df_summary.to_string(index=False))
    print(f"\n  [SAVED] {OUTPUT_GROUPWISE_DIR / 'dataset_audit.csv'}")
    print(f"  [SAVED] {OUTPUT_GROUPWISE_DIR / 'dataset_audit.json'}")

    total_valid_imgs = len(all_valid_images)
    assert total_valid_imgs == 1660, f"Total image count harus tepat 1660, ditemukan: {total_valid_imgs}"
    print(f"  ✓ Verifikasi Total Citra Valid: {total_valid_imgs} citra (Sesuai Target 1.660)")

    return detailed_data, all_valid_images


if __name__ == '__main__':
    audit_original_dataset()
