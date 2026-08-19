"""
dataset_audit.py — Audit Komprehensif Dataset Master Combined
=============================================================
Memisahkan secara eksplisit antara:
  - image_count (jumlah file citra)
  - instance_count (jumlah bounding box / anotasi emosi)
Mendeteksi orphan images, orphan labels, multi-bbox, dan 0-bbox.
"""

import json
from pathlib import Path
import pandas as pd
import numpy as np

BASE_DIR = Path(__file__).resolve().parent.parent
DATASET_DIR = BASE_DIR / 'datasets' / 'master_combined_dataset'
OUTPUT_DIR = BASE_DIR / 'outputs'

CLASS_NAMES = {0: 'engaged', 1: 'confused', 2: 'bored', 3: 'frustrated'}
VALID_IMG_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}

def audit_dataset():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    splits = ['train', 'val', 'test']
    audit_summary = []
    
    orphan_images = []
    orphan_labels = []
    
    detailed_data = {}
    
    for split in splits:
        img_dir = DATASET_DIR / 'images' / split
        lbl_dir = DATASET_DIR / 'labels' / split
        
        img_files = {f.stem: f for f in img_dir.iterdir() if f.suffix.lower() in VALID_IMG_EXTS} if img_dir.exists() else {}
        lbl_files = {f.stem: f for f in lbl_dir.iterdir() if f.suffix.lower() == '.txt'} if lbl_dir.exists() else {}
        
        # Check special files in labels like labels.txt
        non_yolo_lbls = []
        clean_lbl_files = {}
        for stem, p in lbl_files.items():
            if stem == 'labels' or p.name == 'labels.txt':
                non_yolo_lbls.append(p.name)
            else:
                clean_lbl_files[stem] = p
                
        # Orphans
        for stem, img_path in img_files.items():
            if stem not in clean_lbl_files:
                orphan_images.append({
                    'split': split,
                    'filename': img_path.name,
                    'path': str(img_path.relative_to(BASE_DIR))
                })
                
        for stem, lbl_path in clean_lbl_files.items():
            if stem not in img_files:
                orphan_labels.append({
                    'split': split,
                    'filename': lbl_path.name,
                    'path': str(lbl_path.relative_to(BASE_DIR))
                })
        
        # Count bounding boxes and instances
        class_counts = {0: 0, 1: 0, 2: 0, 3: 0}
        images_with_multi_bbox = []
        images_with_zero_bbox = []
        total_bboxes = 0
        
        # Analyze only images that actually exist in dataset
        for stem, img_path in img_files.items():
            lbl_path = clean_lbl_files.get(stem)
            if not lbl_path or not lbl_path.exists():
                images_with_zero_bbox.append(img_path.name)
                continue
                
            bboxes_in_img = 0
            with open(lbl_path, 'r', encoding='utf-8') as f:
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
                images_with_zero_bbox.append(img_path.name)
            elif bboxes_in_img > 1:
                images_with_multi_bbox.append({
                    'filename': img_path.name,
                    'bbox_count': bboxes_in_img
                })
        
        # Instances from orphan labels (if any parsed)
        orphan_label_bboxes = 0
        for stem, lbl_path in clean_lbl_files.items():
            if stem not in img_files:
                with open(lbl_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        if len(line.strip().split()) >= 5:
                            orphan_label_bboxes += 1
                            
        split_stat = {
            'split': split,
            'image_count': len(img_files),
            'label_file_count_total': len(lbl_files),
            'label_file_count_matched': len(img_files) - len([o for o in orphan_images if o['split'] == split]),
            'orphan_images_count': len([o for o in orphan_images if o['split'] == split]),
            'orphan_labels_count': len([o for o in orphan_labels if o['split'] == split]),
            'non_yolo_label_files': non_yolo_lbls,
            'instance_count_in_valid_images': total_bboxes,
            'instance_count_per_class': {
                CLASS_NAMES[c]: class_counts[c] for c in sorted(class_counts.keys())
            },
            'images_with_single_bbox': len(img_files) - len(images_with_multi_bbox) - len(images_with_zero_bbox),
            'images_with_multi_bbox_count': len(images_with_multi_bbox),
            'images_with_zero_bbox_count': len(images_with_zero_bbox),
            'multi_bbox_details': images_with_multi_bbox
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
            'Multi-BBox Images': len(images_with_multi_bbox),
            'Zero-BBox Images': len(images_with_zero_bbox)
        })

    # Total row
    total_imgs = sum(d['Image Count'] for d in audit_summary)
    total_lbls = sum(d['Label Files'] for d in audit_summary)
    total_instances = sum(d['Valid Instances'] for d in audit_summary)
    total_engaged = sum(d['Engaged (0)'] for d in audit_summary)
    total_confused = sum(d['Confused (1)'] for d in audit_summary)
    total_bored = sum(d['Bored (2)'] for d in audit_summary)
    total_frustrated = sum(d['Frustrated (3)'] for d in audit_summary)
    
    audit_summary.append({
        'Split': 'TOTAL',
        'Image Count': total_imgs,
        'Label Files': total_lbls,
        'Orphan Images': sum(d['Orphan Images'] for d in audit_summary[:-1]),
        'Orphan Labels': sum(d['Orphan Labels'] for d in audit_summary[:-1]),
        'Valid Instances': total_instances,
        'Engaged (0)': total_engaged,
        'Confused (1)': total_confused,
        'Bored (2)': total_bored,
        'Frustrated (3)': total_frustrated,
        'Multi-BBox Images': sum(d['Multi-BBox Images'] for d in audit_summary[:-1]),
        'Zero-BBox Images': sum(d['Zero-BBox Images'] for d in audit_summary[:-1])
    })
    
    # Save outputs
    df_summary = pd.DataFrame(audit_summary)
    df_summary.to_csv(OUTPUT_DIR / 'dataset_audit.csv', index=False)
    
    with open(OUTPUT_DIR / 'dataset_audit.json', 'w', encoding='utf-8') as f:
        json.dump(detailed_data, f, indent=2)
        
    df_orphan_imgs = pd.DataFrame(orphan_images)
    df_orphan_imgs.to_csv(OUTPUT_DIR / 'orphan_images.csv', index=False)
    
    df_orphan_lbls = pd.DataFrame(orphan_labels)
    df_orphan_lbls.to_csv(OUTPUT_DIR / 'orphan_labels.csv', index=False)
    
    print("=== DATASET AUDIT RESULTS ===")
    print(df_summary.to_string(index=False))
    print(f"\nOrphan Images: {len(orphan_images)}")
    print(f"Orphan Labels: {len(orphan_labels)}")
    print("Files saved to outputs/")
    return detailed_data

if __name__ == '__main__':
    audit_dataset()
