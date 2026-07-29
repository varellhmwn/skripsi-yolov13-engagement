import os
import argparse
import random
import shutil
import pandas as pd
import yaml
from pathlib import Path
from tqdm import tqdm
from collections import Counter

# Class definitions
TARGET_CLASSES = {
    0: 'engaged',
    1: 'confused',
    2: 'bored',
    3: 'frustrated'
}

VALID_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}

def parse_args():
    parser = argparse.ArgumentParser(description="Stratified Resplit for YOLO Dataset")
    parser.add_argument('--input_dir', type=str, default='datasets/roboflow_4class_yolo')
    parser.add_argument('--output_dir', type=str, default='datasets/roboflow_4class_yolo_stratified')
    parser.add_argument('--train_ratio', type=float, default=0.80)
    parser.add_argument('--val_ratio', type=float, default=0.10)
    parser.add_argument('--test_ratio', type=float, default=0.10)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--overwrite', action='store_true')
    return parser.parse_args()

def validate_yolo_label(lines):
    for line in lines:
        if not line.strip(): continue
        parts = line.strip().split()
        if len(parts) != 5: return False
        try:
            c, x, y, w, h = [float(p) for p in parts]
            c = int(c)
        except ValueError:
            return False
        if c not in TARGET_CLASSES: return False
        if not (0 <= x <= 1 and 0 <= y <= 1): return False
        if not (0 < w <= 1 and 0 < h <= 1): return False
    return True

def detect_sample_class(lines):
    classes = set()
    for line in lines:
        if not line.strip(): continue
        parts = line.strip().split()
        c = int(float(parts[0]))
        classes.add(c)
    return list(classes)

def load_yolo_samples(input_dir):
    samples = []
    multi_class_images = []
    
    base_dir = Path(input_dir)
    splits = ['train', 'val', 'test']
    
    for split in splits:
        img_dir = base_dir / 'images' / split
        lbl_dir = base_dir / 'labels' / split
        
        if not img_dir.exists() or not lbl_dir.exists():
            continue
            
        images = [f for f in img_dir.iterdir() if f.suffix.lower() in VALID_EXTENSIONS]
        
        for img_path in images:
            lbl_path = lbl_dir / f"{img_path.stem}.txt"
            if not lbl_path.exists():
                continue
                
            with open(lbl_path, 'r') as f:
                lines = f.readlines()
                
            if not lines:
                continue
                
            if not validate_yolo_label(lines):
                continue
                
            classes_in_image = detect_sample_class(lines)
            
            if len(classes_in_image) > 1:
                multi_class_images.append({
                    'original_split': split,
                    'image_path': str(img_path),
                    'label_path': str(lbl_path),
                    'classes_found': [TARGET_CLASSES[c] for c in classes_in_image]
                })
            elif len(classes_in_image) == 1:
                c = classes_in_image[0]
                samples.append({
                    'original_split': split,
                    'image_path': img_path,
                    'label_path': lbl_path,
                    'class_id': c,
                    'class_name': TARGET_CLASSES[c],
                    'num_bboxes': len([l for l in lines if l.strip()])
                })
                
    return samples, multi_class_images

def stratified_split(samples, args):
    random.seed(args.seed)
    
    # Group by class
    class_groups = {c: [] for c in TARGET_CLASSES.keys()}
    for s in samples:
        class_groups[s['class_id']].append(s)
        
    resplit_data = []
    
    for c, group in class_groups.items():
        random.shuffle(group)
        n = len(group)
        
        n_train = int(n * args.train_ratio)
        n_val = int(n * args.val_ratio)
        
        train_group = group[:n_train]
        val_group = group[n_train:n_train+n_val]
        test_group = group[n_train+n_val:]
        
        for s in train_group: s['new_split'] = 'train'
        for s in val_group: s['new_split'] = 'val'
        for s in test_group: s['new_split'] = 'test'
        
        resplit_data.extend(train_group + val_group + test_group)
        
    return resplit_data

def copy_samples(resplit_data, out_dir):
    base_out = Path(out_dir)
    
    for split in ['train', 'val', 'test']:
        (base_out / 'images' / split).mkdir(parents=True, exist_ok=True)
        (base_out / 'labels' / split).mkdir(parents=True, exist_ok=True)
        
    for s in tqdm(resplit_data, desc="Copying files"):
        new_split = s['new_split']
        img_dest = base_out / 'images' / new_split / s['image_path'].name
        lbl_dest = base_out / 'labels' / new_split / s['label_path'].name
        
        shutil.copy(s['image_path'], img_dest)
        shutil.copy(s['label_path'], lbl_dest)

def write_data_yaml(out_dir):
    yaml_data = {
        'path': os.path.abspath(out_dir),
        'train': 'images/train',
        'val': 'images/val',
        'test': 'images/test',
        'names': TARGET_CLASSES
    }
    with open(Path(out_dir) / 'data.yaml', 'w') as f:
        yaml.dump(yaml_data, f, sort_keys=False)

def write_reports(resplit_data, multi_class_images, out_dir):
    meta_dir = Path(out_dir) / 'meta'
    meta_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. multi_class_images.csv
    if multi_class_images:
        pd.DataFrame(multi_class_images).to_csv(meta_dir / 'multi_class_images.csv', index=False)
        
    # 2. resplit_log.csv
    log_data = []
    for s in resplit_data:
        log_data.append({
            'original_split': s['original_split'],
            'new_split': s['new_split'],
            'image_path': s['image_path'].name,
            'label_path': s['label_path'].name,
            'class_id': s['class_id'],
            'class_name': s['class_name']
        })
    pd.DataFrame(log_data).to_csv(meta_dir / 'resplit_log.csv', index=False)
    
    # 3. class_distribution_by_split.csv
    dist_data = []
    summary_counts = {'train': Counter(), 'val': Counter(), 'test': Counter()}
    
    for s in resplit_data:
        summary_counts[s['new_split']][s['class_id']] += 1
        
    total_images = len(resplit_data)
    
    for split in ['train', 'val', 'test']:
        split_total = sum(summary_counts[split].values())
        for cid, cname in TARGET_CLASSES.items():
            count = summary_counts[split].get(cid, 0)
            pct = (count / split_total * 100) if split_total > 0 else 0
            
            # Count bboxes
            bboxes = sum([s['num_bboxes'] for s in resplit_data if s['new_split'] == split and s['class_id'] == cid])
            
            dist_data.append({
                'split': split,
                'class_id': cid,
                'class_name': cname,
                'images': count,
                'bboxes': bboxes,
                'percentage': f"{pct:.2f}%"
            })
            
    pd.DataFrame(dist_data).to_csv(meta_dir / 'class_distribution_by_split.csv', index=False)
    
    # 4. resplit_summary.md
    md = ["# Stratified Resplit Summary\n"]
    md.append(f"- **Total Image Input Valid (Single Class)**: {total_images}")
    md.append(f"- **Total Multi-Class Image Skipped**: {len(multi_class_images)}\n")
    
    md.append("## Distribusi Train/Val/Test per Class")
    
    for split in ['train', 'val', 'test']:
        md.append(f"### {split.capitalize()}")
        c_counts = [summary_counts[split].get(cid, 0) for cid in TARGET_CLASSES.keys()]
        if not c_counts or max(c_counts) == 0:
            md.append("Kosong\n")
            continue
            
        min_c = min(c_counts)
        imb = max(c_counts) / min_c if min_c > 0 else float('inf')
        
        md.append(f"Imbalance Ratio: {imb:.2f}")
        for cid, cname in TARGET_CLASSES.items():
            md.append(f"- {cname}: {summary_counts[split].get(cid, 0)}")
        md.append("\n")
        
    # Recommendation
    all_train = [summary_counts['train'].get(cid, 0) for cid in TARGET_CLASSES.keys()]
    all_val = [summary_counts['val'].get(cid, 0) for cid in TARGET_CLASSES.keys()]
    
    rec = "READY_FOR_TRAINING"
    if min(all_train) < 50 or min(all_val) < 10:
        rec = "TRAIN_WITH_CAUTION"
    if min(all_train) == 0 or min(all_val) == 0:
        rec = "NOT_READY"
        
    md.append("## Rekomendasi")
    md.append(f"Status: **{rec}**")
    
    with open(meta_dir / 'resplit_summary.md', 'w') as f:
        f.write("\n".join(md))
        
    return summary_counts, rec

def main():
    args = parse_args()
    print("[INFO] Starting Stratified Resplit")
    
    if abs((args.train_ratio + args.val_ratio + args.test_ratio) - 1.0) > 1e-6:
        print("[ERROR] Ratios must sum to 1.0")
        return
        
    out_dir = Path(args.output_dir)
    if out_dir.exists():
        if not args.overwrite:
            print("[ERROR] Output directory exists. Use --overwrite.")
            return
        shutil.rmtree(out_dir)
        
    samples, multi_class_images = load_yolo_samples(args.input_dir)
    if not samples:
        print("[ERROR] No valid samples found.")
        return
        
    resplit_data = stratified_split(samples, args)
    copy_samples(resplit_data, args.output_dir)
    write_data_yaml(args.output_dir)
    
    summary_counts, rec = write_reports(resplit_data, multi_class_images, args.output_dir)
    
    print("\n" + "="*50)
    print("STRATIFIED RESPLIT YOLO DATASET")
    print("="*50)
    print(f"Input dataset: {args.input_dir}")
    print(f"Output dataset: {args.output_dir}")
    print(f"\nTotal samples: {len(resplit_data)}")
    print(f"Skipped multi-class samples: {len(multi_class_images)}\n")
    
    print("Distribution after resplit:")
    for split in ['train', 'val', 'test']:
        print(f"\n{split.capitalize()}:")
        for cid, cname in TARGET_CLASSES.items():
            print(f"  {cname}: {summary_counts[split].get(cid, 0)}")
            
    print("\nFinal status:")
    print(rec)
    print("="*50)

if __name__ == "__main__":
    main()
