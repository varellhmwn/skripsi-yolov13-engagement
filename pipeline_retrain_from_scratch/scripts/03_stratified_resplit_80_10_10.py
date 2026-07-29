import os
import argparse
import random
import shutil
import pandas as pd
import yaml
from pathlib import Path
from tqdm import tqdm
from collections import Counter

TARGET_CLASSES = {
    0: 'engaged',
    1: 'confused',
    2: 'bored',
    3: 'frustrated'
}

BASE_DIR = Path(__file__).resolve().parent.parent
INPUT_DIR = BASE_DIR / 'datasets_processed' / '02_master_4class_unsplit'
OUT_DIR = BASE_DIR / 'datasets_processed' / '03_master_combined_80_10_10'

def parse_args():
    parser = argparse.ArgumentParser(description="Stratified 80/10/10 Split for YOLO Dataset")
    parser.add_argument('--input_dir', type=str, default=str(INPUT_DIR))
    parser.add_argument('--output_dir', type=str, default=str(OUT_DIR))
    parser.add_argument('--train_ratio', type=float, default=0.80)
    parser.add_argument('--val_ratio', type=float, default=0.10)
    parser.add_argument('--test_ratio', type=float, default=0.10)
    parser.add_argument('--seed', type=int, default=42)
    return parser.parse_args()

def main():
    args = parse_args()
    print("=" * 60)
    print("  TAHAP 3: Stratified Random Split (80% Train / 10% Val / 10% Test)")
    print("=" * 60)

    in_dir = Path(args.input_dir)
    out_dir = Path(args.output_dir)

    if out_dir.exists():
        shutil.rmtree(out_dir)

    # Scan samples
    img_dir = in_dir / 'images'
    lbl_dir = in_dir / 'labels'

    samples = []
    for lbl_path in lbl_dir.glob('*.txt'):
        img_path = None
        for ext in ['.jpg', '.jpeg', '.png']:
            cand = img_dir / f"{lbl_path.stem}{ext}"
            if cand.exists():
                img_path = cand
                break

        if img_path:
            with open(lbl_path, 'r') as f:
                lines = f.readlines()

            classes = set()
            for l in lines:
                if l.strip():
                    c = int(float(l.strip().split()[0]))
                    classes.add(c)

            if classes:
                dom_cls = list(classes)[0]
                samples.append({
                    'image_path': img_path,
                    'label_path': lbl_path,
                    'class_id': dom_cls,
                    'class_name': TARGET_CLASSES[dom_cls]
                })

    print(f"[INFO] Total sampel valid ditemukan: {len(samples)}")

    # Stratified Split per kelas
    random.seed(args.seed)
    class_groups = {c: [] for c in TARGET_CLASSES.keys()}
    for s in samples:
        class_groups[s['class_id']].append(s)

    split_data = {'train': [], 'val': [], 'test': []}

    for c, group in class_groups.items():
        random.shuffle(group)
        n = len(group)
        n_train = int(n * args.train_ratio)
        n_val = int(n * args.val_ratio)

        train_g = group[:n_train]
        val_g = group[n_train:n_train + n_val]
        test_g = group[n_train + n_val:]

        split_data['train'].extend(train_g)
        split_data['val'].extend(val_g)
        split_data['test'].extend(test_g)

    # Buat folder output
    for s in ['train', 'val', 'test']:
        (out_dir / 'images' / s).mkdir(parents=True, exist_ok=True)
        (out_dir / 'labels' / s).mkdir(parents=True, exist_ok=True)

    # Salin file ke folder split masing-masing
    counts = {'train': Counter(), 'val': Counter(), 'test': Counter()}

    for split, items in split_data.items():
        for item in tqdm(items, desc=f"Writing {split} set"):
            dst_img = out_dir / 'images' / split / item['image_path'].name
            dst_lbl = out_dir / 'labels' / split / item['label_path'].name
            shutil.copy2(item['image_path'], dst_img)
            shutil.copy2(item['label_path'], dst_lbl)
            counts[split][item['class_name']] += 1

    # Save data.yaml
    final_yaml = {
        'path': str(out_dir.resolve()),
        'train': 'images/train',
        'val': 'images/val',
        'test': 'images/test',
        'names': TARGET_CLASSES
    }
    with open(out_dir / 'data.yaml', 'w') as f:
        yaml.dump(final_yaml, f, sort_keys=False)

    print("\n" + "=" * 50)
    print("  HASIL TAHAP 3 — STRATIFIED SPLIT 80:10:10")
    print("=" * 50)
    for s in ['train', 'val', 'test']:
        total_s = len(split_data[s])
        pct_s = (total_s / len(samples)) * 100
        print(f"  Split {s.upper():5s} : {total_s:4d} gambar ({pct_s:.1f}%)")
        for cname in TARGET_CLASSES.values():
            cnt = counts[s][cname]
            cpct = (cnt / total_s * 100) if total_s > 0 else 0
            print(f"    - {cname:12s} : {cnt:3d} ({cpct:.1f}%)")
    print("=" * 50 + "\n")

if __name__ == '__main__':
    main()
