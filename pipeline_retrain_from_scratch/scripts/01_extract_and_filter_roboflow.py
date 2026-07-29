import os
import zipfile
import shutil
import yaml
import pandas as pd
from pathlib import Path
from tqdm import tqdm

# Big-data mapping: 0=bored -> 2, 1=confused -> 1, 2=engaged -> 0, 3=frustrated -> 3
BIGDATA_MAP = {
    0: 2, # bored
    1: 1, # confused
    2: 0, # engaged
    3: 3  # frustrated
}

TARGET_NAMES = {
    0: 'engaged',
    1: 'confused',
    2: 'bored',
    3: 'frustrated'
}

BASE_DIR = Path(__file__).resolve().parent.parent
RAW_ZIP = BASE_DIR / 'raw_datasets' / 'big_data_roboflow.zip'
TEMP_DIR = BASE_DIR / 'temp_bigdata_roboflow'
OUT_DIR = BASE_DIR / 'datasets_processed' / '01_bigdata_roboflow_remapped'

def main():
    print("=" * 60)
    print("  TAHAP 1: Ekstrak & Remap Roboflow Big-Data Dataset")
    print("=" * 60)

    if OUT_DIR.exists():
        shutil.rmtree(OUT_DIR, ignore_errors=True)
    if TEMP_DIR.exists():
        shutil.rmtree(TEMP_DIR, ignore_errors=True)

    # 1. Ekstrak ZIP
    print(f"[INFO] Mengekstrak {RAW_ZIP.name}...")
    with zipfile.ZipFile(RAW_ZIP, 'r') as zip_ref:
        zip_ref.extractall(TEMP_DIR)

    # Buat direktori output
    for split in ['train', 'val', 'test']:
        (OUT_DIR / 'images' / split).mkdir(parents=True, exist_ok=True)
        (OUT_DIR / 'labels' / split).mkdir(parents=True, exist_ok=True)

    class_dist = {i: 0 for i in TARGET_NAMES.keys()}
    total_images_processed = 0
    total_images_saved = 0

    # 2. Remap & salin file
    for split in ['train', 'valid', 'test']:
        img_dir = TEMP_DIR / split / 'images'
        lbl_dir = TEMP_DIR / split / 'labels'

        if not img_dir.exists():
            continue

        images = list(img_dir.glob('*.jpg')) + list(img_dir.glob('*.png')) + list(img_dir.glob('*.jpeg'))
        dst_split = 'val' if split == 'valid' else split

        for img_path in tqdm(images, desc=f"Remapping {split}"):
            total_images_processed += 1
            lbl_path = lbl_dir / f"{img_path.stem}.txt"

            if not lbl_path.exists():
                continue

            with open(lbl_path, 'r') as f:
                lines = f.readlines()

            new_lines = []
            for line in lines:
                parts = line.strip().split()
                if len(parts) != 5:
                    continue
                orig_cls = int(float(parts[0]))
                if orig_cls in BIGDATA_MAP:
                    target_cls = BIGDATA_MAP[orig_cls]
                    new_lines.append(f"{target_cls} {parts[1]} {parts[2]} {parts[3]} {parts[4]}\n")
                    class_dist[target_cls] += 1

            if new_lines:
                # Copy image
                dst_img = OUT_DIR / 'images' / dst_split / img_path.name
                shutil.copy2(img_path, dst_img)

                # Write remapped label
                dst_lbl = OUT_DIR / 'labels' / dst_split / f"{img_path.stem}.txt"
                with open(dst_lbl, 'w') as f:
                    f.writelines(new_lines)

                total_images_saved += 1

    # Cleanup temp
    shutil.rmtree(TEMP_DIR, ignore_errors=True)

    # Save data.yaml
    final_yaml = {
        'path': str(OUT_DIR.resolve()),
        'train': 'images/train',
        'val': 'images/val',
        'test': 'images/test',
        'names': TARGET_NAMES
    }
    with open(OUT_DIR / 'data.yaml', 'w') as f:
        yaml.dump(final_yaml, f, sort_keys=False)

    print("\n" + "=" * 50)
    print("  HASIL TAHAP 1 — ROBOFLOW BIG-DATA REMAPPED")
    print("=" * 50)
    print(f"  Total Gambar Diproses : {total_images_processed}")
    print(f"  Total Gambar Disimpan : {total_images_saved}")
    print("\n  Distribusi Bounding Box per Kelas:")
    for cid, cname in TARGET_NAMES.items():
        print(f"    - {cname:12s} : {class_dist[cid]} bboxes")
    print("=" * 50 + "\n")

if __name__ == '__main__':
    main()
