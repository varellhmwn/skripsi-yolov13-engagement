import os
import shutil
import yaml
from pathlib import Path
from tqdm import tqdm

TARGET_NAMES = {
    0: 'engaged',
    1: 'confused',
    2: 'bored',
    3: 'frustrated'
}

CLASS_MAP = {
    'engaged': 0,
    'confused': 1,
    'bored': 2,
    'frustrated': 3
}

BASE_DIR = Path(__file__).resolve().parent.parent
BIGDATA_DIR = BASE_DIR / 'datasets_processed' / '01_bigdata_roboflow_remapped'
HARD_SAMPLES_DIR = BASE_DIR / 'raw_datasets' / 'hard_samples'
OUT_DIR = BASE_DIR / 'datasets_processed' / '02_master_4class_unsplit'

def main():
    print("=" * 60)
    print("  TAHAP 2: Gabungkan Roboflow Big-Data + Hard Samples -> Master Unsplit")
    print("=" * 60)

    if OUT_DIR.exists():
        shutil.rmtree(OUT_DIR, ignore_errors=True)

    # Buat direktori master unsplit
    (OUT_DIR / 'images').mkdir(parents=True, exist_ok=True)
    (OUT_DIR / 'labels').mkdir(parents=True, exist_ok=True)

    kept_roboflow = 0
    kept_hardsamples = 0

    # 1. Salin data Roboflow Big-Data
    print("[INFO] Menggabungkan data dari Roboflow Big-Data...")
    for split in ['train', 'val', 'test']:
        img_dir = BIGDATA_DIR / 'images' / split
        lbl_dir = BIGDATA_DIR / 'labels' / split
        if not img_dir.exists():
            continue

        for lbl_file in lbl_dir.glob('*.txt'):
            img_file = None
            for ext in ['.jpg', '.jpeg', '.png']:
                cand = img_dir / f"{lbl_file.stem}{ext}"
                if cand.exists():
                    img_file = cand
                    break

            if img_file:
                dst_img = OUT_DIR / 'images' / f"rf_{img_file.name}"
                dst_lbl = OUT_DIR / 'labels' / f"rf_{lbl_file.name}"
                shutil.copy2(img_file, dst_img)
                shutil.copy2(lbl_file, dst_lbl)
                kept_roboflow += 1

    print(f"  -> Roboflow Big-Data tersimpan: {kept_roboflow} gambar")

    # 2. Proses folder Hard Samples (cropped face images per subfolder emosi)
    print(f"[INFO] Memproses folder Hard Samples dari {HARD_SAMPLES_DIR.name}...")
    for cname, cid in CLASS_MAP.items():
        sub_folder = HARD_SAMPLES_DIR / cname
        if not sub_folder.exists():
            continue

        images = list(sub_folder.glob('*.jpg')) + list(sub_folder.glob('*.png')) + list(sub_folder.glob('*.jpeg'))
        for img_file in tqdm(images, desc=f"Hard samples ({cname})"):
            dst_img = OUT_DIR / 'images' / f"hs_{cname}_{img_file.name}"
            dst_lbl = OUT_DIR / 'labels' / f"hs_{cname}_{img_file.stem}.txt"

            shutil.copy2(img_file, dst_img)
            # Generate full-face bounding box (class_id 0.5 0.5 1.0 1.0)
            with open(dst_lbl, 'w') as f:
                f.write(f"{cid} 0.5 0.5 1.0 1.0\n")

            kept_hardsamples += 1

    total_combined = kept_roboflow + kept_hardsamples

    # Save data.yaml
    final_yaml = {
        'path': str(OUT_DIR.resolve()),
        'train': 'images',
        'val': 'images',
        'test': 'images',
        'names': TARGET_NAMES
    }
    with open(OUT_DIR / 'data.yaml', 'w') as f:
        yaml.dump(final_yaml, f, sort_keys=False)

    print("\n" + "=" * 50)
    print("  HASIL TAHAP 2 — MASTER COMBINED UNSPLIT")
    print("=" * 50)
    print(f"  Sampel Roboflow Big-Data : {kept_roboflow} gambar")
    print(f"  Sampel Hard Samples      : {kept_hardsamples} gambar")
    print(f"  TOTAL COMBINED           : {total_combined} gambar")
    print("=" * 50 + "\n")

if __name__ == '__main__':
    main()
