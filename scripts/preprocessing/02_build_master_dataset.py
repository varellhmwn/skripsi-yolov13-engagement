import os
import shutil
import zipfile
import yaml
from pathlib import Path
from tqdm import tqdm

# Target Classes (DAiSEE Standard)
TARGET_NAMES = ['engaged', 'confused', 'bored', 'frustrated']
MASTER_DIR = Path('datasets/master_4class')

def create_dirs():
    if MASTER_DIR.exists():
        shutil.rmtree(MASTER_DIR)
    
    for split in ['train', 'valid', 'test']:
        (MASTER_DIR / 'images' / split).mkdir(parents=True, exist_ok=True)
        (MASTER_DIR / 'labels' / split).mkdir(parents=True, exist_ok=True)

def process_finetuned_dataset():
    """Process the already aligned DAiSEE finetuned dataset (contains hard samples)"""
    print("Processing: roboflow_4class_yolo_finetuned")
    src_dir = Path('datasets/roboflow_4class_yolo_finetuned')
    
    stats = {'kept': 0}
    
    for split in ['train', 'valid', 'test']:
        img_dir = src_dir / 'images' / split
        lbl_dir = src_dir / 'labels' / split
        
        if not lbl_dir.exists() or not img_dir.exists():
            continue
            
        for lbl_file in lbl_dir.glob('*.txt'):
            img_path = None
            for ext in ['.jpg', '.jpeg', '.png']:
                cand = img_dir / f"{lbl_file.stem}{ext}"
                if cand.exists():
                    img_path = cand
                    break
                    
            if img_path:
                # Direct copy without remapping
                new_stem = f"finetuned_{lbl_file.stem}"
                dst_img = MASTER_DIR / 'images' / split / f"{new_stem}{img_path.suffix}"
                dst_lbl = MASTER_DIR / 'labels' / split / f"{new_stem}.txt"
                
                shutil.copy2(img_path, dst_img)
                shutil.copy2(lbl_file, dst_lbl)
                stats['kept'] += 1
                
    print(f"  -> Kept: {stats['kept']} (Includes Hard Samples)")
    return stats['kept']

def process_big_data_zip():
    """Process big-data-data and remap its classes"""
    print("Processing: big-data-data.v1i.yolov8.zip")
    zpath = Path('datasets/kumpulan dataset/big-data-data.v1i.yolov8.zip')
    temp_dir = Path('datasets/temp_bigdata')
    
    if not zpath.exists():
        print(f"[ERROR] ZIP not found: {zpath}")
        return 0
        
    with zipfile.ZipFile(zpath, 'r') as zf:
        zf.extractall(temp_dir)
        
    # Mapping for big-data-data
    # big-data classes: 0: bored, 1: confused, 2: engaged, 3: frustrated
    # Target classes:   0: engaged, 1: confused, 2: bored, 3: frustrated
    mapping = {
        0: 2, # bored -> bored
        1: 1, # confused -> confused
        2: 0, # engaged -> engaged
        3: 3  # frustrated -> frustrated
    }
    
    stats = {'kept': 0, 'dropped': 0}
    
    for split in ['train', 'valid', 'test']:
        # Note: Zip might have 'val' instead of 'valid' or similar structure
        img_dir = temp_dir / split / 'images'
        lbl_dir = temp_dir / split / 'labels'
        
        if not img_dir.exists():
            img_dir = temp_dir / split / 'images'
            if split == 'valid' and not img_dir.exists():
                img_dir = temp_dir / 'val' / 'images'
                lbl_dir = temp_dir / 'val' / 'labels'
                
        if not lbl_dir.exists() or not img_dir.exists():
            continue
            
        for lbl_file in lbl_dir.glob('*.txt'):
            new_lines = []
            with open(lbl_file, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if not parts: continue
                    old_id = int(parts[0])
                    if old_id in mapping:
                        new_id = mapping[old_id]
                        parts[0] = str(new_id)
                        new_lines.append(" ".join(parts))
                        
            if not new_lines:
                stats['dropped'] += 1
                continue
                
            img_path = None
            for ext in ['.jpg', '.jpeg', '.png']:
                cand = img_dir / f"{lbl_file.stem}{ext}"
                if cand.exists():
                    img_path = cand
                    break
                    
            if img_path:
                new_stem = f"bigdata_{lbl_file.stem}"
                dst_img = MASTER_DIR / 'images' / split / f"{new_stem}{img_path.suffix}"
                dst_lbl = MASTER_DIR / 'labels' / split / f"{new_stem}.txt"
                
                shutil.copy2(img_path, dst_img)
                with open(dst_lbl, 'w') as f:
                    f.write("\n".join(new_lines) + "\n")
                stats['kept'] += 1
            else:
                stats['dropped'] += 1
                
    shutil.rmtree(temp_dir)
    print(f"  -> Kept: {stats['kept']} (Remapped to DAiSEE format)")
    return stats['kept']

def main():
    print("=" * 60)
    print("BUILDING MASTER 4-CLASS DATASET")
    print("=" * 60)
    
    create_dirs()
    
    total = 0
    total += process_finetuned_dataset()
    total += process_big_data_zip()
    
    # Create data.yaml
    yaml_content = {
        'path': str(MASTER_DIR.absolute()),
        'train': 'images/train',
        'val': 'images/valid',
        'test': 'images/test',
        'nc': 4,
        'names': TARGET_NAMES
    }
    with open(MASTER_DIR / 'data.yaml', 'w') as f:
        yaml.dump(yaml_content, f, sort_keys=False)
        
    print("=" * 60)
    print(f"DONE! Total images combined: {total}")
    print(f"Master 4-Class Dataset saved to: {MASTER_DIR}")
    print("=" * 60)

if __name__ == "__main__":
    main()
