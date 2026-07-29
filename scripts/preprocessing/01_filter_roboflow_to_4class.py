import os
import argparse
import yaml
import shutil
import pandas as pd
from pathlib import Path
from tqdm import tqdm

# Target class mapping
TARGET_MAP = {
    'engaged': 0,
    'confusion': 1,
    'boredom': 2,
    'frustration': 3,
    # Additional handling for variations
    'confused': 1,
    'bored': 2,
    'frustrated': 3
}

TARGET_NAMES = {
    0: 'engaged',
    1: 'confused',
    2: 'bored',
    3: 'frustrated'
}

def load_roboflow_yaml(input_dir):
    yaml_path = Path(input_dir) / 'data.yaml'
    if not yaml_path.exists():
        raise FileNotFoundError(f"data.yaml tidak ditemukan di {input_dir}")
        
    with open(yaml_path, 'r') as f:
        data = yaml.safe_load(f)
        
    names = data.get('names')
    if isinstance(names, list):
        class_names = {i: name for i, name in enumerate(names)}
    elif isinstance(names, dict):
        class_names = names
    else:
        raise ValueError("Format 'names' di data.yaml tidak dikenali.")
        
    return data, class_names

def determine_split_paths(input_dir, data_yaml):
    """Mendeteksi apakah format A (train/images) atau format B (images/train)"""
    splits = {}
    base = Path(input_dir)
    
    # Check common split names
    for split_key in ['train', 'val', 'valid', 'test']:
        yaml_val = data_yaml.get(split_key)
        if not yaml_val:
            # Try guessing if missing in yaml
            if (base / split_key / 'images').exists() or (base / 'images' / split_key).exists():
                pass
            else:
                continue
                
        # Resolve path
        # Check format A: base/train/images
        if (base / split_key / 'images').exists():
            img_dir = base / split_key / 'images'
            lbl_dir = base / split_key / 'labels'
        # Check format B: base/images/train
        elif (base / 'images' / split_key).exists():
            img_dir = base / 'images' / split_key
            lbl_dir = base / 'labels' / split_key
        # Check yaml relative path
        elif yaml_val:
            # Often yaml path is like "../train/images" or "train/images"
            clean_val = str(yaml_val).replace('../', '').replace('./', '')
            img_dir = base / clean_val
            lbl_dir = Path(str(img_dir).replace('images', 'labels'))
        else:
            continue
            
        if img_dir.exists() and lbl_dir.exists():
            # Standardize split name (valid -> val)
            std_split = 'val' if split_key == 'valid' else split_key
            splits[std_split] = {'images': img_dir, 'labels': lbl_dir}
            
    return splits

def create_class_mapping(old_class_names):
    mapping = {}
    remap_log = []
    
    for old_id, old_name in old_class_names.items():
        mapped = False
        old_name_lower = str(old_name).lower().strip()
        
        for target_key, new_id in TARGET_MAP.items():
            if target_key in old_name_lower:
                mapping[old_id] = new_id
                remap_log.append({
                    'old_class_id': old_id,
                    'old_class_name': old_name,
                    'new_class_id': new_id,
                    'new_class_name': TARGET_NAMES[new_id],
                    'action': 'KEEP & REMAP'
                })
                mapped = True
                break
                
        if not mapped:
            remap_log.append({
                'old_class_id': old_id,
                'old_class_name': old_name,
                'new_class_id': None,
                'new_class_name': None,
                'action': 'DROP'
            })
            
    return mapping, remap_log

def filter_dataset(args):
    print(f"[INFO] Membaca dataset Roboflow dari {args.input_dir}")
    
    out_dir = Path(args.output_dir)
    if out_dir.exists():
        if not args.overwrite:
            print(f"[ERROR] Output direktori {out_dir} sudah ada. Gunakan --overwrite.")
            return
        shutil.rmtree(out_dir)
        
    data_yaml, old_class_names = load_roboflow_yaml(args.input_dir)
    print("\n[INFO] Roboflow Class Mapping Asli:")
    for k, v in old_class_names.items():
        print(f"  {k}: {v}")
        
    class_mapping, remap_log = create_class_mapping(old_class_names)
    print("\n[INFO] Rencana Konversi:")
    for log in remap_log:
        if log['action'] == 'DROP':
            print(f"  [DROP] {log['old_class_name']}")
        else:
            print(f"  [REMAP] {log['old_class_name']} -> {log['new_class_name']} (ID: {log['new_class_id']})")
            
    splits = determine_split_paths(args.input_dir, data_yaml)
    if not splits:
        print("[ERROR] Gagal mendeteksi struktur dataset image/label dari input.")
        return
        
    # Setup output directories
    for split in splits.keys():
        (out_dir / 'images' / split).mkdir(parents=True, exist_ok=True)
        (out_dir / 'labels' / split).mkdir(parents=True, exist_ok=True)
    meta_dir = out_dir / 'meta'
    meta_dir.mkdir(parents=True, exist_ok=True)
    
    # Tracking stats
    class_dist = {i: 0 for i in TARGET_NAMES.keys()}
    images_per_class = {i: 0 for i in TARGET_NAMES.keys()}
    skipped_images = []
    
    total_images_processed = 0
    total_images_saved = 0
    
    # Process dataset
    for split, paths in splits.items():
        img_dir = paths['images']
        lbl_dir = paths['labels']
        
        images = list(img_dir.rglob('*.jpg')) + list(img_dir.rglob('*.png')) + list(img_dir.rglob('*.jpeg'))
        
        for img_path in tqdm(images, desc=f"Processing {split}"):
            total_images_processed += 1
            lbl_path = lbl_dir / f"{img_path.stem}.txt"
            
            if not lbl_path.exists():
                skipped_images.append({'image': img_path.name, 'split': split, 'reason': 'Missing label file'})
                continue
                
            with open(lbl_path, 'r') as f:
                lines = f.readlines()
                
            new_lines = []
            found_classes_in_image = set()
            
            for line in lines:
                parts = line.strip().split()
                if len(parts) != 5: continue
                
                try:
                    c, x, y, w, h = [float(p) for p in parts]
                    c = int(c)
                except ValueError:
                    continue
                    
                if c in class_mapping:
                    new_c = class_mapping[c]
                    # Validasi batas 0-1
                    if (0 <= x <= 1) and (0 <= y <= 1) and (0 < w <= 1) and (0 < h <= 1):
                        new_lines.append(f"{new_c} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n")
                        class_dist[new_c] += 1
                        found_classes_in_image.add(new_c)
                        
            if new_lines:
                # Save label
                out_lbl_path = out_dir / 'labels' / split / f"{img_path.stem}.txt"
                with open(out_lbl_path, 'w') as f:
                    f.writelines(new_lines)
                
                # Copy image
                out_img_path = out_dir / 'images' / split / img_path.name
                shutil.copy(img_path, out_img_path)
                
                for fc in found_classes_in_image:
                    images_per_class[fc] += 1
                    
                total_images_saved += 1
            else:
                skipped_images.append({'image': img_path.name, 'split': split, 'reason': 'No target class bbox remaining'})
                
    # Save meta
    pd.DataFrame(remap_log).to_csv(meta_dir / 'remap_log.csv', index=False)
    
    dist_records = []
    for cid, cname in TARGET_NAMES.items():
        dist_records.append({
            'class_id': cid,
            'class_name': cname,
            'total_bboxes': class_dist[cid],
            'total_images': images_per_class[cid]
        })
    pd.DataFrame(dist_records).to_csv(meta_dir / 'class_distribution.csv', index=False)
    
    if skipped_images:
        pd.DataFrame(skipped_images).to_csv(meta_dir / 'skipped_images.csv', index=False)
        
    # Save final data.yaml
    final_yaml = {
        'path': os.path.abspath(out_dir),
        'train': 'images/train',
        'val': 'images/val',
        'test': 'images/test',
        'names': TARGET_NAMES
    }
    with open(out_dir / 'data.yaml', 'w') as f:
        yaml.dump(final_yaml, f, sort_keys=False)
        
    # Summary
    print("\n" + "="*50)
    print("RINGKASAN KONVERSI ROBOFLOW -> 4-CLASS YOLO")
    print("="*50)
    print(f"Total gambar diproses : {total_images_processed}")
    print(f"Total gambar disimpan : {total_images_saved}")
    print(f"Total gambar di-skip  : {len(skipped_images)}")
    print("\nDistribusi Kelas (Bboxes):")
    for rec in dist_records:
        print(f"  - {rec['class_name']}: {rec['total_bboxes']} kotak wajah")
    print("\n" + "="*50)
    print(f"Dataset berhasil disimpan di: {out_dir}")
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str, required=True, help='Path ke dataset Roboflow input')
    parser.add_argument('--output_dir', type=str, required=True, help='Path ke dataset 4-class output')
    parser.add_argument('--overwrite', action='store_true', help='Timpa jika direktori output sudah ada')
    args = parser.parse_args()
    
    filter_dataset(args)
