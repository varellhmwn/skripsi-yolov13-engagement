"""
evaluate_subject_wise.py — Robust Held-Out Test Evaluation & Exporter for YOLOv13n
==================================================================================
Penelitian: "Deteksi Emosi Belajar Mahasiswa pada Pembelajaran Pemrograman Menggunakan YOLOv13n"

Fitur:
  1. Introspeksi metrics.box Ultralytics 8.3.63 yang akurat.
  2. Ekstraksi mAP@0.5, mAP@0.75, dan mAP@0.5:0.95 (mean of 10 IoU thresholds) per kelas.
  3. Perhitungan ground truth images count dan instances count per kelas.
  4. Penyimpanan raw AP matrix (raw_per_class_ap_matrix.csv) untuk audit 10 IoU thresholds.
  5. Backup file exporter lama dan pembuatan corrected files.
  6. Sanity check otomatis matematis (mean AP vs overall mAP).
"""

import sys
import os
import csv
import json
import shutil
import argparse
import logging
from pathlib import Path
from collections import Counter
import numpy as np

import torch
from ultralytics import YOLO, settings

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger('SubjectWiseEvaluator')

CLASS_NAMES = ['engaged', 'confused', 'bored', 'frustrated']
IOU_THRESHOLDS = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluasi Final Held-Out Test Set YOLOv13n")
    parser.add_argument('--weights', type=str, default='runs/train/yolov13_subject_wise_v1/weights/best.pt', help="Path ke weights model best.pt")
    parser.add_argument('--dataset_dir', type=str, default='subject_wise_dataset', help="Direktori dataset subject-wise")
    parser.add_argument('--split', type=str, default='test', choices=['val', 'test'], help="Subset yang dievaluasi")
    parser.add_argument('--project', type=str, default='runs/evaluation', help="Direktori output project")
    parser.add_argument('--name', type=str, default='yolov13_subject_wise_test', help="Nama folder output run evaluasi")
    parser.add_argument('--device', type=str, default='0', help="Device ID (0 atau 'cpu')")
    parser.add_argument('--batch', type=int, default=16, help="Batch size evaluasi")
    parser.add_argument('--imgsz', type=int, default=640, help="Resolusi citra evaluasi")
    return parser.parse_args()


def calculate_f1(precision: float, recall: float) -> float:
    """Menghitung F1-score dengan proteksi division-by-zero."""
    if (precision + recall) <= 1e-8:
        return 0.0
    return 2.0 * (precision * recall) / (precision + recall)


def count_ground_truth(label_dir: Path):
    """Menghitung ground truth images count dan instance count per class id."""
    img_counts = Counter()
    inst_counts = Counter()
    total_imgs = 0
    total_insts = 0

    if label_dir.exists():
        lbl_files = list(label_dir.glob('*.txt'))
        total_imgs = len(lbl_files)
        for f in lbl_files:
            with open(f, 'r', encoding='utf-8') as fp:
                cids = [int(line.strip().split()[0]) for line in fp if line.strip()]
                for cid in set(cids):
                    img_counts[cid] += 1
                for cid in cids:
                    inst_counts[cid] += 1
                total_insts += len(cids)
    return img_counts, inst_counts, total_imgs, total_insts


def extract_per_class_metrics(metrics, class_names, label_dir: Path):
    """
    Mengekstrak metrik per-class secara robust dari metrics.box Ultralytics.
    Mendukung pemetaan ap_class_index, 10 IoU thresholds AP vector, dan ground-truth counts.
    """
    box = metrics.box
    ap_class_idx = list(getattr(box, 'ap_class_index', range(len(class_names))))
    all_ap = getattr(box, 'all_ap', None)
    p_arr = getattr(box, 'p', None)
    r_arr = getattr(box, 'r', None)

    img_counts, inst_counts, total_imgs, total_insts = count_ground_truth(label_dir)

    per_class = {}
    raw_ap_matrix = {}

    for i, cname in enumerate(class_names):
        if i in ap_class_idx:
            k = ap_class_idx.index(i)
            p = float(p_arr[k]) if p_arr is not None and len(p_arr) > k else 0.0
            r = float(r_arr[k]) if r_arr is not None and len(r_arr) > k else 0.0
            f1 = calculate_f1(p, r)

            if all_ap is not None and len(all_ap) > k:
                ap_vec = np.array(all_ap[k], dtype=float)
                ap50 = float(ap_vec[0])
                ap75 = float(ap_vec[5])
                map50_95 = float(ap_vec.mean())
            else:
                ap_vec = np.zeros(10)
                ap50, ap75, map50_95 = 0.0, 0.0, 0.0
        else:
            p, r, f1, ap50, ap75, map50_95 = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
            ap_vec = np.zeros(10)

        per_class[cname] = {
            'images': img_counts[i],
            'instances': inst_counts[i],
            'precision': p,
            'recall': r,
            'f1': f1,
            'ap50': ap50,
            'ap75': ap75,
            'map50_95': map50_95
        }
        raw_ap_matrix[cname] = ap_vec

    precision_all = float(box.mp) if hasattr(box, 'mp') else float(box.p)
    recall_all = float(box.mr) if hasattr(box, 'mr') else float(box.r)
    map50_all = float(box.map50)
    map75_all = float(box.map75)
    map50_95_all = float(box.map)
    f1_all = calculate_f1(precision_all, recall_all)

    speed = metrics.speed
    preprocess_ms = float(speed.get('preprocess', 0.0))
    inference_ms = float(speed.get('inference', 0.0))
    postprocess_ms = float(speed.get('postprocess', 0.0))

    overall = {
        'images': total_imgs,
        'instances': total_insts,
        'precision': precision_all,
        'recall': recall_all,
        'f1': f1_all,
        'map50': map50_all,
        'map75': map75_all,
        'map50_95': map50_95_all,
        'preprocess_ms': preprocess_ms,
        'inference_ms': inference_ms,
        'postprocess_ms': postprocess_ms
    }

    return overall, per_class, raw_ap_matrix


def create_temp_eval_subset(source_meta_rows, target_source: str, temp_dir: Path):
    """Menyalin subset spesifik (public atau private) ke folder temporary untuk evaluasi terisolasi."""
    filtered_rows = [r for r in source_meta_rows if r['source'] == target_source]
    temp_dir.mkdir(parents=True, exist_ok=True)
    img_dir = temp_dir / 'images' / 'test'
    lbl_dir = temp_dir / 'labels' / 'test'
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)

    for r in filtered_rows:
        src_img = Path('subject_wise_dataset') / 'images' / 'test' / r['final_filename']
        src_lbl = Path('subject_wise_dataset') / 'labels' / 'test' / r['label_filename']
        dst_img = img_dir / r['final_filename']
        dst_lbl = lbl_dir / r['label_filename']
        shutil.copy2(src_img, dst_img)
        shutil.copy2(src_lbl, dst_lbl)

    yaml_content = f"""path: {temp_dir.resolve().as_posix()}
train: images/test
val: images/test
test: images/test

nc: 4
names:
  0: engaged
  1: confused
  2: bored
  3: frustrated
"""
    yaml_path = temp_dir / 'data.yaml'
    with open(yaml_path, 'w', encoding='utf-8') as f:
        f.write(yaml_content)

    return yaml_path, len(filtered_rows), lbl_dir


def run_evaluation(args):
    repo_root = Path('.').resolve()
    settings.update({'datasets_dir': str(repo_root.as_posix())})

    weights_path = Path(args.weights).resolve()
    dataset_dir = Path(args.dataset_dir).resolve()
    data_yaml_path = dataset_dir / 'data.yaml'

    test_img_dir = dataset_dir / 'images' / args.split
    test_lbl_dir = dataset_dir / 'labels' / args.split

    # Pre-test verification
    print("\n" + "=" * 70)
    print("PRE-TEST VERIFICATION:")
    print("=" * 70)
    print(f"[MODEL]\n  Frozen best.pt: {weights_path.name} (Ada: {weights_path.exists()})")
    print(f"[DATASET]\n  data.yaml: {data_yaml_path.name} (Ada: {data_yaml_path.exists()})")
    
    test_imgs = list(test_img_dir.glob('*.*')) if test_img_dir.exists() else []
    test_lbls = list(test_lbl_dir.glob('*.txt')) if test_lbl_dir.exists() else []
    print(f"  test images found: {len(test_imgs)}\n  test labels found: {len(test_lbls)}")
    print(f"[MODE]\n  training: DISABLED\n  resume: DISABLED\n  split: {args.split}")
    print("=" * 70 + "\n")

    if not weights_path.exists():
        raise FileNotFoundError(f"Model best.pt tidak ditemukan di {weights_path}")
    if not data_yaml_path.exists():
        raise FileNotFoundError(f"data.yaml tidak ditemukan di {data_yaml_path}")

    # Output directory
    eval_main_dir = Path(args.project) / args.name
    eval_main_dir.mkdir(parents=True, exist_ok=True)

    # Backup file lama jika sudah ada
    for fname in ['final_test_metrics.csv', 'final_test_summary.txt', 'test_evaluation_per_class.csv']:
        orig_file = eval_main_dir / fname
        if orig_file.exists():
            stem = orig_file.stem
            suffix = orig_file.suffix
            backup_file = eval_main_dir / f"{stem}_backup_v1{suffix}"
            if not backup_file.exists():
                shutil.copy2(orig_file, backup_file)
                logger.info(f"Backup dibuat: {backup_file.name}")

    # Baca metadata test set
    meta_path = dataset_dir / 'metadata.csv'
    test_meta_rows = []
    with open(meta_path, 'r', encoding='utf-8') as f:
        for row in csv.DictReader(f):
            if row['assigned_split'] == args.split:
                test_meta_rows.append(row)

    pub_test_cnt = sum(1 for r in test_meta_rows if r['source'] == 'public')
    priv_test_cnt = sum(1 for r in test_meta_rows if r['source'] == 'private')

    # Load frozen model
    logger.info(f"Memuat model FROZEN dari: {weights_path}")
    model = YOLO(str(weights_path))

    # 1. Evaluasi Domain A: Overall Test (156 images)
    logger.info(f"Menjalankan evaluasi split '{args.split}'...")
    metrics_overall = model.val(
        data=str(data_yaml_path),
        split=args.split,
        batch=args.batch,
        imgsz=args.imgsz,
        device=args.device,
        project=str(args.project),
        name=args.name,
        exist_ok=True,
        plots=True,
        save_json=True
    )
    overall_res, per_class_res, raw_ap_matrix = extract_per_class_metrics(metrics_overall, CLASS_NAMES, test_lbl_dir)

    # 2. Evaluasi Domain B: Public Test (95 images)
    temp_pub_dir = eval_main_dir / 'temp_subsets' / 'test_public'
    yaml_pub, pub_cnt, pub_lbl_dir = create_temp_eval_subset(test_meta_rows, 'public', temp_pub_dir)
    metrics_pub = model.val(
        data=str(yaml_pub),
        split='test',
        batch=args.batch,
        imgsz=args.imgsz,
        device=args.device,
        project=str(eval_main_dir),
        name='eval_public_unseen',
        exist_ok=True,
        plots=True,
        save_json=True
    )
    pub_overall_res, pub_per_class_res, _ = extract_per_class_metrics(metrics_pub, CLASS_NAMES, pub_lbl_dir)

    # 3. Evaluasi Domain C: Private Test (61 images)
    temp_priv_dir = eval_main_dir / 'temp_subsets' / 'test_private'
    yaml_priv, priv_cnt, priv_lbl_dir = create_temp_eval_subset(test_meta_rows, 'private', temp_priv_dir)
    metrics_priv = model.val(
        data=str(yaml_priv),
        split='test',
        batch=args.batch,
        imgsz=args.imgsz,
        device=args.device,
        project=str(eval_main_dir),
        name='eval_private_unseen',
        exist_ok=True,
        plots=True,
        save_json=True
    )
    priv_overall_res, priv_per_class_res, _ = extract_per_class_metrics(metrics_priv, CLASS_NAMES, priv_lbl_dir)

    # Bersihkan temp directory
    if (eval_main_dir / 'temp_subsets').exists():
        shutil.rmtree(eval_main_dir / 'temp_subsets', ignore_errors=True)

    # 4. Sanity Checks Rigorous
    mean_ap50 = float(np.mean([d['ap50'] for d in per_class_res.values()]))
    mean_ap75 = float(np.mean([d['ap75'] for d in per_class_res.values()]))
    mean_map50_95 = float(np.mean([d['map50_95'] for d in per_class_res.values()]))

    diff_ap50 = abs(mean_ap50 - overall_res['map50'])
    diff_ap75 = abs(mean_ap75 - overall_res['map75'])
    diff_map50_95 = abs(mean_map50_95 - overall_res['map50_95'])

    sanity_pass = (diff_ap50 < 1e-4) and (diff_ap75 < 1e-4) and (diff_map50_95 < 1e-4)

    # 5. Buat raw_per_class_ap_matrix.csv
    raw_matrix_path = eval_main_dir / 'raw_per_class_ap_matrix.csv'
    with open(raw_matrix_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['class', 'ap50', 'ap55', 'ap60', 'ap65', 'ap70', 'ap75', 'ap80', 'ap85', 'ap90', 'ap95', 'mean_ap50_95'])
        for cname in CLASS_NAMES:
            ap_vec = raw_ap_matrix[cname]
            writer.writerow([cname] + [f"{v:.6f}" for v in ap_vec] + [f"{ap_vec.mean():.6f}"])

    # 6. Buat final_test_metrics_corrected.csv & final_test_metrics.csv
    final_metrics_rows = []
    for cname in CLASS_NAMES:
        d = per_class_res[cname]
        final_metrics_rows.append({
            'class': cname,
            'images': d['images'],
            'instances': d['instances'],
            'precision': f"{d['precision']:.6f}",
            'recall': f"{d['recall']:.6f}",
            'f1': f"{d['f1']:.6f}",
            'map50': f"{d['ap50']:.6f}",
            'map75': f"{d['ap75']:.6f}",
            'map50_95': f"{d['map50_95']:.6f}"
        })
    final_metrics_rows.append({
        'class': 'overall',
        'images': overall_res['images'],
        'instances': overall_res['instances'],
        'precision': f"{overall_res['precision']:.6f}",
        'recall': f"{overall_res['recall']:.6f}",
        'f1': f"{overall_res['f1']:.6f}",
        'map50': f"{overall_res['map50']:.6f}",
        'map75': f"{overall_res['map75']:.6f}",
        'map50_95': f"{overall_res['map50_95']:.6f}"
    })

    fieldnames = ['class', 'images', 'instances', 'precision', 'recall', 'f1', 'map50', 'map75', 'map50_95']
    for out_name in ['final_test_metrics_corrected.csv', 'final_test_metrics.csv']:
        with open(eval_main_dir / out_name, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(final_metrics_rows)

    # 7. Buat test_evaluation_per_class_corrected.csv & test_evaluation_per_class.csv
    domain_per_class_rows = []
    domain_list = [
        ('overall_unseen', per_class_res),
        ('public_unseen', pub_per_class_res),
        ('private_unseen', priv_per_class_res)
    ]
    for domain_name, dom_res in domain_list:
        for cname in CLASS_NAMES:
            d = dom_res[cname]
            domain_per_class_rows.append({
                'subset': domain_name,
                'class': cname,
                'images': d['images'],
                'instances': d['instances'],
                'precision': f"{d['precision']:.6f}",
                'recall': f"{d['recall']:.6f}",
                'f1': f"{d['f1']:.6f}",
                'map50': f"{d['ap50']:.6f}",
                'map75': f"{d['ap75']:.6f}",
                'map50_95': f"{d['map50_95']:.6f}"
            })
    dom_fieldnames = ['subset', 'class', 'images', 'instances', 'precision', 'recall', 'f1', 'map50', 'map75', 'map50_95']
    for out_name in ['test_evaluation_per_class_corrected.csv', 'test_evaluation_per_class.csv']:
        with open(eval_main_dir / out_name, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=dom_fieldnames)
            writer.writeheader()
            writer.writerows(domain_per_class_rows)

    # 8. Buat final_test_summary_corrected.txt & final_test_summary.txt
    summary_lines = [
        "==================================================",
        "YOLOv13n SUBJECT-WISE FINAL HELD-OUT TEST",
        "==================================================",
        "",
        f"Model:\n{weights_path}",
        "",
        f"Dataset:\n{data_yaml_path}",
        "",
        f"Split:\n{args.split}",
        "",
        f"Images: {overall_res['images']}",
        f"Instances: {overall_res['instances']}",
        "",
        "OVERALL RESULTS",
        f"Precision: {overall_res['precision']:.6f}",
        f"Recall: {overall_res['recall']:.6f}",
        f"F1: {overall_res['f1']:.6f}",
        f"mAP@0.5: {overall_res['map50']:.6f}",
        f"mAP@0.75: {overall_res['map75']:.6f}",
        f"mAP@0.5:0.95: {overall_res['map50_95']:.6f}",
        "",
        "SPEED",
        f"Preprocess: {overall_res['preprocess_ms']:.2f} ms",
        f"Inference: {overall_res['inference_ms']:.2f} ms",
        f"Postprocess: {overall_res['postprocess_ms']:.2f} ms",
        "",
        "PER-CLASS RESULTS"
    ]
    for cname in CLASS_NAMES:
        d = per_class_res[cname]
        summary_lines.append(f"{cname:<12}: Images={d['images']:<3} Instances={d['instances']:<3} | P={d['precision']:.6f} | R={d['recall']:.6f} | F1={d['f1']:.6f} | AP50={d['ap50']:.6f} | AP75={d['ap75']:.6f} | AP50-95={d['map50_95']:.6f}")

    summary_lines.extend([
        "",
        "DOMAIN BREAKDOWN (PUBLIC VS PRIVATE)",
        f"Public  Test ({pub_cnt} images) : P={pub_overall_res['precision']:.6f} | R={pub_overall_res['recall']:.6f} | F1={pub_overall_res['f1']:.6f} | mAP50={pub_overall_res['map50']:.6f} | mAP75={pub_overall_res['map75']:.6f} | mAP50-95={pub_overall_res['map50_95']:.6f}",
        f"Private Test ({priv_cnt} images) : P={priv_overall_res['precision']:.6f} | R={priv_overall_res['recall']:.6f} | F1={priv_overall_res['f1']:.6f} | mAP50={priv_overall_res['map50']:.6f} | mAP75={priv_overall_res['map75']:.6f} | mAP50-95={priv_overall_res['map50_95']:.6f}",
        "",
        "METRIC EXPORT VALIDATION & SANITY CHECKS:",
        f"  Mean per-class AP50 vs overall mAP50      : {mean_ap50:.6f} vs {overall_res['map50']:.6f} (diff: {diff_ap50:.2e})",
        f"  Mean per-class AP75 vs overall mAP75      : {mean_ap75:.6f} vs {overall_res['map75']:.6f} (diff: {diff_ap75:.2e})",
        f"  Mean per-class AP50:95 vs overall mAP50:95: {mean_map50_95:.6f} vs {overall_res['map50_95']:.6f} (diff: {diff_map50_95:.2e})",
        f"  Status                                    : {'PASS' if sanity_pass else 'FAIL'}",
        "",
        "==================================================",
        "MODEL WAS NOT RETRAINED OR TUNED USING TEST DATA.",
        "=================================================="
    ])
    summary_str = "\n".join(summary_lines)
    for out_name in ['final_test_summary_corrected.txt', 'final_test_summary.txt']:
        with open(eval_main_dir / out_name, 'w', encoding='utf-8') as f:
            f.write(summary_str)

    # 9. Print Terminal Summary Sesuai Format Bagian 19
    print("\n" + "=" * 75)
    print("EXPORTER CORRECTION & SANITY CHECK REPORT:")
    print("=" * 75)
    print(f"[MODEL]\nFrozen best.pt: {weights_path}")
    print(f"\n[TEST]\nImages: {overall_res['images']}\nInstances: {overall_res['instances']}")
    print(f"\n[OVERALL VERIFIED]\nPrecision : {overall_res['precision']:.6f}\nRecall    : {overall_res['recall']:.6f}\nF1        : {overall_res['f1']:.6f}\nmAP50     : {overall_res['map50']:.6f}\nmAP75     : {overall_res['map75']:.6f}\nmAP50-95  : {overall_res['map50_95']:.6f}")
    
    print("\n[PER CLASS]")
    for cname in CLASS_NAMES:
        d = per_class_res[cname]
        print(f"{cname}:")
        print(f"  images   : {d['images']}")
        print(f"  instances: {d['instances']}")
        print(f"  P        : {d['precision']:.6f}")
        print(f"  R        : {d['recall']:.6f}")
        print(f"  F1       : {d['f1']:.6f}")
        print(f"  AP50     : {d['ap50']:.6f}")
        print(f"  AP75     : {d['ap75']:.6f}")
        print(f"  AP50-95  : {d['map50_95']:.6f}")

    print("\n[SANITY CHECK]")
    print(f"mean AP50      = {mean_ap50:.6f}\noverall mAP50  = {overall_res['map50']:.6f}\ndifference     = {diff_ap50:.2e}")
    print(f"\nmean AP75      = {mean_ap75:.6f}\noverall mAP75  = {overall_res['map75']:.6f}\ndifference     = {diff_ap75:.2e}")
    print(f"\nmean AP50-95      = {mean_map50_95:.6f}\noverall mAP50-95  = {overall_res['map50_95']:.6f}\ndifference        = {diff_map50_95:.2e}")
    print(f"\nSTATUS: {'PASS' if sanity_pass else 'FAIL'}")
    print("=" * 75 + "\n")

    return overall_res, per_class_res, raw_ap_matrix, sanity_pass


if __name__ == '__main__':
    args = parse_args()
    try:
        run_evaluation(args)
    except Exception as e:
        logger.error(f"Evaluasi gagal: {e}")
        sys.exit(1)
