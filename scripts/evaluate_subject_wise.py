"""
evaluate_subject_wise.py — Standalone Evaluation for Subject-Wise YOLOv13n
==========================================================================
Penelitian: "Deteksi Emosi Belajar Mahasiswa pada Pembelajaran Pemrograman Menggunakan YOLOv13n"

Tujuan:
  1. Menjalankan evaluasi independen pada Test Set (156 citra).
  2. Mengevaluasi 3 domain pengujian:
     A. Overall Unseen-Subject Test (156 citra)
     B. Public Unseen-Subject Test (95 citra)
     C. Private Unseen-Subject Test (61 citra)
  3. Menghitung Precision, Recall, F1-Score, mAP@0.5, mAP@0.5:0.95, per-class & latency.
  4. Menghasilkan artefak CSV:
     - test_evaluation_summary.csv
     - test_evaluation_per_class.csv
     - test_manifest.csv
     - legacy_vs_subjectwise_template.csv
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

import torch
from ultralytics import YOLO

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger('SubjectWiseEvaluator')

CLASS_NAMES = ['engaged', 'confused', 'bored', 'frustrated']
CLASS_NAME_TO_ID = {name: i for i, name in enumerate(CLASS_NAMES)}


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluasi Independen Subject-Wise YOLOv13n")
    parser.add_argument('--weights', type=str, required=True, help="Path ke weights model (misal: runs/train/yolov13_subject_wise_v1/weights/best.pt)")
    parser.add_argument('--dataset_dir', type=str, default='subject_wise_dataset', help="Direktori dataset subject-wise")
    parser.add_argument('--split', type=str, default='test', choices=['val', 'test'], help="Subset yang dievaluasi")
    parser.add_argument('--output_dir', type=str, default=None, help="Direktori output hasil evaluasi (default: folder run weights)")
    parser.add_argument('--device', type=str, default='0', help="Device ID (0 atau 'cpu')")
    parser.add_argument('--batch', type=int, default=16, help="Batch size evaluasi")
    parser.add_argument('--imgsz', type=int, default=640, help="Resolusi citra evaluasi")
    return parser.parse_args()


def calculate_f1(precision: float, recall: float) -> float:
    """Menghitung F1-score dengan proteksi division-by-zero."""
    if (precision + recall) <= 1e-8:
        return 0.0
    return 2.0 * (precision * recall) / (precision + recall)


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

    # Buat data.yaml mini untuk subset ini
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

    return yaml_path, len(filtered_rows)


def evaluate_split_domain(model, data_yaml: str, split_name: str, device: str, batch: int, imgsz: int, project_dir: Path, run_name: str):
    """Menjalankan model.val() pada domain pengujian tertentu."""
    logger.info(f"MENJALANKAN EVALUASI: {run_name} (YAML: {data_yaml})")
    metrics = model.val(
        data=str(data_yaml),
        split=split_name,
        batch=batch,
        imgsz=imgsz,
        device=device,
        project=str(project_dir),
        name=run_name,
        exist_ok=True,
        plots=True,
        save_json=True
    )

    box = metrics.box
    precision_all = float(box.p)
    recall_all = float(box.r)
    map50_all = float(box.map50)
    map50_95_all = float(box.map)
    f1_all = calculate_f1(precision_all, recall_all)

    # Speed metrics (ms)
    speed = metrics.speed
    preprocess_ms = speed.get('preprocess', 0.0)
    inference_ms = speed.get('inference', 0.0)
    postprocess_ms = speed.get('postprocess', 0.0)

    # Per-class metrics
    p_per_class = box.p_curve if hasattr(box, 'p_curve') else []
    # Ambil nilai per class langsung dari box properties
    # box.class_result(i) -> (p, r, map50, map)
    per_class_results = {}
    for i, cname in enumerate(CLASS_NAMES):
        try:
            res = box.class_result(i)
            p_c, r_c, m50_c, m95_c = float(res[0]), float(res[1]), float(res[2]), float(res[3])
        except Exception:
            # Fallback jika class_result tidak tersedia
            p_c, r_c, m50_c, m95_c = float(box.p[i]) if hasattr(box.p, '__getitem__') else 0.0, float(box.r[i]) if hasattr(box.r, '__getitem__') else 0.0, float(box.map50), float(box.map)
        f1_c = calculate_f1(p_c, r_c)
        per_class_results[cname] = {
            'precision': p_c,
            'recall': r_c,
            'f1': f1_c,
            'map50': m50_c,
            'map50_95': m95_c
        }

    return {
        'precision': precision_all,
        'recall': recall_all,
        'f1': f1_all,
        'map50': map50_all,
        'map50_95': map50_95_all,
        'preprocess_ms': preprocess_ms,
        'inference_ms': inference_ms,
        'postprocess_ms': postprocess_ms,
        'per_class': per_class_results
    }


def run_evaluation(args):
    weights_path = Path(args.weights).resolve()
    if not weights_path.exists():
        logger.error(f"[FATAL] File weights tidak ditemukan: {weights_path}")
        sys.exit(1)

    dataset_dir = Path(args.dataset_dir).resolve()
    meta_path = dataset_dir / 'metadata.csv'
    if not meta_path.exists():
        logger.error(f"[FATAL] metadata.csv tidak ditemukan di {dataset_dir}")
        sys.exit(1)

    # Tentukan output dir
    if args.output_dir:
        eval_out_dir = Path(args.output_dir).resolve()
    else:
        eval_out_dir = weights_path.parent.parent / 'evaluations_subject_wise'
    eval_out_dir.mkdir(parents=True, exist_ok=True)

    # Baca metadata test set
    test_meta_rows = []
    with open(meta_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row['assigned_split'] == args.split:
                test_meta_rows.append(row)

    logger.info(f"Total citra {args.split} ditemukan: {len(test_meta_rows)}")
    pub_test_cnt = sum(1 for r in test_meta_rows if r['source'] == 'public')
    priv_test_cnt = sum(1 for r in test_meta_rows if r['source'] == 'private')
    logger.info(f"  - Public Test  : {pub_test_cnt} citra")
    logger.info(f"  - Private Test : {priv_test_cnt} citra")

    # 1. Simpan test_manifest.csv
    manifest_rows = []
    for r in test_meta_rows:
        manifest_rows.append({
            'filename': r['final_filename'],
            'subject_id': r['subject_id'],
            'source': r['source'],
            'class_name': r['class_name'],
            'sha256': r['sha256']
        })
    with open(eval_out_dir / 'test_manifest.csv', 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['filename', 'subject_id', 'source', 'class_name', 'sha256'])
        writer.writeheader()
        writer.writerows(manifest_rows)

    # 2. Inisialisasi Model
    logger.info(f"Memuat model evaluasi dari: {weights_path}")
    model = YOLO(str(weights_path))

    # 3. Evaluasi Domain A: Overall Test
    data_yaml_main = dataset_dir / 'data.yaml'
    res_overall = evaluate_split_domain(
        model, data_yaml_main, args.split, args.device, args.batch, args.imgsz,
        eval_out_dir, 'eval_overall_unseen'
    )

    # 4. Evaluasi Domain B: Public Test
    temp_pub_dir = eval_out_dir / 'temp_subsets' / 'test_public'
    yaml_pub, pub_cnt = create_temp_eval_subset(test_meta_rows, 'public', temp_pub_dir)
    res_public = evaluate_split_domain(
        model, yaml_pub, 'test', args.device, args.batch, args.imgsz,
        eval_out_dir, 'eval_public_unseen'
    )

    # 5. Evaluasi Domain C: Private Test
    temp_priv_dir = eval_out_dir / 'temp_subsets' / 'test_private'
    yaml_priv, priv_cnt = create_temp_eval_subset(test_meta_rows, 'private', temp_priv_dir)
    res_private = evaluate_split_domain(
        model, yaml_priv, 'test', args.device, args.batch, args.imgsz,
        eval_out_dir, 'eval_private_unseen'
    )

    # Hapus folder temporary
    if (eval_out_dir / 'temp_subsets').exists():
        shutil.rmtree(eval_out_dir / 'temp_subsets', ignore_errors=True)

    # 6. Simpan test_evaluation_summary.csv
    summary_rows = [
        {
            'subset': 'overall_unseen',
            'images': len(test_meta_rows),
            'precision': round(res_overall['precision'], 4),
            'recall': round(res_overall['recall'], 4),
            'f1': round(res_overall['f1'], 4),
            'map50': round(res_overall['map50'], 4),
            'map50_95': round(res_overall['map50_95'], 4),
            'preprocess_ms': round(res_overall['preprocess_ms'], 2),
            'inference_ms': round(res_overall['inference_ms'], 2),
            'postprocess_ms': round(res_overall['postprocess_ms'], 2)
        },
        {
            'subset': 'public_unseen',
            'images': pub_cnt,
            'precision': round(res_public['precision'], 4),
            'recall': round(res_public['recall'], 4),
            'f1': round(res_public['f1'], 4),
            'map50': round(res_public['map50'], 4),
            'map50_95': round(res_public['map50_95'], 4),
            'preprocess_ms': round(res_public['preprocess_ms'], 2),
            'inference_ms': round(res_public['inference_ms'], 2),
            'postprocess_ms': round(res_public['postprocess_ms'], 2)
        },
        {
            'subset': 'private_unseen',
            'images': priv_cnt,
            'precision': round(res_private['precision'], 4),
            'recall': round(res_private['recall'], 4),
            'f1': round(res_private['f1'], 4),
            'map50': round(res_private['map50'], 4),
            'map50_95': round(res_private['map50_95'], 4),
            'preprocess_ms': round(res_private['preprocess_ms'], 2),
            'inference_ms': round(res_private['inference_ms'], 2),
            'postprocess_ms': round(res_private['postprocess_ms'], 2)
        }
    ]
    with open(eval_out_dir / 'test_evaluation_summary.csv', 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['subset', 'images', 'precision', 'recall', 'f1', 'map50', 'map50_95', 'preprocess_ms', 'inference_ms', 'postprocess_ms'])
        writer.writeheader()
        writer.writerows(summary_rows)

    # 7. Simpan test_evaluation_per_class.csv
    per_class_rows = []
    for domain_name, domain_res, dom_cnt in [('overall_unseen', res_overall, len(test_meta_rows)), ('public_unseen', res_public, pub_cnt), ('private_unseen', res_private, priv_cnt)]:
        for cname in CLASS_NAMES:
            c_m = domain_res['per_class'][cname]
            per_class_rows.append({
                'subset': domain_name,
                'class': cname,
                'precision': round(c_m['precision'], 4),
                'recall': round(c_m['recall'], 4),
                'f1': round(c_m['f1'], 4),
                'map50': round(c_m['map50'], 4),
                'map50_95': round(c_m['map50_95'], 4)
            })
    with open(eval_out_dir / 'test_evaluation_per_class.csv', 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['subset', 'class', 'precision', 'recall', 'f1', 'map50', 'map50_95'])
        writer.writeheader()
        writer.writerows(per_class_rows)

    # 8. Simpan legacy_vs_subjectwise_template.csv
    template_rows = [
        {'metric': 'precision', 'legacy_random_split': '0.994', 'subjectwise_split': str(round(res_overall['precision'], 4))},
        {'metric': 'recall', 'legacy_random_split': '0.982', 'subjectwise_split': str(round(res_overall['recall'], 4))},
        {'metric': 'f1', 'legacy_random_split': '0.988', 'subjectwise_split': str(round(res_overall['f1'], 4))},
        {'metric': 'map50', 'legacy_random_split': '0.994', 'subjectwise_split': str(round(res_overall['map50'], 4))},
        {'metric': 'map50_95', 'legacy_random_split': '0.982', 'subjectwise_split': str(round(res_overall['map50_95'], 4))},
        {'metric': 'inference_ms', 'legacy_random_split': '6.2', 'subjectwise_split': str(round(res_overall['inference_ms'], 2))}
    ]
    with open(eval_out_dir / 'legacy_vs_subjectwise_template.csv', 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['metric', 'legacy_random_split', 'subjectwise_split'])
        writer.writeheader()
        writer.writerows(template_rows)

    logger.info("\n" + "=" * 75)
    logger.info("  EVALUASI TEST SELESAI")
    logger.info("=" * 75)
    logger.info(f"Hasil tersimpan di: {eval_out_dir.resolve()}")


if __name__ == '__main__':
    args = parse_args()
    try:
        run_evaluation(args)
    except Exception as e:
        logger.error(f"Evaluasi gagal: {e}")
        sys.exit(1)
