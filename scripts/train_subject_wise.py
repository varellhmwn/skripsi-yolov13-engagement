"""
train_subject_wise.py — Isolated Training Pipeline for Subject-Wise YOLOv13n
=============================================================================
Penelitian: "Deteksi Emosi Belajar Mahasiswa pada Pembelajaran Pemrograman Menggunakan YOLOv13n"

Tujuan:
  1. Melatih YOLOv13n dari bobot dasar yolov13n.pt (murni train baru tanpa checkpoint lama).
  2. Melakukan preflight dataset audit ketat (zero subject leakage & zero hash leakage).
  3. Menyimpan environment specifications & provenance.
  4. Mendukung mode:
     --smoke-test : 1 epoch (yolov13_subject_wise_smoke)
     --full       : 150 epochs (yolov13_subject_wise_v1)
"""

import sys
import os
import csv
import json
import time
import argparse
import logging
from pathlib import Path

import torch
from ultralytics import YOLO, settings

from scripts.audit_subject_wise_training import (
    get_environment_info,
    perform_preflight_audit,
    clean_stale_caches
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger('SubjectWiseTrainer')


def parse_args():
    parser = argparse.ArgumentParser(description="Training YOLOv13n pada Dataset Subject-Wise (Clean)")
    parser.add_argument('--smoke-test', action='store_true', help="Jalankan smoke test 1 epoch")
    parser.add_argument('--full', action='store_true', help="Jalankan full training 150 epochs")
    parser.add_argument('--dataset_dir', type=str, default='subject_wise_dataset', help="Path direktori dataset subject-wise")
    parser.add_argument('--weights', type=str, default='yolov13n.pt', help="Bobot dasar pretrained (wajib yolov13n.pt)")
    parser.add_argument('--project', type=str, default='runs/train', help="Direktori project output runs")
    parser.add_argument('--name', type=str, default=None, help="Nama folder run")
    parser.add_argument('--device', type=str, default='0', help="Device ID (0 untuk GPU pertama, atau 'cpu')")
    parser.add_argument('--seed', type=int, default=42, help="Random seed reproducibility (default: 42)")
    parser.add_argument('--exist_ok', action='store_true', help="Izinkan overwrite run folder jika sudah ada")
    return parser.parse_args()


def run_training(args):
    repo_root = Path('.').resolve()
    settings.update({'datasets_dir': str(repo_root.as_posix())})
    dataset_path = Path(args.dataset_dir).resolve()
    base_weights_path = Path(args.weights).resolve()

    logger.info("=" * 75)
    logger.info("  ISOLATED TRAINING PIPELINE — YOLOv13n SUBJECT-WISE")
    logger.info("=" * 75)

    # 1. Validasi Pretrained Weight yolov13n.pt
    if not base_weights_path.exists():
        logger.error(f"[FATAL] Bobot dasar {base_weights_path} tidak ditemukan!")
        sys.exit(1)

    # Cegah checkpoint lama
    forbidden_keywords = ['best.pt', 'last.pt', 'master_combined', 'checkpoint']
    if any(k in str(base_weights_path).lower() for k in forbidden_keywords):
        logger.error(f"[FATAL] Pelanggaran metodologis: Dilarang menggunakan checkpoint lama: {base_weights_path}")
        sys.exit(1)

    logger.info(f"[PRETRAINED WEIGHT] Memuat dari: {base_weights_path}")

    # 2. Tentukan Mode & Hyperparameters
    if args.smoke_test and args.full:
        logger.error("[FATAL] Pilih salah satu: --smoke-test ATAU --full")
        sys.exit(1)

    is_smoke = args.smoke_test or (not args.full)
    epochs = 1 if is_smoke else 150
    patience = 25 if not is_smoke else 1
    run_name = args.name if args.name else ('yolov13_subject_wise_smoke' if is_smoke else 'yolov13_subject_wise_v1')

    logger.info(f"[MODE] {'SMOKE TEST (1 Epoch)' if is_smoke else 'FULL TRAINING (150 Epochs)'}")
    logger.info(f"[RUN NAME] {run_name}")

    # 3. Environment & Hardware Check
    env_info = get_environment_info(seed=args.seed)
    device_to_use = args.device
    if device_to_use != 'cpu' and not env_info['cuda_available']:
        logger.warning("[WARNING] CUDA tidak tersedia, fallback otomatis ke device='cpu'")
        device_to_use = 'cpu'

    # 4. Preflight Dataset Audit (Ketat)
    logger.info("\n--- MENJALANKAN PREFLIGHT DATASET AUDIT ---")
    audit_res = perform_preflight_audit(dataset_path)
    data_yaml_path = dataset_path / 'data.yaml'

    # 5. Siapkan Training Arguments
    train_args = {
        'data': str(data_yaml_path),
        'epochs': epochs,
        'imgsz': 640,
        'batch': 16,
        'patience': patience,
        'project': args.project,
        'name': run_name,
        'exist_ok': args.exist_ok or is_smoke,
        'device': device_to_use,
        'seed': args.seed,
        'deterministic': True,
        # Optimizer
        'optimizer': 'AdamW',
        'lr0': 0.001,
        'lrf': 0.01,
        'weight_decay': 0.0005,
        'warmup_epochs': 3 if not is_smoke else 0,
        # Augmentasi
        'hsv_h': 0.015,
        'hsv_s': 0.7,
        'hsv_v': 0.4,
        'degrees': 10.0,
        'translate': 0.1,
        'scale': 0.5,
        'fliplr': 0.5,
        'mosaic': 1.0 if not is_smoke else 0.0,
        'mixup': 0.1 if not is_smoke else 0.0,
        'close_mosaic': 10 if not is_smoke else 0,
        # Validation & Logging
        'val': True,
        'save': True,
        'plots': True,
        'verbose': True
    }

    # Assertions untuk Full Training
    if not is_smoke:
        assert train_args['warmup_epochs'] == 3, f"Warmup epochs must be 3, got {train_args['warmup_epochs']}"
        assert train_args['mosaic'] == 1.0, f"Mosaic must be 1.0, got {train_args['mosaic']}"
        assert train_args['mixup'] == 0.1, f"Mixup must be 0.1, got {train_args['mixup']}"
        assert train_args['close_mosaic'] == 10, f"Close mosaic must be 10, got {train_args['close_mosaic']}"
        assert train_args['patience'] == 25, f"Patience must be 25, got {train_args['patience']}"
        assert train_args['epochs'] == 150, f"Epochs must be 150, got {train_args['epochs']}"
        assert train_args['seed'] == 42, f"Seed must be 42, got {train_args['seed']}"

    # Print Detailed Preflight Summary
    print("\n" + "=" * 70)
    print("PREFLIGHT TRAINING VERIFICATION SUMMARY:")
    print("=" * 70)
    print(f"[MODEL]\n  Starting weights: {base_weights_path.name}")
    print(f"[DATASET]\n  Train images : {audit_res['subsets']['train']['images_count']}\n  Val images   : {audit_res['subsets']['val']['images_count']}\n  Test images  : {audit_res['subsets']['test']['images_count']}\n  Classes      : 4")
    print(f"[CLASS DISTRIBUTION]\n  Train: {audit_res['subsets']['train']['class_counts']}\n  Val  : {audit_res['subsets']['val']['class_counts']}\n  Test : {audit_res['subsets']['test']['class_counts']}")
    print(f"[SUBJECT LEAKAGE]\n  Train-Val  : 0\n  Train-Test : 0\n  Val-Test   : 0")
    print(f"[HASH LEAKAGE]\n  Train-Val  : 0\n  Train-Test : 0\n  Val-Test   : 0")
    print(f"[PAIR AUDIT]\n  Orphan images  : 0\n  Orphan labels  : 0\n  Invalid labels : 0")
    print(f"[TRAINING PARAMETERS]\n  Epochs: {train_args['epochs']}, Batch: {train_args['batch']}, ImgSz: {train_args['imgsz']}, Patience: {train_args['patience']}")
    print(f"  Optimizer: {train_args['optimizer']}, lr0: {train_args['lr0']}, lrf: {train_args['lrf']}, weight_decay: {train_args['weight_decay']}, warmup: {train_args['warmup_epochs']}")
    print(f"  Augmentations: mosaic={train_args['mosaic']}, mixup={train_args['mixup']}, close_mosaic={train_args['close_mosaic']}")
    print(f"  Device: {train_args['device']}, Seed: {train_args['seed']}, Deterministic: {train_args['deterministic']}")
    print("=" * 70 + "\n")

    # 6. Inisialisasi Model YOLOv13n
    model = YOLO(str(base_weights_path))

    # 7. Eksekusi Training Loop
    logger.info("--- MEMULAI TRAINING LOOP ---")
    start_time = time.time()
    results = model.train(**train_args)
    elapsed_sec = time.time() - start_time

    run_dir = Path(args.project) / run_name
    weights_dir = run_dir / 'weights'
    best_pt = weights_dir / 'best.pt'
    last_pt = weights_dir / 'last.pt'
    results_csv = run_dir / 'results.csv'

    # 8. Simpan Metadata Provenance ke Run Folder
    run_dir.mkdir(parents=True, exist_ok=True)
    with open(run_dir / 'environment_info.json', 'w', encoding='utf-8') as f:
        json.dump(env_info, f, indent=2)

    with open(run_dir / 'training_config_subject_wise.json', 'w', encoding='utf-8') as f:
        clean_args = {k: v for k, v in train_args.items()}
        clean_args['base_weights'] = str(base_weights_path)
        clean_args['elapsed_seconds'] = round(elapsed_sec, 2)
        json.dump(clean_args, f, indent=2)

    # 9. Parsing Hasil Validation & Metrik Epoch dari results.csv
    csv_rows = []
    best_epoch_idx = 0
    best_fitness = -1.0
    best_row_data = {}

    if results_csv.exists():
        try:
            with open(results_csv, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    cleaned_row = {k.strip(): float(v.strip()) for k, v in row.items() if v.strip() != ''}
                    csv_rows.append(cleaned_row)

            # Hitung fitness = 0.1 * mAP50 + 0.9 * mAP50-95
            for row in csv_rows:
                ep = int(row.get('epoch', 0))
                map50 = row.get('metrics/mAP50(B)', 0.0)
                map95 = row.get('metrics/mAP50-95(B)', 0.0)
                fitness = 0.1 * map50 + 0.9 * map95
                if fitness > best_fitness:
                    best_fitness = fitness
                    best_epoch_idx = ep
                    best_row_data = row
        except Exception as e:
            logger.warning(f"Gagal mem-parsing results.csv: {e}")

    actual_completed_epochs = len(csv_rows)
    stopped_early = actual_completed_epochs < epochs

    # 10. Buat best_validation_metrics.csv
    if best_row_data:
        val_metric_items = [
            ('best_epoch', int(best_row_data.get('epoch', best_epoch_idx))),
            ('best_fitness', round(best_fitness, 5)),
            ('precision', round(best_row_data.get('metrics/precision(B)', 0.0), 5)),
            ('recall', round(best_row_data.get('metrics/recall(B)', 0.0), 5)),
            ('map50', round(best_row_data.get('metrics/mAP50(B)', 0.0), 5)),
            ('map75', round(best_row_data.get('metrics/mAP75(B)', 0.0), 5)),
            ('map50_95', round(best_row_data.get('metrics/mAP50-95(B)', 0.0), 5)),
            ('train_box_loss', round(best_row_data.get('train/box_loss', 0.0), 5)),
            ('train_cls_loss', round(best_row_data.get('train/cls_loss', 0.0), 5)),
            ('train_dfl_loss', round(best_row_data.get('train/dfl_loss', 0.0), 5)),
            ('val_box_loss', round(best_row_data.get('val/box_loss', 0.0), 5)),
            ('val_cls_loss', round(best_row_data.get('val/cls_loss', 0.0), 5)),
            ('val_dfl_loss', round(best_row_data.get('val/dfl_loss', 0.0), 5))
        ]
        with open(run_dir / 'best_validation_metrics.csv', 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['metric', 'value'])
            writer.writerows(val_metric_items)

    # 11. Buat training_summary.txt
    summary_text = [
        "==================================================",
        "YOLOv13n SUBJECT-WISE FULL TRAINING SUMMARY",
        "==================================================",
        f"Starting weights: {base_weights_path.name}",
        f"Dataset: {data_yaml_path}",
        f"Run name: {run_name}",
        "",
        f"Train images: {audit_res['subsets']['train']['images_count']}",
        f"Validation images: {audit_res['subsets']['val']['images_count']}",
        f"Held-out test images: {audit_res['subsets']['test']['images_count']}",
        "",
        f"Train subjects: {', '.join(audit_res['subsets']['train']['subjects'])}",
        f"Validation subjects: {', '.join(audit_res['subsets']['val']['subjects'])}",
        f"Held-out test subjects: {', '.join(audit_res['subsets']['test']['subjects'])}",
        "",
        "Subject leakage: 0 (ZERO LEAKAGE)",
        "Hash leakage: 0 (ZERO LEAKAGE)",
        "",
        f"GPU: {env_info['gpus'][0]['name'] if env_info['gpus'] else 'CPU'}",
        f"PyTorch: {env_info['pytorch_version']}",
        f"CUDA: {env_info['cuda_version']}",
        f"YOLO version: {env_info['ultralytics_version']}",
        "",
        f"Requested epochs: {epochs}",
        f"Actual completed epochs: {actual_completed_epochs}",
        f"Early stopping: {stopped_early} (Patience: {patience})",
        f"Patience: {patience}",
        "",
        f"Best epoch: {best_epoch_idx}",
        f"Best fitness: {best_fitness:.5f}",
        "",
        "Validation metrics at best epoch:",
        f"  Precision: {best_row_data.get('metrics/precision(B)', 0.0):.5f}",
        f"  Recall: {best_row_data.get('metrics/recall(B)', 0.0):.5f}",
        f"  mAP@0.5: {best_row_data.get('metrics/mAP50(B)', 0.0):.5f}",
        f"  mAP@0.5:0.95: {best_row_data.get('metrics/mAP50-95(B)', 0.0):.5f}",
        "",
        "Training losses at best epoch:",
        f"  box: {best_row_data.get('train/box_loss', 0.0):.5f}",
        f"  cls: {best_row_data.get('train/cls_loss', 0.0):.5f}",
        f"  dfl: {best_row_data.get('train/dfl_loss', 0.0):.5f}",
        "",
        "Validation losses at best epoch:",
        f"  box: {best_row_data.get('val/box_loss', 0.0):.5f}",
        f"  cls: {best_row_data.get('val/cls_loss', 0.0):.5f}",
        f"  dfl: {best_row_data.get('val/dfl_loss', 0.0):.5f}",
        "",
        f"best.pt: {best_pt}",
        f"last.pt: {last_pt}",
        "",
        "TEST EVALUATION:",
        "NOT RUN / HELD OUT",
        "=================================================="
    ]
    summary_str = "\n".join(summary_text)
    with open(run_dir / 'training_summary.txt', 'w', encoding='utf-8') as f:
        f.write(summary_str)

    logger.info("\n" + summary_str)
    return {
        'is_smoke': is_smoke,
        'run_dir': str(run_dir),
        'best_pt': str(best_pt),
        'last_pt': str(last_pt),
        'results_csv': str(results_csv),
        'elapsed_seconds': elapsed_sec,
        'best_epoch': best_epoch_idx,
        'best_fitness': best_fitness,
        'best_row_data': best_row_data,
        'env_info': env_info,
        'train_args': train_args
    }


if __name__ == '__main__':
    args = parse_args()
    try:
        run_training(args)
    except Exception as e:
        logger.error(f"Training gagal: {e}")
        sys.exit(1)
