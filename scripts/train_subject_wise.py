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

    # 6. Inisialisasi Model YOLOv13n
    model = YOLO(str(base_weights_path))

    # 7. Eksekusi Training Loop
    logger.info("\n--- MEMULAI TRAINING LOOP ---")
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

    with open(run_dir / 'training_config.json', 'w', encoding='utf-8') as f:
        clean_args = {k: v for k, v in train_args.items()}
        clean_args['base_weights'] = str(base_weights_path)
        clean_args['elapsed_seconds'] = round(elapsed_sec, 2)
        json.dump(clean_args, f, indent=2)

    # 9. Parsing Hasil Validation & Metrik Epoch
    val_metrics = {}
    if results_csv.exists():
        try:
            with open(results_csv, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                rows = list(reader)
                if rows:
                    last_row = rows[-1]
                    # Bersihkan whitespace key
                    cleaned_last = {k.strip(): v.strip() for k, v in last_row.items()}
                    val_metrics = cleaned_last
        except Exception as e:
            logger.warning(f"Gagal membaca results.csv: {e}")

    logger.info("\n" + "=" * 75)
    logger.info(f"  {'SMOKE TEST' if is_smoke else 'TRAINING'} COMPLETED IN {elapsed_sec:.2f} DETIK")
    logger.info("=" * 75)
    logger.info(f"  Run Directory : {run_dir.resolve()}")
    logger.info(f"  Best Weights  : {best_pt} (Ada: {best_pt.exists()})")
    logger.info(f"  Last Weights  : {last_pt} (Ada: {last_pt.exists()})")
    logger.info(f"  Results CSV   : {results_csv} (Ada: {results_csv.exists()})")

    return {
        'is_smoke': is_smoke,
        'run_dir': str(run_dir),
        'best_pt': str(best_pt),
        'last_pt': str(last_pt),
        'results_csv': str(results_csv),
        'elapsed_seconds': elapsed_sec,
        'val_metrics': val_metrics,
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
