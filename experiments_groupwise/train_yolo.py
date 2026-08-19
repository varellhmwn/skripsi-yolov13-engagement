"""
train_yolo.py — Pelatihan Ulang YOLOv13n pada Dataset Group-Wise (v1)
=====================================================================
Melatih ulang YOLOv13n dari bobot dasar pretrained yolov13n.pt (COCO base)
pada master_combined_groupwise_v1 dengan konfigurasi eksperimen asli penelitian.
"""

import sys
import time
from pathlib import Path
from ultralytics import YOLO

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from experiments_groupwise.config import (
    GROUPWISE_DATA_YAML, PRETRAINED_YOLO_WEIGHTS, RUNS_GROUPWISE_DIR,
    YOLO_TRAIN_PARAMS
)


def train_yolov13_groupwise():
    print("=" * 65)
    print("  TAHAP 6: PELATIHAN ULANG (RETRAINING) YOLOv13n DARI SCRATCH")
    print("=" * 65)
    print(f"  Dataset YAML : {GROUPWISE_DATA_YAML}")
    print(f"  Base Weights : {PRETRAINED_YOLO_WEIGHTS}")
    print(f"  Output Run   : {RUNS_GROUPWISE_DIR}")
    print(f"  Epochs       : {YOLO_TRAIN_PARAMS['epochs']} (Patience: {YOLO_TRAIN_PARAMS['patience']})")
    print(f"  Optimizer    : {YOLO_TRAIN_PARAMS['optimizer']} (lr0={YOLO_TRAIN_PARAMS['lr0']})")
    print(f"  Image Size   : {YOLO_TRAIN_PARAMS['imgsz']} (Batch: {YOLO_TRAIN_PARAMS['batch']})")
    print("=" * 65)

    if not GROUPWISE_DATA_YAML.exists():
        raise FileNotFoundError(f"data.yaml group-wise tidak ditemukan: {GROUPWISE_DATA_YAML}")

    # Load base pretrained model
    model = YOLO(str(PRETRAINED_YOLO_WEIGHTS))

    t0 = time.time()
    results = model.train(
        data=str(GROUPWISE_DATA_YAML),
        epochs=YOLO_TRAIN_PARAMS['epochs'],
        imgsz=YOLO_TRAIN_PARAMS['imgsz'],
        batch=YOLO_TRAIN_PARAMS['batch'],
        patience=YOLO_TRAIN_PARAMS['patience'],
        project=str(RUNS_GROUPWISE_DIR.parent),
        name=RUNS_GROUPWISE_DIR.name,
        exist_ok=True,
        device=YOLO_TRAIN_PARAMS['device'],
        optimizer=YOLO_TRAIN_PARAMS['optimizer'],
        lr0=YOLO_TRAIN_PARAMS['lr0'],
        lrf=YOLO_TRAIN_PARAMS['lrf'],
        weight_decay=YOLO_TRAIN_PARAMS['weight_decay'],
        warmup_epochs=YOLO_TRAIN_PARAMS['warmup_epochs'],
        hsv_h=YOLO_TRAIN_PARAMS['hsv_h'],
        hsv_s=YOLO_TRAIN_PARAMS['hsv_s'],
        hsv_v=YOLO_TRAIN_PARAMS['hsv_v'],
        degrees=YOLO_TRAIN_PARAMS['degrees'],
        translate=YOLO_TRAIN_PARAMS['translate'],
        scale=YOLO_TRAIN_PARAMS['scale'],
        fliplr=YOLO_TRAIN_PARAMS['fliplr'],
        mosaic=YOLO_TRAIN_PARAMS['mosaic'],
        mixup=YOLO_TRAIN_PARAMS['mixup'],
        close_mosaic=YOLO_TRAIN_PARAMS['close_mosaic'],
        plots=YOLO_TRAIN_PARAMS['plots'],
        save=YOLO_TRAIN_PARAMS['save'],
        val=YOLO_TRAIN_PARAMS['val']
    )

    elapsed_mins = (time.time() - t0) / 60.0
    best_weights = RUNS_GROUPWISE_DIR / 'weights' / 'best.pt'

    print("\n" + "=" * 65)
    print("  PELATIHAN YOLOv13n GROUP-WISE SELESAI!")
    print(f"  Total Waktu Training : {elapsed_mins:.2f} menit")
    print(f"  Best Weights Path    : {best_weights}")
    print("=" * 65 + "\n")

    return results, best_weights


if __name__ == '__main__':
    train_yolov13_groupwise()
