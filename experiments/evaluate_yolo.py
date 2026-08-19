"""
evaluate_yolo.py — Evaluasi Native Object Detection YOLOv13n
=============================================================
Menjalankan evaluasi native object detection YOLOv13n pada test set.
Menghitung: Precision, Recall, mAP@0.5, mAP@0.75, mAP@0.5:0.95, per-class AP.
Output:
  - outputs/yolo_detection_metrics.json
"""

import json
from pathlib import Path
from ultralytics import YOLO

from experiments.config import (
    MODEL_WEIGHTS_PATH, DATA_YAML, OUTPUT_DIR, YOLO_IMGSZ,
    BENCHMARK_DEVICE, CLASS_LIST
)


def evaluate_yolo_detection():
    print("=" * 60)
    print("  EVALUASI NATIVE OBJECT DETECTION YOLOv13n (TEST SET)")
    print("=" * 60)
    print(f"  Model Weights : {MODEL_WEIGHTS_PATH}")
    print(f"  Dataset YAML  : {DATA_YAML}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    model = YOLO(str(MODEL_WEIGHTS_PATH))

    # Run native evaluation
    val_results = model.val(
        data=str(DATA_YAML),
        split='test',
        imgsz=YOLO_IMGSZ,
        batch=16,
        device=BENCHMARK_DEVICE,
        workers=0,
        plots=False,
        verbose=False
    )

    mp = float(val_results.box.mp)
    mr = float(val_results.box.mr)
    map50 = float(val_results.box.map50)
    map75 = float(val_results.box.map75)
    map50_95 = float(val_results.box.map)
    f1_det = float(2 * mp * mr / (mp + mr + 1e-8))

    per_class_detection = {}
    for i, name in enumerate(CLASS_LIST):
        if i < len(val_results.box.p):
            per_class_detection[name] = {
                'precision': float(val_results.box.p[i]),
                'recall': float(val_results.box.r[i]),
                'ap50': float(val_results.box.ap50[i]),
            }

    # Speed metrics
    speed = val_results.speed if hasattr(val_results, 'speed') else {}

    detection_metrics = {
        'model_name': 'YOLOv13n',
        'weights_path': str(MODEL_WEIGHTS_PATH),
        'precision': mp,
        'recall': mr,
        'f1_detection': f1_det,
        'mAP_50': map50,
        'mAP_75': map75,
        'mAP_50_95': map50_95,
        'per_class': per_class_detection,
        'native_speed_ms': {
            'preprocess': float(speed.get('preprocess', 0.0)),
            'inference': float(speed.get('inference', 0.0)),
            'loss': float(speed.get('loss', 0.0)),
            'postprocess': float(speed.get('postprocess', 0.0))
        }
    }

    print("\n  " + "-" * 40)
    print(f"  Precision    : {mp:.4f} ({mp*100:.2f}%)")
    print(f"  Recall       : {mr:.4f} ({mr*100:.2f}%)")
    print(f"  mAP@0.5      : {map50:.4f} ({map50*100:.2f}%)")
    print(f"  mAP@0.75     : {map75:.4f} ({map75*100:.2f}%)")
    print(f"  mAP@0.5:0.95 : {map50_95:.4f} ({map50_95*100:.2f}%)")
    print("  " + "-" * 40)

    for cname, pstats in per_class_detection.items():
        print(f"  {cname:<12}: P={pstats['precision']:.4f}, R={pstats['recall']:.4f}, AP50={pstats['ap50']:.4f}")

    json_path = OUTPUT_DIR / 'yolo_detection_metrics.json'
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(detection_metrics, f, indent=2)
    print(f"\n  [SAVED] {json_path}")

    return detection_metrics


if __name__ == '__main__':
    evaluate_yolo_detection()
