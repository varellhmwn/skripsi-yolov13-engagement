"""
evaluate_yolo_classification.py — Evaluasi YOLOv13n pada Test Set
=================================================================
Menjalankan:
  1. Evaluasi native YOLO (mAP, Precision, Recall, dll.)
  2. Image-level classification metrics (Accuracy, Macro F1, dll.)

Image-level classification:
  - Untuk setiap test image, pilih prediksi YOLO terbesar.
  - Gunakan class YOLO sebagai predicted class.
  - Bandingkan dengan ground-truth class.

Output:
  - outputs/yolo_metrics.json
  - outputs/yolo_predictions.csv
  - outputs/yolo_confusion_matrix.png
"""

import sys
import time
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from experiments.utils import (
    DATASET_DIR, CLASS_LIST, VALID_IMG_EXTS, BASE_DIR,
    parse_yolo_annotation, calculate_metrics, plot_confusion_matrix,
    save_metrics_json, TimingContext
)

# Weights — menggunakan run terbaru (wtest_4_kelas)
WEIGHTS_PATH = BASE_DIR / 'runs' / 'yolov13_master_combined_wtest_4_kelas' / 'weights' / 'best.pt'
OUTPUT_DIR = BASE_DIR / 'outputs'


def evaluate_yolo_classification():
    """
    Evaluasi YOLOv13n: native object detection + image-level classification.

    Returns
    -------
    dict
        Metrik evaluasi lengkap (detection + classification).
    """
    print("\n" + "=" * 60)
    print("  EVALUASI YOLOv13n — Test Set")
    print("  Native Detection + Image-Level Classification")
    print("=" * 60)

    from ultralytics import YOLO

    if not WEIGHTS_PATH.exists():
        raise FileNotFoundError(f"Weights tidak ditemukan: {WEIGHTS_PATH}")

    # 1. Native YOLO evaluation (mAP metrics)
    print("\n[1/3] Running native YOLO evaluation (mAP metrics)...")
    model = YOLO(str(WEIGHTS_PATH))
    data_yaml = str(DATASET_DIR / 'data.yaml')

    val_results = model.val(
        data=data_yaml,
        split='test',
        imgsz=640,
        batch=16,
        device=0,
        plots=False,
        verbose=False
    )

    detection_metrics = {
        'precision': float(val_results.box.mp),
        'recall': float(val_results.box.mr),
        'f1_detection': float(2 * val_results.box.mp * val_results.box.mr /
                              (val_results.box.mp + val_results.box.mr + 1e-8)),
        'mAP_50': float(val_results.box.map50),
        'mAP_75': float(val_results.box.map75),
        'mAP_50_95': float(val_results.box.map),
    }

    # Per-class detection
    per_class_detection = {}
    for i, name in enumerate(CLASS_LIST):
        if i < len(val_results.box.p):
            per_class_detection[name] = {
                'precision': float(val_results.box.p[i]),
                'recall': float(val_results.box.r[i]),
                'ap50': float(val_results.box.ap50[i]),
                'ap50_95': float(val_results.box.ap[i] if hasattr(val_results.box, 'ap') else 0),
            }

    print(f"  Precision    : {detection_metrics['precision']:.4f}")
    print(f"  Recall       : {detection_metrics['recall']:.4f}")
    print(f"  mAP@0.5      : {detection_metrics['mAP_50']:.4f}")
    print(f"  mAP@0.5:0.95 : {detection_metrics['mAP_50_95']:.4f}")

    # 2. Image-level classification
    print("\n[2/3] Running image-level classification evaluation...")

    test_images_dir = DATASET_DIR / 'images' / 'test'
    test_labels_dir = DATASET_DIR / 'labels' / 'test'

    img_files = sorted([
        f for f in test_images_dir.iterdir()
        if f.suffix.lower() in VALID_IMG_EXTS
    ])

    predictions = []
    inference_times = []

    for img_path in img_files:
        label_path = test_labels_dir / f"{img_path.stem}.txt"
        if not label_path.exists():
            continue

        # Read ground truth
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        h, w = img.shape[:2]
        gt_anns = parse_yolo_annotation(label_path, w, h)
        if not gt_anns:
            continue
        gt_class = gt_anns[0][0]  # Class dari first/main annotation

        # YOLO inference
        t0 = time.perf_counter()
        results = model.predict(
            str(img_path), imgsz=640, conf=0.25,
            device=0, verbose=False
        )
        infer_time = time.perf_counter() - t0
        inference_times.append(infer_time)

        det = results[0].boxes
        pred_class = -1
        pred_conf = 0.0
        pred_bbox = ""
        face_detected = False

        if len(det) > 0:
            face_detected = True
            # Pilih deteksi terbesar (konsisten dengan pipeline realtime)
            largest_area = 0
            best_idx = 0
            for i in range(len(det)):
                xyxy = det.xyxy[i].cpu().numpy()
                area = (xyxy[2] - xyxy[0]) * (xyxy[3] - xyxy[1])
                if area > largest_area:
                    largest_area = area
                    best_idx = i

            pred_class = int(det.cls[best_idx].item())
            pred_conf = float(det.conf[best_idx].item())
            xyxy = det.xyxy[best_idx].cpu().numpy()
            pred_bbox = f"{xyxy[0]:.1f},{xyxy[1]:.1f},{xyxy[2]:.1f},{xyxy[3]:.1f}"

        predictions.append({
            'filename': img_path.name,
            'face_detected': face_detected,
            'true_class': CLASS_LIST[gt_class] if gt_class < len(CLASS_LIST) else str(gt_class),
            'true_class_id': gt_class,
            'predicted_class': CLASS_LIST[pred_class] if 0 <= pred_class < len(CLASS_LIST) else 'detection_failed',
            'predicted_class_id': pred_class,
            'confidence': pred_conf,
            'yolo_bbox': pred_bbox,
            'correct': pred_class == gt_class,
            'inference_time_ms': infer_time * 1000
        })

    # 3. Calculate classification metrics
    print("\n[3/3] Calculating classification metrics...")

    preds_df = pd.DataFrame(predictions)

    # Filter only detected faces for classification metrics
    detected = preds_df[preds_df['face_detected']]
    detection_failures = preds_df[~preds_df['face_detected']]

    y_true = detected['true_class_id'].values
    y_pred = detected['predicted_class_id'].values

    classification_metrics = calculate_metrics(y_true, y_pred, CLASS_LIST)

    # Timing
    timing = {
        'mean_inference_ms': float(np.mean(inference_times) * 1000),
        'median_inference_ms': float(np.median(inference_times) * 1000),
        'total_inference_sec': float(sum(inference_times)),
        'estimated_fps': float(1.0 / np.mean(inference_times)) if np.mean(inference_times) > 0 else 0,
        'num_images': len(predictions),
        'num_detected': int(len(detected)),
        'num_detection_failed': int(len(detection_failures))
    }

    # Combine all metrics
    all_metrics = {
        'model': 'YOLOv13n',
        'weights': str(WEIGHTS_PATH),
        'detection': detection_metrics,
        'per_class_detection': per_class_detection,
        'classification': classification_metrics,
        'timing': timing,
    }

    # Print summary
    print(f"\n  {'─' * 40}")
    print(f"  [IMAGE-LEVEL CLASSIFICATION]")
    print(f"  Images evaluated : {len(predictions)}")
    print(f"  Detection success: {len(detected)}")
    print(f"  Detection failed : {len(detection_failures)}")
    print(f"  Accuracy         : {classification_metrics['accuracy']:.4f}")
    print(f"  Macro Precision  : {classification_metrics['macro_precision']:.4f}")
    print(f"  Macro Recall     : {classification_metrics['macro_recall']:.4f}")
    print(f"  Macro F1         : {classification_metrics['macro_f1']:.4f}")
    print(f"  Weighted F1      : {classification_metrics['weighted_f1']:.4f}")
    print(f"  {'─' * 40}")
    print(f"  Mean inference   : {timing['mean_inference_ms']:.2f} ms")
    print(f"  Estimated FPS    : {timing['estimated_fps']:.1f}")
    print(f"  {'─' * 40}")

    # Save outputs
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    save_metrics_json(all_metrics, OUTPUT_DIR / 'yolo_metrics.json')

    csv_path = OUTPUT_DIR / 'yolo_predictions.csv'
    preds_df.to_csv(csv_path, index=False)
    print(f"  [SAVED] {csv_path}")

    plot_confusion_matrix(
        classification_metrics['confusion_matrix'],
        CLASS_LIST,
        'YOLOv13n Image-Level Classification',
        OUTPUT_DIR / 'yolo_confusion_matrix.png',
        accuracy=classification_metrics['accuracy']
    )

    return all_metrics


if __name__ == '__main__':
    evaluate_yolo_classification()
