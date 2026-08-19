"""
evaluate_yolo_classification.py — Evaluasi Image-Level Classification YOLOv13n (Group-Wise)
=============================================================================================
Inferensi citra utuh YOLOv13n pada Group-Wise Test Set (166 citra).
Memilih deteksi wajah utama (bounding box terbesar) untuk menentukan kelas prediksi.
Output:
  - outputs_groupwise/yolo_classification_metrics.json
  - outputs_groupwise/yolo_classification_predictions.csv
  - outputs_groupwise/yolo_classification_confusion_matrix.png
"""

import sys
import json
from pathlib import Path
import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from experiments_groupwise.config import (
    TRAINED_GROUPWISE_WEIGHTS, GROUPWISE_DATASET_DIR, OUTPUT_GROUPWISE_DIR,
    YOLO_IMGSZ, YOLO_CONF_THRESHOLD, BENCHMARK_DEVICE, CLASS_LIST, VALID_IMG_EXTS
)
from experiments_groupwise.hog_features import (
    parse_yolo_annotation, calculate_metrics, plot_confusion_matrix
)


def evaluate_yolo_groupwise_classification():
    print("=" * 65)
    print("  TAHAP 9: EVALUASI IMAGE-LEVEL CLASSIFICATION YOLOv13n (GROUP-WISE)")
    print("=" * 65)

    OUTPUT_GROUPWISE_DIR.mkdir(parents=True, exist_ok=True)

    if not TRAINED_GROUPWISE_WEIGHTS.exists():
        raise FileNotFoundError(f"Weights YOLO group-wise belum tersedia: {TRAINED_GROUPWISE_WEIGHTS}")

    test_images_dir = GROUPWISE_DATASET_DIR / 'images' / 'test'
    test_labels_dir = GROUPWISE_DATASET_DIR / 'labels' / 'test'

    img_files = sorted([
        f for f in test_images_dir.iterdir()
        if f.suffix.lower() in VALID_IMG_EXTS
    ])

    print(f"\n[1/2] Loading YOLO model: {TRAINED_GROUPWISE_WEIGHTS}...")
    model = YOLO(str(TRAINED_GROUPWISE_WEIGHTS))

    print(f"\n[2/2] Inferensi pada {len(img_files)} citra test group-wise...")
    predictions = []
    detection_failures = 0

    for img_path in img_files:
        label_path = test_labels_dir / f"{img_path.stem}.txt"
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        h, w = img.shape[:2]
        gt_anns = parse_yolo_annotation(label_path, w, h)
        if not gt_anns:
            continue
        gt_class = gt_anns[0][0]

        results = model.predict(
            str(img_path),
            imgsz=YOLO_IMGSZ,
            conf=YOLO_CONF_THRESHOLD,
            device=BENCHMARK_DEVICE,
            verbose=False
        )

        det = results[0].boxes
        face_detected = len(det) > 0
        pred_class = -1
        pred_conf = 0.0
        pred_bbox = ""

        if face_detected:
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
        else:
            detection_failures += 1

        is_correct = (pred_class == gt_class) if face_detected else False

        predictions.append({
            'filename': img_path.name,
            'face_detected': face_detected,
            'true_class': CLASS_LIST[gt_class],
            'true_class_id': gt_class,
            'predicted_class': CLASS_LIST[pred_class] if face_detected else 'detection_failed',
            'predicted_class_id': pred_class,
            'confidence': pred_conf,
            'yolo_bbox': pred_bbox,
            'correct': is_correct
        })

    preds_df = pd.DataFrame(predictions)

    detected_df = preds_df[preds_df['face_detected']]
    y_true = detected_df['true_class_id'].values
    y_pred = detected_df['predicted_class_id'].values

    class_metrics = calculate_metrics(y_true, y_pred, CLASS_LIST)
    e2e_accuracy = preds_df['correct'].sum() / len(preds_df)

    summary_metrics = {
        'model_name': 'YOLOv13n Image-Level Classification (Group-Wise)',
        'weights_path': str(TRAINED_GROUPWISE_WEIGHTS),
        'total_images': len(preds_df),
        'detected_images': len(detected_df),
        'detection_failures': detection_failures,
        'detection_failure_rate': detection_failures / len(preds_df),
        'end_to_end_accuracy': float(e2e_accuracy),
        'classification_metrics_detected_only': class_metrics,
        'accuracy': class_metrics['accuracy'],
        'macro_precision': class_metrics['macro_precision'],
        'macro_recall': class_metrics['macro_recall'],
        'macro_f1': class_metrics['macro_f1'],
        'weighted_precision': class_metrics['weighted_precision'],
        'weighted_recall': class_metrics['weighted_recall'],
        'weighted_f1': class_metrics['weighted_f1'],
        'per_class': class_metrics['per_class'],
        'confusion_matrix': class_metrics['confusion_matrix']
    }

    print("\n  " + "-" * 40)
    print(f"  Total Images    : {len(preds_df)}")
    print(f"  Detected Faces  : {len(detected_df)} (Failures: {detection_failures})")
    print(f"  Accuracy        : {class_metrics['accuracy']:.4f} ({class_metrics['accuracy']*100:.2f}%)")
    print(f"  Macro Precision : {class_metrics['macro_precision']:.4f}")
    print(f"  Macro Recall    : {class_metrics['macro_recall']:.4f}")
    print(f"  Macro F1-Score  : {class_metrics['macro_f1']:.4f} ({class_metrics['macro_f1']*100:.2f}%)")
    print(f"  Weighted F1     : {class_metrics['weighted_f1']:.4f}")
    print("  " + "-" * 40)

    json_path = OUTPUT_GROUPWISE_DIR / 'yolo_classification_metrics.json'
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(summary_metrics, f, indent=2)
    print(f"  [SAVED] {json_path}")

    csv_path = OUTPUT_GROUPWISE_DIR / 'yolo_classification_predictions.csv'
    preds_df.to_csv(csv_path, index=False)
    print(f"  [SAVED] {csv_path}")

    cm_path = OUTPUT_GROUPWISE_DIR / 'yolo_classification_confusion_matrix.png'
    plot_confusion_matrix(
        class_metrics['confusion_matrix'],
        CLASS_LIST,
        'YOLOv13n Image-Level Classification (Group-Wise Test)',
        cm_path,
        accuracy=class_metrics['accuracy']
    )
    print(f"  [SAVED] {cm_path}")

    return summary_metrics, preds_df


if __name__ == '__main__':
    evaluate_yolo_groupwise_classification()
