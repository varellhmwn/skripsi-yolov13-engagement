"""
evaluate_yolo_hog_knn.py — Eksperimen B: YOLO-based Face Crop + HOG-KNN
========================================================================
Pipeline hybrid:
  1. Full image -> YOLOv13n (hanya mengambil bounding box wajah terbesar)
  2. Crop area wajah hasil deteksi YOLO
  3. Preprocessing HOG (resize 64x64, grayscale, HOG extraction)
  4. KNN classification (K terbaik dari validation tuning)
Kelas hasil YOLO diabaikan pada tahap klasifikasi.
"""

import json
from pathlib import Path
import cv2
import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from ultralytics import YOLO

from experiments.config import (
    MODEL_WEIGHTS_PATH, DATASET_DIR, OUTPUT_DIR, YOLO_IMGSZ,
    YOLO_CONF_THRESHOLD, BENCHMARK_DEVICE, CLASS_LIST, VALID_IMG_EXTS,
    KNN_METRIC
)
from experiments.hog_features import (
    parse_yolo_annotation, crop_face_from_bbox, extract_hog_features,
    calculate_metrics, plot_confusion_matrix, load_dataset_split
)
from experiments.tune_knn import run_knn_tuning


def evaluate_yolo_hog_knn(best_k=None, X_train=None, y_train=None):
    print("=" * 60)
    print("  EKSPERIMEN B: YOLO-based Face Crop + HOG-KNN (Hybrid Pipeline)")
    print("=" * 60)
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if best_k is None or X_train is None or y_train is None:
        best_k, _, X_train, y_train = run_knn_tuning()

    print(f"\n[1/3] Melatih model KNN (K={best_k}) pada {len(X_train)} data latih...")
    knn = KNeighborsClassifier(n_neighbors=best_k, metric=KNN_METRIC)
    knn.fit(X_train, y_train)

    print(f"\n[2/3] Memuat model YOLO: {MODEL_WEIGHTS_PATH}...")
    yolo_model = YOLO(str(MODEL_WEIGHTS_PATH))

    test_images_dir = DATASET_DIR / 'images' / 'test'
    test_labels_dir = DATASET_DIR / 'labels' / 'test'

    img_files = sorted([
        f for f in test_images_dir.iterdir()
        if f.suffix.lower() in VALID_IMG_EXTS
    ])

    print(f"\n[3/3] Menjalankan pipeline hybrid pada {len(img_files)} citra test...")
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

        # 1. YOLO face detection
        results = yolo_model.predict(
            str(img_path),
            imgsz=YOLO_IMGSZ,
            conf=YOLO_CONF_THRESHOLD,
            device=BENCHMARK_DEVICE,
            verbose=False
        )

        det = results[0].boxes
        face_detected = len(det) > 0
        pred_class = -1
        bbox_str = ""

        if face_detected:
            # Pilih bbox terbesar
            largest_area = 0
            best_idx = 0
            for i in range(len(det)):
                xyxy = det.xyxy[i].cpu().numpy()
                area = (xyxy[2] - xyxy[0]) * (xyxy[3] - xyxy[1])
                if area > largest_area:
                    largest_area = area
                    best_idx = i

            xyxy = det.xyxy[best_idx].cpu().numpy()
            x1, y1, x2, y2 = int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])
            bbox_str = f"{x1},{y1},{x2},{y2}"

            # 2. Crop face from YOLO bbox
            crop = crop_face_from_bbox(img, x1, y1, x2, y2)
            if crop is not None:
                # 3. HOG extraction
                feat = extract_hog_features(crop)
                # 4. KNN predict
                pred = knn.predict(feat.reshape(1, -1))
                pred_class = int(pred[0])
            else:
                face_detected = False
                detection_failures += 1
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
            'yolo_bbox': bbox_str,
            'correct': is_correct
        })

    preds_df = pd.DataFrame(predictions)

    # Classification metrics
    detected_df = preds_df[preds_df['face_detected']]
    y_true = detected_df['true_class_id'].values
    y_pred = detected_df['predicted_class_id'].values

    class_metrics = calculate_metrics(y_true, y_pred, CLASS_LIST)
    e2e_accuracy = preds_df['correct'].sum() / len(preds_df)

    summary_metrics = {
        'model_name': 'YOLO-based Face Crop + HOG-KNN Hybrid',
        'weights_path': str(MODEL_WEIGHTS_PATH),
        'best_k': best_k,
        'knn_metric': KNN_METRIC,
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

    # Save outputs
    json_path = OUTPUT_DIR / 'yolo_hog_knn_metrics.json'
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(summary_metrics, f, indent=2)
    print(f"  [SAVED] {json_path}")

    csv_path = OUTPUT_DIR / 'yolo_hog_knn_predictions.csv'
    preds_df.to_csv(csv_path, index=False)
    print(f"  [SAVED] {csv_path}")

    cm_path = OUTPUT_DIR / 'yolo_hog_knn_confusion_matrix.png'
    plot_confusion_matrix(
        class_metrics['confusion_matrix'],
        CLASS_LIST,
        f'YOLO-based Face Crop + HOG-KNN (K={best_k})',
        cm_path,
        accuracy=class_metrics['accuracy']
    )
    print(f"  [SAVED] {cm_path}")

    return summary_metrics, preds_df


if __name__ == '__main__':
    evaluate_yolo_hog_knn()
