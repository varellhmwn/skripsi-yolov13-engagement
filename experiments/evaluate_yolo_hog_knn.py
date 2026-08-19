"""
evaluate_yolo_hog_knn.py — Eksperimen B: YOLO Crop + HOG-KNN
==============================================================
Pipeline hybrid:
  1. YOLO mendeteksi wajah (bounding box saja)
  2. Crop area wajah dari bounding box YOLO
  3. HOG feature extraction
  4. KNN classification

YOLO predicted class DIABAIKAN — hanya bounding box yang digunakan.

Output:
  - outputs/yolo_hog_knn_metrics.json
  - outputs/yolo_hog_knn_predictions.csv
  - outputs/yolo_hog_knn_confusion_matrix.png
"""

import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from experiments.utils import (
    DATASET_DIR, CLASS_LIST, VALID_IMG_EXTS, BASE_DIR,
    parse_yolo_annotation, crop_face_from_bbox, extract_hog_features,
    load_dataset_split, calculate_metrics, plot_confusion_matrix,
    save_metrics_json, TimingContext
)
from experiments.knn_tuning import run_knn_tuning, METRIC

WEIGHTS_PATH = BASE_DIR / 'runs' / 'yolov13_master_combined_wtest_4_kelas' / 'weights' / 'best.pt'
OUTPUT_DIR = BASE_DIR / 'outputs'


def evaluate_yolo_hog_knn(best_k=None, X_train=None, y_train=None):
    """
    Evaluasi pipeline hybrid YOLO crop + HOG-KNN pada test set.

    Parameters
    ----------
    best_k : int, optional
        K terbaik dari tuning.
    X_train, y_train : numpy.ndarray, optional
        Data training HOG features.

    Returns
    -------
    dict
        Metrik evaluasi lengkap.
    """
    print("\n" + "=" * 60)
    print("  EKSPERIMEN B: YOLO-based Face Crop + HOG-KNN")
    print("  (Hybrid Pipeline)")
    print("=" * 60)

    from ultralytics import YOLO

    if not WEIGHTS_PATH.exists():
        raise FileNotFoundError(f"Weights tidak ditemukan: {WEIGHTS_PATH}")

    # 1. Persiapan
    if best_k is None or X_train is None:
        best_k, _, X_train, y_train = run_knn_tuning()

    print(f"\n  Menggunakan K = {best_k}")
    print(f"  YOLO Weights: {WEIGHTS_PATH}")

    # 2. Train KNN
    print("\n[1/3] Training KNN (K={})...".format(best_k))
    knn = KNeighborsClassifier(n_neighbors=best_k, metric=METRIC)
    knn.fit(X_train, y_train)
    print(f"      Training selesai: {len(X_train)} samples")

    # 3. Load YOLO model
    print("\n[2/3] Loading YOLO model...")
    yolo_model = YOLO(str(WEIGHTS_PATH))

    # 4. Run hybrid pipeline on test set
    print("\n[3/3] Running hybrid pipeline on test set...")

    test_images_dir = DATASET_DIR / 'images' / 'test'
    test_labels_dir = DATASET_DIR / 'labels' / 'test'

    img_files = sorted([
        f for f in test_images_dir.iterdir()
        if f.suffix.lower() in VALID_IMG_EXTS
    ])

    predictions = []
    detection_failed_count = 0

    # Timing breakdown
    yolo_times = []
    crop_preprocess_times = []
    knn_times = []
    full_pipeline_times = []

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
        gt_class = gt_anns[0][0]

        # === Full pipeline timing ===
        t_full_start = time.perf_counter()

        # Step 1: YOLO detection
        t0 = time.perf_counter()
        results = yolo_model.predict(
            str(img_path), imgsz=640, conf=0.25,
            device=0, verbose=False
        )
        t_yolo = time.perf_counter() - t0
        yolo_times.append(t_yolo)

        det = results[0].boxes
        pred_class = -1
        face_detected = False
        bbox_str = ""

        if len(det) > 0:
            face_detected = True

            # Pilih deteksi terbesar
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

            # Step 2: Crop + preprocess + HOG
            t0 = time.perf_counter()
            crop = crop_face_from_bbox(img, x1, y1, x2, y2)
            if crop is not None:
                feat = extract_hog_features(crop)
                t_preprocess = time.perf_counter() - t0
                crop_preprocess_times.append(t_preprocess)

                # Step 3: KNN prediction
                t0 = time.perf_counter()
                pred = knn.predict(feat.reshape(1, -1))
                t_knn = time.perf_counter() - t0
                knn_times.append(t_knn)

                pred_class = int(pred[0])
            else:
                face_detected = False
                detection_failed_count += 1
        else:
            detection_failed_count += 1

        t_full = time.perf_counter() - t_full_start
        full_pipeline_times.append(t_full)

        predictions.append({
            'filename': img_path.name,
            'face_detected': face_detected,
            'true_class': CLASS_LIST[gt_class] if gt_class < len(CLASS_LIST) else str(gt_class),
            'true_class_id': gt_class,
            'predicted_class': CLASS_LIST[pred_class] if 0 <= pred_class < len(CLASS_LIST) else 'detection_failed',
            'predicted_class_id': pred_class,
            'correct': pred_class == gt_class if face_detected else False,
            'yolo_bbox': bbox_str
        })

    # 5. Calculate metrics
    preds_df = pd.DataFrame(predictions)
    detected = preds_df[preds_df['face_detected']]
    failed = preds_df[~preds_df['face_detected']]

    print(f"\n  Detection results:")
    print(f"    Detected: {len(detected)}/{len(predictions)}")
    print(f"    Failed:   {len(failed)}/{len(predictions)}")

    # Classification metrics (hanya untuk yang berhasil dideteksi)
    if len(detected) > 0:
        y_true_detected = detected['true_class_id'].values
        y_pred_detected = detected['predicted_class_id'].values
        classification_metrics = calculate_metrics(y_true_detected, y_pred_detected, CLASS_LIST)
    else:
        classification_metrics = {
            'accuracy': 0.0, 'macro_f1': 0.0, 'note': 'No faces detected'
        }

    # End-to-end metrics (detection_failed = salah)
    y_true_all = preds_df['true_class_id'].values
    y_pred_all = preds_df['predicted_class_id'].values
    # Untuk yang gagal deteksi, assign class -1 yang akan salah
    end_to_end_correct = preds_df['correct'].sum()
    end_to_end_accuracy = end_to_end_correct / len(preds_df) if len(preds_df) > 0 else 0

    # Timing
    timing = {
        'yolo_detection_mean_ms': float(np.mean(yolo_times) * 1000) if yolo_times else 0,
        'yolo_detection_median_ms': float(np.median(yolo_times) * 1000) if yolo_times else 0,
        'crop_hog_preprocess_mean_ms': float(np.mean(crop_preprocess_times) * 1000) if crop_preprocess_times else 0,
        'knn_predict_mean_ms': float(np.mean(knn_times) * 1000) if knn_times else 0,
        'full_pipeline_mean_ms': float(np.mean(full_pipeline_times) * 1000) if full_pipeline_times else 0,
        'full_pipeline_median_ms': float(np.median(full_pipeline_times) * 1000) if full_pipeline_times else 0,
        'full_pipeline_total_sec': float(sum(full_pipeline_times)),
        'estimated_fps': float(1.0 / np.mean(full_pipeline_times)) if full_pipeline_times and np.mean(full_pipeline_times) > 0 else 0,
    }

    all_metrics = {
        'model': 'YOLO-HOG-KNN Hybrid',
        'weights': str(WEIGHTS_PATH),
        'best_k': best_k,
        'classification': classification_metrics,
        'end_to_end': {
            'total_images': len(predictions),
            'detected': int(len(detected)),
            'detection_failed': int(len(failed)),
            'end_to_end_accuracy': float(end_to_end_accuracy),
            'end_to_end_correct': int(end_to_end_correct),
        },
        'timing': timing,
    }

    # Print summary
    print(f"\n  {'─' * 45}")
    print(f"  [CLASSIFICATION - Detected Only]")
    if 'accuracy' in classification_metrics:
        print(f"  Accuracy         : {classification_metrics['accuracy']:.4f}")
        print(f"  Macro Precision  : {classification_metrics.get('macro_precision', 0):.4f}")
        print(f"  Macro Recall     : {classification_metrics.get('macro_recall', 0):.4f}")
        print(f"  Macro F1         : {classification_metrics.get('macro_f1', 0):.4f}")
        print(f"  Weighted F1      : {classification_metrics.get('weighted_f1', 0):.4f}")
    print(f"  {'─' * 45}")
    print(f"  [END-TO-END]")
    print(f"  E2E Accuracy     : {end_to_end_accuracy:.4f} ({end_to_end_correct}/{len(predictions)})")
    print(f"  Detection failures: {len(failed)}")
    print(f"  {'─' * 45}")
    print(f"  [TIMING]")
    print(f"  YOLO detection   : {timing['yolo_detection_mean_ms']:.2f} ms")
    print(f"  Crop+HOG preproc : {timing['crop_hog_preprocess_mean_ms']:.2f} ms")
    print(f"  KNN predict      : {timing['knn_predict_mean_ms']:.2f} ms")
    print(f"  Full pipeline    : {timing['full_pipeline_mean_ms']:.2f} ms (mean)")
    print(f"  Estimated FPS    : {timing['estimated_fps']:.1f}")
    print(f"  {'─' * 45}")

    # Save outputs
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    save_metrics_json(all_metrics, OUTPUT_DIR / 'yolo_hog_knn_metrics.json')

    csv_path = OUTPUT_DIR / 'yolo_hog_knn_predictions.csv'
    preds_df.to_csv(csv_path, index=False)
    print(f"  [SAVED] {csv_path}")

    # Confusion matrix (detected only)
    if 'confusion_matrix' in classification_metrics:
        plot_confusion_matrix(
            classification_metrics['confusion_matrix'],
            CLASS_LIST,
            'YOLO-based Face Crop + HOG-KNN',
            OUTPUT_DIR / 'yolo_hog_knn_confusion_matrix.png',
            accuracy=classification_metrics.get('accuracy')
        )

    return all_metrics


if __name__ == '__main__':
    evaluate_yolo_hog_knn()
