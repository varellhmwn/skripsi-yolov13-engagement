"""
error_analysis.py — Analisis Kesalahan & Visual Error Inspection
================================================================
Menganalisis pola kesalahan prediksi dari ketiga pendekatan dan
menghasilkan visualisasi anotasi per-citra pada folder outputs/error_samples/.
"""

import json
from pathlib import Path
import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO

from experiments.config import (
    MODEL_WEIGHTS_PATH, DATASET_DIR, OUTPUT_DIR, YOLO_IMGSZ,
    YOLO_CONF_THRESHOLD, BENCHMARK_DEVICE, CLASS_LIST, CLASS_NAMES
)
from experiments.hog_features import parse_yolo_annotation


def run_error_analysis():
    print("=" * 60)
    print("  ANALISIS KESALAHAN & VISUAL ERROR INSPECTION")
    print("=" * 60)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    error_samples_dir = OUTPUT_DIR / 'error_samples'
    error_samples_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load predictions from all 3 evaluations
    yolo_preds_path = OUTPUT_DIR / 'yolo_classification_predictions.csv'
    knn_gt_preds_path = OUTPUT_DIR / 'hog_knn_gt_predictions.csv'
    hybrid_preds_path = OUTPUT_DIR / 'yolo_hog_knn_predictions.csv'

    if not (yolo_preds_path.exists() and knn_gt_preds_path.exists() and hybrid_preds_path.exists()):
        raise FileNotFoundError("Salah satu file prediksi evaluasi belum tersedia. Jalankan evaluasi terlebih dahulu.")

    df_yolo = pd.read_csv(yolo_preds_path)
    df_knn = pd.read_csv(knn_gt_preds_path)
    df_hyb = pd.read_csv(hybrid_preds_path)

    # Standardize and merge
    merged = df_yolo[['filename', 'true_class', 'predicted_class', 'correct', 'confidence', 'yolo_bbox']].copy()
    merged.columns = ['filename', 'true_class', 'yolo_prediction', 'yolo_correct', 'yolo_conf', 'yolo_bbox']

    knn_sub = df_knn[['filename', 'predicted_class', 'correct']].copy()
    knn_sub.columns = ['filename', 'knn_gt_prediction', 'knn_gt_correct']

    hyb_sub = df_hyb[['filename', 'predicted_class', 'correct']].copy()
    hyb_sub.columns = ['filename', 'hybrid_prediction', 'hybrid_correct']

    df_error = merged.merge(knn_sub, on='filename', how='inner')
    df_error = df_error.merge(hyb_sub, on='filename', how='inner')

    # Categorize detailed error types
    def get_category(row):
        y_ok = row['yolo_correct']
        k_ok = row['knn_gt_correct']
        h_ok = row['hybrid_correct']

        if y_ok and k_ok and h_ok:
            return 'all_correct'
        elif not y_ok and not k_ok and not h_ok:
            return 'all_wrong'
        elif not y_ok and k_ok and h_ok:
            return 'yolo_only_wrong'
        elif y_ok and not k_ok and h_ok:
            return 'knn_only_wrong'
        elif y_ok and k_ok and not h_ok:
            return 'hybrid_only_wrong'
        elif not y_ok and not k_ok and h_ok:
            return 'yolo_and_knn_wrong'
        elif not y_ok and k_ok and not h_ok:
            return 'yolo_and_hybrid_wrong'
        elif y_ok and not k_ok and not h_ok:
            return 'knn_and_hybrid_wrong'
        return 'other'

    df_error['comparison_category'] = df_error.apply(get_category, axis=1)

    csv_path = OUTPUT_DIR / 'error_analysis.csv'
    df_error.to_csv(csv_path, index=False)
    print(f"  [SAVED] {csv_path}")

    # Summary counts
    cat_counts = df_error['comparison_category'].value_counts()
    print("\n  Distribusi Kategori Prediksi:")
    for cat, count in cat_counts.items():
        pct = count / len(df_error) * 100
        print(f"    - {cat:<24}: {count:>3} citra ({pct:>5.1f}%)")

    # ─── VISUAL ERROR INSPECTION ─────────────────────────────
    # Filter all misclassified images (where at least one method is wrong)
    misclassified = df_error[df_error['comparison_category'] != 'all_correct']
    print(f"\n  Menyimpan visualisasi untuk {len(misclassified)} citra dengan kesalahan...")

    test_images_dir = DATASET_DIR / 'images' / 'test'
    test_labels_dir = DATASET_DIR / 'labels' / 'test'

    for _, row in misclassified.iterrows():
        fname = row['filename']
        img_path = test_images_dir / fname
        lbl_path = test_labels_dir / f"{Path(fname).stem}.txt"

        img = cv2.imread(str(img_path))
        if img is None:
            continue

        h, w = img.shape[:2]
        vis = img.copy()

        # Draw Ground Truth BBox (Green)
        gt_anns = parse_yolo_annotation(lbl_path, w, h)
        if gt_anns:
            _, gx1, gy1, gx2, gy2 = gt_anns[0]
            cv2.rectangle(vis, (gx1, gy1), (gx2, gy2), (0, 255, 0), 2)
            cv2.putText(vis, f"GT: {row['true_class']}", (gx1, max(20, gy1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # Draw YOLO BBox (Blue or Red)
        if pd.notna(row['yolo_bbox']) and str(row['yolo_bbox']).strip():
            try:
                bx1, by1, bx2, by2 = map(float, str(row['yolo_bbox']).split(','))
                bx1, by1, bx2, by2 = int(bx1), int(by1), int(bx2), int(by2)
                yolo_color = (255, 100, 0) if row['yolo_correct'] else (0, 0, 255)
                cv2.rectangle(vis, (bx1, by1), (bx2, by2), yolo_color, 2)
                cv2.putText(vis, f"YOLO: {row['yolo_prediction']} ({row['yolo_conf']:.2f})",
                            (bx1, min(h - 10, by2 + 20)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, yolo_color, 2)
            except Exception:
                pass

        # Create info panel on top
        panel_h = 90
        panel = np.zeros((panel_h, w, 3), dtype=np.uint8) + 30
        
        cv2.putText(panel, f"File: {fname}", (10, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1)
        cv2.putText(panel, f"True Class: {row['true_class'].upper()}", (10, 48),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)
        
        yolo_status = "CORRECT" if row['yolo_correct'] else "WRONG"
        knn_status = "CORRECT" if row['knn_gt_correct'] else "WRONG"
        hyb_status = "CORRECT" if row['hybrid_correct'] else "WRONG"
        
        y_text = f"YOLO: {row['yolo_prediction']} [{yolo_status}]"
        k_text = f"HOG-KNN GT: {row['knn_gt_prediction']} [{knn_status}]"
        h_text = f"Hybrid: {row['hybrid_prediction']} [{hyb_status}]"

        cv2.putText(panel, y_text, (10, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 255, 0) if row['yolo_correct'] else (0, 80, 255), 1)
        cv2.putText(panel, k_text, (w//3 + 20, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 255, 0) if row['knn_gt_correct'] else (0, 80, 255), 1)
        cv2.putText(panel, h_text, (2*w//3 + 20, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 255, 0) if row['hybrid_correct'] else (0, 80, 255), 1)

        combined = np.vstack([panel, vis])
        out_sample_path = error_samples_dir / f"error_{row['comparison_category']}_{Path(fname).stem}.jpg"
        cv2.imwrite(str(out_sample_path), combined)

    print(f"  [SAVED] {len(misclassified)} visual error samples saved to {error_samples_dir}")
    return df_error


if __name__ == '__main__':
    run_error_analysis()
