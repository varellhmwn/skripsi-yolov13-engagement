"""
evaluate_knn_gt.py — Eksperimen A: HOG-KNN dengan Ground-Truth Crop
=====================================================================
Evaluasi KNN (K terbaik) pada test set menggunakan crop wajah
berdasarkan ground-truth bounding box.

Output:
  - outputs/hog_knn_gt_metrics.json
  - outputs/hog_knn_gt_predictions.csv
  - outputs/hog_knn_gt_confusion_matrix.png
"""

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from experiments.utils import (
    load_dataset_split, calculate_metrics, plot_confusion_matrix,
    save_metrics_json, TimingContext, RANDOM_SEED, CLASS_LIST
)
from experiments.knn_tuning import run_knn_tuning, METRIC

OUTPUT_DIR = Path(__file__).resolve().parent.parent / 'outputs'


def evaluate_knn_gt(best_k=None, X_train=None, y_train=None):
    """
    Evaluasi HOG-KNN dengan ground-truth crop pada test set.

    Parameters
    ----------
    best_k : int, optional
        K terbaik. Jika None, akan dijalankan tuning terlebih dahulu.
    X_train, y_train : numpy.ndarray, optional
        Data training. Jika None, akan di-load ulang.

    Returns
    -------
    dict
        Metrik evaluasi lengkap.
    """
    print("\n" + "=" * 60)
    print("  EKSPERIMEN A: HOG-KNN Ground-Truth Crop")
    print("=" * 60)

    # 1. Tuning jika belum
    if best_k is None or X_train is None:
        best_k, _, X_train, y_train = run_knn_tuning()

    print(f"\n  Menggunakan K = {best_k}")

    # 2. Train KNN final
    print("\n[1/3] Training KNN final (K={})...".format(best_k))
    with TimingContext("KNN Training") as tc_train:
        knn = KNeighborsClassifier(n_neighbors=best_k, metric=METRIC)
        knn.fit(X_train, y_train)
    print(f"      Training selesai: {tc_train.elapsed:.4f} detik")

    # 3. Load test set
    print("\n[2/3] Loading test data (ground-truth crop + HOG)...")
    X_test, y_test, test_files, test_skipped = load_dataset_split('test')
    print(f"      Test: {len(X_test)} samples loaded, "
          f"{len(test_skipped)} skipped")

    if len(X_test) == 0:
        raise RuntimeError("Test data kosong!")

    # 4. Evaluasi
    print("\n[3/3] Evaluasi pada test set...")

    # Timing: KNN prediction only
    with TimingContext("KNN Prediction") as tc_pred:
        y_pred = knn.predict(X_test)
    knn_pred_time = tc_pred.elapsed
    knn_pred_per_img = (knn_pred_time / len(X_test)) * 1000  # ms

    # Timing: Full classification pipeline per sample
    pipeline_times = []
    for i in range(len(X_test)):
        t0 = time.perf_counter()
        # Simulasi full pipeline: HOG sudah di-extract saat load,
        # jadi kita hanya ukur predict.
        # Catatan: waktu HOG extraction sudah termasuk di load time
        _ = knn.predict(X_test[i:i+1])
        pipeline_times.append(time.perf_counter() - t0)

    # Untuk full pipeline, kita juga perlu mengukur HOG extraction
    # Muat ulang 1 sample untuk mengukur end-to-end
    import cv2
    from experiments.utils import (
        DATASET_DIR, parse_yolo_annotation, crop_face_from_bbox,
        extract_hog_features
    )

    full_pipeline_times = []
    test_images_dir = DATASET_DIR / 'images' / 'test'
    test_labels_dir = DATASET_DIR / 'labels' / 'test'

    for fname in test_files:
        img_path = test_images_dir / fname
        label_path = test_labels_dir / f"{Path(fname).stem}.txt"

        t0 = time.perf_counter()
        img = cv2.imread(str(img_path))
        h, w = img.shape[:2]
        anns = parse_yolo_annotation(label_path, w, h)
        ann = anns[0] if anns else None
        if ann:
            _, x1, y1, x2, y2 = ann
            crop = crop_face_from_bbox(img, x1, y1, x2, y2)
            feat = extract_hog_features(crop)
            _ = knn.predict(feat.reshape(1, -1))
        full_pipeline_times.append(time.perf_counter() - t0)

    # 5. Hitung metrik
    metrics = calculate_metrics(y_test, y_pred, CLASS_LIST)

    # Tambahkan timing
    metrics['timing'] = {
        'knn_predict_total_sec': float(knn_pred_time),
        'knn_predict_per_img_ms': float(knn_pred_per_img),
        'full_pipeline_mean_ms': float(np.mean(full_pipeline_times) * 1000),
        'full_pipeline_median_ms': float(np.median(full_pipeline_times) * 1000),
        'full_pipeline_total_sec': float(sum(full_pipeline_times)),
        'train_time_sec': float(tc_train.elapsed),
        'num_test_samples': len(X_test)
    }
    metrics['config'] = {
        'k': best_k,
        'metric': METRIC,
        'hog_img_size': [64, 64],
        'hog_orientations': 9,
        'hog_pixels_per_cell': [8, 8],
        'hog_cells_per_block': [2, 2],
        'hog_block_norm': 'L2-Hys'
    }

    # 6. Tampilkan hasil
    print(f"\n  {'─' * 40}")
    print(f"  Accuracy        : {metrics['accuracy']:.4f}")
    print(f"  Macro Precision : {metrics['macro_precision']:.4f}")
    print(f"  Macro Recall    : {metrics['macro_recall']:.4f}")
    print(f"  Macro F1        : {metrics['macro_f1']:.4f}")
    print(f"  Weighted F1     : {metrics['weighted_f1']:.4f}")
    print(f"  {'─' * 40}")
    print(f"  KNN predict/img : {knn_pred_per_img:.2f} ms")
    print(f"  Full pipeline   : {np.mean(full_pipeline_times)*1000:.2f} ms (mean)")
    print(f"  {'─' * 40}")

    # 7. Simpan output
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Metrics JSON
    save_metrics_json(metrics, OUTPUT_DIR / 'hog_knn_gt_metrics.json')

    # Predictions CSV
    preds_df = pd.DataFrame({
        'filename': test_files,
        'true_class': [CLASS_LIST[y] for y in y_test],
        'predicted_class': [CLASS_LIST[y] for y in y_pred],
        'true_class_id': y_test.tolist(),
        'predicted_class_id': y_pred.tolist(),
        'correct': (y_test == y_pred).tolist()
    })
    csv_path = OUTPUT_DIR / 'hog_knn_gt_predictions.csv'
    preds_df.to_csv(csv_path, index=False)
    print(f"  [SAVED] {csv_path}")

    # Confusion matrix
    plot_confusion_matrix(
        metrics['confusion_matrix'],
        CLASS_LIST,
        'HOG-KNN Ground-Truth Crop',
        OUTPUT_DIR / 'hog_knn_gt_confusion_matrix.png',
        accuracy=metrics['accuracy']
    )

    return metrics, knn


if __name__ == '__main__':
    evaluate_knn_gt()
