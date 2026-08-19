"""
evaluate_knn_gt.py — Eksperimen A: HOG-KNN Ground-Truth Crop (Group-Wise Test Set)
==================================================================================
Evaluasi performa klasifikasi HOG-KNN pada test set group-wise (166 citra)
menggunakan crop wajah ground truth (baseline klasifikasi murni).
Output:
  - outputs_groupwise/hog_knn_gt_metrics.json
  - outputs_groupwise/hog_knn_gt_predictions.csv
  - outputs_groupwise/hog_knn_gt_confusion_matrix.png
"""

import sys
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from experiments_groupwise.config import (
    KNN_METRIC, OUTPUT_GROUPWISE_DIR, CLASS_LIST
)
from experiments_groupwise.hog_features import (
    load_dataset_split, calculate_metrics, plot_confusion_matrix
)
from experiments_groupwise.tune_knn import run_knn_tuning_groupwise


def evaluate_knn_gt_groupwise(best_k=None, X_train=None, y_train=None):
    print("=" * 65)
    print("  TAHAP 10: EKSPERIMEN A: HOG-KNN GT CROP (GROUP-WISE TEST SET)")
    print("=" * 65)

    OUTPUT_GROUPWISE_DIR.mkdir(parents=True, exist_ok=True)

    if best_k is None or X_train is None or y_train is None:
        best_k, _, X_train, y_train = run_knn_tuning_groupwise()

    print(f"\n[1/3] Melatih model KNN (K={best_k}, metric={KNN_METRIC}) pada {len(X_train)} data latih group-wise...")
    knn = KNeighborsClassifier(n_neighbors=best_k, metric=KNN_METRIC)
    knn.fit(X_train, y_train)

    print("\n[2/3] Memuat data uji (Test Set Group-Wise: 166 citra)...")
    X_test, y_test, test_files, test_skipped = load_dataset_split('test')
    print(f"      Test set loaded: {len(X_test)} samples ({len(test_skipped)} skipped)")

    if len(X_test) == 0:
        raise RuntimeError("Test dataset group-wise kosong!")

    print("\n[3/3] Menjalankan inferensi dan evaluasi...")
    y_pred = knn.predict(X_test)

    metrics = calculate_metrics(y_test, y_pred, CLASS_LIST)
    metrics['best_k'] = best_k
    metrics['metric'] = KNN_METRIC
    metrics['total_test_samples'] = len(X_test)

    print("\n  " + "-" * 40)
    print(f"  Accuracy        : {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
    print(f"  Macro Precision : {metrics['macro_precision']:.4f}")
    print(f"  Macro Recall    : {metrics['macro_recall']:.4f}")
    print(f"  Macro F1-Score  : {metrics['macro_f1']:.4f} ({metrics['macro_f1']*100:.2f}%)")
    print(f"  Weighted F1     : {metrics['weighted_f1']:.4f}")
    print("  " + "-" * 40)

    json_path = OUTPUT_GROUPWISE_DIR / 'hog_knn_gt_metrics.json'
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2)
    print(f"  [SAVED] {json_path}")

    preds_df = pd.DataFrame({
        'filename': test_files,
        'true_class': [CLASS_LIST[y] for y in y_test],
        'true_class_id': y_test.tolist(),
        'predicted_class': [CLASS_LIST[y] for y in y_pred],
        'predicted_class_id': y_pred.tolist(),
        'correct': (y_test == y_pred).tolist()
    })
    preds_csv_path = OUTPUT_GROUPWISE_DIR / 'hog_knn_gt_predictions.csv'
    preds_df.to_csv(preds_csv_path, index=False)
    print(f"  [SAVED] {preds_csv_path}")

    cm_path = OUTPUT_GROUPWISE_DIR / 'hog_knn_gt_confusion_matrix.png'
    plot_confusion_matrix(
        metrics['confusion_matrix'],
        CLASS_LIST,
        f'HOG-KNN GT Crop (Group-Wise Test, K={best_k})',
        cm_path,
        accuracy=metrics['accuracy']
    )
    print(f"  [SAVED] {cm_path}")

    return metrics, knn, preds_df


if __name__ == '__main__':
    evaluate_knn_gt_groupwise()
