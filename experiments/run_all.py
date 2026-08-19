"""
run_all.py — Master Runner: Seluruh Eksperimen Perbandingan
=============================================================
Menjalankan seluruh pipeline secara berurutan:
  1. Validasi data leakage
  2. KNN hyperparameter tuning (validation set)
  3. Evaluasi HOG-KNN Ground-Truth Crop (test set)
  4. Evaluasi YOLOv13n image-level classification (test set)
  5. Evaluasi YOLO-based Face Crop + HOG-KNN (test set)
  6. Perbandingan model & pembuatan laporan

Penggunaan:
    python experiments/run_all.py

Untuk menjalankan per-step secara individual:
    python -c "from experiments.knn_tuning import run_knn_tuning; run_knn_tuning()"
    python -c "from experiments.evaluate_knn_gt import evaluate_knn_gt; evaluate_knn_gt()"
    python -c "from experiments.evaluate_yolo_classification import evaluate_yolo_classification; evaluate_yolo_classification()"
    python -c "from experiments.evaluate_yolo_hog_knn import evaluate_yolo_hog_knn; evaluate_yolo_hog_knn()"
    python -c "from experiments.compare_models import run_comparison; run_comparison()"
"""

import sys
import time
from pathlib import Path

# Ensure project root is in path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def main():
    total_start = time.time()

    print("╔" + "═" * 58 + "╗")
    print("║  EKSPERIMEN PERBANDINGAN YOLOv13n vs HOG-KNN             ║")
    print("║  Facial Emotion Recognition — 4 Class                    ║")
    print("║  engaged | confused | bored | frustrated                 ║")
    print("╚" + "═" * 58 + "╝")
    print()

    # ─── Step 0: Data Leakage Validation ──────────────────────
    print("━" * 60)
    print("  STEP 0: Validasi Data Leakage")
    print("━" * 60)

    from experiments.utils import validate_no_data_leakage, get_class_distribution

    leakage_result = validate_no_data_leakage()
    for detail in leakage_result['details']:
        print(f"  {detail}")

    if not leakage_result['passed']:
        print("\n  [FATAL] Data leakage terdeteksi! Eksperimen dibatalkan.")
        sys.exit(1)
    else:
        print("\n  ✓ Data leakage check PASSED")

    # Print class distribution
    print("\n  Class Distribution:")
    for split in ['train', 'val', 'test']:
        dist = get_class_distribution(split)
        total = sum(dist.values())
        print(f"    {split}: {total} total — "
              f"engaged={dist[0]}, confused={dist[1]}, "
              f"bored={dist[2]}, frustrated={dist[3]}")

    # ─── Step 1: KNN Hyperparameter Tuning ────────────────────
    print("\n" + "━" * 60)
    print("  STEP 1: KNN Hyperparameter Tuning")
    print("━" * 60)

    from experiments.knn_tuning import run_knn_tuning
    best_k, tuning_df, X_train, y_train = run_knn_tuning()

    # ─── Step 2: HOG-KNN Ground-Truth Crop ────────────────────
    print("\n" + "━" * 60)
    print("  STEP 2: HOG-KNN Ground-Truth Crop Evaluation")
    print("━" * 60)

    from experiments.evaluate_knn_gt import evaluate_knn_gt
    knn_gt_metrics, trained_knn = evaluate_knn_gt(
        best_k=best_k, X_train=X_train, y_train=y_train
    )

    # ─── Step 3: YOLO Image-Level Classification ──────────────
    print("\n" + "━" * 60)
    print("  STEP 3: YOLOv13n Image-Level Classification")
    print("━" * 60)

    from experiments.evaluate_yolo_classification import evaluate_yolo_classification
    yolo_metrics = evaluate_yolo_classification()

    # ─── Step 4: YOLO Crop + HOG-KNN ─────────────────────────
    print("\n" + "━" * 60)
    print("  STEP 4: YOLO-based Face Crop + HOG-KNN")
    print("━" * 60)

    from experiments.evaluate_yolo_hog_knn import evaluate_yolo_hog_knn
    hybrid_metrics = evaluate_yolo_hog_knn(
        best_k=best_k, X_train=X_train, y_train=y_train
    )

    # ─── Step 5: Comparison & Report ──────────────────────────
    print("\n" + "━" * 60)
    print("  STEP 5: Model Comparison & Report")
    print("━" * 60)

    from experiments.compare_models import run_comparison
    run_comparison()

    # ─── Summary ──────────────────────────────────────────────
    total_time = time.time() - total_start
    print("\n" + "╔" + "═" * 58 + "╗")
    print("║  SELURUH EKSPERIMEN SELESAI!                             ║")
    print("╚" + "═" * 58 + "╝")
    print(f"\n  Total waktu: {total_time:.1f} detik ({total_time/60:.1f} menit)")
    print()
    print("  Output files:")
    print("  ─────────────")

    outputs_dir = PROJECT_ROOT / 'outputs'
    expected_files = [
        'knn_validation_results.csv',
        'knn_tuning_k_plot.png',
        'hog_knn_gt_metrics.json',
        'hog_knn_gt_predictions.csv',
        'hog_knn_gt_confusion_matrix.png',
        'yolo_metrics.json',
        'yolo_predictions.csv',
        'yolo_confusion_matrix.png',
        'yolo_hog_knn_metrics.json',
        'yolo_hog_knn_predictions.csv',
        'yolo_hog_knn_confusion_matrix.png',
        'model_comparison.csv',
        'error_analysis.csv',
        'comparison_f1_chart.png',
        'comparison_time_chart.png',
        'experiment_report.md',
    ]

    for fname in expected_files:
        path = outputs_dir / fname
        status = "✓" if path.exists() else "✗"
        print(f"    {status} outputs/{fname}")

    print()


if __name__ == '__main__':
    main()
