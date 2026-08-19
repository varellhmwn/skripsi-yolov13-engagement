"""
run_all_experiments.py — Master Pipeline Runner Reproducibility
================================================================
Menjalankan seluruh rangkaian eksperimen komparatif secara otomatis & reproducible:
  1. Dataset Audit (Image Count vs Instance Count, Orphan Detection)
  2. Strict Leakage Audit (Filename, SHA-256, Perceptual Hash, Subject Identity)
  3. KNN Hyperparameter Tuning pada Validation Set (168 citra)
  4. HOG-KNN Ground-Truth Crop Evaluation (173 citra)
  5. YOLOv13n Native Object Detection Evaluation (173 citra)
  6. YOLOv13n Image-Level Classification Evaluation (173 citra)
  7. YOLO-based Face Crop + HOG-KNN Hybrid Evaluation (173 citra)
  8. Standardized High-Precision Runtime Benchmark (Single-image, GPU warmup)
  9. Error Analysis & Visual Sample Annotations (outputs/error_samples/)
  10. Model Comparison Table, Charts & Final Markdown Report Generation
"""

import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from experiments.dataset_audit import audit_dataset
from experiments.leakage_audit import run_leakage_audit
from experiments.tune_knn import run_knn_tuning
from experiments.evaluate_knn_gt import evaluate_knn_gt
from experiments.evaluate_yolo import evaluate_yolo_detection
from experiments.evaluate_yolo_classification import evaluate_yolo_classification
from experiments.evaluate_yolo_hog_knn import evaluate_yolo_hog_knn
from experiments.benchmark_runtime import run_runtime_benchmark
from experiments.error_analysis import run_error_analysis
from experiments.generate_report import run_all_reporting


def main():
    t_global_start = time.time()

    print("=" * 70)
    print("  MASTER RUNNER: AUDIT, EKSPERIMEN & FINALISASI YOLOv13n vs HOG-KNN")
    print("  Student Engagement Emotion Recognition (4 Classes)")
    print("=" * 70)

    # ─── TAHAP 1: DATASET AUDIT ──────────────────────────────
    print("\n>>> [1/10] Menjalankan Dataset Audit...")
    audit_dataset()

    # ─── TAHAP 2: LEAKAGE AUDIT ──────────────────────────────
    print("\n>>> [2/10] Menjalankan Strict Leakage Audit...")
    run_leakage_audit()

    # ─── TAHAP 3: KNN TUNING ─────────────────────────────────
    print("\n>>> [3/10] Menjalankan KNN Hyperparameter Tuning...")
    best_k, tuning_df, X_train, y_train = run_knn_tuning()

    # ─── TAHAP 4: HOG-KNN GT EVALUATION ──────────────────────
    print("\n>>> [4/10] Menjalankan Evaluasi HOG-KNN GT Crop...")
    evaluate_knn_gt(best_k=best_k, X_train=X_train, y_train=y_train)

    # ─── TAHAP 5: YOLO DETECTION EVALUATION ──────────────────
    print("\n>>> [5/10] Menjalankan Evaluasi Native YOLO Detection...")
    evaluate_yolo_detection()

    # ─── TAHAP 6: YOLO CLASSIFICATION EVALUATION ─────────────
    print("\n>>> [6/10] Menjalankan Evaluasi YOLO Image-Level Classification...")
    evaluate_yolo_classification()

    # ─── TAHAP 7: YOLO-HOG-KNN HYBRID EVALUATION ─────────────
    print("\n>>> [7/10] Menjalankan Evaluasi YOLO Crop + HOG-KNN Hybrid...")
    evaluate_yolo_hog_knn(best_k=best_k, X_train=X_train, y_train=y_train)

    # ─── TAHAP 8: RUNTIME BENCHMARK ──────────────────────────
    print("\n>>> [8/10] Menjalankan Standardized Runtime Benchmark...")
    run_runtime_benchmark(best_k=best_k, X_train=X_train, y_train=y_train)

    # ─── TAHAP 9: ERROR ANALYSIS & VISUAL INSPECTION ─────────
    print("\n>>> [9/10] Menjalankan Error Analysis & Visual Sample Generation...")
    run_error_analysis()

    # ─── TAHAP 10: REPORT & CHART GENERATION ─────────────────
    print("\n>>> [10/10] Menyusun Laporan Komprehensif & Grafik Publikasi...")
    run_all_reporting()

    t_global_elapsed = time.time() - t_global_start

    print("\n" + "=" * 70)
    print(f"  SEMUA TAHAP SELESAI DALAM {t_global_elapsed:.2f} DETIK ({t_global_elapsed/60:.2f} MENIT)!")
    print("=" * 70)


if __name__ == '__main__':
    main()
