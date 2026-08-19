"""
run_all.py — Master Pipeline Runner: Group-Wise Retraining & Evaluasi Komparatif
================================================================================
Menjalankan seluruh 13 tahap workflow eksperimen group-wise secara otomatis & reproducible.
Mendukung flag --skip-training jika model sudah dilatih sebelumnya.
"""

import sys
import time
import argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from experiments_groupwise.dataset_audit import audit_original_dataset
from experiments_groupwise.group_discovery import discover_groups
from experiments_groupwise.split_proposal import generate_split_proposal
from experiments_groupwise.leakage_gate import validate_leakage_gate
from experiments_groupwise.materialize_dataset import materialize_groupwise_dataset
from experiments_groupwise.train_yolo import train_yolov13_groupwise
from experiments_groupwise.tune_knn import run_knn_tuning_groupwise
from experiments_groupwise.evaluate_yolo import evaluate_yolo_groupwise_detection
from experiments_groupwise.evaluate_yolo_classification import evaluate_yolo_groupwise_classification
from experiments_groupwise.evaluate_knn_gt import evaluate_knn_gt_groupwise
from experiments_groupwise.evaluate_yolo_hog_knn import evaluate_yolo_hog_knn_groupwise
from experiments_groupwise.benchmark_runtime import run_runtime_benchmark_groupwise
from experiments_groupwise.error_analysis import run_error_analysis_groupwise
from experiments_groupwise.generate_report import run_all_groupwise_reporting
from experiments_groupwise.config import TRAINED_GROUPWISE_WEIGHTS


def main():
    parser = argparse.ArgumentParser(description="Master Pipeline Group-Wise Retraining & Evaluation")
    parser.add_argument('--skip-training', action='store_true', help="Lewati training jika bobot group-wise sudah ada")
    args = parser.parse_args()

    t_start = time.time()

    print("=" * 70)
    print("  MASTER PIPELINE: GROUP-WISE SPLIT, RETRAINING YOLOv13n & EVALUASI")
    print("  Student Engagement Emotion Recognition (4 Classes)")
    print("=" * 70)

    # 1. Dataset Audit
    print("\n>>> [1/13] Audit Dataset 1.660 Citra...")
    audit_original_dataset()

    # 2. Group Discovery
    print("\n>>> [2/13] Identifikasi & Pembentukan Group...")
    discover_groups()

    # 3. Split Proposal
    print("\n>>> [3/13] Optimasi Split Proposal (80:10:10)...")
    generate_split_proposal()

    # 4. Leakage Validation Gate
    print("\n>>> [4/13] Validasi Leakage Gate...")
    validate_leakage_gate()

    # 5. Materialize Group-Wise Dataset
    print("\n>>> [5/13] Materialisasi Direktori Dataset Group-Wise v1...")
    materialize_groupwise_dataset()

    # 6. Retrain YOLOv13n
    print("\n>>> [6/13] Pelatihan Ulang YOLOv13n dari Scratch (yolov13n.pt)...")
    if args.skip_training and TRAINED_GROUPWISE_WEIGHTS.exists():
        print(f"  [SKIP] Flag --skip-training aktif dan bobot {TRAINED_GROUPWISE_WEIGHTS} ditemukan.")
    else:
        train_yolov13_groupwise()

    # 7. KNN Tuning on Validation Set
    print("\n>>> [7/13] Hyperparameter Tuning KNN pada Group-Wise Validation Set...")
    best_k, tuning_df, X_train, y_train = run_knn_tuning_groupwise()

    # 8. YOLO Native Object Detection Evaluation
    print("\n>>> [8/13] Evaluasi Native Detection YOLOv13n pada Group-Wise Test Set...")
    evaluate_yolo_groupwise_detection()

    # 9. YOLO Image-Level Classification Evaluation
    print("\n>>> [9/13] Evaluasi Image-Level Classification YOLOv13n...")
    evaluate_yolo_groupwise_classification()

    # 10. HOG-KNN GT Evaluation
    print("\n>>> [10/13] Evaluasi Baseline HOG-KNN GT Crop...")
    evaluate_knn_gt_groupwise(best_k=best_k, X_train=X_train, y_train=y_train)

    # 11. YOLO-HOG-KNN Hybrid Evaluation
    print("\n>>> [11/13] Evaluasi YOLO Crop + HOG-KNN Hybrid...")
    evaluate_yolo_hog_knn_groupwise(best_k=best_k, X_train=X_train, y_train=y_train)

    # 12. Standardized Runtime Benchmark
    print("\n>>> [12/13] Standardized High-Precision Runtime Benchmark...")
    run_runtime_benchmark_groupwise(best_k=best_k, X_train=X_train, y_train=y_train)

    # 13. Error Analysis & Report Generation
    print("\n>>> [13/13] Analisis Kesalahan & Pembuatan Seluruh Artefak Laporan...")
    run_error_analysis_groupwise()
    run_all_groupwise_reporting()

    elapsed = time.time() - t_start
    print("\n" + "=" * 70)
    print(f"  SEMUA TAHAP SELESAI DALAM {elapsed:.2f} DETIK ({elapsed/60:.2f} MENIT)!")
    print("=" * 70)


if __name__ == '__main__':
    main()
