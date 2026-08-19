"""
tune_knn.py — Pencarian Hyperparameter K Optimal (Group-Wise Validation Set)
=============================================================================
Mencari nilai K terbaik menggunakan VALIDATION SET Group-Wise (167 citra).
Kriteria:
  1. Macro F1-score tertinggi pada validation set
  2. Tiebreaker: Accuracy tertinggi
  3. Tiebreaker: Nilai K terkecil
Output:
  - outputs_groupwise/knn_validation_results.csv
  - outputs_groupwise/knn_validation_macro_f1.png
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from experiments_groupwise.config import (
    KNN_K_SEARCH_LIST, KNN_METRIC, OUTPUT_GROUPWISE_DIR, CLASS_LIST
)
from experiments_groupwise.hog_features import load_dataset_split, calculate_metrics


def run_knn_tuning_groupwise():
    print("=" * 65)
    print("  TAHAP 7: KNN HYPERPARAMETER TUNING (GROUP-WISE VALIDATION SET)")
    print("=" * 65)

    OUTPUT_GROUPWISE_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Load data train group-wise
    print("\n[1/3] Memuat data latih (Train Group-Wise: ground-truth crop + HOG)...")
    X_train, y_train, train_files, train_skipped = load_dataset_split('train')
    print(f"      Train loaded: {len(X_train)} samples ({len(train_skipped)} skipped)")

    # 2. Load data val group-wise
    print("\n[2/3] Memuat data validasi (Val Group-Wise: ground-truth crop + HOG)...")
    X_val, y_val, val_files, val_skipped = load_dataset_split('val')
    print(f"      Val loaded:   {len(X_val)} samples ({len(val_skipped)} skipped)")

    if len(X_train) == 0 or len(X_val) == 0:
        raise RuntimeError("Data train atau validation group-wise kosong!")

    # 3. Tuning K
    print(f"\n[3/3] Evaluasi K in {KNN_K_SEARCH_LIST} (metric={KNN_METRIC})...")
    print(f"      {'K':>3} | {'Accuracy':>10} | {'Macro P':>10} | {'Macro R':>10} | {'Macro F1':>10} | {'Weighted F1':>12}")
    print(f"      {'-'*3}-+-{'-'*10}-+-{'-'*10}-+-{'-'*10}-+-{'-'*10}-+-{'-'*12}")

    results = []
    for k in KNN_K_SEARCH_LIST:
        knn = KNeighborsClassifier(n_neighbors=k, metric=KNN_METRIC)
        knn.fit(X_train, y_train)
        y_pred = knn.predict(X_val)

        metrics = calculate_metrics(y_val, y_pred, CLASS_LIST)
        row = {
            'k': k,
            'accuracy': metrics['accuracy'],
            'macro_precision': metrics['macro_precision'],
            'macro_recall': metrics['macro_recall'],
            'macro_f1': metrics['macro_f1'],
            'weighted_f1': metrics['weighted_f1']
        }
        results.append(row)
        print(f"      {k:>3} | {row['accuracy']:>10.4f} | {row['macro_precision']:>10.4f} | {row['macro_recall']:>10.4f} | {row['macro_f1']:>10.4f} | {row['weighted_f1']:>12.4f}")

    results_df = pd.DataFrame(results)

    # Pilih K terbaik
    results_df_sorted = results_df.sort_values(
        by=['macro_f1', 'accuracy', 'k'],
        ascending=[False, False, True]
    )
    best_row = results_df_sorted.iloc[0]
    best_k = int(best_row['k'])

    print("\n  " + "=" * 45)
    print(f"  HASIL TUNING GROUP-WISE: K TERBAIK = {best_k}")
    print(f"  Macro F1 (Val) = {best_row['macro_f1']:.4f} ({best_row['macro_f1']*100:.2f}%)")
    print(f"  Accuracy (Val) = {best_row['accuracy']:.4f} ({best_row['accuracy']*100:.2f}%)")
    print("  " + "=" * 45)

    # Simpan CSV
    csv_path = OUTPUT_GROUPWISE_DIR / 'knn_validation_results.csv'
    results_df.to_csv(csv_path, index=False)
    print(f"\n  [SAVED] {csv_path}")

    # Plot grafik tuning
    fig, ax1 = plt.subplots(figsize=(9, 5.5))
    color_f1 = '#1976D2'
    color_acc = '#388E3C'

    ax1.plot(results_df['k'], results_df['macro_f1'] * 100,
             'o-', color=color_f1, linewidth=2.2, markersize=8,
             label='Macro F1-Score (%)', zorder=3)
    ax1.set_xlabel('Nilai K (Number of Neighbors)', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Macro F1-Score (%)', fontsize=11, color=color_f1, fontweight='bold')
    ax1.tick_params(axis='y', labelcolor=color_f1)
    ax1.set_xticks(results_df['k'])

    ax2 = ax1.twinx()
    ax2.plot(results_df['k'], results_df['accuracy'] * 100,
             's--', color=color_acc, linewidth=2.0, markersize=7,
             label='Accuracy (%)', zorder=2)
    ax2.set_ylabel('Accuracy (%)', fontsize=11, color=color_acc, fontweight='bold')
    ax2.tick_params(axis='y', labelcolor=color_acc)

    ax1.axvline(x=best_k, color='#D32F2F', linestyle=':', alpha=0.8, linewidth=2)
    ax1.annotate(f'K Terbaik = {best_k}\nMacro F1 = {best_row["macro_f1"]*100:.2f}%\nAccuracy = {best_row["accuracy"]*100:.2f}%',
                 xy=(best_k, best_row['macro_f1'] * 100),
                 xytext=(best_k + 1.2, best_row['macro_f1'] * 100 - 2.5),
                 fontsize=10,
                 fontweight='bold',
                 bbox=dict(boxstyle='round,pad=0.5', facecolor='#FFEBEE', edgecolor='#D32F2F', alpha=0.9),
                 arrowprops=dict(arrowstyle='->', color='#D32F2F', lw=1.5))

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='lower left', fontsize=10)

    plt.title(f'KNN Hyperparameter Tuning pada Group-Wise Validation Set ({len(X_val)} Citra)\nHOG Features (64x64) — Euclidean Distance',
              fontsize=12, pad=15)
    ax1.grid(True, linestyle='--', alpha=0.4)
    plt.tight_layout()

    plot_path = OUTPUT_GROUPWISE_DIR / 'knn_validation_macro_f1.png'
    plt.savefig(str(plot_path), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  [SAVED] {plot_path}")

    return best_k, results_df, X_train, y_train


if __name__ == '__main__':
    run_knn_tuning_groupwise()
