"""
knn_tuning.py — Pencarian Hyperparameter K untuk KNN
=====================================================
Mencari K optimal menggunakan VALIDATION SET.
Kriteria: Macro F1 tertinggi. Tiebreaker: accuracy → K terkecil.

K ∈ {1, 3, 5, 7, 9, 11, 13, 15}
Metric: euclidean

Output:
  - outputs/knn_validation_results.csv
  - outputs/knn_tuning_k_plot.png
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier

# Tambahkan parent directory ke path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from experiments.utils import (
    load_dataset_split, calculate_metrics, RANDOM_SEED, CLASS_LIST
)

# Konfigurasi
K_VALUES = [1, 3, 5, 7, 9, 11, 13, 15]
METRIC = 'euclidean'
OUTPUT_DIR = Path(__file__).resolve().parent.parent / 'outputs'


def run_knn_tuning():
    """
    Jalankan pencarian K pada validation set.

    Returns
    -------
    best_k : int
        K dengan Macro F1 terbaik pada validation set.
    results_df : pd.DataFrame
        Tabel hasil setiap K.
    X_train, y_train : numpy.ndarray
        Data training (untuk digunakan di evaluasi selanjutnya).
    """
    print("=" * 60)
    print("  KNN HYPERPARAMETER TUNING")
    print("  Pencarian K pada Validation Set")
    print("=" * 60)

    # 1. Load data train
    print("\n[1/3] Loading training data (ground-truth crop + HOG)...")
    X_train, y_train, train_files, train_skipped = load_dataset_split('train')
    print(f"      Train: {len(X_train)} samples loaded, "
          f"{len(train_skipped)} skipped")

    # 2. Load data validation
    print("\n[2/3] Loading validation data (ground-truth crop + HOG)...")
    X_val, y_val, val_files, val_skipped = load_dataset_split('val')
    print(f"      Val:   {len(X_val)} samples loaded, "
          f"{len(val_skipped)} skipped")

    if len(X_train) == 0 or len(X_val) == 0:
        raise RuntimeError("Data train atau val kosong!")

    # 3. Tuning K
    print(f"\n[3/3] Tuning K ∈ {K_VALUES} (metric={METRIC})...")
    print(f"      {'K':>3} | {'Accuracy':>10} | {'Macro P':>10} | "
          f"{'Macro R':>10} | {'Macro F1':>10} | {'Weighted F1':>12}")
    print(f"      {'-'*3}-+-{'-'*10}-+-{'-'*10}-+-{'-'*10}-+-{'-'*10}-+-{'-'*12}")

    results = []
    for k in K_VALUES:
        knn = KNeighborsClassifier(n_neighbors=k, metric=METRIC)
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

        print(f"      {k:>3} | {row['accuracy']:>10.4f} | "
              f"{row['macro_precision']:>10.4f} | "
              f"{row['macro_recall']:>10.4f} | "
              f"{row['macro_f1']:>10.4f} | "
              f"{row['weighted_f1']:>12.4f}")

    results_df = pd.DataFrame(results)

    # 4. Pilih K terbaik
    # Kriteria: Macro F1 tertinggi → accuracy tertinggi → K terkecil
    results_df_sorted = results_df.sort_values(
        by=['macro_f1', 'accuracy', 'k'],
        ascending=[False, False, True]
    )
    best_row = results_df_sorted.iloc[0]
    best_k = int(best_row['k'])

    print(f"\n  ╔══════════════════════════════════════╗")
    print(f"  ║  K TERBAIK = {best_k:<3}                     ║")
    print(f"  ║  Macro F1  = {best_row['macro_f1']:.4f}                ║")
    print(f"  ║  Accuracy  = {best_row['accuracy']:.4f}                ║")
    print(f"  ╚══════════════════════════════════════╝")

    # 5. Simpan hasil
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    csv_path = OUTPUT_DIR / 'knn_validation_results.csv'
    results_df.to_csv(csv_path, index=False)
    print(f"\n  [SAVED] {csv_path}")

    # 6. Plot K vs Macro F1
    _plot_k_tuning(results_df, best_k)

    return best_k, results_df, X_train, y_train


def _plot_k_tuning(results_df, best_k):
    """Plot grafik hasil tuning K."""
    fig, ax1 = plt.subplots(figsize=(10, 6))

    color_f1 = '#2196F3'
    color_acc = '#4CAF50'

    # Plot Macro F1
    ax1.plot(results_df['k'], results_df['macro_f1'],
             'o-', color=color_f1, linewidth=2, markersize=8,
             label='Macro F1', zorder=3)
    ax1.set_xlabel('K (Number of Neighbors)', fontsize=12)
    ax1.set_ylabel('Macro F1-Score', fontsize=12, color=color_f1)
    ax1.tick_params(axis='y', labelcolor=color_f1)
    ax1.set_xticks(results_df['k'])

    # Plot Accuracy di axis kedua
    ax2 = ax1.twinx()
    ax2.plot(results_df['k'], results_df['accuracy'],
             's--', color=color_acc, linewidth=2, markersize=7,
             label='Accuracy', zorder=2)
    ax2.set_ylabel('Accuracy', fontsize=12, color=color_acc)
    ax2.tick_params(axis='y', labelcolor=color_acc)

    # Tandai K terbaik
    best_row = results_df[results_df['k'] == best_k].iloc[0]
    ax1.axvline(x=best_k, color='red', linestyle=':', alpha=0.7, linewidth=1.5)
    ax1.annotate(f'Best K={best_k}\nMacro F1={best_row["macro_f1"]:.4f}',
                 xy=(best_k, best_row['macro_f1']),
                 xytext=(best_k + 1.5, best_row['macro_f1'] - 0.02),
                 fontsize=10,
                 arrowprops=dict(arrowstyle='->', color='red'),
                 color='red', fontweight='bold')

    # Legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2,
               loc='lower left', fontsize=10)

    plt.title('KNN Hyperparameter Tuning (Validation Set)\n'
              'HOG Features — Euclidean Distance',
              fontsize=13, pad=15)
    ax1.grid(True, alpha=0.3)
    plt.tight_layout()

    plot_path = OUTPUT_DIR / 'knn_tuning_k_plot.png'
    plt.savefig(str(plot_path), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  [SAVED] {plot_path}")


if __name__ == '__main__':
    best_k, results_df, _, _ = run_knn_tuning()
