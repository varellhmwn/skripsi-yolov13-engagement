"""
compare_models.py — Perbandingan Model & Pembuatan Laporan
============================================================
Membandingkan hasil evaluasi:
  1. YOLOv13n (image-level classification)
  2. HOG-KNN Ground-Truth Crop
  3. YOLO-based Face Crop + HOG-KNN

Output:
  - outputs/model_comparison.csv
  - outputs/error_analysis.csv
  - outputs/comparison_f1_chart.png
  - outputs/comparison_time_chart.png
  - outputs/experiment_report.md
"""

import sys
import json
import platform
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from experiments.utils import CLASS_LIST, BASE_DIR

OUTPUT_DIR = BASE_DIR / 'outputs'


def load_metrics(filename):
    """Load JSON metrics file."""
    path = OUTPUT_DIR / filename
    if not path.exists():
        print(f"  [WARNING] File tidak ditemukan: {path}")
        return None
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def load_predictions(filename):
    """Load predictions CSV."""
    path = OUTPUT_DIR / filename
    if not path.exists():
        print(f"  [WARNING] File tidak ditemukan: {path}")
        return None
    return pd.read_csv(path)


def create_comparison_table():
    """Buat tabel perbandingan model."""
    print("\n[1/5] Creating model comparison table...")

    yolo = load_metrics('yolo_metrics.json')
    knn_gt = load_metrics('hog_knn_gt_metrics.json')
    hybrid = load_metrics('yolo_hog_knn_metrics.json')

    rows = []

    # YOLOv13n
    if yolo:
        cls = yolo.get('classification', {})
        det = yolo.get('detection', {})
        tim = yolo.get('timing', {})
        rows.append({
            'Model': 'YOLOv13n',
            'Accuracy': cls.get('accuracy', 'N/A'),
            'Macro Precision': cls.get('macro_precision', 'N/A'),
            'Macro Recall': cls.get('macro_recall', 'N/A'),
            'Macro F1': cls.get('macro_f1', 'N/A'),
            'Weighted F1': cls.get('weighted_f1', 'N/A'),
            'mAP@0.5': det.get('mAP_50', 'N/A'),
            'mAP@0.5:0.95': det.get('mAP_50_95', 'N/A'),
            'Avg Time (ms)': tim.get('mean_inference_ms', 'N/A')
        })

    # HOG-KNN GT
    if knn_gt:
        tim = knn_gt.get('timing', {})
        rows.append({
            'Model': 'HOG-KNN GT Crop',
            'Accuracy': knn_gt.get('accuracy', 'N/A'),
            'Macro Precision': knn_gt.get('macro_precision', 'N/A'),
            'Macro Recall': knn_gt.get('macro_recall', 'N/A'),
            'Macro F1': knn_gt.get('macro_f1', 'N/A'),
            'Weighted F1': knn_gt.get('weighted_f1', 'N/A'),
            'mAP@0.5': 'N/A',
            'mAP@0.5:0.95': 'N/A',
            'Avg Time (ms)': tim.get('full_pipeline_mean_ms', 'N/A')
        })

    # YOLO Crop + HOG-KNN
    if hybrid:
        cls = hybrid.get('classification', {})
        tim = hybrid.get('timing', {})
        rows.append({
            'Model': 'YOLO Crop + HOG-KNN',
            'Accuracy': cls.get('accuracy', 'N/A'),
            'Macro Precision': cls.get('macro_precision', 'N/A'),
            'Macro Recall': cls.get('macro_recall', 'N/A'),
            'Macro F1': cls.get('macro_f1', 'N/A'),
            'Weighted F1': cls.get('weighted_f1', 'N/A'),
            'mAP@0.5': 'N/A',
            'mAP@0.5:0.95': 'N/A',
            'Avg Time (ms)': tim.get('full_pipeline_mean_ms', 'N/A')
        })

    df = pd.DataFrame(rows)
    csv_path = OUTPUT_DIR / 'model_comparison.csv'
    df.to_csv(csv_path, index=False)
    print(f"  [SAVED] {csv_path}")

    return df


def create_error_analysis():
    """Analisis kesalahan antar model."""
    print("\n[2/5] Creating error analysis...")

    yolo_preds = load_predictions('yolo_predictions.csv')
    knn_gt_preds = load_predictions('hog_knn_gt_predictions.csv')
    hybrid_preds = load_predictions('yolo_hog_knn_predictions.csv')

    if yolo_preds is None or knn_gt_preds is None or hybrid_preds is None:
        print("  [SKIP] Tidak semua prediction files tersedia.")
        return None

    # Merge by filename
    # Standardize column names
    yolo_df = yolo_preds[['filename', 'true_class', 'predicted_class', 'correct']].copy()
    yolo_df.columns = ['filename', 'true_class', 'yolo_prediction', 'yolo_correct']

    knn_df = knn_gt_preds[['filename', 'predicted_class', 'correct']].copy()
    knn_df.columns = ['filename', 'knn_gt_prediction', 'knn_gt_correct']

    hybrid_df = hybrid_preds[['filename', 'predicted_class', 'correct']].copy()
    hybrid_df.columns = ['filename', 'hybrid_prediction', 'hybrid_correct']

    merged = yolo_df.merge(knn_df, on='filename', how='outer')
    merged = merged.merge(hybrid_df, on='filename', how='outer')

    # Kategorisasi perbandingan YOLO vs KNN GT
    def categorize(row):
        yc = row.get('yolo_correct', False)
        kc = row.get('knn_gt_correct', False)
        if pd.isna(yc):
            yc = False
        if pd.isna(kc):
            kc = False
        if yc and kc:
            return 'both_correct'
        elif yc and not kc:
            return 'yolo_correct_knn_wrong'
        elif not yc and kc:
            return 'yolo_wrong_knn_correct'
        else:
            return 'both_wrong'

    merged['comparison_category'] = merged.apply(categorize, axis=1)

    csv_path = OUTPUT_DIR / 'error_analysis.csv'
    merged.to_csv(csv_path, index=False)
    print(f"  [SAVED] {csv_path}")

    # Print summary
    cat_counts = merged['comparison_category'].value_counts()
    print(f"\n  Error Analysis Summary:")
    for cat, count in cat_counts.items():
        pct = count / len(merged) * 100
        print(f"    {cat:<30}: {count:>4} ({pct:.1f}%)")

    return merged


def plot_f1_comparison():
    """Plot perbandingan Macro F1."""
    print("\n[3/5] Plotting F1 comparison chart...")

    yolo = load_metrics('yolo_metrics.json')
    knn_gt = load_metrics('hog_knn_gt_metrics.json')
    hybrid = load_metrics('yolo_hog_knn_metrics.json')

    models = []
    f1_scores = []
    acc_scores = []

    if yolo:
        models.append('YOLOv13n')
        f1_scores.append(yolo['classification']['macro_f1'])
        acc_scores.append(yolo['classification']['accuracy'])
    if knn_gt:
        models.append('HOG-KNN\nGT Crop')
        f1_scores.append(knn_gt['macro_f1'])
        acc_scores.append(knn_gt['accuracy'])
    if hybrid:
        models.append('YOLO Crop +\nHOG-KNN')
        f1_scores.append(hybrid['classification']['macro_f1'])
        acc_scores.append(hybrid['classification']['accuracy'])

    if not models:
        print("  [SKIP] No metrics available.")
        return

    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(models))
    width = 0.35

    bars1 = ax.bar(x - width/2, [s * 100 for s in f1_scores],
                   width, label='Macro F1', color='#2196F3',
                   edgecolor='white', linewidth=0.5)
    bars2 = ax.bar(x + width/2, [s * 100 for s in acc_scores],
                   width, label='Accuracy', color='#4CAF50',
                   edgecolor='white', linewidth=0.5)

    # Nilai di atas bar
    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width() / 2., bar.get_height() + 0.5,
                f'{bar.get_height():.2f}%', ha='center', va='bottom',
                fontsize=10, fontweight='bold')
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width() / 2., bar.get_height() + 0.5,
                f'{bar.get_height():.2f}%', ha='center', va='bottom',
                fontsize=10, fontweight='bold')

    ax.set_ylabel('Score (%)', fontsize=12)
    ax.set_title('Model Comparison — Macro F1 & Accuracy\n(Test Set, 173 Images)',
                 fontsize=13, pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=11)
    ax.legend(fontsize=11)
    ax.set_ylim(0, 105)
    ax.grid(axis='y', alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    path = OUTPUT_DIR / 'comparison_f1_chart.png'
    plt.savefig(str(path), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  [SAVED] {path}")


def plot_time_comparison():
    """Plot perbandingan waktu inferensi."""
    print("\n[4/5] Plotting processing time comparison chart...")

    yolo = load_metrics('yolo_metrics.json')
    knn_gt = load_metrics('hog_knn_gt_metrics.json')
    hybrid = load_metrics('yolo_hog_knn_metrics.json')

    models = []
    times = []
    time_labels = []

    if yolo:
        t = yolo['timing']['mean_inference_ms']
        models.append('YOLOv13n')
        times.append(t)
        time_labels.append(f'{t:.2f} ms')
    if knn_gt:
        t = knn_gt['timing']['full_pipeline_mean_ms']
        models.append('HOG-KNN\nGT Crop')
        times.append(t)
        time_labels.append(f'{t:.2f} ms')
    if hybrid:
        t = hybrid['timing']['full_pipeline_mean_ms']
        models.append('YOLO Crop +\nHOG-KNN')
        times.append(t)
        time_labels.append(f'{t:.2f} ms')

    if not models:
        print("  [SKIP] No metrics available.")
        return

    fig, ax = plt.subplots(figsize=(9, 6))
    colors = ['#FF9800', '#2196F3', '#9C27B0']

    bars = ax.barh(models, times, color=colors[:len(models)],
                   edgecolor='white', linewidth=0.5, height=0.5)

    for bar, label in zip(bars, time_labels):
        ax.text(bar.get_width() + max(times) * 0.02,
                bar.get_y() + bar.get_height() / 2,
                label, va='center', fontsize=11, fontweight='bold')

    ax.set_xlabel('Average Processing Time per Image (ms)', fontsize=12)
    ax.set_title('Average Processing Time Comparison\n(Test Set, 173 Images)',
                 fontsize=13, pad=15)
    ax.grid(axis='x', alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_xlim(0, max(times) * 1.3)

    plt.tight_layout()
    path = OUTPUT_DIR / 'comparison_time_chart.png'
    plt.savefig(str(path), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  [SAVED] {path}")


def generate_report():
    """Generate laporan eksperimen lengkap dalam Markdown."""
    print("\n[5/5] Generating experiment report...")

    yolo = load_metrics('yolo_metrics.json')
    knn_gt = load_metrics('hog_knn_gt_metrics.json')
    hybrid = load_metrics('yolo_hog_knn_metrics.json')
    tuning_df = None
    tuning_path = OUTPUT_DIR / 'knn_validation_results.csv'
    if tuning_path.exists():
        tuning_df = pd.read_csv(tuning_path)

    error_df = None
    error_path = OUTPUT_DIR / 'error_analysis.csv'
    if error_path.exists():
        error_df = pd.read_csv(error_path)

    report = []
    report.append("# Laporan Eksperimen: Perbandingan YOLOv13n vs HOG-KNN")
    report.append(f"\n**Tanggal**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append(f"**Platform**: {platform.system()} {platform.release()}")
    report.append(f"**Processor**: {platform.processor()}")
    report.append("")

    # === A. Dataset ===
    report.append("## A. Dataset")
    report.append("")
    report.append("| Split | Jumlah | Engaged | Confused | Bored | Frustrated |")
    report.append("|-------|-------:|--------:|---------:|------:|-----------:|")

    from experiments.utils import get_class_distribution
    for split in ['train', 'val', 'test']:
        dist = get_class_distribution(split)
        total = sum(dist.values())
        report.append(
            f"| {split} | {total} | {dist[0]} | {dist[1]} | {dist[2]} | {dist[3]} |"
        )

    report.append("")
    report.append("**Catatan dataset:**")
    report.append("- 39 orphan label files di train set (tanpa matching image)")
    report.append("- 8 file train memiliki 2 bounding box (objek utama + wajah kecil di background)")
    report.append("- Semua file test dan val memiliki tepat 1 bounding box")
    report.append("")

    # === B. Konfigurasi HOG ===
    report.append("## B. Konfigurasi HOG")
    report.append("")
    report.append("| Parameter | Nilai |")
    report.append("|-----------|-------|")
    report.append("| Resize | 64 × 64 pixel |")
    report.append("| Color space | Grayscale |")
    report.append("| Orientations | 9 |")
    report.append("| Pixels per cell | 8 × 8 |")
    report.append("| Cells per block | 2 × 2 |")
    report.append("| Block normalization | L2-Hys |")
    report.append("| Library | scikit-image (`skimage.feature.hog`) |")
    report.append("")

    # === C. Tuning KNN ===
    report.append("## C. Tuning KNN")
    report.append("")
    if tuning_df is not None:
        report.append("### Hasil Validasi per-K")
        report.append("")
        report.append("| K | Accuracy | Macro Precision | Macro Recall | Macro F1 | Weighted F1 |")
        report.append("|--:|---------:|----------------:|-------------:|---------:|------------:|")
        for _, row in tuning_df.iterrows():
            report.append(
                f"| {int(row['k'])} | {row['accuracy']:.4f} | "
                f"{row['macro_precision']:.4f} | {row['macro_recall']:.4f} | "
                f"{row['macro_f1']:.4f} | {row['weighted_f1']:.4f} |"
            )

        # K terbaik
        best_row = tuning_df.sort_values(
            by=['macro_f1', 'accuracy', 'k'],
            ascending=[False, False, True]
        ).iloc[0]
        best_k = int(best_row['k'])

        report.append("")
        report.append(f"**K terbaik = {best_k}** (Macro F1 = {best_row['macro_f1']:.4f})")
        report.append("")
        report.append("**Kriteria pemilihan:**")
        report.append("1. Macro F1-score tertinggi pada validation set")
        report.append("2. Tiebreaker: Accuracy tertinggi → K terkecil")
        report.append(f"3. Metric jarak: Euclidean")
        report.append(f"4. Validation set berjumlah 168 citra")
    report.append("")

    # === D. Hasil YOLOv13n ===
    report.append("## D. Hasil YOLOv13n")
    report.append("")
    if yolo:
        det = yolo.get('detection', {})
        cls = yolo.get('classification', {})
        tim = yolo.get('timing', {})

        report.append("### Object Detection Metrics")
        report.append("")
        report.append("| Metrik | Nilai |")
        report.append("|--------|------:|")
        report.append(f"| Precision | {det.get('precision', 0):.4f} |")
        report.append(f"| Recall | {det.get('recall', 0):.4f} |")
        report.append(f"| mAP@0.5 | {det.get('mAP_50', 0):.4f} |")
        report.append(f"| mAP@0.75 | {det.get('mAP_75', 0):.4f} |")
        report.append(f"| mAP@0.5:0.95 | {det.get('mAP_50_95', 0):.4f} |")
        report.append("")

        report.append("### Image-Level Classification Metrics")
        report.append("")
        report.append("| Metrik | Nilai |")
        report.append("|--------|------:|")
        report.append(f"| Accuracy | {cls.get('accuracy', 0):.4f} |")
        report.append(f"| Macro Precision | {cls.get('macro_precision', 0):.4f} |")
        report.append(f"| Macro Recall | {cls.get('macro_recall', 0):.4f} |")
        report.append(f"| Macro F1 | {cls.get('macro_f1', 0):.4f} |")
        report.append(f"| Weighted Precision | {cls.get('weighted_precision', 0):.4f} |")
        report.append(f"| Weighted Recall | {cls.get('weighted_recall', 0):.4f} |")
        report.append(f"| Weighted F1 | {cls.get('weighted_f1', 0):.4f} |")
        report.append("")

        # Per-class
        if 'per_class' in cls:
            report.append("#### Per-Class Performance (YOLO)")
            report.append("")
            report.append("| Kelas | Precision | Recall | F1-Score | Support |")
            report.append("|-------|----------:|-------:|---------:|--------:|")
            for name in CLASS_LIST:
                if name in cls['per_class']:
                    pc = cls['per_class'][name]
                    report.append(
                        f"| {name} | {pc['precision']:.4f} | {pc['recall']:.4f} | "
                        f"{pc['f1']:.4f} | {pc['support']} |"
                    )
            report.append("")

        report.append(f"**Waktu inferensi rata-rata**: {tim.get('mean_inference_ms', 0):.2f} ms/image")
        report.append(f"**Estimasi FPS**: {tim.get('estimated_fps', 0):.1f}")
        report.append(f"**Detection failures**: {tim.get('num_detection_failed', 0)}")
    report.append("")

    # === E. Hasil HOG-KNN Ground Truth ===
    report.append("## E. Hasil HOG-KNN Ground-Truth Crop")
    report.append("")
    if knn_gt:
        report.append("| Metrik | Nilai |")
        report.append("|--------|------:|")
        report.append(f"| Accuracy | {knn_gt.get('accuracy', 0):.4f} |")
        report.append(f"| Macro Precision | {knn_gt.get('macro_precision', 0):.4f} |")
        report.append(f"| Macro Recall | {knn_gt.get('macro_recall', 0):.4f} |")
        report.append(f"| Macro F1 | {knn_gt.get('macro_f1', 0):.4f} |")
        report.append(f"| Weighted Precision | {knn_gt.get('weighted_precision', 0):.4f} |")
        report.append(f"| Weighted Recall | {knn_gt.get('weighted_recall', 0):.4f} |")
        report.append(f"| Weighted F1 | {knn_gt.get('weighted_f1', 0):.4f} |")
        report.append("")

        if 'per_class' in knn_gt:
            report.append("#### Per-Class Performance (HOG-KNN GT)")
            report.append("")
            report.append("| Kelas | Precision | Recall | F1-Score | Support |")
            report.append("|-------|----------:|-------:|---------:|--------:|")
            for name in CLASS_LIST:
                if name in knn_gt['per_class']:
                    pc = knn_gt['per_class'][name]
                    report.append(
                        f"| {name} | {pc['precision']:.4f} | {pc['recall']:.4f} | "
                        f"{pc['f1']:.4f} | {pc['support']} |"
                    )
            report.append("")

        tim = knn_gt.get('timing', {})
        report.append(f"**KNN predict per image**: {tim.get('knn_predict_per_img_ms', 0):.2f} ms")
        report.append(f"**Full pipeline (crop+resize+grayscale+HOG+KNN)**: "
                      f"{tim.get('full_pipeline_mean_ms', 0):.2f} ms/image (mean)")
    report.append("")

    # === F. Hasil YOLO-HOG-KNN ===
    report.append("## F. Hasil YOLO-based Face Crop + HOG-KNN")
    report.append("")
    if hybrid:
        cls = hybrid.get('classification', {})
        e2e = hybrid.get('end_to_end', {})
        tim = hybrid.get('timing', {})

        report.append("### Classification Metrics (Detected Faces Only)")
        report.append("")
        report.append("| Metrik | Nilai |")
        report.append("|--------|------:|")
        report.append(f"| Accuracy | {cls.get('accuracy', 0):.4f} |")
        report.append(f"| Macro Precision | {cls.get('macro_precision', 0):.4f} |")
        report.append(f"| Macro Recall | {cls.get('macro_recall', 0):.4f} |")
        report.append(f"| Macro F1 | {cls.get('macro_f1', 0):.4f} |")
        report.append(f"| Weighted F1 | {cls.get('weighted_f1', 0):.4f} |")
        report.append("")

        if 'per_class' in cls:
            report.append("#### Per-Class Performance (YOLO-HOG-KNN)")
            report.append("")
            report.append("| Kelas | Precision | Recall | F1-Score | Support |")
            report.append("|-------|----------:|-------:|---------:|--------:|")
            for name in CLASS_LIST:
                if name in cls['per_class']:
                    pc = cls['per_class'][name]
                    report.append(
                        f"| {name} | {pc['precision']:.4f} | {pc['recall']:.4f} | "
                        f"{pc['f1']:.4f} | {pc['support']} |"
                    )
            report.append("")

        report.append("### End-to-End Statistics")
        report.append("")
        report.append(f"- Total images: {e2e.get('total_images', 0)}")
        report.append(f"- Detected successfully: {e2e.get('detected', 0)}")
        report.append(f"- Detection failed: {e2e.get('detection_failed', 0)}")
        report.append(f"- End-to-end accuracy: {e2e.get('end_to_end_accuracy', 0):.4f}")
        report.append("")

        report.append("### Timing Breakdown")
        report.append("")
        report.append("| Komponen | Waktu (ms) |")
        report.append("|----------|----------:|")
        report.append(f"| YOLO detection | {tim.get('yolo_detection_mean_ms', 0):.2f} |")
        report.append(f"| Crop + HOG preprocessing | {tim.get('crop_hog_preprocess_mean_ms', 0):.2f} |")
        report.append(f"| KNN prediction | {tim.get('knn_predict_mean_ms', 0):.2f} |")
        report.append(f"| **Full pipeline** | **{tim.get('full_pipeline_mean_ms', 0):.2f}** |")
        report.append("")
    report.append("")

    # === G. Perbandingan ===
    report.append("## G. Perbandingan Model")
    report.append("")

    report.append("### Classification Performance")
    report.append("")
    report.append("| Model | Accuracy | Macro P | Macro R | Macro F1 | Weighted F1 |")
    report.append("|-------|--------:|---------:|--------:|---------:|------------:|")

    if yolo:
        c = yolo['classification']
        report.append(f"| YOLOv13n | {c['accuracy']:.4f} | {c['macro_precision']:.4f} | "
                      f"{c['macro_recall']:.4f} | {c['macro_f1']:.4f} | {c['weighted_f1']:.4f} |")
    if knn_gt:
        report.append(f"| HOG-KNN GT Crop | {knn_gt['accuracy']:.4f} | {knn_gt['macro_precision']:.4f} | "
                      f"{knn_gt['macro_recall']:.4f} | {knn_gt['macro_f1']:.4f} | {knn_gt['weighted_f1']:.4f} |")
    if hybrid:
        c = hybrid['classification']
        report.append(f"| YOLO Crop + HOG-KNN | {c.get('accuracy', 0):.4f} | {c.get('macro_precision', 0):.4f} | "
                      f"{c.get('macro_recall', 0):.4f} | {c.get('macro_f1', 0):.4f} | {c.get('weighted_f1', 0):.4f} |")

    report.append("")
    report.append("### Object Detection Performance (Khusus YOLO)")
    report.append("")
    if yolo:
        det = yolo['detection']
        report.append("| Metrik | Nilai |")
        report.append("|--------|------:|")
        report.append(f"| Precision | {det['precision']:.4f} |")
        report.append(f"| Recall | {det['recall']:.4f} |")
        report.append(f"| mAP@0.5 | {det['mAP_50']:.4f} |")
        report.append(f"| mAP@0.5:0.95 | {det['mAP_50_95']:.4f} |")
    report.append("")
    report.append("> **Catatan:** HOG-KNN tidak menghasilkan bounding box sehingga metrik mAP dan IoU "
                  "tidak dihitung untuk HOG-KNN. Perbandingan mAP antara YOLO dan KNN tidak valid "
                  "karena KNN hanya melakukan klasifikasi, bukan lokalisasi objek.")
    report.append("")

    report.append("### Waktu Pemrosesan")
    report.append("")
    report.append("| Model | Avg Time (ms) | Catatan |")
    report.append("|-------|-------------:|---------|")
    if yolo:
        t = yolo['timing']['mean_inference_ms']
        report.append(f"| YOLOv13n | {t:.2f} | End-to-end (detection + classification) |")
    if knn_gt:
        t = knn_gt['timing']
        report.append(f"| HOG-KNN GT Crop | {t['full_pipeline_mean_ms']:.2f} | "
                      f"crop + resize + grayscale + HOG + KNN predict |")
    if hybrid:
        t = hybrid['timing']
        report.append(f"| YOLO Crop + HOG-KNN | {t['full_pipeline_mean_ms']:.2f} | "
                      f"YOLO detection + crop + HOG + KNN predict |")
    report.append("")
    report.append("> **Catatan waktu:** Waktu `knn.predict()` saja jauh lebih kecil dari waktu total pipeline. "
                  "Membandingkan hanya `knn.predict()` dengan waktu inferensi YOLO tidak fair karena "
                  "tidak memperhitungkan preprocessing (crop, resize, grayscale, HOG extraction).")
    report.append("")

    # === H. Error Analysis ===
    report.append("## H. Error Analysis")
    report.append("")
    if error_df is not None:
        cat_counts = error_df['comparison_category'].value_counts()
        report.append("### Distribusi Kategori Kesalahan (YOLO vs HOG-KNN GT)")
        report.append("")
        report.append("| Kategori | Jumlah | Persentase |")
        report.append("|----------|-------:|-----------:|")
        for cat in ['both_correct', 'yolo_correct_knn_wrong',
                     'yolo_wrong_knn_correct', 'both_wrong']:
            count = cat_counts.get(cat, 0)
            pct = count / len(error_df) * 100
            report.append(f"| {cat} | {count} | {pct:.1f}% |")
        report.append("")

        # Confusion terbesar
        report.append("### Confusion Terbesar")
        report.append("")
        if knn_gt and 'confusion_matrix' in knn_gt:
            cm = np.array(knn_gt['confusion_matrix'])
            report.append("**HOG-KNN GT Crop:**")
            report.append("")
            confusions = []
            for i in range(len(CLASS_LIST)):
                for j in range(len(CLASS_LIST)):
                    if i != j and cm[i][j] > 0:
                        confusions.append((CLASS_LIST[i], CLASS_LIST[j], cm[i][j]))
            confusions.sort(key=lambda x: x[2], reverse=True)
            for true_cls, pred_cls, count in confusions[:5]:
                report.append(f"- {true_cls} → {pred_cls}: {count} kesalahan")
            report.append("")

        if yolo and 'confusion_matrix' in yolo['classification']:
            cm = np.array(yolo['classification']['confusion_matrix'])
            report.append("**YOLOv13n:**")
            report.append("")
            confusions = []
            for i in range(len(CLASS_LIST)):
                for j in range(len(CLASS_LIST)):
                    if i != j and cm[i][j] > 0:
                        confusions.append((CLASS_LIST[i], CLASS_LIST[j], cm[i][j]))
            confusions.sort(key=lambda x: x[2], reverse=True)
            for true_cls, pred_cls, count in confusions[:5]:
                report.append(f"- {true_cls} → {pred_cls}: {count} kesalahan")
            report.append("")
    report.append("")

    # === I. Kesimpulan ===
    report.append("## I. Kesimpulan")
    report.append("")

    if yolo and knn_gt and hybrid:
        yolo_f1 = yolo['classification']['macro_f1']
        knn_f1 = knn_gt['macro_f1']
        hybrid_f1 = hybrid['classification'].get('macro_f1', 0)

        report.append("Berdasarkan hasil eksperimen pada 173 citra test set:")
        report.append("")
        report.append(f"1. **YOLOv13n** memperoleh Macro F1 sebesar **{yolo_f1:.4f}** pada evaluasi "
                      f"image-level classification. Model ini melakukan lokalisasi wajah dan "
                      f"klasifikasi ekspresi secara simultan (end-to-end).")
        report.append("")
        report.append(f"2. **HOG-KNN Ground-Truth Crop** memperoleh Macro F1 sebesar **{knn_f1:.4f}**. "
                      f"Model ini menerima crop wajah yang sudah benar (ground-truth bounding box), "
                      f"sehingga tugasnya murni klasifikasi tanpa perlu lokalisasi.")
        report.append("")
        report.append(f"3. **YOLO-based Face Crop + HOG-KNN** memperoleh Macro F1 sebesar **{hybrid_f1:.4f}**. "
                      f"Pipeline ini menunjukkan performa KNN ketika crop wajah diperoleh secara "
                      f"otomatis dari deteksi YOLO, bukan dari ground truth.")
        report.append("")

        report.append("**Catatan penting:**")
        report.append("- Perbandingan ini harus mempertimbangkan perbedaan tugas dan pipeline masing-masing model.")
        report.append("- YOLO melakukan detection + classification sekaligus, sedangkan HOG-KNN hanya klasifikasi.")
        report.append("- HOG-KNN GT Crop mendapat keuntungan dari crop wajah yang sudah benar.")
        report.append("- mAP tidak relevan untuk HOG-KNN karena tidak menghasilkan bounding box.")
        report.append("- Perbedaan performa antara HOG-KNN GT dan YOLO-HOG-KNN menunjukkan dampak kualitas lokalisasi terhadap klasifikasi.")
    report.append("")

    # === Journal-ready summary ===
    report.append("## Journal-ready Summary")
    report.append("")

    if yolo and knn_gt and hybrid and tuning_df is not None:
        best_row = tuning_df.sort_values(
            by=['macro_f1', 'accuracy', 'k'],
            ascending=[False, False, True]
        ).iloc[0]
        best_k = int(best_row['k'])

        knn_config = knn_gt.get('config', {})

        report.append("### Metode Eksperimen HOG-KNN")
        report.append("")
        report.append(
            f"Sebagai model pembanding, digunakan pendekatan klasifikasi berbasis "
            f"Histogram of Oriented Gradients (HOG) dan K-Nearest Neighbors (KNN). "
            f"Citra wajah diperoleh melalui crop berdasarkan bounding box ground truth "
            f"dari anotasi YOLO. Setiap crop di-resize ke ukuran 64×64 piksel, "
            f"dikonversi ke grayscale, kemudian diekstraksi fitur HOG dengan parameter "
            f"{knn_config.get('hog_orientations', 9)} orientasi, "
            f"pixels per cell {knn_config.get('hog_pixels_per_cell', [8,8])}, "
            f"cells per block {knn_config.get('hog_cells_per_block', [2,2])}, "
            f"dan normalisasi {knn_config.get('hog_block_norm', 'L2-Hys')}. "
            f"Implementasi HOG menggunakan library scikit-image (`skimage.feature.hog`). "
            f"Selain itu, dilakukan eksperimen hybrid di mana bounding box diperoleh "
            f"secara otomatis dari deteksi YOLOv13n, kemudian crop diproses dengan "
            f"HOG-KNN untuk klasifikasi ekspresi."
        )
        report.append("")

        report.append("### Hasil Tuning K")
        report.append("")
        report.append(
            f"Pencarian hyperparameter K dilakukan pada validation set (168 citra) "
            f"dengan nilai K ∈ {{1, 3, 5, 7, 9, 11, 13, 15}} dan metric jarak Euclidean. "
            f"Hasil tuning menunjukkan bahwa K = {best_k} menghasilkan Macro F1-score "
            f"tertinggi pada validation set sebesar {best_row['macro_f1']:.4f} "
            f"dengan accuracy {best_row['accuracy']:.4f}. "
            f"Pemilihan K dilakukan berdasarkan kriteria utama Macro F1-score validation "
            f"tertinggi, diikuti accuracy dan nilai K terkecil sebagai tiebreaker."
        )
        report.append("")

        report.append("### Hasil Pengujian KNN")
        report.append("")
        report.append(
            f"Evaluasi HOG-KNN (K={best_k}) pada 173 citra test set dengan ground-truth "
            f"crop menghasilkan accuracy {knn_gt['accuracy']:.4f}, "
            f"macro precision {knn_gt['macro_precision']:.4f}, "
            f"macro recall {knn_gt['macro_recall']:.4f}, "
            f"dan macro F1-score {knn_gt['macro_f1']:.4f}. "
            f"Pada eksperimen hybrid (YOLO crop + HOG-KNN), di mana crop wajah diperoleh "
            f"secara otomatis dari deteksi YOLOv13n, diperoleh accuracy "
            f"{hybrid['classification'].get('accuracy', 0):.4f} dan "
            f"macro F1-score {hybrid['classification'].get('macro_f1', 0):.4f}."
        )
        report.append("")

        yolo_f1 = yolo['classification']['macro_f1']
        knn_f1 = knn_gt['macro_f1']
        hybrid_f1 = hybrid['classification'].get('macro_f1', 0)

        report.append("### Perbandingan YOLO dan KNN")
        report.append("")
        report.append(
            f"YOLOv13n memperoleh macro F1-score sebesar {yolo_f1:.4f} pada evaluasi "
            f"image-level classification, sedangkan HOG-KNN dengan ground-truth crop "
            f"memperoleh {knn_f1:.4f}. Pipeline hybrid YOLO crop + HOG-KNN menghasilkan "
            f"macro F1-score {hybrid_f1:.4f}. "
            f"Perlu dicatat bahwa perbandingan ini memiliki konteks yang berbeda: "
            f"YOLOv13n melakukan lokalisasi wajah dan klasifikasi ekspresi secara "
            f"simultan (end-to-end), sedangkan HOG-KNN ground-truth menerima crop wajah "
            f"yang sudah benar sehingga tugasnya lebih sederhana (murni klasifikasi). "
            f"Pipeline hybrid menunjukkan performa klasifikasi KNN ketika bergantung "
            f"pada lokalisasi otomatis dari YOLO. Metrik mAP tidak dihitung untuk "
            f"HOG-KNN karena metode ini tidak menghasilkan bounding box prediksi."
        )
        report.append("")

        report.append("### Tabel Ringkas Perbandingan")
        report.append("")
        report.append("| Model | Accuracy | Macro F1 | Weighted F1 | mAP@0.5 | mAP@0.5:0.95 |")
        report.append("|-------|--------:|---------:|------------:|--------:|-------------:|")

        det = yolo.get('detection', {})
        report.append(
            f"| YOLOv13n | {yolo['classification']['accuracy']:.4f} | "
            f"{yolo_f1:.4f} | {yolo['classification']['weighted_f1']:.4f} | "
            f"{det.get('mAP_50', 0):.4f} | {det.get('mAP_50_95', 0):.4f} |"
        )
        report.append(
            f"| HOG-KNN GT Crop | {knn_gt['accuracy']:.4f} | "
            f"{knn_f1:.4f} | {knn_gt['weighted_f1']:.4f} | N/A | N/A |"
        )
        report.append(
            f"| YOLO Crop + HOG-KNN | {hybrid['classification'].get('accuracy', 0):.4f} | "
            f"{hybrid_f1:.4f} | {hybrid['classification'].get('weighted_f1', 0):.4f} | N/A | N/A |"
        )
        report.append("")

        report.append("### Penjelasan mAP")
        report.append("")
        report.append(
            "Metrik mean Average Precision (mAP) tidak dihitung untuk model HOG-KNN "
            "karena metode ini tidak melakukan prediksi bounding box. HOG-KNN hanya "
            "melakukan klasifikasi pada crop wajah yang sudah tersedia, sehingga "
            "tidak menghasilkan output lokalisasi yang diperlukan untuk menghitung "
            "Intersection over Union (IoU) dan Average Precision per kelas. "
            "Membandingkan mAP YOLO dengan HOG-KNN tidak valid karena keduanya "
            "memiliki output yang fundamentally berbeda: YOLO menghasilkan "
            "bounding box + kelas + confidence, sedangkan KNN hanya menghasilkan "
            "kelas prediksi."
        )

    report.append("")

    # Save report
    report_path = OUTPUT_DIR / 'experiment_report.md'
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report))
    print(f"  [SAVED] {report_path}")


def run_comparison():
    """Jalankan seluruh proses perbandingan."""
    print("\n" + "=" * 60)
    print("  MODEL COMPARISON & REPORT GENERATION")
    print("=" * 60)

    create_comparison_table()
    create_error_analysis()
    plot_f1_comparison()
    plot_time_comparison()
    generate_report()

    print("\n" + "=" * 60)
    print("  PERBANDINGAN SELESAI!")
    print("=" * 60)


if __name__ == '__main__':
    run_comparison()
