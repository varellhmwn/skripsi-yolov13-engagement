"""
analyze_realtime_log.py — Analisis Performa Real-Time & Efektivitas Temporal Post-Processing
================================================================================================
Skrip ini membaca file realtime_prediction_log.csv hasil pengujian YOLOv13n secara real-time
dan menghasilkan:
  1. Statistik FPS (mean, median, P5, P95, dll.)
  2. Analisis perubahan label (raw vs. stable)
  3. Segmentasi label (jumlah segmen, durasi, segmen pendek)
  4. Analisis status neutral pada stable_label
  5. Visualisasi dalam bentuk PNG resolusi tinggi
  6. File ringkasan CSV dan JSON

Penggunaan:
    python analyze_realtime_log.py realtime_prediction_log.csv

Penulis: Verell Haziq Maulana Wahid
"""

import sys
import json
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend agar bisa berjalan tanpa GUI
import matplotlib.pyplot as plt


# ════════════════════════════════════════════════════════════════════
# 1. LOAD & VALIDASI DATA
# ════════════════════════════════════════════════════════════════════

def load_and_validate_data(csv_path: str) -> pd.DataFrame:
    """
    Membaca file CSV dan memvalidasi keberadaan kolom-kolom yang dibutuhkan.
    Kolom numerik dikonversi secara aman (errors='coerce').
    Baris yang tidak valid (NaN pada kolom kunci) dihapus dan dilaporkan.
    Data diurutkan berdasarkan frame_index.

    Returns:
        pd.DataFrame yang sudah bersih dan terurut.
    """
    REQUIRED_COLS = [
        'frame_index', 'timestamp_sec', 'raw_class_name', 'raw_confidence',
        'stable_label', 'vote_ratio', 'avg_confidence',
        'bbox_x1', 'bbox_y1', 'bbox_x2', 'bbox_y2', 'fps'
    ]
    NUMERIC_COLS = [
        'frame_index', 'timestamp_sec', 'raw_confidence', 'vote_ratio',
        'avg_confidence', 'bbox_x1', 'bbox_y1', 'bbox_x2', 'bbox_y2', 'fps'
    ]

    # Membaca CSV
    df = pd.read_csv(csv_path)
    print(f"[INFO] File dimuat: {csv_path}")
    print(f"[INFO] Ukuran awal: {len(df)} baris x {len(df.columns)} kolom")

    # Mapping aliases
    rename_map = {
        'frame_id': 'frame_index',
        'timestamp': 'timestamp_sec',
        'raw_label': 'raw_class_name'
    }
    for old_c, new_c in rename_map.items():
        if old_c in df.columns and new_c not in df.columns:
            df.rename(columns={old_c: new_c}, inplace=True)

    # Validasi kolom
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"[ERROR] Kolom tidak ditemukan: {missing}")

    # Konversi tipe numerik secara aman
    for col in NUMERIC_COLS:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # Hitung dan hapus baris invalid (NaN pada kolom kunci)
    key_cols = ['frame_index', 'timestamp_sec', 'raw_class_name', 'stable_label', 'fps']
    invalid_mask = df[key_cols].isna().any(axis=1)
    n_invalid = invalid_mask.sum()
    if n_invalid > 0:
        print(f"[WARNING] {n_invalid} baris tidak valid dihapus (NaN pada kolom kunci).")
        df = df[~invalid_mask].copy()

    # Urutkan berdasarkan frame_index
    df = df.sort_values('frame_index').reset_index(drop=True)
    print(f"[INFO] Data valid setelah pembersihan: {len(df)} baris")

    return df


# ════════════════════════════════════════════════════════════════════
# 2. STATISTIK FPS
# ════════════════════════════════════════════════════════════════════

def calculate_fps_statistics(df: pd.DataFrame, warmup_frames: int = 10) -> dict:
    """
    Menghitung statistik FPS dari kolom 'fps'.

    Parameter:
        warmup_frames: jumlah frame awal yang dikeluarkan dari perhitungan
                       karena biasanya FPS belum stabil (model loading, dll.)

    Rumus yang digunakan:
        - Mean FPS   = sum(fps) / N
        - Median FPS = nilai tengah dari distribusi fps
        - P5, P95    = persentil ke-5 dan ke-95
        - Frames >= 30 FPS = count(fps >= 30) / N * 100

    Returns:
        dict berisi semua statistik FPS.
    """
    total_frames = len(df)
    max_ts = df['timestamp_sec'].max()
    min_ts = df['timestamp_sec'].min()
    test_duration = max_ts - min_ts

    # Keluarkan frame warm-up
    fps_data = df['fps'].iloc[warmup_frames:].values
    n_samples = len(fps_data)

    stats = {
        'total_frames': total_frames,
        'test_duration_sec': round(test_duration, 2),
        'warmup_frames_excluded': warmup_frames,
        'fps_samples': n_samples,
        'mean_fps': round(np.mean(fps_data), 2),
        'median_fps': round(np.median(fps_data), 2),
        'min_fps': round(np.min(fps_data), 2),
        'max_fps': round(np.max(fps_data), 2),
        'p5_fps': round(np.percentile(fps_data, 5), 2),
        'p95_fps': round(np.percentile(fps_data, 95), 2),
        'frames_ge_30': int(np.sum(fps_data >= 30)),
        'pct_frames_ge_30': round(np.sum(fps_data >= 30) / n_samples * 100, 2),
    }
    return stats


# ════════════════════════════════════════════════════════════════════
# 3. PERUBAHAN LABEL (Label Changes)
# ════════════════════════════════════════════════════════════════════

def count_label_changes(series: pd.Series) -> int:
    """
    Menghitung jumlah perubahan label pada sebuah kolom.

    Aturan: perubahan dihitung jika label pada frame ke-i berbeda
    dengan label pada frame ke-(i-1).

    Rumus:
        label_changes = sum(label[i] != label[i-1]) untuk i = 1..N-1

    Returns:
        int jumlah perubahan label.
    """
    return int((series != series.shift(1)).sum() - 1)  # -1 karena frame pertama selalu "berbeda" dari NaN


# ════════════════════════════════════════════════════════════════════
# 4. SEGMENTASI LABEL
# ════════════════════════════════════════════════════════════════════

def extract_segments(df: pd.DataFrame, label_col: str) -> pd.DataFrame:
    """
    Mengekstrak segmen-segmen berurutan dari kolom label tertentu.

    Definisi segmen:
        Satu segmen adalah rangkaian frame berturut-turut yang memiliki
        label yang sama. Contoh:
            engaged, engaged, engaged, bored, bored, confused
            => 3 segmen: engaged(3 frame), bored(2 frame), confused(1 frame)

    Durasi segmen dihitung berdasarkan timestamp_sec (bukan asumsi 30 FPS).
    Untuk segmen 1 frame, durasi diestimasi dari median interval antar-frame.

    Returns:
        pd.DataFrame dengan kolom:
        segment_id, label, start_frame, end_frame, frame_count,
        start_time, end_time, duration_sec
    """
    labels = df[label_col].values
    frames = df['frame_index'].values
    times = df['timestamp_sec'].values

    # Estimasi durasi 1 frame dari median selisih timestamp
    ts_diffs = np.diff(times)
    median_interval = np.median(ts_diffs) if len(ts_diffs) > 0 else 1.0 / 30.0

    segments = []
    seg_start = 0

    for i in range(1, len(labels)):
        if labels[i] != labels[seg_start]:
            # Segmen berakhir di frame sebelumnya (i-1)
            seg_end = i - 1
            frame_count = seg_end - seg_start + 1
            start_time = times[seg_start]
            end_time = times[seg_end]
            # Untuk segmen 1 frame, tambahkan estimasi interval 1 frame
            duration = (end_time - start_time) + median_interval

            segments.append({
                'segment_id': len(segments) + 1,
                'label': labels[seg_start],
                'start_frame': int(frames[seg_start]),
                'end_frame': int(frames[seg_end]),
                'frame_count': frame_count,
                'start_time': round(start_time, 4),
                'end_time': round(end_time, 4),
                'duration_sec': round(duration, 4),
            })
            seg_start = i

    # Segmen terakhir
    seg_end = len(labels) - 1
    frame_count = seg_end - seg_start + 1
    start_time = times[seg_start]
    end_time = times[seg_end]
    duration = (end_time - start_time) + median_interval

    segments.append({
        'segment_id': len(segments) + 1,
        'label': labels[seg_start],
        'start_frame': int(frames[seg_start]),
        'end_frame': int(frames[seg_end]),
        'frame_count': frame_count,
        'start_time': round(start_time, 4),
        'end_time': round(end_time, 4),
        'duration_sec': round(duration, 4),
    })

    return pd.DataFrame(segments)


# ════════════════════════════════════════════════════════════════════
# 5. STATISTIK SEGMEN
# ════════════════════════════════════════════════════════════════════

def calculate_segment_statistics(seg_df: pd.DataFrame) -> dict:
    """
    Menghitung statistik agregat dari hasil segmentasi.

    Metrik:
        - total_segments        : jumlah segmen
        - short_segments (<8 fr): jumlah segmen dengan frame_count < 8
        - pct_short_segments    : persentase segmen pendek
        - median_frame_count    : median panjang segmen (frame)
        - mean_frame_count      : rata-rata panjang segmen (frame)
        - min_frame_count       : segmen terpendek (frame)
        - max_frame_count       : segmen terpanjang (frame)
        - median_duration_sec   : median durasi segmen (detik)

    Returns:
        dict berisi semua statistik segmen.
    """
    counts = seg_df['frame_count'].values
    durations = seg_df['duration_sec'].values
    short_mask = counts < 8

    return {
        'total_segments': len(seg_df),
        'short_segments': int(short_mask.sum()),
        'pct_short_segments': round(short_mask.sum() / len(seg_df) * 100, 2) if len(seg_df) > 0 else 0.0,
        'median_frame_count': int(np.median(counts)),
        'mean_frame_count': round(np.mean(counts), 2),
        'min_frame_count': int(np.min(counts)),
        'max_frame_count': int(np.max(counts)),
        'median_duration_sec': round(np.median(durations), 2),
    }


# ════════════════════════════════════════════════════════════════════
# 6. STATISTIK NEUTRAL
# ════════════════════════════════════════════════════════════════════

def calculate_neutral_statistics(df: pd.DataFrame, stable_segs: pd.DataFrame) -> dict:
    """
    Menghitung statistik kemunculan status 'neutral' pada stable_label.

    neutral muncul ketika confidence model di bawah threshold, sehingga
    bukan merupakan salah satu dari 4 kelas emosi yang dilatih.

    Returns:
        dict berisi:
        - neutral_frames      : jumlah frame neutral
        - neutral_pct         : persentase terhadap seluruh data
        - neutral_segments    : jumlah segmen neutral
        - neutral_median_dur  : median durasi segmen neutral (detik)
    """
    total = len(df)
    n_neutral = int((df['stable_label'] == 'neutral').sum())
    neutral_segs = stable_segs[stable_segs['label'] == 'neutral']

    return {
        'neutral_frames': n_neutral,
        'neutral_pct': round(n_neutral / total * 100, 2) if total > 0 else 0.0,
        'neutral_segments': len(neutral_segs),
        'neutral_median_dur': round(neutral_segs['duration_sec'].median(), 2) if len(neutral_segs) > 0 else 0.0,
    }


# ════════════════════════════════════════════════════════════════════
# 7. VISUALISASI
# ════════════════════════════════════════════════════════════════════

def create_plots(df: pd.DataFrame, raw_changes: int, stable_changes: int,
                 raw_stats: dict, stable_stats: dict, output_dir: Path):
    """
    Membuat 4 file visualisasi PNG resolusi tinggi:
      1. label_changes_comparison.png   — Bar chart perubahan label
      2. short_segments_comparison.png  — Bar chart segmen <8 frame
      3. fps_over_time.png              — Line chart FPS vs waktu
      4. raw_vs_stable_timeline.png     — Dual panel timeline emosi
    """
    # Mapping emosi ke angka untuk timeline (konsisten)
    EMOTION_MAP = {
        'engaged': 4,
        'confused': 3,
        'frustrated': 2,
        'bored': 1,
        'neutral': 0,
    }
    EMOTION_LABELS = ['Neutral', 'Bored', 'Frustrated', 'Confused', 'Engaged']

    # ── Plot 1: Label Changes Comparison ─────────────────────────
    fig, ax = plt.subplots(figsize=(8, 5))
    categories = ['Prediksi Mentah\n(Raw)', 'Pasca-Pemrosesan\n(Stable)']
    values = [raw_changes, stable_changes]
    colors = ['#ef4444', '#22c55e']
    bars = ax.bar(categories, values, color=colors, width=0.5, edgecolor='white', linewidth=1.5)
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 5,
                str(val), ha='center', va='bottom', fontweight='bold', fontsize=14)
    ax.set_ylabel('Jumlah Perubahan Label', fontsize=12)
    ax.set_title('Perbandingan Jumlah Perubahan Label\n(Raw vs Post-Processing)', fontsize=14, fontweight='bold')
    ax.spines[['top', 'right']].set_visible(False)
    ax.set_ylim(0, max(values) * 1.2)
    plt.tight_layout()
    plt.savefig(output_dir / 'label_changes_comparison.png', dpi=300)
    plt.close()

    # ── Plot 2: Short Segments Comparison ────────────────────────
    fig, ax = plt.subplots(figsize=(8, 5))
    values_short = [raw_stats['short_segments'], stable_stats['short_segments']]
    bars = ax.bar(categories, values_short, color=['#f97316', '#3b82f6'], width=0.5,
                  edgecolor='white', linewidth=1.5)
    for bar, val in zip(bars, values_short):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 3,
                str(val), ha='center', va='bottom', fontweight='bold', fontsize=14)
    ax.set_ylabel('Jumlah Segmen < 8 Frame', fontsize=12)
    ax.set_title('Perbandingan Segmen Pendek (< 8 Frame)\n(Raw vs Post-Processing)', fontsize=14, fontweight='bold')
    ax.spines[['top', 'right']].set_visible(False)
    ax.set_ylim(0, max(values_short) * 1.2)
    plt.tight_layout()
    plt.savefig(output_dir / 'short_segments_comparison.png', dpi=300)
    plt.close()

    # ── Plot 3: FPS Over Time ────────────────────────────────────
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(df['timestamp_sec'], df['fps'], color='#6366f1', alpha=0.7, linewidth=0.8, label='FPS')
    ax.axhline(y=30, color='#ef4444', linestyle='--', linewidth=1.5, label='Target 30 FPS')
    ax.set_xlabel('Waktu (detik)', fontsize=12)
    ax.set_ylabel('FPS', fontsize=12)
    ax.set_title('FPS Selama Pengujian Real-Time', fontsize=14, fontweight='bold')
    ax.legend(loc='lower right', fontsize=11)
    ax.spines[['top', 'right']].set_visible(False)
    ax.set_ylim(0, df['fps'].quantile(0.99) * 1.3)
    plt.tight_layout()
    plt.savefig(output_dir / 'fps_over_time.png', dpi=300)
    plt.close()

    # ── Plot 4: Raw vs Stable Timeline ───────────────────────────
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 7), sharex=True)

    # Panel atas: raw_class_name
    raw_numeric = df['raw_class_name'].map(EMOTION_MAP).fillna(-1)
    ax1.step(df['timestamp_sec'], raw_numeric, where='post', color='#ef4444', linewidth=0.8)
    ax1.set_ylabel('Emosi', fontsize=11)
    ax1.set_title('Prediksi Mentah (raw_class_name)', fontsize=12, fontweight='bold')
    ax1.set_yticks(range(5))
    ax1.set_yticklabels(EMOTION_LABELS, fontsize=10)
    ax1.set_ylim(-0.5, 4.5)
    ax1.spines[['top', 'right']].set_visible(False)
    ax1.grid(axis='y', alpha=0.3)

    # Panel bawah: stable_label
    stable_numeric = df['stable_label'].map(EMOTION_MAP).fillna(-1)
    ax2.step(df['timestamp_sec'], stable_numeric, where='post', color='#22c55e', linewidth=0.8)
    ax2.set_ylabel('Emosi', fontsize=11)
    ax2.set_xlabel('Waktu (detik)', fontsize=11)
    ax2.set_title('Prediksi Stabil (stable_label)', fontsize=12, fontweight='bold')
    ax2.set_yticks(range(5))
    ax2.set_yticklabels(EMOTION_LABELS, fontsize=10)
    ax2.set_ylim(-0.5, 4.5)
    ax2.spines[['top', 'right']].set_visible(False)
    ax2.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'raw_vs_stable_timeline.png', dpi=300)
    plt.close()

    print(f"[INFO] 4 file visualisasi disimpan ke: {output_dir}")


# ════════════════════════════════════════════════════════════════════
# 8. SIMPAN RINGKASAN (CSV & JSON)
# ════════════════════════════════════════════════════════════════════

def save_summary(fps_stats: dict, raw_changes: int, stable_changes: int,
                 raw_seg_stats: dict, stable_seg_stats: dict,
                 neutral_stats: dict, output_dir: Path):
    """
    Menyimpan seluruh hasil analisis ke:
      - realtime_analysis_summary.csv  (format: metric, raw, postprocessed, change)
      - realtime_analysis_summary.json (format nested dict)
    """
    # Hitung persentase perubahan
    pct_change_labels = round((raw_changes - stable_changes) / raw_changes * 100, 2) if raw_changes > 0 else 0.0
    pct_change_short = round(
        (raw_seg_stats['short_segments'] - stable_seg_stats['short_segments'])
        / raw_seg_stats['short_segments'] * 100, 2
    ) if raw_seg_stats['short_segments'] > 0 else 0.0

    # ── CSV ──
    rows = [
        ('total_frames', fps_stats['total_frames'], fps_stats['total_frames'], '-'),
        ('test_duration_sec', fps_stats['test_duration_sec'], fps_stats['test_duration_sec'], '-'),
        ('mean_fps', fps_stats['mean_fps'], '-', '-'),
        ('median_fps', fps_stats['median_fps'], '-', '-'),
        ('p5_fps', fps_stats['p5_fps'], '-', '-'),
        ('p95_fps', fps_stats['p95_fps'], '-', '-'),
        ('pct_frames_ge_30', fps_stats['pct_frames_ge_30'], '-', '-'),
        ('label_changes', raw_changes, stable_changes, f"-{pct_change_labels}%"),
        ('total_segments', raw_seg_stats['total_segments'], stable_seg_stats['total_segments'], '-'),
        ('short_segments_lt8', raw_seg_stats['short_segments'], stable_seg_stats['short_segments'],
         f"-{pct_change_short}%"),
        ('pct_short_segments', raw_seg_stats['pct_short_segments'], stable_seg_stats['pct_short_segments'], '-'),
        ('median_frame_count', raw_seg_stats['median_frame_count'], stable_seg_stats['median_frame_count'], '-'),
        ('median_duration_sec', raw_seg_stats['median_duration_sec'], stable_seg_stats['median_duration_sec'], '-'),
        ('neutral_frames', '-', neutral_stats['neutral_frames'], '-'),
        ('neutral_pct', '-', neutral_stats['neutral_pct'], '-'),
        ('neutral_segments', '-', neutral_stats['neutral_segments'], '-'),
        ('neutral_median_dur', '-', neutral_stats['neutral_median_dur'], '-'),
    ]
    summary_csv = pd.DataFrame(rows, columns=['metric', 'raw', 'postprocessed', 'change'])
    csv_path = output_dir / 'realtime_analysis_summary.csv'
    summary_csv.to_csv(csv_path, index=False)

    # ── JSON ──
    summary_json = {
        'fps': fps_stats,
        'label_changes': {
            'raw': raw_changes,
            'stable': stable_changes,
            'reduction_pct': pct_change_labels,
        },
        'segments': {
            'raw': raw_seg_stats,
            'stable': stable_seg_stats,
            'short_segment_reduction_pct': pct_change_short,
        },
        'neutral': neutral_stats,
    }
    json_path = output_dir / 'realtime_analysis_summary.json'
    with open(json_path, 'w') as f:
        json.dump(summary_json, f, indent=2, ensure_ascii=False)

    return csv_path, json_path, pct_change_labels, pct_change_short


# ════════════════════════════════════════════════════════════════════
# 9. SANITY CHECKS (Assertions)
# ════════════════════════════════════════════════════════════════════

def run_sanity_checks(df, raw_changes, stable_changes, raw_segs, stable_segs,
                      raw_seg_stats, stable_seg_stats):
    """
    Melakukan beberapa assertion untuk memastikan konsistensi hasil perhitungan.
    Jika ada assertion yang gagal, skrip akan menampilkan error tetapi tetap berjalan.
    """
    print("\n" + "=" * 50)
    print("  SANITY CHECKS")
    print("=" * 50)

    checks_passed = 0
    checks_total = 0

    # Check 1: jumlah segmen = jumlah perubahan label + 1
    checks_total += 1
    expected_raw_segs = raw_changes + 1
    if len(raw_segs) == expected_raw_segs:
        print(f"  [PASS] Raw segments ({len(raw_segs)}) == raw label changes + 1 ({expected_raw_segs})")
        checks_passed += 1
    else:
        print(f"  [FAIL] Raw segments ({len(raw_segs)}) != raw label changes + 1 ({expected_raw_segs})")

    checks_total += 1
    expected_stable_segs = stable_changes + 1
    if len(stable_segs) == expected_stable_segs:
        print(f"  [PASS] Stable segments ({len(stable_segs)}) == stable label changes + 1 ({expected_stable_segs})")
        checks_passed += 1
    else:
        print(f"  [FAIL] Stable segments ({len(stable_segs)}) != stable label changes + 1 ({expected_stable_segs})")

    # Check 2: total frame pada seluruh segmen = jumlah frame
    checks_total += 1
    total_raw_frames = raw_segs['frame_count'].sum()
    if total_raw_frames == len(df):
        print(f"  [PASS] Total raw segment frames ({total_raw_frames}) == total data frames ({len(df)})")
        checks_passed += 1
    else:
        print(f"  [FAIL] Total raw segment frames ({total_raw_frames}) != total data frames ({len(df)})")

    checks_total += 1
    total_stable_frames = stable_segs['frame_count'].sum()
    if total_stable_frames == len(df):
        print(f"  [PASS] Total stable segment frames ({total_stable_frames}) == total data frames ({len(df)})")
        checks_passed += 1
    else:
        print(f"  [FAIL] Total stable segment frames ({total_stable_frames}) != total data frames ({len(df)})")

    # Check 3: persentase harus antara 0 dan 100
    checks_total += 1
    pcts = [
        raw_seg_stats['pct_short_segments'],
        stable_seg_stats['pct_short_segments'],
    ]
    all_valid = all(0 <= p <= 100 for p in pcts)
    if all_valid:
        print(f"  [PASS] Semua persentase berada dalam rentang 0-100")
        checks_passed += 1
    else:
        print(f"  [FAIL] Ada persentase di luar rentang 0-100: {pcts}")

    print(f"\n  Hasil: {checks_passed}/{checks_total} checks passed")
    print("=" * 50)


# ════════════════════════════════════════════════════════════════════
# 10. MAIN
# ════════════════════════════════════════════════════════════════════

def main():
    # ── Parse argumen ──
    parser = argparse.ArgumentParser(
        description="Analisis performa real-time & efektivitas temporal post-processing YOLOv13n"
    )
    parser.add_argument('csv_path', type=str, help="Path ke file realtime_prediction_log.csv")
    args = parser.parse_args()

    csv_path = Path(args.csv_path)
    if not csv_path.exists():
        print(f"[ERROR] File tidak ditemukan: {csv_path}")
        sys.exit(1)

    # Output disimpan di folder yang sama dengan file CSV
    output_dir = csv_path.parent
    print(f"[INFO] Output akan disimpan ke: {output_dir}\n")

    # ── 1. Load & Validasi ──
    df = load_and_validate_data(str(csv_path))

    # ── 2. Statistik FPS ──
    fps_stats = calculate_fps_statistics(df, warmup_frames=10)

    # ── 3. Perubahan Label ──
    raw_changes = count_label_changes(df['raw_class_name'])
    stable_changes = count_label_changes(df['stable_label'])

    # ── 4. Segmentasi Label ──
    raw_segs = extract_segments(df, 'raw_class_name')
    stable_segs = extract_segments(df, 'stable_label')

    # Simpan segmen detail ke CSV
    raw_segs.to_csv(output_dir / 'raw_segments.csv', index=False)
    stable_segs.to_csv(output_dir / 'stable_segments.csv', index=False)

    # ── 5. Statistik Segmen ──
    raw_seg_stats = calculate_segment_statistics(raw_segs)
    stable_seg_stats = calculate_segment_statistics(stable_segs)

    # ── 6. Statistik Neutral ──
    neutral_stats = calculate_neutral_statistics(df, stable_segs)

    # ── 7. Hitung persentase penurunan ──
    pct_label_reduction = round(
        (raw_changes - stable_changes) / raw_changes * 100, 2
    ) if raw_changes > 0 else 0.0
    pct_short_reduction = round(
        (raw_seg_stats['short_segments'] - stable_seg_stats['short_segments'])
        / raw_seg_stats['short_segments'] * 100, 2
    ) if raw_seg_stats['short_segments'] > 0 else 0.0
    pct_seg_reduction = round(
        (raw_seg_stats['total_segments'] - stable_seg_stats['total_segments'])
        / raw_seg_stats['total_segments'] * 100, 2
    ) if raw_seg_stats['total_segments'] > 0 else 0.0

    # ══════════════════════════════════════════════════════════════
    #  CETAK HASIL KE TERMINAL
    # ══════════════════════════════════════════════════════════════

    print("\n" + "=" * 50)
    print("  REAL-TIME PERFORMANCE")
    print("=" * 50)
    print(f"  Total frames               : {fps_stats['total_frames']}")
    print(f"  Test duration              : {fps_stats['test_duration_sec']} s")
    print(f"  FPS samples after warm-up  : {fps_stats['fps_samples']}")
    print(f"  Mean FPS                   : {fps_stats['mean_fps']}")
    print(f"  Median FPS                 : {fps_stats['median_fps']}")
    print(f"  Min FPS                    : {fps_stats['min_fps']}")
    print(f"  Max FPS                    : {fps_stats['max_fps']}")
    print(f"  P5 - P95                   : {fps_stats['p5_fps']} - {fps_stats['p95_fps']}")
    print(f"  Frames >= 30 FPS           : {fps_stats['frames_ge_30']} ({fps_stats['pct_frames_ge_30']}%)")

    print("\n" + "=" * 50)
    print("  POST-PROCESSING ANALYSIS")
    print("=" * 50)
    print(f"  Raw label changes          : {raw_changes}")
    print(f"  Stable label changes       : {stable_changes}")
    print(f"  Reduction                  : {pct_label_reduction}%")
    print()
    print(f"  Raw segments               : {raw_seg_stats['total_segments']}")
    print(f"  Stable segments            : {stable_seg_stats['total_segments']}")
    print(f"  Segment reduction          : {pct_seg_reduction}%")
    print()
    print(f"  Raw segments < 8 frames    : {raw_seg_stats['short_segments']} ({raw_seg_stats['pct_short_segments']}%)")
    print(f"  Stable segments < 8 frames : {stable_seg_stats['short_segments']} ({stable_seg_stats['pct_short_segments']}%)")
    print(f"  Short segment reduction    : {pct_short_reduction}%")
    print()
    print(f"  Median raw segment         : {raw_seg_stats['median_frame_count']} frames / {raw_seg_stats['median_duration_sec']} s")
    print(f"  Mean raw segment           : {raw_seg_stats['mean_frame_count']} frames")
    print(f"  Median stable segment      : {stable_seg_stats['median_frame_count']} frames / {stable_seg_stats['median_duration_sec']} s")
    print(f"  Mean stable segment        : {stable_seg_stats['mean_frame_count']} frames")
    print()
    print(f"  Neutral frames (stable)    : {neutral_stats['neutral_frames']}")
    print(f"  Neutral percentage         : {neutral_stats['neutral_pct']}%")
    print(f"  Neutral segments           : {neutral_stats['neutral_segments']}")
    print(f"  Neutral median duration    : {neutral_stats['neutral_median_dur']} s")

    # ── 8. Visualisasi ──
    create_plots(df, raw_changes, stable_changes, raw_seg_stats, stable_seg_stats, output_dir)

    # ── 9. Simpan ringkasan ──
    csv_out, json_out, _, _ = save_summary(
        fps_stats, raw_changes, stable_changes,
        raw_seg_stats, stable_seg_stats, neutral_stats, output_dir
    )

    # ── 10. Sanity Checks ──
    run_sanity_checks(df, raw_changes, stable_changes, raw_segs, stable_segs,
                      raw_seg_stats, stable_seg_stats)

    # ── 11. Tampilkan lokasi file output ──
    print("\n" + "=" * 50)
    print("  OUTPUT FILES")
    print("=" * 50)
    output_files = [
        csv_out,
        json_out,
        output_dir / 'raw_segments.csv',
        output_dir / 'stable_segments.csv',
        output_dir / 'label_changes_comparison.png',
        output_dir / 'short_segments_comparison.png',
        output_dir / 'fps_over_time.png',
        output_dir / 'raw_vs_stable_timeline.png',
    ]
    for f in output_files:
        print(f"  -> {f}")
    print("=" * 50)
    print("\n[SELESAI] Analisis berhasil diselesaikan.")


if __name__ == "__main__":
    main()
