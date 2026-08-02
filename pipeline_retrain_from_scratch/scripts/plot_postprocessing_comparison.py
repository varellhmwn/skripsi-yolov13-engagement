"""
plot_postprocessing_comparison.py — Generate Gambar 4.12
=========================================================
Membuat grafik perbandingan 2-panel (Raw vs Stable Post-Processing Timeline)
beresolusi tinggi (300 DPI) untuk Bab 4.
"""

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

BASE_DIR = Path(__file__).resolve().parent.parent.parent
CSV_PATH = BASE_DIR / 'outputs' / 'realtime_smoothed' / 'realtime_prediction_log.csv'
OUTPUT_PATH = BASE_DIR / 'pipeline_retrain_from_scratch' / 'gambar_4_12_perbandingan_postprocessing.png'


def main():
    if not CSV_PATH.exists():
        print(f"[ERROR] File log tidak ditemukan: {CSV_PATH}")
        return

    df = pd.read_csv(CSV_PATH)
    emotion_map = {'engaged': 0, 'confused': 1, 'bored': 2, 'frustrated': 3, 'neutral': 4, 'no_face': 4}
    emotion_labels = ['Engaged', 'Confused', 'Bored', 'Frustrated', 'Neutral']
    colors = {'engaged': '#2ecc71', 'confused': '#e67e22', 'bored': '#3498db', 'frustrated': '#e74c3c', 'neutral': '#95a5a6'}

    time = df['timestamp_sec']
    raw = df['raw_class_name'].map(lambda x: emotion_map.get(str(x), 4))
    stable = df['stable_label'].map(lambda x: emotion_map.get(str(x), 4))

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6), sharex=True, dpi=300)

    # (a) Raw Timeline
    ax1.plot(time, raw, color='#7f8c8d', alpha=0.5, linewidth=1)
    for em_name in colors:
        mask = (df['raw_class_name'] == em_name)
        ax1.scatter(time[mask], raw[mask], color=colors[em_name], s=12)

    ax1.set_yticks(range(5))
    ax1.set_yticklabels(emotion_labels, fontsize=10, fontweight='bold')
    ax1.set_title('(a) Prediksi Mentah YOLOv13n (Tinggi Fluktuasi)', fontsize=11, fontweight='bold', pad=10)
    ax1.grid(True, linestyle='--', alpha=0.4)

    # (b) Stable Timeline
    ax2.plot(time, stable, color='#2c3e50', linewidth=1.8)
    for em_name in colors:
        mask = (df['stable_label'] == em_name)
        ax2.scatter(time[mask], stable[mask], color=colors[em_name], s=16)

    ax2.set_yticks(range(5))
    ax2.set_yticklabels(emotion_labels, fontsize=10, fontweight='bold')
    ax2.set_title('(b) Hasil Setelah Pasca-Pemrosesan (Sliding Window 30 Frame & Majority Voting)', fontsize=11, fontweight='bold', pad=10)
    ax2.set_xlabel('Waktu (detik)', fontsize=10, fontweight='bold', labelpad=8)
    ax2.grid(True, linestyle='--', alpha=0.4)

    plt.suptitle('Gambar 4.12 Perbandingan Perubahan Label Prediksi Mentah dan Hasil Pasca-Pemrosesan', fontsize=12, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    plt.savefig(OUTPUT_PATH, dpi=300)
    print(f"[SUCCESS] Gambar 4.12 berhasil disimpan di: {OUTPUT_PATH}")


if __name__ == '__main__':
    main()
