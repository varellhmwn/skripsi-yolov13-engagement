"""
plot_class_distribution.py — Membuat Grafik Distribusi Citra (Gambar 4.2)
========================================================================
Menghitung jumlah citra per kelas pada Master Combined Dataset (1.660 citra)
dan menyimpan grafik batang beresolusi tinggi (300 DPI) untuk Bab 4.
"""

from pathlib import Path
import matplotlib.pyplot as plt

# ─── Data Distribusi Kelas ──────────────────────────────────────
CLASSES = ['Engaged', 'Confused', 'Bored', 'Frustrated']
COUNTS = [548, 365, 364, 383]
COLORS = ['#2ecc71', '#e67e22', '#3498db', '#e74c3c']

BASE_DIR = Path(__file__).resolve().parent.parent
OUTPUT_PATH = BASE_DIR / 'gambar_4_2_distribusi_kelas.png'


def main():
    plt.figure(figsize=(8, 5), dpi=300)
    bars = plt.bar(CLASSES, COUNTS, color=COLORS, width=0.55, edgecolor='black', linewidth=1.2)

    plt.title('Distribusi Citra Berdasarkan Kelas pada Master Combined Dataset', fontsize=12, fontweight='bold', pad=15)
    plt.xlabel('Kategori Kelas Afektif', fontsize=10, fontweight='bold', labelpad=10)
    plt.ylabel('Jumlah Citra', fontsize=10, fontweight='bold', labelpad=10)
    plt.ylim(0, 650)
    plt.grid(axis='y', linestyle='--', alpha=0.5)

    # Tambahkan angka di atas setiap batang
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2.0, yval + 12, f'{yval:,}', ha='center', va='bottom', fontsize=11, fontweight='bold')

    plt.tight_layout()
    plt.savefig(OUTPUT_PATH, dpi=300)
    print(f"[INFO] Grafik berhasil disimpan di: {OUTPUT_PATH}")


if __name__ == '__main__':
    main()
