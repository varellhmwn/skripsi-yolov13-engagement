"""
evaluate.py — Evaluasi Model YOLOv13n pada Test Set
====================================================
Menjalankan evaluasi model pada test split dari Master Combined Dataset.
Menampilkan metrik: Precision, Recall, mAP@50, mAP@50-95 per kelas.

Penggunaan:
    python scripts/evaluate.py
    python scripts/evaluate.py --weights runs/yolov13_master_combined/weights/best.pt
    python scripts/evaluate.py --split test
"""

import argparse
import sys
from pathlib import Path
from ultralytics import YOLO


# ─── Konfigurasi Default ────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_WEIGHTS = str(BASE_DIR / 'runs' / 'yolov13_master_combined' / 'weights' / 'best.pt')
DEFAULT_DATA = str(BASE_DIR / 'datasets' / 'master_combined_dataset' / 'data.yaml')

CLASS_NAMES = {0: 'engaged', 1: 'confused', 2: 'bored', 3: 'frustrated'}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluasi model YOLOv13n pada test/val set"
    )
    parser.add_argument(
        '--weights', type=str, default=DEFAULT_WEIGHTS,
        help=f"Path ke model weights (default: {DEFAULT_WEIGHTS})"
    )
    parser.add_argument(
        '--data', type=str, default=DEFAULT_DATA,
        help=f"Path ke data.yaml (default: {DEFAULT_DATA})"
    )
    parser.add_argument(
        '--split', type=str, default='test', choices=['val', 'test'],
        help="Split dataset untuk evaluasi (default: test)"
    )
    parser.add_argument(
        '--imgsz', type=int, default=640,
        help="Ukuran input gambar (default: 640)"
    )
    parser.add_argument(
        '--device', type=str, default='0',
        help="Device untuk inference (default: 0 = GPU)"
    )
    parser.add_argument(
        '--conf', type=float, default=0.25,
        help="Confidence threshold (default: 0.25)"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    print("=" * 60)
    print("  EVALUASI MODEL YOLOv13n")
    print("  Student Engagement Detection — 4 Class")
    print("=" * 60)
    print(f"  Weights : {args.weights}")
    print(f"  Dataset : {args.data}")
    print(f"  Split   : {args.split}")
    print(f"  ImgSize : {args.imgsz}")
    print(f"  Device  : {args.device}")
    print()

    # Validasi file
    if not Path(args.weights).exists():
        print(f"[ERROR] Weights tidak ditemukan: {args.weights}")
        sys.exit(1)
    if not Path(args.data).exists():
        print(f"[ERROR] data.yaml tidak ditemukan: {args.data}")
        sys.exit(1)

    # Load model
    print("[INFO] Loading model...")
    model = YOLO(args.weights)

    # Evaluasi
    print(f"[INFO] Menjalankan evaluasi pada split '{args.split}'...")
    results = model.val(
        data=args.data,
        split=args.split,
        imgsz=args.imgsz,
        device=args.device,
        conf=args.conf,
        plots=True,
        save_json=False,
    )

    # Tampilkan hasil
    print(f"\n{'=' * 60}")
    print("  HASIL EVALUASI")
    print(f"{'=' * 60}")
    print(f"  {'Metrik':<20} {'Nilai':>10}")
    print(f"  {'-' * 32}")
    print(f"  {'Precision':<20} {results.box.mp * 100:>9.2f}%")
    print(f"  {'Recall':<20} {results.box.mr * 100:>9.2f}%")
    print(f"  {'mAP@50':<20} {results.box.map50 * 100:>9.2f}%")
    print(f"  {'mAP@50-95':<20} {results.box.map * 100:>9.2f}%")

    # Per-kelas
    print(f"\n  {'Kelas':<15} {'Precision':>10} {'Recall':>10} {'mAP@50':>10}")
    print(f"  {'-' * 47}")
    for i, name in CLASS_NAMES.items():
        if i < len(results.box.p):
            p = results.box.p[i] * 100
            r = results.box.r[i] * 100
            ap50 = results.box.ap50[i] * 100
            print(f"  {name:<15} {p:>9.2f}% {r:>9.2f}% {ap50:>9.2f}%")

    print(f"\n{'=' * 60}")
    print("  EVALUASI SELESAI!")
    print(f"{'=' * 60}")


if __name__ == '__main__':
    main()
