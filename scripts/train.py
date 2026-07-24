"""
train.py — Training YOLOv13n pada Master Combined Dataset
==========================================================
Melatih model YOLOv13n untuk mendeteksi 4 kelas emosi mahasiswa:
  - engaged (0), confused (1), bored (2), frustrated (3)

Dataset: Master Combined Dataset (1.698 gambar)
  - Sumber: DAiSEE finetuned + Big-Data + Hard Samples
  - Split: 80% Train / 10% Val / 10% Test

Penggunaan:
    python scripts/train.py
"""

import sys
from pathlib import Path
from ultralytics import YOLO

# ─── Konfigurasi Path ───────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent        # skripsi_yolov13_engagement/
DATA_YAML = str(BASE_DIR / 'datasets' / 'master_combined_dataset' / 'data.yaml')
OUTPUT_DIR = str(BASE_DIR / 'runs')
RUN_NAME = 'yolov13_master_combined_v2'


def main():
    print("=" * 60)
    print("  TRAINING YOLOv13n — Master Combined 4-Class")
    print("  Student Engagement Detection")
    print("=" * 60)
    print(f"  Dataset : {DATA_YAML}")
    print(f"  Output  : {OUTPUT_DIR}/{RUN_NAME}")
    print(f"  Epochs  : 150")
    print(f"  ImgSize : 640")
    print()

    # Validasi dataset
    if not Path(DATA_YAML).exists():
        print(f"[ERROR] Dataset tidak ditemukan: {DATA_YAML}")
        print("        Pastikan folder datasets/master_combined_dataset/ ada.")
        sys.exit(1)

    # Cek ketersediaan weights (apakah fine-tune dari model terdahulu atau dari yolov13n.pt)
    prev_best = BASE_DIR / 'runs' / 'yolov13_master_combined' / 'weights' / 'best.pt'
    if prev_best.exists():
        print(f"[INFO] Fine-tuning dari bobot terbaik sebelumnya: {prev_best}")
        model_weights = str(prev_best)
    else:
        print("[INFO] Loading pretrained YOLOv13n base weights...")
        model_weights = 'yolov13n.pt'

    model = YOLO(model_weights)

    # Training
    results = model.train(
        data=DATA_YAML,
        epochs=150,
        imgsz=640,
        batch=16,
        patience=25,
        project=OUTPUT_DIR,
        name=RUN_NAME,
        exist_ok=True,
        device=0,
        # Optimizer
        optimizer='AdamW',
        lr0=0.001,
        lrf=0.01,
        weight_decay=0.0005,
        warmup_epochs=3,
        # Augmentasi
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=10.0,
        translate=0.1,
        scale=0.5,
        fliplr=0.5,
        mosaic=1.0,
        mixup=0.1,
        close_mosaic=10,
        # Output
        plots=True,
        save=True,
        val=True,
    )

    print(f"\n{'=' * 60}")
    print("  TRAINING COMPLETE!")
    print(f"{'=' * 60}")
    print(f"  Best weights : {OUTPUT_DIR}/{RUN_NAME}/weights/best.pt")
    print(f"  Results      : {OUTPUT_DIR}/{RUN_NAME}/results.csv")


if __name__ == '__main__':
    main()
