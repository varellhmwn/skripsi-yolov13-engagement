import os
import sys
from pathlib import Path
from ultralytics import YOLO

BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_YAML = BASE_DIR / 'datasets' / 'master_combined_dataset' / 'data.yaml'
OUTPUT_DIR = BASE_DIR / 'runs'
RUN_NAME = 'yolov13n_retrained_scratch'

def main():
    print("=" * 60)
    print("  TAHAP 4: Training YOLOv13n pada Dataset Retrained Scratch")
    print("=" * 60)
    print(f"  Dataset YAML : {DATA_YAML}")
    print(f"  Output Dir   : {OUTPUT_DIR}/{RUN_NAME}")
    print(f"  Epochs       : 150")
    print(f"  Image Size   : 640")
    print("=" * 60)

    if not DATA_YAML.exists():
        print(f"[ERROR] Dataset YAML tidak ditemukan: {DATA_YAML}")
        print("        Pastikan Tahap 1, 2, dan 3 telah dijalankan.")
        sys.exit(1)

    # Inisialisasi model YOLOv13n dari bobot terbaik sebelumnya (V2)
    # Ini disebut Continual Learning / Active Learning
    base_weights = str(BASE_DIR / 'runs' / 'yolov13_master_combined_v2' / 'weights' / 'best.pt')
    print(f"[INFO] Memuat bobot model terbaik sebelumnya: {base_weights}")
    model = YOLO(base_weights)

    # Nama eksperimen baru
    RUN_NAME_V3 = 'yolov13_master_combined_v3'

    # Memulai Pelatihan Lanjutan (Fine-Tuning V3)
    results = model.train(
        data=str(DATA_YAML),
        epochs=100,  # Dikurangi karena model sudah cukup pintar
        imgsz=640,
        batch=16,
        patience=25,
        project=str(OUTPUT_DIR),
        name=RUN_NAME_V3,
        exist_ok=True,
        device=0,
        optimizer='AdamW',
        lr0=0.0001,  # Learning rate diturunkan drastis agar tidak merusak bobot V2 (Catastrophic Forgetting)
        lrf=0.01,
        weight_decay=0.0005,
        warmup_epochs=3,
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
        plots=True,
        save=True,
        val=True
    )

    print("\n" + "=" * 60)
    print("  PELATIHAN V3 SELESAI!")
    print("=" * 60)
    print(f"  Best Weights : {OUTPUT_DIR}/{RUN_NAME_V3}/weights/best.pt")
    print(f"  Results CSV  : {OUTPUT_DIR}/{RUN_NAME_V3}/results.csv")
    print("=" * 60 + "\n")

if __name__ == '__main__':
    main()
