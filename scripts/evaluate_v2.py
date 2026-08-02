"""
evaluate_v2.py — Kode Evaluasi Model YOLOv13n (v2) pada Data Pengujian
======================================================================
Menjalankan evaluasi model yolov13_master_combined_v2 pada test split (173 citra)
dan menyimpan seluruh gambar/kurva evaluasi ke runs/evaluation/yolov13_master_combined_v2_test
"""

from ultralytics import YOLO
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

def main():
    model_path = str(BASE_DIR / 'runs' / 'yolov13_master_combined_v2' / 'weights' / 'best.pt')
    data_yaml = str(BASE_DIR / 'datasets' / 'master_combined_dataset' / 'data.yaml')
    project_dir = str(BASE_DIR / 'runs' / 'evaluation')
    run_name = "yolov13_master_combined_v2_test"

    print("=" * 60)
    print("  EVALUASI MODEL YOLOv13n (V2) — Kode Program 4.2")
    print("=" * 60)
    print(f"  Model   : {model_path}")
    print(f"  Dataset : {data_yaml}")
    print(f"  Output  : {project_dir}/{run_name}")
    print()

    model = YOLO(model_path)

    test_results = model.val(
        data=data_yaml,
        split="test",
        imgsz=640,
        batch=16,
        device=0,
        plots=True,
        project=project_dir,
        name=run_name,
        exist_ok=True
    )

    print("\n" + "=" * 60)
    print("  EVALUASI SELESAI!")
    print(f"  Hasil tersimpan di: {project_dir}/{run_name}")
    print("=" * 60)

if __name__ == '__main__':
    main()
