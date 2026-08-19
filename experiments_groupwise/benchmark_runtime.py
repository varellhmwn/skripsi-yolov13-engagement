"""
benchmark_runtime.py — Benchmarking Waktu Presisi Tinggi (Group-Wise Test Set)
================================================================================
Mengukur waktu inferensi per-citra secara terstandarisasi dengan metodologi identik:
  - Timer: time.perf_counter()
  - Warmup: 20 iterasi per model
  - Evaluasi single-image pada seluruh 166 citra test set group-wise
Output:
  - outputs_groupwise/runtime_raw.csv
  - outputs_groupwise/runtime_summary.csv
"""

import sys
import time
from pathlib import Path
import cv2
import numpy as np
import pandas as pd
import torch
from sklearn.neighbors import KNeighborsClassifier
from ultralytics import YOLO

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from experiments_groupwise.config import (
    TRAINED_GROUPWISE_WEIGHTS, GROUPWISE_DATASET_DIR, OUTPUT_GROUPWISE_DIR,
    YOLO_IMGSZ, YOLO_CONF_THRESHOLD, BENCHMARK_DEVICE, BENCHMARK_WARMUP_ROUNDS,
    KNN_METRIC, VALID_IMG_EXTS
)
from experiments_groupwise.hog_features import (
    parse_yolo_annotation, crop_face_from_bbox, extract_hog_features,
    load_dataset_split
)
from experiments_groupwise.tune_knn import run_knn_tuning_groupwise


def run_runtime_benchmark_groupwise(best_k=None, X_train=None, y_train=None):
    print("=" * 65)
    print("  TAHAP 12: BENCHMARKING RUNTIME PRESISI TINGGI (GROUP-WISE)")
    print("=" * 65)
    print(f"  Warmup Rounds : {BENCHMARK_WARMUP_ROUNDS} iterasi")
    print(f"  Device        : GPU CUDA:0 ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

    OUTPUT_GROUPWISE_DIR.mkdir(parents=True, exist_ok=True)

    if best_k is None or X_train is None or y_train is None:
        best_k, _, X_train, y_train = run_knn_tuning_groupwise()

    if not TRAINED_GROUPWISE_WEIGHTS.exists():
        raise FileNotFoundError(f"Weights YOLO group-wise belum tersedia: {TRAINED_GROUPWISE_WEIGHTS}")

    # 1. Melatih model KNN
    print(f"\n[1/4] Training KNN (K={best_k})...")
    knn = KNeighborsClassifier(n_neighbors=best_k, metric=KNN_METRIC)
    knn.fit(X_train, y_train)

    # 2. Loading YOLO model
    print(f"\n[2/4] Loading YOLO model: {TRAINED_GROUPWISE_WEIGHTS}...")
    yolo_model = YOLO(str(TRAINED_GROUPWISE_WEIGHTS))

    # 3. Pre-loading test images into RAM
    print("\n[3/4] Pre-loading test images into RAM (eliminasi disk I/O latency)...")
    test_images_dir = GROUPWISE_DATASET_DIR / 'images' / 'test'
    test_labels_dir = GROUPWISE_DATASET_DIR / 'labels' / 'test'

    img_files = sorted([
        f for f in test_images_dir.iterdir()
        if f.suffix.lower() in VALID_IMG_EXTS
    ])

    test_samples = []
    for img_p in img_files:
        img_bgr = cv2.imread(str(img_p))
        if img_bgr is None:
            continue
        h, w = img_bgr.shape[:2]
        lbl_p = test_labels_dir / f"{img_p.stem}.txt"
        anns = parse_yolo_annotation(lbl_p, w, h)
        gt_bbox = (anns[0][1], anns[0][2], anns[0][3], anns[0][4]) if anns else (0, 0, w, h)
        test_samples.append({
            'filename': img_p.name,
            'image': img_bgr,
            'gt_bbox': gt_bbox
        })

    print(f"      {len(test_samples)} citra berhasil dimuat ke memory.")

    # 4. WARMUP PHASE
    print(f"\n[4/4] Menjalankan Warm-up ({BENCHMARK_WARMUP_ROUNDS} iterasi)...")
    dummy_img = test_samples[0]['image']
    dummy_crop = dummy_img[10:100, 10:100]
    dummy_feat = extract_hog_features(dummy_crop).reshape(1, -1)

    for _ in range(BENCHMARK_WARMUP_ROUNDS):
        _ = yolo_model.predict(dummy_img, imgsz=YOLO_IMGSZ, conf=YOLO_CONF_THRESHOLD, device=BENCHMARK_DEVICE, verbose=False)
        _ = extract_hog_features(dummy_crop)
        _ = knn.predict(dummy_feat)

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    # BENCHMARK EXECUTION
    print("\n  Memulai pengukuran per-citra secara individual...")
    raw_records = []

    for sample in test_samples:
        img = sample['image']
        fname = sample['filename']
        gt_x1, gt_y1, gt_x2, gt_y2 = sample['gt_bbox']

        # --- A. YOLO Measurement ---
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t_yolo_start = time.perf_counter()
        results = yolo_model.predict(img, imgsz=YOLO_IMGSZ, conf=YOLO_CONF_THRESHOLD, device=BENCHMARK_DEVICE, verbose=False)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t_yolo_total = (time.perf_counter() - t_yolo_start) * 1000 # ms

        speed_dict = getattr(results[0], 'speed', {})
        yolo_native_infer_ms = float(speed_dict.get('inference', 0.0))
        yolo_native_pre_ms = float(speed_dict.get('preprocess', 0.0))
        yolo_native_post_ms = float(speed_dict.get('postprocess', 0.0))

        # --- B. HOG-KNN GT Pipeline ---
        t_crop_start = time.perf_counter()
        gt_crop = crop_face_from_bbox(img, gt_x1, gt_y1, gt_x2, gt_y2)
        if gt_crop is None:
            gt_crop = img
        t_gt_crop = (time.perf_counter() - t_crop_start) * 1000

        t_hog_start = time.perf_counter()
        gt_feat = extract_hog_features(gt_crop)
        t_gt_hog = (time.perf_counter() - t_hog_start) * 1000

        t_knn_start = time.perf_counter()
        _ = knn.predict(gt_feat.reshape(1, -1))
        t_gt_knn = (time.perf_counter() - t_knn_start) * 1000

        t_gt_total_pipeline = t_gt_crop + t_gt_hog + t_gt_knn

        # --- C. Hybrid Pipeline ---
        t_hyb_crop_start = time.perf_counter()
        det = results[0].boxes
        if len(det) > 0:
            largest_area = 0
            best_idx = 0
            for i in range(len(det)):
                xyxy = det.xyxy[i].cpu().numpy()
                area = (xyxy[2] - xyxy[0]) * (xyxy[3] - xyxy[1])
                if area > largest_area:
                    largest_area = area
                    best_idx = i
            xyxy = det.xyxy[best_idx].cpu().numpy()
            hyb_crop = crop_face_from_bbox(img, xyxy[0], xyxy[1], xyxy[2], xyxy[3])
            if hyb_crop is None:
                hyb_crop = img
        else:
            hyb_crop = img
        t_hyb_crop = (time.perf_counter() - t_hyb_crop_start) * 1000

        t_hyb_hog_start = time.perf_counter()
        hyb_feat = extract_hog_features(hyb_crop)
        t_hyb_hog = (time.perf_counter() - t_hyb_hog_start) * 1000

        t_hyb_knn_start = time.perf_counter()
        _ = knn.predict(hyb_feat.reshape(1, -1))
        t_hyb_knn = (time.perf_counter() - t_hyb_knn_start) * 1000

        t_hyb_total_pipeline = t_yolo_total + t_hyb_crop + t_hyb_hog + t_hyb_knn

        raw_records.append({
            'filename': fname,
            'yolo_wall_clock_ms': t_yolo_total,
            'yolo_native_infer_ms': yolo_native_infer_ms,
            'yolo_native_pre_ms': yolo_native_pre_ms,
            'yolo_native_post_ms': yolo_native_post_ms,
            'knn_gt_crop_ms': t_gt_crop,
            'knn_gt_hog_ms': t_gt_hog,
            'knn_gt_predict_ms': t_gt_knn,
            'knn_gt_total_pipeline_ms': t_gt_total_pipeline,
            'hybrid_yolo_det_ms': t_yolo_total,
            'hybrid_crop_ms': t_hyb_crop,
            'hybrid_hog_ms': t_hyb_hog,
            'hybrid_knn_predict_ms': t_hyb_knn,
            'hybrid_total_pipeline_ms': t_hyb_total_pipeline
        })

    df_raw = pd.DataFrame(raw_records)
    df_raw.to_csv(OUTPUT_GROUPWISE_DIR / 'runtime_raw.csv', index=False)

    def calc_stats(series):
        arr = series.values
        m = float(np.mean(arr))
        return {
            'mean': m,
            'median': float(np.median(arr)),
            'std': float(np.std(arr)),
            'p5': float(np.percentile(arr, 5)),
            'p95': float(np.percentile(arr, 95)),
            'fps': float(1000.0 / m) if m > 0 else 0.0
        }

    summary_rows = [
        {'Pipeline / Komponen': 'YOLOv13n (Native Inference Only)', **calc_stats(df_raw['yolo_native_infer_ms'])},
        {'Pipeline / Komponen': 'YOLOv13n (Total Wall-Clock Pipeline)', **calc_stats(df_raw['yolo_wall_clock_ms'])},
        {'Pipeline / Komponen': 'HOG-KNN GT: Crop Wajah', **calc_stats(df_raw['knn_gt_crop_ms'])},
        {'Pipeline / Komponen': 'HOG-KNN GT: HOG Extraction (Resize+Gray+HOG)', **calc_stats(df_raw['knn_gt_hog_ms'])},
        {'Pipeline / Komponen': 'HOG-KNN GT: KNN Predict (Single Sample)', **calc_stats(df_raw['knn_gt_predict_ms'])},
        {'Pipeline / Komponen': 'HOG-KNN GT: Total Pipeline', **calc_stats(df_raw['knn_gt_total_pipeline_ms'])},
        {'Pipeline / Komponen': 'YOLO-HOG-KNN: YOLO Detection Stage', **calc_stats(df_raw['hybrid_yolo_det_ms'])},
        {'Pipeline / Komponen': 'YOLO-HOG-KNN: Crop Stage', **calc_stats(df_raw['hybrid_crop_ms'])},
        {'Pipeline / Komponen': 'YOLO-HOG-KNN: HOG Extraction Stage', **calc_stats(df_raw['hybrid_hog_ms'])},
        {'Pipeline / Komponen': 'YOLO-HOG-KNN: KNN Predict Stage', **calc_stats(df_raw['hybrid_knn_predict_ms'])},
        {'Pipeline / Komponen': 'YOLO-HOG-KNN: Total Hybrid Pipeline', **calc_stats(df_raw['hybrid_total_pipeline_ms'])},
    ]

    df_summary = pd.DataFrame(summary_rows)
    df_summary.to_csv(OUTPUT_GROUPWISE_DIR / 'runtime_summary.csv', index=False)

    print("\n" + "=" * 80)
    print("  RINGKASAN BENCHMARK RUNTIME (GROUP-WISE)")
    print("=" * 80)
    print(df_summary.to_string(index=False))
    print("=" * 80)

    return df_summary, df_raw


if __name__ == '__main__':
    run_runtime_benchmark_groupwise()
