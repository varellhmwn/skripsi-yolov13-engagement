"""
realtime_predict.py — Real-Time Inference via Webcam + Manual Hard Sample Collector
=====================================================================================
Mendeteksi emosi mahasiswa secara real-time menggunakan webcam.
Model YOLOv13n mendeteksi 4 kelas (engaged, confused, bored, frustrated),
ditambah label 'neutral' dari mekanisme Confidence Thresholding.

Fitur:
  - Sliding Window Smoothing (mengurangi flicker antar-frame)
  - Confidence Thresholding (wajah datar → neutral)
  - Bounding box berwarna per emosi
  - FPS counter & legend
  - [BARU] Manual Hard Sample Collector: tekan hotkey angka untuk
    menyimpan frame + label YOLO ke folder kelas yang dipilih.

Penggunaan:
    python scripts/realtime_predict.py
    python scripts/realtime_predict.py --capture_hard_samples
    python scripts/realtime_predict.py --capture_hard_samples --save_csv

Hotkeys (saat jendela webcam aktif):
    q  = Keluar
    s  = Screenshot (frame + anotasi)
    0  = Simpan frame sebagai 'engaged'
    1  = Simpan frame sebagai 'confused'
    2  = Simpan frame sebagai 'bored'
    3  = Simpan frame sebagai 'frustrated'
    c  = Simpan frame sebagai kelas stabil saat ini (auto-label)
"""

import cv2
import argparse
import time
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from collections import deque, Counter
from ultralytics import YOLO


# ─── Konfigurasi Default ────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_WEIGHTS = str(BASE_DIR / 'runs' /
                      'yolov13_master_combined_v3' / 'weights' / 'best.pt')
DEFAULT_OUTPUT_DIR = str(BASE_DIR / 'outputs' / 'realtime_smoothed')

TARGET_CLASSES = {0: 'engaged', 1: 'confused', 2: 'bored', 3: 'frustrated'}
CLASS_NAME_TO_ID = {v: k for k, v in TARGET_CLASSES.items()}

# Warna per kelas (BGR format untuk OpenCV)
CLASS_COLORS = {
    'engaged':    (0, 255, 0),      # Hijau
    'confused':   (255, 0, 0),      # Biru
    'bored':      (0, 165, 255),    # Oranye
    'frustrated': (0, 0, 255),      # Merah
    'neutral':    (200, 200, 200),   # Abu-abu terang
    'no_face':    (128, 128, 128),   # Abu-abu gelap
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Real-Time Student Engagement Detection — YOLOv13n"
    )
    parser.add_argument('--weights', type=str, default=DEFAULT_WEIGHTS,
                        help="Path ke model weights")
    parser.add_argument('--source', type=str, default='0',
                        help="0 untuk webcam, atau path ke video file")
    parser.add_argument('--imgsz', type=int, default=640,
                        help="Ukuran input gambar (default: 640)")
    parser.add_argument('--conf', type=float, default=0.20,
                        help="Minimum confidence YOLO (default: 0.20)")
    parser.add_argument('--device', type=str, default='0',
                        help="Device inference (default: 0 = GPU)")
    parser.add_argument('--window_size', type=int, default=30,
                        help="Ukuran sliding window untuk smoothing (default: 30)")
    parser.add_argument('--min_vote_ratio', type=float, default=0.40,
                        help="Rasio voting minimum (default: 0.40)")
    parser.add_argument('--min_avg_confidence', type=float, default=0.40,
                        help="Threshold confidence untuk neutral trick (default: 0.40)")
    parser.add_argument('--show_raw', action='store_true',
                        help="Tampilkan prediksi mentah di samping prediksi stabil")

    # ─── Output & Hard Sample Collector ──────────────────────
    parser.add_argument('--output_dir', type=str, default=DEFAULT_OUTPUT_DIR,
                        help="Direktori output untuk screenshots, CSV, dan hard samples")
    parser.add_argument('--save_csv', action='store_true',
                        help="Simpan CSV log prediksi setiap frame")
    parser.add_argument('--save_video', action='store_true',
                        help="Simpan video hasil anotasi")
    parser.add_argument('--capture_hard_samples', action='store_true',
                        help="Aktifkan mode pengumpulan hard samples manual via hotkey")

    return parser.parse_args()


def setup_hard_samples_dirs(out_dir):
    """Buat struktur folder hard_samples/{kelas}/ dan screenshots/."""
    hs_dir = Path(out_dir) / 'hard_samples'
    for cls_name in TARGET_CLASSES.values():
        (hs_dir / cls_name).mkdir(parents=True, exist_ok=True)
    (Path(out_dir) / 'screenshots').mkdir(parents=True, exist_ok=True)
    return hs_dir


def save_hard_sample(frame, bbox, cls_name, cls_id, frame_idx, hs_dir):
    """Simpan citra mentah + label YOLO ke folder kelas yang dipilih."""
    # Simpan citra (frame mentah tanpa anotasi, agar bersih untuk training)
    img_path = hs_dir / cls_name / f"{cls_name}_{frame_idx}.jpg"
    cv2.imwrite(str(img_path), frame)

    # Simpan label YOLO (class_id x_center y_center width height)
    lbl_path = hs_dir / cls_name / f"{cls_name}_{frame_idx}.txt"
    if bbox is not None:
        ih, iw = frame.shape[:2]
        x1, y1, x2, y2 = bbox
        xc = ((x1 + x2) / 2.0) / iw
        yc = ((y1 + y2) / 2.0) / ih
        bw = (x2 - x1) / iw
        bh = (y2 - y1) / ih
        with open(lbl_path, 'w') as f:
            f.write(f"{cls_id} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}\n")
    else:
        # Fallback: full face crop jika tidak ada bbox
        with open(lbl_path, 'w') as f:
            f.write(f"{cls_id} 0.500000 0.500000 0.950000 0.950000\n")

    print(f"  [SAVED] {cls_name} -> {img_path.name} + {lbl_path.name}")
    return str(img_path)


def main():
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Validasi weights
    if not Path(args.weights).exists():
        print(f"[ERROR] Weights tidak ditemukan: {args.weights}")
        sys.exit(1)

    # Setup hard samples dirs
    hs_dir = None
    if args.capture_hard_samples:
        hs_dir = setup_hard_samples_dirs(out_dir)
    else:
        (out_dir / 'screenshots').mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("  REAL-TIME STUDENT ENGAGEMENT DETECTION")
    print("  YOLOv13n — 4 Class + Neutral Trick")
    print("=" * 60)
    print(f"  Weights              : {args.weights}")
    print(
        f"  Source               : {'Webcam' if args.source == '0' else args.source}")
    print(f"  Smoothing Window     : {args.window_size} frames")
    print(f"  Confidence Threshold : {args.min_avg_confidence}")
    print(
        f"  Classes              : {list(TARGET_CLASSES.values())} + neutral")
    print(
        f"  Hard Sample Mode     : {'AKTIF (manual hotkey)' if args.capture_hard_samples else 'OFF'}")
    print(f"  Save CSV Log         : {'YA' if args.save_csv else 'TIDAK'}")
    print(f"  Output Dir           : {out_dir}")
    print()

    if args.capture_hard_samples:
        print("  ╔══════════════════════════════════════════════╗")
        print("  ║  HOTKEYS HARD SAMPLE COLLECTOR               ║")
        print("  ║  0 = engaged  │  1 = confused                ║")
        print("  ║  2 = bored    │  3 = frustrated              ║")
        print("  ║  c = auto (gunakan label stabil saat ini)    ║")
        print("  ║  s = screenshot  │  q = keluar               ║")
        print("  ╚══════════════════════════════════════════════╝")
        print()

    # Load model
    print("[INFO] Loading model...")
    model = YOLO(args.weights)

    # Open video source
    source = int(args.source) if args.source.isdigit() else args.source
    cap = cv2.VideoCapture(source)

    if not cap.isOpened():
        print(f"[ERROR] Tidak dapat membuka sumber video: {args.source}")
        sys.exit(1)

    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cam_fps = int(cap.get(cv2.CAP_PROP_FPS))
    if cam_fps == 0 or cam_fps > 100:
        cam_fps = 30

    # Video writer (opsional)
    video_writer = None
    if args.save_video:
        video_path = str(out_dir / 'annotated_video.mp4')
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(video_path, fourcc, cam_fps,
                                       (frame_w, frame_h))

    # Sliding window untuk temporal smoothing
    window = deque(maxlen=args.window_size)

    # Statistik
    frame_idx = 0
    hard_sample_count = 0
    frame_logs = []
    total_no_face = 0
    total_neutral = 0
    raw_dist = Counter()
    stable_dist = Counter()

    print("[INFO] Real-time inference dimulai. Tekan 'q' untuk keluar.\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_idx += 1
        t0 = time.time()

        # ─── Inference ───────────────────────────────────────
        results = model.predict(frame, imgsz=args.imgsz, conf=args.conf,
                                device=args.device, verbose=False)
        det = results[0].boxes

        stable_label = "no_face"
        raw_label = None
        raw_conf = 0.0
        bbox = None
        vote_ratio = 0.0
        avg_conf = 0.0

        if len(det) > 0:
            # Pilih deteksi wajah terbesar (terdekat)
            frame_area = frame_w * frame_h
            largest_area = 0
            best_det = None

            for i in range(len(det)):
                xyxy = det.xyxy[i].cpu().numpy()
                w = xyxy[2] - xyxy[0]
                h = xyxy[3] - xyxy[1]
                area = w * h
                if (area / frame_area) >= 0.02 and area > largest_area:
                    largest_area = area
                    best_det = (int(det.cls[i].item()),
                                float(det.conf[i].item()), xyxy)

            if best_det is not None:
                cls_id, conf, bbox = best_det
                raw_label = TARGET_CLASSES.get(cls_id, "unknown")
                raw_conf = conf
                raw_dist[raw_label] += 1

                window.append({'class_id': cls_id, 'conf': conf})

                # ─── Sliding Window Smoothing ────────────────
                if len(window) >= 8:
                    counts = Counter([w['class_id'] for w in window])
                    dom_id, dom_count = counts.most_common(1)[0]
                    dom_label = TARGET_CLASSES.get(dom_id, "unknown")
                    vote_ratio = dom_count / len(window)
                    dom_confs = [w['conf']
                                 for w in window if w['class_id'] == dom_id]
                    avg_conf = sum(dom_confs) / len(dom_confs)

                    # ─── Neutral Trick (Confidence Thresholding) ─
                    if vote_ratio >= args.min_vote_ratio and avg_conf >= args.min_avg_confidence:
                        stable_label = dom_label
                    else:
                        stable_label = "neutral"
                else:
                    stable_label = "neutral"
            else:
                window.clear()
        else:
            window.clear()

        # Statistik
        if stable_label == "no_face":
            total_no_face += 1
        elif stable_label == "neutral":
            total_neutral += 1
        else:
            stable_dist[stable_label] += 1

        # ─── FPS ─────────────────────────────────────────────
        infer_time = time.time() - t0
        fps = 1.0 / infer_time if infer_time > 0 else 0

        # ─── Log frame ke CSV ────────────────────────────────
        if args.save_csv:
            frame_logs.append({
                'frame_index': frame_idx,
                'timestamp_sec': frame_idx / cam_fps,
                'raw_class_name': raw_label if raw_label else 'no_face',
                'raw_confidence': raw_conf,
                'stable_label': stable_label,
                'vote_ratio': vote_ratio,
                'avg_confidence': avg_conf,
                'bbox_x1': bbox[0] if bbox is not None else 0,
                'bbox_y1': bbox[1] if bbox is not None else 0,
                'bbox_x2': bbox[2] if bbox is not None else 0,
                'bbox_y2': bbox[3] if bbox is not None else 0,
                'fps': fps
            })

        # ─── Visualisasi ─────────────────────────────────────
        display = frame.copy()
        color = CLASS_COLORS.get(stable_label, (255, 255, 255))

        if bbox is not None:
            x1, y1, x2, y2 = map(int, bbox)
            cv2.rectangle(display, (x1, y1), (x2, y2), color, 2)

            # Label utama (stabil)
            label_text = f"{stable_label}"
            (tw, th), _ = cv2.getTextSize(
                label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)
            cv2.rectangle(display, (x1, y1 - th - 10),
                          (x1 + tw + 10, y1), color, -1)
            text_color = (
                255, 255, 255) if stable_label != 'neutral' else (0, 0, 0)
            cv2.putText(display, label_text, (x1 + 5, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, text_color, 2)

            # Info vote & confidence
            info = f"vote: {vote_ratio:.0%} | conf: {avg_conf:.2f}"
            cv2.putText(display, info, (x1, y2 + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

            # Prediksi mentah (opsional)
            if args.show_raw and raw_label:
                cv2.putText(display, f"raw: {raw_label} ({raw_conf:.2f})", (x1, y2 + 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)
        else:
            cv2.putText(display, "No face detected", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (128, 128, 128), 2)

        # FPS counter
        cv2.putText(display, f"FPS: {fps:.0f}", (frame_w - 120, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # Hard sample counter (jika mode aktif)
        if args.capture_hard_samples:
            cv2.putText(display, f"[REC] Samples: {hard_sample_count}  |  0-3=save  c=auto  s=screenshot",
                        (10, frame_h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 255), 1)

        # Legend kelas emosi
        y_legend = frame_h - 170 if args.capture_hard_samples else frame_h - 150
        for cls_name, cls_color in CLASS_COLORS.items():
            if cls_name == 'no_face':
                continue
            cv2.circle(display, (20, y_legend), 8, cls_color, -1)
            cv2.putText(display, cls_name, (35, y_legend + 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            y_legend += 25

        cv2.imshow("YOLOv13 Student Engagement Detection", display)

        if video_writer:
            video_writer.write(display)

        # ─── Keyboard Input ──────────────────────────────────
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break

        # Screenshot (frame + anotasi)
        elif key == ord('s'):
            sp = out_dir / 'screenshots' / f"screenshot_{frame_idx}.jpg"
            cv2.imwrite(str(sp), display)
            print(f"  [SCREENSHOT] Saved: {sp}")

        # ─── Manual Hard Sample Capture ──────────────────────
        if args.capture_hard_samples and hs_dir:
            target_cls_name = None
            target_cls_id = None

            # Hotkey angka: pilih kelas secara manual
            if key == ord('0'):
                target_cls_name = 'engaged'
                target_cls_id = 0
            elif key == ord('1'):
                target_cls_name = 'confused'
                target_cls_id = 1
            elif key == ord('2'):
                target_cls_name = 'bored'
                target_cls_id = 2
            elif key == ord('3'):
                target_cls_name = 'frustrated'
                target_cls_id = 3
            # Hotkey 'c': auto-label menggunakan label stabil saat ini
            elif key == ord('c'):
                if stable_label in CLASS_NAME_TO_ID:
                    target_cls_name = stable_label
                    target_cls_id = CLASS_NAME_TO_ID[stable_label]
                else:
                    print(
                        f"  [SKIP] Label stabil saat ini '{stable_label}', tidak bisa auto-label.")

            if target_cls_name is not None and target_cls_id is not None:
                save_hard_sample(frame, bbox, target_cls_name, target_cls_id,
                                 frame_idx, hs_dir)
                hard_sample_count += 1

    cap.release()
    if video_writer:
        video_writer.release()
    cv2.destroyAllWindows()

    # ─── Simpan CSV Log ──────────────────────────────────────
    if args.save_csv and frame_logs:
        csv_path = out_dir / 'realtime_prediction_log.csv'
        pd.DataFrame(frame_logs).to_csv(csv_path, index=False)
        print(f"\n[CSV] Log disimpan: {csv_path}")

    # ─── Simpan Summary ──────────────────────────────────────
    md = [
        "# Real-time Smoothed Inference Summary\n",
        f"- **Model Weights:** `{args.weights}`",
        f"- **Source:** `{args.source}`",
        "\n## Smoothing Parameters",
        f"- `window_size`: {args.window_size}",
        f"- `min_vote_ratio`: {args.min_vote_ratio}",
        f"- `min_avg_confidence`: {args.min_avg_confidence}",
        "\n## Execution Stats",
        f"- Total frames processed: {frame_idx}",
        f"- Frames with `no_face`: {total_no_face}",
        f"- Frames with `neutral`: {total_neutral}",
        f"- Hard samples collected: {hard_sample_count}",
        "\n## Raw Prediction Distribution"
    ]
    for cls in TARGET_CLASSES.values():
        md.append(f"- {cls}: {raw_dist.get(cls, 0)}")
    md.append("\n## Stable Prediction Distribution")
    for cls in TARGET_CLASSES.values():
        md.append(f"- {cls}: {stable_dist.get(cls, 0)}")

    # Hitung total per kelas di folder hard_samples
    if hs_dir:
        md.append("\n## Hard Samples per Kelas (Kumulatif)")
        for cls_name in TARGET_CLASSES.values():
            count = len(list((hs_dir / cls_name).glob('*.jpg')))
            md.append(f"- {cls_name}: {count} citra + label")

    with open(out_dir / 'summary.md', 'w', encoding='utf-8') as f:
        f.write("\n".join(md))

    print(f"\n[SUCCESS] Sesi selesai. Output: {out_dir}")
    if hard_sample_count > 0:
        print(
            f"  Hard samples tersimpan: {hard_sample_count} (citra + label YOLO)")
        print(f"  Siap digunakan untuk fine-tuning!")


if __name__ == '__main__':
    main()
