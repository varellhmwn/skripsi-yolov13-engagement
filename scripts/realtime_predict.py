"""
realtime_predict.py — Real-Time Inference via Webcam
=====================================================
Mendeteksi emosi mahasiswa secara real-time menggunakan webcam.
Model YOLOv13n mendeteksi 4 kelas (engaged, confused, bored, frustrated),
ditambah label 'neutral' dari mekanisme Confidence Thresholding.

Fitur:
  - Sliding Window Smoothing (mengurangi flicker antar-frame)
  - Confidence Thresholding (wajah datar → neutral)
  - Bounding box berwarna per emosi
  - FPS counter & legend

Penggunaan:
    python scripts/realtime_predict.py
    python scripts/realtime_predict.py --source 0
    python scripts/realtime_predict.py --weights runs/yolov13_master_combined/weights/best.pt
"""

import cv2
import argparse
import time
import sys
from pathlib import Path
from collections import deque, Counter
from ultralytics import YOLO


# ─── Konfigurasi Default ────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_WEIGHTS = str(BASE_DIR / 'runs' / 'yolov13_master_combined' / 'weights' / 'best.pt')

TARGET_CLASSES = {0: 'engaged', 1: 'confused', 2: 'bored', 3: 'frustrated'}

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
    parser.add_argument('--conf', type=float, default=0.25,
                        help="Minimum confidence YOLO (default: 0.25)")
    parser.add_argument('--device', type=str, default='0',
                        help="Device inference (default: 0 = GPU)")
    parser.add_argument('--window_size', type=int, default=30,
                        help="Ukuran sliding window untuk smoothing (default: 30)")
    parser.add_argument('--min_vote_ratio', type=float, default=0.50,
                        help="Rasio voting minimum (default: 0.50)")
    parser.add_argument('--min_avg_confidence', type=float, default=0.65,
                        help="Threshold confidence untuk neutral trick (default: 0.65)")
    parser.add_argument('--show_raw', action='store_true',
                        help="Tampilkan prediksi mentah di samping prediksi stabil")
    return parser.parse_args()


def main():
    args = parse_args()

    # Validasi weights
    if not Path(args.weights).exists():
        print(f"[ERROR] Weights tidak ditemukan: {args.weights}")
        sys.exit(1)

    print("=" * 60)
    print("  REAL-TIME STUDENT ENGAGEMENT DETECTION")
    print("  YOLOv13n — 4 Class + Neutral Trick")
    print("=" * 60)
    print(f"  Weights              : {args.weights}")
    print(f"  Source               : {'Webcam' if args.source == '0' else args.source}")
    print(f"  Smoothing Window     : {args.window_size} frames")
    print(f"  Confidence Threshold : {args.min_avg_confidence}")
    print(f"  Classes              : {list(TARGET_CLASSES.values())} + neutral")
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

    # Sliding window untuk temporal smoothing
    window = deque(maxlen=args.window_size)

    print("[INFO] Real-time inference dimulai. Tekan 'q' untuk keluar.\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

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
                    best_det = (int(det.cls[i].item()), float(det.conf[i].item()), xyxy)

            if best_det is not None:
                cls_id, conf, bbox = best_det
                raw_label = TARGET_CLASSES.get(cls_id, "unknown")
                raw_conf = conf

                window.append({'class_id': cls_id, 'conf': conf})

                # ─── Sliding Window Smoothing ────────────────
                if len(window) >= 8:
                    counts = Counter([w['class_id'] for w in window])
                    dom_id, dom_count = counts.most_common(1)[0]
                    dom_label = TARGET_CLASSES.get(dom_id, "unknown")
                    vote_ratio = dom_count / len(window)
                    dom_confs = [w['conf'] for w in window if w['class_id'] == dom_id]
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

        # ─── FPS ─────────────────────────────────────────────
        fps = 1.0 / (time.time() - t0) if (time.time() - t0) > 0 else 0

        # ─── Visualisasi ─────────────────────────────────────
        display = frame.copy()
        color = CLASS_COLORS.get(stable_label, (255, 255, 255))

        if bbox is not None:
            x1, y1, x2, y2 = map(int, bbox)
            cv2.rectangle(display, (x1, y1), (x2, y2), color, 2)

            # Label utama (stabil)
            label_text = f"{stable_label}"
            (tw, th), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)
            cv2.rectangle(display, (x1, y1 - th - 10), (x1 + tw + 10, y1), color, -1)
            text_color = (255, 255, 255) if stable_label != 'neutral' else (0, 0, 0)
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

        # Legend kelas emosi
        y_legend = frame_h - 150
        for cls_name, cls_color in CLASS_COLORS.items():
            if cls_name == 'no_face':
                continue
            cv2.circle(display, (20, y_legend), 8, cls_color, -1)
            cv2.putText(display, cls_name, (35, y_legend + 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            y_legend += 25

        cv2.imshow("YOLOv13 Student Engagement Detection", display)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("\n[INFO] Sesi selesai.")


if __name__ == '__main__':
    main()
