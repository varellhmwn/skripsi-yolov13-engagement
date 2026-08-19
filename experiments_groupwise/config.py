"""
config.py — Konfigurasi Terpusat Eksperimen Group-Wise Split & Retraining
=========================================================================
Mengatur path dataset baru (group-wise v1), output directories, bobot YOLO,
parameter HOG, hyperparameter training YOLOv13n, dan konfigurasi benchmarking.
"""

from pathlib import Path

# ─── Direktori Utama ─────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent

# Dataset Asli (Tidak Boleh Dimodifikasi)
ORIGINAL_DATASET_DIR = BASE_DIR / 'datasets' / 'master_combined_dataset'

# Dataset Baru Group-Wise (v1)
GROUPWISE_DATASET_DIR = BASE_DIR / 'datasets' / 'master_combined_groupwise_v1'
GROUPWISE_DATA_YAML = GROUPWISE_DATASET_DIR / 'data.yaml'

# Output Direktori Khusus Group-Wise
OUTPUT_GROUPWISE_DIR = BASE_DIR / 'outputs_groupwise'
RUNS_GROUPWISE_DIR = BASE_DIR / 'runs' / 'yolov13_master_groupwise_v1'

# Pretrained Base Weight untuk Training dari Scratch
PRETRAINED_YOLO_WEIGHTS = BASE_DIR / 'yolov13n.pt'

# Bobot Model Final Hasil Retraining Group-Wise
TRAINED_GROUPWISE_WEIGHTS = RUNS_GROUPWISE_DIR / 'weights' / 'best.pt'

# ─── Mapping Kelas ───────────────────────────────────────────
CLASS_NAMES = {0: 'engaged', 1: 'confused', 2: 'bored', 3: 'frustrated'}
CLASS_LIST = ['engaged', 'confused', 'bored', 'frustrated']
NUM_CLASSES = 4

# ─── Parameter HOG ───────────────────────────────────────────
HOG_IMG_SIZE = (64, 64)
HOG_ORIENTATIONS = 9
HOG_PIXELS_PER_CELL = (8, 8)
HOG_CELLS_PER_BLOCK = (2, 2)
HOG_BLOCK_NORM = 'L2-Hys'

# ─── Parameter KNN ───────────────────────────────────────────
KNN_METRIC = 'euclidean'
KNN_K_SEARCH_LIST = [1, 3, 5, 7, 9, 11, 13, 15]
RANDOM_SEED = 42

# ─── Parameter Training YOLOv13n (Sesuai Konfigurasi Asli Penelitian) ───
YOLO_TRAIN_PARAMS = {
    'epochs': 150,
    'imgsz': 640,
    'batch': 16,
    'patience': 25,
    'optimizer': 'AdamW',
    'lr0': 0.001,
    'lrf': 0.01,
    'weight_decay': 0.0005,
    'warmup_epochs': 3,
    'hsv_h': 0.015,
    'hsv_s': 0.7,
    'hsv_v': 0.4,
    'degrees': 10.0,
    'translate': 0.1,
    'scale': 0.5,
    'fliplr': 0.5,
    'mosaic': 1.0,
    'mixup': 0.1,
    'close_mosaic': 10,
    'val': True,
    'save': True,
    'plots': True,
    'device': 0
}

# ─── Parameter Benchmarking ──────────────────────────────────
BENCHMARK_WARMUP_ROUNDS = 20
BENCHMARK_DEVICE = 0
YOLO_CONF_THRESHOLD = 0.25
YOLO_IMGSZ = 640
VALID_IMG_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
