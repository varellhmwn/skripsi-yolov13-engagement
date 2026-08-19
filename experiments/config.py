"""
config.py — Konfigurasi Terpusat Eksperimen YOLOv13n vs HOG-KNN
===============================================================
Mengatur path dataset, bobot model final, parameter HOG, dan output.
"""

from pathlib import Path

# ─── Direktori Utama ─────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent
DATASET_DIR = BASE_DIR / 'datasets' / 'master_combined_dataset'
DATA_YAML = DATASET_DIR / 'data.yaml'
OUTPUT_DIR = BASE_DIR / 'outputs'

# ─── Bobot Model Final ───────────────────────────────────────
# Menggunakan bobot resmi v2 (Master Final TA & Dashboard)
MODEL_WEIGHTS_PATH = BASE_DIR / 'runs' / 'yolov13_master_combined_v2' / 'weights' / 'best.pt'

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

# ─── Parameter Benchmarking ──────────────────────────────────
BENCHMARK_WARMUP_ROUNDS = 20
BENCHMARK_DEVICE = 0  # GPU CUDA:0 (NVIDIA RTX 4060 Laptop)
YOLO_CONF_THRESHOLD = 0.25
YOLO_IMGSZ = 640
VALID_IMG_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
