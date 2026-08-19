"""
utils.py — Backward-compatible wrapper for experiments
"""

from experiments.config import *
from experiments.hog_features import (
    parse_yolo_annotation,
    crop_face_from_bbox,
    extract_hog_features,
    load_dataset_split,
    calculate_metrics,
    plot_confusion_matrix
)
import json
import time

def save_metrics_json(metrics, save_path):
    """Simpan metrik ke JSON."""
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

class TimingContext:
    def __init__(self, name="operation"):
        self.name = name
        self.elapsed = 0.0

    def __enter__(self):
        self.start = time.perf_counter()
        return self

    def __exit__(self, *args):
        self.elapsed = time.perf_counter() - self.start
