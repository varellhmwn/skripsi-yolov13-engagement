"""
utils.py — Fungsi Reusable untuk Eksperimen HOG-KNN vs YOLOv13n
================================================================
Menyediakan utilitas untuk:
  - Parsing anotasi YOLO
  - Crop wajah dari bounding box
  - Ekstraksi fitur HOG
  - Kalkulasi metrik evaluasi
  - Plotting confusion matrix
  - Validasi data leakage
"""

import json
import time
import warnings
from pathlib import Path

import cv2
import numpy as np
import yaml
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from skimage.feature import hog
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, precision_recall_fscore_support
)

# ─── Konstanta ──────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent
DATASET_DIR = BASE_DIR / 'datasets' / 'master_combined_dataset'
DATA_YAML = DATASET_DIR / 'data.yaml'

CLASS_NAMES = {0: 'engaged', 1: 'confused', 2: 'bored', 3: 'frustrated'}
CLASS_LIST = ['engaged', 'confused', 'bored', 'frustrated']
VALID_IMG_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}

# Konfigurasi HOG (konsisten dengan metodologi penelitian)
HOG_IMG_SIZE = (64, 64)
HOG_ORIENTATIONS = 9
HOG_PIXELS_PER_CELL = (8, 8)
HOG_CELLS_PER_BLOCK = (2, 2)
HOG_BLOCK_NORM = 'L2-Hys'

RANDOM_SEED = 42


def load_data_yaml():
    """Load data.yaml dan return konfigurasi dataset."""
    if not DATA_YAML.exists():
        raise FileNotFoundError(f"data.yaml tidak ditemukan: {DATA_YAML}")
    with open(DATA_YAML, 'r') as f:
        cfg = yaml.safe_load(f)
    return cfg


def parse_yolo_annotation(label_path, img_width, img_height):
    """
    Parse file anotasi YOLO (.txt) menjadi list of (class_id, x1, y1, x2, y2).

    Parameters
    ----------
    label_path : Path
        Path ke file .txt anotasi YOLO.
    img_width : int
        Lebar gambar dalam pixel.
    img_height : int
        Tinggi gambar dalam pixel.

    Returns
    -------
    list of tuple
        Setiap tuple: (class_id, x1, y1, x2, y2) dalam pixel coordinates.
    """
    annotations = []
    with open(label_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 5:
                continue
            try:
                class_id = int(parts[0])
            except ValueError:
                # Skip lines where class is not a number (e.g. "engaged")
                continue

            x_center = float(parts[1]) * img_width
            y_center = float(parts[2]) * img_height
            box_w = float(parts[3]) * img_width
            box_h = float(parts[4]) * img_height

            x1 = int(max(0, x_center - box_w / 2))
            y1 = int(max(0, y_center - box_h / 2))
            x2 = int(min(img_width, x_center + box_w / 2))
            y2 = int(min(img_height, y_center + box_h / 2))

            annotations.append((class_id, x1, y1, x2, y2))

    return annotations


def crop_face_from_bbox(image, x1, y1, x2, y2):
    """
    Crop area wajah dari gambar berdasarkan bounding box.

    Returns None jika crop tidak valid.
    """
    # Clamp coordinates
    h, w = image.shape[:2]
    x1 = max(0, min(x1, w))
    y1 = max(0, min(y1, h))
    x2 = max(0, min(x2, w))
    y2 = max(0, min(y2, h))

    if x2 <= x1 or y2 <= y1:
        return None

    crop = image[y1:y2, x1:x2]
    if crop.size == 0:
        return None

    return crop


def extract_hog_features(face_crop):
    """
    Preprocessing dan ekstraksi fitur HOG dari crop wajah.

    Pipeline:
        face_crop → resize 64×64 → grayscale → HOG → feature vector

    Parameters
    ----------
    face_crop : numpy.ndarray
        Crop wajah (BGR format dari OpenCV).

    Returns
    -------
    numpy.ndarray
        Feature vector HOG.
    """
    # Resize ke 64×64
    resized = cv2.resize(face_crop, HOG_IMG_SIZE)

    # Konversi ke grayscale
    if len(resized.shape) == 3:
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    else:
        gray = resized

    # Ekstraksi HOG
    features = hog(
        gray,
        orientations=HOG_ORIENTATIONS,
        pixels_per_cell=HOG_PIXELS_PER_CELL,
        cells_per_block=HOG_CELLS_PER_BLOCK,
        block_norm=HOG_BLOCK_NORM,
        visualize=False
    )

    return features


def load_dataset_split(split_name, use_largest_bbox=True):
    """
    Load seluruh gambar dari split dataset, crop wajah berdasarkan
    ground-truth bounding box, dan ekstrak fitur HOG.

    Parameters
    ----------
    split_name : str
        Nama split: 'train', 'val', atau 'test'.
    use_largest_bbox : bool
        Jika True, gunakan bounding box terbesar jika ada multiple bbox.

    Returns
    -------
    features : numpy.ndarray
        Array fitur HOG (n_samples, n_features).
    labels : numpy.ndarray
        Array class ID (n_samples,).
    filenames : list of str
        Nama file gambar yang berhasil diproses.
    skipped : list of str
        Nama file gambar yang dilewati (error/tidak valid).
    """
    images_dir = DATASET_DIR / 'images' / split_name
    labels_dir = DATASET_DIR / 'labels' / split_name

    if not images_dir.exists():
        raise FileNotFoundError(f"Images directory tidak ditemukan: {images_dir}")
    if not labels_dir.exists():
        raise FileNotFoundError(f"Labels directory tidak ditemukan: {labels_dir}")

    features = []
    labels = []
    filenames = []
    skipped = []

    # Iterate over image files
    img_files = sorted([
        f for f in images_dir.iterdir()
        if f.suffix.lower() in VALID_IMG_EXTS
    ])

    for img_path in img_files:
        label_path = labels_dir / f"{img_path.stem}.txt"
        if not label_path.exists():
            skipped.append(img_path.name)
            continue

        # Read image
        img = cv2.imread(str(img_path))
        if img is None:
            skipped.append(img_path.name)
            continue

        h, w = img.shape[:2]

        # Parse annotations
        annotations = parse_yolo_annotation(label_path, w, h)
        if not annotations:
            skipped.append(img_path.name)
            continue

        # Select bounding box
        if use_largest_bbox and len(annotations) > 1:
            # Pilih bbox terbesar (area terbesar)
            best_ann = max(annotations,
                          key=lambda a: (a[3] - a[1]) * (a[4] - a[2]))
        else:
            best_ann = annotations[0]

        class_id, x1, y1, x2, y2 = best_ann

        # Crop face
        face_crop = crop_face_from_bbox(img, x1, y1, x2, y2)
        if face_crop is None:
            skipped.append(img_path.name)
            continue

        # Extract HOG features
        try:
            feat = extract_hog_features(face_crop)
            features.append(feat)
            labels.append(class_id)
            filenames.append(img_path.name)
        except Exception as e:
            skipped.append(img_path.name)
            continue

    return (np.array(features), np.array(labels), filenames, skipped)


def calculate_metrics(y_true, y_pred, class_names=None):
    """
    Hitung metrik klasifikasi lengkap.

    Returns
    -------
    dict
        Dictionary berisi semua metrik evaluasi.
    """
    if class_names is None:
        class_names = CLASS_LIST

    # Overall metrics
    acc = accuracy_score(y_true, y_pred)
    macro_p = precision_score(y_true, y_pred, average='macro', zero_division=0)
    macro_r = recall_score(y_true, y_pred, average='macro', zero_division=0)
    macro_f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
    weighted_p = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    weighted_r = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    weighted_f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)

    # Per-class metrics
    per_class_p, per_class_r, per_class_f1, per_class_support = \
        precision_recall_fscore_support(
            y_true, y_pred,
            labels=list(range(len(class_names))),
            zero_division=0
        )

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))

    per_class = {}
    for i, name in enumerate(class_names):
        per_class[name] = {
            'precision': float(per_class_p[i]),
            'recall': float(per_class_r[i]),
            'f1': float(per_class_f1[i]),
            'support': int(per_class_support[i])
        }

    metrics = {
        'accuracy': float(acc),
        'macro_precision': float(macro_p),
        'macro_recall': float(macro_r),
        'macro_f1': float(macro_f1),
        'weighted_precision': float(weighted_p),
        'weighted_recall': float(weighted_r),
        'weighted_f1': float(weighted_f1),
        'per_class': per_class,
        'confusion_matrix': cm.tolist()
    }

    return metrics


def plot_confusion_matrix(cm, class_names, title, save_path, accuracy=None):
    """
    Plot dan simpan confusion matrix sebagai PNG.

    Parameters
    ----------
    cm : array-like
        Confusion matrix.
    class_names : list of str
        Nama kelas.
    title : str
        Judul plot.
    save_path : str or Path
        Path untuk menyimpan PNG.
    accuracy : float, optional
        Accuracy untuk ditampilkan di subtitle.
    """
    plt.figure(figsize=(8, 6))
    cm_array = np.array(cm)

    sns.heatmap(
        cm_array,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names,
        square=True,
        linewidths=0.5,
        cbar_kws={'shrink': 0.8}
    )

    if accuracy is not None:
        plt.title(f'{title}\n(Accuracy: {accuracy:.2%})', fontsize=13, pad=15)
    else:
        plt.title(title, fontsize=13, pad=15)

    plt.ylabel('True Label', fontsize=11)
    plt.xlabel('Predicted Label', fontsize=11)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10, rotation=0)
    plt.tight_layout()

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(save_path), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  [SAVED] Confusion matrix: {save_path}")


def validate_no_data_leakage():
    """
    Validasi bahwa train, val, dan test sets tidak overlap.

    Returns
    -------
    dict
        Hasil validasi: pass/fail + detail.
    """
    results = {'passed': True, 'details': []}

    splits = {}
    for split in ['train', 'val', 'test']:
        img_dir = DATASET_DIR / 'images' / split
        if img_dir.exists():
            splits[split] = set(f.stem for f in img_dir.iterdir()
                                if f.suffix.lower() in VALID_IMG_EXTS)
        else:
            splits[split] = set()
            results['details'].append(f"WARNING: {split} images directory not found")

    # Check overlaps
    checks = [
        ('train', 'val'),
        ('train', 'test'),
        ('val', 'test')
    ]

    for s1, s2 in checks:
        overlap = splits[s1] & splits[s2]
        if overlap:
            results['passed'] = False
            results['details'].append(
                f"LEAK: {s1} ∩ {s2} = {len(overlap)} files: "
                f"{list(overlap)[:5]}..."
            )
        else:
            results['details'].append(f"OK: {s1} ∩ {s2} = 0 (no overlap)")

    # Report counts
    for split, files in splits.items():
        results['details'].append(f"  {split}: {len(files)} images")

    return results


def get_class_distribution(split_name):
    """
    Hitung distribusi kelas untuk split tertentu.

    Returns
    -------
    dict
        {class_id: count}
    """
    labels_dir = DATASET_DIR / 'labels' / split_name
    distribution = {i: 0 for i in range(len(CLASS_LIST))}

    if not labels_dir.exists():
        return distribution

    for label_path in labels_dir.iterdir():
        if label_path.suffix != '.txt':
            continue
        try:
            with open(label_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        try:
                            cls_id = int(parts[0])
                            if cls_id in distribution:
                                distribution[cls_id] += 1
                        except ValueError:
                            continue
        except Exception:
            continue

    return distribution


def save_metrics_json(metrics, save_path):
    """Simpan metrik ke file JSON."""
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    print(f"  [SAVED] Metrics: {save_path}")


class TimingContext:
    """Context manager untuk mengukur waktu eksekusi."""

    def __init__(self, name="operation"):
        self.name = name
        self.elapsed = 0.0

    def __enter__(self):
        self.start = time.perf_counter()
        return self

    def __exit__(self, *args):
        self.elapsed = time.perf_counter() - self.start
