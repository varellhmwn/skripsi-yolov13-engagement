"""
hog_features.py — Preprocessing dan Ekstraksi Fitur HOG
======================================================
Fungsi standar ekstraksi fitur HOG yang digunakan secara seragam
oleh semua tahap: training, validation, test ground-truth, dan hybrid.
"""

from pathlib import Path
import cv2
import numpy as np
from skimage.feature import hog
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, precision_recall_fscore_support
)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from experiments.config import (
    HOG_IMG_SIZE, HOG_ORIENTATIONS, HOG_PIXELS_PER_CELL,
    HOG_CELLS_PER_BLOCK, HOG_BLOCK_NORM, DATASET_DIR,
    VALID_IMG_EXTS, CLASS_LIST
)


def parse_yolo_annotation(label_path, img_width, img_height):
    """Parse file anotasi YOLO menjadi list of (class_id, x1, y1, x2, y2)."""
    annotations = []
    if not Path(label_path).exists():
        return annotations
        
    with open(label_path, 'r', encoding='utf-8') as f:
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
    """Crop area wajah dengan coordinate clamping."""
    h, w = image.shape[:2]
    x1 = max(0, min(int(x1), w))
    y1 = max(0, min(int(y1), h))
    x2 = max(0, min(int(x2), w))
    y2 = max(0, min(int(y2), h))

    if x2 <= x1 or y2 <= y1:
        return None

    crop = image[y1:y2, x1:x2]
    if crop.size == 0:
        return None

    return crop


def extract_hog_features(face_crop):
    """
    Ekstraksi fitur HOG terstandarisasi:
    Face Crop -> Resize 64x64 -> Grayscale -> HOG Feature Vector
    """
    resized = cv2.resize(face_crop, HOG_IMG_SIZE)
    if len(resized.shape) == 3:
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    else:
        gray = resized

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
    Load dataset split: crop ground truth wajah & extract HOG.
    Mengabaikan orphan label tanpa file citra.
    """
    images_dir = DATASET_DIR / 'images' / split_name
    labels_dir = DATASET_DIR / 'labels' / split_name

    features = []
    labels = []
    filenames = []
    skipped = []

    img_files = sorted([
        f for f in images_dir.iterdir()
        if f.suffix.lower() in VALID_IMG_EXTS
    ])

    for img_path in img_files:
        label_path = labels_dir / f"{img_path.stem}.txt"
        if not label_path.exists():
            skipped.append(img_path.name)
            continue

        img = cv2.imread(str(img_path))
        if img is None:
            skipped.append(img_path.name)
            continue

        h, w = img.shape[:2]
        annotations = parse_yolo_annotation(label_path, w, h)
        if not annotations:
            skipped.append(img_path.name)
            continue

        if use_largest_bbox and len(annotations) > 1:
            best_ann = max(annotations, key=lambda a: (a[3] - a[1]) * (a[4] - a[2]))
        else:
            best_ann = annotations[0]

        class_id, x1, y1, x2, y2 = best_ann
        face_crop = crop_face_from_bbox(img, x1, y1, x2, y2)
        if face_crop is None:
            skipped.append(img_path.name)
            continue

        try:
            feat = extract_hog_features(face_crop)
            features.append(feat)
            labels.append(class_id)
            filenames.append(img_path.name)
        except Exception:
            skipped.append(img_path.name)
            continue

    return np.array(features), np.array(labels), filenames, skipped


def calculate_metrics(y_true, y_pred, class_names=None):
    """Menghitung metrik evaluasi klasifikasi lengkap."""
    if class_names is None:
        class_names = CLASS_LIST

    acc = accuracy_score(y_true, y_pred)
    macro_p = precision_score(y_true, y_pred, average='macro', zero_division=0)
    macro_r = recall_score(y_true, y_pred, average='macro', zero_division=0)
    macro_f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
    weighted_p = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    weighted_r = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    weighted_f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)

    per_class_p, per_class_r, per_class_f1, per_class_support = \
        precision_recall_fscore_support(
            y_true, y_pred,
            labels=list(range(len(class_names))),
            zero_division=0
        )

    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))

    per_class = {}
    for i, name in enumerate(class_names):
        per_class[name] = {
            'precision': float(per_class_p[i]),
            'recall': float(per_class_r[i]),
            'f1': float(per_class_f1[i]),
            'support': int(per_class_support[i])
        }

    return {
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


def plot_confusion_matrix(cm, class_names, title, save_path, accuracy=None):
    """Plot dan simpan confusion matrix PNG berkualitas publikasi."""
    plt.figure(figsize=(7, 5.5))
    cm_array = np.array(cm)

    sns.heatmap(
        cm_array,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names,
        square=True,
        linewidths=0.7,
        cbar_kws={'shrink': 0.8}
    )

    if accuracy is not None:
        plt.title(f'{title}\n(Accuracy: {accuracy:.2%})', fontsize=12, pad=12)
    else:
        plt.title(title, fontsize=12, pad=12)

    plt.ylabel('True Class', fontsize=11)
    plt.xlabel('Predicted Class', fontsize=11)
    plt.tight_layout()

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(save_path), dpi=300, bbox_inches='tight')
    plt.close()
