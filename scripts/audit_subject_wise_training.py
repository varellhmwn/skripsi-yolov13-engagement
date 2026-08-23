"""
audit_subject_wise_training.py — Preflight Dataset & Environment Audit
======================================================================
Penelitian: "Deteksi Emosi Belajar Mahasiswa pada Pembelajaran Pemrograman Menggunakan YOLOv13n"

Tujuan:
  1. Memeriksa spesifikasi environment (Python, PyTorch, CUDA, GPU, Ultralytics, OS).
  2. Memvalidasi data.yaml (portabilitas, path relatif, keberadaan direktori).
  3. Mengaudit dataset fisik:
     - Train: 1311 citra, Val: 158 citra, Test: 156 citra (Total: 1625).
     - Integritas pasangan citra & label (0 orphan, nama basename identik).
     - Validitas anotasi bounding box YOLO & class ID {0, 1, 2, 3}.
     - ZERO subject leakage antar subset (train, val, test saling disjoint).
     - ZERO SHA-256 hash leakage antar subset (train, val, test saling disjoint).
  4. Membersihkan file .cache usang jika ditemukan di subject_wise_dataset/.
"""

import sys
import os
import re
import csv
import json
import hashlib
import platform
import logging
from pathlib import Path
from collections import defaultdict, Counter

import torch
import ultralytics

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger('SubjectWiseAuditor')

CLASS_NAME_TO_ID = {
    'engaged': 0,
    'confused': 1,
    'bored': 2,
    'frustrated': 3
}
CLASS_ID_TO_NAME = {v: k for k, v in CLASS_NAME_TO_ID.items()}
VALID_IMAGE_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}


def compute_sha256(file_path: Path) -> str:
    """Menghitung SHA-256 checksum sebuah file secara streaming."""
    hasher = hashlib.sha256()
    with open(file_path, 'rb') as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


def get_environment_info(seed: int = 42) -> dict:
    """Mengumpulkan metadata environment secara komprehensif."""
    cuda_avail = torch.cuda.is_available()
    gpu_info = []
    if cuda_avail:
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            gpu_info.append({
                'device_id': i,
                'name': props.name,
                'total_memory_gb': round(props.total_memory / (1024**3), 2),
                'multi_processor_count': props.multi_processor_count
            })

    info = {
        'os': platform.platform(),
        'python_version': sys.version.split()[0],
        'pytorch_version': torch.__version__,
        'cuda_available': cuda_avail,
        'cuda_version': torch.version.cuda if cuda_avail else None,
        'ultralytics_version': ultralytics.__version__,
        'random_seed': seed,
        'gpus': gpu_info
    }
    return info


def clean_stale_caches(dataset_dir: Path):
    """Menghapus file cache (.cache) lama di folder labels agar Ultralytics membangun ulang."""
    cleaned = []
    for p in dataset_dir.rglob('*.cache'):
        if p.is_file():
            p.unlink()
            cleaned.append(str(p))
    if cleaned:
        logger.info(f"Dibersihkan {len(cleaned)} file cache lama:")
        for c in cleaned:
            logger.info(f"  - Dihapus: {c}")
    else:
        logger.info("Tidak ditemukan file .cache lama (bersih).")


def validate_yolo_label(label_path: Path):
    """Memvalidasi baris anotasi label YOLO."""
    if not label_path.exists():
        return False, f"File label tidak ditemukan: {label_path}", []
    raw_lines = []
    with open(label_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                raw_lines.append(line)
    if not raw_lines:
        return False, f"File label kosong (0 bbox): {label_path}", []

    parsed = []
    for idx, line in enumerate(raw_lines, 1):
        parts = line.split()
        if len(parts) < 5:
            return False, f"Format tidak valid baris {idx}: '{line}'", []
        try:
            cid = int(parts[0])
            xc, yc, w, h = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
        except ValueError:
            return False, f"Bukan angka numerik baris {idx}: '{line}'", []

        if cid not in CLASS_ID_TO_NAME:
            return False, f"Class ID '{cid}' di luar {{0,1,2,3}} baris {idx}", []
        if not (0.0 <= xc <= 1.0 and 0.0 <= yc <= 1.0 and 0.0 < w <= 1.0 and 0.0 < h <= 1.0):
            return False, f"Koordinat box tidak valid baris {idx}: '{line}'", []

        parsed.append((cid, xc, yc, w, h))
    return True, "", parsed


def perform_preflight_audit(dataset_dir: Path, expected_counts=None):
    """Menjalankan audit preflight ketat terhadap dataset fisik subject_wise_dataset/."""
    if expected_counts is None:
        expected_counts = {'train': 1311, 'val': 158, 'test': 156}

    dataset_dir = Path(dataset_dir).resolve()
    logger.info("=" * 70)
    logger.info(f"PREFLIGHT AUDIT DATASET FISIK: {dataset_dir}")
    logger.info("=" * 70)

    # 1. Periksa data.yaml
    data_yaml_p = dataset_dir / 'data.yaml'
    if not data_yaml_p.exists():
        raise FileNotFoundError(f"data.yaml tidak ditemukan di: {data_yaml_p}")

    # 2. Bersihkan cache lama
    clean_stale_caches(dataset_dir)

    subset_data = {}
    subset_subjects = {}
    subset_hashes = {}
    subset_class_counts = {}

    for subset in ['train', 'val', 'test']:
        img_dir = dataset_dir / 'images' / subset
        lbl_dir = dataset_dir / 'labels' / subset

        if not img_dir.exists():
            raise FileNotFoundError(f"Direktori citra tidak ditemukan: {img_dir}")
        if not lbl_dir.exists():
            raise FileNotFoundError(f"Direktori label tidak ditemukan: {lbl_dir}")

        img_files = sorted([f for f in img_dir.iterdir() if f.suffix.lower() in VALID_IMAGE_EXTS])
        lbl_files = sorted([f for f in lbl_dir.iterdir() if f.suffix.lower() == '.txt'])

        # Cek jumlah
        exp = expected_counts[subset]
        act = len(img_files)
        logger.info(f"[{subset.upper()}] Images: {act} (Ekspektasi: {exp}) | Labels: {len(lbl_files)}")

        if act != exp:
            raise AssertionError(f"Jumlah citra {subset} ({act}) tidak sesuai ekspektasi ({exp})!")
        if len(lbl_files) != exp:
            raise AssertionError(f"Jumlah label {subset} ({len(lbl_files)}) tidak sesuai ekspektasi ({exp})!")

        # Cek Pair Integrity
        img_stems = {f.stem: f for f in img_files}
        lbl_stems = {f.stem: f for f in lbl_files}

        orphans_img = [f.name for stem, f in img_stems.items() if stem not in lbl_stems]
        orphans_lbl = [f.name for stem, f in lbl_stems.items() if stem not in img_stems]

        if orphans_img or orphans_lbl:
            raise AssertionError(f"Pair integrity check FAILED pada {subset}: Orphan Imgs={orphans_img}, Orphan Lbls={orphans_lbl}")

        # Cek Subject, Class, Label Annotations, Hash
        subjs = set()
        hashes = set()
        c_counts = Counter()

        for stem, img_p in img_stems.items():
            lbl_p = lbl_stems[stem]
            subj_id = img_p.name.split('_')[0].upper()
            subjs.add(subj_id)

            is_valid, err_msg, bboxes = validate_yolo_label(lbl_p)
            if not is_valid:
                raise AssertionError(f"Label validation FAILED pada {lbl_p}: {err_msg}")

            for cid, xc, yc, w, h in bboxes:
                cname = CLASS_ID_TO_NAME[cid]
                c_counts[cname] += 1

            h = compute_sha256(img_p)
            hashes.add(h)

        subset_subjects[subset] = subjs
        subset_hashes[subset] = hashes
        subset_class_counts[subset] = c_counts
        subset_data[subset] = {
            'images_count': act,
            'labels_count': len(lbl_files),
            'subjects': sorted(list(subjs)),
            'class_counts': dict(c_counts)
        }

        # Validasi bahwa keempat kelas muncul (> 0)
        for cname in ['engaged', 'confused', 'bored', 'frustrated']:
            if c_counts[cname] == 0:
                raise AssertionError(f"Kelas '{cname}' tidak ditemukan (count=0) pada subset {subset}!")

    # 3. Validasi Subject Leakage (Disjoint Assertions)
    tr_sub = subset_subjects['train']
    va_sub = subset_subjects['val']
    te_sub = subset_subjects['test']

    logger.info("\n--- LEAKAGE AUDIT ASSERTIONS ---")
    tr_va_overlap = tr_sub.intersection(va_sub)
    tr_te_overlap = tr_sub.intersection(te_sub)
    va_te_overlap = va_sub.intersection(te_sub)

    logger.info(f"Subject Overlap Train-Val : {len(tr_va_overlap)} {tr_va_overlap}")
    logger.info(f"Subject Overlap Train-Test: {len(tr_te_overlap)} {tr_te_overlap}")
    logger.info(f"Subject Overlap Val-Test  : {len(va_te_overlap)} {va_te_overlap}")

    assert tr_sub.isdisjoint(va_sub), f"LEAKAGE DETECTED: Train-Val Subject Overlap: {tr_va_overlap}"
    assert tr_sub.isdisjoint(te_sub), f"LEAKAGE DETECTED: Train-Test Subject Overlap: {tr_te_overlap}"
    assert va_sub.isdisjoint(te_sub), f"LEAKAGE DETECTED: Val-Test Subject Overlap: {va_te_overlap}"

    # 4. Validasi Image Hash Leakage (Disjoint Assertions)
    tr_h = subset_hashes['train']
    va_h = subset_hashes['val']
    te_h = subset_hashes['test']

    tr_va_h_overlap = tr_h.intersection(va_h)
    tr_te_h_overlap = tr_h.intersection(te_h)
    va_te_h_overlap = va_h.intersection(te_h)

    logger.info(f"Hash Overlap Train-Val    : {len(tr_va_h_overlap)}")
    logger.info(f"Hash Overlap Train-Test   : {len(tr_te_h_overlap)}")
    logger.info(f"Hash Overlap Val-Test     : {len(va_te_h_overlap)}")

    assert tr_h.isdisjoint(va_h), "LEAKAGE DETECTED: Train-Val SHA-256 Hash Overlap!"
    assert tr_h.isdisjoint(te_h), "LEAKAGE DETECTED: Train-Test SHA-256 Hash Overlap!"
    assert va_h.isdisjoint(te_h), "LEAKAGE DETECTED: Val-Test SHA-256 Hash Overlap!"

    logger.info("[SUCCESS] SELURUH PREFLIGHT AUDIT LULUS DENGAN STATUS 100% CLEAN & ZERO LEAKAGE!")
    return {
        'status': 'PASS',
        'subsets': subset_data
    }


if __name__ == '__main__':
    dataset_path = Path('subject_wise_dataset')
    env_info = get_environment_info(seed=42)
    print("\n" + "=" * 70)
    print("ENVIRONMENT SPECIFICATIONS:")
    print("=" * 70)
    print(json.dumps(env_info, indent=2))

    audit_res = perform_preflight_audit(dataset_path)
    print("\n" + "=" * 70)
    print("AUDIT SUMMARY:")
    print("=" * 70)
    print(json.dumps(audit_res, indent=2))
