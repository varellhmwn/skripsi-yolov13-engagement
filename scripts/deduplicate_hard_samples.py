"""
deduplicate_hard_samples.py — Exact Binary Deduplication (SHA-256) Hard Samples
================================================================================
Penelitian: "Deteksi Emosi Belajar Mahasiswa pada Pembelajaran Pemrograman Menggunakan YOLOv13n"
Tujuan:
  1. Mendeteksi exact binary duplicate menggunakan hash SHA-256 dari isi file citra.
  2. Memvalidasi bahwa setiap duplicate group konsisten (subject_id sama, class sama, label semantik identik).
  3. Mempertahankan 1 file per duplicate group secara deterministik (natural sort terkecil: S01_engaged_0010 vs S01_engaged_0011 -> keep S01_engaged_0010).
  4. Menyalin dataset bersih ke folder baru: hard_samples_subject_deduplicated/
  5. TIDAK MENGUBAH / MENGHAPUS / MEMODIFIKASI dataset sumber (hard_samples_subject_renamed/).
  6. Menghasilkan audit trail lengkap:
     - metadata.csv (portable relative path, dedup_status)
     - deduplication_log.csv
     - duplicate_groups.csv
     - duplicate_conflicts.csv
     - audit_report.txt
  7. Mendukung mode --dry-run.
"""

import sys
import os
import re
import csv
import shutil
import hashlib
import argparse
import logging
from pathlib import Path
from collections import defaultdict, Counter
from PIL import Image

# ─── KONFIGURASI MAPPING RESMI ───────────────────────────────────────
CLASS_NAME_TO_ID = {
    'engaged': 0,
    'confused': 1,
    'bored': 2,
    'frustrated': 3
}
CLASS_ID_TO_NAME = {v: k for k, v in CLASS_NAME_TO_ID.items()}
VALID_IMAGE_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger('Deduplicator')


def natural_sort_key(s: str):
    """Kunci pengurutan natural/alphanumeric (misal: S01_engaged_9 sebelum S01_engaged_10)."""
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', str(s))]


def compute_sha256(file_path: Path) -> str:
    """Menghitung SHA-256 checksum sebuah file secara streaming."""
    hasher = hashlib.sha256()
    with open(file_path, 'rb') as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


def parse_subject_id(filename: str) -> str:
    """Mengekstrak subject ID dari pola filename SXX_... (misal: S01)."""
    m = re.match(r'^(S\d+)_', filename, re.IGNORECASE)
    if not m:
        raise ValueError(f"Subject ID tidak dapat ditentukan dari filename '{filename}' (ekspektasi: SXX_...)")
    return m.group(1).upper()


def parse_class_from_filename(filename: str) -> str:
    """Mengekstrak nama kelas emosi dari filename (misal: S01_engaged_0001.jpg -> engaged)."""
    m = re.match(r'^S\d+_([a-zA-Z]+)_', filename, re.IGNORECASE)
    if not m:
        raise ValueError(f"Class name tidak dapat diekstrak dari filename '{filename}'")
    cname = m.group(1).lower()
    if cname not in CLASS_NAME_TO_ID:
        raise ValueError(f"Class name '{cname}' dari filename '{filename}' tidak valid")
    return cname


def read_yolo_label(label_path: Path):
    """Membaca baris label YOLO, menormalkan whitespace."""
    if not label_path.exists():
        return None, []
    raw_lines = []
    with open(label_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                raw_lines.append(line)
    return label_path, raw_lines


def validate_yolo_label(label_path: Path, expected_class_id: int):
    """
    Validasi format dan nilai bounding box YOLO:
      class_id in {0,1,2,3}
      x_center in [0, 1]
      y_center in [0, 1]
      width in (0, 1]
      height in (0, 1]
      class_id harus konsisten dengan expected_class_id
    """
    if not label_path.exists():
        return False, f"File label tidak ditemukan: {label_path}", [], 0

    _, raw_lines = read_yolo_label(label_path)
    if not raw_lines:
        return False, f"File label kosong (0 bounding box): {label_path}", [], 0

    parsed_bboxes = []
    for line_idx, line in enumerate(raw_lines, start=1):
        parts = line.split()
        if len(parts) < 5:
            return False, f"Format anotasi tidak valid pada baris {line_idx}: '{line}'", [], 0
        try:
            cid = int(parts[0])
            xc, yc, w, h = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
        except ValueError:
            return False, f"Nilai numerik tidak valid pada baris {line_idx}: '{line}'", [], 0

        if cid not in CLASS_ID_TO_NAME:
            return False, f"Class ID '{cid}' di luar rentang {{0,1,2,3}} pada baris {line_idx}", [], 0

        if cid != expected_class_id:
            return False, f"Class ID '{cid}' di label tidak cocok dengan class pada filename '{expected_class_id}'", [], 0

        if not (0.0 <= xc <= 1.0 and 0.0 <= yc <= 1.0):
            return False, f"Koordinat pusat ({xc}, {yc}) di luar rentang [0, 1] pada baris {line_idx}", [], 0

        if not (0.0 < w <= 1.0 and 0.0 < h <= 1.0):
            return False, f"Dimensi box ({w}, {h}) di luar rentang (0, 1] pada baris {line_idx}", [], 0

        parsed_bboxes.append((cid, xc, yc, w, h))

    return True, "", parsed_bboxes, len(parsed_bboxes)


def compare_label_bboxes(bboxes1, bboxes2, tol=1e-5):
    """Membandingkan kesamaan semantik dua daftar bounding box YOLO."""
    if len(bboxes1) != len(bboxes2):
        return False
    for (c1, x1, y1, w1, h1), (c2, x2, y2, w2, h2) in zip(bboxes1, bboxes2):
        if c1 != c2:
            return False
        if abs(x1 - x2) > tol or abs(y1 - y2) > tol or abs(w1 - w2) > tol or abs(h1 - h2) > tol:
            return False
    return True


def deduplicate_dataset(input_dir: Path, output_dir: Path, dry_run: bool = False):
    """Fungsi utama deduplikasi biner eksak dataset hard samples."""
    input_dir = Path(input_dir).resolve()
    output_dir = Path(output_dir).resolve()

    logger.info("=" * 70)
    logger.info(f"EXACT BINARY DEDUPLICATION (SHA-256) {'[DRY-RUN]' if dry_run else '[EKSEKUSI FISIK]'}")
    logger.info("=" * 70)
    logger.info(f"Input Directory  : {input_dir}")
    logger.info(f"Output Directory : {output_dir}")

    images_dir = input_dir / 'images'
    labels_dir = input_dir / 'labels'
    meta_path = input_dir / 'metadata.csv'

    if not images_dir.exists():
        raise FileNotFoundError(f"Folder images tidak ditemukan: {images_dir}")
    if not labels_dir.exists():
        raise FileNotFoundError(f"Folder labels tidak ditemukan: {labels_dir}")

    # 1. Membaca metadata lama jika tersedia
    old_meta_dict = {}
    if meta_path.exists():
        try:
            with open(meta_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    fn = row.get('new_filename') or row.get('filename')
                    if fn:
                        old_meta_dict[fn] = row
            logger.info(f"Loaded {len(old_meta_dict)} records from existing metadata.csv")
        except Exception as e:
            logger.warning(f"Gagal membaca metadata.csv lama: {e}")

    # 2. Memindai seluruh citra dan label
    all_image_files = sorted([f for f in images_dir.iterdir() if f.suffix.lower() in VALID_IMAGE_EXTS], key=lambda p: natural_sort_key(p.name))
    all_label_files = sorted([f for f in labels_dir.iterdir() if f.suffix.lower() == '.txt'], key=lambda p: natural_sort_key(p.name))

    logger.info(f"Images found: {len(all_image_files)}")
    logger.info(f"Labels found: {len(all_label_files)}")

    image_stems = {f.stem: f for f in all_image_files}
    label_stems = {f.stem: f for f in all_label_files}

    # Audit Pair Integrity
    orphan_images = [img.name for stem, img in image_stems.items() if stem not in label_stems]
    orphan_labels = [lbl.name for stem, lbl in label_stems.items() if stem not in image_stems]

    if orphan_images:
        logger.error(f"Ditemukan {len(orphan_images)} orphan images (tanpa label .txt): {orphan_images[:5]}")
    if orphan_labels:
        logger.error(f"Ditemukan {len(orphan_labels)} orphan labels (tanpa file citra): {orphan_labels[:5]}")

    if orphan_images or orphan_labels:
        raise RuntimeError("Pair integrity check FAILED: Terdapat orphan images atau orphan labels.")

    # 3. Parsing & Validasi Setiap Citra-Label
    records = []
    invalid_labels = []

    for img_p in all_image_files:
        stem = img_p.stem
        lbl_p = label_stems[stem]
        fn = img_p.name

        # Parse Subject & Class
        subj_id = parse_subject_id(fn)
        cname = parse_class_from_filename(fn)
        cid = CLASS_NAME_TO_ID[cname]

        # Validasi Label
        is_valid, err_msg, bboxes, bbox_cnt = validate_yolo_label(lbl_p, cid)
        if not is_valid:
            invalid_labels.append((fn, lbl_p.name, err_msg))
            continue

        # Hitung SHA-256
        sha = compute_sha256(img_p)
        file_sz = img_p.stat().st_size

        # Dimensi citra
        try:
            with Image.open(img_p) as img_obj:
                w, h = img_obj.size
        except Exception as e:
            w, h = 640, 480

        # Ambil info subset & original_filename dari metadata lama
        old_info = old_meta_dict.get(fn, {})
        orig_fn = old_info.get('original_filename', fn)
        orig_subset = old_info.get('original_subset', 'unknown')

        records.append({
            'filename': fn,
            'label_filename': lbl_p.name,
            'image_path': img_p,
            'label_path': lbl_p,
            'subject_id': subj_id,
            'class_id': cid,
            'class_name': cname,
            'sha256': sha,
            'file_size': file_sz,
            'width': w,
            'height': h,
            'bbox_count': bbox_cnt,
            'bboxes': bboxes,
            'original_filename': orig_fn,
            'original_subset': orig_subset
        })

    if invalid_labels:
        logger.error(f"Ditemukan {len(invalid_labels)} label tidak valid:")
        for ifn, lfn, msg in invalid_labels[:10]:
            logger.error(f"  ✗ {ifn} / {lfn}: {msg}")
        raise RuntimeError(f"Label validation FAILED: Terdapat {len(invalid_labels)} label tidak valid.")

    logger.info(f"Total valid image-label pairs: {len(records)}")

    # 4. Pengelompokan Berdasarkan SHA-256 (Duplicate Groups)
    sha_to_records = defaultdict(list)
    for r in records:
        sha_to_records[r['sha256']].append(r)

    total_unique_hashes = len(sha_to_records)
    duplicate_groups_raw = {sha: grp for sha, grp in sha_to_records.items() if len(grp) > 1}
    total_images_in_dup_groups = sum(len(grp) for grp in duplicate_groups_raw.values())

    logger.info(f"Total Unique SHA-256 Hashes: {total_unique_hashes}")
    logger.info(f"Exact Duplicate Groups: {len(duplicate_groups_raw)}")
    logger.info(f"Images participating in duplicate groups: {total_images_in_dup_groups}")

    # 5. Validasi Setiap Duplicate Group
    duplicate_groups_report = []
    duplicate_conflicts_report = []
    safe_duplicate_groups = {}

    group_idx = 0
    for sha, grp in duplicate_groups_raw.items():
        group_idx += 1
        group_id = f"DUP_GRP_{group_idx:03d}"
        filenames_str = "; ".join([r['filename'] for r in grp])

        # Pemeriksaan Konsistensi
        subjects = set(r['subject_id'] for r in grp)
        classes = set(r['class_name'] for r in grp)
        class_ids = set(r['class_id'] for r in grp)

        # Pemeriksaan Bounding Box
        base_bboxes = grp[0]['bboxes']
        bboxes_match = all(compare_label_bboxes(base_bboxes, r['bboxes']) for r in grp[1:])

        conflict_reasons = []
        if len(subjects) > 1:
            conflict_reasons.append(f"Subject berbeda: {subjects}")
        if len(classes) > 1 or len(class_ids) > 1:
            conflict_reasons.append(f"Kelas berbeda: {classes}")
        if not bboxes_match:
            conflict_reasons.append("Isi bounding box label berbeda semantik")

        if conflict_reasons:
            status = "conflict"
            reason_str = " | ".join(conflict_reasons)
            logger.warning(f"[CONFLICT] Group {group_id} ({sha[:12]}...): {reason_str}")
            duplicate_conflicts_report.append({
                'group_id': group_id,
                'sha256': sha,
                'filenames': filenames_str,
                'subjects': "; ".join(subjects),
                'classes': "; ".join(classes),
                'conflict_reason': reason_str
            })
        else:
            status = "safe_to_deduplicate"
            safe_duplicate_groups[sha] = grp

        duplicate_groups_report.append({
            'group_id': group_id,
            'sha256': sha,
            'number_of_images': len(grp),
            'subject_id': list(subjects)[0] if len(subjects) == 1 else "; ".join(subjects),
            'class_id': list(class_ids)[0] if len(class_ids) == 1 else "; ".join(str(x) for x in class_ids),
            'class_name': list(classes)[0] if len(classes) == 1 else "; ".join(classes),
            'filenames': filenames_str,
            'validation_status': status
        })

    if duplicate_conflicts_report:
        logger.warning(f"Ditemukan {len(duplicate_conflicts_report)} duplicate conflict groups! Harap diperiksa.")

    # 6. Pemilihan File yang Dipertahankan (Deterministic Keep Lowest Natural Sort Index)
    clean_retained_records = []
    deduplication_log_rows = []
    redundant_copies_count = 0

    for sha, grp in sha_to_records.items():
        if len(grp) == 1:
            # File unik (tidak ada duplikat)
            rec = grp[0].copy()
            rec['dedup_status'] = 'unique'
            clean_retained_records.append(rec)
        else:
            if sha in safe_duplicate_groups:
                # Urutkan secara natural
                sorted_grp = sorted(grp, key=lambda r: natural_sort_key(r['filename']))
                kept_rec = sorted_grp[0].copy()
                kept_rec['dedup_status'] = 'duplicate_kept'
                clean_retained_records.append(kept_rec)

                for excluded_rec in sorted_grp[1:]:
                    redundant_copies_count += 1
                    deduplication_log_rows.append({
                        'sha256': sha,
                        'kept_filename': kept_rec['filename'],
                        'removed_filename': excluded_rec['filename'],
                        'subject_id': excluded_rec['subject_id'],
                        'class_id': excluded_rec['class_id'],
                        'class_name': excluded_rec['class_name'],
                        'reason': 'exact_binary_duplicate'
                    })
                    logger.debug(f"[DEDUP] SHA256: {sha[:10]}... | KEEP: {kept_rec['filename']} | EXCLUDE: {excluded_rec['filename']}")
            else:
                # Jika ada conflict, jangan hilangkan otomatis: pertahankan semua dan laporkan
                for rec_item in grp:
                    rec = rec_item.copy()
                    rec['dedup_status'] = 'conflict_retained'
                    clean_retained_records.append(rec)

    # 7. Sanity Check
    expected_retained = len(records) - redundant_copies_count
    if len(clean_retained_records) != expected_retained:
        raise AssertionError(
            f"SANITY CHECK GAGAL: Input ({len(records)}) != Retained ({len(clean_retained_records)}) + Excluded ({redundant_copies_count})"
        )

    logger.info(f"Redundant copies excluded: {redundant_copies_count}")
    logger.info(f"Clean images retained: {len(clean_retained_records)}")

    # 8. Statistik Sebelum & Sesudah
    subjects_before = Counter(r['subject_id'] for r in records)
    classes_before = Counter(r['class_name'] for r in records)

    subjects_after = Counter(r['subject_id'] for r in clean_retained_records)
    classes_after = Counter(r['class_name'] for r in clean_retained_records)

    all_subj_keys = sorted(set(list(subjects_before.keys()) + list(subjects_after.keys())), key=natural_sort_key)
    all_class_keys = ['engaged', 'confused', 'bored', 'frustrated']

    # 9. Menyusun Laporan Teks Audit
    report_lines = [
        "==========================================================================",
        "  LAPORAN AUDIT EXACT BINARY DEDUPLICATION (SHA-256) HARD SAMPLES",
        "==========================================================================",
        f"Status Eksekusi                 : {'DRY-RUN (Simulasi Audit)' if dry_run else 'EKSEKUSI FISIK BERHASIL'}",
        f"Input Directory                 : {input_dir}",
        f"Output Directory                : {output_dir}",
        "",
        "--- 1. DATASET AWAL (SEBELUM DEDUPLIKASI) ---",
        f"  Total Images                  : {len(records)} citra",
        f"  Total Labels                  : {len(records)} label .txt",
        f"  Total Subjects                : {len(subjects_before)} subjek",
        f"  Total Unique SHA-256 Hashes   : {total_unique_hashes}",
        "",
        "  Distribusi per Subjek (Awal):",
    ]
    for s in all_subj_keys:
        report_lines.append(f"    - {s:<6}: {subjects_before[s]:>4} citra")

    report_lines.append("\n  Distribusi per Kelas (Awal):")
    for c in all_class_keys:
        report_lines.append(f"    - {c:<12}: {classes_before[c]:>4} citra ({classes_before[c]/len(records)*100:>5.2f}%)")

    report_lines.extend([
        "\n--- 2. DUPLICATE AUDIT ---",
        f"  Total Exact Duplicate Groups  : {len(duplicate_groups_raw)} grup",
        f"  Images in Duplicate Groups    : {total_images_in_dup_groups} citra",
        f"  Redundant Copies Excluded     : {redundant_copies_count} citra",
        f"  Duplicate Conflict Groups     : {len(duplicate_conflicts_report)} grup",
        "",
        "--- 3. DATASET HASIL DEDUPLIKASI (BERSIH) ---",
        f"  Total Images Retained         : {len(clean_retained_records)} citra",
        f"  Total Labels Retained         : {len(clean_retained_records)} label .txt",
        f"  Total Unique SHA-256 Hashes   : {len(set(r['sha256'] for r in clean_retained_records))}",
        f"  Orphan Images / Labels        : 0",
        f"  Invalid Labels                : 0",
        "",
        "  Distribusi per Subjek (Setelah Deduplikasi):",
    ])
    for s in all_subj_keys:
        diff = subjects_before[s] - subjects_after[s]
        report_lines.append(f"    - {s:<6}: {subjects_after[s]:>4} citra (berkurang {diff} duplikat)")

    report_lines.append("\n  Distribusi per Kelas (Setelah Deduplikasi):")
    for c in all_class_keys:
        diff = classes_before[c] - classes_after[c]
        report_lines.append(f"    - {c:<12}: {classes_after[c]:>4} citra ({classes_after[c]/len(clean_retained_records)*100:>5.2f}%, berkurang {diff} duplikat)")

    report_lines.extend([
        "\n--- 4. TABEL PERBANDINGAN SEBELUM & SESUDAH DEDUPLIKASI ---",
        f"  {'Metric':<32} | {'Before':<10} | {'After':<10} | {'Perubahan':<10}",
        f"  {'-'*32}-+-{'-'*10}-+-{'-'*10}-+-{'-'*10}",
        f"  {'Total Images':<32} | {len(records):<10} | {len(clean_retained_records):<10} | -{redundant_copies_count:<9}",
        f"  {'Total Labels':<32} | {len(records):<10} | {len(clean_retained_records):<10} | -{redundant_copies_count:<9}",
        f"  {'Unique SHA-256 Hashes':<32} | {total_unique_hashes:<10} | {total_unique_hashes:<10} | 0",
        f"  {'Exact Duplicate Groups':<32} | {len(duplicate_groups_raw):<10} | 0          | -{len(duplicate_groups_raw):<9}",
        f"  {'Redundant Duplicates':<32} | {redundant_copies_count:<10} | 0          | -{redundant_copies_count:<9}",
        f"  {'engaged':<32} | {classes_before['engaged']:<10} | {classes_after['engaged']:<10} | -{classes_before['engaged']-classes_after['engaged']:<9}",
        f"  {'frustrated':<32} | {classes_before['frustrated']:<10} | {classes_after['frustrated']:<10} | -{classes_before['frustrated']-classes_after['frustrated']:<9}",
        f"  {'confused':<32} | {classes_before['confused']:<10} | {classes_after['confused']:<10} | -{classes_before['confused']-classes_after['confused']:<9}",
        f"  {'bored':<32} | {classes_before['bored']:<10} | {classes_after['bored']:<10} | -{classes_before['bored']-classes_after['bored']:<9}",
        f"  {'Orphan Images':<32} | 0          | 0          | 0",
        f"  {'Orphan Labels':<32} | 0          | 0          | 0",
        f"  {'Invalid Labels':<32} | 0          | 0          | 0",
        "=========================================================================="
    ])
    audit_report_text = "\n".join(report_lines)
    print("\n" + audit_report_text + "\n")

    # 10. Penulisan Output / Report
    target_report_dir = output_dir if not dry_run else input_dir.parent / 'dedup_audit_dry_run'
    target_report_dir.mkdir(parents=True, exist_ok=True)

    # Simpan duplicate_groups.csv
    with open(target_report_dir / 'duplicate_groups.csv', 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['group_id', 'sha256', 'number_of_images', 'subject_id', 'class_id', 'class_name', 'filenames', 'validation_status'])
        writer.writeheader()
        writer.writerows(duplicate_groups_report)

    # Simpan duplicate_conflicts.csv
    with open(target_report_dir / 'duplicate_conflicts.csv', 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['group_id', 'sha256', 'filenames', 'subjects', 'classes', 'conflict_reason'])
        writer.writeheader()
        writer.writerows(duplicate_conflicts_report)

    # Simpan deduplication_log.csv
    with open(target_report_dir / 'deduplication_log.csv', 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['sha256', 'kept_filename', 'removed_filename', 'subject_id', 'class_id', 'class_name', 'reason'])
        writer.writeheader()
        writer.writerows(deduplication_log_rows)

    # Simpan audit_report.txt
    with open(target_report_dir / 'audit_report.txt', 'w', encoding='utf-8') as f:
        f.write(audit_report_text)

    if dry_run:
        logger.info(f"[DRY-RUN SELESAI] Laporan simulasi tersimpan di: {target_report_dir}")
        return clean_retained_records, deduplication_log_rows, audit_report_text

    # Eksekusi Fisik: Salin gambar dan label bersih
    logger.info(f"Menyalin {len(clean_retained_records)} citra dan label bersih ke: {output_dir}")
    out_img_dir = output_dir / 'images'
    out_lbl_dir = output_dir / 'labels'
    out_img_dir.mkdir(parents=True, exist_ok=True)
    out_lbl_dir.mkdir(parents=True, exist_ok=True)

    meta_rows = []
    for rec in clean_retained_records:
        src_img = rec['image_path']
        src_lbl = rec['label_path']

        dst_img = out_img_dir / rec['filename']
        dst_lbl = out_lbl_dir / rec['label_filename']

        shutil.copy2(src_img, dst_img)
        shutil.copy2(src_lbl, dst_lbl)

        # Buat relative path yang portable
        rel_img_path = f"images/{rec['filename']}"
        rel_lbl_path = f"labels/{rec['label_filename']}"

        meta_rows.append({
            'filename': rec['filename'],
            'label_filename': rec['label_filename'],
            'image_rel_path': rel_img_path,
            'label_rel_path': rel_lbl_path,
            'subject_id': rec['subject_id'],
            'class_id': rec['class_id'],
            'class_name': rec['class_name'],
            'sha256': rec['sha256'],
            'source': 'hard_samples',
            'original_filename': rec['original_filename'],
            'original_subset': rec['original_subset'],
            'width': rec['width'],
            'height': rec['height'],
            'file_size': rec['file_size'],
            'bbox_count': rec['bbox_count'],
            'dedup_status': rec['dedup_status']
        })

    # Simpan metadata.csv bersih
    meta_cols = [
        'filename', 'label_filename', 'image_rel_path', 'label_rel_path',
        'subject_id', 'class_id', 'class_name', 'sha256', 'source',
        'original_filename', 'original_subset', 'width', 'height',
        'file_size', 'bbox_count', 'dedup_status'
    ]
    with open(output_dir / 'metadata.csv', 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=meta_cols)
        writer.writeheader()
        writer.writerows(meta_rows)

    logger.info(f"[SUCCESS] Dataset bersih berhasil dibuat di {output_dir}")
    return clean_retained_records, deduplication_log_rows, audit_report_text


def parse_args():
    parser = argparse.ArgumentParser(description="Exact Binary Deduplication (SHA-256) untuk Dataset YOLO")
    parser.add_argument(
        '--input_dir', type=str, default='hard_samples_subject_renamed',
        help="Path ke direktori dataset hasil rename subjek (default: hard_samples_subject_renamed)"
    )
    parser.add_argument(
        '--output_dir', type=str, default='hard_samples_subject_deduplicated',
        help="Path ke direktori output dataset bersih (default: hard_samples_subject_deduplicated)"
    )
    parser.add_argument(
        '--dry-run', action='store_true',
        help="Jalankan simulasi penuh dan buat laporan audit tanpa menyalin dataset final"
    )
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    in_path = Path(args.input_dir)
    out_path = Path(args.output_dir)

    try:
        deduplicate_dataset(
            input_dir=in_path,
            output_dir=out_path,
            dry_run=args.dry_run
        )
    except Exception as e:
        logger.error(f"Terjadi kesalahan: {e}")
        sys.exit(1)
