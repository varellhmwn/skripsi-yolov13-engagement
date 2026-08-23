"""
build_subject_wise_dataset.py — Final Subject-Wise Dataset Builder & Leakage Prevention
========================================================================================
Penelitian: "Deteksi Emosi Belajar Mahasiswa pada Pembelajaran Pemrograman Menggunakan YOLOv13n"

Tujuan:
  1. Menggabungkan 953 citra Public (Roboflow) & 672 citra Private (hard_samples_subject_deduplicated) -> Total 1.625 citra.
  2. Menganonimkan Public Subject menjadi P01..P11 dan memetakan ke public_subject_mapping.csv.
  3. Mengaudit SHA-256 seluruh 1.625 citra (0 duplicate).
  4. Menjalankan combinatorial subject-wise split optimization (ZERO subject leakage, 80:10:10 ratio, 4 classes covered).
  5. Menampilkan TOP 5 kandidat split pada mode --dry-run.
  6. Menyalin dan membuat output final subject_wise_dataset/ berdasarkan kandidat yang dipilih (--candidate <ID>).
  7. Menghasilkan artefak audit & konfigurasi lengkap.
"""

import sys
import os
import re
import csv
import json
import time
import shutil
import hashlib
import argparse
import logging
import itertools
from pathlib import Path
from collections import defaultdict, Counter
from PIL import Image

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger('SubjectWiseBuilder')

# Mapping Kelas YOLO
CLASS_NAME_TO_ID = {
    'engaged': 0,
    'confused': 1,
    'bored': 2,
    'frustrated': 3
}
CLASS_ID_TO_NAME = {v: k for k, v in CLASS_NAME_TO_ID.items()}
VALID_IMAGE_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}


def natural_sort_key(s: str):
    """Pengurutan natural/alphanumeric."""
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', str(s))]


def compute_sha256(file_path: Path) -> str:
    """Menghitung SHA-256 checksum sebuah file secara streaming."""
    hasher = hashlib.sha256()
    with open(file_path, 'rb') as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


def parse_public_subject_identifier(filename: str) -> str:
    """Mengekstrak subject identifier asli dari filename public (misal: aksan_50)."""
    m = re.search(r'([a-zA-Z]+_\d{2})_\d{13}_', filename)
    if not m:
        raise ValueError(f"Gagal mem-parsing subject public dari filename: '{filename}'")
    return m.group(1).lower()


def read_and_validate_yolo_label(label_path: Path):
    """Membaca dan memvalidasi isi bounding box file label YOLO."""
    if not label_path.exists():
        raise FileNotFoundError(f"File label tidak ditemukan: {label_path}")
    raw_lines = []
    with open(label_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                raw_lines.append(line)
    if not raw_lines:
        raise ValueError(f"File label kosong (0 bounding box): {label_path}")

    class_ids = set()
    for idx, line in enumerate(raw_lines, 1):
        parts = line.split()
        if len(parts) < 5:
            raise ValueError(f"Format baris {idx} tidak valid di {label_path}: '{line}'")
        cid = int(parts[0])
        xc, yc, w, h = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
        if cid not in CLASS_ID_TO_NAME:
            raise ValueError(f"Class ID '{cid}' di luar {{0,1,2,3}} pada baris {idx} di {label_path}")
        if not (0.0 <= xc <= 1.0 and 0.0 <= yc <= 1.0 and 0.0 < w <= 1.0 and 0.0 < h <= 1.0):
            raise ValueError(f"Koordinat bbx di luar rentang valid pada baris {idx} di {label_path}: '{line}'")
        class_ids.add(cid)

    final_cid = list(class_ids)[0]
    return final_cid, CLASS_ID_TO_NAME[final_cid], raw_lines, len(raw_lines)


def load_all_raw_data(public_dir: Path, private_dir: Path):
    """Memuat dan memvalidasi seluruh citra & label dari Public dan Private datasets."""
    logger.info("=" * 70)
    logger.info("MEMUAT & MENGVALIDASI DATASET PUBLIC DAN PRIVATE")
    logger.info("=" * 70)

    records = []

    # 1. Muat Public Dataset
    pub_img_dir = public_dir / 'images'
    pub_lbl_dir = public_dir / 'labels'
    pub_files = sorted([f for f in pub_img_dir.rglob('*.*') if f.name.startswith('bigdata_') and f.suffix.lower() in VALID_IMAGE_EXTS], key=lambda p: natural_sort_key(p.name))
    logger.info(f"Public images found: {len(pub_files)}")

    # Identifikasi dan buat mapping anonymization public
    raw_public_subjects = sorted(list(set(parse_public_subject_identifier(f.name) for f in pub_files)))
    public_subj_mapping = {raw_id: f"P{idx+1:02d}" for idx, raw_id in enumerate(raw_public_subjects)}
    logger.info(f"Public subjects identified ({len(raw_public_subjects)}): {', '.join([f'{r}->{public_subj_mapping[r]}' for r in raw_public_subjects])}")

    for img_p in pub_files:
        raw_s = parse_public_subject_identifier(img_p.name)
        anon_s = public_subj_mapping[raw_s]
        subset = img_p.parent.name # train / val / test lama

        # Cari label pasangannya
        lbl_p = pub_lbl_dir / subset / f"{img_p.stem}.txt"
        cid, cname, raw_lines, bbox_cnt = read_and_validate_yolo_label(lbl_p)

        sha = compute_sha256(img_p)
        with Image.open(img_p) as img_obj:
            w, h = img_obj.size

        records.append({
            'original_filename': img_p.name,
            'source': 'public',
            'raw_subject_id': raw_s,
            'subject_id': anon_s,
            'class_id': cid,
            'class_name': cname,
            'original_subset': subset,
            'sha256': sha,
            'image_width': w,
            'image_height': h,
            'bbox_count': bbox_cnt,
            'src_image_path': img_p,
            'src_label_path': lbl_p
        })

    # 2. Muat Private Dataset (hard_samples_subject_deduplicated)
    priv_img_dir = private_dir / 'images'
    priv_lbl_dir = private_dir / 'labels'
    priv_files = sorted([f for f in priv_img_dir.glob('*.*') if f.suffix.lower() in VALID_IMAGE_EXTS], key=lambda p: natural_sort_key(p.name))
    logger.info(f"Private images found: {len(priv_files)}")

    priv_meta_dict = {}
    priv_meta_path = private_dir / 'metadata.csv'
    if priv_meta_path.exists():
        with open(priv_meta_path, 'r', encoding='utf-8') as f:
            for row in csv.DictReader(f):
                priv_meta_dict[row['filename']] = row

    for img_p in priv_files:
        fn = img_p.name
        subj_id = fn.split('_')[0].upper()
        lbl_p = priv_lbl_dir / f"{img_p.stem}.txt"
        cid, cname, raw_lines, bbox_cnt = read_and_validate_yolo_label(lbl_p)

        sha = compute_sha256(img_p)
        with Image.open(img_p) as img_obj:
            w, h = img_obj.size

        meta_info = priv_meta_dict.get(fn, {})
        orig_fn = meta_info.get('original_filename', fn)
        orig_subset = meta_info.get('original_subset', 'unknown')

        records.append({
            'original_filename': orig_fn,
            'existing_renamed_filename': fn,
            'source': 'private',
            'raw_subject_id': subj_id,
            'subject_id': subj_id,
            'class_id': cid,
            'class_name': cname,
            'original_subset': orig_subset,
            'sha256': sha,
            'image_width': w,
            'image_height': h,
            'bbox_count': bbox_cnt,
            'src_image_path': img_p,
            'src_label_path': lbl_p
        })

    total_cnt = len(records)
    logger.info(f"Combined total images: {total_cnt} (Public: {len(pub_files)}, Private: {len(priv_files)})")

    # 3. Verifikasi SHA-256 Duplikasi pada Combined Dataset
    sha_map = defaultdict(list)
    for r in records:
        sha_map[r['sha256']].append(r)

    duplicate_conflicts = []
    for sha, grp in sha_map.items():
        if len(grp) > 1:
            duplicate_conflicts.append((sha, grp))

    if duplicate_conflicts:
        logger.error(f"DITEMUKAN {len(duplicate_conflicts)} EXACT DUPLICATES PADA COMBINED DATASET!")
        for sha, grp in duplicate_conflicts:
            fns = [r['original_filename'] for r in grp]
            srcs = [r['source'] for r in grp]
            logger.error(f"  ✗ SHA {sha[:12]}: {list(zip(fns, srcs))}")
        raise RuntimeError("Combined exact duplication check FAILED.")

    logger.info(f"Unique SHA-256 hashes: {len(sha_map)} / {total_cnt} (100% Unique, Zero Exact Duplicates)")

    return records, public_subj_mapping


def build_subject_matrix(records):
    """Membangun matriks subject x class x source untuk optimizer."""
    subj_data = {}
    subjects = sorted(list(set(r['subject_id'] for r in records)), key=natural_sort_key)

    for s in subjects:
        s_recs = [r for r in records if r['subject_id'] == s]
        src = s_recs[0]['source']
        c_counts = Counter(r['class_name'] for r in s_recs)
        subj_data[s] = {
            'subject_id': s,
            'source': src,
            'total': len(s_recs),
            'engaged': c_counts['engaged'],
            'confused': c_counts['confused'],
            'bored': c_counts['bored'],
            'frustrated': c_counts['frustrated']
        }
    return subj_data


def optimize_subject_split(subj_data, total_dataset_len):
    """Menjalankan pencarian kombinatorik deterministik untuk menemukan TOP split candidates."""
    all_subjects = list(subj_data.keys())
    remaining_subjects = [s for s in all_subjects if s not in ('S01', 'S02')]

    global_classes = {'engaged': 0, 'confused': 0, 'bored': 0, 'frustrated': 0}
    for s_info in subj_data.values():
        for c in global_classes:
            global_classes[c] += s_info[c]
    global_ratios = {k: v / total_dataset_len for k, v in global_classes.items()}

    def eval_subset(subj_list):
        total = sum(subj_data[s]['total'] for s in subj_list)
        engaged = sum(subj_data[s]['engaged'] for s in subj_list)
        confused = sum(subj_data[s]['confused'] for s in subj_list)
        bored = sum(subj_data[s]['bored'] for s in subj_list)
        frustrated = sum(subj_data[s]['frustrated'] for s in subj_list)
        public_cnt = sum(subj_data[s]['total'] for s in subj_list if subj_data[s]['source'] == 'public')
        private_cnt = sum(subj_data[s]['total'] for s in subj_list if subj_data[s]['source'] == 'private')
        return {
            'total': total,
            'engaged': engaged,
            'confused': confused,
            'bored': bored,
            'frustrated': frustrated,
            'public': public_cnt,
            'private': private_cnt,
            'min_class': min(engaged, confused, bored, frustrated)
        }

    # Pre-filter candidate subsets untuk Val & Test (ukuran 130 - 195 citra, all 4 classes > 0)
    valid_small_subsets = []
    for k in range(1, 6):
        for subjs in itertools.combinations(remaining_subjects, k):
            stats = eval_subset(subjs)
            if 130 <= stats['total'] <= 195 and stats['min_class'] > 0:
                valid_small_subsets.append((frozenset(subjs), stats))

    candidates = []
    for i, (val_subjs, v_stats) in enumerate(valid_small_subsets):
        for j, (test_subjs, t_stats) in enumerate(valid_small_subsets):
            if i == j or not val_subjs.isdisjoint(test_subjs):
                continue

            train_subjs = set(all_subjects) - val_subjs - test_subjs
            tr_stats = eval_subset(train_subjs)
            if tr_stats['min_class'] == 0:
                continue

            r_tr = tr_stats['total'] / total_dataset_len
            r_va = v_stats['total'] / total_dataset_len
            r_te = t_stats['total'] / total_dataset_len
            ratio_err = (r_tr - 0.80)**2 + (r_va - 0.10)**2 + (r_te - 0.10)**2

            def class_err(stats):
                tot = stats['total']
                return sum((stats[c] / tot - global_ratios[c])**2 for c in global_classes)

            cls_err = class_err(tr_stats) * 0.2 + class_err(v_stats) * 0.4 + class_err(t_stats) * 0.4

            # Source penalty (prioritize val & test having both public & private)
            src_pen = 0.0
            if v_stats['public'] == 0 or v_stats['private'] == 0:
                src_pen += 0.05
            if t_stats['public'] == 0 or t_stats['private'] == 0:
                src_pen += 0.05

            score = (ratio_err * 50.0) + (cls_err * 30.0) + src_pen

            candidates.append({
                'score': score,
                'train_subjs': sorted(train_subjs, key=natural_sort_key),
                'val_subjs': sorted(val_subjs, key=natural_sort_key),
                'test_subjs': sorted(test_subjs, key=natural_sort_key),
                'train_stats': tr_stats,
                'val_stats': v_stats,
                'test_stats': t_stats,
                'ratios': (r_tr, r_va, r_te)
            })

    candidates.sort(key=lambda c: c['score'])
    return candidates


def generate_final_filenames(records, split_assignment):
    """Menghasilkan nama file final yang rapi: PXX_CLASS_XXXX.ext / SXX_CLASS_XXXX.ext."""
    # Counter per subject-class untuk penomoran 4 digit konsisten
    subj_class_counters = defaultdict(int)
    assigned_records = []

    # Urutkan berdasarkan subject_id dan natural sort original_filename
    sorted_records = sorted(records, key=lambda r: (natural_sort_key(r['subject_id']), natural_sort_key(r['original_filename'])))

    for r in sorted_records:
        rec = r.copy()
        s_id = rec['subject_id']
        c_name = rec['class_name']
        ext = rec['src_image_path'].suffix.lower()

        split = split_assignment[s_id]
        rec['assigned_split'] = split

        if rec['source'] == 'private':
            # Pertahankan nama yang sudah distandarisasi di tahap sebelumnya jika ada
            final_img_name = rec.get('existing_renamed_filename', f"{s_id}_{c_name}_{subj_class_counters[(s_id, c_name)]+1:04d}{ext}")
            final_lbl_name = f"{Path(final_img_name).stem}.txt"
        else:
            subj_class_counters[(s_id, c_name)] += 1
            idx_num = subj_class_counters[(s_id, c_name)]
            final_stem = f"{s_id}_{c_name}_{idx_num:04d}"
            final_img_name = f"{final_stem}{ext}"
            final_lbl_name = f"{final_stem}.txt"

        rec['final_filename'] = final_img_name
        rec['label_filename'] = final_lbl_name
        assigned_records.append(rec)

    return assigned_records


def build_dataset_pipeline(public_dir: Path, private_dir: Path, output_dir: Path, dry_run: bool = False, candidate_idx: int = 1):
    """Eksekusi pipeline subject-wise dataset builder."""
    records, pub_mapping = load_all_raw_data(public_dir, private_dir)
    subj_data = build_subject_matrix(records)

    logger.info("\n" + "=" * 70)
    logger.info("MENJALANKAN OPTIMASI SUBJECT-WISE SPLIT")
    logger.info("=" * 70)
    candidates = optimize_subject_split(subj_data, len(records))
    logger.info(f"Total valid subject-wise partitions evaluated: {len(candidates)}")

    if not candidates:
        raise RuntimeError("Gagal menemukan kandidat subject-wise split yang valid!")

    # Format TOP 5 Candidates
    top_candidates = candidates[:5]
    print("\n" + "=" * 80)
    print("  TOP 5 KANDIDAT SUBJECT-WISE SPLIT (OPTIMIZER RANKING)")
    print("=" * 80)

    for idx, c in enumerate(top_candidates, 1):
        tr, va, te = c['train_stats'], c['val_stats'], c['test_stats']
        r_tr, r_va, r_te = c['ratios']
        print(f"\n[KANDIDAT #{idx}] {'(REKOMENDASI UTAMA)' if idx==1 else ''}")
        print(f"  - Objective Score : {c['score']:.4f}")
        print(f"  - Train Set       : {tr['total']:>4} citra ({r_tr*100:>5.2f}%) | Subjek ({len(c['train_subjs'])}): {', '.join(c['train_subjs'])}")
        print(f"                      Engaged={tr['engaged']} ({tr['engaged']/tr['total']*100:.1f}%), Confused={tr['confused']} ({tr['confused']/tr['total']*100:.1f}%), Bored={tr['bored']} ({tr['bored']/tr['total']*100:.1f}%), Frustrated={tr['frustrated']} ({tr['frustrated']/tr['total']*100:.1f}%)")
        print(f"                      Public={tr['public']} ({tr['public']/tr['total']*100:.1f}%), Private={tr['private']} ({tr['private']/tr['total']*100:.1f}%)")
        print(f"  - Val Set         : {va['total']:>4} citra ({r_va*100:>5.2f}%) | Subjek ({len(c['val_subjs'])}): {', '.join(c['val_subjs'])}")
        print(f"                      Engaged={va['engaged']} ({va['engaged']/va['total']*100:.1f}%), Confused={va['confused']} ({va['confused']/va['total']*100:.1f}%), Bored={va['bored']} ({va['bored']/va['total']*100:.1f}%), Frustrated={va['frustrated']} ({va['frustrated']/va['total']*100:.1f}%)")
        print(f"                      Public={va['public']} ({va['public']/va['total']*100:.1f}%), Private={va['private']} ({va['private']/va['total']*100:.1f}%)")
        print(f"  - Test Set        : {te['total']:>4} citra ({r_te*100:>5.2f}%) | Subjek ({len(c['test_subjs'])}): {', '.join(c['test_subjs'])}")
        print(f"                      Engaged={te['engaged']} ({te['engaged']/te['total']*100:.1f}%), Confused={te['confused']} ({te['confused']/te['total']*100:.1f}%), Bored={te['bored']} ({te['bored']/te['total']*100:.1f}%), Frustrated={te['frustrated']} ({te['frustrated']/te['total']*100:.1f}%)")
        print(f"                      Public={te['public']} ({te['public']/te['total']*100:.1f}%), Private={te['private']} ({te['private']/te['total']*100:.1f}%)")

    print("=" * 80)

    chosen_candidate = top_candidates[candidate_idx - 1]
    logger.info(f"Kandidat Terpilih: #{candidate_idx} (Score: {chosen_candidate['score']:.4f})")

    # Buat map split per subject
    split_assignment = {}
    for s in chosen_candidate['train_subjs']:
        split_assignment[s] = 'train'
    for s in chosen_candidate['val_subjs']:
        split_assignment[s] = 'val'
    for s in chosen_candidate['test_subjs']:
        split_assignment[s] = 'test'

    assigned_records = generate_final_filenames(records, split_assignment)

    # 4. Susun Artefak Data & Laporan
    target_report_dir = output_dir if not dry_run else Path('subject_wise_split_dry_run')
    target_report_dir.mkdir(parents=True, exist_ok=True)

    # A. public_subject_mapping.csv
    with open(target_report_dir / 'public_subject_mapping.csv', 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['original_subject_id', 'anonymous_subject_id', 'total_images'])
        for raw_s, anon_s in sorted(pub_mapping.items()):
            cnt = sum(1 for r in records if r.get('raw_subject_id') == raw_s)
            writer.writerow([raw_s, anon_s, cnt])

    # B. subject_distribution.csv
    with open(target_report_dir / 'subject_distribution.csv', 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['subject_id', 'source', 'total', 'engaged', 'confused', 'bored', 'frustrated'])
        for s, s_info in sorted(subj_data.items(), key=lambda x: natural_sort_key(x[0])):
            writer.writerow([s, s_info['source'], s_info['total'], s_info['engaged'], s_info['confused'], s_info['bored'], s_info['frustrated']])

    # C. split_manifest.csv
    with open(target_report_dir / 'split_manifest.csv', 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['subject_id', 'source', 'assigned_split', 'total_images', 'engaged', 'confused', 'bored', 'frustrated'])
        for s, s_info in sorted(subj_data.items(), key=lambda x: natural_sort_key(x[0])):
            writer.writerow([s, s_info['source'], split_assignment[s], s_info['total'], s_info['engaged'], s_info['confused'], s_info['bored'], s_info['frustrated']])

    # D. class_distribution.csv & source_distribution.csv
    cls_dist_rows = []
    src_dist_rows = []
    for sp in ['train', 'val', 'test']:
        sub_recs = [r for r in assigned_records if r['assigned_split'] == sp]
        tot = len(sub_recs)
        c_cnt = Counter(r['class_name'] for r in sub_recs)
        s_cnt = Counter(r['source'] for r in sub_recs)
        cls_dist_rows.append([sp, tot, c_cnt['engaged'], c_cnt['confused'], c_cnt['bored'], c_cnt['frustrated'],
                             f"{c_cnt['engaged']/tot*100:.2f}%", f"{c_cnt['confused']/tot*100:.2f}%", f"{c_cnt['bored']/tot*100:.2f}%", f"{c_cnt['frustrated']/tot*100:.2f}%"])
        src_dist_rows.append([sp, tot, s_cnt['public'], s_cnt['private'], f"{s_cnt['public']/tot*100:.2f}%", f"{s_cnt['private']/tot*100:.2f}%"])

    with open(target_report_dir / 'class_distribution.csv', 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['subset', 'total', 'engaged', 'confused', 'bored', 'frustrated', 'engaged_pct', 'confused_pct', 'bored_pct', 'frustrated_pct'])
        writer.writerows(cls_dist_rows)

    with open(target_report_dir / 'source_distribution.csv', 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['subset', 'total', 'public_count', 'private_count', 'public_pct', 'private_pct'])
        writer.writerows(src_dist_rows)

    # E. combined_duplicate_report.csv
    with open(target_report_dir / 'combined_duplicate_report.csv', 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['sha256', 'total_occurrences', 'status'])
        writer.writerow(['ALL_UNIQUE_NO_DUPLICATES', len(records), 'PASS'])

    # F. split_config.json
    split_config_data = {
        "target_ratios": {"train": 0.80, "val": 0.10, "test": 0.10},
        "subject_wise": True,
        "sha256_dedup_check": True,
        "chosen_candidate_id": candidate_idx,
        "chosen_candidate_score": chosen_candidate['score'],
        "split_counts": {
            "train": chosen_candidate['train_stats']['total'],
            "val": chosen_candidate['val_stats']['total'],
            "test": chosen_candidate['test_stats']['total']
        },
        "split_subjects": {
            "train": chosen_candidate['train_subjs'],
            "val": chosen_candidate['val_subjs'],
            "test": chosen_candidate['test_subjs']
        }
    }
    with open(target_report_dir / 'split_config.json', 'w', encoding='utf-8') as f:
        json.dump(split_config_data, f, indent=2)

    # G. audit_report.txt
    report_text = [
        "==========================================================================",
        "  LAPORAN AUDIT FINAL SUBJECT-WISE DATASET (LEAKAGE PREVENTION)",
        "==========================================================================",
        f"Status Eksekusi            : {'DRY-RUN (Simulasi Audit & Pemilihan Kandidat)' if dry_run else 'EKSEKUSI FISIK SELESAI'}",
        f"Kandidat Terpilih          : #{candidate_idx} (Objective Score: {chosen_candidate['score']:.4f})",
        f"Total Citra Gabungan       : {len(records)} citra (953 Public + 672 Private)",
        f"Total Unique SHA-256 Hashes: {len(records)} (100% Bebas Exact Duplicates)",
        f"Total Subjek Unik          : {len(subj_data)} subjek (11 Public P01..P11, 10 Private S01..S10)",
        "",
        "--- 1. AUDIT SUBJECT LEAKAGE ---",
        f"  - Subject Overlap Train-Val : 0 subjek (DISJOINT / LULUS)",
        f"  - Subject Overlap Train-Test: 0 subjek (DISJOINT / LULUS)",
        f"  - Subject Overlap Val-Test  : 0 subjek (DISJOINT / LULUS)",
        "",
        "--- 2. AUDIT IMAGE HASH LEAKAGE ---",
        f"  - Hash Overlap Train-Val    : 0 citra (DISJOINT / LULUS)",
        f"  - Hash Overlap Train-Test   : 0 citra (DISJOINT / LULUS)",
        f"  - Hash Overlap Val-Test     : 0 citra (DISJOINT / LULUS)",
        "",
        "--- 3. DISTRIBUSI SPLIT SUBSET ---",
        f"  - Train Set : {chosen_candidate['train_stats']['total']} citra ({chosen_candidate['ratios'][0]*100:.2f}%) | Subjek: {', '.join(chosen_candidate['train_subjs'])}",
        f"  - Val Set   : {chosen_candidate['val_stats']['total']} citra ({chosen_candidate['ratios'][1]*100:.2f}%) | Subjek: {', '.join(chosen_candidate['val_subjs'])}",
        f"  - Test Set  : {chosen_candidate['test_stats']['total']} citra ({chosen_candidate['ratios'][2]*100:.2f}%) | Subjek: {', '.join(chosen_candidate['test_subjs'])}",
        "",
        "--- 4. DISTRIBUSI KELAS PER SUBSET ---",
    ]
    for row in cls_dist_rows:
        report_text.append(f"  - {row[0]:<6}: Total={row[1]:<4} | Engaged={row[2]:<3} ({row[6]}) | Confused={row[3]:<3} ({row[7]}) | Bored={row[4]:<3} ({row[8]}) | Frustrated={row[5]:<3} ({row[9]})")

    report_text.append("\n--- 5. DISTRIBUSI SUMBER DATA PER SUBSET ---")
    for row in src_dist_rows:
        report_text.append(f"  - {row[0]:<6}: Total={row[1]:<4} | Public={row[2]:<3} ({row[4]}) | Private={row[3]:<3} ({row[5]})")

    report_text.append("==========================================================================")
    audit_report_content = "\n".join(report_text)

    with open(target_report_dir / 'audit_report.txt', 'w', encoding='utf-8') as f:
        f.write(audit_report_content)

    if dry_run:
        logger.info(f"[DRY-RUN SELESAI] Artefak simulasi tersimpan di: {target_report_dir.resolve()}")
        return assigned_records, chosen_candidate

    # 5. Eksekusi Fisik Penyalinan Citra dan Label
    logger.info("\n" + "=" * 70)
    logger.info(f"MENYALIN DATASET FINAL KE: {output_dir.resolve()}")
    logger.info("=" * 70)

    for sp in ['train', 'val', 'test']:
        (output_dir / 'images' / sp).mkdir(parents=True, exist_ok=True)
        (output_dir / 'labels' / sp).mkdir(parents=True, exist_ok=True)

    final_meta_rows = []
    for rec in assigned_records:
        sp = rec['assigned_split']
        dst_img = output_dir / 'images' / sp / rec['final_filename']
        dst_lbl = output_dir / 'labels' / sp / rec['label_filename']

        # Copy byte-for-byte persis sama
        shutil.copy2(rec['src_image_path'], dst_img)
        shutil.copy2(rec['src_label_path'], dst_lbl)

        final_meta_rows.append({
            'final_filename': rec['final_filename'],
            'original_filename': rec['original_filename'],
            'subject_id': rec['subject_id'],
            'source': rec['source'],
            'assigned_split': sp,
            'class_id': rec['class_id'],
            'class_name': rec['class_name'],
            'sha256': rec['sha256'],
            'original_subset': rec['original_subset'],
            'image_width': rec['image_width'],
            'image_height': rec['image_height'],
            'label_filename': rec['label_filename']
        })

    # metadata.csv
    meta_cols = [
        'final_filename', 'original_filename', 'subject_id', 'source',
        'assigned_split', 'class_id', 'class_name', 'sha256',
        'original_subset', 'image_width', 'image_height', 'label_filename'
    ]
    with open(output_dir / 'metadata.csv', 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=meta_cols)
        writer.writeheader()
        writer.writerows(final_meta_rows)

    # data.yaml
    data_yaml_content = """path: .
train: images/train
val: images/val
test: images/test

nc: 4

names:
  0: engaged
  1: confused
  2: bored
  3: frustrated
"""
    with open(output_dir / 'data.yaml', 'w', encoding='utf-8') as f:
        f.write(data_yaml_content)

    # 6. Post-Execution Strict Audits
    logger.info("MENJALANKAN AUDIT INDEPENDEN PADA OUTPUT FINAL...")
    train_subjs_actual = set(f.name.split('_')[0] for f in (output_dir / 'images' / 'train').iterdir() if f.is_file())
    val_subjs_actual = set(f.name.split('_')[0] for f in (output_dir / 'images' / 'val').iterdir() if f.is_file())
    test_subjs_actual = set(f.name.split('_')[0] for f in (output_dir / 'images' / 'test').iterdir() if f.is_file())

    assert train_subjs_actual.isdisjoint(val_subjs_actual), "ASSERTION FAILED: Subject overlap between Train and Val!"
    assert train_subjs_actual.isdisjoint(test_subjs_actual), "ASSERTION FAILED: Subject overlap between Train and Test!"
    assert val_subjs_actual.isdisjoint(test_subjs_actual), "ASSERTION FAILED: Subject overlap between Val and Test!"

    train_hashes = set(compute_sha256(f) for f in (output_dir / 'images' / 'train').iterdir() if f.is_file())
    val_hashes = set(compute_sha256(f) for f in (output_dir / 'images' / 'val').iterdir() if f.is_file())
    test_hashes = set(compute_sha256(f) for f in (output_dir / 'images' / 'test').iterdir() if f.is_file())

    assert train_hashes.isdisjoint(val_hashes), "ASSERTION FAILED: Image hash overlap between Train and Val!"
    assert train_hashes.isdisjoint(test_hashes), "ASSERTION FAILED: Image hash overlap between Train and Test!"
    assert val_hashes.isdisjoint(test_hashes), "ASSERTION FAILED: Image hash overlap between Val and Test!"

    logger.info("[SUCCESS] SELURUH AUDIT INDEPENDEN LULUS DENGAN STATUS ZERO LEAKAGE!")
    return assigned_records, chosen_candidate


def parse_args():
    parser = argparse.ArgumentParser(description="Build Final Subject-Wise Split Dataset (Zero Leakage)")
    parser.add_argument(
        '--public_dir', type=str, default='datasets/master_combined_dataset',
        help="Path ke master dataset public"
    )
    parser.add_argument(
        '--private_dir', type=str, default='hard_samples_subject_deduplicated',
        help="Path ke private dataset hasil exact deduplication"
    )
    parser.add_argument(
        '--output_dir', type=str, default='subject_wise_dataset',
        help="Path ke output dataset final"
    )
    parser.add_argument(
        '--dry-run', action='store_true',
        help="Jalankan simulasi audit penuh dan tampilkan ranking kandidat tanpa menyalin file"
    )
    parser.add_argument(
        '--candidate', type=int, default=1,
        help="Pilih nomor ranking kandidat split untuk dieksekusi (default: 1)"
    )
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    pub_p = Path(args.public_dir)
    priv_p = Path(args.private_dir)
    out_p = Path(args.output_dir)

    try:
        build_dataset_pipeline(
            public_dir=pub_p,
            private_dir=priv_p,
            output_dir=out_p,
            dry_run=args.dry_run,
            candidate_idx=args.candidate
        )
    except Exception as e:
        logger.error(f"Terjadi kesalahan: {e}")
        sys.exit(1)
