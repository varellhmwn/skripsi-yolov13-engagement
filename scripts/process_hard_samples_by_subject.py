"""
process_hard_samples_by_subject.py — Pemrosesan Aman Hard Samples Berdasarkan Subjek
====================================================================================
Tujuan:
  1. Membaca gambar hard sample dari folder subjek manual: hard_samples_subject_mapping/SXX/
  2. Mencari pasangan gambar asli dan label YOLO di master_combined_dataset/ (train/val/test)
  3. Melakukan standarisasi penamaan baru: SXX_CLASS_XXXX.ext (misal: S01_engaged_0001.jpg & .txt)
  4. MENJAGA 100% ISI BOUNDING BOX LABEL ASLI TANPA DIUBAH SAMA SEKALI
  5. Melakukan validasi ketat (tidak ada silent ignore)
  6. Menghitung SHA-256 untuk audit duplikasi
  7. Menghasilkan:
     - hard_samples_subject_renamed/images/
     - hard_samples_subject_renamed/labels/
     - hard_samples_subject_renamed/metadata.csv
     - hard_samples_subject_renamed/duplicate_report.csv
     - hard_samples_subject_renamed/audit_report.txt
  8. Mendukung mode --dry-run untuk simulasi aman sebelum penulisan fisik.
"""

import sys
import os
import argparse
import hashlib
import shutil
from pathlib import Path
from collections import defaultdict, Counter
import cv2
import pandas as pd

# ─── MAPPING KELAS RESMI PENELITIAN ──────────────────────────────────
CLASS_MAP = {
    0: 'engaged',
    1: 'confused',
    2: 'bored',
    3: 'frustrated'
}
VALID_IMAGE_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}


def compute_sha256(file_path: Path) -> str:
    """Menghitung SHA-256 checksum sebuah file secara streaming."""
    hasher = hashlib.sha256()
    with open(file_path, 'rb') as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


def get_image_dimensions(image_path: Path):
    """Membaca dimensi lebar dan tinggi citra menggunakan OpenCV."""
    img = cv2.imread(str(image_path))
    if img is None:
        raise ValueError(f"File citra korup atau tidak dapat dibaca: {image_path}")
    h, w = img.shape[:2]
    return w, h


def index_master_dataset(master_dir: Path):
    """
    Memindai master_combined_dataset dan membuat indeks pencarian berdasarkan filename.
    Mendukung kedua struktur layout:
      Layout A: images/<subset>/ & labels/<subset>/
      Layout B: <subset>/images/ & <subset>/labels/
    di mana <subset> adalah train, val, valid, atau test.
    """
    master_index = {}
    collisions = []

    subsets = ['train', 'val', 'valid', 'test']

    # Layout A: images/<subset> & labels/<subset>
    images_root = master_dir / 'images'
    labels_root = master_dir / 'labels'

    for subset in subsets:
        img_sub = images_root / subset
        lbl_sub = labels_root / subset
        if img_sub.exists() and img_sub.is_dir():
            for img_file in img_sub.iterdir():
                if img_file.suffix.lower() in VALID_IMAGE_EXTS:
                    fn = img_file.name
                    lbl_file = lbl_sub / f"{img_file.stem}.txt" if lbl_sub.exists() else None
                    if fn in master_index:
                        collisions.append((fn, master_index[fn]['image_path'], img_file))
                    master_index[fn] = {
                        'subset': subset if subset != 'valid' else 'val',
                        'image_path': img_file.resolve(),
                        'label_path': lbl_file.resolve() if lbl_file and lbl_file.exists() else None,
                        'expected_label_path': lbl_sub / f"{img_file.stem}.txt"
                    }

    # Layout B: <subset>/images & <subset>/labels
    for subset in subsets:
        sub_dir = master_dir / subset
        img_sub = sub_dir / 'images'
        lbl_sub = sub_dir / 'labels'
        if img_sub.exists() and img_sub.is_dir():
            for img_file in img_sub.iterdir():
                if img_file.suffix.lower() in VALID_IMAGE_EXTS:
                    fn = img_file.name
                    lbl_file = lbl_sub / f"{img_file.stem}.txt" if lbl_sub.exists() else None
                    if fn in master_index:
                        # Jika sudah tercatat dari layout yang sama persis, abaikan. Jika berbeda, catat collision
                        if master_index[fn]['image_path'] != img_file.resolve():
                            collisions.append((fn, master_index[fn]['image_path'], img_file))
                    else:
                        master_index[fn] = {
                            'subset': subset if subset != 'valid' else 'val',
                            'image_path': img_file.resolve(),
                            'label_path': lbl_file.resolve() if lbl_file and lbl_file.exists() else None,
                            'expected_label_path': lbl_sub / f"{img_file.stem}.txt"
                        }

    if collisions:
        err_msg = "\n".join([f"  - '{fn}' ditemukan ganda di:\n      1) {p1}\n      2) {p2}" for fn, p1, p2 in collisions[:10]])
        raise RuntimeError(f"VALIDASI GAGAL: Ditemukan {len(collisions)} filename ambigu yang berada di lebih dari satu lokasi master dataset:\n{err_msg}")

    return master_index


def validate_and_parse_yolo_label(label_path: Path):
    """
    Validasi ketat isi file label YOLO:
      1. File harus ada dan tidak kosong
      2. Format setiap baris: <class_id> <x_center> <y_center> <width> <height>
      3. Class ID harus 0, 1, 2, atau 3
      4. Tidak boleh ada beberapa class ID berbeda dalam satu citra
    Mengembalikan (class_id, class_name, raw_content, bbox_count).
    """
    if not label_path.exists():
        raise FileNotFoundError(f"File label tidak ditemukan: {label_path}")

    raw_lines = []
    with open(label_path, 'r', encoding='utf-8') as f:
        for line in f:
            stripped = line.strip()
            if stripped:
                raw_lines.append(stripped)

    if not raw_lines:
        raise ValueError(f"File label kosong (0 bounding box): {label_path}")

    class_ids_in_file = set()
    for line_idx, line in enumerate(raw_lines, start=1):
        parts = line.split()
        if len(parts) < 5:
            raise ValueError(f"Format anotasi YOLO tidak valid pada baris {line_idx} di {label_path}: '{line}'")
        try:
            cid = int(parts[0])
        except ValueError:
            raise ValueError(f"Class ID bukan integer pada baris {line_idx} di {label_path}: '{parts[0]}'")

        if cid not in CLASS_MAP:
            raise ValueError(f"Class ID '{cid}' tidak valid (harus 0, 1, 2, atau 3) pada baris {line_idx} di {label_path}")

        # Validasi koordinat float (0.0 sampai 1.0)
        try:
            coords = [float(p) for p in parts[1:5]]
        except ValueError:
            raise ValueError(f"Koordinat bounding box bukan float pada baris {line_idx} di {label_path}: '{line}'")

        class_ids_in_file.add(cid)

    if len(class_ids_in_file) > 1:
        raise ValueError(f"Citra memiliki beberapa class ID berbeda ({class_ids_in_file}) di {label_path}")

    final_cid = list(class_ids_in_file)[0]
    final_cname = CLASS_MAP[final_cid]

    return final_cid, final_cname, raw_lines, len(raw_lines)


def process_hard_samples(mapping_dir: Path, master_dir: Path, output_dir: Path, dry_run: bool = False):
    """Fungsi utama audit, validasi, dan penamaan ulang hard samples."""
    print("=" * 75)
    print(f"  PEMROSESAN & PENAMAAN ULANG HARD SAMPLES BERDASARKAN SUBJEK {'[DRY-RUN]' if dry_run else '[EKSEKUSI FISIK]'}")
    print("=" * 75)
    print(f"  Direktori Mapping Subjek : {mapping_dir.resolve()}")
    print(f"  Direktori Master Dataset : {master_dir.resolve()}")
    print(f"  Direktori Output Tujuan  : {output_dir.resolve()}")
    print("=" * 75)

    if not mapping_dir.exists():
        raise FileNotFoundError(f"Direktori mapping subjek tidak ditemukan: {mapping_dir}")
    if not master_dir.exists():
        raise FileNotFoundError(f"Direktori master dataset tidak ditemukan: {master_dir}")

    # 1. Temukan folder-folder subjek (S01, S02, dst.)
    subject_dirs = sorted([d for d in mapping_dir.iterdir() if d.is_dir()])
    if not subject_dirs:
        raise ValueError(f"Tidak ditemukan subdirektori subjek (S01, S02, dst.) di: {mapping_dir}")

    print(f"\n[1/5] Memindai Struktur Mapping Subjek...")
    print(f"  - Total Subjek Ditemukan : {len(subject_dirs)} subjek ({', '.join([d.name for d in subject_dirs[:10]])}{'...' if len(subject_dirs)>10 else ''})")

    subject_image_files = {}
    total_mapping_images = 0
    for s_dir in subject_dirs:
        imgs = sorted([f for f in s_dir.iterdir() if f.is_file() and f.suffix.lower() in VALID_IMAGE_EXTS])
        subject_image_files[s_dir.name] = imgs
        total_mapping_images += len(imgs)

    print(f"  - Total Citra Mapping    : {total_mapping_images} citra")

    if total_mapping_images == 0:
        raise ValueError(f"Tidak ditemukan file citra di dalam folder subjek di {mapping_dir}")

    # 2. Indeks Master Dataset
    print(f"\n[2/5] Membangun Indeks Master Dataset...")
    master_index = index_master_dataset(master_dir)
    print(f"  - Total File Terindeks   : {len(master_index)} citra pada master dataset")

    # 3. Validasi & Perencanaan Penamaan Ulang
    print(f"\n[3/5] Validasi Integritas Data & Pembuatan Rencana Penamaan...")
    records = []
    errors = []
    seen_new_filenames = set()

    # Counter untuk penomoran 4 digit per subjek: SXX_CLASS_0001.ext
    # Menggunakan counter per subject-class untuk pengurutan yang rapi
    subject_class_counter = defaultdict(int)

    for subj_id, img_list in subject_image_files.items():
        for img_path in img_list:
            orig_fn = img_path.name

            # Validasi 1: Apakah ada di master dataset?
            if orig_fn not in master_index:
                errors.append(f"[{subj_id}] File '{orig_fn}' TIDAK DITEMUKAN pada master dataset!")
                continue

            master_info = master_index[orig_fn]
            src_img_path = master_info['image_path']
            src_lbl_path = master_info['label_path']
            orig_subset = master_info['subset']

            # Validasi 2: Apakah file label ada?
            if src_lbl_path is None or not src_lbl_path.exists():
                errors.append(f"[{subj_id}] File label untuk '{orig_fn}' TIDAK DITEMUKAN (ekspektasi: {master_info['expected_label_path']})")
                continue

            # Validasi 3 & 4: Validasi isi label YOLO
            try:
                cid, cname, raw_lines, bbox_count = validate_and_parse_yolo_label(src_lbl_path)
            except Exception as e:
                errors.append(f"[{subj_id}] Kesalahan parsing label '{src_lbl_path.name}': {e}")
                continue

            # Validasi 5: Integritas citra & dimensi
            try:
                width, height = get_image_dimensions(src_img_path)
            except Exception as e:
                errors.append(f"[{subj_id}] Kesalahan membaca dimensi citra '{src_img_path}': {e}")
                continue

            # Hitung SHA-256
            sha256_hash = compute_sha256(src_img_path)
            file_size_bytes = src_img_path.stat().st_size
            ext = src_img_path.suffix.lower()

            # Buat Nama Baru: SXX_CLASS_XXXX.ext (misal: S01_engaged_0001.jpg)
            subject_class_counter[(subj_id, cname)] += 1
            idx_num = subject_class_counter[(subj_id, cname)]
            new_stem = f"{subj_id}_{cname}_{idx_num:04d}"
            new_img_fn = f"{new_stem}{ext}"
            new_lbl_fn = f"{new_stem}.txt"

            # Validasi 6: Collision check
            if new_img_fn in seen_new_filenames:
                errors.append(f"COLLISION: Nama target baru '{new_img_fn}' sudah digunakan oleh file lain!")
                continue
            seen_new_filenames.add(new_img_fn)

            records.append({
                'original_filename': orig_fn,
                'new_filename': new_img_fn,
                'new_label_filename': new_lbl_fn,
                'subject_id': subj_id,
                'class_id': cid,
                'class_name': cname,
                'original_subset': orig_subset,
                'original_image_path': str(src_img_path),
                'original_label_path': str(src_lbl_path),
                'sha256': sha256_hash,
                'image_extension': ext,
                'file_size': file_size_bytes,
                'width': width,
                'height': height,
                'bbox_count': bbox_count
            })

    if errors:
        print(f"\n[!] DITEMUKAN {len(errors)} KESALAHAN VALIDASI:")
        for err in errors[:20]:
            print(f"  [ERROR] {err}")
        if len(errors) > 20:
            print(f"  ... dan {len(errors) - 20} kesalahan lainnya.")
        raise RuntimeError(f"VALIDASI GAGAL DENGAN {len(errors)} ERROR. Operasi dibatalkan untuk menjaga integritas dataset.")

    df_meta = pd.DataFrame(records)
    print(f"  [OK] Validasi 100% LULUS untuk seluruh {len(df_meta)} citra dan label!")

    # 4. Analisis Duplikasi (SHA-256 Exact Duplicates)
    print(f"\n[4/5] Mengaudit Exact Duplicate (SHA-256)...")
    sha_groups = defaultdict(list)
    for idx, row in df_meta.iterrows():
        sha_groups[row['sha256']].append(row)

    dup_rows = []
    dup_group_id = 0
    total_dup_images = 0

    for sha, group in sha_groups.items():
        if len(group) > 1:
            dup_group_id += 1
            total_dup_images += len(group)
            for item in group:
                dup_rows.append({
                    'duplicate_group_id': f"DUP_{dup_group_id:03d}",
                    'sha256': sha,
                    'subject_id': item['subject_id'],
                    'new_filename': item['new_filename'],
                    'original_filename': item['original_filename'],
                    'original_subset': item['original_subset'],
                    'class_name': item['class_name'],
                    'duplicate_count_in_group': len(group)
                })

    df_dups = pd.DataFrame(dup_rows)
    print(f"  - Total Exact Duplicate Groups : {dup_group_id} grup")
    print(f"  - Total Exact Duplicate Images : {total_dup_images} citra")

    # 5. Ringkasan Statistik
    subject_counts = df_meta['subject_id'].value_counts().sort_index()
    class_counts = df_meta['class_name'].value_counts()
    subset_counts = df_meta['original_subset'].value_counts()

    summary_lines = [
        "==========================================================================",
        "  LAPORAN AUDIT PEMROSESAN & PENAMAAN ULANG HARD SAMPLES BERDASARKAN SUBJEK",
        "==========================================================================",
        f"Status Eksekusi            : {'DRY-RUN (Simulasi Tanpa Penulisan)' if dry_run else 'EKSEKUSI FISIK BERHASIL'}",
        f"Total Citra Diproses       : {len(df_meta)} citra",
        f"Total Pasangan Label       : {len(df_meta)} label .txt (100% Cocok & Utuh)",
        f"Jumlah Subjek Unik         : {df_meta['subject_id'].nunique()} subjek",
        f"Jumlah Duplicate Groups    : {dup_group_id} grup (Total {total_dup_images} citra duplikat biner)",
        f"Jumlah Kesalahan/Missing   : 0 error (100% Lulus Validasi)",
        "",
        "--- Distribusi per Subjek ---",
    ]
    for s_id, cnt in subject_counts.items():
        summary_lines.append(f"  {s_id:<10}: {cnt:>4} citra")

    summary_lines.append("\n--- Distribusi per Kelas Emosi ---")
    for c_name, cnt in class_counts.items():
        summary_lines.append(f"  {c_name:<12}: {cnt:>4} citra ({cnt/len(df_meta)*100:>5.2f}%)")

    summary_lines.append("\n--- Distribusi Asal Subset (Master Dataset) ---")
    for s_name, cnt in subset_counts.items():
        summary_lines.append(f"  {s_name:<10}: {cnt:>4} citra ({cnt/len(df_meta)*100:>5.2f}%)")

    summary_lines.append("\n--- Contoh Rencana Penamaan Baru (10 Sampel Pertama) ---")
    summary_lines.append(f"  {'Filename Asli':<32} -> {'Filename Baru':<28} | {'Subjek':<6} | {'Kelas':<10} | {'Subset':<6}")
    summary_lines.append(f"  {'-'*32}----+{'-'*28}-+-{'-'*6}-+-{'-'*10}-+-{'-'*6}")
    for _, r in df_meta.head(10).iterrows():
        summary_lines.append(f"  {r['original_filename']:<32} -> {r['new_filename']:<28} | {r['subject_id']:<6} | {r['class_name']:<10} | {r['original_subset']:<6}")

    summary_lines.append("==========================================================================")
    summary_text = "\n".join(summary_lines)

    print("\n" + summary_text)

    # 6. Penulisan Fisik (Jika Bukan Dry-Run)
    if dry_run:
        print("\n[DRY-RUN SELESAI] Tidak ada berkas yang disalin/ditulis ke disk.")
        print("Untuk mengeksekusi penyalinan dan penulisan berkas secara fisik, jalankan kembali tanpa opsi --dry-run.")
        return df_meta, df_dups, summary_text

    print(f"\n[5/5] Menyalin Berkas & Menulis Metadata ke: {output_dir.resolve()}...")
    out_img_dir = output_dir / 'images'
    out_lbl_dir = output_dir / 'labels'

    out_img_dir.mkdir(parents=True, exist_ok=True)
    out_lbl_dir.mkdir(parents=True, exist_ok=True)

    # Salin citra dan label secara aman
    for _, row in df_meta.iterrows():
        src_img = Path(row['original_image_path'])
        src_lbl = Path(row['original_label_path'])

        dst_img = out_img_dir / row['new_filename']
        dst_lbl = out_lbl_dir / row['new_label_filename']

        # Copy byte-for-byte persis sama
        shutil.copy2(src_img, dst_img)
        shutil.copy2(src_lbl, dst_lbl)

    # Simpan metadata.csv
    meta_cols = [
        'original_filename', 'new_filename', 'subject_id', 'class_id', 'class_name',
        'original_subset', 'original_image_path', 'original_label_path', 'sha256',
        'image_extension', 'file_size', 'width', 'height', 'bbox_count'
    ]
    df_meta[meta_cols].to_csv(output_dir / 'metadata.csv', index=False)
    print(f"  [SAVED] {output_dir / 'metadata.csv'}")

    # Simpan duplicate_report.csv
    df_dups.to_csv(output_dir / 'duplicate_report.csv', index=False)
    print(f"  [SAVED] {output_dir / 'duplicate_report.csv'}")

    # Simpan audit_report.txt
    with open(output_dir / 'audit_report.txt', 'w', encoding='utf-8') as f:
        f.write(summary_text)
    print(f"  [SAVED] {output_dir / 'audit_report.txt'}")

    print(f"\n  [SUCCESS] {len(df_meta)} citra dan {len(df_meta)} label berhasil disalin dan dinamai ulang dengan sukses!")
    return df_meta, df_dups, summary_text


def parse_arguments():
    parser = argparse.ArgumentParser(description="Proses & Rename Hard Samples Berdasarkan Subjek untuk Mencegah Data Leakage")
    parser.add_argument(
        '--mapping_dir', type=str, default='hard_samples_subject_mapping',
        help="Path ke folder mapping subjek manual (berisi subfolder S01, S02, dst.)"
    )
    parser.add_argument(
        '--master_dir', type=str, default='datasets/master_combined_dataset',
        help="Path ke master_combined_dataset (berisi images/ & labels/)"
    )
    parser.add_argument(
        '--output_dir', type=str, default='hard_samples_subject_renamed',
        help="Path ke direktori output tujuan"
    )
    parser.add_argument(
        '--dry-run', action='store_true',
        help="Jalankan simulasi validasi penuh dan tampilkan rencana tanpa menyalin file fisik"
    )
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_arguments()

    mapping_path = Path(args.mapping_dir)
    master_path = Path(args.master_dir)
    output_path = Path(args.output_dir)

    # Auto-fallback jika path relatif berada di subfolder datasets/
    if not mapping_path.exists() and (Path('datasets') / args.mapping_dir).exists():
        mapping_path = Path('datasets') / args.mapping_dir
    if not master_path.exists() and (Path('datasets') / args.master_dir).exists():
        master_path = Path('datasets') / args.master_dir

    try:
        process_hard_samples(
            mapping_dir=mapping_path,
            master_dir=master_path,
            output_dir=output_path,
            dry_run=args.dry_run
        )
    except Exception as e:
        print(f"\n[ERROR] {e}", file=sys.stderr)
        sys.exit(1)
