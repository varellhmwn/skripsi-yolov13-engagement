import sys
import os
import subprocess
import time
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
LOG_FILE = BASE_DIR / 'pipeline_execution_log.txt'

def log(msg):
    timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
    formatted = f"[{timestamp}] {msg}"
    print(formatted)
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(formatted + '\n')

def run_script(script_name, args=[]):
    script_path = BASE_DIR / 'scripts' / script_name
    log(f"--- MENJALANKAN: {script_name} ---")
    cmd = [sys.executable, str(script_path)] + args
    res = subprocess.run(cmd, text=True)
    if res.returncode != 0:
        log(f"[ERROR] Gagal mengeksekusi {script_name} (Exit code: {res.returncode})")
        sys.exit(res.returncode)
    log(f"--- SELESAI: {script_name} ---")

def main():
    if LOG_FILE.exists():
        os.remove(LOG_FILE)

    log("=" * 60)
    log("   AUTOMATED RETRAINING PIPELINE FROM SCRATCH")
    log("   Computer Vision Student Engagement Detection (YOLOv13n)")
    log("=" * 60)

    start_time = time.time()

    # Tahap 1: Filter Roboflow
    run_script('01_extract_and_filter_roboflow.py')

    # Tahap 2: Build Master Unsplit
    run_script('02_build_master_dataset.py')

    # Tahap 3: Stratified Resplit 80/10/10
    run_script('03_stratified_resplit_80_10_10.py')

    # Prompt check for training
    log("=" * 60)
    log("   TAHAP PREPROCESSING (1-3) BERHASIL SELESAI!")
    log("   Dataset Siap di: datasets_processed/03_master_combined_80_10_10")
    log("=" * 60)

    if '--skip-train' in sys.argv:
        log("[INFO] Parameter --skip-train terdeteksi. Pelatihan dilewati.")
    else:
        log("[INFO] Memulai Tahap 4: Training YOLOv13n (150 epoch)...")
        run_script('04_train_yolov13.py')

    elapsed = time.time() - start_time
    log("=" * 60)
    log(f"   PIPELINE SELESAI DALAM WAKTU: {elapsed / 60:.2f} MENIT")
    log(f"   Log Eksekusi Tersimpan di: {LOG_FILE.resolve()}")
    log("=" * 60)

if __name__ == '__main__':
    main()
