import os
from pathlib import Path
import cv2
import numpy as np
import yaml
from skimage.feature import hog
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import time

BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_YAML = BASE_DIR / 'datasets' / 'master_combined_dataset' / 'data.yaml'

def load_data_from_yolo_dir(images_dir, labels_dir, img_size=(64, 64)):
    """
    Load images and YOLO labels, crop the bounding box, resize, convert to grayscale,
    and compute HOG features.
    """
    features = []
    labels = []
    
    images_dir = Path(images_dir)
    labels_dir = Path(labels_dir)
    
    if not images_dir.exists() or not labels_dir.exists():
        print(f"[WARNING] Directory not found: {images_dir} or {labels_dir}")
        return np.array(features), np.array(labels)

    # Valid image extensions
    valid_exts = {'.jpg', '.jpeg', '.png'}
    
    for img_path in images_dir.iterdir():
        if img_path.suffix.lower() not in valid_exts:
            continue
            
        label_path = labels_dir / f"{img_path.stem}.txt"
        if not label_path.exists():
            continue
            
        # Read image
        img = cv2.imread(str(img_path))
        if img is None:
            continue
            
        h, w = img.shape[:2]
        
        # Read label (taking the first bounding box if there are multiple)
        with open(label_path, 'r') as f:
            lines = f.readlines()
            if not lines:
                continue
            
            parts = lines[0].strip().split()
            if len(parts) >= 5:
                class_id = int(parts[0])
                x_center = float(parts[1]) * w
                y_center = float(parts[2]) * h
                box_width = float(parts[3]) * w
                box_height = float(parts[4]) * h
                
                x1 = int(max(0, x_center - box_width / 2))
                y1 = int(max(0, y_center - box_height / 2))
                x2 = int(min(w, x_center + box_width / 2))
                y2 = int(min(h, y_center + box_height / 2))
                
                # Crop face
                face_crop = img[y1:y2, x1:x2]
                
                # If crop is valid
                if face_crop.size > 0:
                    # Resize to fixed size
                    face_resized = cv2.resize(face_crop, img_size)
                    # Convert to grayscale
                    face_gray = cv2.cvtColor(face_resized, cv2.COLOR_BGR2GRAY)
                    
                    # Extract HOG features
                    # Parameters tuned for 64x64 face images
                    hog_feat = hog(
                        face_gray, 
                        orientations=9, 
                        pixels_per_cell=(8, 8), 
                        cells_per_block=(2, 2), 
                        block_norm='L2-Hys',
                        visualize=False
                    )
                    
                    features.append(hog_feat)
                    labels.append(class_id)
                    
    return np.array(features), np.array(labels)

def main():
    print("=" * 60)
    print("  TAHAP 5: Training & Evaluasi KNN sebagai Baseline")
    print("=" * 60)
    
    if not DATA_YAML.exists():
        print(f"[ERROR] {DATA_YAML} tidak ditemukan.")
        return
        
    with open(DATA_YAML, 'r') as f:
        data_cfg = yaml.safe_load(f)
        
    class_names = data_cfg.get('names', [])
    if isinstance(class_names, dict):
        # Convert dict to list ordered by key
        class_names = [class_names[k] for k in sorted(class_names.keys())]
        
    print(f"Mendeteksi {len(class_names)} kelas: {class_names}")
    
    dataset_dir = DATA_YAML.parent
    train_images_dir = dataset_dir / 'images' / 'train'
    train_labels_dir = dataset_dir / 'labels' / 'train'
    test_images_dir = dataset_dir / 'images' / 'test'
    test_labels_dir = dataset_dir / 'labels' / 'test'
    
    # 1. Ekstraksi Fitur Train
    print("\n[1/4] Membaca dan mengekstrak fitur HOG dari Data Latih (Train)...")
    start_t = time.time()
    X_train, y_train = load_data_from_yolo_dir(train_images_dir, train_labels_dir)
    print(f"      Selesai dalam {time.time() - start_t:.2f} detik. Total: {len(X_train)} sampel.")
    
    # 2. Ekstraksi Fitur Test
    print("\n[2/4] Membaca dan mengekstrak fitur HOG dari Data Uji (Test)...")
    start_t = time.time()
    X_test, y_test = load_data_from_yolo_dir(test_images_dir, test_labels_dir)
    print(f"      Selesai dalam {time.time() - start_t:.2f} detik. Total: {len(X_test)} sampel.")
    
    if len(X_train) == 0 or len(X_test) == 0:
        print("[ERROR] Data train atau test kosong. Periksa direktori.")
        return
        
    # 3. Training KNN
    print("\n[3/4] Melatih K-Nearest Neighbors (K=5)...")
    start_t = time.time()
    knn = KNeighborsClassifier(n_neighbors=5, metric='euclidean')
    knn.fit(X_train, y_train)
    train_time = time.time() - start_t
    print(f"      Training selesai dalam {train_time:.4f} detik.")
    
    # 4. Evaluasi Model
    print("\n[4/4] Mengevaluasi model pada Data Uji (Test)...")
    start_t = time.time()
    y_pred = knn.predict(X_test)
    test_time = time.time() - start_t
    inference_time_per_img = (test_time / len(X_test)) * 1000 # in ms
    print(f"      Waktu Inferensi Rata-rata: {inference_time_per_img:.2f} ms per citra.")
    
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=class_names, digits=4)
    conf_mat = confusion_matrix(y_test, y_pred)
    
    print("\n" + "=" * 60)
    print("  CLASSIFICATION REPORT (KNN + HOG)")
    print("=" * 60)
    print(f"Akurasi Keseluruhan: {acc:.4f}\n")
    print(report)
    print("Confusion Matrix:")
    print(conf_mat)
    
    # --- SIMPAN HASIL KE FOLDER ---
    OUTPUT_DIR = BASE_DIR / 'runs' / 'evaluation' / 'knn_baseline'
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Simpan report ke txt
    report_path = OUTPUT_DIR / 'knn_classification_report.txt'
    with open(report_path, 'w') as f:
        f.write("=" * 60 + "\n")
        f.write("  CLASSIFICATION REPORT (KNN + HOG)\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Akurasi Keseluruhan: {acc:.4f}\n\n")
        f.write(report + "\n")
        f.write("Confusion Matrix:\n")
        f.write(str(conf_mat) + "\n\n")
        f.write(f"Waktu Latih: {train_time:.4f} detik\n")
        f.write(f"Waktu Inferensi per citra: {inference_time_per_img:.2f} ms\n")
        
    # Simpan confusion matrix ke CSV agar mudah diolah
    import pandas as pd
    cm_df = pd.DataFrame(conf_mat, index=class_names, columns=class_names)
    cm_path = OUTPUT_DIR / 'knn_confusion_matrix.csv'
    cm_df.to_csv(cm_path)
    
    # Simpan plot Confusion Matrix menggunakan Seaborn
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm_df, annot=True, fmt='d', cmap='Reds')
    plt.title(f'Confusion Matrix KNN (Akurasi: {acc:.2%})')
    plt.ylabel('Aktual (True)')
    plt.xlabel('Prediksi (Predicted)')
    plt.tight_layout()
    
    plot_path = OUTPUT_DIR / 'knn_confusion_matrix.png'
    plt.savefig(plot_path, dpi=300)
    plt.close()
    
    print("\n" + "=" * 60)
    print(f"[BERHASIL] HASIL DISIMPAN KE:")
    print(f"   - {report_path}")
    print(f"   - {cm_path}")
    print(f"   - {plot_path}")
    print("=" * 60)

if __name__ == "__main__":
    main()
