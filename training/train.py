"""
train.py - ASL Landmark Extraction + MLP Training Script (v3)
=============================================================
Pipeline fitur:
  1. Translasi (wrist = pusat)
  2. Normalisasi rotasi: rotasi tangan agar selalu menghadap arah kanonik
  3. Normalisasi skala (wrist-to-MCP jari tengah)
  4. Angle features (15 sudut ruas jari) — rotation & scale invariant
  5. Distance features (12 jarak antar titik kunci) — scale-normalized
  6. Augmentasi: cermin-X + noise gaussian + variasi skala + rotasi 3D
  7. Label smoothing: mencegah model terlalu overconfident

Output: 90 fitur per sampel (63 koordinat + 15 sudut + 12 jarak)
"""

import os
import sys
import csv
import random
import math
import argparse
import numpy as np
import pandas as pd
import cv2
from tqdm import tqdm

# ── Import eksplisit MediaPipe (wajib, bukan 'import mediapipe as mp') ─────────
import mediapipe.python.solutions.hands as mp_hands

# ── TensorFlow / Keras ──────────────────────────────────────────────────────────
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# ── Konfigurasi ─────────────────────────────────────────────────────────────────
DATASET_DIR    = os.path.join("dataset_asl")
CSV_TEMP       = "landmarks_temp.csv"
MODEL_OUT      = "model_mlp_asl.keras"
CLASSES_OUT    = "classes.npy"
MAX_PER_CLASS  = 800
IMG_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp")
INPUT_DIM      = 90    # 63 koordinat + 15 sudut + 12 jarak

# ── Seed untuk reproducibility ───────────────────────────────────────────────────
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)


# ── Triplet sudut sendi (sama persis dengan index.html) ─────────────────────────
ANGLE_TRIPLETS = [
    (0,  1,  2),  (1,  2,  3),  (2,  3,  4),    # ibu jari
    (0,  5,  6),  (5,  6,  7),  (6,  7,  8),    # telunjuk
    (0,  9,  10), (9,  10, 11), (10, 11, 12),   # tengah
    (0,  13, 14), (13, 14, 15), (14, 15, 16),   # manis
    (0,  17, 18), (17, 18, 19), (18, 19, 20),   # kelingking
]


def compute_angle_features(coords_normalized: list[float]) -> list[float]:
    """
    Menghitung 15 sudut sendi dari 63 koordinat yang sudah dinormalisasi.

    Sudut dihitung di titik tengah B dari triplet (A, B, C):
      angle = acos( dot(B->A, B->C) / (|B->A| * |B->C|) )

    Fitur sudut bersifat ROTATION-INVARIANT dan SCALE-INVARIANT:
    gestur tetap terdeteksi meski tangan miring atau jaraknya berbeda.
    Output: list 15 float dalam radian [0, pi].
    """
    pts = [
        (coords_normalized[i], coords_normalized[i + 1], coords_normalized[i + 2])
        for i in range(0, 63, 3)
    ]

    def vec_angle(a: int, b: int, c: int) -> float:
        v1 = tuple(pts[a][k] - pts[b][k] for k in range(3))
        v2 = tuple(pts[c][k] - pts[b][k] for k in range(3))
        dot  = sum(v1[k] * v2[k] for k in range(3))
        mag1 = math.sqrt(sum(v ** 2 for v in v1)) + 1e-8
        mag2 = math.sqrt(sum(v ** 2 for v in v2)) + 1e-8
        return math.acos(max(-1.0, min(1.0, dot / (mag1 * mag2))))

    return [vec_angle(a, b, c) for a, b, c in ANGLE_TRIPLETS]


# Pasangan titik untuk 12 fitur jarak (SINKRON dengan index.html)
# Format: (index_a, index_b) — semua menggunakan index landmark MediaPipe 0-20
DIST_PAIRS = [
    # Jempol (lm4) ke ujung jari lain
    (4,  8),   # jempol  ↔ telunjuk
    (4,  12),  # jempol  ↔ tengah
    (4,  16),  # jempol  ↔ manis
    (4,  20),  # jempol  ↔ kelingking
    # Antar ujung jari berdekatan
    (8,  12),  # telunjuk ↔ tengah
    (12, 16),  # tengah   ↔ manis
    (16, 20),  # manis    ↔ kelingking
    # Ujung jari ke wrist (lm0 — sudah di-translasi ke 0,0,0)
    (4,  0),   # jempol  ↔ wrist
    (8,  0),   # telunjuk ↔ wrist
    (12, 0),   # tengah   ↔ wrist
    (16, 0),   # manis    ↔ wrist
    (20, 0),   # kelingking↔ wrist
]


def compute_distance_features(coords_normalized: list[float]) -> list[float]:
    """
    Menghitung 12 jarak Euclidean antar pasang titik landmark.

    Input : 63 koordinat (21 titik x,y,z) yang sudah dinormalisasi skala.
    Output: list 12 float (jarak ternormalisasi, scale-invariant karena
            koordinat sudah dibagi ref_dist sebelumnya).

    Pasangan dipilih untuk membedakan gestur mirip:
      - Jempol ke ujung jari lain    → O, D, F, T, S, E
      - Antar ujung jari berdekatan  → U vs V vs R
      - Ujung jari ke wrist          → apakah jari dikepal atau direntang
    """
    pts = [
        (coords_normalized[i], coords_normalized[i + 1], coords_normalized[i + 2])
        for i in range(0, 63, 3)
    ]

    def eucl(a: int, b: int) -> float:
        return math.sqrt(sum((pts[a][k] - pts[b][k]) ** 2 for k in range(3)))

    return [eucl(a, b) for a, b in DIST_PAIRS]


def align_rotation(translated: list[tuple]) -> list[tuple]:
    """
    Normalisasi rotasi: memutar koordinat tangan ke "frame kanonik".

    Tujuan: dua gambar yang sama gesturenya tapi kamera/posisi berbeda
    akan menghasilkan koordinat yang hampir identik setelah normalisasi ini.

    Metode:
      1. Vektor "palm forward" = dari lm0 (wrist) ke lm9 (tengah MCP).
         Ini mendefinisikan arah "atas" telapak tangan.
      2. Vektor "palm side"   = dari lm0 ke lm17 (kelingking MCP).
         Ini mendefinisikan arah "samping" telapak tangan.
      3. Dari dua vektor ini, bangun basis ortogonal (Gram-Schmidt).
      4. Putar semua titik ke basis tersebut.

    Input/Output: list of 21 tuple (dx, dy, dz) yang sudah ditranslasi.
    """
    # Vektor primer: wrist → index MCP (lm5)
    fx, fy, fz = translated[5]
    mag_f = math.sqrt(fx**2 + fy**2 + fz**2) + 1e-8
    # Normalisasi
    ux, uy, uz = fx / mag_f, fy / mag_f, fz / mag_f   # unit vector "forward"

    # Vektor sekunder: wrist → pinky MCP (lm17), lalu Gram-Schmidt
    sx, sy, sz = translated[17]
    # Kurangi proyeksi ke u agar ortogonal
    dot_su = sx * ux + sy * uy + sz * uz
    rx, ry, rz = sx - dot_su * ux, sy - dot_su * uy, sz - dot_su * uz
    mag_r = math.sqrt(rx**2 + ry**2 + rz**2) + 1e-8
    vx, vy, vz = rx / mag_r, ry / mag_r, rz / mag_r   # unit vector "side"

    # Vektor ketiga: cross product u × v
    wx = uy * vz - uz * vy
    wy = uz * vx - ux * vz
    wz = ux * vy - uy * vx

    # Rotasikan setiap titik ke frame kanonik (matrix multiply)
    rotated = []
    for (px, py, pz) in translated:
        new_x = px * ux + py * uy + pz * uz
        new_y = px * vx + py * vy + pz * vz
        new_z = px * wx + py * wy + pz * wz
        rotated.append((new_x, new_y, new_z))

    return rotated


def extract_landmarks_from_image(image_path: str, hands_model) -> list[float] | None:
    """
    Membaca gambar dan mengembalikan 90 fitur:
      - 63 koordinat (21 titik x,y,z) yang sudah dinormalisasi posisi+skala+rotasi
      - 15 sudut sendi (rotation & scale invariant)
      - 12 jarak antar pasang titik kunci (scale-invariant)

    Pipeline:
      1. Translasi: landmark[0] (wrist) = (0,0,0)
      2. Normalisasi rotasi (align_rotation): frame kanonik berbasis arah telapak
      3. Normalisasi skala: bagi dengan jarak wrist -> Middle MCP (landmark[9])
      4. Angle features: 15 sudut ruas jari
      5. Distance features: 12 jarak antar titik kunci

    Mengembalikan None jika tangan tidak terdeteksi atau pose tidak valid.
    """
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        return None

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    result  = hands_model.process(img_rgb)

    if not result.multi_hand_landmarks:
        return None

    hand = result.multi_hand_landmarks[0]
    is_left_hand = result.multi_handedness[0].classification[0].label == "Left"

    # Langkah 1: Translasi ke wrist sebagai pusat
    base_x = hand.landmark[0].x
    base_y = hand.landmark[0].y
    base_z = hand.landmark[0].z

    translated = []
    for lm in hand.landmark:
        dx = lm.x - base_x
        if is_left_hand:
            dx = -dx
        translated.append((dx, lm.y - base_y, lm.z - base_z))

    # Langkah 2: Normalisasi rotasi (frame kanonik)
    rotated = align_rotation(translated)

    # Langkah 3: Normalisasi skala (jarak wrist ke Middle MCP setelah rotasi)
    ref_dx, ref_dy, ref_dz = rotated[9]
    ref_dist = math.sqrt(ref_dx**2 + ref_dy**2 + ref_dz**2)

    if ref_dist < 1e-6:
        return None

    coords = []
    for dx, dy, dz in rotated:
        coords.extend([dx / ref_dist, dy / ref_dist, dz / ref_dist])

    # Langkah 4: Angle features (15 sudut)
    angles = compute_angle_features(coords)

    # Langkah 5: Distance features (12 jarak)
    distances = compute_distance_features(coords)

    return coords + angles + distances   # total 90 fitur


def rotate_coords_3d(coords: list[float], angle_deg: float, axis: str) -> list[float]:
    """
    Memutar 63 koordinat (21 titik) sebesar `angle_deg` derajat di sekitar `axis`.
    Digunakan untuk augmentasi: mensimulasikan variasi sudut pandang 3D.

    Input : 63 float (koordinat sudah dinormalisasi)
    Output: 63 float baru (koordinat setelah rotasi)
    """
    rad = math.radians(angle_deg)
    c, s = math.cos(rad), math.sin(rad)
    rotated = []
    for i in range(0, 63, 3):
        x, y, z = coords[i], coords[i+1], coords[i+2]
        if axis == 'x':
            rotated.extend([x, c*y - s*z, s*y + c*z])
        elif axis == 'y':
            rotated.extend([c*x + s*z, y, -s*x + c*z])
        else:  # 'z'
            rotated.extend([c*x - s*y, s*x + c*y, z])
    return rotated


def augment_coords(coords: list[float]) -> list[list[float]]:
    """
    Menghasilkan variasi koordinat untuk augmentasi data.

    Teknik:
      - Gaussian noise kecil: simulasi tremor kamera / tracking jitter
      - Variasi skala: simulasi proporsi tangan yang berbeda antar individu
      - Rotasi 3D kecil: simulasi variasi sudut pergelangan tangan

    Input : 90 fitur (63 koordinat + 15 sudut + 12 jarak) yang sudah dinormalisasi.
    Output: list beberapa list[float] augmented.

    Catatan: fitur turunan (sudut & jarak) selalu di-regen ulang dari koordinat
    yang sudah dimodifikasi agar tetap konsisten.
    """
    coord_part = np.array(coords[:63], dtype=np.float32)
    augmented  = []

    # Gaussian noise (2 variasi)
    for sigma in [0.012, 0.020]:
        noisy_coords = (coord_part + np.random.normal(0, sigma, coord_part.shape).astype(np.float32)).tolist()
        noisy_angles = compute_angle_features(noisy_coords)
        noisy_dists  = compute_distance_features(noisy_coords)
        augmented.append(noisy_coords + noisy_angles + noisy_dists)

    # Variasi skala (2 variasi)
    for scale in [0.90, 1.10]:
        scaled_coords = (coord_part * scale).tolist()
        scaled_angles = compute_angle_features(scaled_coords)
        scaled_dists  = compute_distance_features(scaled_coords)
        augmented.append(scaled_coords + scaled_angles + scaled_dists)

    # Rotasi 3D dihapus karena merusak fitur sudut untuk gestur halus seperti G
    # for axis in ['x', 'y', 'z']:
    #     for angle in [-10.0, 10.0]:
    #         rot_coords = rotate_coords_3d(coord_part.tolist(), angle, axis)
    #         rot_angles = compute_angle_features(rot_coords)
    #         rot_dists  = compute_distance_features(rot_coords)
    #         augmented.append(rot_coords + rot_angles + rot_dists)

    return augmented


def mirror_augment_in_memory(
    X: np.ndarray,
    y: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Menghasilkan sampel mirror (tangan kiri) secara in-memory dari training set.

    Berguna saat CSV tidak di-regenerate: kita tambah variasi tangan kiri
    langsung dari data yang sudah ada di RAM, tanpa re-ekstraksi gambar.

    Pipeline per sampel:
      1. Flip koordinat-X (index 0, 3, 6, …, 60 dari 63 total)
      2. Hitung ulang angle features dari koordinat yang sudah di-flip
      3. Hitung ulang distance features dari koordinat yang sudah di-flip
      4. Tambah 1 variasi noise kecil (σ = 0.012) untuk robustness

    Dipanggil HANYA pada X_train (bukan X_val) untuk menghindari data leakage.

    Return: (X_augmented, y_augmented) — training set 3× lebih besar.
    """
    print(f"[INFO] Mirror augmentation in-memory pada {len(X):,} sampel training...")
    new_X: list[list[float]] = []
    new_y: list[int]         = []

    for i in tqdm(range(len(X)), desc="  Mirror aug", leave=False):
        sample = X[i]

        # ── Flip koordinat-X saja (index 0, 3, 6, …, 60) ───────────────────────
        coords_m: list[float] = []
        for j in range(0, 63, 3):
            coords_m.extend([-float(sample[j]),
                              float(sample[j + 1]),
                              float(sample[j + 2])])

        # ── Hitung ulang angle & distance dari koordinat yang sudah di-flip ────
        angles_m = compute_angle_features(coords_m)
        dists_m  = compute_distance_features(coords_m)
        new_X.append(coords_m + angles_m + dists_m)
        new_y.append(int(y[i]))

        # ── Variasi noise kecil dari mirror ────────────────────────────────────
        noise    = np.random.normal(0, 0.012, 63).tolist()
        noisy_m  = [coords_m[k] + noise[k] for k in range(63)]
        na       = compute_angle_features(noisy_m)
        nd       = compute_distance_features(noisy_m)
        new_X.append(noisy_m + na + nd)
        new_y.append(int(y[i]))

    mirror_X = np.array(new_X, dtype=np.float32)
    mirror_y = np.array(new_y, dtype=np.int64)

    X_out = np.vstack([X, mirror_X])
    y_out = np.concatenate([y, mirror_y])
    print(f"[INFO] Training set setelah mirror aug: {len(X_out):,} sampel (3× dari {len(X):,})")
    return X_out, y_out


def build_csv(dataset_dir: str, csv_path: str) -> None:
    """
    Iterasi tiap subfolder (kelas), ekstrak landmark 90 fitur, simpan ke CSV.
    Augmentasi per sampel:
      - noise gaussian (2x)
      - skala (2x)
    Total: 1 asli + 4 augmentasi = 5 baris per gambar.
    """
    classes = sorted([
        d for d in os.listdir(dataset_dir)
        if os.path.isdir(os.path.join(dataset_dir, d))
    ])
    if not classes:
        print(f"[ERROR] Tidak ada subfolder di '{dataset_dir}'. "
              "Pastikan path dataset sudah benar.")
        sys.exit(1)

    print(f"[INFO] Ditemukan {len(classes)} kelas: {classes}")
    print(f"[INFO] Setiap gambar akan menghasilkan 12 sampel (asli + 11 augmentasi)")

    # Header: label + 63 kolom koordinat + 15 kolom sudut + 12 kolom jarak
    coord_cols = [f"{ax}{i}" for i in range(21) for ax in ("x", "y", "z")]
    angle_cols = [f"angle{i}" for i in range(15)]
    dist_cols  = [f"dist{i}"  for i in range(12)]
    header = ["label"] + coord_cols + angle_cols + dist_cols

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)

        with mp_hands.Hands(
            static_image_mode=True,
            max_num_hands=1,
            min_detection_confidence=0.5,
        ) as hands:
            for cls in classes:
                cls_dir = os.path.join(dataset_dir, cls)
                imgs = [
                    p for p in os.listdir(cls_dir)
                    if os.path.splitext(p)[1].lower() in IMG_EXTENSIONS
                ]
                random.shuffle(imgs)
                imgs = imgs[:MAX_PER_CLASS]

                success = 0
                for img_name in tqdm(imgs, desc=f"  Kelas [{cls}]", leave=False):
                    img_path = os.path.join(cls_dir, img_name)
                    features = extract_landmarks_from_image(img_path, hands)
                    if features is None:
                        continue

                    # 1. Sampel asli (90 fitur)
                    writer.writerow([cls] + features)

                    # 2. Augmentasi noise & scale (4 variasi, 3D rotation dihapus)
                    for aug in augment_coords(features):
                        writer.writerow([cls] + aug)

                    success += 1

                total_samples = success * 5
                print(f"  [OK] [{cls}]: {success}/{len(imgs)} gambar "
                      f"({total_samples} sampel total)")

    print(f"\n[INFO] CSV sementara disimpan ke '{csv_path}'")


def build_model(num_classes: int, input_dim: int = INPUT_DIM) -> keras.Model:
    """
    Arsitektur MLP v3 — 4 Dense layer dengan AdamW.

    Perubahan dari v2:
      - Input: 90 fitur (63 koordinat + 15 sudut + 12 jarak)
      - Layer 1 diperlebar ke 512 untuk mengakomodasi fitur baru
      - Arsitektur: 512 -> 256 -> 128 -> 64
    """
    inputs = keras.Input(shape=(input_dim,), name="landmarks_input")

    # Layer 1: Representasi awal
    x = layers.Dense(512, activation="relu")(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.4)(x)

    # Layer 2: Abstraksi menengah
    x = layers.Dense(256, activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.35)(x)

    # Layer 3: Refinement
    x = layers.Dense(128, activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)

    # Layer 4: Kompresi akhir
    x = layers.Dense(64, activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.2)(x)

    outputs = layers.Dense(num_classes, activation="softmax", name="predictions")(x)

    model = keras.Model(inputs=inputs, outputs=outputs, name="ASL_MLP_v3")
    return model


def train(csv_path: str) -> None:
    """Membaca CSV, encode label, split data, compile & latih model."""
    print("\n[INFO] Membaca dataset dari CSV...")
    df = pd.read_csv(csv_path)

    if df.empty:
        print("[ERROR] CSV kosong - tidak ada landmark yang berhasil diekstrak.")
        sys.exit(1)

    X = df.drop(columns=["label"]).values.astype(np.float32)
    y_raw = df["label"].values

    le = LabelEncoder()
    y  = le.fit_transform(y_raw)
    classes = le.classes_
    num_classes = len(classes)

    print(f"[INFO] Total sampel    : {len(X)}")
    print(f"[INFO] Dimensi fitur   : {X.shape[1]} (target: {INPUT_DIM})")
    print(f"[INFO] Jumlah kelas    : {num_classes} -> {classes.tolist()}")

    if X.shape[1] != INPUT_DIM:
        print(f"[ERROR] Dimensi fitur tidak sesuai! "
              f"CSV={X.shape[1]}, TARGET={INPUT_DIM}. "
              f"Hapus CSV lama dan jalankan ulang.")
        sys.exit(1)

    # Simpan mapping kelas
    np.save(CLASSES_OUT, classes)
    print(f"[INFO] Mapping kelas disimpan ke '{CLASSES_OUT}'")

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.15, random_state=SEED, stratify=y
    )

    # Mirror augmentation in-memory dihapus karena semua tangan sudah 
    # dinormalisasi ke representasi tangan kanan saat ekstraksi fitur.
    # X_train, y_train = mirror_augment_in_memory(X_train, y_train)

    # Label Smoothing: label keras (0/1) → lembut (0.1/0.9)
    # Mencegah model terlalu overconfident, meningkatkan generalisasi
    LABEL_SMOOTHING = 0.1
    y_train_cat = keras.utils.to_categorical(y_train, num_classes)
    y_val_cat   = keras.utils.to_categorical(y_val,   num_classes)

    print(f"\n[INFO] Train: {len(X_train)}, Val: {len(X_val)}")
    print(f"[INFO] Label smoothing: {LABEL_SMOOTHING}")

    model = build_model(num_classes)
    model.summary()

    # AdamW = Adam + weight_decay bawaan (regularisasi L2 implisit)
    try:
        optimizer = keras.optimizers.AdamW(learning_rate=1e-3, weight_decay=1e-4)
    except AttributeError:
        # Fallback untuk versi Keras yang lebih lama
        print("[INFO] AdamW tidak tersedia, menggunakan Adam biasa.")
        optimizer = keras.optimizers.Adam(learning_rate=1e-3)

    model.compile(
        optimizer=optimizer,
        loss=keras.losses.CategoricalCrossentropy(label_smoothing=LABEL_SMOOTHING),
        metrics=["accuracy"],
    )

    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_accuracy", patience=12, restore_best_weights=True, verbose=1
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=6, verbose=1
        ),
        keras.callbacks.ModelCheckpoint(
            MODEL_OUT, monitor="val_accuracy", save_best_only=True, verbose=1
        ),
    ]

    print("\n[INFO] Memulai pelatihan...")
    model.fit(
        X_train, y_train_cat,
        validation_data=(X_val, y_val_cat),
        epochs=50,
        batch_size=64,
        callbacks=callbacks,
        verbose=1,
    )

    print(f"\n[OK] Model terbaik disimpan ke '{MODEL_OUT}'")
    print("[OK] Pelatihan selesai!")


# ── Entry Point ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="ASL MLP Trainer v3 — Ekstraksi landmark & pelatihan model."
    )
    parser.add_argument(
        "--skip-extract",
        action="store_true",
        help=(
            f"Lewati ekstraksi landmark dan langsung latih dari '{CSV_TEMP}' "
            "yang sudah ada. Berguna saat CSV masih valid dan hanya ingin "
            "melatih ulang model (misalnya setelah perbaikan augmentasi)."
        ),
    )
    args = parser.parse_args()

    if not os.path.isdir(DATASET_DIR):
        print(f"[ERROR] Folder dataset tidak ditemukan: '{DATASET_DIR}'")
        print("  Pastikan folder 'dataset_asl/' berada satu level dengan train.py")
        sys.exit(1)

    # Langkah 1: Ekstraksi landmark ke CSV
    if args.skip_extract:
        if not os.path.exists(CSV_TEMP):
            print(f"[ERROR] --skip-extract dipilih tapi '{CSV_TEMP}' tidak ditemukan!")
            print(f"  Jalankan tanpa --skip-extract agar CSV dibuat terlebih dahulu.")
            sys.exit(1)
        print(f"[INFO] --skip-extract: melewati ekstraksi, menggunakan '{CSV_TEMP}'.")
        skip_extract = True
    elif os.path.exists(CSV_TEMP):
        answer = input(
            f"[INFO] '{CSV_TEMP}' sudah ada. Lewati ekstraksi dan langsung latih? [y/n]: "
        ).strip().lower()
        skip_extract = answer == "y"
    else:
        skip_extract = False

    if not skip_extract:
        print("\n=== LANGKAH 1: EKSTRAKSI LANDMARK (90 FITUR) ===")
        build_csv(DATASET_DIR, CSV_TEMP)

    # Langkah 2: Pelatihan model (dengan mirror augmentation in-memory)
    print("\n=== LANGKAH 2: PELATIHAN MODEL MLP v3 (+ Mirror Aug In-Memory) ===")
    train(CSV_TEMP)
