"""
main.py - FastAPI Backend Server untuk ASL Real-Time Detector (v3)
==================================================================
Menerima array JSON [90 angka] dari WebSocket klien:
  - 63 koordinat (21 titik x,y,z) ternormalisasi posisi+rotasi+skala
  - 15 sudut sendi (angle features)
  - 12 jarak antar titik kunci (distance features)

Optimasi v3:
  - INPUT_DIM = 90 (dari 78)
  - Normalisasi rotasi canonical frame di sisi klien
  - Inferensi via model() langsung (lebih cepat dari model.predict())
  - Warm-up TF graph saat startup
  - Temporal smoothing: rata-rata 5 prediksi terakhir per koneksi

Jalankan dengan:
    uvicorn main:app --host 0.0.0.0 --port 8000
"""

import os
import time
import json
import numpy as np
from collections import deque
from contextlib import asynccontextmanager
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import tensorflow as tf
from tensorflow import keras
from spellchecker import SpellChecker

# ── Inisialisasi SpellChecker ───────────────────────────────────────────────────
spell = SpellChecker(distance=1)

# ── Konfigurasi ─────────────────────────────────────────────────────────────────
MODEL_PATH           = "models/model_mlp_asl.h5"
CLASSES_PATH         = "models/classes.npy"
CONFIDENCE_THRESHOLD = 0.50
INPUT_DIM            = 90
SMOOTH_BUFFER_SIZE   = 3

# ── State global (diisi saat startup) ───────────────────────────────────────────
app_state: dict = {
    "model":   None,
    "classes": None,
    "session_stats": {
        "start_time":      time.time(),
        "total_frames":    0,
        "total_predicted": 0,
        "letter_counts":   {},
        "avg_confidence":  0.0,
        "conf_sum":        0.0,
    }
}


# ── Lifespan: muat model, warm-up, verifikasi dimensi ───────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Memuat model dan mapping kelas sebelum server siap menerima request."""
    print("[INFO] Memuat model ASL...")

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Model tidak ditemukan: '{MODEL_PATH}'. "
            "Jalankan train.py terlebih dahulu."
        )
    if not os.path.exists(CLASSES_PATH):
        raise FileNotFoundError(
            f"Mapping kelas tidak ditemukan: '{CLASSES_PATH}'. "
            "Jalankan train.py terlebih dahulu."
        )

    model   = keras.models.load_model(MODEL_PATH, compile=False)
    classes = np.load(CLASSES_PATH, allow_pickle=True)

    model_input_dim = model.input_shape[-1]
    if model_input_dim != INPUT_DIM:
        raise ValueError(
            f"[ERROR] Dimensi model ({model_input_dim}) != INPUT_DIM ({INPUT_DIM}). "
            f"Hapus landmarks_temp.csv dan latih ulang dengan train.py."
        )

    dummy = np.zeros((1, INPUT_DIM), dtype=np.float32)
    _ = model(dummy, training=False)
    print("[INFO] TF graph warm-up selesai.")

    app_state["model"]   = model
    app_state["classes"] = classes

    num_classes = len(classes)
    print(f"[OK] Model dimuat. {num_classes} kelas, input_dim={INPUT_DIM}")
    print(f"     Kelas: {classes.tolist()}")
    print("[OK] Server siap menerima koneksi WebSocket.")

    yield

    print("[INFO] Server dimatikan.")


# ── Inisialisasi FastAPI ─────────────────────────────────────────────────────────
app = FastAPI(
    title="ASL Real-Time Detector API v2",
    description=(
        "Backend inferensi MLP untuk mendeteksi huruf ASL dari 78 fitur tangan. "
        "Input: 63 koordinat ternormalisasi + 15 sudut sendi."
    ),
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/assets", StaticFiles(directory="assets"), name="assets")


# ── Frontend Route ─────────────────────────────────────────────────────────────
@app.get("/", tags=["UI"])
async def serve_index():
    return FileResponse("index.html")

@app.get("/index.html", tags=["UI"])
async def serve_index_explicit():
    return FileResponse("index.html")


# ── Health Check ─────────────────────────────────────────────────────────────────
@app.get("/health", tags=["Health"])
async def health_check():
    model_loaded = app_state["model"] is not None
    return {
        "status":      "running",
        "version":     "2.0.0",
        "input_dim":   INPUT_DIM,
        "model_loaded": model_loaded,
        "classes":     app_state["classes"].tolist() if model_loaded else [],
    }



# ── Spell Check Endpoint ─────────────────────────────────────────────────────────
@app.get("/spellcheck", tags=["NLP"])
async def spellcheck(word: str):
    """
    Koreksi kata dari Word Builder (contoh: HELLOQ -> HELLO).
    Menggunakan jarak Levenshtein dari dictionary standar.
    """
    if not word or len(word) == 0:
        return {"original": word, "corrected": word}
        
    word = word.lower()
    
    corrected = spell.correction(word)
    
    if corrected:
        return {"original": word, "corrected": corrected.upper()}
    else:
        return {"original": word, "corrected": word.upper()}


# ── WebSocket Endpoint ───────────────────────────────────────────────────────────
@app.websocket("/ws/predict")
async def websocket_predict(websocket: WebSocket):
    """
    Menerima array JSON [78 float], melakukan inferensi MLP dengan temporal
    smoothing, dan mengembalikan prediksi huruf ASL.

    Format masuk  : JSON array panjang 78
    Format keluar : {"prediction": "A", "confidence": 0.92}
                    {"prediction": null, "confidence": 0.45, "message": "..."}
                    {"error": "..."}
    """
    await websocket.accept()
    print(f"[WS] Klien terhubung: {websocket.client}")

    model   = app_state["model"]
    classes = app_state["classes"]

    pred_buffer: deque = deque(maxlen=SMOOTH_BUFFER_SIZE)

    try:
        while True:
            raw = await websocket.receive_text()

            try:
                data = json.loads(raw)
            except json.JSONDecodeError:
                await websocket.send_text(json.dumps({
                    "error": f"Format tidak valid. Kirim JSON array berisi {INPUT_DIM} angka."
                }))
                continue

            client_timestamp = None
            if isinstance(data, dict) and "features" in data and "timestamp" in data:
                client_timestamp = data["timestamp"]
                features_list = data["features"]
            else:
                features_list = data

            if not isinstance(features_list, list) or len(features_list) != INPUT_DIM:
                received = len(features_list) if isinstance(features_list, list) else "bukan list"
                await websocket.send_text(json.dumps({
                    "error": f"Array harus {INPUT_DIM} elemen, diterima: {received}"
                }))
                continue

            try:
                features = np.array(features_list, dtype=np.float32).reshape(1, INPUT_DIM)
            except (ValueError, TypeError) as e:
                await websocket.send_text(json.dumps({"error": f"Konversi data gagal: {e}"}))
                continue

            # ── Inferensi cepat via model() langsung (bukan model.predict()) ──────
            raw_probs = model(features, training=False).numpy()[0]

            # ── Temporal Smoothing ──────────────────────────────────────────────
            pred_buffer.append(raw_probs)
            avg_probs    = np.mean(list(pred_buffer), axis=0)

            confidence   = float(np.max(avg_probs))
            class_index  = int(np.argmax(avg_probs))
            predicted_cls = str(classes[class_index])

            top2_idx = np.argsort(avg_probs)[-2:][::-1]

            stats = app_state["session_stats"]
            stats["total_frames"] += 1

            if confidence >= CONFIDENCE_THRESHOLD:
                stats["total_predicted"] += 1
                stats["letter_counts"][predicted_cls] = stats["letter_counts"].get(predicted_cls, 0) + 1
                stats["conf_sum"] += confidence
                stats["avg_confidence"] = stats["conf_sum"] / stats["total_predicted"]
                
                response = {
                    "prediction": predicted_cls,
                    "confidence": round(confidence, 4),
                    "runner_up":  str(classes[top2_idx[1]]),
                }
            else:
                response = {
                    "prediction": None,
                    "confidence": round(confidence, 4),
                    "message":    f"Confidence rendah ({confidence:.1%})",
                }

            if client_timestamp is not None:
                response["timestamp"] = client_timestamp

            await websocket.send_text(json.dumps(response))

    except WebSocketDisconnect:
        print(f"[WS] Klien terputus: {websocket.client}")
    except Exception as e:
        print(f"[ERROR] {e}")
        try:
            await websocket.send_text(json.dumps({"error": str(e)}))
        except Exception:
            pass
