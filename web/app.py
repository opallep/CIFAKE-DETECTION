"""
web/app.py - Server Web CekCitra (Deteksi Citra AI-Generated vs Asli)
=====================================================================
Menyajikan frontend statis (web/index.html + web/assets/) dan API
prediksi menggunakan model terlatih dari folder models/.

Jalankan:
    python web/app.py            → http://127.0.0.1:5000 (hanya laptop ini)
    python web/app.py --lan      → bisa dibuka dari HP di jaringan Wi-Fi yang sama

Model yang dipakai bisa diganti lewat variabel lingkungan CEKCITRA_MODEL
(LightweightCNN | ResNet50 | EfficientNetV2B0). Default: ResNet50.
"""

import io
import json
import os
import socket
import sys
import time

WEB_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(WEB_DIR))   # agar config/models/data_loader bisa di-import

import torch
from flask import Flask, jsonify, request, send_from_directory
from PIL import Image, ImageOps, UnidentifiedImageError
from torchvision import transforms

import config
from data_loader import get_transforms
from models import count_parameters, get_model

ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png", "webp"}
MAX_UPLOAD_MB = 10

MODEL_LABELS = {
    config.MODEL_LIGHTWEIGHT: "LightweightCNN, 8 lapisan konvolusi",
    config.MODEL_RESNET50: "ResNet-50, transfer learning ImageNet",
    config.MODEL_EFFICIENTNET: "EfficientNetV2-B0, transfer learning ImageNet",
}

app = Flask(
    __name__,
    static_folder=os.path.join(WEB_DIR, "assets"),
    static_url_path="/assets",
)
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_MB * 1024 * 1024


# ============================================================
# MODEL
# ============================================================

def load_model(model_name: str):
    """Muat bobot terbaik dari models/<nama>_best.pt. Kembalikan None jika belum ada."""
    path = os.path.join(config.MODEL_DIR, f"{model_name}_best.pt")
    if not os.path.exists(path):
        print(f"[WARNING] Bobot model tidak ditemukan: {path}")
        print("[WARNING] Website berjalan dalam mode simulasi.")
        return None
    model = get_model(model_name)
    model.load_state_dict(torch.load(path, map_location=config.DEVICE, weights_only=True))
    model.eval()
    # Pemanasan: inferensi pertama di GPU lambat (inisialisasi CUDA), jadi dilakukan saat start.
    size = config.IMG_SIZE_LIGHTWEIGHT if model_name == config.MODEL_LIGHTWEIGHT else config.IMG_SIZE_TRANSFER
    with torch.no_grad():
        model(torch.zeros(1, 3, size, size, device=config.DEVICE))
    print(f"[Model] {model_name} dimuat dari {path} ({config.DEVICE})")
    return model


def load_model_info(model_name: str, model) -> dict:
    info = {
        "label": MODEL_LABELS.get(model_name, model_name),
        "params": count_parameters(model)["total"],
        "input_size": config.IMG_SIZE_LIGHTWEIGHT if model_name == config.MODEL_LIGHTWEIGHT
        else config.IMG_SIZE_TRANSFER,
        "test_accuracy": None,
        "auc_roc": None,
    }
    metrics_path = os.path.join(config.RESULT_DIR, f"{model_name}_metrics.json")
    if os.path.exists(metrics_path):
        with open(metrics_path) as f:
            metrics = json.load(f)
        info["test_accuracy"] = metrics.get("accuracy")
        info["auc_roc"] = metrics.get("auc_roc")
    return info


MODEL_NAME = os.environ.get("CEKCITRA_MODEL", config.MODEL_RESNET50)
MODEL = load_model(MODEL_NAME)
MODEL_INFO = load_model_info(MODEL_NAME, MODEL) if MODEL is not None else None

# Gambar diperkecil dulu ke 32×32 agar sama dengan citra CIFAKE yang dipelajari model,
# lalu diproses dengan transform evaluasi model (resize ke input model + normalisasi).
TO_CIFAKE = transforms.Resize((config.IMG_SIZE_LIGHTWEIGHT, config.IMG_SIZE_LIGHTWEIGHT))
EVAL_TRANSFORM = get_transforms(MODEL_NAME, is_training=False)


@torch.no_grad()
def predict_image(img: Image.Image) -> dict:
    start = time.perf_counter()
    x = EVAL_TRANSFORM(TO_CIFAKE(img)).unsqueeze(0).to(config.DEVICE)
    probs = torch.softmax(MODEL(x), dim=1)[0].float().cpu().tolist()
    if config.DEVICE.type == "cuda":
        torch.cuda.synchronize()
    elapsed_ms = (time.perf_counter() - start) * 1000

    p_fake, p_real = probs[0], probs[1]      # indeks kelas: 0 = FAKE, 1 = REAL
    label = "REAL" if p_real >= p_fake else "FAKE"
    return {
        "label": label,
        "confidence": max(p_fake, p_real),
        "probabilities": {"FAKE": p_fake, "REAL": p_real},
        "model": MODEL_NAME,
        "inference_ms": elapsed_ms,
    }


# ============================================================
# ROUTES
# ============================================================

def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route("/")
def index():
    return send_from_directory(WEB_DIR, "index.html")


@app.route("/api/status")
def status():
    return jsonify(model_ready=MODEL is not None, model=MODEL_NAME if MODEL else None, info=MODEL_INFO)


@app.route("/api/predict", methods=["POST"])
def predict():
    if MODEL is None:
        return jsonify(error=f"Bobot model {MODEL_NAME} belum tersedia. Latih model terlebih dahulu."), 503

    file = request.files.get("image")
    if file is None or file.filename == "":
        return jsonify(error="Tidak ada gambar yang dikirim."), 400
    if not allowed_file(file.filename):
        return jsonify(error="Format tidak didukung. Gunakan JPG, PNG, atau WEBP."), 400

    try:
        img = Image.open(io.BytesIO(file.read()))
        img = ImageOps.exif_transpose(img).convert("RGB")
    except (UnidentifiedImageError, OSError):
        return jsonify(error="File tidak bisa dibaca sebagai gambar. Coba file lain."), 400

    return jsonify(predict_image(img))


@app.errorhandler(413)
def too_large(_):
    return jsonify(error=f"Ukuran file melebihi {MAX_UPLOAD_MB} MB."), 413


def lan_ip() -> str:
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80))
        return s.getsockname()[0]
    except OSError:
        return "127.0.0.1"
    finally:
        s.close()


if __name__ == "__main__":
    lan = "--lan" in sys.argv
    host = "0.0.0.0" if lan else "127.0.0.1"
    print(f"\n  CekCitra berjalan di http://127.0.0.1:5000")
    if lan:
        print(f"  Dari HP (Wi-Fi yang sama): http://{lan_ip()}:5000\n")
    # use_reloader=False agar model tidak dimuat dua kali
    app.run(host=host, port=5000, debug=False, use_reloader=False, threaded=True)
