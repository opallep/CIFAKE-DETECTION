"""
web/app.py - Server Web CekCitra (Deteksi Citra AI-Generated vs Asli)
=====================================================================
Frontend berupa HTML statis (web/index.html + web/assets/). File ini
bisa dibuka langsung di browser (mode simulasi) atau disajikan lewat
server ini.

Tahap 1b: endpoint /api/predict belum terhubung ke model; akan diisi
pada tahap 5 setelah model terbaik dipilih dari hasil perbandingan.

Jalankan:
    python web/app.py
lalu buka http://127.0.0.1:5000
"""

import os

from flask import Flask, jsonify, request, send_from_directory

WEB_DIR = os.path.dirname(os.path.abspath(__file__))
ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png", "webp"}
MAX_UPLOAD_MB = 10

app = Flask(
    __name__,
    static_folder=os.path.join(WEB_DIR, "assets"),
    static_url_path="/assets",
)
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_MB * 1024 * 1024

# Diisi pada tahap 5 (model terpilih + nama model)
MODEL = None
MODEL_NAME = None


def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route("/")
def index():
    return send_from_directory(WEB_DIR, "index.html")


@app.route("/api/status")
def status():
    return jsonify(model_ready=MODEL is not None, model=MODEL_NAME)


@app.route("/api/predict", methods=["POST"])
def predict():
    file = request.files.get("image")
    if file is None or file.filename == "":
        return jsonify(error="Tidak ada gambar yang dikirim."), 400
    if not allowed_file(file.filename):
        return jsonify(error="Format tidak didukung. Gunakan JPG, PNG, atau WEBP."), 400

    # TODO (tahap 5): load model terpilih, preprocessing, inferensi.
    # Format respons yang diharapkan frontend:
    # {
    #   "label": "FAKE" | "REAL",
    #   "confidence": 0.973,
    #   "probabilities": {"FAKE": 0.973, "REAL": 0.027},
    #   "model": "LightweightCNN",
    #   "inference_ms": 12.4
    # }
    return jsonify(error="Model belum dihubungkan ke website (tahap 5)."), 503


@app.errorhandler(413)
def too_large(_):
    return jsonify(error=f"Ukuran file melebihi {MAX_UPLOAD_MB} MB."), 413


if __name__ == "__main__":
    app.run(debug=True)
