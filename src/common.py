"""
common.py - Utilitas bersama pipeline CD-2 (path, YAML, seed, label).

Konvensi label (brief Bagian 5): kelas positif "ai" = 1, kelas negatif "real" = 0.
"""

import hashlib
import os
import random
import subprocess

import numpy as np
import torch
import yaml

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

LABELS = {"real": 0, "ai": 1}
LABEL_NAMES = {v: k for k, v in LABELS.items()}
IMAGE_TYPES = ["manusia", "hewan", "objek", "pemandangan", "manusia_edit_ai"]
SPLITS = ["train", "val", "test_seen", "test_unseen"]

DISCLAIMER = (
    "Hasil ini bersifat probabilistik sebagai informasi pendukung dan "
    "bukan alat bukti hukum."
)


def project_path(path: str) -> str:
    """Path relatif diartikan relatif terhadap root proyek."""
    return path if os.path.isabs(path) else os.path.join(PROJECT_ROOT, path)


def report_path(path: str) -> str:
    """Lokasi keluaran laporan (docs/*.md, results/verification). Env CD2_REPORT_DIR
    mengalihkannya, dipakai uji smoke agar laporan proyek tidak tertimpa."""
    root = os.environ.get("CD2_REPORT_DIR")
    out = os.path.join(root, path) if root else project_path(path)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    return out


def load_yaml(path: str) -> dict:
    with open(project_path(path), "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def file_sha256(path: str, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()


def git_commit() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT,
            capture_output=True, text=True, check=True,
        )
        dirty = subprocess.run(
            ["git", "status", "--porcelain"], cwd=PROJECT_ROOT,
            capture_output=True, text=True,
        ).stdout.strip()
        return out.stdout.strip() + ("-dirty" if dirty else "")
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
