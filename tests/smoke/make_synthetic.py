"""
make_synthetic.py - Dataset SINTETIS kecil untuk uji end-to-end pipeline CD-2.

Hanya untuk memastikan kode berjalan (manifest -> split -> harmonize ->
verifikasi -> train -> evaluate -> rekap). Angka yang dihasilkan TIDAK
BOLEH dipakai sebagai hasil eksperimen (brief Bagian 11 no. 2).

Sengaja memuat: 10 generator (3 unseen), citra real JPEG vs ai PNG beresolusi
berbeda (bias format yang harus hilang setelah harmonisasi, T3), satu
duplikat lintas sumber (T2), dan satu berkas rusak.

Pemakaian:
    python -m tests.smoke.make_synthetic --out <folder_kosong> [--per-class 40]
"""

import argparse
import os

import numpy as np
import yaml
from PIL import Image

SEEN = ["g01", "g02", "g03", "g04", "g05", "g06", "g07"]
UNSEEN = ["g08", "g09", "g10"]
TYPES = ["manusia", "hewan", "objek", "pemandangan"]


def make_image(rng, label, size):
    """Latar acak + beberapa persegi berwarna acak, sehingga tiap citra unik (dHash berbeda)."""
    h, w = size
    coarse = rng.uniform(0, 255, (8, 8, 3))
    base = np.asarray(Image.fromarray(coarse.astype(np.uint8)).resize((w, h), Image.Resampling.BICUBIC),
                      dtype=np.float64)
    for _ in range(4):
        y0, x0 = rng.integers(0, h - 64), rng.integers(0, w - 64)
        base[y0:y0 + rng.integers(32, 160), x0:x0 + rng.integers(32, 160)] = rng.uniform(0, 255, 3)
    base += rng.normal(0, 8, base.shape)
    if label == "ai":   # pola periodik halus sebagai "artefak generator"
        yy, xx = np.mgrid[0:h, 0:w]
        base += 10 * np.sign(np.sin(xx * 1.7) * np.sin(yy * 1.7))[..., None]
    return Image.fromarray(np.clip(base, 0, 255).astype(np.uint8))


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("--out", required=True)
    parser.add_argument("--per-class", type=int, default=40)
    args = parser.parse_args()
    rng = np.random.default_rng(0)
    raw = os.path.join(args.out, "raw")

    sources = []
    for gi, gen in enumerate(SEEN + UNSEEN):
        image_type = TYPES[gi % len(TYPES)]
        for label in ("ai", "real"):
            folder = os.path.join(raw, f"{gen}_{label}")
            os.makedirs(folder, exist_ok=True)
            for i in range(args.per_class):
                if label == "ai":
                    make_image(rng, label, (512, 512)).save(os.path.join(folder, f"{i:04d}.png"))
                else:
                    make_image(rng, label, (480, 640)).save(os.path.join(folder, f"{i:04d}.jpg"), quality=85)
            sources.append({"name": f"{gen}_{label}", "root": f"{gen}_{label}", "include": ["*"],
                            "label": label, "generator": gen, "image_type": image_type,
                            "source": "synthetic", "license": "synthetic (uji kode)"})

    # duplikat lintas sumber + berkas rusak
    dup = os.path.join(raw, "g02_real", "dup_of_g01.jpg")
    with open(os.path.join(raw, "g01_real", "0000.jpg"), "rb") as f, open(dup, "wb") as g:
        g.write(f.read())
    with open(os.path.join(raw, "g03_ai", "corrupt.png"), "wb") as f:
        f.write(b"not an image")

    data_cfg = {
        "data_root": raw, "manifest_dir": os.path.join(args.out, "manifests"),
        "harmonized_dir": os.path.join(args.out, "harmonized"), "seed": 42,
        "max_images_per_source": None, "min_test_images_per_class": 4,
        "generators": {"seen": SEEN, "unseen": UNSEEN},
        "split_ratio": {"train": 0.7, "val": 0.15, "test_seen": 0.15},
        "harmonize": {"short_side": 256, "crop": 256, "format": "JPEG", "quality": 95},
        "dedup": {"phash_max_distance": 2}, "sources": sources,
    }
    with open(os.path.join(args.out, "data.yaml"), "w") as f:
        yaml.safe_dump(data_cfg, f, sort_keys=False)

    here = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(here, "..", "..", "configs", "train.yaml"), encoding="utf-8") as f:
        train_cfg = yaml.safe_load(f)
    train_cfg.update({"manifest": os.path.join(args.out, "manifests", "manifest.csv"),
                      "experiments_dir": os.path.join(args.out, "experiments"), "seeds": [42]})
    train_cfg["train"].update({"epochs": 2, "batch_size": 16, "num_workers": 0})
    with open(os.path.join(args.out, "train.yaml"), "w") as f:
        yaml.safe_dump(train_cfg, f, sort_keys=False)
    print(f"[Smoke] dataset sintetis + config -> {args.out}")


if __name__ == "__main__":
    main()
