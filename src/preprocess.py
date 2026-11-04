"""
preprocess.py - SATU modul praproses untuk pelatihan, evaluasi, dan server (T3, T7).

Alur:
    berkas asli --load_rgb--> harmonize (resize sisi pendek, center crop,
    encode ulang JPEG) --> [augmentasi, hanya training] --> crop 224 --> tensor

harmonize() dipakai offline oleh src/data/harmonize.py untuk kedua kelas,
dan dipakai server sebelum inferensi, sehingga citra yang dilihat model
saat inferensi melalui jalur yang sama persis dengan data uji.
Encode ulang tanpa metadata juga membuang EXIF (H1).
"""

import io
import random

from PIL import Image, ImageEnhance, ImageFilter, ImageOps
from torchvision import transforms

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

HARMONIZE_DEFAULT = {"short_side": 256, "crop": 256, "format": "JPEG", "quality": 95}


def load_rgb(fp) -> Image.Image:
    """Buka citra (path atau file-like), terapkan orientasi EXIF, ubah ke RGB."""
    with Image.open(fp) as img:
        img = ImageOps.exif_transpose(img)
        return img.convert("RGB")


def harmonize_bytes(img: Image.Image, short_side: int = 256, crop: int = 256,
                    format: str = "JPEG", quality: int = 95) -> bytes:
    """Seragamkan resolusi dan kompresi; kembalikan berkas hasil encode (tanpa EXIF)."""
    w, h = img.size
    scale = short_side / min(w, h)
    if scale != 1:
        img = img.resize((max(crop, round(w * scale)), max(crop, round(h * scale))),
                         Image.Resampling.BICUBIC)
    w, h = img.size
    left, top = (w - crop) // 2, (h - crop) // 2
    img = img.crop((left, top, left + crop, top + crop))

    buf = io.BytesIO()
    if format.upper() == "JPEG":
        img.save(buf, format="JPEG", quality=quality, subsampling=0)
    else:
        img.save(buf, format=format)
    return buf.getvalue()


def harmonize(img: Image.Image, **cfg) -> Image.Image:
    """harmonize_bytes lalu decode kembali, identik dengan membaca berkas hasil harmonize offline."""
    params = {**HARMONIZE_DEFAULT, **cfg}
    return Image.open(io.BytesIO(harmonize_bytes(img, **params))).convert("RGB")


def jpeg_recompress(img: Image.Image, quality: int) -> Image.Image:
    """Kompresi ulang JPEG in-memory (T6 dan augmentasi)."""
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


# ============================================================
# AUGMENTASI (diterapkan sama pada kedua kelas)
# ============================================================

class RandomJPEG:
    def __init__(self, p: float, quality_range):
        self.p, self.qmin, self.qmax = p, quality_range[0], quality_range[1]

    def __call__(self, img):
        if random.random() < self.p:
            return jpeg_recompress(img, random.randint(self.qmin, self.qmax))
        return img


class RandomBlur:
    def __init__(self, p: float, radius=(0.1, 1.0)):
        self.p, self.radius = p, radius

    def __call__(self, img):
        if random.random() < self.p:
            return img.filter(ImageFilter.GaussianBlur(random.uniform(*self.radius)))
        return img


class RandomFilter:
    """Filter ringan ala aplikasi foto: kecerahan, kontras, saturasi."""

    def __init__(self, p: float, strength: float):
        self.p, self.s = p, strength

    def __call__(self, img):
        if random.random() < self.p:
            for enhancer in (ImageEnhance.Brightness, ImageEnhance.Contrast, ImageEnhance.Color):
                img = enhancer(img).enhance(random.uniform(1 - self.s, 1 + self.s))
        return img


def _to_tensor():
    return [transforms.ToTensor(), transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)]


def train_transform(input_size: int, aug: dict):
    return transforms.Compose([
        transforms.RandomCrop(input_size),
        transforms.RandomHorizontalFlip(aug.get("hflip_p", 0.5)),
        RandomFilter(aug.get("color_p", 0.0), aug.get("color_strength", 0.15)),
        RandomBlur(aug.get("blur_p", 0.0)),
        RandomJPEG(aug.get("jpeg_p", 0.0), aug.get("jpeg_quality", [60, 100])),
        *_to_tensor(),
    ])


def eval_transform(input_size: int):
    return transforms.Compose([transforms.CenterCrop(input_size), *_to_tensor()])


def preprocess_inference(fp, input_size: int = 224, harmonize_cfg: dict | None = None):
    """Jalur server (T7): berkas unggahan -> tensor (1, 3, H, W)."""
    img = harmonize(load_rgb(fp), **(harmonize_cfg or {}))
    return eval_transform(input_size)(img).unsqueeze(0)
