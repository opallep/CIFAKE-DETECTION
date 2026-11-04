"""
dataset.py - Dataset PyTorch berbasis manifest.csv (pengganti ImageFolder CIFAKE).

Mengembalikan (tensor, label, index_baris) agar evaluasi bisa memetakan
prediksi kembali ke generator dan image_type setiap citra.
"""

import pandas as pd
from torch.utils.data import DataLoader, Dataset

from src.common import LABELS, project_path
from src.preprocess import eval_transform, jpeg_recompress, load_rgb, train_transform


def load_manifest(path: str) -> pd.DataFrame:
    return pd.read_csv(project_path(path))


class ManifestDataset(Dataset):
    def __init__(self, df: pd.DataFrame, transform, jpeg_quality: int | None = None):
        self.df = df.reset_index(drop=True)
        self.paths = [project_path(p) for p in self.df["path"]]
        self.labels = self.df["label"].map(LABELS).tolist()
        self.transform = transform
        self.jpeg_quality = jpeg_quality

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        img = load_rgb(self.paths[i])
        if self.jpeg_quality is not None:
            img = jpeg_recompress(img, self.jpeg_quality)
        return self.transform(img), self.labels[i], i


def make_loader(df: pd.DataFrame, cfg: dict, training: bool, jpeg_quality=None) -> DataLoader:
    size = cfg["input_size"]
    transform = train_transform(size, cfg["augment"]) if training else eval_transform(size)
    t = cfg["train"]
    return DataLoader(
        ManifestDataset(df, transform, jpeg_quality),
        batch_size=t["batch_size"], shuffle=training, drop_last=training,
        num_workers=t["num_workers"], pin_memory=True,
        persistent_workers=training and t["num_workers"] > 0,
    )
