"""
verify_t3_metadata_bias.py - T3: keseragaman berkas antarkelas.

Melatih pengklasifikasi sederhana yang HANYA memakai metadata berkas
(format, lebar, tinggi, kualitas JPEG) pada split train, lalu mengukur
akurasi pada test_seen + test_unseen. Target akurasi <= 55%.
Ukuran berkas dilaporkan terpisah dan tidak menentukan kelulusan, karena
setelah format/resolusi/kompresi seragam ukuran berkas ditentukan isi citra
(lihat docs/decisions.md D-05). Uji yang sama pada metadata berkas asli
(kolom orig_*) menunjukkan besarnya bias sebelum harmonisasi.

Pemakaian:
    python -m tests.verification.verify_t3_metadata_bias --config configs/data.yaml
"""

import argparse
import os

import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.common import LABELS, load_yaml, project_path
from src.metrics import confusion_counts, accuracy
from tests.verification._record import record

TARGET = 0.55


def features(df: pd.DataFrame, prefix: str = "", with_size: bool = False) -> pd.DataFrame:
    x = pd.DataFrame({
        "width": df[f"{prefix}width"].astype(float),
        "height": df[f"{prefix}height"].astype(float),
        "quality": df[f"{prefix}quality"].fillna(-1).astype(float),
    })
    if with_size:
        x["file_size"] = df[f"{prefix}file_size"].astype(float)
    fmt = pd.get_dummies(df[f"{prefix}format"].fillna("NA"), prefix="fmt").astype(float)
    return pd.concat([x, fmt], axis=1)


def evaluate(df: pd.DataFrame, prefix: str, with_size: bool = False) -> dict:
    train = df[df["split"] == "train"]
    test = df[df["split"].isin(["test_seen", "test_unseen"])]
    x_all = features(df, prefix, with_size)
    y_train, y_test = train["label"].map(LABELS), test["label"].map(LABELS)
    out = {}
    for name, clf in {
        "logistic_regression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)),
        "gradient_boosting": GradientBoostingClassifier(random_state=42),
    }.items():
        clf.fit(x_all.loc[train.index], y_train)
        pred = clf.predict(x_all.loc[test.index])
        out[name] = accuracy(confusion_counts(y_test.to_numpy(), pred))
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("--config", default="configs/data.yaml")
    args = parser.parse_args()
    cfg = load_yaml(args.config)
    df = pd.read_csv(os.path.join(project_path(cfg["manifest_dir"]), "manifest.csv"))

    harmonized = evaluate(df, "")
    original = evaluate(df, "orig_")
    with_size = evaluate(df, "", with_size=True)
    best = max(harmonized.values())
    record("T3", "Keseragaman berkas antarkelas",
           f"akurasi pengklasifikasi metadata <= {TARGET:.0%}",
           f"akurasi tertinggi {best:.2%} (berkas asli sebelum harmonisasi: {max(original.values()):.2%}; "
           f"info: + ukuran berkas {max(with_size.values()):.2%})",
           best <= TARGET,
           {"harmonized": harmonized, "original": original, "harmonized_with_file_size": with_size})


if __name__ == "__main__":
    main()
