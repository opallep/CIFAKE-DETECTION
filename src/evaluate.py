"""
evaluate.py - Evaluasi satu run: generalisasi, ketahanan JPEG, keseimbangan kelas (T5, T6, H5, H6).

Kondisi uji: test_seen dan test_unseen, masing-masing tanpa kompresi tambahan
dan dengan kompresi ulang JPEG kualitas 90, 75, 60. Hasil per citra disimpan
ke predictions.csv; metrik (rumus src/metrics.py) ke metrics.json di folder run.

Pemakaian:
    python -m src.evaluate --run experiments/resnet50_se/seed42
    python -m src.evaluate --all
"""

import argparse
import glob
import json
import os

import numpy as np
import pandas as pd
import torch
from torch.amp import autocast
from tqdm import tqdm

from src.common import load_yaml, project_path
from src.dataset import load_manifest, make_loader
from src.metrics import classification_report, compression_drop, generalization_gap, range_gap
from src.models import build_model

JPEG_QUALITIES = [90, 75, 60]
TEST_SPLITS = ["test_seen", "test_unseen"]


def load_run(run_dir: str):
    cfg = load_yaml(os.path.join(run_dir, "config.yaml"))
    ckpt = torch.load(os.path.join(run_dir, "best.pt"), map_location="cpu", weights_only=False)
    info = ckpt["run_info"]
    model = build_model(cfg, info["arch"], info["use_se"], pretrained=False)
    model.load_state_dict(ckpt["state_dict"])
    return cfg, model.eval(), info


@torch.no_grad()
def predict(model, loader, device) -> np.ndarray:
    probs = np.zeros(len(loader.dataset), dtype=np.float64)
    for images, _, idx in tqdm(loader, leave=False, ncols=90):
        with autocast(device_type=device.type, enabled=device.type == "cuda"):
            logits = model(images.to(device, non_blocking=True))
        probs[idx.numpy()] = torch.softmax(logits.float(), dim=1)[:, 1].cpu().numpy()
    return probs


def group_reports(df: pd.DataFrame, key: str) -> dict:
    return {k: classification_report(g["y_true"], g["y_pred"]) for k, g in df.groupby(key)}


def summarize(pred: pd.DataFrame) -> dict:
    out = {"conditions": {}}
    for (cond, split), g in pred.groupby(["condition", "split"]):
        out["conditions"].setdefault(cond, {})[split] = {
            "overall": classification_report(g["y_true"], g["y_pred"]),
            "per_generator": group_reports(g, "generator"),
            "per_image_type": group_reports(g, "image_type"),
        }
    for cond, g in pred.groupby("condition"):
        out["conditions"][cond]["all_test"] = {
            "overall": classification_report(g["y_true"], g["y_pred"]),
            "per_image_type": group_reports(g, "image_type"),
        }

    none = out["conditions"]["none"]
    acc_seen = none["test_seen"]["overall"]["accuracy"]
    acc_unseen = none["test_unseen"]["overall"]["accuracy"]
    all_none = none["all_test"]
    out["summary"] = {
        # T5
        "acc_seen": acc_seen, "acc_unseen": acc_unseen,
        "generalization_gap": generalization_gap(acc_seen, acc_unseen),
        # H5 (semua foto asli pada data uji)
        "fpr": all_none["overall"]["fpr"], "recall_gap": all_none["overall"]["recall_gap"],
        # H6
        "image_type_gap": range_gap([r["accuracy"] for r in all_none["per_image_type"].values()]),
        # T6
        "jpeg_drop": {
            q: compression_drop(all_none["overall"]["accuracy"],
                                out["conditions"][f"jpeg{q}"]["all_test"]["overall"]["accuracy"])
            for q in JPEG_QUALITIES
        },
    }
    return out


def evaluate_run(run_dir: str, threshold: float = 0.5) -> dict:
    run_dir = project_path(run_dir)
    cfg, model, info = load_run(run_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    manifest = load_manifest(cfg["manifest"])
    print(f"[Eval] {info['config_name']} seed={info['seed']}")

    frames = []
    for split in TEST_SPLITS:
        df = manifest[manifest["split"] == split].reset_index(drop=True)
        if df.empty:
            raise SystemExit(f"Split {split} kosong di manifest")
        for q in [None] + JPEG_QUALITIES:
            probs = predict(model, make_loader(df, cfg, training=False, jpeg_quality=q), device)
            frames.append(pd.DataFrame({
                "path": df["path"], "generator": df["generator"], "image_type": df["image_type"],
                "split": split, "condition": "none" if q is None else f"jpeg{q}",
                "y_true": (df["label"] == "ai").astype(int), "prob_ai": probs,
                "y_pred": (probs >= threshold).astype(int),
            }))
    pred = pd.concat(frames, ignore_index=True)
    pred.to_csv(os.path.join(run_dir, "predictions.csv"), index=False)

    metrics = {"run_info": info, "threshold": threshold, **summarize(pred)}
    with open(os.path.join(run_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2, default=float)
    s = metrics["summary"]
    print(f"  seen {s['acc_seen']:.2%} | unseen {s['acc_unseen']:.2%} | gap {s['generalization_gap'] * 100:.2f} pp"
          f" | FPR {s['fpr']:.2%} | JPEG drop " +
          ", ".join(f"Q{q}: {d * 100:.2f} pp" for q, d in s["jpeg_drop"].items()))
    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("--run", help="folder run, mis. experiments/resnet50_se/seed42")
    parser.add_argument("--all", action="store_true", help="semua run di experiments/")
    parser.add_argument("--experiments-dir", default="experiments")
    args = parser.parse_args()
    if args.all:
        runs = sorted(os.path.dirname(p) for p in glob.glob(
            os.path.join(project_path(args.experiments_dir), "*", "seed*", "run_info.json")))
    elif args.run:
        runs = [args.run]
    else:
        raise SystemExit("Gunakan --run <folder> atau --all")
    for run in runs:
        evaluate_run(run)


if __name__ == "__main__":
    main()
