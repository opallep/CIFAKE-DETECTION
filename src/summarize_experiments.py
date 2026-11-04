"""
summarize_experiments.py - Rekap ablation lintas seed + verifikasi T4, T5, T6, H5, H6, E1, B2.

Konfigurasi terpilih ditentukan HANYA dari rerata akurasi validasi (generator
seen), tidak pernah dari hasil test_unseen (T2). Target dinilai pada rerata
lintas seed konfigurasi terpilih.

Pemakaian:
    python -m src.summarize_experiments
"""

import argparse
import glob
import json
import os

import numpy as np
import pandas as pd

from src.common import load_yaml, project_path, report_path
from tests.verification._record import record

TARGETS = {"acc_unseen": 0.90, "gap": 0.05, "jpeg_drop": 0.10, "fpr": 0.10,
           "recall_gap": 0.10, "image_type_gap": 0.10, "params": 30e6, "gpu_hours": 150}


def load_runs(experiments_dir: str) -> list:
    runs = []
    for info_path in glob.glob(os.path.join(project_path(experiments_dir), "*", "seed*", "run_info.json")):
        run = os.path.dirname(info_path)
        with open(info_path) as f:
            info = json.load(f)
        metrics_path = os.path.join(run, "metrics.json")
        metrics = None
        if os.path.exists(metrics_path):
            with open(metrics_path) as f:
                metrics = json.load(f)
        runs.append({"dir": run, "info": info, "metrics": metrics})
    return runs


def run_row(r) -> dict:
    info, s = r["info"], r["metrics"]["summary"]
    return {
        "config": info["config_name"], "seed": info["seed"], "val_acc": info["best_val_acc"],
        "acc_seen": s["acc_seen"], "acc_unseen": s["acc_unseen"], "gap": s["generalization_gap"],
        "fpr": s["fpr"], "recall_gap": s["recall_gap"], "image_type_gap": s["image_type_gap"],
        **{f"jpeg_drop_q{q}": d for q, d in s["jpeg_drop"].items()},
        "params": info["params"]["total"], "gpu_hours": info["gpu_hours"],
    }


def per_group_mean(runs, key) -> pd.DataFrame:
    rows = []
    for r in runs:
        for split in ("test_seen", "test_unseen", "all_test"):
            block = r["metrics"]["conditions"]["none"][split].get(key, {})
            for name, rep in block.items():
                rows.append({"split": split, key: name, **{m: rep[m] for m in
                             ("n", "accuracy", "precision", "recall", "f1", "fpr")}})
    return pd.DataFrame(rows).groupby(["split", key]).mean(numeric_only=True).reset_index()


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("--config", default="configs/train.yaml")
    args = parser.parse_args()
    cfg = load_yaml(args.config)

    runs = load_runs(cfg["experiments_dir"])
    if not runs:
        raise SystemExit("Belum ada run selesai di experiments/")

    # ---- T4: 6 konfigurasi x >= 3 pengulangan, data dan prosedur sama ----
    expected = {f"{a}_{'se' if se else 'base'}" for a in cfg["architectures"] for se in cfg["se_options"]}
    per_config = pd.Series([r["info"]["config_name"] for r in runs]).value_counts()
    manifests = {r["info"]["manifest_sha256"] for r in runs}
    min_reps = min(int(per_config.get(c, 0)) for c in expected)
    record("T4", "Rancangan eksperimen terkendali", "6 konfigurasi x >= 3 pengulangan, data dan prosedur sama",
           f"{sum(c in per_config for c in expected)}/6 konfigurasi; pengulangan minimum {min_reps}; "
           f"{len(manifests)} versi manifest",
           len(expected) == 6 and min_reps >= 3 and len(manifests) == 1,
           {"runs_per_config": per_config.to_dict(), "manifest_sha256": sorted(manifests)})

    # ---- B2: total GPU-jam ----
    gpu_hours = sum(r["info"]["gpu_hours"] for r in runs)
    record("B2", "Biaya pelatihan", f"<= {TARGETS['gpu_hours']} GPU-jam", f"{gpu_hours:.2f} GPU-jam ({len(runs)} run)",
           gpu_hours <= TARGETS["gpu_hours"])

    evaluated = [r for r in runs if r["metrics"]]
    if not evaluated:
        raise SystemExit("Belum ada run yang dievaluasi (python -m src.evaluate --all)")
    df = pd.DataFrame([run_row(r) for r in evaluated])
    agg = df.drop(columns="seed").groupby("config").agg(["mean", "std"])
    n_seeds = df.groupby("config").size()

    best = df.groupby("config")["val_acc"].mean().idxmax()
    m = df[df["config"] == best].mean(numeric_only=True)
    best_runs = [r for r in evaluated if r["info"]["config_name"] == best]
    tag = f"[{best}, rerata {len(best_runs)} seed]"

    record("T5", "Kinerja generalisasi lintas generator",
           "akurasi unseen >= 90%; gap <= 5 pp",
           f"{tag} unseen {m['acc_unseen']:.2%}; gap {m['gap'] * 100:.2f} pp",
           m["acc_unseen"] >= TARGETS["acc_unseen"] and m["gap"] <= TARGETS["gap"])
    drops = {q: m[f"jpeg_drop_q{q}"] for q in (90, 75, 60)}
    record("T6", "Ketahanan terhadap kompresi", "penurunan <= 10 pp pada Q90, Q75, Q60",
           f"{tag} " + ", ".join(f"Q{q} {d * 100:.2f} pp" for q, d in drops.items()),
           all(d <= TARGETS["jpeg_drop"] for d in drops.values()))
    record("H5", "Keseimbangan kedua kelas", "FPR <= 10%; selisih recall <= 10 pp",
           f"{tag} FPR {m['fpr']:.2%}; selisih recall {m['recall_gap'] * 100:.2f} pp",
           m["fpr"] <= TARGETS["fpr"] and m["recall_gap"] <= TARGETS["recall_gap"])
    record("H6", "Kewajaran antarjenis citra", "selisih akurasi jenis tertinggi-terendah <= 10 pp",
           f"{tag} {m['image_type_gap'] * 100:.2f} pp", m["image_type_gap"] <= TARGETS["image_type_gap"])
    record("E1", "Ukuran model", "model tunggal; <= 30 juta parameter",
           f"{tag} {m['params'] / 1e6:.2f} juta parameter; terbesar di ablation "
           f"{df['params'].max() / 1e6:.2f} juta", m["params"] <= TARGETS["params"])

    lines = ["# Hasil Ablation (dibuat otomatis)", "",
             "Sumber: `experiments/*/seed*/metrics.json`. Nilai = rerata ± simpangan baku lintas seed.",
             f"Konfigurasi terpilih (rerata akurasi validasi seen tertinggi): **{best}**.", "",
             "| Konfigurasi | n seed | Val (%) | Seen (%) | Unseen (%) | Gap (pp) | FPR (%) | "
             "Drop Q90 | Drop Q75 | Drop Q60 | Param (juta) |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]
    for c in agg.index:
        def ms(col, scale=100):
            mean, std = agg.loc[c, (col, "mean")], agg.loc[c, (col, "std")]
            std = 0 if np.isnan(std) else std
            return f"{mean * scale:.2f} ± {std * scale:.2f}"
        lines.append(f"| {c} | {n_seeds[c]} | {ms('val_acc')} | {ms('acc_seen')} | {ms('acc_unseen')} | "
                     f"{ms('gap')} | {ms('fpr')} | {ms('jpeg_drop_q90')} | {ms('jpeg_drop_q75')} | "
                     f"{ms('jpeg_drop_q60')} | {agg.loc[c, ('params', 'mean')] / 1e6:.2f} |")
    for key, title in (("per_generator", "per generator"), ("per_image_type", "per jenis citra")):
        lines += ["", f"## {best}: metrik {title} (rerata seed, tanpa kompresi)", "",
                  per_group_mean(best_runs, key).to_markdown(index=False, floatfmt=".4f")]
    out = report_path("docs/experiment_results.md")
    with open(out, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"[Save] {out}")


if __name__ == "__main__":
    main()
