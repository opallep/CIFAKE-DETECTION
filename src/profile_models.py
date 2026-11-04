"""
profile_models.py - Pelaporan efisiensi keenam konfigurasi pada kondisi sama (E1, E4).

Mengukur: jumlah parameter, FLOPs (1 citra 224x224), ukuran bobot (state_dict),
waktu inferensi CPU (rerata dan p95, batch 1, tanpa GPU sesuai E2/B3),
memori proses (RSS) setelah inferensi, dan waktu pelatihan dari run_info.json.

Pemakaian:
    python -m src.profile_models [--repeats 50] [--threads 4]
"""

import argparse
import glob
import io
import json
import os
import platform
import time

import pandas as pd
import torch
from torch.utils.flop_counter import FlopCounterMode

from src.common import load_yaml, project_path, report_path
from src.metrics import mean_time, percentile_95
from src.models import build_model, config_name, count_parameters
from tests.verification._record import record


def rss_mb() -> float:
    try:
        import psutil
        return psutil.Process().memory_info().rss / 2**20
    except ImportError:
        return float("nan")


def profile(cfg, arch, use_se, repeats):
    model = build_model(cfg, arch, use_se, pretrained=False).eval()
    size = cfg["input_size"]
    x = torch.randn(1, 3, size, size)
    with torch.no_grad(), FlopCounterMode(display=False) as fc:
        model(x)
    buf = io.BytesIO()
    torch.save(model.state_dict(), buf)

    times = []
    with torch.no_grad():
        for _ in range(5):
            model(x)
        for _ in range(repeats):
            t0 = time.perf_counter()
            model(x)
            times.append(time.perf_counter() - t0)

    name = config_name(arch, use_se)
    train_hours = []
    for p in glob.glob(os.path.join(project_path(cfg["experiments_dir"]), name, "seed*", "run_info.json")):
        with open(p) as f:
            train_hours.append(json.load(f)["train_seconds"] / 3600)
    return {
        "config": name, "params_m": count_parameters(model)["total"] / 1e6,
        "gmacs": fc.get_total_flops() / 2 / 1e9,   # FlopCounterMode menghitung 2 FLOP per multiply-add
        "weights_mb": buf.tell() / 2**20,
        "cpu_mean_ms": mean_time(times) * 1000, "cpu_p95_ms": percentile_95(times) * 1000,
        "rss_mb": rss_mb(),
        "train_hours_mean": sum(train_hours) / len(train_hours) if train_hours else float("nan"),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("--config", default="configs/train.yaml")
    parser.add_argument("--repeats", type=int, default=50)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    cfg = load_yaml(args.config)
    torch.set_num_threads(args.threads)

    rows = [profile(cfg, a, se, args.repeats) for a in cfg["architectures"] for se in cfg["se_options"]]
    df = pd.DataFrame(rows)
    print(df.to_string(index=False, float_format="%.2f"))
    out = report_path("docs/efficiency_report.md")
    with open(out, "w", encoding="utf-8") as f:
        f.write("# Laporan Efisiensi (E1, E4)\n\nDibuat oleh `python -m src.profile_models`. "
                f"CPU: {platform.processor()}, {args.threads} thread, batch 1, {args.repeats} pengulangan. "
                "GMACs = FLOPs / 2 (multiply-add dihitung satu).\n\n")
        f.write(df.to_markdown(index=False, floatfmt=".2f") + "\n")
    print(f"[Save] {out}")

    complete = df["train_hours_mean"].notna().all()
    record("E4", "Pelaporan efisiensi", "parameter, FLOPs, bobot, waktu inferensi, memori, waktu latih 6 konfigurasi",
           f"{len(df)} konfigurasi diprofil; waktu latih {'lengkap' if complete else 'belum lengkap'}",
           len(df) == 6 and complete)


if __name__ == "__main__":
    main()
