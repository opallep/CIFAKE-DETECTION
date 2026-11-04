"""
run_experiments.py - Jalankan seluruh ablation: 3 arsitektur x (tanpa/dengan SE) x seed (T4).

Setiap run dijalankan di proses terpisah (memori GPU bersih). Run yang sudah
punya run_info.json dilewati, sehingga skrip aman dihentikan dan dilanjutkan.

Pemakaian:
    python -m src.run_experiments                  # latih semua yang belum selesai
    python -m src.run_experiments --evaluate       # + evaluasi setiap run yang selesai
    python -m src.run_experiments --dry-run        # tampilkan daftar run saja
"""

import argparse
import os
import subprocess
import sys

from src.common import PROJECT_ROOT, load_yaml
from src.models import config_name
from src.train import run_dir


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("--config", default="configs/train.yaml")
    parser.add_argument("--evaluate", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    cfg = load_yaml(args.config)

    runs = [(a, se, s) for a in cfg["architectures"] for se in cfg["se_options"] for s in cfg["seeds"]]
    print(f"[Ablation] {len(runs)} run ({len(cfg['architectures'])} arsitektur x "
          f"{len(cfg['se_options'])} opsi SE x {len(cfg['seeds'])} seed)")
    for arch, se, seed in runs:
        out = run_dir(cfg, arch, se, seed)
        done = os.path.exists(os.path.join(out, "run_info.json"))
        evaluated = os.path.exists(os.path.join(out, "metrics.json"))
        print(f"  {config_name(arch, se):<26} seed{seed}  "
              f"{'selesai' if done else 'belum'}{' + dievaluasi' if evaluated else ''}")
        if args.dry_run:
            continue
        if not done:
            subprocess.run([sys.executable, "-m", "src.train", "--config", args.config,
                            "--arch", arch, "--se" if se else "--no-se", "--seed", str(seed)],
                           cwd=PROJECT_ROOT, check=True)
        if args.evaluate and not evaluated:
            subprocess.run([sys.executable, "-m", "src.evaluate", "--run", out],
                           cwd=PROJECT_ROOT, check=True)


if __name__ == "__main__":
    main()
