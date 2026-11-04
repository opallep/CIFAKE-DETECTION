"""
train.py - Pelatihan satu konfigurasi ablation (T4, K3, B2).

Checkpoint terbaik dipilih dari akurasi split "val" yang hanya berisi
generator seen; generator unseen tidak pernah dibaca di sini (T2).
Setiap run menyimpan snapshot konfigurasi, commit, hash manifest, versi
library, seed, dan waktu pelatihan (GPU-jam, B2) ke experiments/<konfig>/seed<k>/.

Pemakaian:
    python -m src.train --arch resnet50 --se --seed 42
    python -m src.train --arch efficientnetv2_b0 --no-se --seed 43 --epochs 1   # uji cepat
"""

import argparse
import json
import os
import platform
import time

import timm
import torch
import torch.nn as nn
import torchvision
import yaml
from torch.amp import GradScaler, autocast
from tqdm import tqdm

from src.common import file_sha256, git_commit, load_yaml, project_path, set_seed
from src.dataset import load_manifest, make_loader
from src.models import build_model, config_name, count_parameters


def run_epoch(model, loader, device, criterion, optimizer=None, scaler=None, amp=True):
    training = optimizer is not None
    model.train(training)
    total_loss, correct, total = 0.0, 0, 0
    with torch.set_grad_enabled(training):
        for images, labels, _ in tqdm(loader, desc="  train" if training else "  val  ",
                                      leave=False, ncols=90):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            with autocast(device_type=device.type, enabled=amp and device.type == "cuda"):
                logits = model(images)
                loss = criterion(logits, labels)
            if training:
                optimizer.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            total_loss += loss.item() * labels.size(0)
            correct += (logits.argmax(1) == labels).sum().item()
            total += labels.size(0)
    return total_loss / total, correct / total


def run_dir(cfg, arch, use_se, seed) -> str:
    return project_path(os.path.join(cfg["experiments_dir"], config_name(arch, use_se), f"seed{seed}"))


def train(cfg: dict, arch: str, use_se: bool, seed: int) -> str:
    set_seed(seed)
    torch.backends.cudnn.benchmark = False
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = run_dir(cfg, arch, use_se, seed)
    os.makedirs(out_dir, exist_ok=True)
    t = cfg["train"]

    manifest = load_manifest(cfg["manifest"])
    train_df = manifest[manifest["split"] == "train"]
    val_df = manifest[manifest["split"] == "val"]
    train_loader = make_loader(train_df, cfg, training=True)
    val_loader = make_loader(val_df, cfg, training=False)

    model = build_model(cfg, arch, use_se).to(device)
    params = count_parameters(model)
    print(f"[Run] {config_name(arch, use_se)} seed={seed} device={device}")
    print(f"[Data] train {len(train_df):,} | val {len(val_df):,} (generator seen saja)")
    print(f"[Model] {cfg['architectures'][arch]} | parameter {params['total']:,}")

    optimizer = torch.optim.AdamW(model.param_groups(t["lr_backbone"], t["lr_head"]),
                                  weight_decay=t["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=t["epochs"])
    scaler = GradScaler(device.type, enabled=t["amp"] and device.type == "cuda")
    criterion = nn.CrossEntropyLoss()

    run_info = {
        "config_name": config_name(arch, use_se), "arch": arch, "timm_name": cfg["architectures"][arch],
        "use_se": use_se, "seed": seed, "commit": git_commit(),
        "manifest": cfg["manifest"], "manifest_sha256": file_sha256(project_path(cfg["manifest"])),
        "n_train": len(train_df), "n_val": len(val_df), "params": params,
        "versions": {"python": platform.python_version(), "torch": torch.__version__,
                     "torchvision": torchvision.__version__, "timm": timm.__version__},
        "device": torch.cuda.get_device_name(0) if device.type == "cuda" else platform.processor(),
        "started": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    with open(os.path.join(out_dir, "config.yaml"), "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)

    history, best_acc, best_epoch, patience = [], -1.0, 0, 0
    start = time.time()
    for epoch in range(1, t["epochs"] + 1):
        e0 = time.time()
        tr_loss, tr_acc = run_epoch(model, train_loader, device, criterion, optimizer, scaler, t["amp"])
        va_loss, va_acc = run_epoch(model, val_loader, device, criterion, amp=t["amp"])
        scheduler.step()
        history.append({"epoch": epoch, "train_loss": tr_loss, "train_acc": tr_acc,
                        "val_loss": va_loss, "val_acc": va_acc, "seconds": time.time() - e0})
        improved = va_acc > best_acc
        if improved:
            best_acc, best_epoch, patience = va_acc, epoch, 0
            torch.save({"state_dict": model.state_dict(), "run_info": run_info, "epoch": epoch,
                        "val_acc": va_acc}, os.path.join(out_dir, "best.pt"))
        else:
            patience += 1
        print(f"  epoch {epoch:>3} | train {tr_loss:.4f} / {tr_acc:.2%} | "
              f"val {va_loss:.4f} / {va_acc:.2%} | {time.time() - e0:.0f}s{' *' if improved else ''}")
        with open(os.path.join(out_dir, "history.json"), "w") as f:
            json.dump(history, f, indent=2)
        if patience >= t["early_stopping_patience"]:
            print(f"[EarlyStopping] val tidak membaik {patience} epoch")
            break

    elapsed = time.time() - start
    run_info.update({
        "finished": time.strftime("%Y-%m-%dT%H:%M:%S"), "best_epoch": best_epoch,
        "best_val_acc": best_acc, "epochs_run": len(history), "train_seconds": elapsed,
        "gpu_hours": elapsed / 3600 if device.type == "cuda" else 0.0,
    })
    with open(os.path.join(out_dir, "run_info.json"), "w") as f:
        json.dump(run_info, f, indent=2)
    print(f"[Selesai] best val {best_acc:.2%} (epoch {best_epoch}) | {elapsed / 60:.1f} menit -> {out_dir}")
    return out_dir


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("--config", default="configs/train.yaml")
    parser.add_argument("--arch", required=True)
    parser.add_argument("--se", action=argparse.BooleanOptionalAction, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--manifest", default=None)
    parser.add_argument("--epochs", type=int, default=None, help="override (uji cepat)")
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    if args.arch not in cfg["architectures"]:
        raise SystemExit(f"--arch harus salah satu dari {list(cfg['architectures'])}")
    if args.manifest:
        cfg["manifest"] = args.manifest
    if args.epochs:
        cfg["train"]["epochs"] = args.epochs
    train(cfg, args.arch, args.se, args.seed)


if __name__ == "__main__":
    main()
