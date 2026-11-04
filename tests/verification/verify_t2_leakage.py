"""
verify_t2_leakage.py - T2: pembagian data tanpa kebocoran.

Target: citra tumpang tindih antar split = 0 (sha256, dHash identik, dan dHash
dengan jarak <= phash_max_distance); generator unseen tidak muncul di
train/val/test_seen; generator seen tidak muncul di test_unseen.
--rehash menghitung ulang sha256 berkas untuk memastikan manifest sesuai isi disk.

Pemakaian:
    python -m tests.verification.verify_t2_leakage --config configs/data.yaml [--rehash]
"""

import argparse
import os

import pandas as pd
from tqdm import tqdm

from src.common import file_sha256, load_yaml, project_path
from src.data.split_by_generator import duplicate_groups
from tests.verification._record import record


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("--config", default="configs/data.yaml")
    parser.add_argument("--rehash", action="store_true")
    args = parser.parse_args()
    cfg = load_yaml(args.config)
    df = pd.read_csv(os.path.join(project_path(cfg["manifest_dir"]), "manifest.csv"))

    mismatched = 0
    if args.rehash:
        for row in tqdm(df.itertuples(), total=len(df), desc="Rehash", ncols=90):
            if file_sha256(project_path(row.path)) != row.sha256:
                mismatched += 1

    max_d = cfg.get("dedup", {}).get("phash_max_distance", 0)
    df = df.reset_index(drop=True)
    df["group"] = duplicate_groups(df, max_d)
    splits_per_group = df.groupby("group")["split"].nunique()
    leaking_groups = splits_per_group[splits_per_group > 1].index
    leaking = df[df["group"].isin(leaking_groups)]

    unseen = set(cfg["generators"]["unseen"])
    unseen_in_training = int((df["generator"].isin(unseen) & (df["split"] != "test_unseen")).sum())
    seen_in_unseen = int((~df["generator"].isin(unseen) & (df["split"] == "test_unseen")).sum())

    measured = (f"tumpang tindih {len(leaking)}; unseen di train/val/test_seen {unseen_in_training}; "
                f"seen di test_unseen {seen_in_unseen}")
    if args.rehash:
        measured += f"; sha256 tidak cocok {mismatched}"
    record("T2", "Pembagian data tanpa kebocoran",
           "tumpang tindih antar split = 0; unseen tidak dipakai train/val",
           measured,
           len(leaking) == 0 and unseen_in_training == 0 and seen_in_unseen == 0 and mismatched == 0,
           {"phash_max_distance": max_d, "rehash": args.rehash, "n_images": len(df),
            "leaking_examples": leaking.head(20)[["path", "split"]].to_dict("records")})


if __name__ == "__main__":
    main()
