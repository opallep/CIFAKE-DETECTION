"""
verify_t1_h7_inventory.py - T1 (cakupan dataset) dan H7 (pencatatan lisensi sumber data).

T1: >= 10 generator; >= 3 generator unseen; >= 400 citra per kelas pada data uji tiap generator.
H7: 100% sumber data memiliki lisensi tercatat.
Menulis docs/data_inventory.md (jumlah per split/generator/kelas) dan docs/data_sources.md.

Pemakaian:
    python -m tests.verification.verify_t1_h7_inventory --config configs/data.yaml
"""

import argparse
import os

import pandas as pd

from src.common import load_yaml, project_path, report_path
from tests.verification._record import record


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("--config", default="configs/data.yaml")
    args = parser.parse_args()
    cfg = load_yaml(args.config)
    df = pd.read_csv(os.path.join(project_path(cfg["manifest_dir"]), "manifest.csv"))
    min_test = cfg.get("min_test_images_per_class", 400)

    counts = df.groupby(["split", "generator", "label"]).size().unstack(fill_value=0)
    test = df[df["split"].isin(["test_seen", "test_unseen"])]
    test_counts = test.groupby(["generator", "label"]).size().unstack(fill_value=0)
    for col in ("ai", "real"):
        if col not in test_counts:
            test_counts[col] = 0
    test_counts["min_per_class"] = test_counts[["ai", "real"]].min(axis=1)

    generators = sorted(df.loc[df["label"] == "ai", "generator"].unique())
    unseen = sorted(df.loc[df["split"] == "test_unseen", "generator"].unique())
    all_gens = sorted(df["generator"].unique())
    per_gen = test_counts.reindex(all_gens, fill_value=0)
    short = per_gen[per_gen["min_per_class"] < min_test]

    passed_t1 = len(generators) >= 10 and len(unseen) >= 3 and short.empty
    record("T1", "Cakupan dataset multi-generator",
           f">= 10 generator; >= 3 unseen; >= {min_test} citra/kelas uji tiap generator",
           f"{len(generators)} generator; {len(unseen)} unseen; "
           f"{len(all_gens) - len(short)}/{len(all_gens)} generator memenuhi jumlah uji",
           passed_t1,
           {"generators": generators, "unseen": unseen,
            "generators_below_min_test": {k: int(v) for k, v in short["min_per_class"].items()}})

    sources = df.groupby(["source", "license"]).agg(
        n_images=("path", "size"), labels=("label", lambda s: ", ".join(sorted(set(s))))
    ).reset_index()
    with_license = int(sources["license"].astype(str).str.strip().ne("").sum())
    record("H7", "Kepatuhan sumber data", "100% sumber berlisensi dan tercatat",
           f"{with_license}/{len(sources)} sumber tercatat dengan lisensi",
           with_license == len(sources), {"sources": sources.to_dict("records")})

    with open(report_path("docs/data_inventory.md"), "w", encoding="utf-8") as f:
        f.write("# Inventaris Dataset (T1)\n\nDibuat oleh `verify_t1_h7_inventory.py`.\n\n")
        f.write(f"Total citra: {len(df):,}\n\n## Jumlah per split, generator, kelas\n\n")
        f.write(counts.reset_index().to_markdown(index=False) + "\n\n")
        f.write("## Data uji per generator\n\n" + test_counts.reset_index().to_markdown(index=False) + "\n")
    with open(report_path("docs/data_sources.md"), "w", encoding="utf-8") as f:
        f.write("# Sumber Data dan Lisensi (H7, B1)\n\nDibuat oleh `verify_t1_h7_inventory.py`.\n\n")
        f.write(sources.to_markdown(index=False) + "\n")
    print("[Save] docs/data_inventory.md, docs/data_sources.md")


if __name__ == "__main__":
    main()
