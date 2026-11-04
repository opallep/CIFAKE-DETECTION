"""
build_report.py - Susun docs/verification_report.md dari results/verification/*.json.

Pemakaian:
    python -m tests.verification.build_report
"""

import glob
import json

from src.common import report_path

ALL_CODES = ["T1", "T2", "T3", "T4", "T5", "T6", "T7", "T8", "T9", "T10",
             "H1", "H2", "H3", "H4", "H5", "H6", "H7",
             "E1", "E2", "E3", "E4", "K1", "K2", "K3", "B1", "B2", "B3", "B4"]


def main():
    results = {}
    for path in glob.glob(report_path("results/verification/*.json")):
        with open(path, encoding="utf-8") as f:
            entry = json.load(f)
        results[entry["code"]] = entry

    lines = [
        "# Laporan Verifikasi Spesifikasi CD-2",
        "",
        "Dibuat otomatis oleh `python -m tests.verification.build_report` dari hasil eksekusi nyata",
        "di `results/verification/`. Kode tanpa hasil berarti belum diuji.",
        "",
        "| Kode | Spesifikasi | Hasil terukur | Target | Status | Waktu | Commit |",
        "|------|-------------|---------------|--------|--------|-------|--------|",
    ]
    for code in ALL_CODES:
        e = results.get(code)
        if e is None:
            lines.append(f"| {code} | | | | belum diuji | | |")
        else:
            status = "lulus" if e["passed"] else "**tidak lulus**"
            lines.append(f"| {code} | {e['spec']} | {e['measured']} | {e['target']} | {status} "
                         f"| {e['timestamp']} | `{e['commit'][:12]}` |")
    out = report_path("docs/verification_report.md")
    with open(out, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"[Save] {out}")


if __name__ == "__main__":
    main()
