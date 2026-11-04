"""
_record.py - Simpan hasil verifikasi per kode spesifikasi ke results/verification/<KODE>.json.
build_report.py menggabungkannya menjadi docs/verification_report.md.
"""

import json
from datetime import datetime

from src.common import git_commit, report_path


def record(code: str, spec: str, target: str, measured: str, passed: bool, details=None):
    entry = {
        "code": code, "spec": spec, "target": target, "measured": measured,
        "passed": bool(passed), "details": details or {},
        "timestamp": datetime.now().isoformat(timespec="seconds"), "commit": git_commit(),
    }
    with open(report_path(f"results/verification/{code}.json"), "w", encoding="utf-8") as f:
        json.dump(entry, f, indent=2, ensure_ascii=False)
    status = "LULUS" if passed else "TIDAK LULUS"
    print(f"[{code}] {spec}: {measured} (target {target}) -> {status}")
    return entry
