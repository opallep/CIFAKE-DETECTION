"""
metrics.py - Rumus metrik brief CD-2 Bagian 7, diimplementasikan persis.

Kelas positif = ai (1), negatif = real (0).
Pembagian dengan nol menghasilkan NaN (metrik tidak terdefinisi), bukan 0,
agar tidak tampak seperti hasil buruk/baik yang sebenarnya tidak terukur.
"""

import math

import numpy as np


def _div(a, b):
    return a / b if b else float("nan")


def confusion_counts(y_true, y_pred) -> dict:
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    return {
        "TP": int(((y_true == 1) & (y_pred == 1)).sum()),
        "TN": int(((y_true == 0) & (y_pred == 0)).sum()),
        "FP": int(((y_true == 0) & (y_pred == 1)).sum()),
        "FN": int(((y_true == 1) & (y_pred == 0)).sum()),
    }


def accuracy(c):            # (TP + TN) / (TP + TN + FP + FN)
    return _div(c["TP"] + c["TN"], c["TP"] + c["TN"] + c["FP"] + c["FN"])


def precision(c):           # TP / (TP + FP)
    return _div(c["TP"], c["TP"] + c["FP"])


def recall_ai(c):           # TP / (TP + FN)
    return _div(c["TP"], c["TP"] + c["FN"])


def recall_real(c):         # TN / (TN + FP)
    return _div(c["TN"], c["TN"] + c["FP"])


def f1_score(c):            # 2 * P * R / (P + R)
    p, r = precision(c), recall_ai(c)
    return _div(2 * p * r, p + r) if not (math.isnan(p) or math.isnan(r)) else float("nan")


def false_positive_rate(c):  # FP / (FP + TN)
    return _div(c["FP"], c["FP"] + c["TN"])


def recall_gap(c):          # |Recall_ai - Recall_real|
    return abs(recall_ai(c) - recall_real(c))


def classification_report(y_true, y_pred) -> dict:
    c = confusion_counts(y_true, y_pred)
    return {
        **c, "n": sum(c.values()),
        "accuracy": accuracy(c), "precision": precision(c), "recall": recall_ai(c),
        "f1": f1_score(c), "fpr": false_positive_rate(c),
        "recall_real": recall_real(c), "recall_gap": recall_gap(c),
    }


def generalization_gap(acc_seen, acc_unseen):    # Akurasi_seen - Akurasi_unseen
    return acc_seen - acc_unseen


def compression_drop(acc_initial, acc_q):        # Akurasi_awal - Akurasi_Q
    return acc_initial - acc_q


def range_gap(values):                           # Akurasi_maks - Akurasi_min
    values = [v for v in values if not math.isnan(v)]
    return max(values) - min(values) if values else float("nan")


def max_prob_diff(p_eval, p_server):             # max_i |p_i,eval - p_i,server|
    return float(np.max(np.abs(np.asarray(p_eval) - np.asarray(p_server))))


def mean_time(times):                            # (1/n) sum t_i
    return float(np.mean(times))


def percentile_95(times):                        # t_(ceil(0.95 n)), urutan menaik, indeks 1-based
    t = sorted(times)
    return float(t[math.ceil(0.95 * len(t)) - 1])


def sus_score(responses):
    """Skor SUS satu responden: 2,5 x [sum_ganjil (s_i - 1) + sum_genap (5 - s_i)]."""
    if len(responses) != 10 or any(s not in (1, 2, 3, 4, 5) for s in responses):
        raise ValueError("SUS butuh 10 jawaban bernilai 1..5")
    odd = sum(s - 1 for s in responses[0::2])    # butir 1, 3, 5, 7, 9
    even = sum(5 - s for s in responses[1::2])   # butir 2, 4, 6, 8, 10
    return 2.5 * (odd + even)


def old_generator_drop(acc_pre, acc_post):       # Akurasi_lama,pra - Akurasi_lama,pasca
    return acc_pre - acc_post


def retrain_diff(acc_1, acc_2):                  # |Akurasi_1 - Akurasi_2|
    return abs(acc_1 - acc_2)
