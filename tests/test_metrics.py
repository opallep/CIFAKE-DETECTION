"""
test_metrics.py - Uji rumus brief Bagian 7 dengan angka yang dihitung manual.

    python -m unittest tests.test_metrics
"""

import math
import unittest

from PIL import Image

from src import metrics as m
from src.preprocess import harmonize, preprocess_inference


class TestClassificationMetrics(unittest.TestCase):
    def setUp(self):
        # TP=3, TN=4, FP=1, FN=2
        self.y_true = [1, 1, 1, 1, 1, 0, 0, 0, 0, 0]
        self.y_pred = [1, 1, 1, 0, 0, 0, 0, 0, 0, 1]
        self.c = m.confusion_counts(self.y_true, self.y_pred)

    def test_counts(self):
        self.assertEqual(self.c, {"TP": 3, "TN": 4, "FP": 1, "FN": 2})

    def test_formulas(self):
        self.assertAlmostEqual(m.accuracy(self.c), 7 / 10)
        self.assertAlmostEqual(m.precision(self.c), 3 / 4)
        self.assertAlmostEqual(m.recall_ai(self.c), 3 / 5)
        self.assertAlmostEqual(m.recall_real(self.c), 4 / 5)
        self.assertAlmostEqual(m.f1_score(self.c), 2 * 0.75 * 0.6 / (0.75 + 0.6))
        self.assertAlmostEqual(m.false_positive_rate(self.c), 1 / 5)
        self.assertAlmostEqual(m.recall_gap(self.c), abs(0.6 - 0.8))

    def test_undefined_is_nan(self):
        c = m.confusion_counts([0, 0], [0, 0])   # tidak ada kelas ai
        self.assertTrue(math.isnan(m.recall_ai(c)))
        self.assertTrue(math.isnan(m.precision(c)))


class TestOtherFormulas(unittest.TestCase):
    def test_gaps(self):
        self.assertAlmostEqual(m.generalization_gap(0.95, 0.91), 0.04)
        self.assertAlmostEqual(m.compression_drop(0.93, 0.85), 0.08)
        self.assertAlmostEqual(m.range_gap([0.9, 0.82, float("nan"), 0.95]), 0.13)
        self.assertAlmostEqual(m.max_prob_diff([0.1, 0.5], [0.1004, 0.4999]), 0.0004)
        self.assertAlmostEqual(m.old_generator_drop(0.94, 0.92), 0.02)
        self.assertAlmostEqual(m.retrain_diff(0.91, 0.915), 0.005)

    def test_percentile_95(self):
        times = list(range(1, 21))               # n=20 -> ceil(19) = 19 -> t_19 = 19
        self.assertEqual(m.percentile_95(times), 19)
        self.assertEqual(m.percentile_95(list(range(1, 201))), 190)
        self.assertAlmostEqual(m.mean_time([1, 2, 3]), 2)

    def test_sus(self):
        self.assertEqual(m.sus_score([5, 1] * 5), 100.0)
        self.assertEqual(m.sus_score([1, 5] * 5), 0.0)
        self.assertEqual(m.sus_score([3] * 10), 50.0)
        with self.assertRaises(ValueError):
            m.sus_score([3] * 9)


class TestPreprocess(unittest.TestCase):
    def test_server_path_matches_offline_harmonize(self):
        """T7 (sebagian): jalur server = harmonize offline + eval transform, bit demi bit."""
        import io
        from src.preprocess import eval_transform, harmonize_bytes, load_rgb
        img = Image.new("RGB", (640, 480))
        img.putdata([((x * 7) % 256, (y * 3) % 256, (x + y) % 256) for y in range(480) for x in range(640)])
        raw = io.BytesIO()
        img.save(raw, format="PNG")
        offline = load_rgb(io.BytesIO(harmonize_bytes(load_rgb(io.BytesIO(raw.getvalue())))))
        server = preprocess_inference(io.BytesIO(raw.getvalue()))
        self.assertEqual((server - eval_transform(224)(offline).unsqueeze(0)).abs().max().item(), 0.0)
        self.assertEqual(harmonize(img).size, (256, 256))


if __name__ == "__main__":
    unittest.main()
