"""Unit tests for DriftGuard. Run with:  python -m unittest -v"""
import tempfile
import unittest

import numpy as np

from src import config
from src.data_generator import drift_phase, generate_batch
from src.drift_detector import DriftDetector, psi, psi_level
from src.registry import ModelRegistry
from src.train import evaluate, train


class TestPSI(unittest.TestCase):
    def test_identical_distributions_have_near_zero_psi(self):
        rng = np.random.default_rng(0)
        x = rng.normal(0, 1, 5000)
        self.assertLess(psi(x, x), 0.01)

    def test_shifted_distribution_is_critical(self):
        rng = np.random.default_rng(0)
        ref, cur = rng.normal(0, 1, 5000), rng.normal(1, 1, 5000)
        self.assertEqual(psi_level(psi(ref, cur)), "critical")

    def test_psi_levels(self):
        self.assertEqual(psi_level(0.05), "ok")
        self.assertEqual(psi_level(0.15), "warning")
        self.assertEqual(psi_level(0.30), "critical")


class TestDataGenerator(unittest.TestCase):
    def test_columns_and_size(self):
        b = generate_batch(500, 1)
        self.assertEqual(len(b), 500)
        for f in config.FEATURES + [config.TARGET]:
            self.assertIn(f, b.columns)

    def test_phases(self):
        self.assertEqual(drift_phase(2), "stable")
        self.assertEqual(drift_phase(6), "data_drift")
        self.assertEqual(drift_phase(11), "concept_drift")

    def test_same_seed_is_reproducible(self):
        a, b = generate_batch(200, 3), generate_batch(200, 3)
        self.assertTrue(a.equals(b))


class TestDriftDetector(unittest.TestCase):
    def setUp(self):
        self.ref = generate_batch(3000, 0)
        self.detector = DriftDetector(self.ref)

    def test_no_drift_in_stable_week(self):
        rep = self.detector.check(generate_batch(2000, 2), 2)
        self.assertFalse(rep.data_drift)

    def test_detects_data_drift(self):
        rep = self.detector.check(generate_batch(2000, 8), 8)
        self.assertTrue(rep.data_drift)
        self.assertTrue(rep.retrain_needed)

    def test_performance_drift_flag(self):
        rep = self.detector.check(generate_batch(2000, 1), 1, auc=0.70, baseline_auc=0.86)
        self.assertTrue(rep.performance_drift)


class TestTrainingAndRegistry(unittest.TestCase):
    def test_model_beats_random(self):
        model, metrics, _ = train(generate_batch(3000, 0))
        self.assertGreater(metrics["auc"], 0.75)

    def test_registry_versions_and_promotion(self):
        with tempfile.TemporaryDirectory() as tmp:
            reg = ModelRegistry(tmp)
            model, metrics, holdout = train(generate_batch(1500, 0))
            v1 = reg.register(model, metrics, "week 0")
            reg.promote(v1)
            v2 = reg.register(model, metrics, "week 1")
            reg.promote(v2)
            self.assertEqual(reg.production_entry()["version"], 2)
            self.assertEqual(reg.entries[0]["status"], "archived")
            loaded = reg.load()
            self.assertAlmostEqual(evaluate(loaded, holdout)["auc"], metrics["auc"], places=6)


if __name__ == "__main__":
    unittest.main()
