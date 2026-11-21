"""Adversarial checks of evaluator/learner boundaries, not proof certificates.

These tests deliberately corrupt evaluator-only channels. A change in learned
coefficients or physical commands would reveal an unintended information path.
They keep the plant and sensor paths intact, including their true coefficients.
"""
import copy
from collections import deque as real_deque
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))
sys.path.insert(0, str(ROOT / "examples" / "legacy_suite" / "simulations"))
import contact
import fourier_online
import burgers


def poison_second_history():
    """The simulations create measured history first and clean history second."""
    created = 0

    class PoisonedHistory(real_deque):
        def append(self, value):
            value = np.asarray(value, dtype=float)
            # A time-dependent perturbation also survives endpoint differences.
            self._append_count = getattr(self, "_append_count", 0) + 1
            super().append(value + 37.0 * self._append_count)

    def factory(*args, **kwargs):
        nonlocal created
        created += 1
        return (PoisonedHistory if created == 2 else real_deque)(*args, **kwargs)

    return factory


def columns(rows, names):
    return np.asarray([[row[name] for name in names] for row in rows])


class InformationBoundaryTests(unittest.TestCase):
    def test_contact_diagnostics_cannot_change_control_or_learning(self):
        cfg = copy.deepcopy(contact.CONFIG)
        cfg["duration"] = 0.3
        original = contact.differential
        ordinary, _ = contact.simulate(23, "learned", cfg)
        with patch.object(contact, "differential", side_effect=lambda *a, **k: 11.0 + original(*a, **k)), \
                patch.object(contact, "deque", side_effect=poison_second_history()):
            poisoned, _ = contact.simulate(23, "learned", cfg)
        causal = ["x", "normal_force", "u1", "u2", "preupdate_energy", "s1", "s2", "a", "r1", "r2", "b", "B11", "B12", "B21", "B22"]
        np.testing.assert_array_equal(columns(ordinary, causal), columns(poisoned, causal))
        self.assertNotEqual(ordinary[-1]["Fx_true"], poisoned[-1]["Fx_true"])
        self.assertNotEqual(ordinary[-1]["clean_energy"], poisoned[-1]["clean_energy"])

    def test_fourier_diagnostics_cannot_change_control_or_learning(self):
        cfg = copy.deepcopy(fourier_online.CONFIG)
        cfg["duration"] = 0.3
        original = fourier_online.differential
        ordinary = fourier_online.simulate(23, "learned", cfg)
        with patch.object(fourier_online, "differential", side_effect=lambda *a, **k: 11.0 + original(*a, **k)), \
                patch.object(fourier_online, "deque", side_effect=poison_second_history()):
            poisoned = fourier_online.simulate(23, "learned", cfg)
        causal = ["state", "control", "preupdate_energy", "K0_same_stream_energy"] + [f"c{j}" for j in range(6)]
        np.testing.assert_array_equal(columns(ordinary, causal), columns(poisoned, causal))
        self.assertNotEqual(ordinary[-1]["phase_partial_true"], poisoned[-1]["phase_partial_true"])
        self.assertNotEqual(ordinary[-1]["clean_energy"], poisoned[-1]["clean_energy"])

    def test_burgers_diagnostics_cannot_change_control_or_learning(self):
        cfg = burgers.load_config()
        cfg.update(duration=0.2, evaluation_window=0.1)
        original = burgers.implicit_jacobian
        ordinary = burgers.simulate_episode(cfg, "learned", 23)
        with patch.object(burgers, "implicit_jacobian", side_effect=lambda *a, **k: 11.0 + original(*a, **k)), \
                patch.object(burgers, "deque", side_effect=poison_second_history()):
            poisoned = burgers.simulate_episode(cfg, "learned", 23)
        excluded = {"F_relative_error_preupdate", "F_min_row_norm_preupdate", "sample_energy_clean_preupdate"}
        ix = [j for j, name in enumerate(burgers.TRACE_COLUMNS) if name not in excluded]
        np.testing.assert_array_equal(ordinary["trace"][:, ix], poisoned["trace"][:, ix])
        for lhs, rhs in zip(ordinary["snapshots"], poisoned["snapshots"]):
            for a, b in zip(lhs, rhs):
                np.testing.assert_array_equal(a, b)
        j = burgers.TRACE_COLUMNS.index("sample_energy_clean_preupdate")
        self.assertNotEqual(ordinary["trace"][20, j], poisoned["trace"][20, j])

    def test_measured_rate_noise_reaches_fourier_learning(self):
        # A positive control for the noninterference tests: sensor perturbations
        # should affect the estimator, while the first premeasurement command
        # and its pre-update coefficients remain the same.
        quiet = copy.deepcopy(fourier_online.CONFIG)
        quiet.update(duration=0.3, rate_noise=0.0)
        noisy = dict(quiet, rate_noise=0.02)
        a = fourier_online.simulate(23, "learned", quiet)
        b = fourier_online.simulate(23, "learned", noisy)
        self.assertEqual(a[0]["control"], b[0]["control"])
        names = [f"c{j}" for j in range(6)]
        np.testing.assert_array_equal(columns(a[:1], names), columns(b[:1], names))
        self.assertGreater(np.linalg.norm(columns(a[-1:], names) - columns(b[-1:], names)), 1e-8)

    def test_burgers_learner_does_not_access_truth_configuration(self):
        class TruthForbidden(dict):
            def __getitem__(self, key):
                if key == "true_coefficients":
                    raise AssertionError("True coefficients entered the learner")
                return super().__getitem__(key)

        cfg = TruthForbidden(burgers.load_config())
        learner = burgers.ConstrainedEnergyLearner(cfg)
        samples = np.random.default_rng(23).normal(size=(9, 4))
        learner.update(samples)
        self.assertTrue(np.all(np.isfinite(learner.coefficients)))


if __name__ == "__main__":
    unittest.main()
