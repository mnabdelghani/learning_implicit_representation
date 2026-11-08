"""Meaningful consistency checks for the new Burgers reconstruction."""
import copy
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from simulations.burgers import (
    ConstrainedEnergyLearner, TRACE_COLUMNS, implicit_jacobian, implicit_residual,
    incompatibility, inverse_control, load_config, plant_rhs, reference_field,
    residual_features, rk4_step, simulate_episode, spatial_grid,
    spectral_derivatives,
)


class BurgersTests(unittest.TestCase):
    def setUp(self):
        self.cfg = load_config()
        self.s = spatial_grid(64)
        self.truth = np.array([1.0, 0.08, 1.0])

    def test_fourier_spatial_derivatives(self):
        w = 0.2 + np.sin(3 * self.s) + 0.3 * np.cos(2 * self.s)
        ws, wss = spectral_derivatives(w)
        np.testing.assert_allclose(ws, 3 * np.cos(3 * self.s) - 0.6 * np.sin(2 * self.s), atol=5e-14)
        np.testing.assert_allclose(wss, -9 * np.sin(3 * self.s) - 1.2 * np.cos(2 * self.s), atol=1e-12)

    def test_F_is_directional_derivative_of_implicit_equation(self):
        rng = np.random.default_rng(41)
        z, direction = rng.normal(size=(32, 5)), rng.normal(size=(32, 5))
        eps = 1e-6
        numeric = (implicit_residual(z + eps * direction, self.truth) -
                   implicit_residual(z - eps * direction, self.truth)) / (2 * eps)
        analytic = np.sum(implicit_jacobian(z, self.truth) * direction, axis=1)
        np.testing.assert_allclose(numeric, analytic, atol=2e-9, rtol=2e-9)

    def test_complete_feature_difference_clean_compatibility(self):
        w0 = 0.2 * np.sin(self.s) + 0.1 * np.cos(2 * self.s)
        w1 = 0.23 * np.sin(self.s) + 0.07 * np.cos(2 * self.s)
        u0, u1 = 0.2 * np.cos(self.s), 0.3 * np.sin(3 * self.s)
        delta = 0.05
        a = residual_features(w0, u0, plant_rhs(w0, u0, self.truth))
        b = residual_features(w1, u1, plant_rhs(w1, u1, self.truth))
        rates = (b - a) / delta
        np.testing.assert_allclose(incompatibility(rates, self.truth), 0, atol=3e-13)
        ws0, _ = spectral_derivatives(w0)
        ws1, _ = spectral_derivatives(w1)
        np.testing.assert_allclose(rates[:, 1], (w1 * ws1 - w0 * ws0) / delta, atol=1e-14)
        omitted_product_term = w1 * (ws1 - ws0) / delta
        self.assertGreater(np.linalg.norm(rates[:, 1] - omitted_product_term), 0.1)

    def test_unforced_mass_conservation_and_viscous_dissipation(self):
        w = 0.37 + 0.2 * np.sin(self.s) + 0.1 * np.cos(2 * self.s)
        mean0, energy0 = float(np.mean(w)), float(np.mean(w * w))
        for _ in range(200):
            w = rk4_step(w, np.zeros_like(w), 0.005, self.truth)
        self.assertAlmostEqual(float(np.mean(w)), mean0, delta=2e-13)
        self.assertLess(float(np.mean(w * w)), energy0)

    def test_pure_diffusion_matches_known_fourier_decay(self):
        w = 0.4 + np.sin(3 * self.s)
        for _ in range(100):
            w = rk4_step(w, np.zeros_like(w), 0.005, [0, 0.08, 0])
        expected = 0.4 + np.exp(-0.08 * 9 * 0.5) * np.sin(3 * self.s)
        np.testing.assert_allclose(w, expected, atol=2e-10, rtol=0)

    def test_exact_instantaneous_inverse_readout(self):
        w = 0.23 * np.sin(self.s) - 0.06 * np.cos(2 * self.s)
        wd, wdt = reference_field(self.s, 1.23)
        u = inverse_control(w, self.s, 1.23, self.truth)
        np.testing.assert_allclose(plant_rhs(w, u, self.truth) - wdt,
                                   -2 * (w - wd), atol=5e-14, rtol=0)

    def test_frozen_memory_projected_energy_descent(self):
        learner = ConstrainedEnergyLearner(self.cfg)
        rng = np.random.default_rng(21)
        rates = rng.normal(size=(256, 4))
        rates[:, 0] = -rates[:, 1:] @ np.array([1.2, 0.07, 1.3]) + 0.05 * rng.normal(size=256)
        learner.accumulate(rates)
        initial = learner.energy()
        outcome = learner.projected_steps()
        self.assertLess(learner.energy(), initial)
        self.assertTrue(np.all(np.diff(outcome["energies"]) <= 1e-13))
        self.assertLessEqual(outcome["max_descent_violation"], 1e-13)
        self.assertTrue(np.all(learner.coefficients >= self.cfg["bounds_lower"]))
        self.assertTrue(np.all(learner.coefficients <= self.cfg["bounds_upper"]))
        self.assertGreaterEqual(np.linalg.eigvalsh(learner.moment)[0], -1e-13)

    def test_short_closed_loop_and_causal_startup(self):
        cfg = copy.deepcopy(self.cfg)
        cfg.update(duration=0.2, evaluation_window=0.1)
        result = simulate_episode(cfg, "learned", 10)
        self.assertEqual(result["trace"].shape, (41, len(TRACE_COLUMNS)))
        self.assertTrue(np.isfinite(result["metrics"]["tracking_rms"]))
        self.assertGreaterEqual(result["metrics"]["F_min_row_norm"], 1.0)
        lookup = {name: result["trace"][:, i] for i, name in enumerate(TRACE_COLUMNS)}
        self.assertTrue(np.isnan(lookup["sample_energy_noisy_preupdate"][:10]).all())
        self.assertTrue(np.isfinite(lookup["sample_energy_noisy_preupdate"][10:]).all())
        np.testing.assert_allclose(result["trace"][0, 11:14], cfg["initial_coefficients"])
        self.assertEqual(lookup["command_applied"][-1], 0)
        self.assertEqual(lookup["command_applied"][:-1].sum(), 40)
        self.assertLessEqual(result["metrics"]["largest_descent_bound_violation"], 1e-12)


if __name__ == "__main__":
    unittest.main()
