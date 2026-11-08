"""Meaningful algebra, causality, readout, and controller checks for Acrobot."""
import importlib.util
from pathlib import Path
import unittest

import numpy as np

MODULE_PATH = Path(__file__).resolve().parents[1] / "simulations" / "acrobot.py"
SPEC = importlib.util.spec_from_file_location("acrobot_under_test", MODULE_PATH)
acrobot = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(acrobot)


class AcrobotTests(unittest.TestCase):
    def setUp(self):
        self.c = np.array([5/3, 0.5, 1/3, 14.715, 4.905])
        self.rng = np.random.default_rng(204)

    def test_complete_features_equal_implicit_residual(self):
        for _ in range(30):
            z = self.rng.normal(size=7)
            actual = acrobot.complete_features(z[:2], z[2:4], z[4:6]) @ self.c - acrobot.ACTUATOR * z[6]
            np.testing.assert_allclose(actual, acrobot.residual(z, self.c), atol=1e-13)

    def test_all_partial_columns_against_finite_differences(self):
        epsilon = 1e-6
        for _ in range(20):
            z = self.rng.normal(size=7)
            numeric = np.column_stack([
                (acrobot.residual(z + epsilon*np.eye(7)[j], self.c)
                 - acrobot.residual(z - epsilon*np.eye(7)[j], self.c))/(2*epsilon)
                for j in range(7)
            ])
            np.testing.assert_allclose(acrobot.implicit_partials(z, self.c), numeric,
                                       rtol=2e-7, atol=7e-9)

    def test_complete_product_differentiation_agrees_with_F_zdot(self):
        # A random direction need not be a true plant tangent. The full chain
        # rule must hold for every candidate model and data-space direction.
        candidate = np.array([1.4, 0.23, 0.31, 12.2, 3.8])
        epsilon = 1e-6
        for _ in range(20):
            z = self.rng.normal(size=7)
            direction = self.rng.normal(size=7)
            high, low = z + epsilon*direction, z - epsilon*direction
            dy = (acrobot.complete_features(high[:2], high[2:4], high[4:6])
                  - acrobot.complete_features(low[:2], low[2:4], low[4:6]))/(2*epsilon)
            observed_directional = dy @ candidate - acrobot.ACTUATOR * direction[6]
            np.testing.assert_allclose(observed_directional,
                acrobot.implicit_partials(z, candidate) @ direction, rtol=1e-7, atol=1e-8)

    def test_plant_tangent_is_in_kernel_of_true_differential(self):
        for _ in range(20):
            q, w = self.rng.normal(size=(2, 2))
            u, udot = self.rng.normal(size=2)
            acceleration = acrobot.forward_readout(q, w, u, self.c)
            z = np.concatenate((q, w, acceleration, [u]))
            F = acrobot.implicit_partials(z, self.c)
            jerk = np.linalg.solve(F[:, 4:6],
                acrobot.ACTUATOR * udot - F[:, :2] @ w - F[:, 2:4] @ acceleration)
            zdot = np.concatenate((w, acceleration, jerk, [udot]))
            np.testing.assert_allclose(F @ zdot, np.zeros(2), atol=2e-13)

    def test_feasible_inverse_enforces_passive_joint(self):
        for _ in range(40):
            q = self.rng.uniform(-np.pi, np.pi, 2)
            w = self.rng.uniform(-2, 2, 2)
            target = self.rng.uniform(-3, 3)
            acceleration, torque = acrobot.feasible_inverse(q, w, target, self.c)
            np.testing.assert_allclose(acrobot.residual(np.concatenate((q, w, acceleration, [torque])), self.c),
                                       np.zeros(2), atol=8e-15)
            np.testing.assert_allclose(acrobot.forward_readout(q, w, torque, self.c), acceleration, atol=2e-14)
            self.assertAlmostEqual(acceleration[1], target, places=14)

    def test_mass_bound_covers_all_configurations(self):
        bound = acrobot.mass_positivity_bound(self.c)
        sampled_minimum = min(np.linalg.eigvalsh(acrobot.mass_matrix([0, q2], self.c))[0]
                              for q2 in np.linspace(-np.pi, np.pi, 701))
        self.assertGreater(bound, 0)
        self.assertAlmostEqual(bound, sampled_minimum, places=13)
        with self.assertRaises(ValueError):
            acrobot.lqr_gain([0.2, 0.8, 0.4, 12, 3], np.eye(4), 0.8)

    def test_learner_is_causal_and_decreases_frozen_memory_objective(self):
        prior = np.array([1.5, 0.3, 0.4, 12, 3.5])
        scales = np.array([1, 1, 1, 10, 10])
        learner = acrobot.ImplicitDifferentialLearner(prior, scales, 0.01, 0.03, 2.0, 10.0)
        history = []
        manual_H = np.zeros((5, 5))
        manual_g = np.zeros(5)
        for index in range(18):
            q, w, acceleration = self.rng.normal(size=(3, 2))
            u = float(self.rng.normal())
            previous_c = learner.coefficients.copy()
            history.append((acrobot.complete_features(q, w, acceleration), u))
            observed = learner.observe(q, w, acceleration, u)
            np.testing.assert_array_equal(observed["coefficients_before"], previous_c)
            if index < 3:
                self.assertFalse(observed["updated"])
                np.testing.assert_array_equal(learner.H, np.zeros((5, 5)))
                np.testing.assert_allclose(learner.coefficients, prior)
            else:
                self.assertTrue(observed["updated"])
                dy = (history[index][0] - history[index-3][0])/0.03
                du = (history[index][1] - history[index-3][1])/0.03
                matrix, known = dy*scales, -acrobot.ACTUATOR*du
                manual_H = learner.beta*manual_H + 0.01*(matrix.T@matrix)
                manual_g = learner.beta*manual_g + 0.01*(matrix.T@known)
                np.testing.assert_allclose(learner.H, manual_H, rtol=1e-13, atol=1e-10)
                np.testing.assert_allclose(learner.g, manual_g, rtol=1e-13, atol=1e-10)
                self.assertLessEqual(observed["memory_objective_after"],
                                     observed["memory_objective_before"] + 1e-8)

    def test_lqr_linearizes_actual_forward_readout(self):
        star = np.array([np.pi, 0, 0, 0])
        K, A, G, P = acrobot.lqr_gain(self.c, np.diag([30, 30, 4, 4]), 0.8, star)
        epsilon = 1e-6
        numeric_A = np.column_stack([
            (acrobot.state_rhs(star + epsilon*np.eye(4)[j], 0, self.c)
             - acrobot.state_rhs(star - epsilon*np.eye(4)[j], 0, self.c))/(2*epsilon)
            for j in range(4)
        ])
        numeric_G = ((acrobot.state_rhs(star, epsilon, self.c)
                     - acrobot.state_rhs(star, -epsilon, self.c))/(2*epsilon))[:, None]
        np.testing.assert_allclose(A, numeric_A, rtol=1e-8, atol=2e-8)
        np.testing.assert_allclose(G, numeric_G, rtol=1e-12, atol=1e-12)
        self.assertLess(np.max(np.real(np.linalg.eigvals(A-G@K))), 0)
        np.testing.assert_allclose(P, P.T, atol=1e-12)


if __name__ == "__main__":
    unittest.main()
