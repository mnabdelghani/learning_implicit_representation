"""Independent numerical and information-boundary checks of direct Q learning."""
from pathlib import Path
import sys
import unittest

import numpy as np
from scipy.integrate import solve_ivp

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))
from direct_field import coupled_step, complete_rates, random_frame


class DirectFieldTests(unittest.TestCase):
    def test_single_and_multirow_match_independent_joint_ode(self):
        rng = np.random.default_rng(101)
        for m, d in [(1, 5), (3, 8)]:
            q = random_frame(m, d, 27)
            factor = rng.normal(size=(d, d))
            R = factor @ factor.T / d
            v = rng.normal(size=d)
            self.assertGreater(np.linalg.norm(R @ np.outer(v, v) - np.outer(v, v) @ R), .1)
            dt, decay, eta = .17, .4, 1.2

            def joint_rhs(t, flat):
                a, B = flat[:m*d].reshape(m, d), flat[m*d:].reshape(d, d)
                gradient = a @ B
                tangent = gradient - ((gradient @ a.T + a @ gradient.T) / 2) @ a
                return np.r_[-eta * tangent.ravel(), (np.outer(v, v) - decay * B).ravel()]

            reference = solve_ivp(joint_rhs, (0, dt), np.r_[q.ravel(), R.ravel()],
                                  method="RK45", rtol=2e-12, atol=2e-14,
                                  max_step=dt / 40).y[:, -1]
            actual_q, actual_R, check = coupled_step(q, R, v, dt, decay, eta)
            np.testing.assert_allclose(actual_q, reference[:m*d].reshape(m, d), atol=2e-10, rtol=2e-10)
            np.testing.assert_allclose(actual_R, reference[m*d:].reshape(d, d), atol=2e-12, rtol=2e-12)
            self.assertLess(check["polar_correction"], 1e-9)

    def test_every_entry_can_change_and_scores_use_preupdate_field(self):
        q = random_frame(3, 8, 19)
        v = np.random.default_rng(7).normal(size=8)
        updated, R, check = coupled_step(q, np.zeros((8, 8)), v, .4, .2, 2.)
        self.assertTrue(np.all(np.abs(updated - q) > 1e-8))
        np.testing.assert_allclose(check["preupdate_incompatibility"], q @ v, atol=0)
        np.testing.assert_allclose(updated @ updated.T, np.eye(3), atol=2e-14)
        expected = -np.expm1(-.2 * .4) / .2 * np.outer(v, v)
        np.testing.assert_allclose(R, expected, atol=1e-14)

    def test_causal_readout_and_row_gauge_for_compatible_rates(self):
        q = random_frame(2, 5, 11)
        S, C = [0, 1, 2], [3, 4]
        supplied = np.array([.2, -.5, .7])
        result = complete_rates(q, S, supplied, C)
        np.testing.assert_allclose(q[:, S] @ supplied + q[:, C] @ result, 0., atol=1e-14)
        G = np.array([[2., .3], [-.2, .9]])
        np.testing.assert_allclose(complete_rates(G @ q, S, supplied, C), result, atol=1e-14)
        bad = q.copy(); bad[:, C] = 0
        with self.assertRaises(ValueError):
            complete_rates(bad, S, supplied, C)
        with self.assertRaises(ValueError):
            complete_rates(bad, S, supplied, C, minimum_singular_value=0.)
        with self.assertRaises(ValueError):
            complete_rates(np.empty((0, 5)), S, supplied, C)

    def test_learns_unknown_constant_plane_from_tangents_and_reuses_for_readout(self):
        # Truth is used only by this test's data generator and evaluator.
        truth = random_frame(1, 4, 5)
        tangent = np.eye(4) - truth.T @ truth
        q = random_frame(1, 4, 12)
        R = np.zeros((4, 4))
        rng = np.random.default_rng(13)
        initial_error = np.linalg.norm(q.T @ q - truth.T @ truth)
        for _ in range(400):
            observed = tangent @ rng.normal(size=4)
            q, R, _ = coupled_step(q, R, observed, .04, .5, 3.)
        self.assertLess(np.linalg.norm(q.T @ q - truth.T @ truth), initial_error / 1000)
        supplied = np.array([.7, -.2, .1])
        estimated = complete_rates(q, [0, 1, 2], supplied, [3])
        expected = -(truth[0, :3] @ supplied) / truth[0, 3]
        np.testing.assert_allclose(estimated, [expected], atol=1e-5)


if __name__ == "__main__":
    unittest.main()
