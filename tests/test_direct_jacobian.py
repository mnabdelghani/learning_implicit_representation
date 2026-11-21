"""Exact-Jacobian checks independent of the direct-learning simulation loop."""
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))
from direct_jacobian_audit import evaluated, lift, residual


class DirectJacobianTests(unittest.TestCase):
    def test_all_candidate_rows_are_exact_context_jacobians(self):
        rng = np.random.default_rng(912)
        for _ in range(30):
            q = rng.normal(size=4)
            z = rng.normal(size=2)
            eps = 1e-6
            fd = np.array([(residual(z + eps * np.eye(2)[j], q) -
                            residual(z - eps * np.eye(2)[j], q)) / (2 * eps) for j in range(2)])
            np.testing.assert_allclose(evaluated(q, z[0]), fd, atol=2e-9, rtol=1e-7)

    def test_rank_three_and_physical_nullspace(self):
        phase = np.arange(4096) * 2 * np.pi / 4096
        for A in [.6, 1., 1.6]:
            psi = np.array([lift([1., -A * np.cos(x)], x) for x in phase])
            target = np.array([0., 1., A, 0.])
            np.testing.assert_allclose(psi @ target, 0., atol=1e-14)
            actual = np.linalg.eigvalsh(psi.T @ psi / len(psi))
            np.testing.assert_allclose(actual, sorted([0., .5, 1., (1 + A * A) / 2]), atol=2e-14)

    def test_tangent_lift_is_complete_derivative(self):
        rng = np.random.default_rng(83)
        for _ in range(30):
            q = rng.normal(size=4)
            v = rng.normal(size=2)
            phase = rng.normal()
            self.assertAlmostEqual(q @ lift(v, phase), evaluated(q, phase) @ v, places=13)


if __name__ == "__main__": unittest.main()
