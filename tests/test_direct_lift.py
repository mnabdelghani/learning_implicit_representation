"""Independent checks of the direct lifted-flow supplementary experiment."""
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))
from direct_lift_audit import evaluated, flow_step, lift


class DirectLiftTests(unittest.TestCase):
    def test_analytic_nullspace_and_positive_complement(self):
        phase = np.arange(4096) * 2 * np.pi / 4096
        for amplitude in [.6, 1., 1.6]:
            psi = np.array([lift([1., -amplitude * np.cos(t)], t) for t in phase])
            target = np.array([0., 1., amplitude, 0., 0., 0.])
            np.testing.assert_allclose(psi @ target, 0., atol=1e-14)
            eig = np.linalg.eigvalsh(psi.T @ psi / len(psi))
            self.assertLess(abs(eig[0]), 1e-12)
            self.assertGreater(eig[1], .01)

    def test_exact_flow_derivative_and_energy(self):
        rng = np.random.default_rng(31)
        for _ in range(12):
            X = rng.normal(size=(9, 6)); R = X.T @ X / 9
            q = rng.normal(size=6); q /= np.linalg.norm(q)
            gain = 3.
            actual = (flow_step(q, R, gain, 1e-7) - q) / 1e-7
            expected = -gain * (q @ R - (q @ R @ q) * q)
            np.testing.assert_allclose(actual, expected, atol=4e-6, rtol=4e-6)
            updated = flow_step(q, R, gain, .04)
            self.assertAlmostEqual(np.linalg.norm(updated), 1., places=14)
            self.assertLessEqual(updated @ R @ updated, q @ R @ q + 1e-12)

    def test_complete_lift_matches_evaluated_differential(self):
        rng = np.random.default_rng(17)
        for _ in range(20):
            q = rng.normal(size=6); v = rng.normal(size=2); phase = rng.normal()
            self.assertAlmostEqual(q @ lift(v, phase), evaluated(q, phase) @ v, places=13)


if __name__ == "__main__":
    unittest.main()
