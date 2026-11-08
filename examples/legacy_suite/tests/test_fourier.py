"""Checks the lifted null relation and the nonzero field constraint."""
import unittest
import numpy as np
from simulations.fourier import evaluated_partials, certified_norm_bound


class FourierChecks(unittest.TestCase):
    def test_true_partials_annihilate_rates(self):
        phase = np.linspace(0, 2 * np.pi, 87, endpoint=False)
        row = np.array([0, 1, 1, 0, 0, 0]) / np.sqrt(2)
        rates = np.column_stack([np.ones(len(phase)), -np.cos(phase)])
        residual = np.einsum("ij,ij->i", evaluated_partials(row, phase), rates)
        self.assertLess(np.max(np.abs(residual)), 1e-14)

    def test_norm_bound_certifies_the_continuous_cycle(self):
        row = np.array([0, 1, 1, 0, 0, 0]) / np.sqrt(2)
        bound = certified_norm_bound(row)
        exact_minimum = 1 / np.sqrt(2)
        self.assertGreater(bound, 0.1)
        self.assertLessEqual(bound, exact_minimum)


if __name__ == "__main__":
    unittest.main()
