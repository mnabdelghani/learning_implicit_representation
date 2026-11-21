import sys
import unittest
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'experiments'))
import estimator_comparison as comparison
import fourier_online


class EstimatorComparisonTests(unittest.TestCase):
    def test_box_solver_known_active_optimum(self):
        # Objective (theta0-3)^2 + 2(theta1+2)^2; both optimum bounds active.
        A = np.array([[1., 0., -3.], [0., np.sqrt(2), 2 * np.sqrt(2)]])
        x, info = comparison.solve_box_memory(A.T @ A, [-1., -1.], [1., 1.])
        np.testing.assert_allclose(x, [1., -1.], atol=1e-12)
        self.assertEqual(info['solver_success'], 1)

    def test_box_solver_matches_independent_constrained_optimizer(self):
        rng = np.random.default_rng(20261118)
        for _ in range(10):
            A = rng.normal(size=(30, 7))
            R = A.T @ A
            lo, hi = np.full(6, -.15), np.full(6, .2)
            x, _ = comparison.solve_box_memory(R, lo, hi)
            def fun(theta):
                q = np.r_[theta, 1.]
                return q @ R @ q / 2
            result = minimize(fun, np.zeros(6), jac=lambda z: R[:-1, :-1] @ z + R[:-1, -1],
                              bounds=list(zip(lo, hi)), method='SLSQP',
                              options=dict(ftol=1e-12, maxiter=200))
            self.assertTrue(result.success)
            self.assertLessEqual(abs(fun(x) - fun(result.x)), 1e-9)

    def test_same_stream_original_controller_and_causal_scores(self):
        cfg = dict(fourier_online.CONFIG, duration=.3)
        original = fourier_online.simulate(10, 'learned', cfg)
        histories, stream = comparison.simulate(10, cfg=cfg)
        for old, new in zip(original, histories['memory_pg']):
            for key in ('state', 'control', 'preupdate_energy', 'F_error'):
                np.testing.assert_allclose(old[key], new[key], rtol=1e-12, atol=1e-12)
            for j in range(6):
                self.assertEqual(old[f'c{j}'], new[f'c{j}'])
        lag = round(cfg['difference_window'] / cfg['dt'])
        for method, rows in histories.items():
            for n in range(lag, len(stream)):
                def feature(row):
                    h = fourier_online.harmonics(row['phase'])
                    return np.r_[h * row['observed_rate'], h * row['state'], -row['control']]
                v = (feature(stream[n]) - feature(stream[n - lag])) / cfg['difference_window']
                q = np.r_[[rows[n][f'c{j}'] for j in range(6)], 1.]
                self.assertAlmostEqual((v @ q) ** 2 / 2, rows[n]['preupdate_energy'], places=12)

    def test_current_candidate_memory_and_final_frozen_update(self):
        cfg = fourier_online.CONFIG
        rng = np.random.default_rng(20)
        learners = [comparison.make_learner(method, cfg) for method in comparison.METHODS]
        history = []
        for _ in range(20):
            free, supplied = rng.normal(size=6), rng.normal()
            history.append(learners[0].channels(free, supplied)[0])
            for learner in learners:
                learner.observe(free, supplied)
        q = np.r_[np.arange(6) / 7, 1.]
        beta = learners[0].beta
        direct = sum((1 - beta) * beta ** (len(history) - i - 1) * (v @ q) ** 2 / 2
                     for i, v in enumerate(history))
        for learner in learners:
            self.assertAlmostEqual(q @ learner.R @ q / 2, direct, places=12)
            saved = learner.coefficients.copy()
            learner.observe(np.ones(6), 5., adapt=False)
            np.testing.assert_array_equal(saved, learner.coefficients)

    def test_unknown_partial_metric_excludes_supplied_coefficient(self):
        cfg = fourier_online.CONFIG
        c = np.asarray(cfg['initial'])
        e = comparison.evaluated_errors(c, .3, -.4, 1.2, cfg)
        F = np.array([e[f'F_true_{j}'] for j in range(4)]) * cfg['context_scales']
        Fest = np.array([e[f'F_est_{j}'] for j in range(4)]) * cfg['context_scales']
        ix = [0, 1, 3]
        self.assertAlmostEqual(e['learned_partial_error'], np.linalg.norm((Fest-F)[ix]) /
                               np.linalg.norm(F[ix]))
        self.assertGreater(e['learned_partial_error'], e['F_error'])

    def test_instant_nlms_projected_sample_descent(self):
        learner = comparison.make_learner('instant_nlms', fourier_online.CONFIG)
        rng = np.random.default_rng(45)
        for _ in range(100):
            free, supplied = rng.normal(size=6), rng.normal()
            v = learner.channels(free, supplied)[0]
            before = float((v @ np.r_[learner.theta, 1.]) ** 2 / 2)
            learner.observe(free, supplied)
            after = float((v @ np.r_[learner.theta, 1.]) ** 2 / 2)
            self.assertLessEqual(after, before + 1e-12)
            self.assertTrue(np.all(learner.theta >= learner.lower))
            self.assertTrue(np.all(learner.theta <= learner.upper))


if __name__ == '__main__':
    unittest.main()
