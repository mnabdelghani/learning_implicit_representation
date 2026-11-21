"""Independent checks of the coupled, fully homogeneous Fourier mechanism."""
import copy
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))
import normalized_fourier as model
from audit_normalized_fourier import coupled_step as reference_step, projected_steps


def config(duration=.18):
    cfg = dict(model.fourier.CONFIG, **json.loads(model.PROTOCOL_PATH.read_text()))
    cfg["duration"] = duration
    return cfg


class NormalizedFourierTests(unittest.TestCase):
    def test_every_candidate_has_the_complete_four_context_jacobian(self):
        rng = np.random.default_rng(91)
        S = np.asarray(config()["scales"])
        for _ in range(40):
            q, z = rng.normal(size=7), rng.normal(size=4)
            q /= np.linalg.norm(q)
            epsilon = 1e-6
            fd = np.array([(model.residual(z + epsilon * axis, q, S) -
                            model.residual(z - epsilon * axis, q, S)) / (2 * epsilon)
                           for axis in np.eye(4)])
            np.testing.assert_allclose(model.differential(z, q, S), fd, rtol=2e-7, atol=2e-9)
            self.assertAlmostEqual(model.differential(z, q, S)[2], -q[6] * S[6])

    def test_complete_features_differentiate_state_phase_products(self):
        rng = np.random.default_rng(32)
        S = np.asarray(config()["scales"])
        for _ in range(15):
            q, z, velocity = rng.normal(size=7), rng.normal(size=4), rng.normal(size=4)
            epsilon = 2e-6
            feature_rate = (model.features(z + epsilon * velocity) -
                            model.features(z - epsilon * velocity)) / (2 * epsilon)
            np.testing.assert_allclose((q * S) @ feature_rate,
                                       model.differential(z, q, S) @ velocity,
                                       rtol=2e-7, atol=2e-9)

    def test_coupled_step_matches_direct_nonlinear_equations_with_noncommuting_memory(self):
        rng = np.random.default_rng(731)
        for dt in [.01, .15]:
            factor = rng.normal(size=(7, 7))
            R = factor @ factor.T / 4
            w = rng.normal(size=7)
            q = rng.normal(size=7)
            q /= np.linalg.norm(q)
            self.assertGreater(np.linalg.norm(R @ np.outer(w, w) - np.outer(w, w) @ R), 1.)
            actual_q, actual_R, _ = model.coupled_step(q, R, w, dt, 8., 1.)
            expected_q, expected_R = reference_step(q, R, w, dt, 8., 1.)
            np.testing.assert_allclose(actual_q, expected_q, atol=2e-9, rtol=2e-9)
            np.testing.assert_allclose(actual_R, expected_R, atol=3e-13, rtol=3e-13)
            self.assertAlmostEqual(np.linalg.norm(actual_q), 1., places=14)

    def test_memory_uses_integral_kernel_and_all_seven_coefficients_can_adapt(self):
        q = np.asarray([1., .2, -.3, .8, -.1, .4, 1.])
        q /= np.linalg.norm(q)
        w = np.asarray([.8, -1.1, .4, .3, -.7, .9, -1.2])
        q_next, R_next, _ = model.coupled_step(q, np.zeros((7, 7)), w, .2, 8., 1.)
        expected = 8. * (-np.expm1(-.2 / 8.)) * np.outer(w, w)
        np.testing.assert_allclose(R_next, expected, atol=1e-15, rtol=1e-15)
        self.assertTrue(np.all(np.abs(q_next - q) > 1e-7))
        self.assertNotAlmostEqual(q_next[6], q[6], places=7)

    def test_command_is_gauge_invariant_and_uses_frozen_prior_when_guarded(self):
        cfg = config()
        q = np.asarray([1.1, .7, -.2, .9, .3, -.1, .8])
        q /= np.linalg.norm(q)
        ordinary = model.command(q, -.32, 1.7, cfg)
        self.assertFalse(ordinary[1])
        for gauge in [-13., -.03, .02, 4.]:
            transformed = model.command(gauge * q, -.32, 1.7, cfg)
            self.assertFalse(transformed[1])
            self.assertAlmostEqual(transformed[0], ordinary[0], places=13)
        unsafe = np.asarray([1., 3., 2., -4., 1., 2., 1e-6])
        expected = model.command(model.initial_q(cfg), -.32, 1.7, cfg)[0]
        for gauge in [-1e4, -.1, 1., 1e6]:
            command, guarded = model.command(gauge * unsafe, -.32, 1.7, cfg)
            self.assertTrue(guarded)
            self.assertEqual(command, expected)

    def test_preupdate_scores_and_final_no_assimilation(self):
        cfg = config()
        histories, _, arrays, _ = model.simulate(23, cfg)
        lag = round(cfg["difference_window"] / cfg["dt"])
        q0 = model.initial_q(cfg)
        np.testing.assert_array_equal(arrays["R_pre"][:lag + 1], 0.)
        np.testing.assert_allclose(arrays["q_pre"][:lag + 1], np.broadcast_to(q0, (lag + 1, 3, 7)))
        np.testing.assert_array_equal(arrays["q_pre"][1:], arrays["q_post"][:-1])
        np.testing.assert_array_equal(arrays["R_pre"][1:], arrays["R_post"][:-1])
        np.testing.assert_array_equal(arrays["q_post"][-1], arrays["q_pre"][-1])
        np.testing.assert_array_equal(arrays["R_post"][-1], arrays["R_pre"][-1])
        self.assertEqual(arrays["nfev"][-1], 0)
        for method_index, method in enumerate(model.METHODS):
            rows = histories[method]
            q = np.asarray([[row[f"q{j}"] for j in range(7)] for row in rows])
            np.testing.assert_array_equal(q, arrays["q_pre"][:, method_index])
            expected = np.sum(q[lag:] * arrays["w"][lag:], axis=1) ** 2 / 2
            np.testing.assert_allclose([row["preupdate_energy"] for row in rows[lag:]], expected, atol=1e-15)
        np.testing.assert_array_equal(arrays["q_post"][:, 2], np.broadcast_to(q0, (len(rows), 7)))

    def test_evaluator_poisoning_cannot_change_control_or_learning(self):
        cfg = config()
        _, ordinary_observations, ordinary_arrays, _ = model.simulate(29, cfg)
        original = model.evaluate_candidate

        def poison(*args, **kwargs):
            out = original(*args, **kwargs)
            return {key: 37. + float(value) for key, value in out.items()}

        with patch.object(model, "evaluate_candidate", side_effect=poison):
            poisoned_histories, poisoned_observations, poisoned_arrays, _ = model.simulate(29, cfg)
        self.assertEqual(ordinary_observations, poisoned_observations)
        for key in ordinary_arrays:
            np.testing.assert_array_equal(ordinary_arrays[key], poisoned_arrays[key])
        self.assertGreater(poisoned_histories["normalized_flow"][0]["aligned_F_error"], 37.)

    def test_replay_clean_rate_is_not_a_learner_channel(self):
        cfg = config()
        _, stream, _, _ = model.simulate(31, cfg)
        ordinary, _, arrays, _ = model.simulate(31, cfg, replay=stream)
        poisoned_stream = copy.deepcopy(stream)
        for n, row in enumerate(poisoned_stream):
            row["evaluator_actual_rate"] += 17. * (n + 1)
        poisoned, _, poisoned_arrays, _ = model.simulate(31, cfg, replay=poisoned_stream)
        for key in arrays:
            np.testing.assert_array_equal(arrays[key], poisoned_arrays[key])
        self.assertNotEqual(ordinary["normalized_flow"][-1]["clean_energy"],
                            poisoned["normalized_flow"][-1]["clean_energy"])

    def test_sensor_noise_reaches_learning_after_the_first_premeasurement_command(self):
        quiet, noisy = config(), config()
        quiet["rate_noise"], noisy["rate_noise"] = 0., .03
        _, a, aa, _ = model.simulate(31, quiet)
        _, b, bb, _ = model.simulate(31, noisy)
        self.assertEqual(a[0]["control"], b[0]["control"])
        np.testing.assert_array_equal(aa["q_pre"][0], bb["q_pre"][0])
        self.assertGreater(np.linalg.norm(aa["q_post"][-1, 0] - bb["q_post"][-1, 0]), 1e-6)

    def test_bottom_estimate_solves_unit_objective_including_degeneracy(self):
        rng = np.random.default_rng(93)
        U = np.linalg.qr(rng.normal(size=(7, 7)))[0]
        previous = rng.normal(size=7)
        previous /= np.linalg.norm(previous)
        for eigenvalues in [np.arange(7), np.asarray([0., 0., 0., 1., 2., 3., 4.])]:
            R = (U * eigenvalues) @ U.T
            estimate, _, multiplicity = model.bottom_estimate(R, previous)
            self.assertAlmostEqual(estimate @ R @ estimate, eigenvalues[0], places=12)
            self.assertAlmostEqual(np.linalg.norm(estimate), 1., places=14)
            self.assertEqual(multiplicity, np.count_nonzero(eigenvalues == eigenvalues[0]))
            self.assertGreaterEqual(estimate @ previous, 0.)

    def test_independent_batched_checker_agrees_with_coupled_reference(self):
        rng = np.random.default_rng(14)
        q = rng.normal(size=(3, 7))
        q /= np.linalg.norm(q, axis=1)[:, None]
        factors = rng.normal(size=(3, 7, 7))
        R = factors @ np.transpose(factors, (0, 2, 1))
        w = rng.normal(size=(3, 7))
        checked = projected_steps(q, R, w, .01, 8., 1.)
        expected = np.asarray([reference_step(q[k], R[k], w[k], .01, 8., 1.)[0] for k in range(3)])
        np.testing.assert_allclose(checked, expected, atol=2e-11, rtol=2e-11)

    def test_integration_failure_retains_every_preceding_observation_and_state(self):
        cfg = config()
        ordinary_histories, ordinary_stream, ordinary_arrays, _ = model.simulate(17, cfg)
        original = model.coupled_step
        calls = 0

        def fail_third(*args, **kwargs):
            nonlocal calls
            calls += 1
            if calls == 3:
                raise FloatingPointError("deliberate audit failure")
            return original(*args, **kwargs)

        with patch.object(model, "coupled_step", side_effect=fail_third):
            with self.assertRaises(model.SimulationFailure) as captured:
                model.simulate(17, cfg)
        histories, stream, arrays, rng = captured.exception.partial
        lag = round(cfg["difference_window"] / cfg["dt"])
        self.assertEqual(len(stream), lag + 3)
        self.assertEqual(stream, ordinary_stream[:lag + 3])
        for key in arrays:
            np.testing.assert_array_equal(arrays[key], ordinary_arrays[key][:lag + 2])
        for method in model.METHODS:
            self.assertEqual(len(histories[method]), lag + 3)
            for j in range(7):
                np.testing.assert_array_equal([row[f"q{j}"] for row in histories[method]],
                                              [row[f"q{j}"] for row in ordinary_histories[method][:lag + 3]])
        self.assertTrue(rng["used"])


if __name__ == "__main__":
    unittest.main()
