"""Numerical-algebra checks for the new unvalidated airfoil prototype.

These verify the projector, the implicit Frechet derivative, and matched
zero-input trajectories. They do not constitute aerodynamic validation.
"""
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from simulations.airfoil import FlowPlant, DifferentialLearner, load_config, readout_control, solve_box_quadratic


class AirfoilNumericsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cfg = load_config()
        cls.cfg.update(nx=48, ny=32, dt=0.002, sample_dt=0.01)
        cls.plant = FlowPlant(cls.cfg)

    def test_projection_is_solenoidal_and_idempotent(self):
        raw = np.random.default_rng(34).normal(size=(2, self.cfg["ny"], self.cfg["nx"]))
        projected = self.plant.project(raw)
        self.assertLess(np.sqrt(np.mean(self.plant.divergence(projected) ** 2)), 2e-13)
        np.testing.assert_allclose(self.plant.project(projected), projected, atol=2e-14)
        self.assertLess(np.linalg.norm(projected), np.linalg.norm(raw))

    def test_implicit_frechet_action_matches_centered_difference(self):
        p = self.plant
        rng = np.random.default_rng(16)
        state = p.project(rng.normal(size=p.uniform.shape) * 0.08) + p.uniform
        direction = p.project(rng.normal(size=p.uniform.shape)) * 0.1
        rate = p.project(rng.normal(size=p.uniform.shape))
        delta_rate = p.project(rng.normal(size=p.uniform.shape))
        theta = np.array([0.83, 0.017, 0.66])
        command, delta_input, eps = 0.31, -0.24, 2e-6
        derivative = p.implicit_differential_action(state, direction, delta_rate, delta_input, theta)
        plus = p.implicit_residual(state + eps * direction, rate + eps * delta_rate,
                                   command + eps * delta_input, theta)
        minus = p.implicit_residual(state - eps * direction, rate - eps * delta_rate,
                                    command - eps * delta_input, theta)
        np.testing.assert_allclose((plus - minus) / (2 * eps), derivative, rtol=2e-7, atol=3e-8)

    def test_coupled_box_quadratic_uses_true_active_set_solution(self):
        hessian = np.array([[3.0, 2.0, 0.0], [2.0, 3.0, 0.0], [0.0, 0.0, 1.0]])
        unconstrained = np.array([2.0, 0.3, 0.5])
        linear = hessian @ unconstrained
        result = solve_box_quadratic(hessian, linear, np.zeros(3), np.ones(3))
        np.testing.assert_allclose(result, [1.0, (4.9 - 2.0) / 3.0, 0.5], atol=1e-14)
        objective = lambda x: 0.5 * x @ hessian @ x - linear @ x
        self.assertLess(objective(result), objective(np.clip(unconstrained, 0, 1)))
        old = np.array([0.7, 0.8, 0.6])
        self.assertLessEqual(objective(old + 0.25 * (result - old)), objective(old))

    def test_default_grid_projector_removes_cutoff_modes(self):
        p = FlowPlant(load_config())
        raw = np.random.default_rng(55).normal(size=p.uniform.shape)
        projected = p.project(raw)
        self.assertTrue(np.isrealobj(projected))
        modes = p._to_hat(projected)
        self.assertLess(np.max(np.abs(modes[:, ~p.cutoff])), 3e-12)
        self.assertLess(np.sqrt(np.mean(p.divergence(projected) ** 2)), 3e-13)

    def test_known_damping_jacobian(self):
        p = self.plant
        direction = p.project(np.random.default_rng(6).normal(size=p.uniform.shape))
        eps = 1e-5
        numerical = (p.known_forcing(p.uniform + eps * direction)
                     - p.known_forcing(p.uniform - eps * direction)) / (2 * eps)
        np.testing.assert_allclose(numerical, p.known_forcing_differential(direction),
                                   rtol=3e-9, atol=2e-9)

    def test_nonzero_acceleration_block_is_identity(self):
        p = self.plant
        rate_direction = np.random.default_rng(7).normal(size=p.uniform.shape)
        derivative = p.implicit_differential_action(p.uniform, np.zeros_like(p.uniform),
                                                  rate_direction, 0.0, np.zeros(3))
        np.testing.assert_allclose(derivative, rate_direction)

    def test_zero_bound_controller_matches_no_actuation(self):
        p = self.plant
        cfg = dict(self.cfg, input_bound=0.0)
        observed = p.initial_state(10)
        baseline = observed.copy()
        theta = np.array([0.7, 0.018, 0.7])
        for _ in range(5):
            command, _ = readout_control(p, observed, theta, 0.1, 0.0, cfg)
            self.assertEqual(command, 0.0)
            observed = p.advance(observed, command, cfg["sample_dt"], 0.55)
            baseline = p.advance(baseline, 0.0, cfg["sample_dt"], 0.55)
        np.testing.assert_array_equal(observed, baseline)
        self.assertTrue(np.all(np.isfinite(observed)))

    def test_frozen_observer_does_not_modify_parameters(self):
        p = self.plant
        learner = DifferentialLearner(self.cfg)
        original = learner.theta.copy()
        state = p.initial_state(3)
        for command in (-0.3, 0.25, 0.1):
            after = p.advance(state, command, self.cfg["sample_dt"], 0.9)
            learner.observe(p, state, after, command, self.cfg["sample_dt"], update=False)
            state = after
        np.testing.assert_array_equal(learner.theta, original)

    def test_body_force_has_zero_integral_and_nonzero_readout_coupling(self):
        p = self.plant
        self.assertLess(abs(p.body_force[1].sum()), 1e-12)
        self.assertGreater(p.CB, 0.01)
        np.testing.assert_allclose(p.project(p.body_force), p.B, atol=1e-14)


if __name__ == "__main__":
    unittest.main()
