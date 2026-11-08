#!/usr/bin/env python3
"""NEW RECONSTRUCTION of the Acrobot implicit-differential experiment.

The learner is given measured (q, w, acceleration, u), a residual family,
and the known actuator location. Its data loss is a causal complete-feature
approximation to F(z) zdot. True coefficients/partials are used by the plant,
the oracle comparator, and explicitly named evaluation diagnostics only.

Example:
  python simulations/acrobot.py --output results/acrobot --seed 10
  python simulations/acrobot.py --output results/acrobot --seeds 10 11 12
"""
from __future__ import annotations

import argparse
from collections import deque
import copy
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy.linalg import solve_continuous_are

STATUS = "NEW_RECONSTRUCTION"
COEFFICIENT_NAMES = ("a", "b", "d", "p", "r")
ACTUATOR = np.array([0.0, 1.0])
DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / "configs" / "acrobot.json"


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    with open(path or DEFAULT_CONFIG, encoding="utf-8") as stream:
        config = json.load(stream)
    validate_config(config)
    return config


def validate_config(config: dict[str, Any]) -> None:
    for name in ("true_coefficients", "prior_coefficients", "coefficient_scales"):
        value = np.asarray(config[name], dtype=float)
        if value.shape != (5,) or not np.all(np.isfinite(value)):
            raise ValueError(f"{name} must contain five finite numbers")
    if np.any(np.asarray(config["coefficient_scales"]) <= 0):
        raise ValueError("coefficient_scales must be positive")
    for name in ("true_coefficients", "prior_coefficients"):
        if mass_positivity_bound(config[name]) <= 0:
            raise ValueError(f"{name} must define positive mass at every configuration")
    identify = config["identification"]
    for key in ("duration", "step", "memory_time", "learning_gain", "feature_difference_interval"):
        if identify[key] <= 0:
            raise ValueError(f"identification.{key} must be positive")
    ratio = identify["feature_difference_interval"] / identify["step"]
    if ratio < 1 or abs(ratio - round(ratio)) > 1e-10:
        raise ValueError("feature_difference_interval must be an integer number of samples")
    if identify["acceleration_noise_std"] < 0:
        raise ValueError("acceleration noise standard deviation cannot be negative")
    for section in ("identification", "control"):
        ratio = config[section]["duration"] / config[section]["step"]
        if abs(ratio - round(ratio)) > 1e-8:
            raise ValueError(f"{section} duration must be an integer number of steps")
    if config["queries"]["count"] < 1:
        raise ValueError("queries.count must be positive")
    if config["control"]["R"] <= 0 or config["control"]["step"] <= 0:
        raise ValueError("control R and step must be positive")
    if len(config["control"]["Q_diagonal"]) != 4 or min(config["control"]["Q_diagonal"]) <= 0:
        raise ValueError("Q_diagonal must contain four positive numbers")
    if config["control"]["evaluation_window"] > config["control"]["duration"]:
        raise ValueError("control evaluation_window cannot exceed duration")


def mass_matrix(q: np.ndarray, coefficients: np.ndarray) -> np.ndarray:
    """M(q) for q=(q1,q2), c=(a,b,d,p,r)."""
    a, b, d, _, _ = np.asarray(coefficients, dtype=float)
    cosine = np.cos(q[1])
    return np.array([[a + 2.0 * b * cosine, d + b * cosine],
                     [d + b * cosine, d]], dtype=float)


def force_vector(q: np.ndarray, w: np.ndarray, coefficients: np.ndarray) -> np.ndarray:
    _, b, _, p, r = np.asarray(coefficients, dtype=float)
    sine = np.sin(q[1])
    return np.array([
        -b * sine * (2.0 * w[0] * w[1] + w[1] ** 2)
        + p * np.sin(q[0]) + r * np.sin(q[0] + q[1]),
        b * sine * w[0] ** 2 + r * np.sin(q[0] + q[1]),
    ])


def residual(z: np.ndarray, coefficients: np.ndarray) -> np.ndarray:
    """The actual implicit family f_c(z)=M(q) acceleration+h(q,w)-B u."""
    z = np.asarray(z, dtype=float)
    return mass_matrix(z[:2], coefficients) @ z[4:6] + force_vector(z[:2], z[2:4], coefficients) - ACTUATOR * z[6]


def complete_features(q: np.ndarray, w: np.ndarray, acceleration: np.ndarray) -> np.ndarray:
    """Complete Y in f_c=Y c-B u; differencing Y includes all product terms."""
    sine, cosine = np.sin(q[1]), np.cos(q[1])
    a1, a2 = acceleration
    return np.array([
        [a1, cosine * (2.0 * a1 + a2) - sine * (2.0 * w[0] * w[1] + w[1] ** 2),
         a2, np.sin(q[0]), np.sin(q[0] + q[1])],
        [0.0, cosine * a1 + sine * w[0] ** 2,
         a1 + a2, 0.0, np.sin(q[0] + q[1])],
    ])


def implicit_partials(z: np.ndarray, coefficients: np.ndarray) -> np.ndarray:
    """Exact 2 by 7 D_z f_c, including derivatives of M and all products.

    z=(q1,q2,w1,w2,acceleration1,acceleration2,u). This function evaluates
    either a learned model or a diagnostic reference; it is never a label
    passed into ImplicitDifferentialLearner.
    """
    q1, q2, w1, w2, a1, a2, _ = np.asarray(z, dtype=float)
    _, b, _, p, r = np.asarray(coefficients, dtype=float)
    s2, c2, c12 = np.sin(q2), np.cos(q2), np.cos(q1 + q2)
    result = np.zeros((2, 7))
    result[:, 0] = [p * np.cos(q1) + r * c12, r * c12]
    result[:, 1] = [-b * s2 * (2 * a1 + a2) - b * c2 * (2 * w1 * w2 + w2 ** 2) + r * c12,
                    -b * s2 * a1 + b * c2 * w1 ** 2 + r * c12]
    result[:, 2:4] = [[-2 * b * s2 * w2, -2 * b * s2 * (w1 + w2)],
                      [2 * b * s2 * w1, 0.0]]
    result[:, 4:6] = mass_matrix(np.array([q1, q2]), coefficients)
    result[:, 6] = -ACTUATOR
    return result


def mass_positivity_bound(coefficients: np.ndarray) -> float:
    """Exact minimum eigenvalue over all q (minimum of the two endpoints).

    M is a symmetric matrix linear in cos(q2). Its minimum eigenvalue is
    concave in that scalar, so its interval minimum occurs at an endpoint.
    """
    endpoints = (mass_matrix(np.array([0.0, 0.0]), coefficients),
                 mass_matrix(np.array([0.0, np.pi]), coefficients))
    return float(min(np.linalg.eigvalsh(matrix)[0] for matrix in endpoints))


def forward_readout(q: np.ndarray, w: np.ndarray, u: float, coefficients: np.ndarray) -> np.ndarray:
    return np.linalg.solve(mass_matrix(q, coefficients), ACTUATOR * u - force_vector(q, w, coefficients))


def feasible_inverse(q: np.ndarray, w: np.ndarray, desired_a2: float,
                     coefficients: np.ndarray) -> tuple[np.ndarray, float]:
    """Return feasible acceleration pair and torque after enforcing row 1."""
    mass = mass_matrix(q, coefficients)
    force = force_vector(q, w, coefficients)
    if abs(mass[0, 0]) < 1e-12:
        raise ValueError("The passive-joint completion is singular: M11 is zero")
    a1 = -(mass[0, 1] * desired_a2 + force[0]) / mass[0, 0]
    acceleration = np.array([a1, desired_a2])
    torque = float(mass[1] @ acceleration + force[1])
    return acceleration, torque


def state_rhs(state: np.ndarray, u: float, coefficients: np.ndarray) -> np.ndarray:
    return np.concatenate((state[2:4], forward_readout(state[:2], state[2:4], u, coefficients)))


def rk4_step(rhs, time: float, state: np.ndarray, step: float) -> np.ndarray:
    k1 = rhs(time, state)
    k2 = rhs(time + 0.5 * step, state + 0.5 * step * k1)
    k3 = rhs(time + 0.5 * step, state + 0.5 * step * k2)
    k4 = rhs(time + step, state + step * k3)
    return state + step * (k1 + 2 * k2 + 2 * k3 + k4) / 6.0


def excitation(time: float, config: dict[str, Any]) -> float:
    amplitudes = np.asarray(config["excitation_amplitudes"])
    phases = np.asarray(config["excitation_phases"])
    frequencies = np.asarray(config["excitation_frequencies"])
    return float(amplitudes @ np.sin(frequencies * time + phases))


class ImplicitDifferentialLearner:
    """Causal learner that has no access to true coefficients or partials.

    The known actuator column fixes the residual scale. The differentiated
    family gives xi=(Delta Y/Delta t)c - B(Delta u/Delta t). Candidate
    coefficients are held fixed when differentiating in the data direction.
    """
    def __init__(self, prior_coefficients: np.ndarray, coefficient_scales: np.ndarray,
                 step: float, difference_interval: float, memory_time: float,
                 learning_gain: float):
        self.scales = np.asarray(coefficient_scales, dtype=float).copy()
        self.theta = np.asarray(prior_coefficients, dtype=float) / self.scales
        self.step = float(step)
        self.difference_interval = float(difference_interval)
        self.lag = int(round(difference_interval / step))
        if self.lag < 1 or abs(self.lag * step - difference_interval) > 1e-12:
            raise ValueError("difference interval must be an integer multiple of step")
        self.beta = float(np.exp(-step / memory_time))
        self.gain = float(learning_gain)
        self.H = np.zeros((5, 5))
        self.g = np.zeros(5)
        self.constant = 0.0
        self.history = deque(maxlen=self.lag + 1)
        self.samples = 0

    @property
    def coefficients(self) -> np.ndarray:
        return self.scales * self.theta

    def observe(self, q: np.ndarray, w: np.ndarray, measured_acceleration: np.ndarray,
                u: float) -> dict[str, Any]:
        before = self.coefficients.copy()
        self.history.append((complete_features(q, w, measured_acceleration), float(u)))
        self.samples += 1
        if len(self.history) <= self.lag:
            return {"updated": False, "coefficients_before": before,
                    "coefficients_after": before.copy()}
        new_y, new_u = self.history[-1]
        old_y, old_u = self.history[0]
        derivative_y = (new_y - old_y) / self.difference_interval
        derivative_u = (new_u - old_u) / self.difference_interval
        matrix = derivative_y * self.scales[np.newaxis, :]
        known = -ACTUATOR * derivative_u
        residual_before = matrix @ self.theta + known
        self.H = self.beta * self.H + self.step * (matrix.T @ matrix)
        self.H = 0.5 * (self.H + self.H.T)
        self.g = self.beta * self.g + self.step * (matrix.T @ known)
        self.constant = self.beta * self.constant + 0.5 * self.step * float(known @ known)
        objective_before = self.objective(self.theta)
        eta_h = self.gain * self.step
        self.theta = np.linalg.solve(np.eye(5) + eta_h * self.H,
                                     self.theta - eta_h * self.g)
        residual_after = matrix @ self.theta + known
        return {
            "updated": True,
            "coefficients_before": before,
            "coefficients_after": self.coefficients.copy(),
            "derivative_features": derivative_y,
            "derivative_input": derivative_u,
            "energy_before": 0.5 * float(residual_before @ residual_before),
            "energy_after": 0.5 * float(residual_after @ residual_after),
            "memory_objective_before": objective_before,
            "memory_objective_after": self.objective(self.theta),
        }

    def objective(self, theta: np.ndarray) -> float:
        return float(0.5 * theta @ self.H @ theta + theta @ self.g + self.constant)


def relative_coefficient_rms(estimate: np.ndarray, reference: np.ndarray) -> float:
    return float(np.sqrt(np.mean(((estimate - reference) / reference) ** 2)))


def relative_partials_error(estimate: np.ndarray, reference: np.ndarray) -> float:
    return float(np.linalg.norm(estimate - reference) / max(np.linalg.norm(reference), 1e-15))


def identify(config: dict[str, Any], seed: int) -> tuple[np.ndarray, list[dict[str, float]], dict[str, Any]]:
    parameters = config["identification"]
    reference = np.asarray(config["true_coefficients"], dtype=float)
    prior = np.asarray(config["prior_coefficients"], dtype=float)
    learner = ImplicitDifferentialLearner(prior, config["coefficient_scales"],
        parameters["step"], parameters["feature_difference_interval"],
        parameters["memory_time"], parameters["learning_gain"])
    rng = np.random.default_rng(seed)
    step = parameters["step"]
    samples = int(round(parameters["duration"] / step))
    state = np.asarray(parameters["initial_state"], dtype=float)
    clean_history = deque(maxlen=learner.lag + 1)
    rows = []
    invalid_online_mass_count = 0
    finite_online_forward_errors = []
    descent_violations = []
    for index in range(samples + 1):
        time = index * step
        u = excitation(time, parameters)
        clean_acceleration = forward_readout(state[:2], state[2:], u, reference)
        measured_acceleration = clean_acceleration + rng.normal(0, parameters["acceleration_noise_std"], 2)
        clean_y = complete_features(state[:2], state[2:], clean_acceleration)
        clean_history.append(clean_y)
        before = learner.coefficients.copy()
        # These are predictions before the current observation is assimilated.
        try:
            prediction = forward_readout(state[:2], state[2:], u, before)
            forward_error = float(np.linalg.norm(prediction - clean_acceleration))
            if np.isfinite(forward_error):
                finite_online_forward_errors.append((time, forward_error))
        except np.linalg.LinAlgError:
            forward_error = np.nan
        z = np.concatenate((state, clean_acceleration, [u]))
        true_F = implicit_partials(z, reference)  # evaluation only
        learned_F = implicit_partials(z, before)
        frozen_F = implicit_partials(z, prior)
        observed = learner.observe(state[:2], state[2:], measured_acceleration, u)
        after = learner.coefficients.copy()
        mass_bound = mass_positivity_bound(after)
        if mass_bound <= 0:
            invalid_online_mass_count += 1
        row = {"time": time, "q1": state[0], "q2": state[1], "w1": state[2], "w2": state[3],
               "input": u, "true_a1": clean_acceleration[0], "true_a2": clean_acceleration[1],
               "measured_a1": measured_acceleration[0], "measured_a2": measured_acceleration[1],
               "coefficient_relative_rms_pre": relative_coefficient_rms(before, reference),
               "coefficient_relative_rms_post": relative_coefficient_rms(after, reference),
               "partials_relative_error_pre": relative_partials_error(learned_F, true_F),
               "forward_vector_error_pre": forward_error,
               "mass_minimum_eigenvalue_post": mass_bound,
               "information_minimum_eigenvalue": float(np.linalg.eigvalsh(learner.H)[0]),
               "updated": float(observed["updated"])}
        for j, name in enumerate(COEFFICIENT_NAMES):
            row[f"c_{name}_pre"] = before[j]
            row[f"c_{name}_post"] = after[j]
        for i in range(2):
            for j in range(7):
                row[f"F{i+1}{j+1}_learned_pre"] = learned_F[i, j]
                row[f"F{i+1}{j+1}_true"] = true_F[i, j]
                row[f"F{i+1}{j+1}_frozen"] = frozen_F[i, j]
        energy_fields = ("energy_measured_pre", "energy_measured_post", "energy_measured_oracle",
                         "energy_clean_pre", "energy_clean_post", "energy_clean_oracle",
                         "memory_objective_before", "memory_objective_after")
        row.update({key: np.nan for key in energy_fields})
        if observed["updated"]:
            dy = observed["derivative_features"]
            du = observed["derivative_input"]
            clean_dy = (clean_history[-1] - clean_history[0]) / learner.difference_interval
            oracle_residual = dy @ reference - ACTUATOR * du
            clean_pre = clean_dy @ before - ACTUATOR * du
            clean_post = clean_dy @ after - ACTUATOR * du
            clean_oracle = clean_dy @ reference - ACTUATOR * du
            row.update({"energy_measured_pre": observed["energy_before"],
                        "energy_measured_post": observed["energy_after"],
                        "energy_measured_oracle": 0.5 * float(oracle_residual @ oracle_residual),
                        "energy_clean_pre": 0.5 * float(clean_pre @ clean_pre),
                        "energy_clean_post": 0.5 * float(clean_post @ clean_post),
                        "energy_clean_oracle": 0.5 * float(clean_oracle @ clean_oracle),
                        "memory_objective_before": observed["memory_objective_before"],
                        "memory_objective_after": observed["memory_objective_after"]})
            descent_violations.append(observed["memory_objective_after"] - observed["memory_objective_before"])
        rows.append(row)
        if index < samples:
            state = rk4_step(lambda t, x: state_rhs(x, excitation(t, parameters), reference), time, state, step)
            if not np.all(np.isfinite(state)) or np.linalg.norm(state) > 1e6:
                raise RuntimeError(f"Identification plant became nonfinite at t={time:g}")
    final = learner.coefficients.copy()
    theta_minimizer = np.linalg.lstsq(learner.H, -learner.g, rcond=None)[0]
    evaluation_start = max(0.0, parameters["duration"] - 2.0)
    final_rows = [row for row in rows if row["time"] >= evaluation_start and row["updated"]]
    metric = {
        "duration_s": parameters["duration"], "observations": len(rows),
        "learning_updates": sum(int(row["updated"]) for row in rows),
        "final_coefficients": final.tolist(),
        "final_relative_coefficient_rms": relative_coefficient_rms(final, reference),
        "final_information_minimum_eigenvalue": float(np.linalg.eigvalsh(learner.H)[0]),
        "final_information_eigenvalues": np.linalg.eigvalsh(learner.H).tolist(),
        "final_mass_minimum_eigenvalue_all_configurations": mass_positivity_bound(final),
        "nonpositive_mass_observations_during_learning": invalid_online_mass_count,
        "final_memory_objective": learner.objective(learner.theta),
        "frozen_memory_objective": learner.objective(prior / learner.scales),
        "oracle_memory_objective": learner.objective(reference / learner.scales),
        "fitted_noisy_memory_minimizer_coefficients": (theta_minimizer * learner.scales).tolist(),
        "scaled_distance_to_noisy_memory_minimizer": float(np.linalg.norm(learner.theta - theta_minimizer)),
        "maximum_frozen_objective_increase": max(0.0, float(max(descent_violations, default=0))),
        "tail_window_s": min(2.0, parameters["duration"]),
        "tail_mean_measured_incompatibility_energy_pre": float(np.mean([r["energy_measured_pre"] for r in final_rows])),
        "tail_mean_measured_oracle_energy": float(np.mean([r["energy_measured_oracle"] for r in final_rows])),
        "tail_mean_clean_incompatibility_energy_pre": float(np.mean([r["energy_clean_pre"] for r in final_rows])),
        "tail_max_clean_oracle_energy": float(max(r["energy_clean_oracle"] for r in final_rows)),
        "tail_mean_relative_partials_error_pre": float(np.mean([r["partials_relative_error_pre"] for r in final_rows])),
        "online_forward_vector_rmse_pre": float(np.sqrt(np.mean([error**2 for _, error in finite_online_forward_errors]))),
        "tail_online_forward_vector_rmse_pre": float(np.sqrt(np.mean([error**2 for t, error in finite_online_forward_errors if t >= evaluation_start]))),
    }
    return final, rows, metric


def evaluate_readouts(config: dict[str, Any], models: dict[str, np.ndarray]) -> tuple[dict[str, Any], list[dict[str, float]]]:
    settings = config["queries"]
    rng = np.random.default_rng(settings["seed"])
    count = settings["count"]
    qs = rng.uniform(-settings["angle_bound"], settings["angle_bound"], (count, 2))
    ws = rng.uniform(-settings["speed_bound"], settings["speed_bound"], (count, 2))
    inputs = rng.uniform(-settings["input_bound"], settings["input_bound"], count)
    commands = rng.uniform(-settings["desired_acceleration_bound"], settings["desired_acceleration_bound"], count)
    true_c = np.asarray(config["true_coefficients"], dtype=float)
    result: dict[str, Any] = {}
    valid_models = {name: c for name, c in models.items() if mass_positivity_bound(c) > 0}
    for name in models:
        if name not in valid_models:
            result[name] = {"status": "NONPOSITIVE_MASS", "reason": "Readout evaluation rejected a final nonphysical mass model"}
    errors = {name: {"forward": [], "inverse": [], "inverse_actual_a2": [], "partial": [], "feasibility": []} for name in valid_models}
    rows = []
    for index, (q, w, u, desired) in enumerate(zip(qs, ws, inputs, commands)):
        actual = forward_readout(q, w, float(u), true_c)
        _, true_torque = feasible_inverse(q, w, float(desired), true_c)
        z = np.concatenate((q, w, actual, [u]))
        true_F = implicit_partials(z, true_c)
        row = {"query": index, "q1": q[0], "q2": q[1], "w1": w[0], "w2": w[1],
               "forward_input": u, "desired_a2": desired, "true_a1": actual[0],
               "true_a2": actual[1], "true_inverse_torque": true_torque}
        for name, c in valid_models.items():
            acceleration = forward_readout(q, w, float(u), c)
            completed, torque = feasible_inverse(q, w, float(desired), c)
            actual_after_inverse = forward_readout(q, w, torque, true_c)
            feasible_residual = residual(np.concatenate((q, w, completed, [torque])), c)
            diagnostics = errors[name]
            diagnostics["forward"].append(float(np.sum((acceleration - actual) ** 2)))
            diagnostics["inverse"].append((torque - true_torque) ** 2)
            diagnostics["inverse_actual_a2"].append((actual_after_inverse[1] - desired) ** 2)
            diagnostics["partial"].append(relative_partials_error(implicit_partials(z, c), true_F) ** 2)
            diagnostics["feasibility"].append(float(np.linalg.norm(feasible_residual)))
            row.update({f"{name}_forward_a1": acceleration[0], f"{name}_forward_a2": acceleration[1],
                        f"{name}_inverse_a1": completed[0], f"{name}_inverse_torque": torque,
                        f"{name}_inverse_actual_a2": actual_after_inverse[1]})
        rows.append(row)
    for name, diagnostic in errors.items():
        result[name] = {"status": "OK", "forward_acceleration_vector_rmse_rad_s2": float(np.sqrt(np.mean(diagnostic["forward"]))),
                        "inverse_torque_rmse_Nm": float(np.sqrt(np.mean(diagnostic["inverse"]))),
                        "inverse_actual_actuated_acceleration_rmse_rad_s2": float(np.sqrt(np.mean(diagnostic["inverse_actual_a2"]))),
                        "relative_partials_rms_on_query_states": float(np.sqrt(np.mean(diagnostic["partial"]))),
                        "maximum_fitted_inverse_residual": max(diagnostic["feasibility"])}
    return result, rows


def lqr_gain(coefficients: np.ndarray, Q: np.ndarray, R: float,
             equilibrium: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Linearize the actual forward readout, then solve continuous LQR."""
    if mass_positivity_bound(coefficients) <= 0:
        raise ValueError("Cannot construct physical LQR: final mass model is not positive definite")
    equilibrium = np.asarray(equilibrium if equilibrium is not None else [np.pi, 0, 0, 0], dtype=float)
    if np.linalg.norm(state_rhs(equilibrium, 0.0, coefficients)) > 1e-9:
        raise ValueError("Specified LQR operating point is not an unforced equilibrium")
    partials = implicit_partials(np.concatenate((equilibrium, [0.0, 0.0, 0.0])), coefficients)
    mass = partials[:, 4:6]
    A = np.zeros((4, 4))
    A[:2, 2:] = np.eye(2)
    A[2:, :4] = -np.linalg.solve(mass, partials[:, :4])
    G = np.zeros((4, 1))
    G[2:, 0] = np.linalg.solve(mass, ACTUATOR)
    P = solve_continuous_are(A, G, Q, np.array([[R]], dtype=float))
    K = (G.T @ P) / R
    return K, A, G, P


def simulate_control(config: dict[str, Any], coefficients: np.ndarray) -> tuple[list[dict[str, float]], dict[str, Any]]:
    settings = config["control"]
    true_c = np.asarray(config["true_coefficients"], dtype=float)
    equilibrium = np.asarray(settings["equilibrium"], dtype=float)
    try:
        K, A, G, P = lqr_gain(coefficients, np.diag(settings["Q_diagonal"]), settings["R"], equilibrium)
    except (ValueError, np.linalg.LinAlgError) as error:
        return [], {"status": "CONTROLLER_UNAVAILABLE", "reason": str(error), "stabilized_in_final_window": False}
    state = np.asarray(settings["initial_state"], dtype=float)
    step = settings["step"]
    samples = int(round(settings["duration"] / step))
    rows = []
    status = "OK"
    for index in range(samples + 1):
        time = index * step
        error = state - equilibrium
        requested_u = -float((K @ error)[0])
        u = float(np.clip(requested_u, -settings["torque_limit"], settings["torque_limit"]))
        angle_error = np.arctan2(np.sin(error[:2]), np.cos(error[:2]))
        row = {"time": time, "q1": state[0], "q2": state[1], "w1": state[2], "w2": state[3],
               "q1_error_wrapped": angle_error[0], "q2_error_wrapped": angle_error[1],
               "angle_error_norm": float(np.linalg.norm(angle_error)), "speed_norm": float(np.linalg.norm(state[2:])),
               "requested_torque": requested_u, "applied_torque": u,
               "saturated": float(abs(requested_u) > settings["torque_limit"])}
        rows.append(row)
        if index < samples:
            state = rk4_step(lambda t, x: state_rhs(x, u, true_c), time, state, step)
            if not np.all(np.isfinite(state)) or np.linalg.norm(state) > 1e6:
                status = "PLANT_DIVERGED"
                break
    times = np.array([r["time"] for r in rows])
    angles = np.array([r["angle_error_norm"] for r in rows])
    speeds = np.array([r["speed_norm"] for r in rows])
    torque = np.array([r["applied_torque"] for r in rows])
    tail = times >= settings["duration"] - settings["evaluation_window"] - 1e-12
    completed = status == "OK" and len(rows) == samples + 1 and np.any(tail)
    final_angle_max = float(np.max(angles[tail])) if np.any(tail) else None
    final_speed_max = float(np.max(speeds[tail])) if np.any(tail) else None
    summary = {"status": status, "duration_requested_s": settings["duration"], "duration_completed_s": float(times[-1]),
               "gain": K.tolist(), "linearization_A": A.tolist(), "linearization_G": G.tolist(),
               "fitted_closed_loop_eigenvalues_real": np.linalg.eigvals(A - G @ K).real.tolist(),
               "fitted_closed_loop_eigenvalues_imag": np.linalg.eigvals(A - G @ K).imag.tolist(),
               "riccati_residual_norm": float(np.linalg.norm(A.T @ P + P @ A - P @ G @ G.T @ P / settings["R"] + np.diag(settings["Q_diagonal"]))),
               "full_episode_angle_vector_rmse_rad": float(np.sqrt(np.mean(angles ** 2))),
               "full_episode_per_angle_rmse_rad": float(np.sqrt(np.mean(angles ** 2) / 2)),
               "final_window_angle_vector_rmse_rad": float(np.sqrt(np.mean(angles[tail] ** 2))) if np.any(tail) else None,
               "final_window_per_angle_rmse_rad": float(np.sqrt(np.mean(angles[tail] ** 2) / 2)) if np.any(tail) else None,
               "final_window_maximum_angle_norm_rad": final_angle_max,
               "final_window_maximum_speed_norm_rad_s": final_speed_max,
               "final_window_s": settings["evaluation_window"],
               "maximum_absolute_torque_Nm": float(np.max(np.abs(torque))),
               "rms_torque_Nm": float(np.sqrt(np.mean(torque ** 2))),
               "saturation_fraction": float(np.mean([r["saturated"] for r in rows])),
               "stabilized_in_final_window": bool(completed and final_angle_max < settings["angle_norm_threshold"] and final_speed_max < settings["speed_norm_threshold"])}
    return rows, summary


def write_csv(path: Path, rows: list[dict[str, float]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def json_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_value(item) for item in value]
    if isinstance(value, np.ndarray):
        return json_value(value.tolist())
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, np.integer):
        return int(value)
    return value


def write_json(path: Path, value: Any) -> None:
    with path.open("w", encoding="utf-8") as stream:
        json.dump(json_value(value), stream, indent=2, allow_nan=False)
        stream.write("\n")


def plot_results(output: Path, identify_rows: list[dict[str, float]], controls: dict[str, list[dict[str, float]]],
                 config: dict[str, Any]) -> list[str]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
    artifacts = []
    times = np.array([r["time"] for r in identify_rows])
    def values(key):
        return np.asarray([row[key] for row in identify_rows], dtype=float)
    def rolling_mean(data, width):
        valid = np.isfinite(data)
        numerator = np.convolve(np.where(valid, data, 0), np.ones(width), "same")
        denominator = np.convolve(valid.astype(float), np.ones(width), "same")
        return np.divide(numerator, denominator, out=np.full_like(data, np.nan), where=denominator > 0)
    def save(figure, stem):
        for extension in ("png", "pdf"):
            name = f"{stem}.{extension}"
            figure.savefig(output / name, dpi=180, bbox_inches="tight")
            artifacts.append(name)
        plt.close(figure)
    fig, axes = plt.subplots(2, 2, figsize=(10, 6.6), constrained_layout=True)
    for key, label in (("energy_measured_pre", "learned, measured data"),
                       ("energy_measured_oracle", "oracle, measured data"),
                       ("energy_clean_pre", "learned, clean diagnostic")):
        energy = rolling_mean(values(key), max(1, round(0.1 / config["identification"]["step"])))
        axes[0, 0].semilogy(times, np.maximum(energy, 1e-15), label=label)
    axes[0, 0].set(xlabel="Time (s)", ylabel="Differential incompatibility energy",
                    title="100 ms displayed moving average")
    axes[0, 0].legend(fontsize=7)
    axes[0, 1].semilogy(times, np.maximum(values("coefficient_relative_rms_post"), 1e-15), label="coefficient relative RMS")
    axes[0, 1].semilogy(times, np.maximum(values("partials_relative_error_pre"), 1e-15), label="evaluated partial relative error")
    axes[0, 1].set(xlabel="Time (s)", ylabel="Diagnostic relative error")
    axes[0, 1].legend(fontsize=7)
    true_c = np.asarray(config["true_coefficients"])
    for j, name in enumerate(COEFFICIENT_NAMES):
        axes[1, 0].plot(times, values(f"c_{name}_post") / true_c[j], label=name)
    axes[1, 0].axhline(1, color="k", linewidth=.8, linestyle=":")
    axes[1, 0].set(xlabel="Time (s)", ylabel="Estimated / true coefficient")
    axes[1, 0].legend(ncol=5, fontsize=8)
    axes[1, 1].plot(times, values("information_minimum_eigenvalue"))
    axes[1, 1].set(xlabel="Time (s)", ylabel="Smallest scaled information eigenvalue")
    fig.suptitle("Acrobot identification — new reconstruction")
    save(fig, "acrobot_identification")
    fig, axes = plt.subplots(2, 2, figsize=(10, 6.2), sharex=True, constrained_layout=True)
    for axis, (i, j, label) in zip(axes.flat, [(1, 1, "partial f1 / partial q1"),
                                               (1, 2, "partial f1 / partial q2"),
                                               (1, 5, "partial f1 / partial acceleration1 = M11"),
                                               (2, 6, "partial f2 / partial acceleration2 = d")]):
        axis.plot(times, values(f"F{i}{j}_true"), color="black", linewidth=1.2, label="true (diagnostic)")
        axis.plot(times, values(f"F{i}{j}_learned_pre"), color="C0", linewidth=.9, label="learned before update")
        axis.plot(times, values(f"F{i}{j}_frozen"), color="C1", alpha=.75, linestyle=":", label="frozen prior")
        axis.set(title=label, xlabel="Time (s)")
    axes[0, 0].legend(fontsize=7)
    fig.suptitle("Evaluated F(z) along the identification trajectory")
    save(fig, "acrobot_partials")
    fig, axes = plt.subplots(2, 2, figsize=(10, 6.2), sharex=True, constrained_layout=True)
    for name, rows in controls.items():
        if not rows:
            continue
        ts = np.array([row["time"] for row in rows])
        axes[0, 0].semilogy(ts, np.maximum([row["angle_error_norm"] for row in rows], 1e-12), label=name)
        axes[0, 1].semilogy(ts, np.maximum([row["speed_norm"] for row in rows], 1e-12), label=name)
        axes[1, 0].plot(ts, [row["q2_error_wrapped"] for row in rows], label=name)
        axes[1, 1].plot(ts, [row["applied_torque"] for row in rows], label=name)
    axes[0, 0].axhline(config["control"]["angle_norm_threshold"], color="gray", linestyle=":")
    axes[0, 1].axhline(config["control"]["speed_norm_threshold"], color="gray", linestyle=":")
    axes[1, 0].axhline(0, color="gray", linestyle=":")
    labels = ("Wrapped angle-error norm (rad)", "Angular-speed norm (rad/s)",
              "q2 error (rad); target is zero", "Applied torque (N m)")
    for axis, ylabel in zip(axes.flat, labels):
        axis.set(xlabel="Time (s)", ylabel=ylabel)
        axis.axvspan(config["control"]["duration"] - config["control"]["evaluation_window"],
                     config["control"]["duration"], color="gray", alpha=.08)
    axes[0, 0].legend(fontsize=8)
    fig.suptitle("Separate nonlinear local-LQR episode — new reconstruction")
    save(fig, "acrobot_control")
    return artifacts


def run_experiment(config: dict[str, Any], seed: int, output: str | Path,
                   make_plots: bool = True) -> dict[str, Any]:
    validate_config(config)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    learned, id_rows, identification_metrics = identify(config, seed)
    models = {"learned": learned, "frozen": np.asarray(config["prior_coefficients"], dtype=float),
              "oracle": np.asarray(config["true_coefficients"], dtype=float)}
    readout_metrics, query_rows = evaluate_readouts(config, models)
    control_rows = {}
    control_metrics = {}
    for name, coefficients in models.items():
        control_rows[name], control_metrics[name] = simulate_control(config, coefficients)
    write_csv(output / "identification_trace.csv", id_rows)
    write_csv(output / "readout_queries.csv", query_rows)
    for name, rows in control_rows.items():
        write_csv(output / f"control_{name}.csv", rows)
    artifact_names = ["identification_trace.csv", "readout_queries.csv"] + [f"control_{name}.csv" for name, rows in control_rows.items() if rows]
    if make_plots:
        artifact_names.extend(plot_results(output, id_rows, control_rows, config))
    summary = {
        "status": STATUS, "seed": int(seed), "mode": config.get("run_mode", "main"),
        "claim": "New executable reconstruction with explicit missing-setting assumptions; not a reproduction of the manuscript table.",
        "config": copy.deepcopy(config),
        "identification": identification_metrics, "readouts": readout_metrics, "control": control_metrics,
        "metric_definitions": {
            "differential_energy": "0.5 * ||(Y_n-Y_(n-L)) c / (L h) - B (u_n-u_(n-L))/(L h)||_2^2; candidate c held fixed in the data derivative.",
            "instantaneous_energy": "Noisy instantaneous incompatibility need not be monotone or converge to zero. Oracle measured energy estimates the differentiation/noise floor.",
            "memory_energy": "Exponential sum of 0.5*h*||xi||^2. Before/after pairs use the same just-updated memory; each coefficient step decreases that frozen objective, up to roundoff.",
            "clean_energy": "Evaluation-only finite differences formed from simulator acceleration, never used to update the learner.",
            "coefficient_relative_rms": "sqrt(mean(((estimated_c-true_c)/true_c)^2)) across the five coefficients; diagnostic only.",
            "forward_vector_rmse": "sqrt(mean(||predicted_acceleration-true_acceleration||_2^2)); no division by the two coordinates.",
            "inverse_torque_rmse": "sqrt(mean((learned_feasible_inverse_torque-true_feasible_inverse_torque)^2)).",
            "partials_error": "Frobenius relative error in the fixed mechanical/actuator convention, evaluated on the same clean context; true F is not a learning target.",
            "control_angle_vector_rmse": "sqrt(mean(||wrap(q-q_star)||_2^2)); the per-angle metric additionally divides the squared norm by two.",
            "control_success": "Both wrapped angle-error norm and angular-speed norm remain below their thresholds at every sampled point of the final evaluation window; this is local stabilization, not swing-up.",
        },
        "artifacts": artifact_names + ["summary.json"],
    }
    write_json(output / "summary.json", summary)
    return summary


def aggregate_summaries(summaries: list[dict[str, Any]]) -> dict[str, Any]:
    aggregate: dict[str, Any] = {"status": STATUS, "seeds": [s["seed"] for s in summaries],
        "number_of_runs": len(summaries), "statistics": {},
        "claim": "Statistics over new reconstructed simulations; original table values were not used as targets."}
    def summarize(name, values):
        finite = np.array([v for v in values if v is not None and np.isfinite(v)], dtype=float)
        aggregate["statistics"][name] = {"count": len(finite),
            "mean": float(np.mean(finite)) if len(finite) else None,
            "sample_sd": float(np.std(finite, ddof=1)) if len(finite) > 1 else None,
            "minimum": float(np.min(finite)) if len(finite) else None,
            "maximum": float(np.max(finite)) if len(finite) else None}
    for key in ("final_relative_coefficient_rms", "final_information_minimum_eigenvalue",
                "final_mass_minimum_eigenvalue_all_configurations", "tail_mean_measured_incompatibility_energy_pre",
                "tail_mean_measured_oracle_energy", "tail_mean_clean_incompatibility_energy_pre"):
        summarize(f"identification.{key}", [s["identification"][key] for s in summaries])
    for name in ("learned", "frozen", "oracle"):
        for key in ("forward_acceleration_vector_rmse_rad_s2", "inverse_torque_rmse_Nm", "relative_partials_rms_on_query_states"):
            summarize(f"readouts.{name}.{key}", [s["readouts"][name].get(key) for s in summaries])
        for key in ("full_episode_angle_vector_rmse_rad", "final_window_angle_vector_rmse_rad", "final_window_maximum_speed_norm_rad_s"):
            summarize(f"control.{name}.{key}", [s["control"][name].get(key) for s in summaries])
        aggregate[f"{name}_control_success_count"] = sum(int(s["control"][name]["stabilized_in_final_window"]) for s in summaries)
    return aggregate


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=Path("results/acrobot"))
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--seed", type=int, default=None)
    group.add_argument("--seeds", nargs="+", type=int, default=None, help="Explicit list of measurement-noise seeds")
    parser.add_argument("--quick", action="store_true", help="Smoke run: 2 s identification, 1 s control, 200 queries; not the main protocol")
    parser.add_argument("--no-plots", action="store_true", help="Save traces and JSON only")
    args = parser.parse_args(argv)
    config = load_config(args.config)
    config["run_mode"] = "quick" if args.quick else "main"
    if args.quick:
        config["identification"]["duration"] = 2.0
        config["control"]["duration"] = 1.0
        config["control"]["evaluation_window"] = 0.25
        config["queries"]["count"] = 200
    seeds = args.seeds if args.seeds is not None else [10 if args.seed is None else args.seed]
    if len(set(seeds)) != len(seeds):
        parser.error("Seeds must be distinct")
    args.output.mkdir(parents=True, exist_ok=True)
    summaries = []
    for seed in seeds:
        output = args.output / f"seed_{seed}"
        summary = run_experiment(config, seed, output, make_plots=not args.no_plots)
        summaries.append(summary)
        print(json.dumps({"status": STATUS, "seed": seed, "output": str(output.resolve()),
                          "coefficient_relative_rms": summary["identification"]["final_relative_coefficient_rms"],
                          "learned_forward_vector_rmse": summary["readouts"]["learned"].get("forward_acceleration_vector_rmse_rad_s2"),
                          "learned_local_lqr_success": summary["control"]["learned"]["stabilized_in_final_window"]}), flush=True)
    write_json(args.output / "aggregate_summary.json", aggregate_summaries(summaries))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
