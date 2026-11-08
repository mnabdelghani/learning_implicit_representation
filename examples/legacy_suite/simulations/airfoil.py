"""New incompressible-flow / implicit differential-learning prototype.

A projected Fourier velocity solver uses a smooth immersed NACA0012 mask,
a fringe sponge, and an explicitly stated wake BODY FORCE. It is neither a
surface-jet CFD benchmark nor a validated aerodynamic model. Full velocity
fields are observed. Only three coefficients of a specified implicit PDE
operator family are learned; truth is reserved for independent evaluation.

Run: python simulations/airfoil.py --output results/airfoil --seed 10
Only numpy, scipy, and matplotlib are required. See docs/airfoil_implementation.txt.
"""
from __future__ import annotations

import argparse
import copy
import csv
import json
import time
from itertools import product
from pathlib import Path

import numpy as np
from scipy import fft
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]


def load_config(path: Path | None = None, quick: bool = False) -> dict:
    cfg = json.loads((path or ROOT / "configs" / "airfoil.json").read_text())
    if quick:
        cfg.update(nx=96, ny=48, dt=0.006, sample_dt=0.03,
                   spinup_time=1.5, training_time=1.8, control_time=3.6,
                   gain_change_time=1.65, target_period=3.0, diagnostic_every=5)
    if cfg["nx"] % 2 or cfg["ny"] % 2:
        raise ValueError("Use even grid dimensions for the Fourier projector.")
    if min(cfg["dt"], cfg["sample_dt"], cfg["penalty_time"]) <= 0:
        raise ValueError("Time steps and penalty time must be positive.")
    ratio = cfg["sample_dt"] / cfg["dt"]
    if abs(ratio - round(ratio)) > 1e-9:
        raise ValueError("sample_dt must be an integer multiple of dt.")
    return cfg


class FlowPlant:
    """Independent projected Navier--Stokes time integrator.

    Physical velocity arrays have shape (2, ny, nx). Every operation uses
    the same truncated, solenoidal Fourier space. A smooth volume penalty
    approximates the immersed solid. Periodic edges plus fringe forcing
    approximate a finite uniform-flow region; they are not exact far-field
    airfoil boundary conditions. No lift or drag is inferred.
    """

    def __init__(self, cfg: dict):
        self.cfg = copy.deepcopy(cfg)
        self.nx, self.ny = cfg["nx"], cfg["ny"]
        self.dx, self.dy = cfg["lx"] / self.nx, cfg["ly"] / self.ny
        self.x = np.arange(self.nx) * self.dx
        self.y = np.arange(self.ny) * self.dy
        self.X, self.Y = np.meshgrid(self.x, self.y)
        kx = 2 * np.pi * fft.rfftfreq(self.nx, d=self.dx)
        ky = 2 * np.pi * fft.fftfreq(self.ny, d=self.dy)
        self.KX, self.KY = np.meshgrid(kx, ky)
        self.k2 = self.KX ** 2 + self.KY ** 2
        self.safe_k2 = np.where(self.k2 == 0, 1.0, self.k2)
        # Strict cutoffs remove Nyquist modes and ambiguous boundary modes.
        ix = fft.rfftfreq(self.nx) * self.nx
        iy = fft.fftfreq(self.ny) * self.ny
        self.cutoff = ((np.abs(iy[:, None]) < self.ny / 3)
                       & (ix[None, :] < self.nx / 3))
        self.chi = self._airfoil_mask()
        self.sponge = self._sponge()
        self.damping = self.chi / cfg["penalty_time"] + self.sponge
        self.uniform = np.zeros((2, self.ny, self.nx))
        self.uniform[0] = 1.0
        self.fringe_source = self.uniform * self.sponge
        main = self._gaussian(cfg["actuator_x"], cfg["actuator_y"],
                              cfg["actuator_sigma_x"], cfg["actuator_sigma_y"])
        returning = self._gaussian(cfg["actuator_x"], cfg["return_force_y"],
                                   cfg["actuator_sigma_x"], cfg["actuator_sigma_y"])
        main *= 1 - self.chi
        returning *= 1 - self.chi
        # Equal/opposite integrated forcing avoids mean-momentum injection.
        self.body_force = np.zeros_like(self.uniform)
        self.body_force[1] = main - returning * (main.sum() / returning.sum())
        self.B = self.project(self.body_force)
        weight = self._gaussian(cfg["readout_x"], cfg["readout_y"],
                                cfg["readout_sigma_x"], cfg["readout_sigma_y"])
        weight *= 1 - self.chi
        self.readout_weight = weight / weight.sum()
        self.CB = self.output(self.B)
        if abs(self.CB) < 1e-8:
            raise ValueError("This body-force/readout pair lacks relative degree one.")

    def _gaussian(self, x0: float, y0: float, sx: float, sy: float) -> np.ndarray:
        return np.exp(-0.5 * (((self.X - x0) / sx) ** 2
                             + ((self.Y - y0) / sy) ** 2))

    def _airfoil_mask(self) -> np.ndarray:
        c = self.cfg
        angle = np.deg2rad(c["angle_deg"])
        xx = self.X - c["airfoil_center_x"]
        yy = self.Y - c["airfoil_center_y"]
        chord_coordinate = (np.cos(angle) * xx + np.sin(angle) * yy) / c["chord"] + 0.5
        normal_coordinate = (-np.sin(angle) * xx + np.cos(angle) * yy) / c["chord"]
        q = np.clip(chord_coordinate, 0, 1)
        # Closed-trailing-edge NACA four-digit thickness polynomial.
        half_thickness = 5 * c["thickness_ratio"] * (
            0.2969 * np.sqrt(q) - 0.1260 * q - 0.3516 * q ** 2
            + 0.2843 * q ** 3 - 0.1036 * q ** 4)
        eps = c["mask_width_cells"] * max(self.dx, self.dy) / c["chord"]
        inside_thickness = 0.5 * (1 + np.tanh((half_thickness - np.abs(normal_coordinate)) / eps))
        leading = 0.5 * (1 + np.tanh(chord_coordinate / eps))
        trailing = 0.5 * (1 + np.tanh((1 - chord_coordinate) / eps))
        return inside_thickness * leading * trailing

    def _sponge(self) -> np.ndarray:
        c = self.cfg
        sx = 0.5 * (1 - np.tanh((self.X - c["sponge_left"]) / c["sponge_width"]))
        sx += 0.5 * (1 + np.tanh((self.X - c["sponge_right"]) / c["sponge_width"]))
        edge = c["sponge_vertical_width"]
        sy = 0.5 * (1 - np.tanh((self.Y - edge) / c["sponge_width"]))
        sy += 0.5 * (1 + np.tanh((self.Y - (c["ly"] - edge)) / c["sponge_width"]))
        return c["sponge_strength"] * np.maximum(sx, sy)

    def _to_hat(self, vector: np.ndarray) -> np.ndarray:
        return fft.rfft2(vector, axes=(-2, -1), workers=1)

    def _from_hat(self, vector_hat: np.ndarray) -> np.ndarray:
        return fft.irfft2(vector_hat, s=(self.ny, self.nx), axes=(-2, -1), workers=1)

    def _project_hat(self, field_hat: np.ndarray) -> np.ndarray:
        out = field_hat.copy()
        dot = (self.KX * out[0] + self.KY * out[1]) / self.safe_k2
        out[0] -= self.KX * dot
        out[1] -= self.KY * dot
        out *= self.cutoff
        return out

    def project(self, vector: np.ndarray) -> np.ndarray:
        return self._from_hat(self._project_hat(self._to_hat(vector)))

    def laplacian(self, vector: np.ndarray) -> np.ndarray:
        return self._from_hat(-self.k2 * self._to_hat(vector))

    def vorticity(self, vector: np.ndarray) -> np.ndarray:
        vh = self._to_hat(vector)
        return self._from_hat(1j * (self.KX * vh[1] - self.KY * vh[0]))

    def divergence(self, vector: np.ndarray) -> np.ndarray:
        vh = self._to_hat(vector)
        return self._from_hat(1j * (self.KX * vh[0] + self.KY * vh[1]))

    def nonlinear(self, vector: np.ndarray) -> np.ndarray:
        omega = self.vorticity(vector)
        return self.project(np.stack((-vector[1] * omega, vector[0] * omega)))

    def nonlinear_differential(self, vector: np.ndarray, direction: np.ndarray) -> np.ndarray:
        omega = self.vorticity(vector)
        delta_omega = self.vorticity(direction)
        return self.project(np.stack((
            -direction[1] * omega - vector[1] * delta_omega,
            direction[0] * omega + vector[0] * delta_omega)))

    def known_forcing(self, vector: np.ndarray) -> np.ndarray:
        return self.project(self.fringe_source - self.damping * vector)

    def known_forcing_differential(self, direction: np.ndarray) -> np.ndarray:
        return self.project(-self.damping * direction)

    def operators(self, vector: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return self.nonlinear(vector), self.laplacian(vector), self.known_forcing(vector)

    def output(self, vector: np.ndarray) -> float:
        return float(np.sum(self.readout_weight * vector[1]))

    def rhs(self, vector: np.ndarray, command: float, actuator_gain: float) -> np.ndarray:
        """Plant-only truth; the learner/controller never calls this function."""
        vh = self._to_hat(vector)
        omega = self._from_hat(1j * (self.KX * vh[1] - self.KY * vh[0]))
        rotation = np.stack((-vector[1] * omega, vector[0] * omega))
        forcing = (-self.cfg["true_advection"] * rotation
                   + self.fringe_source - self.damping * vector
                   + actuator_gain * command * self.body_force)
        rhs_hat = self._project_hat(self._to_hat(forcing))
        rhs_hat -= self.cfg["true_viscosity"] * self.k2 * vh
        return self._from_hat(rhs_hat)

    def step(self, vector: np.ndarray, command: float, dt: float,
             actuator_gain: float) -> np.ndarray:
        """Three-stage strong-stability-preserving Runge--Kutta."""
        v1 = vector + dt * self.rhs(vector, command, actuator_gain)
        v2 = 0.75 * vector + 0.25 * (v1 + dt * self.rhs(v1, command, actuator_gain))
        return vector / 3 + (2 / 3) * (v2 + dt * self.rhs(v2, command, actuator_gain))

    def advance(self, vector: np.ndarray, command: float, duration: float,
                actuator_gain: float) -> np.ndarray:
        steps = int(round(duration / self.cfg["dt"]))
        if steps < 1:
            raise ValueError("Duration must span at least one solver step.")
        for _ in range(steps):
            vector = self.step(vector, command, duration / steps, actuator_gain)
        if not np.all(np.isfinite(vector)) or np.max(np.abs(vector)) > 20:
            raise FloatingPointError("Flow integration diverged: reduce dt or forcing.")
        return vector

    def initial_state(self, seed: int) -> np.ndarray:
        # A fixed small solenoidal disturbance breaks exact spatial symmetries.
        rng = np.random.default_rng(seed)
        noise = self.project(rng.normal(size=self.uniform.shape))
        noise /= max(np.sqrt(np.mean(noise ** 2)), 1e-12)
        return self.project(self.uniform + 0.002 * noise)

    def implicit_residual(self, state: np.ndarray, rate: np.ndarray, command: float,
                          parameters: np.ndarray) -> np.ndarray:
        n, lap, known = self.operators(state)
        ca, nu, beta = parameters
        return rate + ca * n - nu * lap - beta * self.B * command - known

    def implicit_differential_action(self, state: np.ndarray, delta_state: np.ndarray,
                                     delta_rate: np.ndarray, delta_input: float,
                                     parameters: np.ndarray) -> np.ndarray:
        ca, nu, beta = parameters
        return (delta_rate + ca * self.nonlinear_differential(state, delta_state)
                - nu * self.laplacian(delta_state) - beta * self.B * delta_input
                - self.known_forcing_differential(delta_state))


def solve_box_quadratic(hessian: np.ndarray, linear: np.ndarray,
                        lower: np.ndarray, upper: np.ndarray) -> np.ndarray:
    """Minimize .5*x.T*H*x-b.T*x in a three-variable box.

    Enumerate all 27 free/lower/upper active sets. On each face solve its
    free normal equations and retain feasible candidates. For the strictly
    convex regularized objective this includes the global minimizer.
    """
    if hessian.shape != (3, 3):
        raise ValueError("This tiny active-set solver is for exactly three parameters.")
    best, best_value = None, np.inf
    for active in product((-1, 0, 1), repeat=3):
        active = np.asarray(active)
        free = np.flatnonzero(active == 0)
        fixed = np.flatnonzero(active != 0)
        candidate = np.zeros(3)
        candidate[active == -1] = lower[active == -1]
        candidate[active == 1] = upper[active == 1]
        if free.size:
            rhs = linear[free] - hessian[np.ix_(free, fixed)] @ candidate[fixed]
            try:
                candidate[free] = np.linalg.solve(hessian[np.ix_(free, free)], rhs)
            except np.linalg.LinAlgError:
                continue
        if np.any(candidate < lower - 1e-10) or np.any(candidate > upper + 1e-10):
            continue
        # Remove only numerical roundoff at a boundary, not unconstrained violations.
        candidate = np.minimum(np.maximum(candidate, lower), upper)
        objective = 0.5 * candidate @ hessian @ candidate - linear @ candidate
        if objective < best_value:
            best_value, best = float(objective), candidate.copy()
    if best is None:
        raise FloatingPointError("No feasible candidate in the parameter box.")
    return best


class DifferentialLearner:
    """Nonzero implicit PDE residual with bounded parameter updates from increments.

    The fixed acceleration block I prevents a trivial zero residual. No
    teacher parameters or true partials enter observe(). Secant differences
    of the complete nonlinear features approximate F(z) z_dot. The numerical
    update solves the three-parameter regularized quadratic over its box
    by enumerating all active sets, then relaxes toward that minimizer.
    Only the frozen regularized objective has the corresponding decrease;
    no monotonicity is claimed for online pre-update compatibility.
    """

    def __init__(self, cfg: dict):
        self.theta = np.asarray(cfg["initial_parameters"], dtype=float)
        self.scales = np.asarray(cfg["parameter_scales"], dtype=float)
        self.lower = np.asarray(cfg["parameter_lower_bounds"], dtype=float)
        self.upper = np.asarray(cfg["parameter_upper_bounds"], dtype=float)
        self.initial_scaled = self.theta / self.scales
        self.gram = np.zeros((3, 3))
        self.cross = np.zeros(3)
        self.previous = None
        self.memory = cfg["memory_time"]
        self.ridge = cfg["ridge"]
        self.relaxation = cfg["update_relaxation"]
        self.samples = 0

    def observe(self, plant: FlowPlant, before: np.ndarray, after: np.ndarray,
                command: float, interval: float, update: bool = True) -> dict:
        middle = 0.5 * (before + after)
        acceleration = (after - before) / interval
        n, lap, known = plant.operators(middle)
        features = np.stack((-n, lap, plant.B * command))
        target = acceleration - known
        prediction = np.einsum("i,ijkl->jkl", self.theta, features, optimize=False)
        level_error = target - prediction
        result = {
            "level_residual_rms": float(np.sqrt(np.mean(level_error ** 2))),
            "differential_energy": float("nan"),
            "relative_differential_energy": float("nan"),
            "information_min_eigenvalue": float("nan"),
            "regularized_update_decrease": float("nan"),
        }
        if self.previous is not None:
            previous_features, previous_target = self.previous
            df = (features - previous_features) / interval
            dy = (target - previous_target) / interval
            residual = dy - np.einsum("i,ijkl->jkl", self.theta, df, optimize=False)
            result["differential_energy"] = float(0.5 * np.mean(residual ** 2))
            result["relative_differential_energy"] = float(np.mean(residual ** 2)
                                                          / (np.mean(dy ** 2) + 1e-12))
            design = (df * self.scales[:, None, None, None]).reshape(3, -1)
            # All updates use only the previous interval and the actual command.
            gram = design @ design.T / design.shape[1]
            cross = design @ dy.ravel() / design.shape[1]
            if update:
                retention = np.exp(-interval / self.memory)
                self.gram = retention * self.gram + (1 - retention) * gram
                self.cross = retention * self.cross + (1 - retention) * cross
                ridge = self.ridge * max(float(np.trace(self.gram) / 3), 1e-10)
                hessian = self.gram + ridge * np.eye(3)
                linear = self.cross + ridge * self.initial_scaled
                scaled = solve_box_quadratic(hessian, linear,
                                             self.lower / self.scales, self.upper / self.scales)
                old = self.theta / self.scales
                relaxed = old + self.relaxation * (scaled - old)
                objective = lambda x: 0.5 * x @ hessian @ x - linear @ x
                result["regularized_update_decrease"] = float(objective(old) - objective(relaxed))
                self.theta = self.scales * relaxed
            result["information_min_eigenvalue"] = float(np.linalg.eigvalsh(self.gram)[0])
        self.previous = (features, target)
        self.samples += 1
        return result


def readout_control(plant: FlowPlant, state: np.ndarray, theta: np.ndarray,
                    desired: float, desired_rate: float, cfg: dict) -> tuple[float, float]:
    """Regularized bounded readout solve at the current measured full state."""
    n, lap, known = plant.operators(state)
    drift = plant.output(-theta[0] * n + theta[1] * lap + known)
    gain = theta[2] * plant.CB
    target_rate = desired_rate - cfg["feedback_gain"] * (plant.output(state) - desired)
    unconstrained = gain * (target_rate - drift) / (gain * gain + cfg["control_regularization"])
    return float(np.clip(unconstrained, -cfg["input_bound"], cfg["input_bound"])), float(unconstrained)


def reference(t: float, initial_output: float, cfg: dict) -> tuple[float, float]:
    omega = 2 * np.pi / cfg["target_period"]
    settling = cfg["target_settling_time"]
    offset = initial_output * np.exp(-t / settling)
    return (offset + cfg["target_amplitude"] * np.sin(omega * t),
            -offset / settling + cfg["target_amplitude"] * omega * np.cos(omega * t))


def evaluation_direction(plant: FlowPlant, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed + 739)
    direction_hat = plant._to_hat(rng.normal(size=plant.uniform.shape))
    direction_hat *= np.exp(-0.12 * plant.k2)
    direction = plant._from_hat(plant._project_hat(direction_hat))
    return direction / max(np.sqrt(np.mean(direction ** 2)), 1e-12)


def partial_action_error(plant: FlowPlant, state: np.ndarray, theta: np.ndarray,
                         true_parameters: np.ndarray, direction: np.ndarray) -> float:
    """Evaluation ONLY. Truth never enters DifferentialLearner or readout_control."""
    zero_rate = np.zeros_like(state)
    predicted = plant.implicit_differential_action(state, direction, zero_rate, 1.0, theta)
    exact = plant.implicit_differential_action(state, direction, zero_rate, 1.0, true_parameters)
    return float(np.sqrt(np.mean((predicted - exact) ** 2))
                 / max(np.sqrt(np.mean(exact ** 2)), 1e-12))


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def clean_json(value):
    if isinstance(value, dict):
        return {k: clean_json(v) for k, v in value.items()}
    if isinstance(value, list):
        return [clean_json(v) for v in value]
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def run(output: Path, cfg: dict, seed: int = 10) -> dict:
    started = time.perf_counter()
    output.mkdir(parents=True, exist_ok=True)
    plant = FlowPlant(cfg)
    state = plant.initial_state(seed)
    learner = DifferentialLearner(cfg)
    h = cfg["sample_dt"]
    initial_gain = cfg["true_gain_initial"]
    true_before = np.array([cfg["true_advection"], cfg["true_viscosity"], initial_gain])
    direction = evaluation_direction(plant, seed)
    for _ in range(int(round(cfg["spinup_time"] / h))):
        state = plant.advance(state, 0.0, h, initial_gain)
    print("Airfoil: spin-up complete", flush=True)

    phases = np.random.default_rng(seed + 1).uniform(0, 2 * np.pi, 3)
    training_rows = []
    training_steps = int(round(cfg["training_time"] / h))
    for index in range(training_steps):
        now = index * h
        command = cfg["exploration_amplitude"] * float(np.dot(
            [0.50, 0.30, 0.20], np.sin(2 * np.pi * np.array([0.45, 1.15, 2.35]) * now + phases)))
        after = plant.advance(state, command, h, initial_gain)
        diagnostics = learner.observe(plant, state, after, command, h, update=True)
        state = after
        row = dict(time=(index + 1) * h, output=plant.output(state), input=command,
                   ca_hat=float(learner.theta[0]), nu_hat=float(learner.theta[1]),
                   beta_hat=float(learner.theta[2]), **diagnostics)
        row["partial_action_relative_error"] = partial_action_error(
            plant, state, learner.theta, true_before, direction)
        training_rows.append(row)
    common_state = state.copy()
    trained_learner = copy.deepcopy(learner)
    initial_output = plant.output(common_state)
    print("Airfoil: identification complete; learned parameters", learner.theta, flush=True)

    traces, final_fields, branch_summary = [], {}, {}
    control_steps = int(round(cfg["control_time"] / h))
    for branch in ("no_control", "frozen", "adaptive"):
        state = common_state.copy()
        branch_learner = copy.deepcopy(trained_learner)
        # Reset the secant pair across the switch of experiment segments.
        branch_learner.previous = None
        branch_rows = []
        for index in range(control_steps):
            now = index * h
            desired, desired_rate = reference(now, initial_output, cfg)
            if branch == "no_control":
                command, unconstrained = 0.0, 0.0
            else:
                command, unconstrained = readout_control(
                    plant, state, branch_learner.theta, desired, desired_rate, cfg)
            # The change schedule is seen by the plant only, never by the learner/controller.
            true_gain = (initial_gain if now < cfg["gain_change_time"]
                         else cfg["true_gain_after"])
            theta_before = branch_learner.theta.copy()
            after = plant.advance(state, command, h, true_gain)
            diagnostics = branch_learner.observe(
                plant, state, after, command, h, update=(branch == "adaptive"))
            state = after
            next_time = (index + 1) * h
            next_desired, _ = reference(next_time, initial_output, cfg)
            row = dict(branch=branch, time=next_time, output=plant.output(state),
                       target=float(next_desired), error=plant.output(state) - next_desired,
                       input=command, unconstrained_input=unconstrained,
                       ca_hat=float(theta_before[0]), nu_hat=float(theta_before[1]),
                       beta_hat=float(theta_before[2]), true_gain=float(true_gain),
                       input_partial_relative_error=float(abs(theta_before[2] - true_gain) / abs(true_gain)),
                       **diagnostics)
            if index % cfg["diagnostic_every"] == 0 or index == control_steps - 1:
                truth = np.array([cfg["true_advection"], cfg["true_viscosity"], true_gain])
                row["partial_action_relative_error"] = partial_action_error(
                    plant, state, theta_before, truth, direction)
                row["divergence_rms"] = float(np.sqrt(np.mean(plant.divergence(state) ** 2)))
            else:
                row["partial_action_relative_error"] = float("nan")
                row["divergence_rms"] = float("nan")
            row["max_speed"] = float(np.sqrt(np.sum(state * state, axis=0)).max())
            branch_rows.append(row)
        traces.extend(branch_rows)
        final_fields[branch] = state.copy()
        errors = np.array([r["error"] for r in branch_rows])
        commands = np.array([r["input"] for r in branch_rows])
        late = np.array([r["time"] > cfg["gain_change_time"] + 0.7 for r in branch_rows])
        valid_energy = [r["differential_energy"] for r in branch_rows[-max(10, control_steps // 5):]
                        if np.isfinite(r["differential_energy"])]
        branch_summary[branch] = {
            "rmse": float(np.sqrt(np.mean(errors ** 2))),
            "late_rmse": float(np.sqrt(np.mean(errors[late] ** 2))) if late.any() else None,
            "max_absolute_error": float(np.abs(errors).max()),
            "input_rms": float(np.sqrt(np.mean(commands ** 2))),
            "max_absolute_input": float(np.abs(commands).max()),
            "saturation_fraction": float(np.mean(np.abs(commands) >= cfg["input_bound"] - 1e-10)),
            "final_parameters": branch_learner.theta.tolist(),
            "final_input_partial_relative_error": float(abs(branch_learner.theta[2] - true_gain) / abs(true_gain)),
            "late_median_pre_update_energy": float(np.median(valid_energy)),
            "final_partial_action_relative_error": float(branch_rows[-1]["partial_action_relative_error"]),
            "final_divergence_rms": float(np.sqrt(np.mean(plant.divergence(state) ** 2))),
            "final_immersed_body_speed_rms": float(np.sqrt(
                np.sum(plant.chi * np.sum(state ** 2, axis=0)) / plant.chi.sum())),
            "maximum_speed": float(max(r["max_speed"] for r in branch_rows)),
        }
        print("Airfoil:", branch, "RMSE", branch_summary[branch]["rmse"], flush=True)

    write_csv(output / "training_trace.csv", training_rows)
    write_csv(output / "trace.csv", traces)
    np.savez_compressed(output / "fields.npz", x=plant.x, y=plant.y, mask=plant.chi,
                        sponge=plant.sponge, body_force=plant.body_force,
                        readout_weight=plant.readout_weight, common_state=common_state,
                        **{f"final_{key}": value for key, value in final_fields.items()})
    make_figures(output, plant, traces, training_rows, final_fields, cfg)
    summary = {
        "status": "new_unvalidated_numerical_prototype",
        "description": "Projected incompressible Fourier flow with immersed NACA0012 and wake body-force actuation",
        "not_a_surface_jet_or_lift_benchmark": True,
        "no_grid_time_penalty_domain_convergence_validation": True,
        "seed": seed, "config": cfg,
        "nominal_reynolds_number": float(cfg["chord"] / cfg["true_viscosity"]),
        "learning_inputs": "Full observed velocity fields, sampled increments, applied input, known geometry/forcing operators; no true coefficient/partial targets",
        "implicit_normalization": "The coefficient of the measured field acceleration is fixed to the identity; learned parameters obey declared bounds",
        "loss_note": "Pre-update secant compatibility approximates F(z) zdot in constant-parameter regimes; no zero-energy claim across the unknown gain jump",
        "update_note": "Exact three-variable box-quadratic active-set solve, followed by relaxation; only the frozen regularized objective decreases, not necessarily online pre-update compatibility",
        "operator_error_note": "One fixed smooth divergence-free evaluation direction with delta_rate=0 and delta_input=1; includes known geometry blocks; not a complete operator norm",
        "readout": "Normalized Gaussian-weighted vertical wake velocity",
        "reference": "y_initial*exp(-t/target_settling_time) + target_amplitude*sin(2*pi*t/target_period)",
        "output_input_coupling_CB": plant.CB,
        "trained_parameters": trained_learner.theta.tolist(),
        "true_parameters_during_training_evaluation_only": true_before.tolist(),
        "training_final_partial_action_relative_error": training_rows[-1]["partial_action_relative_error"],
        "branches": branch_summary,
        "elapsed_seconds": time.perf_counter() - started,
        "files": ["trace.csv", "training_trace.csv", "fields.npz", "airfoil.png", "airfoil.pdf", "airfoil_flow.png", "parameters.png"],
    }
    (output / "summary.json").write_text(json.dumps(clean_json(summary), indent=2, allow_nan=False) + "\n")
    return summary


def make_figures(output, plant, traces, training, final_fields, cfg):
    colors = {"no_control": "#888888", "frozen": "#c56a13", "adaptive": "#146da3"}
    labels = {"no_control": "No control", "frozen": "Frozen learner", "adaptive": "Online learner"}
    fig, axs = plt.subplots(2, 2, figsize=(11.6, 7.0), layout="constrained")
    for branch in colors:
        rows = [r for r in traces if r["branch"] == branch]
        t = np.array([r["time"] for r in rows])
        axs[0, 0].plot(t, [r["output"] for r in rows], color=colors[branch], label=labels[branch])
        axs[0, 1].plot(t, [r["input"] for r in rows], color=colors[branch], label=labels[branch])
        if branch != "no_control":
            axs[1, 0].semilogy(t, np.maximum([r["differential_energy"] for r in rows], 1e-16),
                               color=colors[branch], label=labels[branch])
            selected = [r for r in rows if np.isfinite(r["partial_action_relative_error"])]
            axs[1, 1].semilogy([r["time"] for r in selected],
                               np.maximum([r["partial_action_relative_error"] for r in selected], 1e-12),
                               color=colors[branch], label=labels[branch] + " F probe")
            axs[1, 1].semilogy(t, np.maximum([r["input_partial_relative_error"] for r in rows], 1e-12),
                               color=colors[branch], linestyle="--", label=labels[branch] + " input block")
    rows = [r for r in traces if r["branch"] == "adaptive"]
    axs[0, 0].plot([r["time"] for r in rows], [r["target"] for r in rows],
                    "k--", linewidth=1.3, label="Desired wake velocity")
    axs[0, 0].set(title="(a) Selected wake output", ylabel="Weighted vertical velocity")
    axs[0, 1].set(title="(b) Actual applied body-force command", ylabel="Input amplitude")
    axs[1, 0].set(title="(c) Pre-update secant incompatibility", ylabel="Mean squared residual / 2")
    axs[1, 1].set(title="(d) Partial errors, evaluation only", ylabel="Relative error")
    for ax in axs.flat:
        ax.axvline(cfg["gain_change_time"], color="#8d366c", linestyle=":", linewidth=1.2)
        ax.set_xlabel("Control time (chord / free-stream speed)")
        ax.grid(alpha=0.22)
        ax.legend(frameon=False, fontsize=8)
    fig.suptitle("New flow prototype — immersed airfoil, wake body force; no aerodynamic validation", fontsize=11)
    fig.savefig(output / "airfoil.png", dpi=180)
    fig.savefig(output / "airfoil.pdf")
    plt.close(fig)

    state = final_fields["adaptive"]
    omega = plant.vorticity(state)
    lim = max(float(np.quantile(np.abs(omega), 0.98)), 0.1)
    fig, ax = plt.subplots(figsize=(10.5, 4.1), layout="constrained")
    image = ax.imshow(omega, extent=(0, cfg["lx"], 0, cfg["ly"]), origin="lower",
                      cmap="RdBu_r", vmin=-lim, vmax=lim, interpolation="nearest", aspect="equal")
    ax.contour(plant.X, plant.Y, plant.chi, levels=[0.5], colors="black", linewidths=1.4)
    ax.contour(plant.X, plant.Y, plant.readout_weight,
               levels=[0.5 * plant.readout_weight.max()], colors="#3b145e", linestyles="--")
    ax.set(xlabel="x / chord", ylabel="y / chord", title="Final online-control vorticity; dashed contour marks the wake readout")
    fig.colorbar(image, ax=ax, label="Nondimensional vorticity")
    fig.savefig(output / "airfoil_flow.png", dpi=180)
    plt.close(fig)

    fig, axs = plt.subplots(1, 3, figsize=(11, 3.3), layout="constrained")
    for axis, key, truth_key, title in zip(axs, ["ca_hat", "nu_hat", "beta_hat"],
                                          ["true_advection", "true_viscosity", "true_gain_initial"],
                                          ["Advection multiplier", "Viscosity", "Actuator efficacy"]):
        axis.plot([r["time"] - cfg["training_time"] for r in training], [r[key] for r in training],
                   color="#146da3", label="Learning")
        axis.plot([r["time"] for r in rows], [r[key] for r in rows], color="#146da3")
        if key == "beta_hat":
            axis.plot([-cfg["training_time"], cfg["gain_change_time"], cfg["gain_change_time"], cfg["control_time"]],
                      [cfg["true_gain_initial"], cfg["true_gain_initial"], cfg["true_gain_after"], cfg["true_gain_after"]],
                      "k--", linewidth=1, label="Truth, evaluation only")
        else:
            axis.axhline(cfg[truth_key], color="black", linestyle="--", linewidth=1, label="Truth, evaluation only")
        axis.axvline(0, color="#999999", linestyle=":")
        axis.axvline(cfg["gain_change_time"], color="#8d366c", linestyle=":")
        axis.set(title=title, xlabel="Time; control begins at zero")
        axis.grid(alpha=0.2)
    axs[0].legend(frameon=False, fontsize=8)
    fig.savefig(output / "parameters.png", dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "results" / "airfoil")
    parser.add_argument("--seed", type=int, default=10)
    parser.add_argument("--quick", action="store_true", help="Use a smaller grid and shorter run")
    parser.add_argument("--config", type=Path, help="JSON configuration; defaults to configs/airfoil.json")
    parser.add_argument("--nx", type=int, help="Override grid columns")
    parser.add_argument("--ny", type=int, help="Override grid rows")
    parser.add_argument("--control-time", type=float, help="Override the control horizon")
    args = parser.parse_args()
    cfg = load_config(args.config, args.quick)
    for key, value in {"nx": args.nx, "ny": args.ny, "control_time": args.control_time}.items():
        if value is not None:
            cfg[key] = value
    result = run(args.output.resolve(), cfg, args.seed)
    print(json.dumps(clean_json(result), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
