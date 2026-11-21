"""Independent raw-record checks for the normalized Fourier experiment.

This module deliberately does not import the simulation or its integrator.
All recorded updates are checked using nonlinear projected-flow RK4, with
selected updates also checked by a tighter coupled (q, R) integration.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import platform
import shutil
import sys

import numpy as np
from scipy.integrate import solve_ivp
import scipy

METHODS = ("normalized_flow", "bottom_eigenvector", "frozen_prior")


def projected_steps(q, memories, samples, dt, memory, gain, subdivisions=32):
    """Independent batched RK4 of qdot, using exact held-sample covariance.

    No normalization is performed inside or after this check. Thus both the
    direction and the preservation of the sphere are tested against the runner.
    """
    q = np.asarray(q, float).copy()
    memories = np.asarray(memories, float)
    samples = np.asarray(samples, float)

    def rhs(s, candidate):
        decay = np.exp(-s / memory)
        Rq = decay * np.einsum("nij,nj->ni", memories, candidate)
        Rq += memory * (-np.expm1(-s / memory)) * samples * np.sum(samples * candidate, axis=1)[:, None]
        return -gain * (Rq - candidate * np.sum(candidate * Rq, axis=1)[:, None])

    h = dt / subdivisions
    for k in range(subdivisions):
        s = k * h
        k1 = rhs(s, q)
        k2 = rhs(s + h / 2, q + h * k1 / 2)
        k3 = rhs(s + h / 2, q + h * k2 / 2)
        k4 = rhs(s + h, q + h * k3)
        q += h * (k1 + 2 * k2 + 2 * k3 + k4) / 6
    return q


def coupled_step(q, R, w, dt, memory, gain):
    """Reference integration of the actual coupled nonlinear equations."""
    dimension = len(q)
    outer = np.outer(w, w)

    def rhs(_s, state):
        candidate, covariance = state[:dimension], state[dimension:].reshape(dimension, dimension)
        Rq = covariance @ candidate
        return np.r_[-gain * (Rq - (candidate @ Rq) * candidate),
                     (outer - covariance / memory).ravel()]

    result = solve_ivp(rhs, (0., dt), np.r_[q, R.ravel()], method="DOP853",
                       rtol=2e-13, atol=2e-15, max_step=dt / 8)
    if not result.success:
        raise RuntimeError(result.message)
    return result.y[:dimension, -1], result.y[dimension:, -1].reshape(dimension, dimension)


def projector_distance(a, b):
    """Stable Frobenius row-projector distance; works on vectorized rows."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    a = a / np.linalg.norm(a, axis=-1, keepdims=True)
    b = b / np.linalg.norm(b, axis=-1, keepdims=True)
    return np.linalg.norm(a[..., :, None] * a[..., None, :] -
                          b[..., :, None] * b[..., None, :], axis=(-2, -1))


class Audit:
    def __init__(self):
        self.checks = 0
        self.scalar_comparisons = 0
        self.maxima = {}

    def close(self, a, b, key, atol=2e-10, rtol=2e-8):
        a, b = np.asarray(a), np.asarray(b)
        np.testing.assert_allclose(a, b, atol=atol, rtol=rtol, equal_nan=True,
                                   err_msg=key)
        self.checks += 1
        self.scalar_comparisons += max(a.size, b.size)
        finite = np.isfinite(a) & np.isfinite(b)
        if np.any(finite):
            value = float(np.max(np.abs((a - b)[finite])))
            self.maxima[key] = max(self.maxima.get(key, 0.), value)

    def require(self, condition, message):
        if not condition:
            raise AssertionError(message)
        self.checks += 1


def load(path):
    return np.atleast_1d(np.genfromtxt(path, delimiter=",", names=True))


def physical_partials(coefficients, h, hp, state, rate):
    coefficients = np.broadcast_to(coefficients, (len(state), 7))
    return np.column_stack((np.sum(coefficients[:, 3:6] * h, axis=1),
                            np.sum(coefficients[:, :3] * h, axis=1), -coefficients[:, 6],
                            np.sum(coefficients[:, :3] * hp, axis=1) * rate +
                            np.sum(coefficients[:, 3:6] * hp, axis=1) * state))


def audit_pair(root, mode, seed, cfg, summaries, checker, selected_count=12):
    folder = root / mode / f"seed_{seed}"
    stream = load(folder / "observations.csv")
    with np.load(folder / "state.npz") as saved:
        arrays = {name: saved[name] for name in saved.files}
    count = round(cfg["duration"] / cfg["dt"]) + 1
    lag = round(cfg["difference_window"] / cfg["dt"])
    checker.require(len(stream) == count, "Missing observation rows")
    t = np.arange(count) * cfg["dt"]
    checker.close(stream["time"], t, "sample_clock", atol=1e-14, rtol=0.)
    checker.close(stream["phase"], t * cfg["phase_rate"], "phase_clock", atol=1e-13, rtol=0.)
    S = np.asarray(cfg["scales"])
    h = np.column_stack((np.ones(count), np.cos(stream["phase"]), np.sin(stream["phase"])))
    hp = np.column_stack((np.zeros(count), -np.sin(stream["phase"]), np.cos(stream["phase"])))
    observed = np.column_stack((h * stream["observed_rate"][:, None], h * stream["state"][:, None], -stream["control"]))
    clean = np.column_stack((h * stream["evaluator_actual_rate"][:, None], h * stream["state"][:, None], -stream["control"]))
    w, wc = np.full((count, 7), np.nan), np.full((count, 7), np.nan)
    w[lag:] = (observed[lag:] - observed[:-lag]) * S / cfg["difference_window"]
    wc[lag:] = (clean[lag:] - clean[:-lag]) * S / cfg["difference_window"]
    checker.close(arrays["w"], w, "complete_secants", atol=2e-12, rtol=2e-12)
    q0 = np.asarray(cfg["initial_homogeneous"]) / S
    q0 /= np.linalg.norm(q0)
    qpre, qpost = arrays["q_pre"], arrays["q_post"]
    checker.require(qpre.shape == qpost.shape == (count, 3, 7), "Wrong coefficient-array shape")
    checker.close(qpre[:lag + 1], np.broadcast_to(q0, (lag + 1, 3, 7)), "initial_coefficients", atol=0., rtol=0.)
    checker.close(qpre[1:], qpost[:-1], "causal_parameter_state", atol=0., rtol=0.)
    checker.close(qpost[-1], qpre[-1], "final_no_parameter_update", atol=0., rtol=0.)
    checker.close(qpost[:, 2], np.broadcast_to(q0, (count, 7)), "frozen_prior", atol=0., rtol=0.)
    checker.close(np.linalg.norm(qpre, axis=2), 1., "unit_norm", atol=2e-14, rtol=0.)
    checker.close(np.linalg.norm(qpost, axis=2), 1., "unit_norm", atol=2e-14, rtol=0.)
    R, clean_R = np.zeros((7, 7)), np.zeros((7, 7))
    expected_pre, expected_post, clean_pre = [], [], []
    beta = np.exp(-cfg["dt"] / cfg["memory"])
    weight = cfg["memory"] * (-np.expm1(-cfg["dt"] / cfg["memory"]))
    for n in range(count):
        expected_pre.append(R.copy())
        clean_pre.append(clean_R.copy())
        if lag <= n < count - 1:
            R = beta * R + weight * np.outer(w[n], w[n])
            clean_R = beta * clean_R + weight * np.outer(wc[n], wc[n])
        expected_post.append(R.copy())
    expected_pre, expected_post, clean_pre = np.asarray(expected_pre), np.asarray(expected_post), np.asarray(clean_pre)
    checker.close(arrays["R_pre"], expected_pre, "memory_before", atol=1e-12, rtol=1e-12)
    checker.close(arrays["R_post"], expected_post, "memory_after", atol=1e-12, rtol=1e-12)
    checker.close(arrays["R_post"][-1], arrays["R_pre"][-1], "final_no_memory_update", atol=0., rtol=0.)
    checker.require(np.all(arrays["nfev"][:lag] == 0) and arrays["nfev"][-1] == 0,
                    "Unexpected integration before features or at final sample")
    checker.require(np.all(arrays["nfev"][lag:-1] > 0), "Missing flow integration")
    active = np.arange(lag, count - 1)
    flow_checked = projected_steps(qpre[active, 0], expected_pre[active], w[active],
                                   cfg["dt"], cfg["memory"], cfg["gain"])
    checker.close(qpost[active, 0], flow_checked, "all_projected_flow_updates", atol=3e-9, rtol=3e-9)
    chosen = np.unique(np.linspace(lag, count - 2, selected_count, dtype=int))
    for n in chosen:
        reference_q, reference_R = coupled_step(qpre[n, 0], expected_pre[n], w[n],
                                               cfg["dt"], cfg["memory"], cfg["gain"])
        checker.close(qpost[n, 0], reference_q, "selected_coupled_flow_updates", atol=3e-9, rtol=3e-9)
        checker.close(expected_post[n], reference_R, "selected_coupled_memory", atol=2e-12, rtol=2e-12)
    values, vectors = np.linalg.eigh(expected_post[active])
    tolerance = cfg["eigen_tie_relative_tolerance"] * np.maximum(1., np.abs(values[:, -1]))
    tied = values <= values[:, :1] + tolerance[:, None]
    previous = qpre[active, 1]
    weights = np.einsum("nji,nj->ni", vectors, previous) * tied
    exact = np.einsum("nij,nj->ni", vectors, weights)
    singular_projection = np.linalg.norm(exact, axis=1) < 1e-12
    if np.any(singular_projection):
        U = vectors[singular_projection]
        mask = tied[singular_projection]
        diagonal = np.sum(U * U * mask[:, None, :], axis=2)
        j = np.argmax(diagonal, axis=1)
        coordinates = U[np.arange(len(U)), j, :] * mask
        exact[singular_projection] = np.einsum("nij,nj->ni", U, coordinates)
    exact /= np.linalg.norm(exact, axis=1)[:, None]
    exact[np.sum(exact * previous, axis=1) < 0.] *= -1.
    checker.close(qpost[active, 1], exact, "all_eigenvector_updates", atol=2e-8, rtol=2e-8)
    eigen_rayleigh = np.einsum("ni,nij,nj->n", qpost[active, 1], expected_post[active], qpost[active, 1])
    checker.require(np.all(eigen_rayleigh - values[:, 0] <= 2 * tolerance + 1e-12),
                    "Bottom estimator does not minimize the matched unit objective")
    eigenvalues, eigenvectors = np.linalg.eigh(expected_pre)
    truth = np.r_[cfg["true_coefficients"], 1.]
    target = truth / S
    target /= np.linalg.norm(target)
    context = np.asarray(cfg["context_scales"])
    F = physical_partials(truth, h, hp, stream["state"], stream["evaluator_actual_rate"])
    desired = .45 * np.sin(.7 * t) + .2 * np.sin(1.3 * t)
    desired_rate = .315 * np.cos(.7 * t) + .26 * np.cos(1.3 * t)
    checker.close(stream["desired"], desired, "reference", atol=2e-15, rtol=2e-15)
    true_rate = (stream["control"] - F[:, 0] * stream["state"]) / F[:, 1]
    checker.close(stream["evaluator_actual_rate"], true_rate, "plant_rate", atol=1e-12, rtol=1e-12)
    noise = np.random.default_rng(seed).normal(0., cfg["rate_noise"], count)
    checker.close(stream["observed_rate"] - true_rate, noise, "sensor_noise", atol=2e-12, rtol=2e-12)
    late = (t >= cfg["evaluation_window"][0]) & (t < cfg["evaluation_window"][1])
    expected_controller_guard = np.zeros(count, dtype=int)
    if mode == "own_trajectory":
        physical = qpre[:, 0] * S
        expected_controller_guard = (np.abs(physical[:, 6]) < cfg["input_guard"]).astype(int)
        c = physical[:, :6] / physical[:, 6, None]
        c[expected_controller_guard.astype(bool)] = np.asarray(cfg["initial_homogeneous"][:6]) / cfg["initial_homogeneous"][6]
        expected_u = np.sum(c[:, :3] * h, axis=1) * (desired_rate - cfg["feedback_gain"] * (stream["state"] - desired)) + np.sum(c[:, 3:] * h, axis=1) * stream["state"]
        checker.close(stream["control"], expected_u, "causal_control", atol=1e-12, rtol=1e-12)

        def rhs(x, time, u):
            phase = cfg["phase_rate"] * time
            local_h = np.column_stack((np.ones_like(time), np.cos(phase), np.sin(phase)))
            rho, kappa = local_h @ truth[:3], local_h @ truth[3:6]
            return (u - kappa * x) / rho

        x, u, time, dt = stream["state"][:-1], stream["control"][:-1], t[:-1], cfg["dt"]
        k1 = rhs(x, time, u)
        k2 = rhs(x + dt * k1 / 2, time + dt / 2, u)
        k3 = rhs(x + dt * k2 / 2, time + dt / 2, u)
        k4 = rhs(x + dt * k3, time + dt, u)
        checker.close(stream["state"][1:], x + dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6,
                      "plant_rk4", atol=2e-13, rtol=2e-13)
    results = []
    for j, method in enumerate(METHODS):
        trace = load(folder / method / "trace.csv")
        checker.require(len(trace) == count, "Missing estimator trace rows")
        q = qpre[:, j]
        checker.close(np.column_stack([trace[f"q{k}"] for k in range(7)]), q, "csv_npz_parameters", atol=0., rtol=0.)
        for column in ("time", "state", "control", "desired"):
            checker.close(trace[column], stream[column], "common_observations", atol=0., rtol=0.)
        checker.close(trace["error"], stream["state"] - desired, "tracking_error", atol=1e-15, rtol=1e-15)
        physical = q * S
        estimated = physical_partials(physical, h, hp, stream["state"], stream["evaluator_actual_rate"])
        guard = np.abs(physical[:, 6]) < cfg["input_guard"]
        aligned = np.full((count, 4), np.nan)
        aligned[~guard] = estimated[~guard] / physical[~guard, 6, None]
        checker.close(trace["readout_guarded"], guard, "readout_guard", atol=0., rtol=0.)
        checker.close(trace["controller_guarded"], expected_controller_guard, "controller_guard", atol=0., rtol=0.)
        checker.close(np.column_stack([trace[f"F_true_{k}"] for k in range(4)]), F, "true_complete_jacobian")
        checker.close(np.column_stack([trace[f"F_est_{k}"] for k in range(4)]), estimated, "learned_complete_jacobian")
        checker.close(np.column_stack([trace[f"F_aligned_{k}"] for k in range(4)]), aligned, "gauge_aligned_jacobian")
        derivative_error = projector_distance(estimated * context, F * context)
        coefficient_error = projector_distance(q, target)
        aligned_error = np.linalg.norm((aligned - F) * context, axis=1) / np.linalg.norm(F * context, axis=1)
        phase_error = aligned[:, 3] - F[:, 3]
        checker.close(trace["derivative_projector_error"], derivative_error, "derivative_projector_error")
        checker.close(trace["coefficient_projector_error"], coefficient_error, "coefficient_projector_error")
        checker.close(trace["aligned_F_error"], aligned_error, "aligned_F_error")
        checker.close(trace["phase_partial_error"], phase_error, "phase_partial_error")
        checker.close(trace["input_coefficient"], physical[:, 6], "input_coefficient", atol=0., rtol=0.)
        checker.close(trace["row_norm"], np.linalg.norm(estimated, axis=1), "row_norm")
        checker.close(trace["coefficient_norm"], np.linalg.norm(q, axis=1), "reported_norm", atol=5e-16, rtol=0.)
        energy = np.sum(q * w, axis=1) ** 2 / 2
        clean_energy = np.sum(q * wc, axis=1) ** 2 / 2
        true_energy = (w @ target) ** 2 / 2
        checker.close(trace["preupdate_energy"], energy, "preupdate_energy")
        checker.close(trace["clean_energy"], clean_energy, "clean_energy")
        checker.close(trace["true_row_energy"], true_energy, "true_row_energy")
        Rq = np.einsum("nij,nj->ni", expected_pre, q)
        rayleigh = np.sum(q * Rq, axis=1)
        gradient = Rq - rayleigh[:, None] * q
        checker.close(trace["memory_energy"], rayleigh / 2, "memory_objective")
        checker.close(trace["normalized_gradient_norm"], np.linalg.norm(gradient, axis=1), "normalized_gradient")
        checker.close(trace["flow_dissipation"], -cfg["gain"] * np.sum(gradient ** 2, axis=1), "flow_dissipation")
        gap = eigenvalues[:, 1] - eigenvalues[:, 0]
        checker.close(trace["memory_gap"], gap, "memory_gap")
        checker.close(trace["memory_lambda_min"], eigenvalues[:, 0], "memory_minimum")
        checker.close(trace["memory_lambda_max"], eigenvalues[:, -1], "memory_maximum")
        distance = projector_distance(q, eigenvectors[:, :, 0])
        ambiguous = gap <= cfg["eigen_tie_relative_tolerance"] * np.maximum(1., np.abs(eigenvalues[:, -1]))
        distance[ambiguous] = np.nan
        checker.close(trace["coefficient_to_memory_error"], distance, "coefficient_to_memory")
        metrics = {name + "_mean": float(np.mean(value[late])) for name, value in dict(
            derivative_projector_error=derivative_error, coefficient_projector_error=coefficient_error,
            aligned_F_error=aligned_error, preupdate_energy=energy, clean_energy=clean_energy,
            true_row_energy=true_energy, coefficient_to_memory_error=distance, memory_gap=gap).items()}
        metrics.update(phase_partial_rmse=float(np.sqrt(np.mean(phase_error[late] ** 2))),
                       min_abs_input_coefficient=float(np.min(np.abs(physical[:, 6]))),
                       readout_guard_count=int(np.count_nonzero(guard)),
                       max_norm_error=float(np.max(np.abs(np.linalg.norm(q, axis=1) - 1.))),
                       min_row_norm=float(np.min(np.linalg.norm(estimated, axis=1))), final_q=q[-1].tolist())
        if mode == "own_trajectory" and method == "normalized_flow":
            metrics.update(tracking_rmse=float(np.sqrt(np.mean((stream["state"][late] - desired[late]) ** 2))),
                           control_rms=float(np.sqrt(np.mean(stream["control"][late] ** 2))),
                           controller_guard_count=int(np.sum(expected_controller_guard)))
        summary = summaries[(mode, seed, method)]
        checker.require(set(metrics) == set(summary["metrics"]), "Summary metric set differs")
        for name, value in metrics.items():
            checker.close(value, summary["metrics"][name], "summary_metrics")
        disk = json.loads((folder / method / "summary.json").read_text())
        checker.require(disk == summary, "Per-run and aggregate summaries differ")
        results.append(dict(mode=mode, seed=seed, method=method, rows=count, metrics=metrics))
    # R(s) is on the segment joining its two endpoints during each held sample.
    # For unit q and PSD R, ||qdot|| <= gain*(lambda_max-lambda_min)/2.
    # This gives a conservative coefficient guard bound over the entire interval.
    spectral_upper = np.maximum(eigenvalues[active, -1], values[:, -1])
    guard_bound = np.abs(S[6]) * (np.abs(qpre[active, 0, 6]) - cfg["gain"] * cfg["dt"] * spectral_upper / 2)
    clean_values = np.linalg.eigvalsh(clean_pre)
    true_clean_residual = wc[lag:] @ target
    checker.close(true_clean_residual, 0., "clean_true_secant_annihilation", atol=1e-12, rtol=0.)
    checker.close(np.einsum("nij,j->ni", clean_pre, target), 0., "clean_memory_true_nullvector", atol=2e-12, rtol=0.)
    clean_rank_tolerance = 1e-10 * np.maximum(1., np.abs(clean_values[:, -1]))
    diagnostics = dict(mode=mode, seed=seed,
                       all_seven_normalized_coefficients_changed=bool(np.all(np.ptp(qpre[:, 0], axis=0) > 1e-12)),
                       normalized_coefficient_ranges=np.column_stack((np.min(qpre[:, 0], axis=0), np.max(qpre[:, 0], axis=0))).tolist(),
                       continuous_input_lower_bound=float(np.min(guard_bound)),
                       continuous_input_guard_margin=float(np.min(guard_bound) - cfg["input_guard"]),
                       intervals_without_positive_guard_margin=int(np.count_nonzero(guard_bound <= cfg["input_guard"])),
                       clean_true_secant_max_abs_residual=float(np.max(np.abs(true_clean_residual))),
                       clean_memory_final_eigenvalues=clean_values[-1].tolist(),
                       minimum_late_clean_eigengap=float(np.min(clean_values[late, 1] - clean_values[late, 0])),
                       minimum_late_clean_rank=int(np.min(np.sum(clean_values[late] > clean_rank_tolerance[late, None], axis=1))),
                       maximum_late_clean_rank=int(np.max(np.sum(clean_values[late] > clean_rank_tolerance[late, None], axis=1))),
                       clean_rank_relative_tolerance=1e-10,
                       limits="Guard bound uses floating-point endpoint spectra and the exact held-sample equations; it is not an interval-arithmetic certificate. Clean covariance rank/gap are executed-trajectory diagnostics, not a uniform excitation theorem.")
    print(f"{mode} seed {seed}: {len(active)} flow transitions and all three estimator traces audited", flush=True)
    return results, len(active), len(chosen), diagnostics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path,
                        help="New audit directory, separate from immutable run records")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Use a new audit output directory")
    cfg = json.loads((args.input / "config.json").read_text())
    checker = Audit()
    summary = json.loads((args.input / "summary.json").read_text())
    checker.require(not summary["failures"], "Experiment contains failures; preserve and inspect them")
    summaries = {(row["mode"], row["seed"], row["method"]): row for row in summary["runs"]}
    expected = {(mode, seed, method) for mode in ("replay", "own_trajectory") for seed in cfg["seeds"] for method in METHODS}
    checker.require(set(summaries) == expected and len(summary["runs"]) == len(expected), "Missing or duplicate planned runs")
    manifest = json.loads((args.input / "file_manifest.json").read_text())
    for name, digest in manifest.items():
        checker.require(hashlib.sha256((args.input / name).read_bytes()).hexdigest() == digest, "Run-file hash mismatch: " + name)
    environment = json.loads((args.input / "environment.json").read_text())
    for name, digest in environment["source_sha256"].items():
        checker.require(hashlib.sha256((args.input / "source_snapshot" / name).read_bytes()).hexdigest() == digest,
                        "Source snapshot hash mismatch: " + name)
    source_replays_checked = []
    unavailable_source_paths = []
    for seed, metadata in environment["replay_inputs"].items():
        local_source = args.input / "replay" / f"seed_{seed}" / "source_observations.csv"
        checker.require(hashlib.sha256(local_source.read_bytes()).hexdigest() == metadata["sha256"],
                        "Archived replay source hash mismatch")
        original = load(local_source)
        recorded = load(args.input / "replay" / f"seed_{seed}" / "observations.csv")
        checker.require(set(original.dtype.names) == set(recorded.dtype.names), "Replay columns changed")
        for name in original.dtype.names:
            checker.close(original[name], recorded[name], "unchanged_replay_observations", atol=0., rtol=0.)
        source_replays_checked.append(int(seed))
        source = Path(metadata["path"])
        if source.exists():
            checker.require(hashlib.sha256(source.read_bytes()).hexdigest() == metadata["sha256"], "Replay source changed")
        else:
            unavailable_source_paths.append(str(source))
    results, diagnostics, updates, selected = [], [], 0, 0
    for mode in ("replay", "own_trajectory"):
        for seed in cfg["seeds"]:
            run_results, run_updates, run_selected, run_diagnostics = audit_pair(args.input, mode, seed, cfg, summaries, checker)
            results.extend(run_results)
            diagnostics.append(run_diagnostics)
            updates += run_updates
            selected += run_selected
    args.output.mkdir(parents=True)
    shutil.copyfile(__file__, args.output / Path(__file__).name)
    report = dict(status="passed", runs=len(results), rows=sum(row["rows"] for row in results),
                  independent_streams=len(cfg["seeds"]) * 2, flow_transitions=updates,
                  selected_coupled_transitions=selected, checks=checker.checks,
                  scalar_comparisons=checker.scalar_comparisons, maxima=checker.maxima,
                  unchanged_source_replay_seeds=source_replays_checked,
                  unavailable_original_source_paths=unavailable_source_paths,
                  trajectory_conditions=diagnostics,
                  minimum_continuous_input_lower_bound=min(row["continuous_input_lower_bound"] for row in diagnostics),
                  minimum_continuous_input_guard_margin=min(row["continuous_input_guard_margin"] for row in diagnostics),
                  minimum_late_clean_eigengap=min(row["minimum_late_clean_eigengap"] for row in diagnostics),
                  minimum_late_clean_rank=min(row["minimum_late_clean_rank"] for row in diagnostics),
                  method="Independent all-step nonlinear projected-flow RK4 (32 substeps), selected coupled q/R DOP853, every covariance, secant, eigensolver update, preupdate derivative/energy, summary, own-loop command/noise/RK4 and immutable file hashes; no experiment imports",
                  limits="Numerical and information-flow checks, not a theorem proof, physical validation, or evidence that other application mechanisms match this normalized law.",
                  environment=dict(python=sys.version, numpy=np.__version__, scipy=scipy.__version__,
                                   platform=platform.platform(), command=sys.argv,
                                   auditor_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()),
                  runs_data=results)
    (args.output / "audit.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({key: value for key, value in report.items() if key not in ("runs_data", "environment", "trajectory_conditions")}, indent=2))


if __name__ == "__main__":
    main()
