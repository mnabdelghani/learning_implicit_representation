#!/usr/bin/env python3
"""NEW RECONSTRUCTION: constrained-energy learning and control of viscous Burgers.

The revised manuscript specifies the plant/protocol but not all implementation
choices. See docs/burgers_notes.txt and the saved config. No published numerical
result is used by this program. Importing the module does not run a simulation.
"""
from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import platform
from collections import deque
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
METHODS = ("learned", "frozen", "oracle")
TRACE_COLUMNS = (
    "time", "command_applied", "tracking_rms", "rate_rmse_clean_preupdate",
    "rate_rmse_measured_preupdate", "control_rms", "control_peak", "field_mean",
    "field_energy", "F_relative_error_preupdate", "F_min_row_norm_preupdate",
    "ca_pre", "nu_pre", "b_pre", "ca_post", "nu_post", "b_post",
    "sample_energy_noisy_preupdate", "sample_energy_clean_preupdate",
    "history_energy_before_steps", "history_energy_after_steps",
    "history_min_eigenvalue_scaled", "largest_inner_energy_increase",
    "largest_descent_bound_violation",
)


def load_config(path=None):
    with open(path or ROOT / "configs" / "burgers.json", encoding="utf-8") as f:
        cfg = json.load(f)
    validate_config(cfg)
    return cfg


def validate_config(cfg):
    n, dt, duration = cfg["grid_points"], cfg["dt"], cfg["duration"]
    if n < 8 or n % 2 or dt <= 0 or duration <= cfg["difference_window"]:
        raise ValueError("Use an even grid >=8, positive dt, and duration > difference_window.")
    for name in ("duration", "difference_window", "snapshot_interval"):
        ratio = cfg[name] / dt
        if not np.isclose(ratio, round(ratio), atol=1e-9):
            raise ValueError(f"{name} must be an integer multiple of dt")
    if not (0 < cfg["evaluation_window"] <= duration):
        raise ValueError("evaluation_window must lie in (0,duration]")
    if cfg["memory_time"] <= 0 or cfg["sensor_noise_std"] < 0:
        raise ValueError("Invalid memory_time or sensor_noise_std")
    lo, hi = np.array(cfg["bounds_lower"]), np.array(cfg["bounds_upper"])
    scale, initial = np.array(cfg["coefficient_scales"]), np.array(cfg["initial_coefficients"])
    if any(x.shape != (3,) for x in (lo, hi, scale, initial)):
        raise ValueError("Coefficient entries must each have three elements: ca,nu,b")
    if np.any(lo >= hi) or np.any(scale <= 0) or np.any(initial < lo) or np.any(initial > hi):
        raise ValueError("Invalid bounds, scales, or initial coefficients")
    if lo[1] <= 0 or lo[2] <= 0 or cfg["projected_steps"] < 1:
        raise ValueError("Positive viscosity/input-gain bounds and update count are required")


def spatial_grid(n, length=2 * np.pi):
    return np.arange(n, dtype=float) * length / n


def spectral_derivatives(w, length=2 * np.pi):
    """First and second derivatives on an evenly spaced periodic real grid."""
    w = np.asarray(w, dtype=float)
    k = 2 * np.pi * np.fft.fftfreq(w.size, d=length / w.size)
    wh = np.fft.fft(w)
    return np.fft.ifft(1j * k * wh).real, np.fft.ifft(-k * k * wh).real


def initial_field(s):
    return 0.3 * np.sin(s) + 0.1 * np.cos(2 * s)


def reference_field(s, t):
    wd = 0.4 * np.sin(s) * np.cos(0.7 * t) + 0.2 * np.cos(2 * s) * np.sin(1.1 * t)
    wdt = -0.28 * np.sin(s) * np.sin(0.7 * t) + 0.22 * np.cos(2 * s) * np.cos(1.1 * t)
    return wd, wdt


def plant_rhs(w, u, coefficients, length=2 * np.pi):
    ca, nu, b = coefficients
    ws, wss = spectral_derivatives(w, length)
    return -ca * w * ws + nu * wss + b * u


def rk4_step(w, u, dt, coefficients, length=2 * np.pi):
    """Classical RK4; exactly the same selected input is held at all stages."""
    k1 = plant_rhs(w, u, coefficients, length)
    k2 = plant_rhs(w + dt * k1 / 2, u, coefficients, length)
    k3 = plant_rhs(w + dt * k2 / 2, u, coefficients, length)
    k4 = plant_rhs(w + dt * k3, u, coefficients, length)
    return w + dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6


def implicit_residual(z, coefficients):
    """z has last-axis ordering (w,ws,wss,u,wt)."""
    ca, nu, b = np.asarray(coefficients)
    z = np.asarray(z)
    return z[..., 4] + ca * z[..., 0] * z[..., 1] - nu * z[..., 2] - b * z[..., 3]


def implicit_jacobian(z, coefficients):
    """Return F = partial f / partial z in the manuscript coordinate convention."""
    ca, nu, b = np.asarray(coefficients)
    z = np.asarray(z)
    out = np.empty_like(z, dtype=float)
    out[..., 0] = ca * z[..., 1]
    out[..., 1] = ca * z[..., 0]
    out[..., 2] = -nu
    out[..., 3] = -b
    out[..., 4] = 1.0
    return out


def residual_features(w, u, observed_rate, length=2 * np.pi):
    """Complete features (wt,w*ws,-wss,-u); differentiate these as a whole."""
    ws, wss = spectral_derivatives(w, length)
    return np.column_stack((observed_rate, w * ws, -wss, -u))


def incompatibility(feature_rates, coefficients):
    return np.asarray(feature_rates) @ np.r_[1.0, coefficients]


def inverse_control(w, s, t, coefficients, gain=2.0, length=2 * np.pi):
    ca, nu, b = coefficients
    if b <= 0:
        raise ValueError("The input gain must be positive for this inverse readout")
    wd, wdt = reference_field(s, t)
    ws, wss = spectral_derivatives(w, length)
    return (wdt + ca * w * ws - nu * wss - gain * (w - wd)) / b


class ConstrainedEnergyLearner:
    """Projected gradient on a stored complete-feature incompatibility energy.

    True plant coefficients never enter this object. The fixed wt partial is 1;
    the other three coefficients are restricted to a declared box. Scaling is
    only a choice of optimization coordinates.
    """
    def __init__(self, cfg, initial=None):
        self.scales = np.array(cfg["coefficient_scales"], dtype=float)
        self.lower = np.array(cfg["bounds_lower"], dtype=float) / self.scales
        self.upper = np.array(cfg["bounds_upper"], dtype=float) / self.scales
        self.theta = np.array(initial if initial is not None else cfg["initial_coefficients"], dtype=float) / self.scales
        self.beta = float(np.exp(-cfg["dt"] / cfg["memory_time"]))
        self.n_steps = cfg["projected_steps"]
        self.trace_floor = cfg["curvature_trace_floor"]
        self.moment = np.zeros((4, 4))
        self.memory_mass = 0.0

    @property
    def coefficients(self):
        return self.scales * self.theta

    def energy(self, theta=None):
        p = np.r_[1.0, self.scales * (self.theta if theta is None else theta)]
        return float(0.5 * p @ self.moment @ p)

    def scaled_free_moment(self):
        return self.scales[:, None] * self.moment[1:, 1:] * self.scales[None, :]

    def accumulate(self, feature_rates):
        r = np.asarray(feature_rates, dtype=float)
        if r.ndim != 2 or r.shape[1] != 4 or not np.all(np.isfinite(r)):
            raise ValueError("Expected finite spatial samples of four complete-feature rates")
        self.moment = self.beta * self.moment + (1 - self.beta) * (r.T @ r / r.shape[0])
        self.moment = 0.5 * (self.moment + self.moment.T)
        self.memory_mass = self.beta * self.memory_mass + (1 - self.beta)

    def projected_steps(self, n_steps=None):
        step = 1 / max(float(np.trace(self.scaled_free_moment())), self.trace_floor)
        energies = [self.energy()]
        violations = []
        for _ in range(self.n_steps if n_steps is None else n_steps):
            p = np.r_[1.0, self.coefficients]
            gradient = self.scales * (self.moment[1:, :] @ p)
            candidate = np.clip(self.theta - step * gradient, self.lower, self.upper)
            change = candidate - self.theta
            new_energy = self.energy(candidate)
            violations.append(float(new_energy - energies[-1] + change @ change / (2 * step)))
            self.theta = candidate
            energies.append(new_energy)
        return {
            "energies": energies,
            "step_size": step,
            "max_increase": max([0.0] + list(np.diff(energies))),
            "max_descent_violation": max([0.0] + violations),
        }

    def update(self, feature_rates, adapt=True):
        self.accumulate(feature_rates)
        return self.projected_steps(self.n_steps if adapt else 0)


def rms(x):
    return float(np.sqrt(np.mean(np.asarray(x) ** 2)))


def simulate_episode(cfg, method="learned", seed=10):
    validate_config(cfg)
    if method not in METHODS:
        raise ValueError(f"method must be one of {METHODS}")
    n, dt, duration = cfg["grid_points"], cfg["dt"], cfg["duration"]
    length = cfg["domain_length"]
    count = int(round(duration / dt))
    lag = int(round(cfg["difference_window"] / dt))
    snapshot_stride = int(round(cfg["snapshot_interval"] / dt))
    truth = np.array(cfg["true_coefficients"], dtype=float)
    initial = truth if method == "oracle" else cfg["initial_coefficients"]
    learner = ConstrainedEnergyLearner(cfg, initial)
    rng = np.random.default_rng(seed)
    measured_history, clean_history = deque(maxlen=lag + 1), deque(maxlen=lag + 1)
    s = spatial_grid(n, length)
    w = initial_field(s)
    rows, snapshots = [], []
    for index in range(count + 1):
        t = index * dt
        pre = learner.coefficients.copy()
        wd, wdt = reference_field(s, t)
        u = inverse_control(w, s, t, pre, cfg["feedback_gain"], length)
        actual_rate = plant_rhs(w, u, truth, length)
        measured_rate = actual_rate + rng.normal(0, cfg["sensor_noise_std"], size=n)
        predicted_rate = plant_rhs(w, u, pre, length)
        measured_history.append(residual_features(w, u, measured_rate, length))
        clean_history.append(residual_features(w, u, actual_rate, length))
        ws, wss = spectral_derivatives(w, length)
        z = np.column_stack((w, ws, wss, u, measured_rate))
        F_true, F_pred = implicit_jacobian(z, truth), implicit_jacobian(z, pre)
        enoise = eclean = hbefore = hafter = mineig = float("nan")
        inc = violation = 0.0
        if len(measured_history) == lag + 1:
            rates = (measured_history[-1] - measured_history[0]) / cfg["difference_window"]
            clean_rates = (clean_history[-1] - clean_history[0]) / cfg["difference_window"]
            enoise = 0.5 * float(np.mean(incompatibility(rates, pre) ** 2))
            eclean = 0.5 * float(np.mean(incompatibility(clean_rates, pre) ** 2))
            diagnostic = learner.update(rates, adapt=(method == "learned" and index < count))
            hbefore, hafter = diagnostic["energies"][0], diagnostic["energies"][-1]
            mineig = float(np.linalg.eigvalsh(learner.scaled_free_moment())[0])
            inc, violation = diagnostic["max_increase"], diagnostic["max_descent_violation"]
        post = learner.coefficients.copy()
        rows.append((t, int(index < count), rms(w - wd), rms(predicted_rate - actual_rate),
                     rms(predicted_rate - measured_rate), rms(u), float(np.max(np.abs(u))),
                     float(np.mean(w)), float(0.5 * np.mean(w * w)),
                     float(np.linalg.norm(F_pred - F_true) / np.linalg.norm(F_true)),
                     float(np.min(np.linalg.norm(F_pred, axis=1))),
                     *pre, *post, enoise, eclean, hbefore, hafter, mineig, inc, violation))
        if index % snapshot_stride == 0 or index == count:
            snapshots.append((t, w.copy(), wd.copy(), u.copy(), actual_rate.copy(), predicted_rate.copy()))
        if index < count:
            w = rk4_step(w, u, dt, truth, length)
            if not np.all(np.isfinite(w)):
                raise FloatingPointError(f"Nonfinite plant at t={t+dt:g}, method={method}, seed={seed}")
    trace = np.array(rows, dtype=float)
    by_name = {name: trace[:, j] for j, name in enumerate(TRACE_COLUMNS)}
    selection = (by_name["time"] >= duration - cfg["evaluation_window"] - 1e-10) & (by_name["command_applied"] == 1)
    valid = selection & np.isfinite(by_name["sample_energy_noisy_preupdate"])
    metric = {}
    for name in ("tracking_rms", "rate_rmse_clean_preupdate", "rate_rmse_measured_preupdate", "control_rms"):
        metric[name] = rms(by_name[name][selection])
    metric["control_peak"] = float(np.max(by_name["control_peak"][selection]))
    metric["control_energy_full_run"] = float(dt * np.sum(by_name["control_rms"][:-1] ** 2))
    metric["F_relative_error_mean"] = float(np.mean(by_name["F_relative_error_preupdate"][selection]))
    metric["F_min_row_norm"] = float(np.min(by_name["F_min_row_norm_preupdate"]))
    coeffs = trace[selection, 11:14]
    for j, name in enumerate(("ca", "nu", "b")):
        metric[name + "_relative_rms_error"] = rms((coeffs[:, j] - truth[j]) / truth[j])
    metric["sample_energy_noisy_mean"] = float(np.mean(by_name["sample_energy_noisy_preupdate"][valid]))
    metric["sample_energy_clean_mean"] = float(np.mean(by_name["sample_energy_clean_preupdate"][valid]))
    metric["largest_inner_energy_increase"] = float(np.max(by_name["largest_inner_energy_increase"]))
    metric["largest_descent_bound_violation"] = float(np.max(by_name["largest_descent_bound_violation"]))
    metric["maximum_absolute_field_mean"] = float(np.max(np.abs(by_name["field_mean"])))
    return {"method": method, "seed": seed, "metrics": metric,
            "final_coefficients": learner.coefficients.tolist(), "grid": s,
            "trace": trace, "trace_columns": TRACE_COLUMNS, "snapshots": snapshots,
            "evaluation_interval": [duration - cfg["evaluation_window"], duration],
            "right_endpoint_excluded": True, "time_steps": count}


def save_episode(episode, directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / "trace.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(episode["trace_columns"])
        writer.writerows(episode["trace"])
    with (directory / "fields.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(("time", "s", "w", "desired", "u", "true_rate", "predicted_rate_preupdate"))
        for t, w, wd, u, actual_rate, predicted_rate in episode["snapshots"]:
            writer.writerows(zip(np.full(w.size, t), episode["grid"], w, wd, u, actual_rate, predicted_rate))
    short = {k: episode[k] for k in ("method", "seed", "metrics", "final_coefficients", "evaluation_interval", "right_endpoint_excluded", "time_steps")}
    short["status"] = "NEW RECONSTRUCTION"
    (directory / "summary.json").write_text(json.dumps(short, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return short


def make_plot(episodes, cfg, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 3, figsize=(13.2, 7.3), constrained_layout=True)
    def column(episode, name):
        return episode["trace"][:, TRACE_COLUMNS.index(name)]
    learned = episodes["learned"]
    t = column(learned, "time")
    for name, true in zip(("ca", "nu", "b"), cfg["true_coefficients"]):
        axes[0, 0].plot(t, column(learned, name + "_pre") / true, label=name)
    axes[0, 0].axhline(1, color="k", lw=0.7, ls="--")
    axes[0, 0].set(title="Learned coefficient / true coefficient", ylabel="Ratio")
    axes[0, 0].legend()
    for name, label in (("sample_energy_noisy_preupdate", "Measured incompatibility"), ("sample_energy_clean_preupdate", "Clean diagnostic")):
        axes[0, 1].semilogy(t, np.maximum(column(learned, name), 1e-20), label=label, alpha=0.8, lw=1)
    floor = (cfg["sensor_noise_std"] / cfg["difference_window"]) ** 2
    if floor > 0:
        axes[0, 1].axhline(floor, color="k", lw=0.8, ls="--", label="True-model noise floor")
    axes[0, 1].set(title="Preupdate sample energy", ylabel="Half spatial mean square")
    axes[0, 1].legend(fontsize=8)
    for method in METHODS:
        ep = episodes[method]
        axes[0, 2].semilogy(column(ep, "time"), np.maximum(column(ep, "tracking_rms"), 1e-20), label=method)
        axes[1, 0].plot(column(ep, "time"), column(ep, "control_rms"), label=method)
        if method != "oracle":
            axes[1, 1].semilogy(column(ep, "time"), np.maximum(column(ep, "F_relative_error_preupdate"), 1e-20), label=method)
    axes[0, 2].set(title="Tracking on each method's trajectory", ylabel="Spatial RMS error")
    axes[1, 0].set(title="Distributed control effort", ylabel="Spatial RMS input")
    axes[1, 1].set(title="Partial-matrix error (fixed coordinates)", ylabel="Relative Frobenius error")
    for ax in (axes[0, 2], axes[1, 0], axes[1, 1]):
        ax.legend(fontsize=8)
    snapshot_times = np.array([r[0] for r in learned["snapshots"]])
    errors = np.array([r[1] - r[2] for r in learned["snapshots"]]).T
    image = axes[1, 2].pcolormesh(snapshot_times, learned["grid"], errors, shading="auto", cmap="RdBu_r")
    axes[1, 2].set(title="Learned-controller field error", ylabel="Spatial coordinate")
    fig.colorbar(image, ax=axes[1, 2], label="w - desired")
    for ax in axes.flat:
        ax.set_xlabel("Time (s)")
        ax.grid(alpha=0.15)
    suffix = "SMOKE RUN" if cfg.get("quick_run", False) else "full protocol"
    fig.suptitle(f"Burgers — NEW RECONSTRUCTION, seed {learned['seed']}, {suffix}", fontsize=14)
    fig.savefig(str(path) + ".png", dpi=170)
    fig.savefig(str(path) + ".pdf")
    plt.close(fig)


def parse_seeds(seed, seeds):
    if seeds is None:
        return [seed]
    if len(seeds) == 0:
        return list(range(10, 20))
    out = []
    for item in seeds:
        for token in item.split(","):
            if ":" in token:
                start, end = map(int, token.split(":"))
                if end < start:
                    raise ValueError("Seed ranges are inclusive and ascending")
                out.extend(range(start, end + 1))
            else:
                out.append(int(token))
    return list(dict.fromkeys(out))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "results" / "burgers")
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "burgers.json")
    parser.add_argument("--seed", type=int, default=10)
    parser.add_argument("--seeds", nargs="*", help="Integers, comma lists, inclusive a:b ranges; flag alone uses 10:19")
    parser.add_argument("--quick", action="store_true", help="2 s smoke run, last 0.5 s evaluation; otherwise same discretization")
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args(argv)
    cfg = copy.deepcopy(load_config(args.config))
    if args.quick:
        cfg.update(duration=2.0, evaluation_window=0.5, quick_run=True)
    validate_config(cfg)
    seeds = parse_seeds(args.seed, args.seeds)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "config_used.json").write_text(json.dumps(cfg, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    summaries = []
    for seed in seeds:
        episodes = {}
        for method in METHODS:
            episodes[method] = simulate_episode(cfg, method, seed)
            short = save_episode(episodes[method], args.output / f"seed_{seed}" / method)
            summaries.append(short)
            print(f"seed={seed} {method}: tracking_rms={short['metrics']['tracking_rms']:.9g}, coefficients={short['final_coefficients']}", flush=True)
        if not args.no_plots:
            make_plot(episodes, cfg, args.output / f"seed_{seed}" / "burgers")
    aggregate = {}
    for method in METHODS:
        selected = [r for r in summaries if r["method"] == method]
        aggregate[method] = {name: {"mean": float(np.mean([r["metrics"][name] for r in selected])),
                                    "std": float(np.std([r["metrics"][name] for r in selected], ddof=1)) if len(selected) > 1 else None,
                                    "count": len(selected)} for name in selected[0]["metrics"]}
    result = {"status": "NEW RECONSTRUCTION", "reproduces_original_paper_numbers": False,
              "quick_run": bool(cfg.get("quick_run", False)), "seeds": seeds,
              "noise_floor_at_true_coefficients": (cfg["sensor_noise_std"] / cfg["difference_window"]) ** 2,
              "config": cfg, "aggregate": aggregate, "episodes": summaries,
              "code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "runtime": {"python": platform.python_version(), "numpy": np.__version__}}
    (args.output / "burgers_summary.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"Saved actual traces and summaries to {args.output.resolve()}", flush=True)
    return result


if __name__ == "__main__":
    main()
