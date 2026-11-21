"""New supplementary direct-differential experiment; no original records replaced.

All six lifted coefficients are learned on the unit sphere. The learner receives
only measured tangent vectors and phase, never a target derivative or coefficient.
Each update exactly integrates the normalized gradient flow with memory frozen.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
from pathlib import Path

import numpy as np

from common import dump_json, save_trace

PROTOCOL = dict(
    identity="direct_lift_20261118_v1",
    purpose="Supplementary direct normalized differential learning, separate from original protocols",
    dt=.02, duration=120., memory=8., gain=10., rate_noise=.005,
    seeds=list(range(10, 20)), scenarios=["stationary", "jump"],
    change_time=60., amplitude_before=1., amplitude_after=1.6,
    initial=[.3, 1., -.3, .15, .2, -.1],
    lifted_order=["v1", "v2", "cos(x)*v1", "cos(x)*v2", "sin(x)*v1", "sin(x)*v2"],
    reporting_windows={"before": [40., 60.], "change": [60., 80.], "late": [100., 120.]},
    update="q exp(-gain*dt*R), normalized; R first assimilates the current measured tangent",
    sensing="x=t and v1=1 are exact clock channels; v2=-A*cos(x)+independent Gaussian noise",
    scoring="All residuals and learned partials use the pre-update q; final sample is not assimilated",
    tuning="Fixed before running; no coefficient or derivative targets used for tuning or adaptation",
    scope="Known harmonic basis; no control; jump recovery is not a smooth-drift theorem test",
)


def lift(v, phase):
    return np.kron(np.array([1., np.cos(phase), np.sin(phase)]), np.asarray(v))


def evaluated(q, phase):
    return np.array([1., np.cos(phase), np.sin(phase)]) @ np.asarray(q).reshape(3, 2)


def flow_step(q, memory, gain, dt):
    """Exact solution of qdot=-gain*(qR-(qRq^T)q) for a frozen symmetric R."""
    values, vectors = np.linalg.eigh(memory)
    candidate = ((q @ vectors) * np.exp(-gain * dt * (values - values[0]))) @ vectors.T
    return candidate / np.linalg.norm(candidate)


def projector_distance(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if min(na, nb) <= 1e-12:
        return float("nan")
    return float(np.linalg.norm(np.outer(a / na, a / na) - np.outer(b / nb, b / nb)))


def simulate(seed, scenario, cfg=PROTOCOL):
    if scenario not in cfg["scenarios"]:
        raise ValueError(scenario)
    rng = np.random.default_rng(seed)
    q = np.array(cfg["initial"], float)
    q /= np.linalg.norm(q)
    initial = q.copy()
    memory = np.zeros((6, 6))
    beta = np.exp(-cfg["dt"] / cfg["memory"])
    count = round(cfg["duration"] / cfg["dt"])
    rows = []
    for n in range(count + 1):
        t = n * cfg["dt"]
        amplitude = cfg["amplitude_after"] if scenario == "jump" and t >= cfg["change_time"] else cfg["amplitude_before"]
        clean_v = np.array([1., -amplitude * np.cos(t)])
        observed_v = clean_v + np.array([0., rng.normal(0, cfg["rate_noise"])])
        psi = lift(observed_v, t)
        clean_psi = lift(clean_v, t)
        target_q = np.array([0., 1., amplitude, 0., 0., 0.])
        target_q /= np.linalg.norm(target_q)
        true_F = np.array([amplitude * np.cos(t), 1.])
        estimated_F = evaluated(q, t)
        row = dict(
            time=t, observed_v1=observed_v[0], observed_v2=observed_v[1],
            clean_v2=clean_v[1], amplitude=amplitude,
            F1_true=true_F[0], F2_true=true_F[1], F1_est=estimated_F[0], F2_est=estimated_F[1],
            derivative_projector_error=projector_distance(estimated_F, true_F),
            frozen_derivative_projector_error=projector_distance(evaluated(initial, t), true_F),
            evaluated_row_norm=float(np.linalg.norm(estimated_F)),
            coefficient_projector_error=projector_distance(q, target_q),
            preupdate_energy=float((psi @ q) ** 2 / 2),
            clean_energy=float((clean_psi @ q) ** 2 / 2),
            same_stream_true_energy=float((psi @ target_q) ** 2 / 2),
            coefficient_norm=float(np.linalg.norm(q)),
        )
        row.update({f"q{k}": float(q[k]) for k in range(6)})
        if n < count:
            memory = beta * memory + (1 - beta) * np.outer(psi, psi)
            eig, U = np.linalg.eigh(memory)
            row["coefficient_to_memory_error"] = projector_distance(q, U[:, 0])
            row["memory_to_physical_error"] = projector_distance(U[:, 0], target_q)
            row["memory_gap"] = float(eig[1] - eig[0])
            row["memory_energy_before"] = float(q @ memory @ q / 2)
            q = flow_step(q, memory, cfg["gain"], cfg["dt"])
            row["memory_energy_after"] = float(q @ memory @ q / 2)
        else:
            row.update({k: float("nan") for k in ["coefficient_to_memory_error", "memory_to_physical_error", "memory_gap", "memory_energy_before", "memory_energy_after"]})
        rows.append(row)
    return rows


def summarize(rows, cfg=PROTOCOL):
    result = {}
    for name, (lo, hi) in cfg["reporting_windows"].items():
        window = [r for r in rows if lo <= r["time"] < hi]
        result[name] = {k: float(np.mean([r[k] for r in window])) for k in [
            "derivative_projector_error", "frozen_derivative_projector_error",
            "coefficient_projector_error", "preupdate_energy", "clean_energy",
            "same_stream_true_energy", "coefficient_to_memory_error", "memory_to_physical_error",
        ]}
    result["minimum_evaluated_row_norm"] = min(r["evaluated_row_norm"] for r in rows)
    result["max_norm_deviation"] = max(abs(r["coefficient_norm"] - 1) for r in rows)
    result["max_memory_energy_increase"] = max(0., max(r["memory_energy_after"] - r["memory_energy_before"] for r in rows[:-1]))
    result["invalid_partial_rows"] = sum(not np.isfinite(r["derivative_projector_error"]) for r in rows)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError("Use a new output directory; existing outcomes are preserved")
    args.output.mkdir(parents=True, exist_ok=True)
    dump_json(args.output / "protocol.json", PROTOCOL)
    source = Path(__file__).resolve()
    repo = source.parents[1]
    dump_json(args.output / "environment.json", dict(
        python=sys.version, numpy=np.__version__, platform=platform.platform(),
        command=sys.argv, source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        git_head=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip(),
    ))
    (args.output / source.name).write_bytes(source.read_bytes())
    summaries = []
    for scenario in PROTOCOL["scenarios"]:
        for seed in PROTOCOL["seeds"]:
            rows = simulate(seed, scenario)
            dest = args.output / scenario / f"seed_{seed}"
            save_trace(dest / "trace.csv", rows)
            summary = dict(scenario=scenario, seed=seed, metrics=summarize(rows))
            dump_json(dest / "summary.json", summary)
            summaries.append(summary)
            print(json.dumps(dict(scenario=scenario, seed=seed, late_error=summary["metrics"]["late"]["derivative_projector_error"])), flush=True)
    dump_json(args.output / "summary.json", summaries)
    hashes = {str(p.relative_to(args.output)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(args.output.rglob("*")) if p.is_file()}
    dump_json(args.output / "sha256.json", hashes)


if __name__ == "__main__":
    main()
