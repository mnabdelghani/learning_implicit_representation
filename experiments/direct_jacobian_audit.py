"""New exact-Jacobian companion to the separate six-coefficient rate-field audit.

All four coefficients adapt on the sphere. Every fixed-candidate evaluated row
is the exact context Jacobian of q0*x+q1*y+q2*sin(x)+q3*cos(x). No derivative
entry or true coefficient is supplied to the learner.
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
from direct_lift_audit import flow_step, projector_distance

PROTOCOL = dict(
    identity="direct_jacobian_20261118_v1",
    purpose="New exact-Jacobian companion; original main and six-dimensional protocols unchanged",
    dt=.02, duration=120., memory=8., gain=10., rate_noise=.005,
    seeds=list(range(10, 20)), scenarios=["stationary", "jump"],
    change_time=60., amplitude_before=1., amplitude_after=1.6,
    initial=[.3, 1., -.3, .2],
    residual_family="q0*x+q1*y+q2*sin(x)+q3*cos(x); all four coefficients learned",
    lifted_order=["v1", "v2", "cos(x)*v1", "-sin(x)*v1"],
    reporting_windows={"before": [40., 60.], "change": [60., 80.], "late": [100., 120.]},
    update="q exp(-gain*dt*R), normalized; R assimilates current measured tangent first",
    sensing="Exact x=t and v1=1; v2=-A*cos(x)+independent Gaussian noise",
    scoring="Pre-update q; final sample recorded but not assimilated",
    tuning="Protocol and initial [.3,1,-.3,.2] fixed before execution; no outcome-based changes",
    scope="Known integrable feature family and clock; no control; no jump-through smooth-drift theorem",
)


def residual(z, q):
    x, y = np.asarray(z)
    return q[0] * x + q[1] * y + q[2] * np.sin(x) + q[3] * np.cos(x)


def evaluated(q, phase):
    return np.array([q[0] + q[2] * np.cos(phase) - q[3] * np.sin(phase), q[1]])


def lift(v, phase):
    return np.array([v[0], v[1], np.cos(phase) * v[0], -np.sin(phase) * v[0]])


def simulate(seed, scenario, cfg=PROTOCOL):
    if scenario not in cfg["scenarios"]: raise ValueError(scenario)
    rng = np.random.default_rng(seed)
    q = np.asarray(cfg["initial"], float)
    q = q / np.linalg.norm(q)
    initial = q.copy()
    R = np.zeros((4, 4))
    beta = np.exp(-cfg["dt"] / cfg["memory"])
    count = round(cfg["duration"] / cfg["dt"])
    rows = []
    for n in range(count + 1):
        t = n * cfg["dt"]
        A = cfg["amplitude_after"] if scenario == "jump" and t >= cfg["change_time"] else cfg["amplitude_before"]
        clean = np.array([1., -A * np.cos(t)])
        measured = clean + [0., rng.normal(0, cfg["rate_noise"])]
        psi = lift(measured, t)
        target = np.array([0., 1., A, 0.]) / np.sqrt(1 + A * A)
        true_F = np.array([A * np.cos(t), 1.])
        estimated_F = evaluated(q, t)
        row = dict(
            time=t, observed_v1=measured[0], observed_v2=measured[1], clean_v2=clean[1], amplitude=A,
            F1_true=true_F[0], F2_true=true_F[1], F1_est=estimated_F[0], F2_est=estimated_F[1],
            derivative_projector_error=projector_distance(estimated_F, true_F),
            frozen_derivative_projector_error=projector_distance(evaluated(initial, t), true_F),
            evaluated_row_norm=float(np.linalg.norm(estimated_F)),
            coefficient_projector_error=projector_distance(q, target),
            preupdate_energy=float((psi @ q) ** 2 / 2),
            clean_energy=float((lift(clean, t) @ q) ** 2 / 2),
            same_stream_true_energy=float((psi @ target) ** 2 / 2),
            coefficient_norm=float(np.linalg.norm(q)),
        )
        row.update({f"q{k}": float(q[k]) for k in range(4)})
        if n < count:
            R = beta * R + (1 - beta) * np.outer(psi, psi)
            eig, U = np.linalg.eigh(R)
            row.update(coefficient_to_memory_error=projector_distance(q, U[:, 0]),
                       memory_to_physical_error=projector_distance(U[:, 0], target),
                       memory_gap=float(eig[1] - eig[0]), memory_energy_before=float(q @ R @ q / 2))
            q = flow_step(q, R, cfg["gain"], cfg["dt"])
            row["memory_energy_after"] = float(q @ R @ q / 2)
        else:
            row.update({name: float("nan") for name in ["coefficient_to_memory_error", "memory_to_physical_error",
                        "memory_gap", "memory_energy_before", "memory_energy_after"]})
        rows.append(row)
    return rows


def summarize(rows, cfg=PROTOCOL):
    result = {}
    names = ["derivative_projector_error", "frozen_derivative_projector_error", "coefficient_projector_error",
             "preupdate_energy", "clean_energy", "same_stream_true_energy", "coefficient_to_memory_error", "memory_to_physical_error"]
    for label, (lo, hi) in cfg["reporting_windows"].items():
        selection = [r for r in rows if lo <= r["time"] < hi]
        result[label] = {name: float(np.mean([r[name] for r in selection])) for name in names}
    result.update(minimum_evaluated_row_norm=min(r["evaluated_row_norm"] for r in rows),
                  max_norm_deviation=max(abs(r["coefficient_norm"] - 1) for r in rows),
                  max_memory_energy_increase=max(0., max(r["memory_energy_after"] - r["memory_energy_before"] for r in rows[:-1])),
                  invalid_partial_rows=sum(not np.isfinite(r["derivative_projector_error"]) for r in rows))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()): raise FileExistsError("Use an empty new output directory")
    args.output.mkdir(parents=True, exist_ok=True)
    dump_json(args.output / "protocol.json", PROTOCOL)
    source = Path(__file__).resolve()
    repo = source.parents[1]
    sources = [source, source.with_name("common.py"), source.with_name("direct_lift_audit.py")]
    hashes = {str(p.relative_to(repo)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
    dump_json(args.output / "environment.json", dict(python=sys.version, numpy=np.__version__, platform=platform.platform(),
              command=sys.argv, source_sha256=hashes,
              git_head=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()))
    for p in sources: (args.output / p.name).write_bytes(p.read_bytes())
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
    hashes = {str(p.relative_to(args.output)): hashlib.sha256(p.read_bytes()).hexdigest()
              for p in sorted(args.output.rglob("*")) if p.is_file()}
    dump_json(args.output / "sha256.json", hashes)


if __name__ == "__main__": main()
