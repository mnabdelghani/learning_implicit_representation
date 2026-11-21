"""Read-only record audit for the separately declared exact-Jacobian companion."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import scipy
from scipy.linalg import expm

from plot_direct_lift_audit import distances, fmt, mean_sd


def audit_run(path, cfg):
    data = np.genfromtxt(path / "trace.csv", delimiter=",", names=True)
    saved = json.loads((path / "summary.json").read_text())
    assert saved["scenario"] == path.parent.name
    assert saved["seed"] == int(path.name.split("_")[-1])
    assert len(data) == round(cfg["duration"] / cfg["dt"]) + 1
    t = np.arange(len(data)) * cfg["dt"]
    np.testing.assert_array_equal(data["time"], t)
    q = np.column_stack([data[f"q{j}"] for j in range(4)])
    A = np.full(len(t), cfg["amplitude_before"])
    if saved["scenario"] == "jump": A[t >= cfg["change_time"]] = cfg["amplitude_after"]
    c, s = np.cos(t), np.sin(t)
    np.testing.assert_array_equal(data["observed_v1"], np.ones(len(t)))
    psi = np.column_stack((np.ones(len(t)), data["observed_v2"], c, -s))
    clean = np.column_stack((np.ones(len(t)), -A*c, c, -s))
    F = np.column_stack((A*c, np.ones(len(t))))
    Fhat = np.column_stack((q[:, 0] + q[:, 2]*c - q[:, 3]*s, q[:, 1]))
    target = np.column_stack((np.zeros(len(t)), np.ones(len(t)), A, np.zeros(len(t))))
    target /= np.linalg.norm(target, axis=1, keepdims=True)
    initial = np.asarray(cfg["initial"], float)
    initial /= np.linalg.norm(initial)
    np.testing.assert_allclose(q[0], initial, atol=1e-15, rtol=0)
    frozen = np.column_stack((initial[0] + initial[2]*c - initial[3]*s, np.full(len(t), initial[1])))
    expected = dict(amplitude=A, clean_v2=-A*c, F1_true=F[:, 0], F2_true=F[:, 1],
                    F1_est=Fhat[:, 0], F2_est=Fhat[:, 1], evaluated_row_norm=np.linalg.norm(Fhat, axis=1),
                    coefficient_norm=np.linalg.norm(q, axis=1), derivative_projector_error=distances(Fhat, F),
                    frozen_derivative_projector_error=distances(frozen, F), coefficient_projector_error=distances(q, target),
                    preupdate_energy=np.einsum("ni,ni->n", psi, q)**2/2,
                    clean_energy=np.einsum("ni,ni->n", clean, q)**2/2,
                    same_stream_true_energy=np.einsum("ni,ni->n", psi, target)**2/2)
    memory_keys = ["coefficient_to_memory_error", "memory_to_physical_error", "memory_gap", "memory_energy_before", "memory_energy_after"]
    expected.update({key: np.full(len(t), np.nan) for key in memory_keys})
    R = np.zeros((4, 4))
    beta = np.exp(-cfg["dt"] / cfg["memory"])
    check_at = set(range(0, len(t)-1, 500)) | {len(t)-2, round(cfg["change_time"]/cfg["dt"])-1, round(cfg["change_time"]/cfg["dt"])}
    transition_errors = []
    for i in range(len(t)-1):
        R = beta*R + (1-beta)*np.outer(psi[i], psi[i])
        eigenvalues, eigenvectors = np.linalg.eigh(R)
        expected["coefficient_to_memory_error"][i] = distances(q[i], eigenvectors[:, 0])
        expected["memory_to_physical_error"][i] = distances(eigenvectors[:, 0], target[i])
        expected["memory_gap"][i] = eigenvalues[1]-eigenvalues[0]
        expected["memory_energy_before"][i] = q[i]@R@q[i]/2
        expected["memory_energy_after"][i] = q[i+1]@R@q[i+1]/2
        if i in check_at:
            prediction = q[i]@expm(-cfg["gain"]*cfg["dt"]*R)
            prediction /= np.linalg.norm(prediction)
            transition_errors.append(float(np.linalg.norm(prediction-q[i+1])))
    discrepancies = {}
    for key, value in expected.items():
        np.testing.assert_allclose(data[key], value, atol=2e-11, rtol=2e-9, equal_nan=True, err_msg=f"{path}: {key}")
        discrepancies[key] = float(np.nanmax(np.abs(data[key]-value)))
    np.testing.assert_allclose(transition_errors, 0, atol=2e-12, rtol=0)
    metrics = {}
    for window, (lo, hi) in cfg["reporting_windows"].items():
        selected = (t >= lo) & (t < hi)
        metrics[window] = {key: float(np.mean(expected[key][selected])) for key in saved["metrics"][window]}
        for key, value in metrics[window].items():
            np.testing.assert_allclose(value, saved["metrics"][window][key], atol=2e-11, rtol=2e-9)
    metrics.update(minimum_evaluated_row_norm=float(np.min(expected["evaluated_row_norm"])),
                   max_norm_deviation=float(np.max(np.abs(expected["coefficient_norm"]-1))),
                   max_memory_energy_increase=float(max(0, np.max(expected["memory_energy_after"][:-1]-expected["memory_energy_before"][:-1]))),
                   invalid_partial_rows=int(np.sum(~np.isfinite(expected["derivative_projector_error"]))))
    for key in ["minimum_evaluated_row_norm", "max_norm_deviation", "max_memory_energy_increase", "invalid_partial_rows"]:
        np.testing.assert_allclose(metrics[key], saved["metrics"][key], atol=2e-12, rtol=2e-9)
    return dict(scenario=saved["scenario"], seed=saved["seed"], rows=len(t), metrics=metrics,
                trace_sha256=hashlib.sha256((path/"trace.csv").read_bytes()).hexdigest(),
                max_column_discrepancies=discrepancies, independent_matrix_exponential_checks=len(transition_errors),
                maximum_transition_error=max(transition_errors))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()): raise FileExistsError("Use a new output directory")
    args.output.mkdir(parents=True, exist_ok=True)
    cfg = json.loads((args.input/"protocol.json").read_text())
    raw_hashes = json.loads((args.input/"sha256.json").read_text())
    for relative, digest in raw_hashes.items():
        assert hashlib.sha256((args.input/relative).read_bytes()).hexdigest() == digest, relative
    reports = [audit_run(args.input/scenario/f"seed_{seed}", cfg)
               for scenario in cfg["scenarios"] for seed in cfg["seeds"]]
    aggregate = {}
    for scenario in cfg["scenarios"]:
        records = [r["metrics"] for r in reports if r["scenario"] == scenario]
        aggregate[scenario] = {window: {name: mean_sd([r[window][name] for r in records]) for name in records[0][window]}
                               for window in cfg["reporting_windows"]}
        aggregate[scenario].update(minimum_evaluated_row_norm=min(r["minimum_evaluated_row_norm"] for r in records),
                                   maximum_norm_deviation=max(r["max_norm_deviation"] for r in records),
                                   maximum_memory_energy_increase=max(r["max_memory_energy_increase"] for r in records),
                                   invalid_partial_rows=sum(r["invalid_partial_rows"] for r in records))
    sources = [Path(__file__).resolve(), Path(__file__).with_name("plot_direct_lift_audit.py").resolve()]
    provenance = dict(command=sys.argv, python=sys.version, numpy=np.__version__, scipy=scipy.__version__,
                      source_sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sources})
    for p in sources: (args.output/p.name).write_bytes(p.read_bytes())
    def save(name, value): (args.output/name).write_text(json.dumps(value, indent=2, allow_nan=False)+"\n")
    save("audit.json", dict(status="passed", runs=len(reports), original_files_hash_checked=len(raw_hashes),
                            provenance=provenance, reports=reports))
    save("aggregate.json", aggregate)
    lines = [r"\begin{table*}[t]", r"\centering",
             r"\caption{Exact-Jacobian companion: mean $\pm$ sample standard deviation across all ten seeds per scenario, after averaging each pre-update diagnostic over its half-open time window. Every candidate is an exact Jacobian and all four coefficients adapt.}",
             r"\label{tab:direct_jacobian_results}", r"\begin{tabular}{llcccc}", r"\toprule",
             r"Scenario & Window (s) & Learned derivative & Frozen derivative & Optimizer lag & Memory bias\\", r"\midrule"]
    for scenario in cfg["scenarios"]:
        for window, (lo, hi) in cfg["reporting_windows"].items():
            vals = [fmt(aggregate[scenario][window][key]) for key in ["derivative_projector_error", "frozen_derivative_projector_error", "coefficient_to_memory_error", "memory_to_physical_error"]]
            lines.append(f"{scenario.title()} & {lo:g}--{hi:g} & " + " & ".join(vals) + r"\\")
        if scenario == "stationary": lines.append(r"\midrule")
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table*}"])
    (args.output/"direct_jacobian_table.tex").write_text("\n".join(lines)+"\n")
    print(json.dumps(dict(status="passed", runs=len(reports), rows=sum(r["rows"] for r in reports),
                          independent_transition_checks=sum(r["independent_matrix_exponential_checks"] for r in reports))))


if __name__ == "__main__": main()
