"""Independently audit saved direct-lift traces and prepare supplement assets.

This postprocessor does not import or run the simulation. It never modifies its
input records. The supplied output directory holds audits, aggregates and plots;
copying selected public assets into the supplement is a separate explicit step.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.linalg import expm


def load_json(path):
    return json.loads(Path(path).read_text())


def save_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def distances(a, b):
    a = np.asarray(a, float)
    b = np.broadcast_to(np.asarray(b, float), a.shape)
    a = a / np.linalg.norm(a, axis=-1, keepdims=True)
    b = b / np.linalg.norm(b, axis=-1, keepdims=True)
    return np.linalg.norm(a[..., :, None] * a[..., None, :] -
                          b[..., :, None] * b[..., None, :], axis=(-2, -1))


def audit_run(directory, cfg):
    trace_path = directory / "trace.csv"
    tr = np.genfromtxt(trace_path, delimiter=",", names=True)
    saved = load_json(directory / "summary.json")
    q = np.column_stack([tr[f"q{j}"] for j in range(6)])
    t = tr["time"]
    h = np.column_stack((np.ones(len(t)), np.cos(t), np.sin(t)))
    expected_amplitude = np.full(len(t), cfg["amplitude_before"])
    if saved["scenario"] == "jump":
        expected_amplitude[t >= cfg["change_time"]] = cfg["amplitude_after"]
    v = np.column_stack((tr["observed_v1"], tr["observed_v2"]))
    vc = np.column_stack((np.ones(len(t)), -expected_amplitude * np.cos(t)))
    psi = (h[:, :, None] * v[:, None, :]).reshape(-1, 6)
    psic = (h[:, :, None] * vc[:, None, :]).reshape(-1, 6)
    F = np.column_stack((expected_amplitude * np.cos(t), np.ones(len(t))))
    Fhat = np.einsum("ni,nij->nj", h, q.reshape(-1, 3, 2))
    target = np.column_stack((np.zeros(len(t)), np.ones(len(t)), expected_amplitude,
                              np.zeros((len(t), 3))))
    target /= np.linalg.norm(target, axis=1, keepdims=True)
    initial = np.asarray(cfg["initial"], float)
    initial /= np.linalg.norm(initial)
    Ffrozen = h @ initial.reshape(3, 2)
    expected = {
        "amplitude": expected_amplitude,
        "clean_v2": vc[:, 1], "F1_true": F[:, 0], "F2_true": F[:, 1],
        "F1_est": Fhat[:, 0], "F2_est": Fhat[:, 1],
        "derivative_projector_error": distances(Fhat, F),
        "frozen_derivative_projector_error": distances(Ffrozen, F),
        "coefficient_projector_error": distances(q, target),
        "coefficient_norm": np.linalg.norm(q, axis=1),
        "evaluated_row_norm": np.linalg.norm(Fhat, axis=1),
        "preupdate_energy": np.einsum("ni,ni->n", psi, q) ** 2 / 2,
        "clean_energy": np.einsum("ni,ni->n", psic, q) ** 2 / 2,
        "same_stream_true_energy": np.einsum("ni,ni->n", psi, target) ** 2 / 2,
    }
    np.testing.assert_allclose(t, np.arange(len(t)) * cfg["dt"], atol=1e-13, rtol=0)
    np.testing.assert_array_equal(v[:, 0], np.ones(len(t)))
    np.testing.assert_allclose(q[0], initial, atol=1e-15, rtol=0)
    # Reconstruct every retained moment. Check posterior energies against the
    # next row's pre-update coefficients; independently use matrix exponentials
    # at fixed indices to test transitions without the simulator's eigensolver.
    beta = np.exp(-cfg["dt"] / cfg["memory"])
    R = np.zeros((6, 6))
    memory_keys = ["coefficient_to_memory_error", "memory_to_physical_error",
                   "memory_gap", "memory_energy_before", "memory_energy_after"]
    expected.update({name: np.full(len(t), np.nan) for name in memory_keys})
    checks_at = set(range(0, len(t) - 1, 500)) | {len(t) - 2}
    jump_index = round(cfg["change_time"] / cfg["dt"])
    checks_at |= {jump_index - 1, jump_index}
    transition_errors = []
    for i in range(len(t) - 1):
        R = beta * R + (1 - beta) * np.outer(psi[i], psi[i])
        values, vectors = np.linalg.eigh(R)
        expected["coefficient_to_memory_error"][i] = distances(q[i], vectors[:, 0])
        expected["memory_to_physical_error"][i] = distances(vectors[:, 0], target[i])
        expected["memory_gap"][i] = values[1] - values[0]
        expected["memory_energy_before"][i] = q[i] @ R @ q[i] / 2
        expected["memory_energy_after"][i] = q[i + 1] @ R @ q[i + 1] / 2
        if i in checks_at:
            independently_updated = q[i] @ expm(-cfg["gain"] * cfg["dt"] * R)
            independently_updated /= np.linalg.norm(independently_updated)
            transition_errors.append(float(np.linalg.norm(independently_updated - q[i + 1])))
    discrepancies = {}
    for name, values in expected.items():
        np.testing.assert_allclose(tr[name], values, atol=2e-11, rtol=2e-9, equal_nan=True,
                                   err_msg=f"{directory}: {name}")
        discrepancies[name] = float(np.nanmax(np.abs(tr[name] - values)))
    np.testing.assert_allclose(transition_errors, 0., atol=2e-12, rtol=0)
    all_metrics = {}
    metric_names = list(saved["metrics"]["before"])
    for label, (lo, hi) in cfg["reporting_windows"].items():
        mask = (t >= lo) & (t < hi)
        all_metrics[label] = {name: float(np.mean(expected[name][mask])) for name in metric_names}
        for name, value in all_metrics[label].items():
            np.testing.assert_allclose(value, saved["metrics"][label][name], atol=2e-11, rtol=2e-9)
    all_metrics.update(
        minimum_evaluated_row_norm=float(np.min(expected["evaluated_row_norm"])),
        max_norm_deviation=float(np.max(np.abs(expected["coefficient_norm"] - 1))),
        max_memory_energy_increase=float(max(0, np.max(expected["memory_energy_after"][:-1] -
                                                       expected["memory_energy_before"][:-1]))),
        invalid_partial_rows=int(np.sum(~np.isfinite(expected["derivative_projector_error"]))),
    )
    for name in ["minimum_evaluated_row_norm", "max_norm_deviation", "max_memory_energy_increase", "invalid_partial_rows"]:
        np.testing.assert_allclose(all_metrics[name], saved["metrics"][name], atol=2e-12, rtol=2e-9)
    report = dict(scenario=saved["scenario"], seed=saved["seed"], rows=len(tr),
                  trace_sha256=hashlib.sha256(trace_path.read_bytes()).hexdigest(),
                  max_column_discrepancies=discrepancies,
                  independent_matrix_exponential_checks=len(transition_errors),
                  maximum_transition_error=max(transition_errors), metrics=all_metrics)
    return report, (t, expected)


def mean_sd(values):
    return dict(mean=float(np.mean(values)), std=float(np.std(values, ddof=1)), count=len(values))


def fmt(stat):
    # Fixed scientific notation avoids visually disappearing standard deviations.
    exponent = int(np.floor(np.log10(abs(stat["mean"])))) if stat["mean"] else 0
    scale = 10. ** exponent
    return rf"$({stat['mean']/scale:.3f}\pm{stat['std']/scale:.3f})\times10^{{{exponent}}}$"


def table(aggregate):
    rows = [r"\begin{table*}[t]", r"\centering", r"\caption{Direct lifted differential learning: time means followed by mean $\pm$ sample standard deviation across all ten seeds per scenario. Physical derivative errors compare row projectors; optimizer lag and memory bias compare coefficient projectors. Every window excludes its right endpoint.}",
            r"\label{tab:direct_lift_results}", r"\begin{tabular}{llcccc}", r"\toprule",
            r"Scenario & Window (s) & Learned derivative & Frozen derivative & Optimizer lag & Memory bias\\", r"\midrule"]
    names = ["derivative_projector_error", "frozen_derivative_projector_error", "coefficient_to_memory_error", "memory_to_physical_error"]
    for scenario in ["stationary", "jump"]:
        for window, label in [("before", "40--60"), ("change", "60--80"), ("late", "100--120")]:
            vals = " & ".join(fmt(aggregate[scenario][window][name]) for name in names)
            rows.append(f"{scenario.title()} & {label} & {vals}" + r"\\")
        if scenario == "stationary": rows.append(r"\midrule")
    rows.extend([r"\bottomrule", r"\end{tabular}", r"\end{table*}"])
    return "\n".join(rows) + "\n"


def plot(series, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 10, "axes.labelsize": 10, "legend.fontsize": 8.5})
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.7), sharex=True, layout="constrained")
    for column, scenario in enumerate(["stationary", "jump"]):
        ts = series[scenario][0][0]
        for row, specs in enumerate([
            [("derivative_projector_error", "Learned derivative", "#0072B2"),
             ("frozen_derivative_projector_error", "Frozen derivative", "#666666")],
            [("coefficient_to_memory_error", "Optimizer lag", "#009E73"),
             ("memory_to_physical_error", "Memory bias", "#D55E00")],
        ]):
            ax = axes[row, column]
            # Avoid the nonunique early empirical eigenspace in lower panels.
            selected = (np.arange(len(ts)) % 5 == 0) & (ts < ts[-1])
            if row: selected &= ts >= 2 * np.pi
            for name, label, color in specs:
                values = np.array([record[1][name] for record in series[scenario]])
                avg, sd = np.mean(values, axis=0), np.std(values, axis=0, ddof=1)
                ax.plot(ts[selected], avg[selected], color=color, label=label, lw=1.2)
                ax.fill_between(ts[selected], np.maximum(avg[selected] - sd[selected], 1e-9),
                                avg[selected] + sd[selected], color=color, alpha=.18, linewidth=0)
            ax.set_yscale("log")
            ax.set_ylim(2e-5, 2 if row == 0 else .5)
            ax.grid(True, which="major", alpha=.2)
            ax.legend(loc="upper right")
            if scenario == "jump": ax.axvline(60, color="black", ls=":", lw=1)
            if column == 0: ax.set_ylabel("Row-projector distance" if row == 0 else "Coefficient-projector distance")
            if row: ax.set_xlabel("Time (s)")
        axes[0, column].set_title("Stationary amplitude" if scenario == "stationary" else "Amplitude changes at 60 s")
    fig.savefig(output / "direct_lift_diagnostics.pdf")
    fig.savefig(output / "direct_lift_diagnostics.png", dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError("Use a new analysis directory")
    args.output.mkdir(parents=True, exist_ok=True)
    cfg = load_json(args.input / "protocol.json")
    source_hashes = load_json(args.input / "sha256.json")
    for relative, digest in source_hashes.items():
        assert hashlib.sha256((args.input / relative).read_bytes()).hexdigest() == digest, relative
    reports = []
    series = {scenario: [] for scenario in cfg["scenarios"]}
    for scenario in cfg["scenarios"]:
        for seed in cfg["seeds"]:
            report, data = audit_run(args.input / scenario / f"seed_{seed}", cfg)
            reports.append(report)
            series[scenario].append(data)
    aggregate = {}
    for scenario in cfg["scenarios"]:
        records = [r["metrics"] for r in reports if r["scenario"] == scenario]
        aggregate[scenario] = {window: {name: mean_sd([r[window][name] for r in records])
                                       for name in records[0][window]} for window in cfg["reporting_windows"]}
        aggregate[scenario].update(
            minimum_evaluated_row_norm=min(r["minimum_evaluated_row_norm"] for r in records),
            maximum_norm_deviation=max(r["max_norm_deviation"] for r in records),
            maximum_memory_energy_increase=max(r["max_memory_energy_increase"] for r in records),
            invalid_partial_rows=sum(r["invalid_partial_rows"] for r in records),
        )
    save_json(args.output / "audit.json", dict(status="passed", runs=len(reports),
              original_files_hash_checked=len(source_hashes), reports=reports,
              postprocessor_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()))
    save_json(args.output / "aggregate.json", aggregate)
    (args.output / "direct_lift_table.tex").write_text(table(aggregate))
    plot(series, args.output)
    print(json.dumps(dict(status="passed", runs=len(reports), rows=sum(r["rows"] for r in reports),
                          independent_transition_checks=sum(r["independent_matrix_exponential_checks"] for r in reports))))


if __name__ == "__main__":
    main()
