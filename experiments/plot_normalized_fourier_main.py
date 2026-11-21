"""Main Fourier panels and current-rate readouts from archived normalized runs.

This is postprocessing only: no plant, learner, controller, or solver is rerun.
The rate completion uses pre-update q, current state/input/phase, and the known
zero residual level. Actual rate is accessed only afterwards for scoring. This
is an online current-rate completion, not future prediction or a held-out test.
Run with explicit --records, --output (new directory), and optionally --figures.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import platform
import shutil
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, obj):
    path.write_text(json.dumps(obj, indent=2, allow_nan=False) + "\n")


def load(path):
    return np.genfromtxt(path, delimiter=",", names=True)


def rate_completion(q, state, control, phase, scales, guard):
    """Solve f_q=0 for current rate without accepting either measured/true rate."""
    h = np.column_stack((np.ones_like(phase), np.cos(phase), np.sin(phase)))
    a = q * scales
    rho = np.sum(a[:, :3] * h, axis=1)
    kappa = np.sum(a[:, 3:6] * h, axis=1)
    normalized_denominator = np.abs(rho) / np.linalg.norm(q, axis=1)
    valid = np.isfinite(normalized_denominator) & (normalized_denominator >= guard)
    rate = np.divide(a[:, 6] * control - kappa * state, rho,
                     out=np.full(len(state), np.nan), where=valid)
    return rate, valid, normalized_denominator


def partials(q, observations, scales):
    phase = observations["phase"]
    h = np.column_stack((np.ones_like(phase), np.cos(phase), np.sin(phase)))
    hp = np.column_stack((np.zeros_like(phase), -np.sin(phase), np.cos(phase)))
    a = q * scales
    return np.column_stack((np.sum(a[:, 3:6] * h, axis=1),
                            np.sum(a[:, :3] * h, axis=1), -a[:, 6],
                            np.sum(a[:, :3] * hp, axis=1) * observations["evaluator_actual_rate"]
                            + np.sum(a[:, 3:6] * hp, axis=1) * observations["state"]))


def rolling(values, width):
    """Trailing, startup-truncated mean; invalid startup energies stay absent."""
    valid = np.isfinite(values)
    sums = np.r_[0., np.cumsum(np.where(valid, values, 0.))]
    counts = np.r_[0, np.cumsum(valid)]
    end = np.arange(1, len(values) + 1)
    start = np.maximum(0, end - width)
    count = counts[end] - counts[start]
    return np.divide(sums[end] - sums[start], count,
                     out=np.full(len(values), np.nan), where=count > 0)


def check_close(label, actual, expected, checks, atol=2e-12):
    if not np.allclose(actual, expected, rtol=2e-11, atol=atol, equal_nan=True):
        raise AssertionError(label)
    finite = np.isfinite(actual) & np.isfinite(expected)
    checks[label] = float(np.max(np.abs(actual[finite] - expected[finite]))) if np.any(finite) else 0.


def process_seed(records, seed, cfg, output, hashes):
    directory = records / "own_trajectory" / f"seed_{seed}"
    paths = [directory / "observations.csv", directory / "state.npz"]
    paths += [directory / method / "trace.csv" for method in ("normalized_flow", "frozen_prior")]
    for path in paths:
        hashes[str(path.relative_to(records))] = sha(path)
    observations, flow, frozen = load(paths[0]), load(paths[2]), load(paths[3])
    t = observations["time"]
    late = (t >= cfg["evaluation_window"][0]) & (t < cfg["evaluation_window"][1])
    assert len(t) == round(cfg["duration"] / cfg["dt"]) + 1
    assert np.count_nonzero(late) == round(np.diff(cfg["evaluation_window"])[0] / cfg["dt"])
    scales = np.asarray(cfg["scales"])
    checks, scores, columns = {}, {}, {key: observations[key] for key in observations.dtype.names}
    with np.load(paths[1]) as saved:
        q_pre = saved["q_pre"]
    for method, trace, index in (("normalized_flow", flow, 0), ("frozen_prior", frozen, 2)):
        for key in ("time", "state", "desired", "control"):
            check_close(f"{method}_{key}", trace[key], observations[key], checks, atol=0.)
        q = np.column_stack([trace[f"q{j}"] for j in range(7)])
        check_close(f"{method}_preupdate_q", q, q_pre[:, index, :], checks, atol=0.)
        F = partials(q, observations, scales)
        check_close(f"{method}_all_partials", F,
                    np.column_stack([trace[f"F_est_{j}"] for j in range(4)]), checks)
        check_close(f"{method}_input_aligned_partials", F / (q[:, 6] * scales[6])[:, None],
                    np.column_stack([trace[f"F_aligned_{j}"] for j in range(4)]), checks)
        # The diagnostic guard is declared in unit-q coordinates, just like the
        # original input guard. It does not change the recorded controller.
        rate, valid, denominator = rate_completion(q, observations["state"], observations["control"],
                                                   observations["phase"], scales, cfg["input_guard"])
        if method == "normalized_flow":
            requested_rate = .315 * np.cos(.7 * t) + .26 * np.cos(1.3 * t)
            requested_rate -= cfg["feedback_gain"] * (observations["state"] - observations["desired"])
            unguarded = valid & (trace["controller_guarded"] == 0)
            check_close("inverse_forward_identity", rate[unguarded], requested_rate[unguarded], checks)
        else:
            check_close("frozen_prior_query", rate[valid],
                        (observations["control"] - observations["state"])[valid], checks)
        check_close(f"{method}_rate_solve", F[valid, 1] * rate[valid]
                    + F[valid, 0] * observations["state"][valid]
                    + F[valid, 2] * observations["control"][valid], np.zeros(np.count_nonzero(valid)), checks)
        error = rate - observations["evaluator_actual_rate"]
        for suffix, values in (("rate_completion", rate), ("error", error),
                               ("valid", valid.astype(int)), ("normalized_rate_denominator", denominator)):
            columns[f"{method}_{suffix}"] = values
        metrics = {}
        for name, mask in (("late", late), ("complete_run", np.ones(len(t), dtype=bool))):
            selected = mask & valid
            metrics[f"{name}_rmse"] = float(np.sqrt(np.mean(error[selected] ** 2))) if np.any(selected) else None
            metrics[f"{name}_samples"] = int(np.count_nonzero(mask))
            metrics[f"{name}_valid_samples"] = int(np.count_nonzero(selected))
            metrics[f"{name}_invalid_samples"] = int(np.count_nonzero(mask & ~valid))
        metrics["minimum_normalized_rate_denominator"] = float(np.min(denominator))
        metrics["input_guard_count"] = int(np.sum(trace["readout_guarded"]))
        scores[method] = metrics
    # Recompute the three energies independently from complete feature secants.
    phase = observations["phase"]
    h = np.column_stack((np.ones_like(phase), np.cos(phase), np.sin(phase)))
    measured = np.column_stack((h * observations["observed_rate"][:, None],
                                h * observations["state"][:, None], -observations["control"]))
    clean = measured.copy()
    clean[:, :3] = h * observations["evaluator_actual_rate"][:, None]
    lag = round(cfg["difference_window"] / cfg["dt"])
    q = np.column_stack([flow[f"q{j}"] for j in range(7)])
    target = np.r_[cfg["true_coefficients"], 1.] / scales
    target /= np.linalg.norm(target)
    for key, features, row in (("preupdate_energy", measured, q[lag:]),
                                ("clean_energy", clean, q[lag:]),
                                ("true_row_energy", measured, target[None, :])):
        w = (features[lag:] - features[:-lag]) * scales / cfg["difference_window"]
        energy = np.sum(row * w, axis=1) ** 2 / 2
        check_close(key, energy, flow[key][lag:], checks)
        assert np.all(np.isnan(flow[key][:lag]))
    raw_metrics = {key + "_mean": float(np.mean(flow[key][late])) for key in
                   ("derivative_projector_error", "coefficient_projector_error", "aligned_F_error",
                    "preupdate_energy", "clean_energy", "true_row_energy")}
    raw_metrics.update(phase_partial_rmse=float(np.sqrt(np.mean(flow["phase_partial_error"][late] ** 2))),
                       frozen_aligned_F_error_mean=float(np.mean(frozen["aligned_F_error"][late])),
                       tracking_rmse=float(np.sqrt(np.mean((flow["state"][late] - flow["desired"][late]) ** 2))),
                       control_rms=float(np.sqrt(np.mean(flow["control"][late] ** 2))),
                       controller_guard_count=int(np.sum(flow["controller_guarded"])))
    with (output / f"seed_{seed}_current_rate.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(columns)
        writer.writerows(zip(*columns.values()))
    return dict(seed=seed, raw_late_metrics=raw_metrics, rate_completion=scores, checks_max_abs_difference=checks), flow


def figures(trace, cfg, output):
    plt.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42})
    t = trace["time"]
    transforms = {}

    def canvas(title, ylabel):
        fig, ax = plt.subplots(figsize=(2.30, 1.65))
        ax.set_title(title, fontsize=8, loc="left", pad=3)
        ax.set_xlabel("Time (s)", fontsize=7)
        ax.set_ylabel(ylabel, fontsize=7, labelpad=2)
        ax.tick_params(labelsize=6.5, pad=2)
        ax.xaxis.set_major_locator(MaxNLocator(4))
        ax.grid(alpha=.18, linewidth=.4)
        return fig, ax

    def save(fig, ax, name, description, legend=True):
        if legend:
            ax.legend(fontsize=6, frameon=False, handlelength=1.8, labelspacing=.18, borderpad=.1)
        fig.tight_layout(pad=.35)
        for suffix in ("pdf", "png"):
            fig.savefig(output / f"normalized_fourier_main_{name}.{suffix}", dpi=180)
        plt.close(fig)
        transforms[name] = description

    def line(ax, x, y, label, style="-", color="C0", log=False):
        ax.plot(x, np.maximum(y, 1e-18) if log else y, style, color=color, lw=.8, label=label)
        if log:
            ax.set_yscale("log")

    initial = t <= 16.
    for name, title, ylabel, j in (("rate", "(a) Rate partial: initial 16 s", r"$\widehat f_{\dot x}/a_u$", 1),
                                  ("phase", "(b) Clock partial: initial 16 s", r"$\widehat f_\phi/a_u$", 3)):
        fig, ax = canvas(title, ylabel)
        line(ax, t[initial], trace[f"F_true_{j}"][initial], "True", "--", "C1")
        line(ax, t[initial], trace[f"F_aligned_{j}"][initial], "Learned")
        save(fig, ax, name, dict(columns=[f"F_true_{j}", f"F_aligned_{j}"], interval="0 <= t <= 16 s",
                                stride=1, smoothing="none", gauge="estimate divided by learned input coefficient a_u"))
    fig, ax = canvas("(c) Full differential error", "Error")
    for key, label, color in (("derivative_projector_error", "Projector distance", "C0"),
                               ("aligned_F_error", r"Input-aligned $e_F$", "C1")):
        line(ax, t[::3], trace[key][::3], label, color=color, log=True)
    save(fig, ax, "error", dict(columns=["derivative_projector_error", "aligned_F_error"], stride=3,
                                smoothing="none", log_floor=1e-18, context_scales=cfg["context_scales"]))
    fig, ax = canvas("(d) Pre-update incompatibility", "Secant energy")
    for key, label, color, style in (("preupdate_energy", "Measured", "C0", "-"),
                                    ("true_row_energy", "True-row reference", "C1", "--"),
                                    ("clean_energy", "Clean diagnostic", "C2", "-")):
        line(ax, t[::3], rolling(trace[key], round(1 / cfg["dt"]))[::3], label, style, color, log=True)
    save(fig, ax, "energy", dict(columns=["preupdate_energy", "true_row_energy", "clean_energy"], stride=3,
                                 smoothing="causal trailing mean of at most 100 samples (1 s); finite values only; startup truncated",
                                 missing_startup_samples=5, log_floor=1e-18))
    fig, ax = canvas("(e) State tracking", "State")
    line(ax, t[::3], trace["desired"][::3], "Desired", "--", "C1")
    line(ax, t[::3], trace["state"][::3], "Realized")
    save(fig, ax, "task", dict(columns=["desired", "state"], stride=3, smoothing="none"))
    fig, ax = canvas("(f) Applied input", "Input")
    line(ax, t[::3], trace["control"][::3], "Learned readout")
    save(fig, ax, "control", dict(columns=["control"], stride=3, smoothing="none"))
    return transforms


def aggregate(rows, group):
    keys = rows[0][group]
    out = {}
    for key in keys:
        values = np.array([row[group][key] for row in rows], dtype=float)
        out[key] = dict(mean=float(np.mean(values)), sample_sd=float(np.std(values, ddof=1)),
                        minimum=float(np.min(values)), maximum=float(np.max(values)), seeds=len(values))
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="New scratch analysis directory")
    parser.add_argument("--figures", type=Path, help="Optional destination for the six new panel PDFs/PNGs")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    cfg = json.loads((args.records / "config.json").read_text())
    if cfg["identity"] != "normalized_fourier_coupled_20261118_v1":
        raise ValueError("Unexpected archived protocol")
    hashes = {"config.json": sha(args.records / "config.json")}
    for path in sorted((args.records / "source_snapshot").iterdir()):
        if path.is_file():
            hashes[str(path.relative_to(args.records))] = sha(path)
    rows, seed10 = [], None
    for seed in cfg["seeds"]:
        row, flow = process_seed(args.records, seed, cfg, args.output, hashes)
        rows.append(row)
        if seed == 10:
            seed10 = flow
    if seed10 is None:
        raise ValueError("Preselected illustrative seed 10 is missing")
    figure_dir = args.output / "figures"
    figure_dir.mkdir()
    transforms = figures(seed10, cfg, figure_dir)
    rate_summary = {}
    for method in ("normalized_flow", "frozen_prior"):
        temporary = [{"metrics": row["rate_completion"][method]} for row in rows]
        rate_summary[method] = aggregate(temporary, "metrics")
    summary = dict(identity="normalized_fourier_main_postprocessing_20261119", source_protocol=cfg["identity"],
                   records=str(args.records.resolve()), seeds=cfg["seeds"], illustrative_seed=10,
                   mode="own_trajectory", evaluation_interval="60 <= t < 80 s", per_seed=rows,
                   aggregate_late_metrics=aggregate(rows, "raw_late_metrics"), current_rate_completion=rate_summary,
                   rate_completion_scope="Postprocessing of pre-update q on each recorded normalized-flow-controlled stream. Current state/input/phase and calibrated zero level determine the rate query. No measured/true rate is supplied to the readout. Actual rate is evaluator-only. Frozen prior is queried on the SAME stream; it is not a separate controller. On unguarded samples the learned forward query necessarily returns the rate requested by the same relation's inverse controller; its agreement with actual rate measures realized relation/readout error. This is not a future prediction, an independent prediction task, or an independent held-out test.",
                   rate_guard=dict(condition="abs((q*S)[:3] @ h)/norm(q) >= threshold", threshold=cfg["input_guard"],
                                   purpose="postprocessing query validity; does not alter any recorded update/controller"),
                   figure_transformations=transforms, input_sha256=hashes,
                   source_sha256=sha(__file__), command=sys.argv,
                   environment=dict(python=platform.python_version(), numpy=np.__version__, matplotlib=matplotlib.__version__),
                   source_record_policy="Inputs unchanged; no simulation or learner rerun; output directory created fresh")
    write_json(args.output / "summary.json", summary)
    macro_metrics = {"NormalizedFourierTask": summary["aggregate_late_metrics"]["tracking_rmse"],
                     "NormalizedFourierFError": summary["aggregate_late_metrics"]["aligned_F_error_mean"],
                     "NormalizedFourierProjector": summary["aggregate_late_metrics"]["derivative_projector_error_mean"],
                     "NormalizedFourierFrozenFError": summary["aggregate_late_metrics"]["frozen_aligned_F_error_mean"],
                     "NormalizedFourierForwardRMSE": rate_summary["normalized_flow"]["late_rmse"],
                     "NormalizedFourierFrozenForwardRMSE": rate_summary["frozen_prior"]["late_rmse"]}
    macro_lines = ["% Generated from all 10 archived own-trajectory runs, 60 <= t < 80 s.",
                   "% Means and sample standard deviations; current-rate query is postprocessing."]
    for name, values in macro_metrics.items():
        mean, sd = values["mean"], values["sample_sd"]
        power = int(np.floor(np.log10(abs(mean)))) if mean else 0
        if power <= -3:
            rendered = f"({mean / 10 ** power:.3f}\\pm{sd / 10 ** power:.3f})\\times10^{{{power}}}"
        else:
            decimals = max(5, min(9, int(np.ceil(-np.log10(sd))) + 1)) if sd else 5
            rendered = f"{mean:.{decimals}f}\\pm{sd:.{decimals}f}"
        macro_lines.append(f"\\newcommand{{\\{name}}}{{{rendered}}}")
    (args.output / "normalized_results_macros.tex").write_text("\n".join(macro_lines) + "\n")
    shutil.copy2(__file__, args.output / Path(__file__).name)
    manifest = {str(path.relative_to(args.output)): sha(path) for path in sorted(args.output.rglob("*")) if path.is_file()}
    write_json(args.output / "output_manifest.json", manifest)
    if args.figures:
        args.figures.mkdir(parents=True, exist_ok=True)
        for path in sorted(figure_dir.iterdir()):
            destination = args.figures / path.name
            if destination.exists():
                raise FileExistsError(destination)
            shutil.copy2(path, destination)
    print(json.dumps(dict(output=str(args.output), seeds=len(rows), panels=len(transforms),
                          late_metrics=summary["aggregate_late_metrics"], rate_completion=rate_summary), indent=2))


if __name__ == "__main__":
    main()
