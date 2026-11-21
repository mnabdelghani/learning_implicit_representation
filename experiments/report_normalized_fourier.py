"""Postprocess all predeclared normalized Fourier records and refine seed 10.

This does not rerun the closed-loop experiments. The integration refinement
replays the archived seed-10 input stream with tighter solver tolerances. It
checks accumulated numerical error on that fixed stream, not closed-loop error.
Figures are review artifacts; no manuscript or published figures are replaced.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import platform
import shutil
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy

METHODS = ("normalized_flow", "bottom_eigenvector", "frozen_prior")
LABELS = ("Normalized flow", "Memory minimum", "Frozen prior")
COLORS = ("#2166ac", "#d95f02", "#7570b3")
MODES = ("replay", "own_trajectory")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def serializable(value):
    if isinstance(value, dict):
        return {str(k): serializable(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [serializable(v) for v in value]
    if isinstance(value, np.ndarray):
        return serializable(value.tolist())
    if isinstance(value, np.generic):
        return serializable(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def write_json(path, value):
    path.write_text(json.dumps(serializable(value), indent=2, allow_nan=False) + "\n")


def finite_mean(values):
    values = np.asarray(values)
    return float(np.mean(values[np.isfinite(values)])) if np.any(np.isfinite(values)) else np.nan


def metrics(record, cfg, control):
    """Recalculate metrics from pre-update trace rows, without runner summaries."""
    t = record["time"]
    late = (t >= cfg["evaluation_window"][0]) & (t < cfg["evaluation_window"][1])
    if np.count_nonzero(late) != round(np.diff(cfg["evaluation_window"])[0] / cfg["dt"]):
        raise ValueError("Incomplete evaluation window")
    keys = ("derivative_projector_error", "coefficient_projector_error", "aligned_F_error",
            "preupdate_energy", "clean_energy", "true_row_energy", "coefficient_to_memory_error",
            "memory_gap", "normalized_gradient_norm")
    result = {key + "_mean": finite_mean(record[key][late]) for key in keys}
    result["phase_partial_rmse"] = float(np.sqrt(finite_mean(record["phase_partial_error"][late] ** 2)))
    excess = record["memory_energy"] - record["memory_lambda_min"] / 2
    result.update(memory_objective_excess_mean=finite_mean(excess[late]),
                  memory_objective_excess_min=float(np.min(excess)),
                  min_abs_input_coefficient=float(np.min(np.abs(record["input_coefficient"]))),
                  readout_guard_count=int(np.sum(record["readout_guarded"])),
                  max_norm_error=float(np.max(np.abs(record["coefficient_norm"] - 1))),
                  min_row_norm=float(np.min(record["row_norm"])))
    if control:
        result.update(tracking_rmse=float(np.sqrt(np.mean(record["error"][late] ** 2))),
                      control_rms=float(np.sqrt(np.mean(record["control"][late] ** 2))),
                      controller_guard_count=int(np.sum(record["controller_guarded"])))
    counts = {key: {"all_nonfinite": int(np.count_nonzero(~np.isfinite(record[key]))),
                    "evaluation_nonfinite": int(np.count_nonzero(~np.isfinite(record[key][late])))}
              for key in (*keys, "phase_partial_error")}
    return dict(metrics=result, nonfinite_counts=counts,
                evaluation_samples=int(np.count_nonzero(late)),
                final_q=[float(record[f"q{j}"][-1]) for j in range(7)])


def aggregate(per_seed, seeds):
    result = {}
    for mode in MODES:
        for method in METHODS:
            rows = [r for r in per_seed if r["mode"] == mode and r["method"] == method]
            group = dict(expected_seeds=seeds, completed_seeds=[r["seed"] for r in rows], metrics={})
            for key in rows[0]["metrics"] if rows else ():
                values = np.asarray([r["metrics"][key] for r in rows], float)
                finite = values[np.isfinite(values)]
                group["metrics"][key] = dict(mean=float(np.mean(finite)) if len(finite) else np.nan,
                                            sample_sd=float(np.std(finite, ddof=1)) if len(finite) > 1 else np.nan,
                                            minimum=float(np.min(finite)) if len(finite) else np.nan,
                                            maximum=float(np.max(finite)) if len(finite) else np.nan,
                                            finite_seed_count=len(finite),
                                            nonfinite_seed_count=len(values) - len(finite))
            result[f"{mode}/{method}"] = group
    return result


def projector(a, b):
    a, b = a / np.linalg.norm(a), b / np.linalg.norm(b)
    return float(np.linalg.norm(np.outer(a, a) - np.outer(b, b)))


def refinement(raw, output, cfg, record, track_input):
    """Independent orchestration of every held sample with the frozen runner."""
    source = raw / "source_snapshot" / "normalized_fourier.py"
    sys.dont_write_bytecode = True  # The archived source directory is immutable evidence.
    sys.path.insert(0, str(source.parent))
    spec = importlib.util.spec_from_file_location("archived_normalized_fourier", source)
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    directory = raw / "replay" / "seed_10"
    state_path, observations_path = directory / "state.npz", directory / "observations.csv"
    track_input(state_path)
    track_input(observations_path)
    # NPZ is a compressed lazy archive: cache arrays once rather than expanding
    # the complete matrix history at every individual time index.
    with np.load(state_path) as archive:
        arrays = {name: archive[name] for name in ("R_pre", "w")}
    observations = np.genfromtxt(observations_path, delimiter=",", names=True)
    q, R = runner.initial_q(cfg), np.zeros((7, 7))
    rows = []
    total_nfev = 0
    maximum_R_difference = 0.
    for n, observation in enumerate(observations):
        old = np.array([record[f"q{j}"][n] for j in range(7)])
        u_refined, guard_refined = runner.command(q, observation["state"], observation["time"], cfg)
        u_original, guard_original = runner.command(old, observation["state"], observation["time"], cfg)
        z = [observation["state"], observation["evaluator_actual_rate"], observation["control"], observation["phase"]]
        F_refined = runner.differential(z, q, np.asarray(cfg["scales"]))
        F_original = runner.differential(z, old, np.asarray(cfg["scales"]))
        context = np.asarray(cfg["context_scales"])
        maximum_R_difference = max(maximum_R_difference, float(np.max(np.abs(R - arrays["R_pre"][n]))))
        rows.append(dict(time=float(observation["time"]), coefficient_projector_difference=projector(q, old),
                         derivative_projector_difference=projector(F_refined * context, F_original * context),
                         learned_command_difference=float(u_refined - u_original),
                         guard_disagreement=int(guard_refined != guard_original),
                         refined_q=q.copy()))
        w = arrays["w"][n]
        if np.all(np.isfinite(w)) and n < len(observations) - 1:
            q, R, nfev = runner.coupled_step(q, R, w, cfg["dt"], cfg["memory"], cfg["gain"],
                                           rtol=1e-11, atol=1e-13, max_step=.005)
            total_nfev += nfev
    np.savez_compressed(output / "seed10_refinement.npz", **{key: np.array([r[key] for r in rows]) for key in rows[0]})
    result = dict(scenario="replay", seed=10, samples=len(rows), rtol=1e-11, atol=1e-13,
                  max_step=.005, total_nfev=total_nfev, maximum_R_difference=maximum_R_difference,
                  guard_disagreements=sum(r["guard_disagreement"] for r in rows),
                  scope="All held samples on the archived PG-controller observation stream. The learned command is a counterfactual readout at those same states; this is not a refined closed-loop plant trajectory.")
    for key in ("coefficient_projector_difference", "derivative_projector_difference", "learned_command_difference"):
        values = np.array([r[key] for r in rows])
        result[key] = dict(max_abs=float(np.max(np.abs(values))), rms=float(np.sqrt(np.mean(values ** 2))),
                           final=float(values[-1]))
    return result


def binned(record, key, cfg, rms=False):
    # Nonoverlapping one-second bins, excluding the final scored-only sample.
    width = round(1 / cfg["dt"])
    values = record[key][:-1].reshape(-1, width)
    if rms:
        values = values ** 2
    count = np.sum(np.isfinite(values), axis=1)
    means = np.divide(np.nansum(values, axis=1), count, out=np.full(len(values), np.nan), where=count > 0)
    return np.sqrt(means) if rms else means


def figures(records, output, cfg):
    plt.rcParams.update({"font.size": 9, "axes.labelsize": 9, "legend.fontsize": 8,
                         "pdf.fonttype": 42, "ps.fonttype": 42})
    fig, axes = plt.subplots(4, 2, figsize=(9.0, 10.3), constrained_layout=True)
    fields = (r"$\partial f/\partial x$", r"$\partial f/\partial \dot{x}$",
              r"$\partial f/\partial u$", r"$\partial f/\partial \phi$")
    replay = [records[("replay", 10, method)] for method in METHODS]
    t = replay[0]["time"]
    selected = (t >= 60) & (t < 65)
    for j, ax in enumerate(axes[:2].ravel()):
        for record, label, color in zip(replay, LABELS, COLORS):
            ax.plot(t[selected], record[f"F_aligned_{j}"][selected], label=label, color=color, lw=1.1)
        ax.plot(t[selected], replay[0][f"F_true_{j}"][selected], "k--", label="Evaluator truth", lw=1.3)
        ax.set_ylabel(fields[j])
        ax.set_xlabel("Time (s)")
        ax.grid(alpha=.2)
    axes[0, 0].set_title("Seed 10 replay: input-gauge-aligned partials", loc="left", fontsize=9)
    axes[1, 0].set_title("Input partial = -1 by gauge alignment", loc="left", fontsize=9)
    for record, label, color in zip(replay, LABELS, COLORS):
        axes[2, 0].plot(t[::10], record["input_coefficient"][::10], color=color, label=label, lw=1.1)
        axes[2, 1].plot(t[::10], record["derivative_projector_error"][::10], color=color, lw=1.1)
    target = np.r_[cfg["true_coefficients"], 1.] / np.asarray(cfg["scales"])
    target /= np.linalg.norm(target)
    axes[2, 0].axhline(target[-1] * cfg["scales"][-1], color="black", linestyle="--", lw=1.3)
    axes[2, 0].set_ylabel("Learned raw input coefficient")
    axes[2, 1].set_ylabel("Differential projector error")
    own = records[("own_trajectory", 10, "normalized_flow")]
    axes[3, 0].plot(t[::10], own["state"][::10], color=COLORS[0], label="Realized state", lw=1.)
    axes[3, 0].plot(t[::10], own["desired"][::10], "k--", label="Reference", lw=.9)
    axes[3, 0].set_ylabel("Own-trajectory state")
    axes[3, 0].legend(frameon=False)
    axes[3, 1].plot(t[::10], own["control"][::10], color=COLORS[0], lw=.9)
    axes[3, 1].set_ylabel("Own-trajectory input")
    for ax in axes[2:].ravel():
        ax.set_xlabel("Time (s)")
        ax.grid(alpha=.2)
    fig.legend(*axes[0, 0].get_legend_handles_labels(), loc="outside upper center",
               ncol=4, frameon=False,
               title="Normalized Fourier learning: recorded pre-update estimates", title_fontsize=11)
    for suffix in ("pdf", "png"):
        fig.savefig(output / f"normalized_fourier_partials.{suffix}", dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(4, 2, figsize=(9., 10.3), constrained_layout=True, sharey="row")
    panel_keys = ("derivative_projector_error", "preupdate_energy", "phase_partial_error", "coefficient_to_memory_error")
    labels = ("Differential projector error", "Pre-update secant energy", "Aligned phase-partial RMSE", "Coefficient-to-memory\nprojector distance")
    for col, mode in enumerate(MODES):
        axes[0, col].set_title("Same archived PG stream" if mode == "replay" else "Normalized-flow plant stream", fontsize=10)
        for method, label, color in zip(METHODS, LABELS, COLORS):
            traces = [records[(mode, seed, method)] for seed in cfg["seeds"]]
            for row, key in enumerate(panel_keys):
                values = np.stack([binned(record, key, cfg, rms=(key == "phase_partial_error")) for record in traces])
                mean, sd = np.mean(values, axis=0), np.std(values, axis=0, ddof=1)
                centers = np.arange(len(mean)) + .5
                axes[row, col].plot(centers, mean, label=label, color=color, lw=1.3)
                floor = 1e-14 if row == 1 else 0.
                axes[row, col].fill_between(centers, np.maximum(mean - sd, floor), mean + sd,
                                            color=color, alpha=.18, linewidth=0)
        truth = np.stack([binned(records[(mode, seed, "normalized_flow")], "true_row_energy", cfg) for seed in cfg["seeds"]])
        axes[1, col].plot(np.arange(truth.shape[1]) + .5, np.mean(truth, axis=0), "k--", label="True-row reference", lw=1.)
        axes[1, col].set_yscale("log")
        for row in range(4):
            axes[row, col].set_xlabel("Time (s)")
            axes[row, col].set_ylabel(labels[row])
            axes[row, col].grid(alpha=.2)
    fig.legend(*axes[1, 0].get_legend_handles_labels(), loc="outside upper center",
               ncol=4, frameon=False,
               title="All 10 seeds: one-second means and across-seed sample SD", title_fontsize=11)
    for suffix in ("pdf", "png"):
        fig.savefig(output / f"normalized_fourier_summary.{suffix}", dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    raw, output = args.input.resolve(), args.output.resolve()
    if output.exists():
        raise FileExistsError("Existing analysis output is never overwritten")
    manifest_path = raw / "file_manifest.json"
    manifest = json.loads(manifest_path.read_text())  # Presence means runner finished all planned attempts.
    inputs = {"file_manifest.json": sha(manifest_path)}

    def track_input(path):
        relative = str(path.relative_to(raw))
        digest = sha(path)
        if manifest.get(relative) != digest:
            raise ValueError(f"Raw-manifest mismatch: {relative}")
        inputs[relative] = digest

    for name in ("config.json", "protocol.json", "environment.json", "summary.json"):
        track_input(raw / name)
    cfg = json.loads((raw / "config.json").read_text())
    summary = json.loads((raw / "summary.json").read_text())
    output.mkdir(parents=True)
    records, per_seed, missing = {}, [], []
    for mode in MODES:
        for seed in cfg["seeds"]:
            for method in METHODS:
                path = raw / mode / f"seed_{seed}" / method / "trace.csv"
                if not path.is_file():
                    missing.append(dict(mode=mode, seed=seed, method=method))
                    continue
                track_input(path)
                record = np.genfromtxt(path, delimiter=",", names=True)
                records[(mode, seed, method)] = record
                per_seed.append(dict(mode=mode, seed=seed, method=method,
                                     **metrics(record, cfg, mode == "own_trajectory" and method == "normalized_flow")))
    report = dict(protocol_identity=cfg["identity"], evaluation_window=cfg["evaluation_window"],
                  planned_streams=len(MODES) * len(cfg["seeds"]), completed_estimator_traces=len(records),
                  failures=summary["failures"], missing=missing, per_seed=per_seed,
                  groups=aggregate(per_seed, cfg["seeds"]),
                  conventions=["Each seed first produces a metric from unsmoothed pre-update samples; group SD uses ddof=1.",
                               "Memory minimum is the bottom eigenvector of the same scaled unnormalized covariance; objective excess is q R q / 2 - lambda_min(R) / 2.",
                               "Memory and derivative projector errors have different coordinate spaces. A small noisy-memory objective does not prove physical identification.",
                               "Physical partials divide by the learned input coefficient; that aligned input partial is -1 by convention, not independent evidence of learning.",
                               "Guard counts include the final scored-only sample. Only the normalized-flow own-trajectory method is credited with control metrics.",
                               "Nonfinite counts are explicit; finite-only diagnostic means do not hide excluded seeds. Startup energy is undefined before a complete 0.05 s window."])
    if not missing:
        for path in sorted((raw / "source_snapshot").iterdir()):
            if path.is_file():
                track_input(path)
        report["integration_refinement"] = refinement(raw, output, cfg, records[("replay", 10, "normalized_flow")], track_input)
        figures(records, output, cfg)
    write_json(output / "report.json", report)
    plotting = dict(seed_figure="Seed 10 was preselected. Four replay partials use every sample in [60,65) s; remaining full-duration panels show every 10th sample (0.1 s), without smoothing.",
                    aggregate_figure="Each 1 s bin first averages the available finite samples within each seed (phase error uses RMS); lines and bands are mean +/- sample SD across all 10 seeds. Final sample at 80 s is excluded. Negative lower bands are truncated at zero; energy uses a 1e-14 visual floor. Metrics use unsmoothed data.",
                    panels=dict(seed10=["replay F_x aligned", "replay F_rate aligned", "replay F_u aligned (identically -1 by gauge alignment)", "replay F_phase aligned", "raw learned input coefficient and positive unit-normalized true coefficient reference", "replay derivative projector", "own state/reference", "own control"],
                                all_seeds="Rows: derivative projector, measured energy plus true-row reference, aligned phase RMSE, coefficient-to-memory distance. Columns: replay and own-trajectory streams. Frozen/minimum estimators are shadows on the own stream."),
                    limitations="Input-gauge-aligned F_u is identically -1 by construction when valid. Memory-minimum coefficients and low energy do not certify physical excitation. Refinement is on a fixed stream, not a new plant run. These experiments do not repair the contact or Burgers mechanism gap.")
    write_json(output / "figure_provenance.json", plotting)
    shutil.copyfile(Path(__file__), output / "report_normalized_fourier.py")
    write_json(output / "provenance.json", dict(command=sys.argv, python=sys.version, numpy=np.__version__,
               scipy=scipy.__version__, matplotlib=matplotlib.__version__, platform=platform.platform(),
               raw_root=str(raw), raw_sha256=inputs, script_sha256=sha(__file__)))
    write_json(output / "file_manifest.json", {str(p.relative_to(output)): sha(p) for p in sorted(output.rglob("*")) if p.is_file()})
    print(json.dumps(serializable(dict(completed_estimator_traces=len(records), missing=missing,
                                      failures=summary["failures"], integration_refinement=report.get("integration_refinement")))))


if __name__ == "__main__":
    main()
