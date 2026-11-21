"""Repair layout of exactly three legacy supplement panels from saved data.

No simulation runs or statistics change. The original data columns, rolling
window, decimation, rendering floors, axis scales and limits are preserved.
Output goes only to the requested empty directory; asset replacement is manual.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def rolling(y, n):
    # Identical causal finite-value mean to experiments/make_figures.py.
    y = np.asarray(y, float)
    valid = np.isfinite(y)
    sums = np.r_[0., np.cumsum(np.where(valid, y, 0.))]
    counts = np.r_[0, np.cumsum(valid)]
    end = np.arange(1, len(y) + 1)
    start = np.maximum(0, end - n)
    number = counts[end] - counts[start]
    return np.divide(sums[end] - sums[start], number,
                     out=np.full(len(y), np.nan), where=number > 0)


def fingerprint(ax):
    lines = []
    for line in ax.lines:
        x, y = np.asarray(line.get_xdata(), float), np.asarray(line.get_ydata(), float)
        lines.append(dict(label=line.get_label(), points=len(x),
                          x_sha256=hashlib.sha256(x.tobytes()).hexdigest(),
                          y_sha256=hashlib.sha256(y.tobytes()).hexdigest()))
    return dict(xlim=list(ax.get_xlim()), ylim=list(ax.get_ylim()),
                xscale=ax.get_xscale(), yscale=ax.get_yscale(), lines=lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError("Use an empty new layout-repair output directory")
    args.output.mkdir(parents=True, exist_ok=True)
    inputs = [ROOT / "results/contact/seed_10/learned/trace.csv",
              ROOT / "results/contact_excitation/seed_10/task_only/trace.csv",
              ROOT / "results/contact_excitation/seed_10/bounded_probe/trace.csv",
              ROOT / "results/contact_excitation/aggregate.json"]
    before_hashes = {str(p.relative_to(ROOT)): sha(p) for p in inputs}
    contact = np.genfromtxt(inputs[0], delimiter=",", names=True)
    paired = [np.genfromtxt(p, delimiter=",", names=True) for p in inputs[1:3]]
    aggregate = json.loads(inputs[3].read_text())
    reports = {}

    def finish(fig, ax, name, changes, tight_pad=None):
        before = fingerprint(ax)
        if name == "contact_activity":
            ax.set_title("Contact: compatibility\nand activity", fontsize=8, loc="left", pad=3)
        elif name == "excitation_energy":
            ax.set_ylabel("Pre-update\nincompatibility")
        else:
            ax.set_xlabel("Clean free-feature covariance\neigenvalue index")
        if tight_pad is None: fig.tight_layout()
        else: fig.tight_layout(pad=tight_pad)
        after = fingerprint(ax)
        if before != after:
            raise AssertionError(f"Numerical curve or axis change during layout repair: {name}")
        fig.savefig(args.output / f"{name}.pdf", bbox_inches="tight", pad_inches=.06)
        fig.savefig(args.output / f"{name}.png", dpi=180, bbox_inches="tight", pad_inches=.06)
        if fingerprint(ax) != before:
            raise AssertionError(f"Numerical curve or axis change during rendering: {name}")
        reports[name] = dict(layout_changes=changes, numerical_fingerprint=before,
                             identical_before_after=True,
                             previous_pdf_sha256=sha(ROOT / "supplement/figures" / f"{name}.pdf"),
                             output_pdf_sha256=sha(args.output / f"{name}.pdf"),
                             output_png_sha256=sha(args.output / f"{name}.png"))
        plt.close(fig)

    # Match original make_figures.canvas/line/finish settings and data exactly.
    fig = plt.figure(figsize=(2.30, 1.65))
    ax = fig.add_subplot(111)
    ax.set_title("Contact: compatibility and activity", fontsize=8, loc="left", pad=3)
    ax.set_xlabel("Time (s)", fontsize=7)
    ax.set_ylabel("Normalized quantity", fontsize=7, labelpad=2)
    ax.tick_params(labelsize=6.5, pad=2)
    ax.xaxis.set_major_locator(MaxNLocator(4))
    ax.grid(alpha=.18, linewidth=.4)
    for key, label in [("normalized_energy", "Normalized energy"), ("feature_power", "Feature power")]:
        ax.semilogy(contact["time"][::4], np.maximum(rolling(contact[key], 100)[::4], 1e-18),
                    "-", lw=.8, label=label)
    ax.axvline(60., ls="--", linewidth=.6)
    ax.legend(fontsize=6, frameon=False, ncol=1, handlelength=1.8, labelspacing=.18, borderpad=.1)
    finish(fig, ax, "contact_activity", ["Wrapped existing title at a word boundary",
           "Tight bounding box with 0.06 inch padding; all data/axis values unchanged"], tight_pad=.35)

    # Match original plot_excitation.make energy panel, including stride8/floor.
    fig, ax = plt.subplots(figsize=(3.3, 2.1))
    for data, label in zip(paired, ["Task only", "Bounded probing"]):
        ax.plot(data["time"][::8], np.maximum(data["preupdate_energy"], 1e-12)[::8], label=label, lw=.9)
    ax.axvline(60, ls="--", lw=.7, color="gray")
    ax.set_yscale("log")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Pre-update incompatibility")
    ax.legend(frameon=False, fontsize=7)
    ax.grid(alpha=.2)
    finish(fig, ax, "excitation_energy", ["Wrapped existing ylabel at a word boundary",
           "Tight bounding box with 0.06 inch padding; stride 8 and 1e-12 floor unchanged"])

    # Match the original aggregate eigenvalue panel; no trace/statistic recompute.
    fig, ax = plt.subplots(figsize=(3.3, 2.1))
    for label, protocol in [("Task only", "task_only"), ("Bounded probing", "bounded_probe")]:
        values = np.maximum(aggregate[protocol]["clean_information_eigenvalues_mean"], 1e-19)
        ax.semilogy(np.arange(1, 7), values, "o-", label=label, lw=.9)
    ax.set_xlabel("Clean free-feature covariance eigenvalue index")
    ax.set_ylabel("Eigenvalue")
    ax.legend(frameon=False, fontsize=7)
    ax.grid(alpha=.2)
    finish(fig, ax, "excitation_information", ["Wrapped existing xlabel at a word boundary",
           "Tight bounding box includes ylabel; aggregate and 1e-19 floor unchanged"])

    assert before_hashes == {str(p.relative_to(ROOT)): sha(p) for p in inputs}
    result = dict(status="passed", purpose="Layout only; three panels; no simulation/statistic changes",
                  data_sha256=before_hashes,
                  source_sha256={str(p.relative_to(ROOT)): sha(p) for p in
                                 [Path(__file__).resolve(), ROOT / "experiments/make_figures.py", ROOT / "experiments/plot_excitation.py"]},
                  matplotlib=matplotlib.__version__, numpy=np.__version__, panels=reports)
    (args.output / "figure_repair_report.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(dict(status="passed", panels=list(reports), unchanged_data_files=len(inputs))))


if __name__ == "__main__": main()
