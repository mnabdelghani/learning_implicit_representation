"""Normalized Fourier-memory diagnostic from the revised manuscript.

The observations come from f(x,y)=y+sin(x)=0. The true coefficient row is
used for evaluation only; the learner sees the lifted observed rates.
This retains the numerical algorithm of the diagnostic delivered with the
revised paper and adds a portable command-line interface.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def evaluated_partials(row: np.ndarray, phase: np.ndarray) -> np.ndarray:
    blocks = np.asarray(row).reshape(3, 2)
    return (blocks[0] + np.cos(phase)[:, None] * blocks[1]
            + np.sin(phase)[:, None] * blocks[2])


def certified_norm_bound(row: np.ndarray, nodes: int = 512) -> float:
    phase = np.arange(nodes) * (2 * np.pi / nodes)
    lipschitz = np.linalg.norm(np.asarray(row).reshape(3, 2)[1:])
    # ||-c_cos sin(t)+c_sin cos(t)|| <= ||[c_cos;c_sin]||_F.
    return float(np.linalg.norm(evaluated_partials(row, phase), axis=1).min()
                 - lipschitz * np.pi / nodes)


def run(output: Path, samples: int = 4096, iterations: int = 1600) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    phase = np.arange(samples) * (2 * np.pi / samples)
    rates = np.column_stack([np.ones(samples), -np.cos(phase)])
    lifted = np.column_stack([rates, np.cos(phase)[:, None] * rates,
                             np.sin(phase)[:, None] * rates])
    covariance = lifted.T @ lifted / samples
    target = np.array([0, 1, 1, 0, 0, 0], dtype=float) / np.sqrt(2)
    row = np.array([-0.2, 0.8, 0.6, 0.1, 0.05, -0.1], dtype=float)
    row /= np.linalg.norm(row)
    initial = row.copy()
    step = 0.5 / np.linalg.eigvalsh(covariance)[-1]
    minimum_norm = 0.1
    trace = []
    backtracks = 0

    def energy(candidate):
        return float(0.5 * np.mean((lifted @ candidate) ** 2))

    for iteration in range(iterations + 1):
        projector_error = np.linalg.norm(np.outer(row, row) - np.outer(target, target))
        trace.append((iteration, energy(row), projector_error,
                      certified_norm_bound(row), np.linalg.norm(row)))
        if iteration == iterations:
            break
        gradient = row @ covariance - (row @ covariance @ row) * row
        alpha = step
        while True:
            trial = row - alpha * gradient
            trial /= np.linalg.norm(trial)
            if (certified_norm_bound(trial) >= minimum_norm
                    and energy(trial) <= energy(row) + 1e-28):
                break
            alpha *= 0.5
            backtracks += 1
            if alpha < 1e-14:
                trial = row.copy()
                break
        row = trial

    trace = np.asarray(trace)
    # Gauge alignment below is for the evaluation plot only.
    if row @ target < 0:
        row = -row
    np.savetxt(output / "trace.csv", trace, delimiter=",", comments="",
               header="iteration,energy,projector_error,certified_norm_lower_bound,row_norm")
    summary = {
        "status": "regenerated_revised_manuscript_diagnostic",
        "system": "f(x,y)=y+sin(x)=0",
        "samples": samples, "iterations": iterations,
        "initial_row": initial.tolist(), "final_row": row.tolist(),
        "step": float(step), "eigenvalues": np.linalg.eigvalsh(covariance).tolist(),
        "initial_energy": float(trace[0, 1]), "final_energy": float(trace[-1, 1]),
        "final_projector_error": float(trace[-1, 2]),
        "minimum_certified_norm_bound": float(trace[:, 3].min()),
        "final_certified_norm_bound": float(trace[-1, 3]),
        "maximum_norm_error": float(np.max(np.abs(trace[:, 4] - 1))),
        "backtracking_steps": backtracks,
        "learning_targets": "observed lifted rates only; no coefficient or partial labels",
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    np.savez_compressed(output / "model.npz", row=row, covariance=covariance)

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), layout="constrained")
    axes[0].semilogy(trace[:, 0], np.maximum(trace[:, 1], 1e-32), label="Energy")
    axes[0].semilogy(trace[:, 0], np.maximum(trace[:, 2], 1e-32), label="Projector error")
    axes[0].set(xlabel="Gradient iteration", ylabel="Error / energy",
                title="(a) Frozen-covariance learning")
    axes[0].legend(frameon=False)
    query = np.linspace(0, 2 * np.pi, 300)
    axes[1].plot(query, np.cos(query) / np.sqrt(2), "k", label="True normalized partial")
    axes[1].plot(query, evaluated_partials(row, query)[:, 0], "--", label="Learned")
    axes[1].plot(query, evaluated_partials(initial, query)[:, 0], ":", label="Initial")
    axes[1].set(xlabel="Phase (rad)", ylabel="First differential entry",
                title="(b) Partials evaluated over a cycle")
    axes[1].legend(frameon=False, fontsize=8)
    for ax in axes:
        ax.grid(alpha=0.2)
    fig.savefig(output / "fourier.pdf", bbox_inches="tight")
    fig.savefig(output / "fourier.png", dpi=170, bbox_inches="tight")
    plt.close(fig)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("results/fourier"))
    parser.add_argument("--quick", action="store_true", help="Use 1024 samples and 250 steps")
    parser.add_argument("--seed", type=int, default=10,
                        help="Accepted for the shared runner; this diagnostic is deterministic")
    args = parser.parse_args()
    print(json.dumps(run(args.output, samples=1024 if args.quick else 4096,
                         iterations=250 if args.quick else 1600), indent=2))


if __name__ == "__main__":
    main()
