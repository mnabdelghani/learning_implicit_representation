"""New, predeclared Fourier estimator comparison; never replaces original records.

All learners consume the same observed complete-feature secants in the primary
comparison.  The original projected-memory learner controls that common stream.
EWLS solves the same box-constrained memory objective numerically; instantaneous
NLMS changes both memory and update.  Separate own-trajectory runs assess control.
"""
from __future__ import annotations

import argparse
from collections import deque
import hashlib
import json
from pathlib import Path
import platform
import shutil
import sys
import time
import traceback

import numpy as np
import scipy
from scipy.optimize import lsq_linear

from common import ImplicitLearner, dump_json, rms, save_trace
import fourier_online as fourier

METHODS = ('memory_pg', 'memory_ewls', 'instant_nlms')
SOLVER_TOL = 1e-10
SOLVER_MAX_ITER = 100


def solve_box_memory(R, lower, upper):
    """BVLS solution of min 1/2 [theta,1]' R [theta,1] on the box.

    The covariance square root preserves its objective (up to floating-point
    eigendecomposition). No statistical ridge or additional prior is introduced.
    """
    eigenvalues, eigenvectors = np.linalg.eigh((R + R.T) / 2)
    if eigenvalues[0] < -1e-12 * max(1., float(eigenvalues[-1])):
        raise ValueError('Memory covariance is materially indefinite')
    factor = np.sqrt(np.maximum(eigenvalues, 0.))[:, None] * eigenvectors.T
    solution = lsq_linear(factor[:, :-1], -factor[:, -1],
                          bounds=(lower, upper), method='bvls',
                          tol=SOLVER_TOL, max_iter=SOLVER_MAX_ITER)
    return solution.x, dict(solver_success=int(solution.success),
                           solver_status=int(solution.status),
                           solver_iterations=int(solution.nit),
                           solver_optimality=float(solution.optimality),
                           negative_covariance_roundoff=float(min(eigenvalues[0], 0.)))


class AlternativeLearner(ImplicitLearner):
    """Same channels, gauge, box, scales and covariance as original learner."""
    def __init__(self, method, *args):
        super().__init__(*args)
        if method not in ('memory_ewls', 'instant_nlms'):
            raise ValueError(method)
        self.method = method

    def observe(self, free, supplied, adapt=True):
        # Preserve preupdate score and exactly the original covariance recursion.
        diagnostics = super().observe(free, supplied, adapt=False)
        diagnostics.update(solver_success=1, solver_status=0, solver_iterations=0,
                           solver_optimality=0., negative_covariance_roundoff=0.)
        if adapt:
            if self.method == 'memory_ewls':
                self.theta, solver = solve_box_memory(self.R, self.lower, self.upper)
                diagnostics.update(solver)
            else:
                channels = self.channels(free, supplied)
                if len(channels) != 1:
                    raise ValueError('Instantaneous NLMS protocol uses one sample')
                a = channels[0, :-1]
                residual = channels[0] @ np.r_[self.theta, 1.]
                self.theta = np.clip(self.theta - a * residual / max(float(a @ a), 1e-12),
                                     self.lower, self.upper)
        q = np.r_[self.theta, 1.]
        diagnostics['memory_after'] = float(q @ self.R @ q / 2)
        return diagnostics


def make_learner(method, cfg):
    args = [np.asarray(cfg[key]) for key in ('initial', 'scales', 'lower', 'upper')]
    args.extend((cfg['dt'], cfg['memory'], int(cfg['steps'])))
    return ImplicitLearner(*args) if method == 'memory_pg' else AlternativeLearner(method, *args)


def evaluated_errors(coefficients, x, actual_rate, phi, cfg):
    """Evaluator-only derivatives in the fixed supplied-input convention."""
    truth = np.asarray(cfg['true_coefficients'])
    F = fourier.differential(x, actual_rate, phi, truth)
    estimate = fourier.differential(x, actual_rate, phi, coefficients)
    scales = np.asarray(cfg['context_scales'])
    Fs, Es = F * scales, estimate * scales
    learned = np.array([0, 1, 3])  # Exclude the supplied, fixed input partial -1.
    out = dict(F_error=float(np.linalg.norm(Es - Fs) / np.linalg.norm(Fs)),
               learned_partial_error=float(np.linalg.norm((Es - Fs)[learned]) /
                                           np.linalg.norm(Fs[learned])),
               phase_partial_error=float(estimate[3] - F[3]),
               projector_error=float(np.sqrt(max(0., 2 * (1 - (Fs @ Es) ** 2 /
                                                   ((Fs @ Fs) * (Es @ Es)))))),
               coefficient_relative_error=float(np.linalg.norm(coefficients - truth) /
                                                np.linalg.norm(truth)))
    for j in range(4):
        out[f'F_true_{j}'] = float(F[j])
        out[f'F_est_{j}'] = float(estimate[j])
    return out


def simulate(seed=10, controller='memory_pg', methods=METHODS, cfg=fourier.CONFIG):
    """Causal command -> sense -> score -> update -> advance, as original.

    Only controller/current state/reference and stored parameters determine u.
    Plant truth determines noisy measurements; clean actual_rate and truth are
    passed solely to plant/evaluator functions, never to observe().
    """
    if controller not in methods:
        raise ValueError('Controller must be among the recorded learners')
    learners = {method: make_learner(method, cfg) for method in methods}
    histories = {method: [] for method in methods}
    stream = []
    rng = np.random.default_rng(seed)
    dt = cfg['dt']
    lag = round(cfg['difference_window'] / dt)
    N = round(cfg['duration'] / dt)
    measured, clean = deque(maxlen=lag + 1), deque(maxlen=lag + 1)
    truth = np.asarray(cfg['true_coefficients'])
    x = cfg['initial_state']
    for n in range(N + 1):
        t = n * dt
        phi = cfg['phase_rate'] * t
        coefficients = {method: learner.coefficients.copy() for method, learner in learners.items()}
        rho, kappa = fourier.evaluated(coefficients[controller], phi)
        rho_true, kappa_true = fourier.evaluated(truth, phi)
        desired, desired_rate = fourier.refs(t)
        u = rho * (desired_rate - cfg['feedback_gain'] * (x - desired)) + kappa * x
        actual_rate = (u - kappa_true * x) / rho_true
        observed_rate = actual_rate + rng.normal(0, cfg['rate_noise'])
        h = fourier.harmonics(phi)
        measured.append(np.r_[h * observed_rate, h * x, -u])
        clean.append(np.r_[h * actual_rate, h * x, -u])
        stream.append(dict(time=t, state=x, desired=desired, control=u, phase=phi,
                           observed_rate=observed_rate, evaluator_actual_rate=actual_rate))
        for method, learner in learners.items():
            c = coefficients[method]
            row = dict(time=t, state=x, desired=desired, error=x - desired, control=u,
                       preupdate_energy=np.nan, clean_energy=np.nan,
                       same_stream_oracle_energy=np.nan, memory_before=np.nan,
                       memory_after=np.nan, information_min_eigenvalue=np.nan,
                       information_max_eigenvalue=np.nan, information_condition=np.nan,
                       memory_kkt_residual=np.nan, active_bounds=0, solver_success=1,
                       solver_status=0, solver_iterations=0, solver_optimality=0.,
                       negative_covariance_roundoff=0., update_seconds=0.)
            row.update(evaluated_errors(c, x, actual_rate, phi, cfg))
            for j in range(6):
                row[f'c{j}'] = float(c[j])
            if len(measured) == lag + 1:
                v = (measured[-1] - measured[0]) / cfg['difference_window']
                vc = (clean[-1] - clean[0]) / cfg['difference_window']
                row['preupdate_energy'] = float((v @ np.r_[c, 1.]) ** 2 / 2)
                row['clean_energy'] = float((vc @ np.r_[c, 1.]) ** 2 / 2)
                row['same_stream_oracle_energy'] = float((v @ np.r_[truth, 1.]) ** 2 / 2)
                started = time.perf_counter()
                diagnostics = learner.observe(v[:6], v[-1], adapt=n < N)
                row['update_seconds'] = time.perf_counter() - started
                row.update({key: value for key, value in diagnostics.items() if key in row})
                eigenvalues = np.linalg.eigvalsh(learner.R[:-1, :-1])
                row['information_min_eigenvalue'] = float(eigenvalues[0])
                row['information_max_eigenvalue'] = float(eigenvalues[-1])
                row['information_condition'] = float(eigenvalues[-1] / eigenvalues[0]) if eigenvalues[0] > 0 else np.inf
                gradient = learner.R[:-1, :-1] @ learner.theta + learner.R[:-1, -1]
                row['memory_kkt_residual'] = float(np.max(np.abs(learner.theta -
                    np.clip(learner.theta - gradient, learner.lower, learner.upper))))
                row['active_bounds'] = int(np.count_nonzero(
                    (np.abs(learner.theta - learner.lower) < 1e-8) |
                    (np.abs(learner.theta - learner.upper) < 1e-8)))
            histories[method].append(row)
        if n < N:
            x = fourier.rk4(x, u, t, dt, truth, cfg['phase_rate'])
    return histories, stream


def summarize(rows, cfg):
    late = [row for row in rows if cfg['duration'] - 20 <= row['time'] < cfg['duration']]
    observed = [row for row in rows if np.isfinite(row['preupdate_energy'])]
    means = ('F_error', 'learned_partial_error', 'projector_error', 'coefficient_relative_error',
             'preupdate_energy', 'clean_energy', 'same_stream_oracle_energy',
             'information_min_eigenvalue', 'information_condition', 'active_bounds')
    out = {key + '_mean': float(np.nanmean([row[key] for row in late])) for key in means}
    out.update(tracking_rmse=rms([row['error'] for row in late]),
               control_rms=rms([row['control'] for row in late]),
               phase_partial_rmse=rms([row['phase_partial_error'] for row in late]),
               solver_failure_count=sum(row['solver_success'] == 0 for row in observed),
               max_solver_optimality=max(row['solver_optimality'] for row in observed),
               max_memory_kkt_residual=max(row['memory_kkt_residual'] for row in observed),
               min_memory_eigenvalue=min(row['information_min_eigenvalue'] for row in observed),
               update_seconds_total=sum(row['update_seconds'] for row in observed),
               final_coefficients=[rows[-1][f'c{j}'] for j in range(6)])
    return out


def save_run(output, seed, controller, methods, mode, cfg):
    histories, stream = simulate(seed, controller, methods, cfg)
    folder = output / mode / f'seed_{seed}'
    stream_folder = folder if mode == 'common_stream' else folder / controller
    save_trace(stream_folder / 'observations.csv', stream)
    summaries = []
    for method, rows in histories.items():
        destination = folder / method
        save_trace(destination / 'trace.csv', rows)
        summary = dict(seed=seed, method=method, controller=controller, mode=mode,
                       metrics=summarize(rows, cfg))
        dump_json(destination / 'summary.json', summary)
        summaries.append(summary)
    return summaries


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--protocol', required=True, type=Path)
    parser.add_argument('--seeds', nargs='+', type=int, default=list(range(10, 20)))
    parser.add_argument('--duration', type=float, default=80.)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Use a new output directory; existing records are never overwritten')
    args.output.mkdir(parents=True)
    shutil.copyfile(args.protocol, args.output / 'predeclared_protocol.json')
    cfg = dict(fourier.CONFIG, duration=args.duration)
    dump_json(args.output / 'config.json', cfg)
    source_dir = Path(__file__).parent
    sources = {name: hashlib.sha256((source_dir / name).read_bytes()).hexdigest()
               for name in ('estimator_comparison.py', 'fourier_online.py', 'common.py')}
    snapshot = args.output / 'source_snapshot'
    snapshot.mkdir()
    for name in sources:
        shutil.copyfile(source_dir / name, snapshot / name)
    dump_json(args.output / 'environment.json', dict(python=sys.version, numpy=np.__version__,
              scipy=scipy.__version__, platform=platform.platform(), argv=sys.argv,
              source_sha256=sources, seeds=args.seeds, worker_count=1))
    all_summaries, failures = [], []
    for seed in args.seeds:
        branches = [('common_stream', 'memory_pg', METHODS),
                    ('own_trajectory', 'memory_ewls', ('memory_ewls',)),
                    ('own_trajectory', 'instant_nlms', ('instant_nlms',))]
        for mode, controller, methods in branches:
            try:
                summaries = save_run(args.output, seed, controller, methods, mode, cfg)
                all_summaries.extend(summaries)
                print(seed, mode, controller, 'saved', flush=True)
            except Exception:
                failure = dict(seed=seed, mode=mode, controller=controller,
                               traceback=traceback.format_exc())
                failures.append(failure)
                print(json.dumps(failure), flush=True)
            dump_json(args.output / 'summaries.json', dict(runs=all_summaries, failures=failures))
    groups = {}
    for run in all_summaries:
        key = run['mode'] + '/' + run['method']
        groups.setdefault(key, []).append(run)
    aggregate = {}
    for key, group in groups.items():
        metrics = {}
        for metric, value in group[0]['metrics'].items():
            if isinstance(value, list):
                continue
            values = [run['metrics'][metric] for run in group]
            metrics[metric] = dict(mean=float(np.mean(values)),
                                  sample_sd=float(np.std(values, ddof=1)) if len(values) > 1 else None)
        aggregate[key] = dict(seeds=[run['seed'] for run in group], metrics=metrics)
    dump_json(args.output / 'aggregate.json', dict(groups=aggregate, failures=failures))
    raise SystemExit(1 if failures else 0)


if __name__ == '__main__':
    main()
