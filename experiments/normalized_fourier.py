"""Coupled normalized learning in the original Fourier plant, with a new identity.

Every fixed candidate is the Jacobian of an integrable seven-coefficient family.
The input coefficient is learned too. Complete-feature secants include jumps
of the held command. They are not instantaneous raw-context rates.
"""
from __future__ import annotations

import argparse
from collections import deque
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import traceback

import numpy as np
import scipy
from scipy.integrate import solve_ivp

from common import dump_json, save_trace
import fourier_online as fourier

PROTOCOL_PATH = Path(__file__).with_name('configs') / 'normalized_fourier_20261118.json'
METHODS = ('normalized_flow', 'bottom_eigenvector', 'frozen_prior')


class SimulationFailure(RuntimeError):
    """Retains all observations and complete state transitions before failure."""


def features(z):
    x, rate, u, phi = np.asarray(z)
    h = fourier.harmonics(phi)
    return np.r_[h * rate, h * x, -u]


def residual(z, q, scales):
    return float((q * scales) @ features(z))


def differential(z, q, scales):
    x, rate, _, phi = np.asarray(z)
    a = q * scales
    h, hp = fourier.harmonics(phi), fourier.harmonic_rates(phi)
    return np.array([a[3:6] @ h, a[:3] @ h, -a[6],
                     (a[:3] @ hp) * rate + (a[3:6] @ hp) * x])


def projector_distance(a, b):
    """Stable rank-one Frobenius projector distance, in declared coordinates."""
    a, b = np.asarray(a), np.asarray(b)
    if np.linalg.norm(a) == 0 or np.linalg.norm(b) == 0:
        return float('nan')
    a, b = a / np.linalg.norm(a), b / np.linalg.norm(b)
    return float(np.linalg.norm(np.outer(a, a) - np.outer(b, b)))


def initial_q(cfg):
    q = np.asarray(cfg['initial_homogeneous'], float) / np.asarray(cfg['scales'])
    return q / np.linalg.norm(q)


def memory_at(R, w, elapsed, memory):
    beta = np.exp(-elapsed / memory)
    return beta * R + (-memory * np.expm1(-elapsed / memory)) * np.outer(w, w)


def coupled_step(q, R, w, dt, memory, gain, rtol=1e-9, atol=1e-11, max_step=np.inf):
    """Integrate the actual coupled law for a causal zero-order-held secant.

    If y'=-gain*R(s)y, normalization q=y/||y|| obeys exactly
    q'=-gain*(Rq-(q.T R q)q). R(s) is analytic, not held fixed here.
    Numerical integration uses DOP853; normalization removes radial roundoff.
    """
    outer = np.outer(w, w)

    def rhs(s, y):
        beta = np.exp(-s / memory)
        local_R = beta * R + (-memory * np.expm1(-s / memory)) * outer
        return -gain * (local_R @ y)

    solved = solve_ivp(rhs, (0., dt), np.asarray(q), method='DOP853',
                       rtol=rtol, atol=atol, max_step=max_step)
    if not solved.success:
        raise RuntimeError(solved.message)
    y = solved.y[:, -1]
    length = np.linalg.norm(y)
    if not np.isfinite(length) or length <= np.finfo(float).tiny:
        raise FloatingPointError('Unnormalizable integration result')
    return y / length, memory_at(R, w, dt, memory), int(solved.nfev)


def bottom_estimate(R, previous, tolerance=1e-12):
    values, vectors = np.linalg.eigh((R + R.T) / 2)
    tied = values <= values[0] + tolerance * max(1., abs(values[-1]))
    U = vectors[:, tied]
    estimate = U @ (U.T @ previous)
    if np.linalg.norm(estimate) < 1e-12:
        # Basis-independent deterministic choice from the minimizing subspace.
        j = int(np.argmax(np.sum(U * U, axis=1)))
        estimate = U @ U[j, :]
    estimate /= np.linalg.norm(estimate)
    if estimate @ previous < 0:
        estimate = -estimate
    return estimate, values, int(np.count_nonzero(tied))


def command(q, state, t, cfg):
    """Only current learned q, state, known clock and reference enter control."""
    scales = np.asarray(cfg['scales'])
    a = q * scales
    guarded = np.linalg.norm(q) == 0 or abs(a[-1]) / np.linalg.norm(q) < cfg['input_guard']
    if guarded:
        c = np.asarray(cfg['initial_homogeneous'][:6]) / cfg['initial_homogeneous'][6]
    else:
        c = a[:6] / a[6]
    rho, kappa = fourier.evaluated(c, cfg['phase_rate'] * t)
    desired, desired_rate = fourier.refs(t)
    u = rho * (desired_rate - cfg['feedback_gain'] * (state - desired)) + kappa * state
    return float(u), bool(guarded)


def evaluate_candidate(q, state, actual_rate, u, phi, cfg):
    """Evaluator-only truth/clean channels; returned values never affect control."""
    scales = np.asarray(cfg['scales'])
    z = [state, actual_rate, u, phi]
    truth = np.asarray(cfg['true_coefficients'])
    F = fourier.differential(state, actual_rate, phi, truth)
    estimate = differential(z, q, scales)
    context = np.asarray(cfg['context_scales'])
    target = np.r_[truth, 1.] / scales
    target /= np.linalg.norm(target)
    a_u = q[-1] * scales[-1]
    aligned = estimate / a_u if abs(a_u) >= cfg['input_guard'] else np.full(4, np.nan)
    out = dict(derivative_projector_error=projector_distance(estimate * context, F * context),
               coefficient_projector_error=projector_distance(q, target),
               aligned_F_error=float(np.linalg.norm((aligned - F) * context) / np.linalg.norm(F * context)),
               phase_partial_error=float(aligned[-1] - F[-1]),
               input_coefficient=float(a_u), row_norm=float(np.linalg.norm(estimate)))
    for j in range(4):
        out[f'F_est_{j}'] = float(estimate[j])
        out[f'F_true_{j}'] = float(F[j])
        out[f'F_aligned_{j}'] = float(aligned[j])
    return out


def read_stream(path):
    with Path(path).open(newline='') as handle:
        return [{key: float(value) for key, value in row.items()}
                for row in csv.DictReader(handle)]


def simulate(seed, cfg, replay=None):
    """Command -> sense -> score -> integrate learner -> advance held-input plant.

    Replay uses only saved state/input/phase/noisy-rate channels in the learner.
    The separately named clean rate is passed exclusively to the evaluator.
    """
    dt, memory = cfg['dt'], cfg['memory']
    N = round(cfg['duration'] / dt)
    lag = round(cfg['difference_window'] / dt)
    if not np.isclose(lag * dt, cfg['difference_window']):
        raise ValueError('Difference window must be an integer number of samples')
    if replay is not None and len(replay) != N + 1:
        raise ValueError('Replay stream length does not match protocol')
    q0 = initial_q(cfg)
    candidates = {name: q0.copy() for name in METHODS}
    R = np.zeros((7, 7))
    scales = np.asarray(cfg['scales'])
    rng = np.random.default_rng(seed)
    rng_initial = rng.bit_generator.state
    truth = np.asarray(cfg['true_coefficients'])
    measured, clean = deque(maxlen=lag + 1), deque(maxlen=lag + 1)
    histories, observations = {name: [] for name in METHODS}, []
    arrays = {key: [] for key in ('R_pre', 'R_post', 'w', 'q_pre', 'q_post', 'nfev')}
    x = cfg['initial_state']
    try:
        for n in range(N + 1):
            t = n * dt
            phi = cfg['phase_rate'] * t
            if replay is None:
                u, control_guarded = command(candidates['normalized_flow'], x, t, cfg)
                rho_true, kappa_true = fourier.evaluated(truth, phi)
                actual_rate = (u - kappa_true * x) / rho_true
                observed_rate = actual_rate + rng.normal(0., cfg['rate_noise'])
                desired = fourier.refs(t)[0]
            else:
                sample = replay[n]
                if not np.isclose(sample['time'], t, atol=1e-12, rtol=0.) or not np.isclose(sample['phase'], phi):
                    raise ValueError('Replay time/phase differs from protocol')
                x, u = sample['state'], sample['control']
                observed_rate, actual_rate = sample['observed_rate'], sample['evaluator_actual_rate']
                desired = sample['desired']
                control_guarded = False  # Source controller is the original PG learner.
            if not np.all(np.isfinite([x, u, observed_rate])):
                raise FloatingPointError(f'Nonfinite plant/observation at sample {n}')
            observations.append(dict(time=t, state=x, desired=desired, control=u, phase=phi,
                                     observed_rate=observed_rate, evaluator_actual_rate=actual_rate))
            measured.append(features([x, observed_rate, u, phi]))
            clean.append(features([x, actual_rate, u, phi]))
            valid = len(measured) == lag + 1
            w = scales * (measured[-1] - measured[0]) / cfg['difference_window'] if valid else np.full(7, np.nan)
            wc = scales * (clean[-1] - clean[0]) / cfg['difference_window'] if valid else np.full(7, np.nan)
            before = np.array([candidates[name].copy() for name in METHODS])
            R_before = R.copy()
            target = np.r_[truth, 1.] / scales
            target /= np.linalg.norm(target)
            eig, U = np.linalg.eigh(R)
            for name in METHODS:
                q = candidates[name]
                gradient = R @ q - (q @ R @ q) * q
                row = dict(time=t, state=x, desired=desired, error=x - desired, control=u,
                           readout_guarded=int(abs(q[-1] * scales[-1]) < cfg['input_guard']),
                           controller_guarded=int(control_guarded), coefficient_norm=float(np.linalg.norm(q)),
                           preupdate_energy=float((q @ w) ** 2 / 2), clean_energy=float((q @ wc) ** 2 / 2),
                           true_row_energy=float((target @ w) ** 2 / 2), memory_energy=float(q @ R @ q / 2),
                           memory_gap=float(eig[1] - eig[0]), memory_lambda_min=float(eig[0]),
                           memory_lambda_max=float(eig[-1]), normalized_gradient_norm=float(np.linalg.norm(gradient)),
                           flow_dissipation=-cfg['gain'] * float(gradient @ gradient),
                           coefficient_to_memory_error=projector_distance(q, U[:, 0]) if eig[1] - eig[0] > cfg['eigen_tie_relative_tolerance'] * max(1., abs(eig[-1])) else np.nan)
                row.update(evaluate_candidate(q, x, actual_rate, u, phi, cfg))
                row.update({f'q{j}': float(q[j]) for j in range(7)})
                histories[name].append(row)
            nfev = 0
            if valid and n < N:
                candidates['normalized_flow'], R, nfev = coupled_step(
                    candidates['normalized_flow'], R, w, dt, memory, cfg['gain'], cfg['rtol'], cfg['atol'])
                candidates['bottom_eigenvector'], _, _ = bottom_estimate(
                    R, candidates['bottom_eigenvector'], cfg['eigen_tie_relative_tolerance'])
            for key, value in dict(R_pre=R_before, R_post=R.copy(), w=w, q_pre=before,
                                   q_post=np.array([candidates[name].copy() for name in METHODS]), nfev=nfev).items():
                arrays[key].append(value)
            if replay is None and n < N:
                x = fourier.rk4(x, u, t, dt, truth, cfg['phase_rate'])
    except Exception as exc:
        failure = SimulationFailure(str(exc))
        failure.partial = (histories, observations,
                           {key: np.asarray(value) for key, value in arrays.items()},
                           dict(initial=rng_initial, final=rng.bit_generator.state, used=replay is None))
        raise failure from exc
    arrays = {key: np.asarray(value) for key, value in arrays.items()}
    return histories, observations, arrays, dict(initial=rng_initial, final=rng.bit_generator.state,
                                                used=replay is None)


def summarize(rows, cfg, own_trajectory):
    lo, hi = cfg['evaluation_window']
    late = [r for r in rows if lo <= r['time'] < hi]
    metrics = ('derivative_projector_error', 'coefficient_projector_error', 'aligned_F_error',
               'preupdate_energy', 'clean_energy', 'true_row_energy', 'coefficient_to_memory_error',
               'memory_gap')
    out = {key + '_mean': float(np.mean([row[key] for row in late])) for key in metrics}
    out.update(phase_partial_rmse=float(np.sqrt(np.mean([row['phase_partial_error'] ** 2 for row in late]))),
               min_abs_input_coefficient=min(abs(row['input_coefficient']) for row in rows),
               readout_guard_count=sum(row['readout_guarded'] for row in rows),
               max_norm_error=max(abs(row['coefficient_norm'] - 1.) for row in rows),
               min_row_norm=min(row['row_norm'] for row in rows),
               final_q=[rows[-1][f'q{j}'] for j in range(7)])
    if own_trajectory:
        out.update(tracking_rmse=float(np.sqrt(np.mean([row['error'] ** 2 for row in late]))),
                   control_rms=float(np.sqrt(np.mean([row['control'] ** 2 for row in late]))),
                   controller_guard_count=sum(row['controller_guarded'] for row in rows))
    return out


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--replay-root', required=True, type=Path,
                        help='Root of archived estimator-comparison common_stream records')
    parser.add_argument('--protocol', type=Path, default=PROTOCOL_PATH)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Existing output directories are never overwritten')
    protocol = json.loads(args.protocol.read_text())
    cfg = dict(fourier.CONFIG, **protocol)
    # This protocol replaces the six-entry scales with the declared seven-entry S.
    inputs = {str(seed): args.replay_root / f'seed_{seed}' / 'observations.csv' for seed in cfg['seeds']}
    for path in inputs.values():
        if not path.is_file():
            raise FileNotFoundError(path)
    args.output.mkdir(parents=True)
    dump_json(args.output / 'protocol.json', protocol)
    dump_json(args.output / 'config.json', cfg)
    repo = Path(__file__).resolve().parents[1]
    source_paths = [Path(__file__).resolve(), repo / 'experiments/fourier_online.py',
                    repo / 'experiments/common.py', args.protocol.resolve()]
    snapshot = args.output / 'source_snapshot'
    snapshot.mkdir()
    for source in source_paths:
        shutil.copyfile(source, snapshot / source.name)
    dump_json(args.output / 'environment.json', dict(started_utc=datetime.now(timezone.utc).isoformat(),
              python=sys.version, numpy=np.__version__, scipy=scipy.__version__,
              platform=platform.platform(), command=sys.argv, worker_count=1,
              source_sha256={p.name: sha(p) for p in source_paths},
              git_head=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=repo, text=True).strip(),
              replay_inputs={seed: dict(path=str(path), sha256=sha(path)) for seed, path in inputs.items()}))
    (args.output / 'git_status.txt').write_text(subprocess.check_output(['git', 'status', '--short'], cwd=repo, text=True))
    (args.output / 'tracked_diff.patch').write_bytes(subprocess.check_output(['git', 'diff', '--binary'], cwd=repo))
    summaries, failures = [], []
    for mode in ('replay', 'own_trajectory'):
        for seed in cfg['seeds']:
            destination = args.output / mode / f'seed_{seed}'
            destination.mkdir(parents=True)
            try:
                if mode == 'replay':
                    shutil.copyfile(inputs[str(seed)], destination / 'source_observations.csv')
                replay = read_stream(inputs[str(seed)]) if mode == 'replay' else None
                histories, stream, arrays, rng = simulate(seed, cfg, replay)
                save_trace(destination / 'observations.csv', stream)
                np.savez_compressed(destination / 'state.npz', **arrays)
                dump_json(destination / 'rng.json', rng)
                for method, rows in histories.items():
                    save_trace(destination / method / 'trace.csv', rows)
                    summary = dict(mode=mode, seed=seed, method=method,
                                   metrics=summarize(rows, cfg, mode == 'own_trajectory' and method == 'normalized_flow'))
                    dump_json(destination / method / 'summary.json', summary)
                    summaries.append(summary)
                print(json.dumps(dict(mode=mode, seed=seed, normalized=summaries[-3]['metrics'])), flush=True)
            except Exception as exc:
                failure = dict(mode=mode, seed=seed, traceback=traceback.format_exc())
                if isinstance(exc, SimulationFailure):
                    histories, stream, arrays, rng = exc.partial
                    if stream:
                        save_trace(destination / 'partial_observations.csv', stream)
                    for method, rows in histories.items():
                        if rows:
                            save_trace(destination / ('partial_' + method) / 'trace.csv', rows)
                    np.savez_compressed(destination / 'partial_state.npz', **arrays)
                    dump_json(destination / 'partial_rng.json', rng)
                    failure['retained_observations'] = len(stream)
                    failure['retained_complete_transitions'] = len(arrays['q_post'])
                failures.append(failure)
                dump_json(destination / 'failure.json', failure)
                print(json.dumps(failure), flush=True)
            dump_json(args.output / 'summary.json', dict(runs=summaries, failures=failures))
    manifest = {str(p.relative_to(args.output)): sha(p) for p in sorted(args.output.rglob('*')) if p.is_file()}
    dump_json(args.output / 'file_manifest.json', manifest)
    if failures:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
