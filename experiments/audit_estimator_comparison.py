"""Independent raw-record check; deliberately imports no experiment functions."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--input', required=True, type=Path)
ROOT = parser.parse_args().input
CFG = json.loads((ROOT / 'config.json').read_text())
SCALES = np.array(CFG['scales'])
LOWER = np.array(CFG['lower']) / SCALES
UPPER = np.array(CFG['upper']) / SCALES
CONTEXT = np.array(CFG['context_scales'])
TRUTH = np.array(CFG['true_coefficients'])
METHODS = ('memory_pg', 'memory_ewls', 'instant_nlms')
checks = 0
maxima = dict(energy_error=0., differential_error=0., summary_error=0.,
              update_error=0., ewls_kkt=0., memory_after_error=0.,
              ewls_objective_minus_pg=-float('inf'))


def close(a, b, key, atol=1e-10, rtol=1e-8):
    global checks
    a, b = np.asarray(a), np.asarray(b)
    np.testing.assert_allclose(a, b, atol=atol, rtol=rtol)
    finite = np.isfinite(a) & np.isfinite(b)
    if np.any(finite):
        maxima[key] = max(maxima[key], float(np.max(np.abs(a[finite] - b[finite]))))
    checks += 1


def load(path):
    return np.genfromtxt(path, delimiter=',', names=True)


def audit(run):
    seed, method, mode = run['seed'], run['method'], run['mode']
    folder = ROOT / mode / f'seed_{seed}'
    traces = load(folder / method / 'trace.csv')
    stream = load(folder / ('observations.csv' if mode == 'common_stream' else method + '/observations.csv'))
    t = stream['time']
    lag = round(CFG['difference_window'] / CFG['dt'])
    late = (t >= CFG['duration'] - 20) & (t < CFG['duration'])
    h = np.column_stack((np.ones(len(t)), np.cos(stream['phase']), np.sin(stream['phase'])))
    hp = np.column_stack((np.zeros(len(t)), -np.sin(stream['phase']), np.cos(stream['phase'])))
    features = np.column_stack((h * stream['observed_rate'][:, None], h * stream['state'][:, None], -stream['control']))
    v = (features[lag:] - features[:-lag]) / CFG['difference_window']
    c = np.column_stack([traces[f'c{j}'] for j in range(6)])
    q = np.column_stack((c, np.ones(len(t))))
    energy = np.sum(v * q[lag:], axis=1) ** 2 / 2
    close(energy, traces['preupdate_energy'][lag:], 'energy_error')
    def partials(c):
        rho = np.sum(h * c[:, :3], axis=1)
        kappa = np.sum(h * c[:, 3:], axis=1)
        clock = np.sum(hp * c[:, :3], axis=1) * stream['evaluator_actual_rate'] + np.sum(hp * c[:, 3:], axis=1) * stream['state']
        return np.column_stack((kappa, rho, -np.ones(len(t)), clock))
    F = partials(np.broadcast_to(TRUTH, c.shape))
    Fest = partials(c)
    close(F, np.column_stack([traces[f'F_true_{j}'] for j in range(4)]), 'differential_error')
    close(Fest, np.column_stack([traces[f'F_est_{j}'] for j in range(4)]), 'differential_error')
    e = np.linalg.norm((Fest-F)*CONTEXT, axis=1) / np.linalg.norm(F*CONTEXT, axis=1)
    ix = [0, 1, 3]
    eu = np.linalg.norm(((Fest-F)*CONTEXT)[:, ix], axis=1) / np.linalg.norm((F*CONTEXT)[:, ix], axis=1)
    close(e, traces['F_error'], 'differential_error')
    close(eu, traces['learned_partial_error'], 'differential_error')
    metrics = dict(F_error_mean=float(np.mean(e[late])),
                   learned_partial_error_mean=float(np.mean(eu[late])),
                   preupdate_energy_mean=float(np.mean(energy[late[lag:]])),
                   phase_partial_rmse=float(np.sqrt(np.mean((Fest[late,3]-F[late,3])**2))),
                   tracking_rmse=float(np.sqrt(np.mean((stream['state'][late]-stream['desired'][late])**2))),
                   control_rms=float(np.sqrt(np.mean(stream['control'][late]**2))))
    for key, value in metrics.items():
        close(value, run['metrics'][key], 'summary_error')
    theta = c / SCALES
    assert np.all(theta >= LOWER-1e-12) and np.all(theta <= UPPER+1e-12)
    np.testing.assert_array_equal(c[0], CFG['initial'])
    beta = np.exp(-CFG['dt'] / CFG['memory'])
    R = np.zeros((7,7))
    for i, unscaled in enumerate(v):
        n = i+lag
        channels = np.r_[unscaled[:6]*SCALES, unscaled[6]]
        R = beta*R+(1-beta)*np.outer(channels,channels)
        if n == len(t)-1:
            continue  # Final sample is not adapted; no next preupdate state.
        following = theta[n+1]
        H, b = R[:-1,:-1], R[:-1,-1]
        if method == 'memory_pg':
            expected = theta[n].copy()
            step = 1/max(float(np.trace(H)),1e-12)
            for _ in range(CFG['steps']):
                expected = np.clip(expected-step*(H@expected+b),LOWER,UPPER)
            close(following, expected, 'update_error')
        elif method == 'instant_nlms':
            a = channels[:-1]
            expected = np.clip(theta[n]-a*(channels@np.r_[theta[n],1])/max(float(a@a),1e-12),LOWER,UPPER)
            close(following, expected, 'update_error')
        else:
            projected_gradient = following-np.clip(following-(H@following+b),LOWER,UPPER)
            maxima['ewls_kkt'] = max(maxima['ewls_kkt'],float(np.max(abs(projected_gradient))))
            assert np.max(abs(projected_gradient)) < 1e-7
        postq = np.r_[following,1]
        close(postq@R@postq/2,traces['memory_after'][n],'memory_after_error')
    if mode == 'common_stream' and method == 'memory_ewls':
        pg = load(folder/'memory_pg'/'trace.csv')
        valid = np.isfinite(traces['memory_after']) & (t<CFG['duration'])
        difference = traces['memory_after'][valid]-pg['memory_after'][valid]
        maxima['ewls_objective_minus_pg'] = max(maxima['ewls_objective_minus_pg'],float(np.max(difference)))
        assert np.max(difference) < 1e-9
        close(traces['information_min_eigenvalue'],pg['information_min_eigenvalue'],'summary_error')
    print(seed,mode,method,'audited',flush=True)
    return dict(seed=seed, mode=mode, method=method, rows=len(t), audited_metrics=metrics)


summary = json.loads((ROOT/'summaries.json').read_text())
assert not summary['failures']
assert len(summary['runs']) == 50
audited = [audit(run) for run in summary['runs']]
sources = json.loads((ROOT/'environment.json').read_text())['source_sha256']
for filename, digest in sources.items():
    assert hashlib.sha256((ROOT/'source_snapshot'/filename).read_bytes()).hexdigest() == digest
result = dict(status='passed', audited_runs=len(audited), seeds=list(range(10,20)),
              checks=checks, maxima=maxima, runs=audited,
              scope='Independent CSV metric, full differential incl clock, all numerical update, EWLS KKT, current-candidate memory and common-stream objective audit; no experiment functions imported')
(ROOT/'raw_audit.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({k:v for k,v in result.items() if k!='runs'},indent=2))
