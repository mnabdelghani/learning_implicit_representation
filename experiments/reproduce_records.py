"""Reproduce all declared records without overwriting historical results.

This orchestrator changes no simulation equations or protocol. It snapshots the
original modules, runs those copies, and records provenance before computation.
The output must be a new directory. These are regenerated records, not recovery
of either historical archive named in DATA_MANIFEST.json.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
SOURCES = [
    'experiments/common.py', 'experiments/contact.py',
    'experiments/fourier_online.py', 'experiments/burgers_online.py',
    'experiments/contact_excitation.py', 'experiments/reproduce_records.py',
    'examples/legacy_suite/simulations/burgers.py',
    'examples/legacy_suite/configs/burgers.json', 'requirements-tested.txt',
]
METHODS = {'contact': ('learned', 'frozen', 'oracle'),
           'fourier': ('learned', 'K0', 'frozen', 'oracle'),
           'burgers': ('learned', 'frozen', 'oracle'),
           'contact_excitation': ('task_only', 'bounded_probe')}


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def timestamp():
    return datetime.now(timezone.utc).isoformat()


def run_case(job):
    study, seed, method, output = job
    output = Path(output)
    snapshot = output / 'source_snapshot'
    sys.path.insert(0, str(snapshot / 'experiments'))
    import numpy as np
    from common import save_trace, dump_json
    start = time.monotonic()
    record = dict(study=study, seed=seed, method=method, started_utc=timestamp())
    try:
        dest = output / 'records' / study / f'seed_{seed}' / method
        cfg = json.loads((output / 'records' / study / 'config.json').read_text())
        if study == 'contact':
            import contact
            rows, partials = contact.simulate(seed, method, cfg)
            save_trace(dest / 'trace.csv', rows)
            np.savez_compressed(dest / 'partials.npz', **partials)
            summary = dict(seed=seed, method=method, metrics=contact.summarize(rows, cfg))
            dump_json(dest / 'summary.json', summary)
        elif study == 'fourier':
            import fourier_online
            rows = fourier_online.simulate(seed, method, cfg)
            save_trace(dest / 'trace.csv', rows)
            summary = dict(seed=seed, method=method, metrics=fourier_online.summarize(rows, cfg))
            dump_json(dest / 'summary.json', summary)
        elif study == 'burgers':
            import burgers_online
            ep = burgers_online.burgers.simulate_episode(cfg, method, seed)
            summary = burgers_online.burgers.save_episode(ep, dest)
            np.savez_compressed(dest / 'partials.npz', **burgers_online.additional_partials(
                ep, cfg['true_coefficients'], cfg['domain_length']))
        else:
            import contact_excitation
            summary = contact_excitation.run_case((seed, method, output / 'records' / study))
        record.update(status='complete', summary=str((dest / 'summary.json').relative_to(output)))
    except Exception:
        record.update(status='failed', traceback=traceback.format_exc())
    record.update(finished_utc=timestamp(), elapsed_seconds=time.monotonic() - start)
    write_json(output / 'execution' / f'{study}_{seed}_{method}.json', record)
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=2)
    args = parser.parse_args()
    if not 1 <= args.workers <= 2:
        parser.error('workers must be 1 or 2')
    output = args.output.resolve()
    if output.exists():
        parser.error('output already exists; use a new directory')
    output.mkdir(parents=True)
    for name in SOURCES:
        target = output / 'source_snapshot' / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / name, target)
    # Snapshot records are the declared protocols; check current source defaults
    # before any computation so a modified protocol is never silently substituted.
    sys.path.insert(0, str(output / 'source_snapshot' / 'experiments'))
    import contact, fourier_online
    for study in METHODS:
        cfg = json.loads((ROOT / 'results' / study / 'config.json').read_text())
        if study == 'contact' and cfg != contact.CONFIG:
            raise ValueError('Contact source defaults differ from saved protocol')
        if study == 'fourier' and cfg != fourier_online.CONFIG:
            raise ValueError('Fourier source defaults differ from saved protocol')
        if study == 'burgers':
            expected = json.loads((output / 'source_snapshot' / 'examples' / 'legacy_suite' /
                                   'configs' / 'burgers.json').read_text())
            if cfg != expected:
                raise ValueError('Burgers source configuration differs from saved protocol')
        if study == 'contact_excitation':
            expected = dict(contact.CONFIG, duration=80., change_time=1.e9)
            if cfg['plant'] != expected or cfg['seeds'] != list(range(10, 20)):
                raise ValueError('Transfer source defaults differ from saved protocol')
        write_json(output / 'records' / study / 'config.json', cfg)
    git = lambda *a: subprocess.check_output(['git', *a], cwd=ROOT, text=True)
    (output / 'git_status_at_start.txt').write_text(git('status', '--short'))
    (output / 'git_diff_at_start.patch').write_text(git('diff', '--no-ext-diff', '--binary'))
    import numpy as np
    with (output / 'numpy_configuration.txt').open('w') as stream:
        from contextlib import redirect_stdout
        with redirect_stdout(stream):
            np.show_config()
    packages = {d.metadata['Name']: d.version for d in importlib.metadata.distributions()}
    jobs = [(study, seed, method, str(output)) for seed in range(10, 20)
            for study, methods in METHODS.items() for method in methods]
    provenance = dict(status='running', started_utc=timestamp(),
        identity='New records from unchanged historical reconstruction protocols; not recovered archives',
        git_head=git('rev-parse', 'HEAD').strip(), argv=sys.argv,
        executable=sys.executable, python=sys.version, platform=platform.platform(),
        packages=packages, workers=args.workers,
        numerical_threads={k: os.environ.get(k) for k in
            ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS')},
        source_sha256={name: sha256(output / 'source_snapshot' / name) for name in SOURCES},
        planned_runs=[dict(study=s, seed=n, method=m) for s, n, m, _ in jobs],
        historical_runtime=json.loads((ROOT / 'docs/runtime.json').read_text()),
        historical_summaries_sha256={str(p.relative_to(ROOT)): sha256(p)
            for study in METHODS for p in sorted((ROOT / 'results' / study).rglob('summary.json'))})
    write_json(output / 'provenance.json', provenance)
    results = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(run_case, job) for job in jobs]
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            print(json.dumps(dict(completed=len(results), total=len(jobs), **result)), flush=True)
    failures = [r for r in results if r['status'] != 'complete']
    if not failures:
        for study in METHODS:
            if study == 'contact_excitation':
                import contact_excitation
                contact_excitation.aggregate(output / 'records' / study, list(range(10, 20)))
            else:
                runs = [json.loads((output / 'records' / study / f'seed_{s}' / m /
                                   'summary.json').read_text())
                        for s in range(10, 20) for m in METHODS[study]]
                write_json(output / 'records' / study / 'summary.json',
                           dict(status=provenance['identity'], runs=runs))
    provenance.update(status='failed' if failures else 'complete', finished_utc=timestamp(),
                      completed_runs=len(results)-len(failures), failed_runs=failures)
    write_json(output / 'provenance.json', provenance)
    if failures:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
