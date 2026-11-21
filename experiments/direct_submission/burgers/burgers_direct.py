#!/usr/bin/env python3
"""Prespecified direct-partial multi-seed study with causal position sensing.

The native learner sees only observed five-channel secants. The learned-row
controller sees measured positions and either declared ideal or causal estimated
rates. Clean positions/rates/partials remain simulator/evaluator channels.
"""
from __future__ import annotations
import argparse
from collections import deque
import copy
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import sys
import time
import traceback
import numpy as np

HERE = Path(__file__).resolve().parent
import paired_burgers as old
import native_kernel

class NativeRows(old.DirectRows):
    def step(self, observed_secant, dt, adapt=True):
        self.q, self.R, subdivisions = self.native_step(
            self.q, self.R, observed_secant, dt, tau=self.tau, gain=self.gain, adapt=adapt)
        return subdivisions

from causal_rate import CausalQuadraticRate

PROTOCOL_SHA256 = '22e225ae5ad2c275dd199bad2e93ea1f0221e51bba851c1e278fb7b4328f7980'
SENSOR_SOURCE_SHA256 = '64f6490330a9328b7c4561b5c0fc1cebffbef4c10ed2a5a47ec8e073698d9412'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


@dataclass
class State:
    w: np.ndarray
    u: np.ndarray
    learner: old.DirectRows
    history: deque
    rng: np.random.Generator
    rate_estimator: CausalQuadraticRate | None
    index: int = 0
    attempt: dict | None = None


def initial_state(cfg):
    s = old.plant.spatial_grid(cfg['grid_points'])
    learner = NativeRows(cfg['grid_points'], cfg['memory'], cfg['gain'], cfg['initial_seed'])
    learner.native_step = native_kernel.native_step
    estimator = (CausalQuadraticRate(cfg['dt'], cfg['rate_fit_window'])
                 if cfg['sensing_mode'] == 'causal' else None)
    return State(.2+.015*np.sin(s), np.zeros_like(s), learner,
        deque(maxlen=round(cfg['window']/cfg['dt'])+1),
        np.random.default_rng(cfg['sensor_seed']), estimator)


def observe_and_step(state, cfg, training, adaptive, terminal=False):
    """Sense current/past positions; query with q_pre; then assimilate/advance."""
    t, dt = state.index*cfg['dt'], cfg['dt']
    state.attempt = {}
    innovation = state.rng.standard_normal(len(state.w))
    noise = cfg['position_noise_std']*innovation
    measured_w = state.w+noise
    state.attempt.update(stage=np.array('sensing'), measured_w=measured_w.copy(),
                         sensor_innovation=innovation.copy(), sensor_noise=noise.copy())
    # In causal conditions no true-rate value is supplied to this sensor path.
    if state.rate_estimator is None:
        observed_rate = old.plant.plant_rhs(state.w, state.u, cfg['plant_coefficients'])
    else:
        observed_rate = state.rate_estimator.update(measured_w)
    rate_valid = observed_rate is not None
    if not rate_valid:
        observed_rate = np.full_like(measured_w, np.nan)
    measured_ws, measured_wss = old.plant.spectral_derivatives(measured_w)
    context = np.column_stack((measured_w, measured_ws, measured_wss, state.u, observed_rate))
    state.attempt.update(context=context.copy(), observed_rate=observed_rate.copy())
    if rate_valid:
        state.history.append(context.copy())
    if not training and not rate_valid:
        raise ValueError('control requested before causal rate window is available')
    q_pre, R_pre = state.learner.q.copy(), state.learner.R.copy()
    s = old.plant.spatial_grid(cfg['grid_points'])
    selected = old.command(q_pre, context, s, t, cfg, training)
    state.attempt.update(q_pre=q_pre.copy(), R_pre=R_pre.copy(),
                         **{key:value.copy() for key,value in selected.items()})
    state.attempt['stage'] = np.array('commanded')
    secant = np.full_like(context, np.nan)
    secant_valid = rate_valid and len(state.history) == state.history.maxlen
    memory_applied = secant_valid and not terminal
    subdivisions = 0
    if secant_valid:
        secant = (state.history[-1]-state.history[0])/cfg['window']
        state.attempt['secant'] = secant.copy()
        if memory_applied:
            subdivisions = state.learner.step(secant, dt, adapt=adaptive)
    state.attempt.update(stage=np.array('assimilated'), q_post=state.learner.q.copy(),
                         R_post=state.learner.R.copy())
    # All clean channels below are evaluator-only, except the declared ideal sensor.
    true_w = state.w.copy()
    true_wt = old.plant.plant_rhs(state.w, state.u, cfg['plant_coefficients'])
    true_ws, true_wss = old.plant.spectral_derivatives(state.w)
    clean_context = np.column_stack((true_w, true_ws, true_wss, state.u, true_wt))
    physical_F = old.plant.implicit_jacobian(clean_context, cfg['plant_coefficients'])
    record = dict(index=state.index, time=t, training=training,
        command_applied=not terminal, rate_valid=rate_valid, secant_valid=secant_valid,
        memory_applied=memory_applied, learning_applied=memory_applied and adaptive,
        true_w=true_w, measured_w=measured_w, sensor_innovation=innovation, sensor_noise=noise,
        actual_wt=true_wt, observed_rate=observed_rate,
        rate_error_evaluator=observed_rate-true_wt, clean_context=clean_context,
        context=context, secant=secant, q_pre=q_pre, q_post=state.learner.q.copy(),
        physical_F=physical_F, F_error=old.distance(q_pre, physical_F),
        initial_F_error=old.distance(state.learner.initial, physical_F),
        incompatibility_pre=np.sum(q_pre*secant, axis=1),
        subdivisions=subdivisions, **selected)
    memory = dict(index=state.index, R_pre=R_pre, R_post=state.learner.R.copy())
    if not (np.all(np.isfinite(state.w)) and np.all(np.isfinite(state.learner.q))):
        raise old.RecordedStepFailure('nonfinite physical state or learned field', record, memory)
    if np.max(np.abs(state.w)) > cfg['field_stop']:
        raise old.RecordedStepFailure('declared field magnitude stop exceeded', record, memory)
    if not terminal:
        state.w, state.u = old.advance(state.w, state.u, selected['udot'], dt, cfg['plant_coefficients'])
        state.index += 1
    return record, memory


def save_checkpoint(directory, stem, state):
    arrays = dict(w=state.w, u=state.u, q=state.learner.q, R=state.learner.R,
        initial_q=state.learner.initial, history=np.asarray(state.history), index=np.array(state.index))
    meta = dict(index=state.index, rng_state=state.rng.bit_generator.state,
                history_maxlen=state.history.maxlen, has_rate_estimator=state.rate_estimator is not None)
    if state.rate_estimator is not None:
        est = state.rate_estimator
        arrays.update(rate_history=np.asarray(est.history), rate_weights=est.weights,
                      rate_sample_offsets=est.sample_offsets)
        meta.update(rate_dt=est.dt, rate_window=est.window, rate_history_maxlen=est.history.maxlen)
    for suffix in ('.npz', '.json'):
        if (directory/(stem+suffix)).exists():
            raise FileExistsError(directory/(stem+suffix))
    np.savez_compressed(directory/(stem+'.npz'), **arrays)
    (directory/(stem+'.json')).write_text(json.dumps(meta, indent=2)+'\n')


def load_checkpoint(cfg, directory, stem):
    state = initial_state(cfg)
    meta = json.loads((directory/(stem+'.json')).read_text())
    with np.load(directory/(stem+'.npz')) as saved:
        state.w, state.u = saved['w'].copy(), saved['u'].copy()
        state.learner.q, state.learner.R = saved['q'].copy(), saved['R'].copy()
        state.learner.initial = saved['initial_q'].copy()
        state.history = deque([z.copy() for z in saved['history']], maxlen=meta['history_maxlen'])
        state.index = int(saved['index'])
        if meta['has_rate_estimator']:
            if state.rate_estimator is None:
                raise ValueError('checkpoint rate-estimator mode differs')
            est = state.rate_estimator
            if est.dt != meta['rate_dt'] or est.window != meta['rate_window']:
                raise ValueError('checkpoint rate-estimator timing differs')
            np.testing.assert_array_equal(est.weights, saved['rate_weights'])
            np.testing.assert_array_equal(est.sample_offsets, saved['rate_sample_offsets'])
            est.history = deque([w.copy() for w in saved['rate_history']], maxlen=meta['rate_history_maxlen'])
        elif state.rate_estimator is not None:
            raise ValueError('checkpoint lacks required rate estimator')
    state.rng.bit_generator.state = copy.deepcopy(meta['rng_state'])
    return state


class Recorder(old.Recorder):
    def add(self, record, memory):
        self.records.append(record)
        if record['index'] % round(.05/self.cfg['dt']) == 0 or not record['command_applied']:
            self.memories.append(memory)
        self.total += 1
        self.last_time = record['time']
        self.summaries.append(dict(time=record['time'], applied=record['command_applied'],
            task_mse=float(np.mean((record['true_w']-record['target_w'])**2)),
            F_error=float(np.mean(record['F_error'])),
            energy=float(.5*np.mean(record['incompatibility_pre']**2)),
            rate_mse=float(np.mean(record['rate_error_evaluator']**2)),
            guard=float(np.mean(record['guard'])), clip=float(np.mean(record['clip'])),
            input_min=float(np.min(np.abs(record['q_pre'][:, 3])))))
        if len(self.records) >= round(1./self.cfg['dt']):
            self.flush()

    def summary(self):
        result = dict(rows=self.total, last_recorded_time=self.last_time, chunks=self.chunk)
        for name, (a, b) in self.cfg['evaluation_windows'].items():
            rows = [r for r in self.summaries if r['applied'] and a <= r['time'] < b]
            if rows:
                def mean_value(key, rms=False):
                    values = np.array([r[key] for r in rows])
                    if not np.all(np.isfinite(values)):
                        return None
                    result = float(np.mean(values))
                    return float(np.sqrt(result)) if rms else result
                result[name] = dict(samples=len(rows), task_rms=mean_value('task_mse', rms=True),
                    F_error_mean=mean_value('F_error'),
                    energy_mean=mean_value('energy'),
                    rate_rms=mean_value('rate_mse', rms=True),
                    guard_fraction=float(np.mean([r['guard'] for r in rows])),
                    clip_fraction=float(np.mean([r['clip'] for r in rows])),
                    minimum_input_partial=float(np.min([r['input_min'] for r in rows])))
        return result


def run_phase(name, state, cfg, output, end_index, training, adaptive, endpoint, study_deadline):
    started = time.monotonic()
    recorder = Recorder(output/name, cfg)
    status, failure, pre_attempt = 'completed', None, None
    subdivisions_total = 0
    try:
        while state.index < end_index or (endpoint and state.index == end_index):
            pre_attempt = copy.deepcopy(state)
            state.attempt = None
            if time.monotonic() >= study_deadline:
                raise TimeoutError('declared study wall-runtime cap exceeded')
            if time.monotonic()-started > cfg['phase_wall_limit_seconds']:
                raise TimeoutError('declared independent phase wall-runtime cap exceeded')
            terminal = endpoint and state.index == end_index
            record, memory = observe_and_step(state, cfg, training, adaptive, terminal)
            recorder.add(record, memory)
            subdivisions_total += record['subdivisions']
            if subdivisions_total > cfg['per_phase_subdivision_cap']:
                raise FloatingPointError('declared independent phase numerical work cap exceeded')
            if state.index % round(5./cfg['dt']) == 0 and not terminal:
                save_checkpoint(output/name, f'checkpoint_{state.index:07d}', state)
            if terminal:
                break
    except BaseException as exc:
        if pre_attempt is not None:
            save_checkpoint(output/name, 'failure_pre_attempt', pre_attempt)
            np.savez_compressed(output/name/'failure_observation.npz', **(state.attempt or {}))
            (output/name/'failure_observation.json').write_text(json.dumps(dict(
                attempted_index=pre_attempt.index, fields=list((state.attempt or {}).keys()),
                command_applied='not established outside any preserved complete record'), indent=2)+'\n')
        if isinstance(exc, old.RecordedStepFailure):
            recorder.add(exc.record, exc.memory)
        status = 'stopped'
        failure = dict(type=type(exc).__name__, message=str(exc), traceback=traceback.format_exc())
    finally:
        recorder.flush()
        save_checkpoint(output/name, 'final_state', state)
        result = dict(status=status, failure=failure, **recorder.summary(),
            elapsed_seconds=time.monotonic()-started, total_subdivisions=subdivisions_total,
            independent_phase_wall_limit_seconds=cfg['phase_wall_limit_seconds'],
            independent_phase_subdivision_cap=cfg['per_phase_subdivision_cap'])
        (output/name/'summary.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    return result


def verify_sources():
    expected=json.loads((HERE/'package_sources.json').read_text())
    for name, wanted in expected.items():
        if sha(HERE/name) != wanted:
            raise ValueError(f'Packaged source changed: {name}')
    native_kernel.build_kernel()



def read_protocol():
    path = HERE/'protocol.json'
    if sha(path) != PROTOCOL_SHA256:
        raise ValueError('study protocol is not frozen or its identity changed; no full runs permitted')
    return json.loads(path.read_text())


def run_case(cfg, output, protocol_path, study_deadline):
    verify_sources()
    output.mkdir(parents=True, exist_ok=False)
    (output/'config.json').write_text(json.dumps(cfg, indent=2)+'\n')
    shutil.copy2(protocol_path, output/'protocol.json')
    sources = output/'sources'; sources.mkdir()
    source_files = [Path(__file__), HERE/'causal_rate.py', Path(old.__file__),
        old.PLANT_SOURCE,
        Path(native_kernel.__file__), native_kernel.SOURCE, native_kernel.LIBRARY,
        native_kernel.MANIFEST, HERE/'package_sources.json', HERE/'source_mapping.json']
    for path in source_files:
        shutil.copy2(path, sources/path.name)
    manifest = dict(protocol_sha256=sha(protocol_path), argv=sys.argv,
        source_hashes={str(p):sha(p) for p in source_files}, python=platform.python_version(),
        python_executable=sys.executable, numpy=np.__version__, platform=platform.platform(),
        thread_environment={k:os.environ.get(k) for k in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS')},
        sensing_mode=cfg['sensing_mode'], learner_information='Observed-context causal secants only',
        truth_boundary='Physical coefficients create simulated dynamics and evaluator records; causal sensor and controller receive no true wt or F',
        no_training_extension=True, no_seed_replacement=True)
    (output/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    state = initial_state(cfg)
    save_checkpoint(output, 'initial', state)
    outcomes = {}
    started = time.monotonic()
    try:
        outcomes['training'] = run_phase('training', state, cfg, output,
            round(cfg['training_time']/cfg['dt']), True, True, False, study_deadline)
        print(f"{cfg['case_id']} training: {outcomes['training']['status']}, last sample {outcomes['training']['last_recorded_time']}", flush=True)
        if outcomes['training']['status'] == 'completed':
            save_checkpoint(output, 'branch', state)
            arms = {name:copy.deepcopy(state) for name in ('adaptive', 'frozen')}
            for name in ('adaptive', 'frozen'):
                outcomes[name] = run_phase(name, arms[name], cfg, output,
                    round(cfg['duration']/cfg['dt']), False, name=='adaptive', True, study_deadline)
                print(f"{cfg['case_id']} {name}: {outcomes[name]['status']}, last sample {outcomes[name]['last_recorded_time']}", flush=True)
        else:
            for name in ('adaptive', 'frozen'):
                outcomes[name] = dict(status='not_started_training_incomplete', rows=0)
    finally:
        result = dict(case_id=cfg['case_id'], condition=cfg['condition'], initial_seed=cfg['initial_seed'],
            dt=cfg['dt'], outcomes=outcomes, elapsed_seconds=time.monotonic()-started)
        (output/'summary.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
        hashes = {str(p.relative_to(output)):sha(p) for p in sorted(output.rglob('*')) if p.is_file()}
        (output/'artifact_hashes.json').write_text(json.dumps(hashes, indent=2)+'\n')
    return result


def case_config(protocol, case):
    cfg = copy.deepcopy(protocol['common_configuration'])
    cfg.update(copy.deepcopy(case))
    cfg['protocol'] = protocol['identity']
    cfg['evaluation_windows'] = {key:copy.deepcopy(value) for key,value in protocol['windows'].items()
                                 if isinstance(value, list)}
    cfg['right_endpoint_excluded'] = protocol['windows']['right_endpoint_excluded']
    return cfg


def validate_protocol_configuration(protocol):
    cases = protocol['cases']
    order = [case['case_id'] for case in cases]
    if order != protocol['execution']['order'] or len(cases) != protocol['execution']['case_count']:
        raise ValueError('case order/count differs from frozen execution plan')
    if len(set(order)) != len(order):
        raise ValueError('duplicate case identity')
    cfg = protocol['common_configuration']
    if cfg['per_update_subdivision_cap'] != 10000 or cfg['excitation_frequencies'] != [3.,5.,7.,11.]:
        raise ValueError('fixed imported kernel/probe does not match protocol')
    if cfg['domain_length'] != 2*np.pi or cfg['grid_points'] != 32:
        raise ValueError('fixed plant discretization does not match protocol')
    for case in cases:
        for period in (cfg['window'], cfg['rate_fit_window'], 1., 5., cfg['training_time'], cfg['duration']):
            if not np.isclose(period/case['dt'], round(period/case['dt']), rtol=0, atol=1e-9):
                raise ValueError('declared physical periods must align with sample indices')
        if case['sensor_seed'] != 100000+case['initial_seed']:
            raise ValueError('sensor seed mapping differs')
    # Historical binary identities remain in the frozen protocol as provenance.
    # This portable source package builds its own local binary and records it.
    verify_sources()


def preserve_unattempted(cfg, output, reason):
    output.mkdir(parents=True, exist_ok=False)
    (output/'config.json').write_text(json.dumps(cfg, indent=2)+'\n')
    shutil.copy2(HERE/'protocol.json', output/'protocol.json')
    result = dict(case_id=cfg['case_id'], condition=cfg['condition'], initial_seed=cfg['initial_seed'],
        dt=cfg['dt'], status='not_attempted', reason=reason, physical_samples=0, outcomes={})
    (output/'summary.json').write_text(json.dumps(result, indent=2)+'\n')
    (output/'artifact_hashes.json').write_text(json.dumps(
        {p.name:sha(p) for p in sorted(output.iterdir()) if p.is_file()}, indent=2)+'\n')
    return result


def run_study(protocol, output_root):
    validate_protocol_configuration(protocol)
    output_root.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    deadline = started+protocol['execution']['study_physical_runtime_limit_seconds']
    study_manifest = dict(protocol_sha256=sha(HERE/'protocol.json'),
        runner_source_sha256=sha(Path(__file__)), fixed_order=protocol['execution']['order'],
        runtime_cap_seconds=protocol['execution']['study_physical_runtime_limit_seconds'],
        runtime_scope='One wall clock includes all case execution, recording and integrity checks; this conservatively bounds physical runtime',
        argv=sys.argv, python_executable=sys.executable)
    (output_root/'study_manifest.json').write_text(json.dumps(study_manifest, indent=2)+'\n')
    results, abort_reason = [], None
    for case in protocol['cases']:
        cfg = case_config(protocol, case)
        destination = output_root/case['case_id']
        if abort_reason is not None or time.monotonic() >= deadline:
            reason = abort_reason or 'declared study runtime cap reached before this case'
            results.append(preserve_unattempted(cfg, destination, reason))
        else:
            try:
                results.append(run_case(cfg, destination, HERE/'protocol.json', deadline))
            except BaseException as exc:
                abort_reason = 'infrastructure exception; remaining cases not started'
                failure = dict(type=type(exc).__name__, message=str(exc), traceback=traceback.format_exc())
                destination.mkdir(parents=True, exist_ok=True)
                (destination/'infrastructure_failure.json').write_text(json.dumps(failure, indent=2)+'\n')
                results.append(dict(case_id=case['case_id'], status='infrastructure_failure', failure=failure))
        progress = dict(protocol=protocol['identity'], completed_case_records=len(results),
            planned_case_count=len(protocol['cases']), elapsed_seconds=time.monotonic()-started,
            results=results)
        temporary = output_root/'study_progress.tmp.json'
        temporary.write_text(json.dumps(progress, indent=2, allow_nan=False)+'\n')
        temporary.replace(output_root/'study_progress.json')
    final = dict(protocol=protocol['identity'], planned_case_count=len(protocol['cases']),
        elapsed_seconds=time.monotonic()-started, results=results,
        no_seed_replacement=True, no_parameter_tuning=True)
    (output_root/'study_summary.json').write_text(json.dumps(final, indent=2, allow_nan=False)+'\n')
    return final


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True,
                        help='New destination; existing runs are never overwritten')
    group=parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--case', choices=[c['case_id'] for c in read_protocol()['cases']],
                       help='Reproduce one unchanged prespecified case')
    group.add_argument('--all', action='store_true', help='Reproduce all 19 cases; about 25 minutes on recorded machine')
    args=parser.parse_args()
    protocol=read_protocol()
    verify_sources()
    for variable in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
        if os.environ.get(variable) != '1':
            raise ValueError(f'{variable}=1 required for the bounded study')
    if args.all:
        result=run_study(protocol,args.output)
    else:
        validate_protocol_configuration(protocol)
        case=next(c for c in protocol['cases'] if c['case_id']==args.case)
        result=run_case(case_config(protocol,case),args.output,HERE/'protocol.json',
                        time.monotonic()+protocol['execution']['study_physical_runtime_limit_seconds'])
    print(json.dumps(result),flush=True)

if __name__=='__main__':
    main()
