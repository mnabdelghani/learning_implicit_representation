"""Extracted unchanged numerical functions for direct five-entry learning."""
from pathlib import Path
import numpy as np
import plant_burgers as plant
PLANT_SOURCE=Path(plant.__file__)

def unit_rows(q):
    norms = np.linalg.norm(q, axis=1)
    if np.any(norms < 1e-15) or not np.all(np.isfinite(norms)):
        raise FloatingPointError('nonfinite or vanishing generic row')
    return q/norms[:, None]


class DirectRows:
    """No plant coefficients, true F, reference, or physical features enter here."""
    def __init__(self, count, tau=4., gain=10000., seed=732):
        self.q = unit_rows(np.random.default_rng(seed).normal(size=(count, 5)))
        self.initial = self.q.copy()
        self.R = np.zeros((count, 5, 5))
        self.tau, self.gain = float(tau), float(gain)

    def step(self, observed_secant, dt, adapt=True):
        v = np.asarray(observed_secant)
        if v.shape != self.q.shape or not np.all(np.isfinite(v)):
            raise ValueError('finite raw-context secants required')
        outer = v[:, :, None]*v[:, None, :]
        old = self.R.copy()
        def memory(t):
            e = np.exp(-t/self.tau)
            return e*old-self.tau*np.expm1(-t/self.tau)*outer
        end = memory(dt)
        if not adapt:
            self.R = end
            return 0
        bound = max(float(np.max(np.trace(old, axis1=1, axis2=2))),
                    float(np.max(np.trace(end, axis1=1, axis2=2))))
        subdivisions = max(4, int(np.ceil(self.gain*dt*bound/.25)))
        if subdivisions > 10000:
            raise FloatingPointError('10000-subdivision per-sample numerical work cap exceeded')
        def tangent(q, t):
            qr = np.einsum('ni,nij->nj', q, memory(t))
            return -self.gain*(qr-np.sum(qr*q, axis=1)[:, None]*q)
        q, ds = self.q.copy(), dt/subdivisions
        for k in range(subdivisions):
            t = k*ds
            k1 = tangent(q, t)
            k2 = tangent(q+.5*ds*k1, t+.5*ds)
            k3 = tangent(q+.5*ds*k2, t+.5*ds)
            k4 = tangent(q+ds*k3, t+ds)
            q = unit_rows(q+ds*(k1+2*k2+2*k3+k4)/6)
        self.q, self.R = q, end
        return subdivisions


def reference(s, t):
    return (.2+.015*np.sin(s)*np.cos(.3*t)+.01*np.cos(2*s)*np.sin(.5*t),
            -.0045*np.sin(s)*np.sin(.3*t)+.005*np.cos(2*s)*np.cos(.5*t),
            -.00135*np.sin(s)*np.cos(.3*t)-.0025*np.cos(2*s)*np.sin(.5*t))


def exploration_rate(s, t, amplitude=.02):
    return amplitude/2*sum(om*np.sin((k+1)*s+.37*k)*np.cos(om*t)
                           for k, om in enumerate((3., 5., 7., 11.)))


def command(q, context, s, t, cfg, training):
    w, u, wt = context[:, 0], context[:, 3], context[:, 4]
    target_w, target_wt, reference_wtt = reference(s, t)
    target_wtt = reference_wtt-cfg['kp']*(w-target_w)-cfg['kd']*(wt-target_wt)
    wt_s, wt_ss = plant.spectral_derivatives(wt)
    supplied = np.column_stack((wt, wt_s, wt_ss, target_wtt))
    if training:
        raw = exploration_rate(s, t, cfg['probe_amplitude'])
        valid = np.ones(len(s), dtype=bool)
    else:
        valid = np.abs(q[:, 3]) >= cfg['input_guard']
        raw = -u.copy()  # Model-independent input decay when completion guarded.
        raw[valid] = -np.sum(q[valid][:, [0, 1, 2, 4]]*supplied[valid], axis=1)/q[valid, 3]
    applied = np.clip(raw, -cfg['input_rate_limit'], cfg['input_rate_limit'])
    return dict(udot_raw=raw, udot=applied, guard=~valid,
                clip=np.abs(raw)>cfg['input_rate_limit'], supplied_rates=supplied,
                target_w=target_w, target_wt=target_wt, target_wtt=target_wtt)


def advance(w, u, udot, dt, truth):
    k1 = plant.plant_rhs(w, u, truth)
    k2 = plant.plant_rhs(w+.5*dt*k1, u+.5*dt*udot, truth)
    k3 = plant.plant_rhs(w+.5*dt*k2, u+.5*dt*udot, truth)
    k4 = plant.plant_rhs(w+dt*k3, u+dt*udot, truth)
    return w+dt*(k1+2*k2+2*k3+k4)/6, u+dt*udot


def distance(q, F):
    return np.sqrt(np.maximum(0., 2*(1-np.sum(unit_rows(q)*unit_rows(F), axis=1)**2)))


class RecordedStepFailure(FloatingPointError):
    """Preserve a fully formed attempted sample before physical advancement."""
    def __init__(self, message, record, memory):
        super().__init__(message)
        self.record, self.memory = record, memory
        self.record['command_applied'] = False


class Recorder:
    """Incremental immutable 1 s chunks; finally flushes partial interval."""
    def __init__(self, directory, cfg):
        self.directory, self.cfg = directory, cfg
        directory.mkdir(parents=True, exist_ok=False)
        self.records, self.memories, self.chunk, self.total = [], [], 0, 0
        self.summaries = []
        self.last_time = None

    def add(self, record, memory):
        self.records.append(record)
        if record['index'] % round(.05/self.cfg['dt']) == 0 or not record['command_applied']:
            self.memories.append(memory)
        self.total += 1
        self.last_time = record['time']
        self.summaries.append(dict(time=record['time'], applied=record['command_applied'],
            task_mse=float(np.mean((record['context'][:, 0]-record['target_w'])**2)),
            F_error=float(np.mean(record['F_error'])),
            incompatibility_energy=float(.5*np.mean(record['incompatibility_pre']**2)),
            guard=float(np.mean(record['guard'])), clip=float(np.mean(record['clip'])),
            input_min=float(np.min(np.abs(record['q_pre'][:, 3])))))
        if len(self.records) >= round(1./self.cfg['dt']):
            self.flush()

    def flush(self):
        if not self.records:
            return
        arrays = {key: np.asarray([r[key] for r in self.records]) for key in self.records[0]}
        arrays.update({f'memory_{key}': np.asarray([r[key] for r in self.memories])
                       for key in ('index', 'R_pre', 'R_post')})
        np.savez_compressed(self.directory/f'chunk_{self.chunk:05d}.npz', **arrays)
        self.records, self.memories = [], []
        self.chunk += 1

    def summary(self):
        result = dict(rows=self.total, last_recorded_time=self.last_time, chunks=self.chunk)
        for name, a, b in [('control', 20., 40.), ('late', 36., 40.)]:
            rows = [r for r in self.summaries if r['applied'] and a <= r['time'] < b]
            if rows:
                result[name] = dict(samples=len(rows),
                    task_rms=float(np.sqrt(np.mean([r['task_mse'] for r in rows]))),
                    F_error_mean=float(np.mean([r['F_error'] for r in rows])),
                    energy_mean=float(np.mean([r['incompatibility_energy'] for r in rows])),
                    guard_fraction=float(np.mean([r['guard'] for r in rows])),
                    clip_fraction=float(np.mean([r['clip'] for r in rows])),
                    minimum_input_partial=float(np.min([r['input_min'] for r in rows])))
        return result

