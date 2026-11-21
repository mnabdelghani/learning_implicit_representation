#!/usr/bin/env python3
"""Independent reconstruction for the paired direct-field Burgers validation.

No runner/learner imports. Derives evaluator F, references, spectral kinematics,
readout completion, exact held-secant memory and selected joint Q/R ODE checks
from raw records. This script only reads experiments and writes audit artifacts.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from scipy.integrate import solve_ivp


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def normalize(x):
    return x / np.linalg.norm(x, axis=-1, keepdims=True)


def projector_distance(a, b):
    inner = np.sum(normalize(a) * normalize(b), axis=-1)
    return np.sqrt(np.maximum(0., 2. * (1. - np.minimum(1., inner**2))))


def physical_field(context, coefficients):
    ca, nu, b = coefficients
    F = np.empty_like(context)
    F[..., 0] = ca * context[..., 1]
    F[..., 1] = ca * context[..., 0]
    F[..., 2] = -nu
    F[..., 3] = -b
    F[..., 4] = 1.
    return F


def reference(times, count):
    t = np.asarray(times)[:, None]
    s = 2. * np.pi * np.arange(count)[None, :] / count
    w = .2 + .015 * np.sin(s) * np.cos(.3*t) + .01 * np.cos(2*s) * np.sin(.5*t)
    wt = -.0045*np.sin(s)*np.sin(.3*t) + .005*np.cos(2*s)*np.cos(.5*t)
    wtt = -.00135*np.sin(s)*np.cos(.3*t) - .0025*np.cos(2*s)*np.sin(.5*t)
    return w, wt, wtt


def spectral(values):
    n = values.shape[-1]
    k = np.fft.fftfreq(n, d=1./n)
    vhat = np.fft.fft(values, axis=-1)
    return (np.fft.ifft(vhat * (1j*k), axis=-1).real,
            np.fft.ifft(vhat * (-k*k), axis=-1).real)


def rms(x):
    return float(np.sqrt(np.mean(np.square(x)))) if np.size(x) else None


def finite_summary(x):
    values = np.asarray(x)
    values = values[np.isfinite(values)]
    if not len(values):
        return dict(count=0, mean=None, p50=None, p95=None, maximum=None)
    return dict(count=len(values), mean=float(np.mean(values)), p50=float(np.median(values)),
                p95=float(np.quantile(values,.95)), maximum=float(np.max(values)))


def joint_ode_check(q, R, v, q_saved, R_saved, cfg):
    """Integrate the literal coupled Q,R ODE, without exact-memory substitution.

    Independent DOP853 evolves all five Q and twenty-five R entries at each
    selected spatial site. The returned endpoint is not renormalized.
    """
    shape_q, shape_R = q.shape, R.shape
    nq = q.size
    outer = v[..., :, None] * v[..., None, :]
    gain, tau, dt = cfg['gain'], cfg['memory'], cfg['dt']
    def fun(t, y):
        qq = y[:nq].reshape(shape_q)
        RR = y[nq:].reshape(shape_R)
        qr = np.einsum('ni,nij->nj', qq, RR)
        qdot = -gain * (qr - np.sum(qr*qq, axis=-1)[:,None]*qq)
        Rdot = outer - RR/tau
        return np.r_[qdot.ravel(), Rdot.ravel()]
    beta = np.exp(-dt/tau)
    R_end = beta*R + tau*(-np.expm1(-dt/tau))*outer
    curvature = max(float(np.max(np.trace(R,axis1=-2,axis2=-1))),
                    float(np.max(np.trace(R_end,axis1=-2,axis2=-1))))
    max_step = min(dt, .1 / max(1e-30, gain*curvature))
    solved = solve_ivp(fun, (0.,dt), np.r_[q.ravel(), R.ravel()], method='DOP853',
                       rtol=2e-11, atol=2e-13, max_step=max_step)
    qq = solved.y[:nq,-1].reshape(shape_q)
    RR = solved.y[nq:,-1].reshape(shape_R)
    return dict(success=bool(solved.success), nfev=int(solved.nfev),
                q_max_abs_error=float(np.max(np.abs(qq-q_saved))),
                R_max_abs_error=float(np.max(np.abs(RR-R_saved))),
                joint_R_vs_analytic_max_abs=float(np.max(np.abs(RR-R_end))),
                uncorrected_joint_row_norm_error=float(np.max(np.abs(np.sum(qq*qq,axis=-1)-1.))))


# Raw-record adapter and command-line entry point are added after runner schema
# review; numerical/evaluator functions above do not depend on its implementation.

def load_records(directory):
    files = sorted(directory.glob('chunk_*.npz'))
    if not files:
        return {}, []
    parts = {}
    for p in files:
        with np.load(p) as data:
            for key in data.files:
                value = data[key]
                if len(value):
                    parts.setdefault(key, []).append(value)
    return {k: np.concatenate(v, axis=0) for k,v in parts.items()}, files


def maximum(x):
    return float(np.max(np.abs(x))) if np.size(x) else 0.


def phase_metrics(data, cfg, initial, low, high):
    t, z, q = data['time'], data['context'], data['q_pre']
    mask = (t >= low-1e-10) & (t < high-1e-10) & data['command_applied']
    if not np.any(mask):
        return None
    F = physical_field(z[mask], cfg['plant_coefficients'])
    error = projector_distance(q[mask], F)
    initial_error = projector_distance(initial, F)
    target, target_rate, _ = reference(t[mask], z.shape[1])
    rates = data['secant'][mask]
    incompat = np.sum(q[mask]*rates, axis=-1)
    physical_incompat = np.sum(normalize(F)*rates, axis=-1)
    valid_rates = np.all(np.isfinite(rates), axis=-1)
    out = dict(samples=int(np.count_nonzero(mask)), spatial_points=z.shape[1], interval=[low,high],
        first_observed_time=float(t[mask][0]),last_observed_time=float(t[mask][-1]),
        expected_samples=int(round((high-low)/cfg['dt'])),
        full_interval_observed=bool(np.count_nonzero(mask)==round((high-low)/cfg['dt'])),
        right_endpoint_excluded=True, field=finite_summary(error),
        field_p99=float(np.quantile(error,.99)),
        field_worst_site_mean=float(np.max(np.mean(error,axis=0))),
        field_per_site_mean=np.mean(error,axis=0).tolist(),
        initial_shadow_field_mean=float(np.mean(initial_error)),
        learned_to_initial_error_ratio=float(np.mean(error)/np.mean(initial_error)),
        task_global_rms=rms(z[mask,:,0]-target),
        rate_global_rms=rms(data['actual_wt'][mask]-target_rate),
        incompatibility_half_mean_square=float(.5*np.mean(incompat[valid_rates]**2)),
        physical_row_secant_half_mean_square=float(.5*np.mean(physical_incompat[valid_rates]**2)),
        observed_secant_global_rms=rms(rates[valid_rates]),
        guard_count=int(np.count_nonzero(data['guard'][mask])),
        clipping_count=int(np.count_nonzero(data['clip'][mask])),
        query_count=int(error.size), guard_fraction=float(np.mean(data['guard'][mask])),
        clipping_fraction=float(np.mean(data['clip'][mask])),
        input_partial_abs_min=float(np.min(np.abs(q[mask,:,3]))),
        input_global_rms=rms(z[mask,:,3]), input_rate_global_rms=rms(data['udot'][mask]))
    if low >= cfg['training_time']:
        supplied = data['supplied_rates'][mask]
        oracle = -np.sum(F[:,:,[0,1,2,4]]*supplied,axis=-1)/F[:,:,3]
        accepted = ~data['guard'][mask]
        out['accepted_query_count'] = int(np.count_nonzero(accepted))
        out['unclipped_learned_vs_true_completion_rms'] = rms((data['udot_raw'][mask]-oracle)[accepted])
        out['applied_vs_true_completion_rms_accepted'] = rms((data['udot'][mask]-oracle)[accepted])
        out['applied_vs_true_completion_rms_all'] = rms(data['udot'][mask]-oracle)
        out['true_completion_global_rms_accepted'] = rms(oracle[accepted])
    return out


def memory_audit(data, cfg, initial_R, initial_history):
    lag = round(cfg['window']/cfg['dt'])
    z, q = data['context'], data['q_pre']
    history = list(initial_history.copy())
    R = initial_R.copy()
    beta = np.exp(-cfg['dt']/cfg['memory'])
    alpha = cfg['memory']*(-np.expm1(-cfg['dt']/cfg['memory']))
    sparse = {int(index):j for j,index in enumerate(data.get('memory_index',[]))}
    discrepancies = dict(secant=0., R_pre=0., R_post=0.)
    diagnostic = []
    for j,index in enumerate(data['index']):
        history.append(z[j])
        history = history[-(lag+1):]
        if len(history) == lag+1:
            expected_v = (history[-1]-history[0])/cfg['window']
            discrepancies['secant'] = max(discrepancies['secant'],maximum(data['secant'][j]-expected_v))
        else:
            expected_v = None
            if not np.all(np.isnan(data['secant'][j])):
                raise AssertionError('warmup secant is not unavailable')
        if int(index) in sparse:
            k = sparse[int(index)]
            discrepancies['R_pre'] = max(discrepancies['R_pre'],maximum(data['memory_R_pre'][k]-R))
            eigenvalues, vectors = np.linalg.eigh(R)
            F = physical_field(z[j],cfg['plant_coefficients'])
            scale = np.maximum(eigenvalues[:,-1],1e-30)
            diagnostic.append(dict(time=float(data['time'][j]), applied=bool(data['command_applied'][j]),
                bottom_to_physical=projector_distance(vectors[:,:,0],F),
                q_to_bottom=projector_distance(q[j],vectors[:,:,0]),
                eigvals=eigenvalues, relative_minimum=eigenvalues[:,0]/scale,
                gap=eigenvalues[:,1]-eigenvalues[:,0],
                rank_1e8=np.sum(eigenvalues>1e-8*scale[:,None],axis=-1),
                rank_1e4=np.sum(eigenvalues>1e-4*scale[:,None],axis=-1)))
        if expected_v is not None and data['time'][j] < cfg['duration']-1e-10:
            R = beta*R + alpha*expected_v[:,:,None]*expected_v[:,None,:]
        if int(index) in sparse:
            k = sparse[int(index)]
            discrepancies['R_post'] = max(discrepancies['R_post'],maximum(data['memory_R_post'][k]-R))
    phases = {}
    for label,lo,hi in [('training_evaluation',16.,20.),('control',20.,40.),('late',36.,40.)]:
        selected = [d for d in diagnostic if d['applied'] and lo <= d['time'] < hi]
        if selected:
            phases[label] = {key:finite_summary(np.asarray([d[key] for d in selected]))
                for key in ('bottom_to_physical','q_to_bottom','relative_minimum','gap','rank_1e8','rank_1e4')}
            phases[label]['eigenvalue_means'] = np.mean([d['eigvals'] for d in selected],axis=(0,1)).tolist()
            phases[label]['snapshots'] = len(selected)
    return discrepancies, phases, R


def audit_phase(directory, cfg, checkpoint, label, ode_checks):
    data, files = load_records(directory)
    if not data:
        return dict(status='no raw data'), data
    initial = checkpoint['initial_q']
    F = physical_field(data['context'],cfg['plant_coefficients'])
    target, target_rate, target_acc = reference(data['time'],cfg['grid_points'])
    z, q = data['context'], data['q_pre']
    ws, wss = spectral(z[:,:,0])
    wt_s, wt_ss = spectral(z[:,:,4])
    desired_acc = target_acc-cfg['kp']*(z[:,:,0]-target)-cfg['kd']*(z[:,:,4]-target_rate)
    supplied = np.stack((z[:,:,4],wt_s,wt_ss,desired_acc),axis=-1)
    derived_energy = np.sum(q*data['secant'],axis=-1)
    check = dict(physical_F=maximum(F-data['physical_F']),
        projector_error=maximum(projector_distance(q,F)-data['F_error']),
        initial_shadow_error=maximum(projector_distance(initial,F)-data['initial_F_error']),
        reference=maximum(target-data['target_w']), reference_rate=maximum(target_rate-data['target_wt']),
        desired_acceleration=maximum(desired_acc-data['target_wtt']),
        supplied_rates=maximum(supplied-data['supplied_rates']),
        observed_ws=maximum(ws-z[:,:,1]), observed_wss=maximum(wss-z[:,:,2]),
        ideal_wt_sensor=maximum(z[:,:,4]-data['actual_wt']),
        row_norm=maximum(np.linalg.norm(q,axis=-1)-1.),
        q_transition=maximum(q[1:]-data['q_post'][:-1]),
        incompatibility=maximum((derived_energy-data['incompatibility_pre'])[np.isfinite(derived_energy)]),
        initial_q_from_seed=maximum(initial-normalize(np.random.default_rng(cfg['initial_seed']).normal(size=initial.shape))))
    ca,nu,b = cfg['plant_coefficients']
    check['plant_sensor_rhs'] = maximum(data['actual_wt']-(-ca*z[:,:,0]*ws+nu*wss+b*z[:,:,3]))
    if label != 'training':
        valid = np.abs(q[:,:,3]) >= cfg['input_guard']
        raw = -z[:,:,3].copy()
        raw[valid] = (-np.sum(q[:,:,[0,1,2,4]]*supplied,axis=-1)/q[:,:,3])[valid]
        check['guard_mask_mismatch_count'] = int(np.count_nonzero(data['guard'] != ~valid))
        check['raw_command'] = maximum(raw-data['udot_raw'])
        check['applied_command'] = maximum(np.clip(raw,-cfg['input_rate_limit'],cfg['input_rate_limit'])-data['udot'])
        check['clip_mask_mismatch_count'] = int(np.count_nonzero(data['clip'] != (np.abs(raw)>cfg['input_rate_limit'])))
        # Independent sign-gauge check on every recorded accepted query.
        signs = np.where(np.arange(cfg['grid_points'])%2, -1., 1.)
        sq = q*signs[None,:,None]
        signed_command = -np.sum(sq[:,:,[0,1,2,4]]*supplied,axis=-1)/sq[:,:,3]
        check['sign_gauge_command'] = maximum((signed_command-raw)[valid])
        check['sign_gauge_projector'] = maximum(projector_distance(sq,F)-projector_distance(q,F))
    else:
        s = 2*np.pi*np.arange(cfg['grid_points'])/cfg['grid_points']
        raw = cfg['probe_amplitude']/2*sum(om*np.sin((k+1)*s+.37*k)[None,:]*np.cos(om*data['time'])[:,None]
            for k,om in enumerate((3.,5.,7.,11.)))
        check['probing_command'] = maximum(raw-data['udot_raw'])
    mem_check, mem_diag, final_R = memory_audit(data,cfg,checkpoint['R'],checkpoint['history'])
    final = np.load(directory/'final_state.npz')
    check.update({'memory_'+k:v for k,v in mem_check.items()})
    check['final_R'] = maximum(final_R-final['R'])
    check['final_q'] = maximum(data['q_post'][-1]-final['q'])
    check['checkpoint_initial_q_pre'] = maximum(data['q_pre'][0]-checkpoint['q'])
    check['checkpoint_initial_w'] = maximum(data['context'][0,:,0]-checkpoint['w'])
    check['checkpoint_initial_u'] = maximum(data['context'][0,:,3]-checkpoint['u'])
    if label == 'frozen':
        check['frozen_q_drift'] = maximum(data['q_pre']-checkpoint['q'])
        check['frozen_q_post_drift'] = maximum(data['q_post']-checkpoint['q'])
    terminal = ~data['command_applied']
    check['terminal_no_q_update'] = maximum(data['q_pre'][terminal]-data['q_post'][terminal])
    check['terminal_samples'] = int(np.count_nonzero(terminal))
    # Independent RK4 plant/input transition, batched across all actual adjacent records.
    mask = data['command_applied'][:-1]
    w,u,udot,dt = z[:-1,:,0],z[:-1,:,3],data['udot'][:-1],cfg['dt']
    def rhs(ww,uu):
        d1,d2 = spectral(ww)
        return -ca*ww*d1+nu*d2+b*uu
    k1=rhs(w,u);k2=rhs(w+dt*k1/2,u+dt*udot/2)
    k3=rhs(w+dt*k2/2,u+dt*udot/2);k4=rhs(w+dt*k3,u+dt*udot)
    check['plant_RK4_transition'] = maximum((w+dt*(k1+2*k2+2*k3+k4)/6-z[1:,:,0])[mask])
    check['input_integral_transition'] = maximum((u+dt*udot-z[1:,:,3])[mask])
    checks = []
    if ode_checks and label != 'frozen':
        desired_times = [1., 10., 19.5] if label=='training' else [20.,25.,39.]
        by_index = {int(i):j for j,i in enumerate(data['index'])}
        by_memory = {int(i):j for j,i in enumerate(data['memory_index'])}
        if label == 'adaptive':
            candidates=[by_index[i] for i in by_memory if data['command_applied'][by_index[i]]]
            if candidates:
                stiff=max(candidates,key=lambda j:int(data['subdivisions'][j]))
                desired_times.append(float(data['time'][stiff]))
        for time in sorted(set(desired_times)):
            index = round(time/cfg['dt'])
            if index not in by_index or index not in by_memory:
                continue
            j,k = by_index[index],by_memory[index]
            result = joint_ode_check(q[j], data['memory_R_pre'][k], data['secant'][j],
                data['q_post'][j], data['memory_R_post'][k],cfg)
            checks.append(dict(time=time, **result))
    result = dict(raw_chunks=len(files),raw_rows=len(data['time']),
        first_time=float(data['time'][0]),last_time=float(data['time'][-1]),
        checks=check, memory_diagnostics=mem_diag, independent_joint_ODE_checks=checks,
        all_entries_max_abs_change_from_initial=np.max(np.abs(data['q_post']-initial),axis=(0,1)).tolist())
    for name,lo,hi in [('training_evaluation',16.,20.),('control',20.,40.),('late',36.,40.)]:
        metrics = phase_metrics(data,cfg,initial,lo,hi)
        if metrics is not None:
            result[name] = metrics
    return result,data



def training_site_diagnostics(data, cfg, initial):
    """Relate uneven recovery to retained local memories without tuning."""
    q, z, t = data['q_pre'], data['context'], data['time']
    mask = (t >= 16.) & (t < 20.)
    errors = projector_distance(q[mask], physical_field(z[mask],cfg['plant_coefficients']))
    by_index = {int(value):j for j,value in enumerate(data['index'])}
    values=[]
    for k,index in enumerate(data['memory_index']):
        j=by_index[int(index)]
        if not (16. <= t[j] < 20.):
            continue
        eig,vec=np.linalg.eigh(data['memory_R_pre'][k])
        bottom=vec[:,:,0]
        true=normalize(physical_field(z[j],cfg['plant_coefficients']))
        values.append(dict(gap=eig[:,1]-eig[:,0],
            eig0=eig[:,0],eig1=eig[:,1],eigmax=eig[:,-1],
            bottom_physical=projector_distance(bottom,true),
            q_bottom=projector_distance(q[j],bottom),
            initial_physical_overlap=np.abs(np.sum(initial*true,axis=-1)),
            initial_bottom_overlap=np.abs(np.sum(initial*bottom,axis=-1))))
    table=[]
    for site in np.argsort(-np.mean(errors,axis=0)):
        row=dict(site=int(site),spatial_coordinate=float(2*np.pi*site/cfg['grid_points']),
            physical_error_mean=float(np.mean(errors[:,site])),
            physical_error_max=float(np.max(errors[:,site])))
        for key in values[0]:
            column=np.asarray([v[key][site] for v in values])
            row[key+'_mean']=float(np.mean(column))
            row[key+'_minimum']=float(np.min(column))
        row['gain_times_mean_gap']=cfg['gain']*row['gap_mean']
        table.append(row)
    return dict(interval=[16.,20.],sites_sorted_by_descending_mean_error=table,
        interpretation='Descriptive only. Small initial alignment or weak local gap can delay convergence; empirical-bottom bias separates memory mismatch. No single diagnostic alone proves causality.')


def audit_pair(directory, ode_checks=True):
    cfg = json.loads((directory/'config.json').read_text())
    summaries = json.loads((directory/'summary.json').read_text())
    initial,branch = np.load(directory/'initial.npz'),np.load(directory/'branch.npz')
    hashes = json.loads((directory/'artifact_hashes.json').read_text())
    mismatch = [name for name,digest in hashes.items() if sha(directory/name)!=digest]
    result = dict(dt=cfg['dt'],statuses=summaries['outcomes'],artifact_hash_mismatches=mismatch,
        protocol_sha256=sha(directory/'protocol.json'),phases={})
    raw = {}
    for name in ('training','adaptive','frozen'):
        result['phases'][name],raw[name] = audit_phase(directory/name,cfg,initial if name=='training' else branch,name,ode_checks)
    result['training_site_diagnostics'] = training_site_diagnostics(raw['training'],cfg,initial['initial_q'])
    same_keys=('context','q_pre','secant','udot_raw','udot','supplied_rates','target_wtt')
    result['branch_equal_first_sample'] = ({k:maximum(raw['adaptive'][k][0]-raw['frozen'][k][0]) for k in same_keys}
        if raw['adaptive'] and raw['frozen'] else None)
    result['branch_checkpoint_vs_training_final'] = {}
    train_final = np.load(directory/'training/final_state.npz')
    for key in ('w','u','q','R','initial_q','history','index'):
        result['branch_checkpoint_vs_training_final'][key] = maximum(branch[key]-train_final[key])
    result['branch_rng_equals_training_final'] = (json.loads((directory/'branch.json').read_text()) ==
        json.loads((directory/'training/final_state.json').read_text()))
    train = result['phases']['training'].get('training_evaluation')
    late = result['phases']['adaptive'].get('late')
    complete = all(summaries['outcomes'].get(phase,{}).get('status')=='completed' for phase in ('training','adaptive','frozen'))
    result['prespecified_assessment'] = dict(
        completed=complete,
        training_mean_below_point1=bool(train and train['field']['mean']<=.1),
        training_error_below_quarter_initial=bool(train and train['learned_to_initial_error_ratio']<=.25),
        adaptive_late_evaluation_available=bool(late),
        adaptive_late_mean_below_point1=bool(late['field']['mean']<=.1) if late else None,
        adaptive_late_p95_below_point25=bool(late['field']['p95']<=.25) if late else None,
        adaptive_late_readout_task=bool(complete and late['guard_fraction']<.01 and late['clipping_fraction']<.01 and late['task_global_rms']<=.001) if late else None)
    return result,raw


def refinement(coarse, fine):
    out={}
    for arm in ('training','adaptive','frozen'):
        a,b=coarse[arm],fine[arm]
        if not a or not b:
            out[arm] = dict(status='No common recorded trajectory; phase stopped before observations')
            continue
        keys={round(float(t),8):j for j,t in enumerate(b['time'])}
        ai=np.asarray([j for j,t in enumerate(a['time']) if round(float(t),8) in keys],dtype=int)
        bi=np.asarray([keys[round(float(a['time'][j]),8)] for j in ai],dtype=int)
        applied=a['command_applied'][ai]&b['command_applied'][bi]
        ai,bi=ai[applied],bi[applied]
        target,_,_=reference(a['time'][ai],a['context'].shape[1])
        out[arm]=dict(common_applied_samples=len(ai),
            first_common_time=float(a['time'][ai][0]),last_common_time=float(a['time'][ai][-1]),
            coarse_common_task_rms=rms(a['context'][ai,:,0]-target),
            fine_common_task_rms=rms(b['context'][bi,:,0]-target),
            coarse_common_field_mean=float(np.mean(projector_distance(a['q_pre'][ai],physical_field(a['context'][ai],[1.,.08,1.])))),
            fine_common_field_mean=float(np.mean(projector_distance(b['q_pre'][bi],physical_field(b['context'][bi],[1.,.08,1.])))),
            state_rms_difference=rms(a['context'][ai,:,0]-b['context'][bi,:,0]),
            state_max_abs_difference=maximum(a['context'][ai,:,0]-b['context'][bi,:,0]),
            input_rms_difference=rms(a['context'][ai,:,3]-b['context'][bi,:,3]),
            input_max_abs_difference=maximum(a['context'][ai,:,3]-b['context'][bi,:,3]),
            learned_row_projector_mean_difference=float(np.mean(projector_distance(a['q_pre'][ai],b['q_pre'][bi]))),
            learned_row_projector_max_difference=float(np.max(projector_distance(a['q_pre'][ai],b['q_pre'][bi]))))
    return out


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--root',type=Path,default=Path(__file__).parent)
    parser.add_argument('--skip-ode',action='store_true')
    args=parser.parse_args()
    results,raws={},{}
    for name in ('dt_0p005','dt_0p0025'):
        cache=args.root/'independent_audit_coarse.json'
        reuse=(name=='dt_0p005' and cache.exists() and not args.skip_ode)
        results[name],raws[name]=audit_pair(args.root/name,not args.skip_ode and not reuse)
        if reuse:
            prior=json.loads(cache.read_text())
            for label in ('training','adaptive'):
                results[name]['phases'][label]['independent_joint_ODE_checks']=prior['phases'][label]['independent_joint_ODE_checks']
            results[name]['joint_ODE_cache_sha256']=sha(cache)
    ref=refinement(raws['dt_0p005'],raws['dt_0p0025'])
    scalar={}
    for arm in ('adaptive','frozen'):
        a=results['dt_0p005']['phases'][arm].get('late')
        b=results['dt_0p0025']['phases'][arm].get('late')
        scalar[arm]={}
        if a is None or b is None:
            scalar[arm] = dict(status='Late interval absent in at least one discretization; no extrapolation')
            continue
        for key,first,second,absolute_threshold in (
            ('field_mean',a['field']['mean'],b['field']['mean'],1e-5),
            ('task_global_rms',a['task_global_rms'],b['task_global_rms'],1e-4)):
            change=abs(second-first)
            relative=change/abs(second) if abs(second)>1e-30 else None
            scalar[arm][key]=dict(coarse=first,fine=second,absolute_change=change,
                relative_change_against_fine=relative,
                relative_flag=bool(abs(second)>1e-6 and relative>.1),
                absolute_flag=bool(change>absolute_threshold))
    final=dict(audit_source_sha256=sha(__file__),implementation_imported=False,
        scope='Independent raw-data reconstruction; joint coupled Q/R step audit; no simulation reruns or parameter changes.',
        runs=results,common_grid_refinement=ref,late_metric_refinement=scalar)
    path=args.root/'independent_audit.json'
    path.write_text(json.dumps(final,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(output=str(path),assessment={k:v['prespecified_assessment'] for k,v in results.items()},
        late_metric_refinement=scalar),indent=2))


if __name__=='__main__':
    main()
