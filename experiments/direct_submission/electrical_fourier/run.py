"""Direct-partial electrical study; run one seed into a NEW path.

All scientific modules are snapshotted and hashed before import. Failed tasks
retain their completed prefixes and have null full-horizon error metrics.
"""
from dataclasses import asdict, replace
from datetime import datetime, timezone
from pathlib import Path
import argparse
import hashlib
import importlib.util
import json
import platform
import sys
import time
import traceback

sys.dont_write_bytecode=True

import numpy as np
import scipy
from scipy.integrate import solve_ivp


HERE=Path(__file__).resolve().parent
SOURCE_FILES=('run.py','protocol.json','electrical_simulator.py','period_estimator.py','direct_learner.py')


def save_json(path,value):
    path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_snapshot(path,name):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec)
    sys.modules[name]=module
    spec.loader.exec_module(module)
    return module


def save_arrays(path,values):
    np.savez_compressed(path,**{k:v for k,v in values.items() if isinstance(v,np.ndarray)})


def voltage_training(t,cfg):
    p=cfg['training_voltage']
    return p['offset_v']+sum(a*np.sin(2*np.pi*f*t+ph) for a,f,ph in
            zip(p['amplitudes_v'],p['frequencies_hz'],p['phases_rad']))


def voltage_prediction(t,start,cfg):
    p=cfg['prediction_voltage'];s=t-start
    return (p['offset_v']+p['sin_amplitude_v']*np.sin(2*np.pi*p['sin_frequency_hz']*s+p['sin_phase_rad'])
            +p['cos_amplitude_v']*np.cos(2*np.pi*p['cos_frequency_hz']*s))


def control_reference(relative_time,cfg):
    p=cfg['control_reference'];t=np.asarray(relative_time)
    desired=p['offset_a']+sum(a*np.sin(2*np.pi*f*t) for a,f in zip(p['amplitudes_a'],p['frequencies_hz']))
    rate=sum(a*2*np.pi*f*np.cos(2*np.pi*f*t) for a,f in zip(p['amplitudes_a'],p['frequencies_hz']))
    return desired,rate


def stable_projector_error(rows,targets):
    """Explicit normalized projector difference, stable near exact recovery."""
    a=rows/np.linalg.norm(rows,axis=-1,keepdims=True)
    b=targets/np.linalg.norm(targets,axis=-1,keepdims=True)
    return np.linalg.norm(a[..., :,None]*a[...,None,:]-b[..., :,None]*b[...,None,:],axis=(-2,-1))


def guard_ratio(learner,weights,t,freq,order,index):
    row=learner.evaluate(weights,t,freq,order)
    length=np.linalg.norm(row)
    return float(abs(row[index])/length) if np.isfinite(length) and length>1e-12 else 0.


def phase_diagnostics(learner,weights,freq,order,cfg):
    count=cfg['phase_grid_points']
    phases=np.arange(count)*2*np.pi/count
    rows=learner.evaluate(weights,phases/(2*np.pi*freq),freq,order)
    lengths=np.linalg.norm(rows,axis=1)
    normalized=np.divide(np.abs(rows),lengths[:,None],out=np.zeros_like(rows),where=lengths[:,None]>1e-12)
    blocks=weights.reshape(2*order+1,4)
    amplitudes=sum((np.hypot(blocks[2*k-1],blocks[2*k]) for k in range(1,order+1)),start=np.zeros(4))
    abs_lower=np.maximum(0.,np.abs(blocks[0])-amplitudes)
    vector_amplitudes=sum(np.sqrt(np.sum(blocks[2*k-1]**2)+np.sum(blocks[2*k]**2)) for k in range(1,order+1))
    norm_upper=float(np.linalg.norm(blocks[0])+vector_amplitudes)
    norm_lower=float(max(0.,np.linalg.norm(blocks[0])-vector_amplitudes))
    lower=abs_lower/norm_upper if norm_upper>0 else np.zeros(4)
    return dict(grid_points=count,grid_min_row_norm=float(np.min(lengths)),
                grid_min_normalized_entries=np.min(normalized,axis=0).tolist(),
                conservative_row_norm_lower_bound=norm_lower,
                conservative_row_norm_upper_bound=norm_upper,
                conservative_absolute_entry_lower_bounds=abs_lower.tolist(),
                conservative_normalized_entry_lower_bounds=lower.tolist(),
                prediction_conservative_guard_certified=bool(lower[0]>=cfg['normalized_computed_entry_guard']),
                control_conservative_guard_certified=bool(lower[2]>=cfg['normalized_computed_entry_guard']),
                scope='Finite-grid minima are not continuum certificates; amplitude bounds are conservative and do not veto valid actual queries')


class QueryFailure(RuntimeError):
    def __init__(self,time_s,ratio,supplied_rates):
        self.detail=dict(time_s=float(time_s),normalized_computed_entry=float(ratio),supplied_rates=list(map(float,supplied_rates)))
        super().__init__('Normalized computed-entry guard failed')


def predict(output,name,learner,weights,order,freq,start,cfg,times):
    scales=np.array(cfg['scales']);guard=cfg['normalized_computed_entry_guard']
    states=[cfg['prediction_initial_current_a']];ratios=[guard_ratio(learner,weights,times[0],freq,order,0)]
    failure=None;solver_evaluations=0
    def rhs(t,y):
        supplied=[0.,float(y[0]),float(voltage_prediction(t,start,cfg)),1.]
        rate,okay=learner.complete(weights,t,freq,order,scales,supplied,0,guard)
        if not okay:
            raise QueryFailure(t,guard_ratio(learner,weights,t,freq,order,0),supplied)
        return [rate]
    for n in range(len(times)-1):
        try:
            solved=solve_ivp(rhs,(times[n],times[n+1]),[states[-1]],method='DOP853',t_eval=[times[n+1]],
                             rtol=cfg['prediction_solver_rtol'],atol=cfg['prediction_solver_atol'],max_step=cfg['task_dt_s'])
            solver_evaluations+=int(solved.nfev)
            if not solved.success or solved.y.shape!=(1,1) or not np.all(np.isfinite(solved.y)):
                raise RuntimeError(solved.message)
            states.append(float(solved.y[0,-1]));ratios.append(guard_ratio(learner,weights,times[n+1],freq,order,0))
        except QueryFailure as error:
            failure=dict(kind='guard',**error.detail);break
        except Exception as error:
            failure=dict(kind='solver',time_s=float(times[n]),message=str(error),traceback=traceback.format_exc());break
    used=times[:len(states)]
    np.savez_compressed(output/f'prediction_{name}.npz',time_s=used,time_relative_s=used-start,
                        predicted_current_a=np.array(states),applied_voltage_v=voltage_prediction(used,start,cfg),
                        guard_ratio=np.array(ratios))
    complete=failure is None and len(states)==len(times)
    return dict(status='complete' if complete else 'aborted',rmse_a=None,guard_failures=int(failure is not None and failure['kind']=='guard'),
                completed_samples=len(states),requested_samples=len(times),solver_nfev=solver_evaluations,
                minimum_completed_sample_guard_ratio=float(min(ratios)),failure=failure)


def control(output,name,simulator,parameters,learner,weights,order,freq,start,cfg):
    dt=cfg['task_dt_s'];count=round(cfg['task_duration_s']/dt)
    times=start+np.arange(count+1)*dt;scales=np.array(cfg['scales']);guard=cfg['normalized_computed_entry_guard']
    desired,desired_rate=control_reference(times-start,cfg)
    currents=[cfg['control_initial_current_a']];commands=[];requested=[];ratios=[];saturation_flags=[]
    failure=None
    for n,t in enumerate(times[:-1]):
        target_rate=desired_rate[n]+cfg['control_feedback_gain_per_s']*(desired[n]-currents[-1])
        supplied=[float(target_rate),float(currents[-1]),0.,1.]
        u,okay=learner.complete(weights,t,freq,order,scales,supplied,2,guard)
        ratio=guard_ratio(learner,weights,t,freq,order,2)
        if not okay:
            failure=dict(kind='guard',time_s=float(t),normalized_computed_entry=ratio,supplied_rates=supplied);break
        limited=float(np.clip(u,-cfg['control_voltage_limit_v'],cfg['control_voltage_limit_v']))
        try:
            current=simulator.step_current(currents[-1],t,dt,limited,parameters,
                                           max_substep=cfg['plant_max_substep_s'])
            if not np.isfinite(current):
                raise FloatingPointError('Nonfinite simulator current')
        except Exception as error:
            failure=dict(kind='plant_solver',time_s=float(t),attempted_voltage_v=limited,message=str(error),traceback=traceback.format_exc());break
        requested.append(float(u));commands.append(limited);ratios.append(ratio)
        saturation_flags.append(bool(abs(u)>cfg['control_voltage_limit_v']));currents.append(current)
    used=times[:len(currents)];current=np.array(currents)
    np.savez_compressed(output/f'control_{name}.npz',time_s=used,time_relative_s=used-start,current_a=current,
                        reference_current_a=desired[:len(currents)],reference_rate_a_per_s=desired_rate[:len(currents)],
                        command_time_s=times[:len(commands)],requested_voltage_v=np.array(requested),
                        applied_voltage_v=np.array(commands),guard_ratio=np.array(ratios),saturated=np.array(saturation_flags,dtype=bool))
    complete=failure is None and len(currents)==len(times)
    lo,hi=cfg['control_late_window_relative_s'];relative=used-start
    late=(relative>=lo-1e-10)&(relative<=hi+1e-10)
    full_rmse=float(np.sqrt(np.mean((current-desired)**2))) if complete else None
    late_rmse=float(np.sqrt(np.mean((current[late]-desired[late])**2))) if complete else None
    return dict(status='complete' if complete else 'aborted',full_rmse_a=full_rmse,late_rmse_a=late_rmse,
                guard_failures=int(failure is not None and failure['kind']=='guard'),saturations=int(sum(saturation_flags)),
                completed_samples=len(currents),requested_samples=len(times),failure=failure,
                minimum_completed_query_guard_ratio=float(min(ratios)) if ratios else None)


def execute(output,cfg,seed,simulator,estimator,learner):
    phase_rng=np.random.default_rng(seed+cfg['phase_rng_seed_offset'])
    learner_rng=np.random.default_rng(seed)
    rng_record=dict(phase_seed=seed+cfg['phase_rng_seed_offset'],learner_seed=seed,
                    phase_before=phase_rng.bit_generator.state,learner_before=learner_rng.bit_generator.state)
    parameters=replace(simulator.ElectricalParameters(),initial_phase_rad=float(phase_rng.uniform(0.,2*np.pi)))
    draw=learner_rng.normal(size=4*(2*cfg['harmonic_order']+1));c=draw/np.linalg.norm(draw)
    c0=draw[:4]/np.linalg.norm(draw[:4]);initial=c.copy();initial0=c0.copy()
    rng_record.update(phase_after=phase_rng.bit_generator.state,learner_after=learner_rng.bit_generator.state,
                      raw_initial_draw=draw.tolist(),initial_C=initial.tolist(),initial_C0=initial0.tolist())
    save_json(output/'initialization.json',rng_record)
    save_json(output/'simulator_only_parameters.json',asdict(parameters))
    sensor_seed=seed+cfg['sensor_rng_seed_offset'];dt=cfg['learner_dt_s'];tau=cfg['memory_time_s']
    cal,caltruth=simulator.generate_stream(dt,cfg['calibration_duration_s'],sensor_seed,
              lambda t:cfg['calibration_voltage_v'],parameters=parameters,initial_current=cfg['initial_calibration_current_a'],
              current_noise_std=cfg['current_noise_std_a'],current_rate_noise_std=cfg['current_rate_noise_std_a_per_s'])
    save_arrays(output/'calibration_observations.npz',cal);save_arrays(output/'calibration_evaluator.npz',caltruth)
    period=estimator.estimate_period(cal['time_s'],cal['current_a'],**cfg['period_estimator'])
    save_json(output/'period_estimate.json',period)
    if period.get('status')!='estimated' or period.get('ambiguous',True):
        return dict(status='calibration_aborted',reason='Unresolved or ambiguous observed period',tasks={},criteria_pass=False)
    freq=float(period['frequency_hz']);print(json.dumps(dict(stage='calibrated',frequency_hz=freq)),flush=True)
    start=cfg['calibration_duration_s']
    obs,truth=simulator.generate_stream(dt,cfg['training_duration_s'],sensor_seed,
              lambda t:voltage_training(t,cfg),parameters=parameters,initial_current=caltruth['current_a'][-1],start_time=start,
              current_noise_std=cfg['current_noise_std_a'],current_rate_noise_std=cfg['current_rate_noise_std_a_per_s'])
    t=obs['time_s'];rates=np.column_stack((obs['current_rate_a_per_s'],obs['current_a'],obs['applied_voltage_v'],np.ones(len(t))))
    C=np.empty((len(t),len(c)));C0=np.empty((len(t),len(c0)));R=np.zeros((len(c),len(c)));R0=np.zeros((len(c0),len(c0)))
    scales=np.array(cfg['scales']);order=cfg['harmonic_order'];failure=None;updates=dict(fourier=0,K0=0);retained=0
    for n in range(len(t)):
        C[n]=c;C0[n]=c0;retained=n+1
        if n==len(t)-1:break
        try:
            c,R=learner.memory_step(c,R,rates[n],t[n],freq,order=order,dt=dt,tau=tau,gain=cfg['gain'],scales=scales,
                                    rtol=cfg['learner_solver_rtol'],atol=cfg['learner_solver_atol']);updates['fourier']+=1
            c0,R0=learner.memory_step(c0,R0,rates[n],t[n],freq,order=0,dt=dt,tau=tau,gain=cfg['gain'],scales=scales,
                                      rtol=cfg['learner_solver_rtol'],atol=cfg['learner_solver_atol']);updates['K0']+=1
        except Exception as error:
            failure=dict(sample=n,time_s=float(t[n]),kind='learning_solver',message=str(error),traceback=traceback.format_exc());break
        if n and n%round(20/dt)==0:print(json.dumps(dict(stage='training',relative_seconds=n*dt)),flush=True)
    C=C[:retained];C0=C0[:retained];used=t[:retained];rates=rates[:retained]
    H=np.einsum('ni,nij->nj',learner.basis(used,freq,order),C.reshape(-1,2*order+1,4))
    energy=.5*np.sum(H*(rates/scales),axis=1)**2;energy0=.5*np.sum(C0*(rates/scales),axis=1)**2
    lengths=np.linalg.norm(H,axis=1);guard_curr=np.abs(H[:,0])/lengths;guard_volt=np.abs(H[:,2])/lengths
    np.savez_compressed(output/'training.npz',time_s=used,time_relative_s=used-start,observed_rates=rates,
                        C_pre=C,C0_pre=C0,R_final=R,R0_final=R0,C_final=c,C0_final=c0,
                        energy_fourier_pre=energy,energy_K0_pre=energy0,
                        evaluated_normalized_guard_current=guard_curr,evaluated_normalized_guard_voltage=guard_volt)
    target=truth['physical_differential'][:retained];eP=stable_projector_error(H,target*scales);eP0=stable_projector_error(C0,target*scales)
    evaluator={key:value[:retained] for key,value in truth.items() if isinstance(value,np.ndarray)}
    evaluator.update(projector_error_fourier=eP,projector_error_K0=eP0)
    save_arrays(output/'training_evaluator.npz',evaluator)
    late_lo,late_hi=cfg['training_metric_window_relative_s'];relative=used-start
    late=(relative>=late_lo-1e-10)&(relative<late_hi-1e-10)
    complete_training=failure is None and retained==len(t)
    summary=dict(status='complete' if complete_training else 'training_aborted',frequency_hz=freq,
                 training=dict(status='complete' if complete_training else 'aborted',updates_completed=updates,failure=failure,
                   late_projector_error_mean=float(np.mean(eP[late])) if complete_training else None,
                   late_K0_projector_error_mean=float(np.mean(eP0[late])) if complete_training else None,
                   late_energy_fourier_mean=float(np.mean(energy[late])) if complete_training else None,
                   late_energy_K0_mean=float(np.mean(energy0[late])) if complete_training else None,
                   final_projector_error=float(eP[-1]),final_K0_projector_error=float(eP0[-1]),
                   max_coefficient_norm_error=float(np.max(np.abs(np.linalg.norm(C,axis=1)-1))),
                   all_coefficient_total_variations=np.sum(np.abs(np.diff(C,axis=0)),axis=0).tolist(),
                   memory_eigenvalues=np.linalg.eigvalsh(R).tolist()),tasks={},full_cycle={})
    if not complete_training:
        summary['criteria_pass']=False;return summary
    task_start=float(t[-1])
    task_obs,task_truth=simulator.generate_stream(cfg['task_dt_s'],cfg['task_duration_s'],sensor_seed,
                 lambda s:voltage_prediction(s,task_start,cfg),parameters=parameters,
                 initial_current=cfg['prediction_initial_current_a'],start_time=task_start)
    save_arrays(output/'prediction_evaluator.npz',task_truth)
    for name,weights,k in [('fourier',c,order),('K0',c0,0)]:
        summary['full_cycle'][name]=phase_diagnostics(learner,weights,freq,k,cfg)
        prediction=predict(output,name,learner,weights,k,freq,task_start,cfg,task_obs['time_s'])
        if prediction['status']=='complete':
            with np.load(output/f'prediction_{name}.npz') as saved:
                prediction['rmse_a']=float(np.sqrt(np.mean((saved['predicted_current_a']-task_truth['current_a'])**2)))
        summary['tasks'][name]=dict(prediction=prediction,
                                   control=control(output,name,simulator,parameters,learner,weights,k,freq,task_start,cfg))
    criteria=cfg['criteria'];learned=summary['tasks']['fourier'];pred=learned['prediction'];ctrl=learned['control']
    checks=dict(late_field=summary['training']['late_projector_error_mean']<criteria['late_projector_error_mean_max'],
                prediction=pred['rmse_a'] is not None and pred['rmse_a']<criteria['prediction_rmse_a_max'],
                late_control=ctrl['late_rmse_a'] is not None and ctrl['late_rmse_a']<criteria['control_late_rmse_a_max'],
                prediction_guards=pred['guard_failures']<=criteria['learned_prediction_guard_failures_max'],
                control_guards=ctrl['guard_failures']<=criteria['learned_control_guard_failures_max'],
                control_saturations=ctrl['saturations']<=criteria['learned_control_saturations_max'],
                completed_tasks=pred['status']=='complete' and ctrl['status']=='complete')
    summary.update(criteria_checks=checks,criteria_pass=bool(all(checks.values())),
                   evaluator_frequency_error_hz=freq-parameters.frequency_hz,
                   task_start_time_s=task_start,
                   information_boundary='True parameters/partials belong only to simulator/evaluator; learned frequency uses calibration time/current alone')
    return summary


def run(output,seed,learner_dt=None,plant_substep=None):
    # No scientific module has been imported before these immutable copies.
    output=output.resolve();output.mkdir(parents=True,exist_ok=False);source=output/'source';source.mkdir()
    for name in SOURCE_FILES:(source/name).write_bytes((HERE/name).read_bytes())
    save_json(output/'source_manifest.json',{name:digest(source/name) for name in SOURCE_FILES})
    cfg=json.loads((source/'protocol.json').read_text())
    if seed in cfg['confirmation_seeds'] and cfg['status']!='SEALED':
        raise ValueError('Confirmation seeds cannot run until the audited protocol is explicitly SEALED')
    actual_dt=cfg['learner_dt_s'] if learner_dt is None else float(learner_dt)
    actual_sub=cfg['plant_max_substep_s'] if plant_substep is None else float(plant_substep)
    if actual_dt not in cfg['allowed_learner_dt_s'] or actual_sub not in cfg['allowed_plant_max_substep_s']:
        raise ValueError('Undeclared numerical refinement; create a new protocol identity')
    cfg['learner_dt_s']=actual_dt;cfg['plant_max_substep_s']=actual_sub
    suffix=f"__learner_dt_{actual_dt:.7f}__plant_substep_{actual_sub:.7f}"
    cfg['run_identity']=cfg['identity']+suffix;cfg['seed']=int(seed)
    save_json(output/'effective_protocol.json',cfg)
    save_json(output/'environment.json',dict(python=sys.version,numpy=np.__version__,scipy=scipy.__version__,
                 platform=platform.platform(),command=sys.argv,started_utc=datetime.now(timezone.utc).isoformat()))
    started=time.monotonic()
    simulator=load_snapshot(source/'electrical_simulator.py','frozen_electrical_simulator')
    estimator=load_snapshot(source/'period_estimator.py','frozen_period_estimator')
    learner=load_snapshot(source/'direct_learner.py','frozen_direct_learner')
    try:
        result=execute(output,cfg,seed,simulator,estimator,learner)
    except Exception as error:
        result=dict(status='execution_failed',criteria_pass=False,message=str(error),traceback=traceback.format_exc())
    result.update(protocol_identity=cfg['run_identity'],protocol_status=cfg['status'],seed=seed,elapsed_seconds=time.monotonic()-started)
    save_json(output/'summary.json',result)
    save_json(output/'manifest.json',{str(path.relative_to(output)):digest(path) for path in sorted(output.rglob('*')) if path.is_file()})
    print(json.dumps(result,indent=2),flush=True)
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--seed',type=int,required=True)
    parser.add_argument('--learner-dt',type=float)
    parser.add_argument('--plant-substep',type=float)
    args=parser.parse_args()
    result=run(args.output,args.seed,args.learner_dt,args.plant_substep)
    if result['status']=='execution_failed':raise SystemExit(1)
