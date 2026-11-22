"""Independent raw-record audit; never imports the run's scientific modules.

Checks hashes, all recorded metrics, the full memory recurrence, selected
projected-flow steps, physical identities, and independent task propagation.
The optional oracle is an evaluator-only diagnostic written outside raw runs.
"""
from pathlib import Path
import argparse
import hashlib
import json

import numpy as np
from scipy.integrate import solve_ivp


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text())


def harmonics(t, f, k):
    angle=2*np.pi*f*np.asarray(t)
    return np.stack([np.ones_like(angle)]+[fun(j*angle) for j in range(1,k+1)
                    for fun in (np.cos,np.sin)],axis=-1)


def evaluated(c,t,f,k):
    return harmonics(t,f,k)@np.asarray(c).reshape(2*k+1,4)


def emf(t,p):
    phase=2*np.pi*p['frequency_hz']*np.asarray(t)+p['initial_phase_rad']
    return (p['emf_first_v']*np.sin(phase+p['emf_first_phase_rad'])+
            p['emf_second_v']*np.sin(2*phase+p['emf_second_phase_rad']))


def exact_held_step(current,t,h,voltage,p):
    """Exact independent voltage-held solution of L idot+R i+e(t)=u."""
    decay_rate=p['resistance_ohm']/p['inductance_h']
    decay=np.exp(-decay_rate*h)
    result=decay*current+voltage/p['resistance_ohm']*(-np.expm1(-decay_rate*h))
    for harmonic,amp,phase in [(1,p['emf_first_v'],p['emf_first_phase_rad']),
                               (2,p['emf_second_v'],p['emf_second_phase_rad'])]:
        omega=harmonic*2*np.pi*p['frequency_hz']
        initial=omega*t+harmonic*p['initial_phase_rad']+phase
        convolution=np.imag(np.exp(1j*initial)*(np.exp(1j*omega*h)-decay)/(decay_rate+1j*omega))
        result-=amp/p['inductance_h']*convolution
    return result


def exact_prediction(t,start,initial,p,cfg):
    """Exact independent response to the declared smooth prediction voltage."""
    s=np.asarray(t)-start;a=p['resistance_ohm']/p['inductance_h'];decay=np.exp(-a*s)
    v=cfg['prediction_voltage']
    current=initial*decay+v['offset_v']/p['resistance_ohm']*(-np.expm1(-a*s))
    components=[(v['sin_amplitude_v'],v['sin_frequency_hz'],v['sin_phase_rad']),
                (v['cos_amplitude_v'],v['cos_frequency_hz'],np.pi/2),
                (-p['emf_first_v'],p['frequency_hz'],2*np.pi*p['frequency_hz']*start+p['initial_phase_rad']+p['emf_first_phase_rad']),
                (-p['emf_second_v'],2*p['frequency_hz'],4*np.pi*p['frequency_hz']*start+2*p['initial_phase_rad']+p['emf_second_phase_rad'])]
    for amp,f,phase in components:
        omega=2*np.pi*f
        current+=amp/p['inductance_h']*np.imag(np.exp(1j*phase)*(np.exp(1j*omega*s)-decay)/(a+1j*omega))
    return current


def audit(run,oracle=False):
    run=run.resolve();checks=[];metrics={};oracle_arrays=None
    def require(name,condition,**detail):
        checks.append(dict(name=name,passed=bool(condition),**detail))
    def close(name,a,b,atol=1e-10,rtol=1e-8):
        aa,bb=np.asarray(a),np.asarray(b)
        shape=aa.shape==bb.shape
        error=float(np.max(np.abs(aa-bb))) if shape and aa.size else 0.
        require(name,shape and np.all(np.isfinite(aa)) and np.all(np.isfinite(bb)) and np.allclose(aa,bb,atol=atol,rtol=rtol),
                maximum_absolute_error=error,absolute_tolerance=atol,relative_tolerance=rtol)
    manifest=read(run/'manifest.json');source=read(run/'source_manifest.json')
    for name,expected in manifest.items():
        path=run/name;require('record_hash:'+name,path.is_file() and sha(path)==expected)
    for name,expected in source.items():
        path=run/'source'/name;require('source_hash:'+name,path.is_file() and sha(path)==expected)
    required={'run.py','protocol.json','electrical_simulator.py','period_estimator.py','direct_learner.py'}
    require('scientific_source_set',required.issubset(source))
    cfg=read(run/'effective_protocol.json');sealed=read(run/'source'/'protocol.json');summary=read(run/'summary.json')
    allowed_changes={'learner_dt_s','plant_max_substep_s','run_identity','seed'}
    require('effective_protocol_changes_declared',all(cfg.get(k)==v for k,v in sealed.items() if k not in allowed_changes))
    require('declared_numerical_refinement',cfg['learner_dt_s'] in cfg['allowed_learner_dt_s'] and cfg['plant_max_substep_s'] in cfg['allowed_plant_max_substep_s'])
    require('confirmation_seal',cfg['seed'] not in cfg['confirmation_seeds'] or cfg['status']=='SEALED')
    p=read(run/'simulator_only_parameters.json');init=read(run/'initialization.json');period=read(run/'period_estimate.json')
    phase_rng=np.random.default_rng(cfg['seed']+cfg['phase_rng_seed_offset'])
    close('phase_rng_reconstruction',p['initial_phase_rad'],phase_rng.uniform(0,2*np.pi),atol=0,rtol=0)
    draw=np.random.default_rng(cfg['seed']).normal(size=4*(2*cfg['harmonic_order']+1))
    close('initial_draw_reconstruction',init['raw_initial_draw'],draw,atol=0,rtol=0)
    close('initial_C_reconstruction',init['initial_C'],draw/np.linalg.norm(draw),atol=0,rtol=0)
    close('initial_C0_reconstruction',init['initial_C0'],draw[:4]/np.linalg.norm(draw[:4]),atol=0,rtol=0)
    cal=np.load(run/'calibration_observations.npz');caltruth=np.load(run/'calibration_evaluator.npz')
    close('calibration_voltage',cal['applied_voltage_v'],np.full(len(cal['time_s']),cfg['calibration_voltage_v']))
    require('calibration_precedes_training',abs(cal['time_s'][-1]-cfg['calibration_duration_s'])<1e-10)
    if summary['status']=='calibration_aborted':
        require('calibration_abort_reason',period.get('status')!='estimated' or period.get('ambiguous',True))
        require('calibration_abort_has_no_task_claims',summary['tasks']=={} and not summary['criteria_pass'])
        return dict(run=str(run),audit_source_sha256=sha(Path(__file__)),integrity_pass=all(x['passed'] for x in checks),
                    run_status=summary['status'],study_criteria_pass=False,checks=checks,metrics=metrics),None
    require('calibration_resolved',period['status']=='estimated' and not period['ambiguous'])
    require('frequency_fit_nyquist',period['fit_nyquist_hz']>max(period['search_hz'])*period['fitted_harmonics'])
    close('calibration_selected_frequency',summary['frequency_hz'],period['frequency_hz'],atol=0,rtol=0)
    require('calibration_candidates_preserved',len(period['candidates'])==period['candidate_count'])
    a=np.load(run/'training.npz');truth=np.load(run/'training_evaluator.npz')
    t=a['time_s'];relative=a['time_relative_s'];rates=a['observed_rates'];scales=np.array(cfg['scales']);f=period['frequency_hz'];k=cfg['harmonic_order']
    close('training_clock',relative,t-cfg['calibration_duration_s'])
    close('training_clock_rate',rates[:,3],np.ones(len(t)),atol=0,rtol=0)
    vp=cfg['training_voltage'];u=vp['offset_v']+sum(amp*np.sin(2*np.pi*fr*t+ph) for amp,fr,ph in zip(vp['amplitudes_v'],vp['frequencies_hz'],vp['phases_rad']))
    close('training_exogenous_voltage',rates[:,2],u)
    actual_F=np.column_stack((np.full(len(t),p['inductance_h']),np.full(len(t),p['resistance_ohm']),-np.ones(len(t)),emf(t,p)))
    close('physical_partials_from_saved_parameters',truth['physical_differential'],actual_F)
    if cfg['current_noise_std_a']==0 and cfg['current_rate_noise_std_a_per_s']==0:
        close('ideal_observation_boundary',rates[:,:3],np.column_stack((truth['current_rate_a_per_s'],truth['current_a'],truth['applied_voltage_v'])))
        close('physical_compatibility',np.sum(actual_F*rates,axis=1),np.zeros(len(t)),atol=2e-12)
        close('calibration_ideal_observation',cal['current_a'],caltruth['current_a'],atol=0,rtol=0)
    target=actual_F*scales;target_unit=target/np.linalg.norm(target,axis=1)[:,None]
    late=(relative>=cfg['training_metric_window_relative_s'][0]-1e-10)&(relative<cfg['training_metric_window_relative_s'][1]-1e-10)
    training_complete=summary['training']['status']=='complete'
    branch={}
    for name,Ckey,Rkey,finalkey,order in [('fourier','C_pre','R_final','C_final',k),('K0','C0_pre','R0_final','C0_final',0)]:
        C=a[Ckey];h=harmonics(t,f,order);rows=np.einsum('ni,nij->nj',h,C.reshape(-1,2*order+1,4))
        unit=rows/np.linalg.norm(rows,axis=1)[:,None]
        # Independent angle expression instead of explicit projector subtraction.
        residual=unit-np.sum(unit*target_unit,axis=1)[:,None]*target_unit
        errors=np.sqrt(2)*np.linalg.norm(residual,axis=1)
        energy=.5*np.square(np.einsum('ni,ni->n',rows,rates/scales))
        close(name+':projector_error_array',truth['projector_error_'+name],errors,atol=2e-12)
        close(name+':preupdate_energy_array',a['energy_'+name+'_pre'],energy,atol=1e-12)
        if name=='fourier':
            close('training_current_guard_array',a['evaluated_normalized_guard_current'],np.abs(rows[:,0])/np.linalg.norm(rows,axis=1))
            close('training_voltage_guard_array',a['evaluated_normalized_guard_voltage'],np.abs(rows[:,2])/np.linalg.norm(rows,axis=1))
        close(name+':unit_coefficient_rows',np.linalg.norm(C,axis=1),np.ones(len(t)),atol=2e-12,rtol=0)
        close(name+':initial_coefficients',C[0],init['initial_C' if name=='fourier' else 'initial_C0'],atol=0,rtol=0)
        if training_complete:
            close(name+':final_coefficients',a[finalkey],C[-1],atol=0,rtol=0)
            close(name+':late_projector_metric',summary['training']['late_projector_error_mean' if name=='fourier' else 'late_K0_projector_error_mean'],np.mean(errors[late]),atol=2e-12)
            close(name+':late_energy_metric',summary['training']['late_energy_'+name+'_mean'],np.mean(energy[late]),atol=2e-12)
        close(name+':final_projector_metric',summary['training']['final_projector_error' if name=='fourier' else 'final_K0_projector_error'],errors[-1],atol=2e-12)
        dt=cfg['learner_dt_s'];tau=cfg['memory_time_s'];gain=cfg['gain'];R=np.zeros((C.shape[1],C.shape[1]));factor=np.exp(-dt/tau)
        updates=summary['training']['updates_completed'][name]
        selected=set(np.linspace(0,max(0,min(updates,len(C)-1)-1),5,dtype=int));step_errors=[]
        for n in range(updates):
            w=np.kron(h[n],rates[n]/scales)
            if n in selected and n+1<len(C):
                d=C.shape[1];outer=np.outer(w,w)
                def ode(s,state):
                    q=state[:d];M=state[d:].reshape(d,d)
                    return np.r_[-gain*(M@q-(q@M@q)*q),(outer-M/tau).ravel()]
                independent=solve_ivp(ode,(0,dt),np.r_[C[n],R.ravel()],method='DOP853',rtol=1e-11,atol=1e-13)
                require(name+':projected_ODE_solver:'+str(n),independent.success)
                error=float(np.max(np.abs(independent.y[:d,-1]-C[n+1])))
                step_errors.append(dict(index=int(n),maximum_absolute_error=error))
                require(name+':projected_ODE_step:'+str(n),error<2e-7,maximum_absolute_error=error)
            R=factor*R+tau*(1-factor)*np.outer(w,w)
        close(name+':full_memory_recurrence',a[Rkey],R,atol=2e-10,rtol=2e-10)
        metrics[name]=dict(late_projector_error_mean=float(np.mean(errors[late])) if np.any(late) else None,
                           late_energy_mean=float(np.mean(energy[late])) if np.any(late) else None,
                           selected_step_checks=step_errors)
        branch[name]=(a[finalkey],order)
    close('all_coefficient_total_variations',summary['training']['all_coefficient_total_variations'],np.sum(abs(np.diff(a['C_pre'],axis=0)),axis=0))
    close('max_coefficient_norm_error',summary['training']['max_coefficient_norm_error'],np.max(abs(np.linalg.norm(a['C_pre'],axis=1)-1)),atol=1e-14)
    close('memory_eigenvalues',summary['training']['memory_eigenvalues'],np.linalg.eigvalsh(a['R_final']),atol=1e-12)
    if training_complete:
        predtruth=np.load(run/'prediction_evaluator.npz');taskstart=summary['task_start_time_s'];taskdt=cfg['task_dt_s'];guard=cfg['normalized_computed_entry_guard']
        close('prediction_truth_independent_exact_solution',predtruth['current_a'],exact_prediction(predtruth['time_s'],taskstart,cfg['prediction_initial_current_a'],p,cfg),atol=2e-8,rtol=1e-8)
        for name,(weights,order) in branch.items():
            phase_t=np.arange(cfg['phase_grid_points'])/(cfg['phase_grid_points']*f);phase_rows=evaluated(weights,phase_t,f,order)
            norms=np.linalg.norm(phase_rows,axis=1);ratios=np.abs(phase_rows)/norms[:,None];diag=summary['full_cycle'][name]
            close(name+':fullcycle_grid_norm',diag['grid_min_row_norm'],norms.min())
            close(name+':fullcycle_grid_guard',diag['grid_min_normalized_entries'],ratios.min(axis=0))
            blocks=weights.reshape(2*order+1,4)
            amplitudes=np.zeros(4);vector_amplitudes=0.
            for harmonic in range(1,order+1):
                amplitudes+=np.hypot(blocks[2*harmonic-1],blocks[2*harmonic])
                vector_amplitudes+=np.sqrt(np.sum(blocks[2*harmonic-1]**2)+np.sum(blocks[2*harmonic]**2))
            abs_lower=np.maximum(0,np.abs(blocks[0])-amplitudes)
            norm_upper=np.linalg.norm(blocks[0])+vector_amplitudes
            norm_lower=max(0,np.linalg.norm(blocks[0])-vector_amplitudes)
            lower=abs_lower/norm_upper
            close(name+':fullcycle_conservative_norm_lower',diag['conservative_row_norm_lower_bound'],norm_lower)
            close(name+':fullcycle_conservative_norm_upper',diag['conservative_row_norm_upper_bound'],norm_upper)
            close(name+':fullcycle_conservative_entry_lower',diag['conservative_absolute_entry_lower_bounds'],abs_lower)
            close(name+':fullcycle_conservative_normalized_lower',diag['conservative_normalized_entry_lower_bounds'],lower)
            require(name+':fullcycle_certificate_flags',diag['prediction_conservative_guard_certified']==bool(lower[0]>=guard) and diag['control_conservative_guard_certified']==bool(lower[2]>=guard))
            for task in ('prediction','control'):
                data=np.load(run/(task+'_'+name+'.npz'));info=summary['tasks'][name][task]
                times=data['time_s'];require(name+':'+task+':sample_count',len(times)==info['completed_samples'])
                require(name+':'+task+':requested_samples',info['requested_samples']==round(cfg['task_duration_s']/taskdt)+1)
                require(name+':'+task+':clock_start',times[0]==taskstart)
                close(name+':'+task+':time_step',np.diff(times),np.full(max(0,len(times)-1),taskdt),atol=1e-12)
                if task=='prediction':
                    rows=evaluated(weights,times,f,order);gr=np.abs(rows[:,0])/np.linalg.norm(rows,axis=1)
                    close(name+':prediction_guard_ratios',data['guard_ratio'],gr)
                    close(name+':prediction_initial',data['predicted_current_a'][0],cfg['prediction_initial_current_a'],atol=0,rtol=0)
                    close(name+':prediction_min_guard_metric',info['minimum_completed_sample_guard_ratio'],gr.min())
                    if info['status']=='complete':
                        err=data['predicted_current_a']-predtruth['current_a'];rmse=float(np.sqrt(np.mean(err**2)))
                        close(name+':prediction_RMSE',info['rmse_a'],rmse)
                        require(name+':prediction_guard_pass',np.all(gr>=guard) and info['guard_failures']==0)
                        pv=cfg['prediction_voltage']
                        def rhs(s,y):
                            q=evaluated(weights,s,f,order)/scales
                            voltage=pv['offset_v']+pv['sin_amplitude_v']*np.sin(2*np.pi*pv['sin_frequency_hz']*(s-taskstart)+pv['sin_phase_rad'])+pv['cos_amplitude_v']*np.cos(2*np.pi*pv['cos_frequency_hz']*(s-taskstart))
                            return [-(q[1]*y[0]+q[2]*voltage+q[3])/q[0]]
                        replay=solve_ivp(rhs,(times[0],times[-1]),[cfg['prediction_initial_current_a']],t_eval=times,method='RK45',rtol=1e-11,atol=1e-13,max_step=taskdt)
                        require(name+':prediction_independent_solver',replay.success and replay.y.shape==(1,len(times)))
                        close(name+':prediction_independent_propagation',data['predicted_current_a'],replay.y[0],atol=2e-7,rtol=2e-7)
                        metrics[name]['prediction_rmse_a']=rmse
                    else:
                        require(name+':prediction_abort_null',info['rmse_a'] is None and info['failure'] is not None)
                else:
                    current=data['current_a'];reference=data['reference_current_a'];rate=data['reference_rate_a_per_s'];ct=data['command_time_s']
                    require(name+':control_command_count',len(ct)==len(current)-1)
                    close(name+':control_initial',current[0],cfg['control_initial_current_a'],atol=0,rtol=0)
                    rr=cfg['control_reference'];rel=times-taskstart
                    desired=rr['offset_a']+sum(amp*np.sin(2*np.pi*fr*rel) for amp,fr in zip(rr['amplitudes_a'],rr['frequencies_hz']))
                    desiredrate=sum(amp*2*np.pi*fr*np.cos(2*np.pi*fr*rel) for amp,fr in zip(rr['amplitudes_a'],rr['frequencies_hz']))
                    close(name+':control_reference',reference,desired);close(name+':control_reference_rate',rate,desiredrate)
                    if len(ct):
                        rows=evaluated(weights,ct,f,order);gr=np.abs(rows[:,2])/np.linalg.norm(rows,axis=1)
                        query=np.column_stack((rate[:-1]+cfg['control_feedback_gain_per_s']*(reference[:-1]-current[:-1]),current[:-1],np.zeros(len(ct)),np.ones(len(ct))))
                        requested=-np.sum(rows*(query/scales),axis=1)/rows[:,2]*scales[2]
                        applied=np.clip(requested,-cfg['control_voltage_limit_v'],cfg['control_voltage_limit_v'])
                        close(name+':control_requested_readout',data['requested_voltage_v'],requested)
                        close(name+':control_clipping',data['applied_voltage_v'],applied)
                        close(name+':control_guard_ratios',data['guard_ratio'],gr)
                        require(name+':control_completed_guards',np.all(gr>=guard))
                        close(name+':control_min_guard_metric',info['minimum_completed_query_guard_ratio'],gr.min())
                        saturated=abs(requested)>cfg['control_voltage_limit_v']
                        require(name+':control_saturation_flags',np.array_equal(data['saturated'],saturated) and info['saturations']==int(saturated.sum()))
                        exact=exact_held_step(current[:-1],ct,taskdt,applied,p)
                        close(name+':every_control_plant_transition_exact',current[1:],exact,atol=2e-8,rtol=1e-8)
                    if info['status']=='complete':
                        lo,hi=cfg['control_late_window_relative_s'];late=(rel>=lo-1e-10)&(rel<=hi+1e-10)
                        full=float(np.sqrt(np.mean((current-reference)**2)));lateerr=float(np.sqrt(np.mean((current[late]-reference[late])**2)))
                        close(name+':control_full_RMSE',info['full_rmse_a'],full);close(name+':control_late_RMSE',info['late_rmse_a'],lateerr)
                        require(name+':control_complete_samples',len(current)==round(cfg['task_duration_s']/taskdt)+1 and info['guard_failures']==0)
                        metrics[name].update(control_full_rmse_a=full,control_late_rmse_a=lateerr)
                    else:
                        require(name+':control_abort_null',info['full_rmse_a'] is None and info['late_rmse_a'] is None and info['failure'] is not None)
                failure=info['failure']
                require(name+':'+task+':guard_failure_count',info['guard_failures']==int(failure is not None and failure['kind']=='guard'))
                if failure is not None and failure['kind']=='guard':
                    failed_row=evaluated(weights,failure['time_s'],f,order);index=0 if task=='prediction' else 2
                    failed_ratio=abs(failed_row[index])/np.linalg.norm(failed_row)
                    close(name+':'+task+':failed_query_ratio',failure['normalized_computed_entry'],failed_ratio)
                    require(name+':'+task+':failed_query_guard',failed_ratio<guard or np.linalg.norm(failed_row)<=1e-12)
            if oracle and name=='fourier':
                data=np.load(run/'control_fourier.npz');times=data['time_s'];reference=data['reference_current_a'];rate=data['reference_rate_a_per_s']
                currents=np.empty(len(times));currents[0]=cfg['control_initial_current_a'];commands=np.empty(len(times)-1);sat=0
                for n,time in enumerate(times[:-1]):
                    target=rate[n]+cfg['control_feedback_gain_per_s']*(reference[n]-currents[n])
                    command=p['inductance_h']*target+p['resistance_ohm']*currents[n]+emf(time,p)
                    sat+=abs(command)>cfg['control_voltage_limit_v'];commands[n]=np.clip(command,-cfg['control_voltage_limit_v'],cfg['control_voltage_limit_v'])
                    currents[n+1]=exact_held_step(currents[n],time,taskdt,commands[n],p)
                lo,hi=cfg['control_late_window_relative_s'];rel=times-taskstart;late=(rel>=lo-1e-10)&(rel<=hi+1e-10)
                metrics['oracle_control_diagnostic']=dict(scope='Evaluator-only true-field completion; independent trajectory, identical reference/gain/voltage bound/hold, exact plant propagation; never a learner input',
                    full_rmse_a=float(np.sqrt(np.mean((currents-reference)**2))),late_rmse_a=float(np.sqrt(np.mean((currents[late]-reference[late])**2))),saturations=int(sat),
                    learned_minus_oracle_current_rms_a=float(np.sqrt(np.mean((data['current_a']-currents)**2))),
                    learned_minus_oracle_current_max_abs_a=float(np.max(abs(data['current_a']-currents))),
                    learned_minus_oracle_applied_voltage_rms_v=float(np.sqrt(np.mean((data['applied_voltage_v']-commands)**2))))
                oracle_arrays=dict(time_s=times,reference_current_a=reference,oracle_current_a=currents,oracle_applied_voltage_v=commands,
                    learned_minus_oracle_current_a=data['current_a']-currents,learned_minus_oracle_applied_voltage_v=data['applied_voltage_v']-commands)
        c=cfg['criteria'];m=metrics['fourier'];pred=summary['tasks']['fourier']['prediction'];ctrl=summary['tasks']['fourier']['control']
        independent_criteria=dict(late_field=m['late_projector_error_mean']<c['late_projector_error_mean_max'],
            prediction=m.get('prediction_rmse_a',np.inf)<c['prediction_rmse_a_max'],late_control=m.get('control_late_rmse_a',np.inf)<c['control_late_rmse_a_max'],
            prediction_guards=pred['guard_failures']<=c['learned_prediction_guard_failures_max'],control_guards=ctrl['guard_failures']<=c['learned_control_guard_failures_max'],
            control_saturations=ctrl['saturations']<=c['learned_control_saturations_max'],completed_tasks=pred['status']=='complete' and ctrl['status']=='complete')
        require('criteria_checks_reconstructed',independent_criteria==summary['criteria_checks'])
        require('criteria_pass_reconstructed',bool(all(independent_criteria.values()))==summary['criteria_pass'])
        close('evaluator_frequency_error',summary['evaluator_frequency_error_hz'],f-p['frequency_hz'],atol=0,rtol=0)
    return dict(run=str(run),audit_source_sha256=sha(Path(__file__)),integrity_pass=all(x['passed'] for x in checks),
                run_status=summary['status'],study_criteria_pass=summary['criteria_pass'],checks=checks,metrics=metrics,
                scope='Hashes and independent numerical reconstruction; selected adaptive steps, not every ODE transition; no theorem or hardware validation'),oracle_arrays


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True);parser.add_argument('--oracle-control',action='store_true');args=parser.parse_args()
    if args.output.exists():raise FileExistsError(args.output)
    report,arrays=audit(args.run,args.oracle_control)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    if arrays is not None:
        archive=args.output.with_name(args.output.stem+'_oracle.npz')
        if archive.exists():raise FileExistsError(archive)
        np.savez_compressed(archive,**arrays);report['oracle_diagnostic_archive']=dict(path=str(archive),sha256=sha(archive))
    args.output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(integrity_pass=report['integrity_pass'],study_criteria_pass=report['study_criteria_pass'],checks=len(report['checks']),
                         failed=[item for item in report['checks'] if not item['passed']],metrics=report['metrics']),indent=2))
    if not report['integrity_pass']:raise SystemExit(1)
