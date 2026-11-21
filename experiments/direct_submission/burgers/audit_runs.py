#!/usr/bin/env python3
"""Independent streaming audit. No experiment runner, sensor, or learner imports.
Reads immutable completed cases; writes evaluator-only audit artifacts.
"""
from __future__ import annotations
import argparse, copy, hashlib, importlib.util, json
from pathlib import Path
import numpy as np

HERE=Path(__file__).resolve().parent
PACKAGE=HERE
PRIOR=PACKAGE/'independent_math.py'
spec=importlib.util.spec_from_file_location('independent_math', PRIOR)
math=importlib.util.module_from_spec(spec); spec.loader.exec_module(math)
sha=math.sha; spectral=math.spectral; reference=math.reference
Fphysical=math.physical_field; distance=math.projector_distance; summary=math.finite_summary
rms=math.rms; maximum=math.maximum
WINDOWS={'training_evaluation':(76.,80.),'control':(80.,100.),'late':(96.,100.)}
AUDIT_SOURCE_BYTES=Path(__file__).read_bytes()
AUDIT_SOURCE_HASH=hashlib.sha256(AUDIT_SOURCE_BYTES).hexdigest()


def dump(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


def discrepancy(checks,key,a,b):
    a,b=np.asarray(a),np.asarray(b)
    if a.size==b.size==0:
        checks[key]=0.;return
    if a.shape!=b.shape: raise AssertionError(f'{key}: shape {a.shape} != {b.shape}')
    if not np.array_equal(np.isnan(a),np.isnan(b)): raise AssertionError(f'{key}: NaN mask differs')
    if not np.array_equal(np.isinf(a),np.isinf(b)): raise AssertionError(f'{key}: infinity mask differs')
    finite=np.isfinite(a)&np.isfinite(b)
    checks[key]=max(checks.get(key,0.), maximum(a[finite]-b[finite]))


def exact(checks,key,a,b):
    if not np.array_equal(a,b,equal_nan=True): raise AssertionError(f'{key}: exact invariant fails')
    checks[key]=True


def rhs(w,u,cfg):
    ws,wss=spectral(w); ca,nu,b=cfg['plant_coefficients']
    return -ca*w*ws+nu*wss+b*u


def advance(w,u,du,cfg):
    h=cfg['dt']; k1=rhs(w,u,cfg); k2=rhs(w+h*k1/2,u+h*du/2,cfg)
    k3=rhs(w+h*k2/2,u+h*du/2,cfg); k4=rhs(w+h*k3,u+h*du,cfg)
    return w+h*(k1+2*k2+2*k3+k4)/6,u+h*du


def weights(cfg):
    count=round(cfg['rate_fit_window']/cfg['dt'])+1
    r=np.arange(-(count-1),1)*cfg['dt']/cfg['rate_fit_window']
    X=np.column_stack((np.ones(count),r,r*r))
    Q,R=np.linalg.qr(X,mode='reduced')
    return np.linalg.solve(R,Q.T)[1]/cfg['rate_fit_window']


def fitted(values,history,coef):
    prior=len(history); combined=np.concatenate((history,values)) if prior else values
    out=np.full_like(values,np.nan); count=len(coef)
    if len(combined)>=count:
        windows=np.lib.stride_tricks.sliding_window_view(combined,count,axis=0)
        first=max(0,count-1-prior)
        starts=np.arange(first,len(values))+prior-count+1
        out[first:]=np.einsum('nsc,c->ns',windows[starts],coef)
    return out,combined[-(count-1):].copy()


def context_secants(z,valid,history,cfg):
    lag=round(cfg['window']/cfg['dt']); values=z[valid]
    prior=len(history); combined=np.concatenate((history,values)) if prior else values
    out=np.full_like(z,np.nan)
    positions=np.flatnonzero(valid); slots=np.arange(len(values))+prior
    ready=slots>=lag
    out[positions[ready]]=(combined[slots[ready]]-combined[slots[ready]-lag])/cfg['window']
    return out,combined[-(lag+1):].copy()


def initial_state(cfg):
    n=cfg['grid_points']; q=np.random.default_rng(cfg['initial_seed']).normal(size=(n,5));q/=np.linalg.norm(q,axis=-1,keepdims=True)
    return dict(q=q.copy(),initial=q.copy(),R=np.zeros((n,5,5)),Rc=np.zeros((n,5,5)),
        history=np.empty((0,n,5)),clean_history=np.empty((0,n,5)),
        fit_history=np.empty((0,n)),clean_fit_history=np.empty((0,n)),rng=np.random.default_rng(cfg['sensor_seed']),
        w=.2+.015*np.sin(2*np.pi*np.arange(n)/n),u=np.zeros(n),index=0)


def checkpoint_checks(path,state,checks,prefix,causal):
    with np.load(path.with_suffix('.npz')) as d:
        for key,sk in [('w','w'),('u','u'),('q','q'),('R','R'),('initial_q','initial'),('history','history')]:
            discrepancy(checks,prefix+'_'+key,d[key],state[sk])
        exact(checks,prefix+'_index',d['index'],state['index'])
        if causal:
            # Stored estimator retains N samples; audit keeps N-1 for next convolution.
            saved=d['rate_history']; retained=state['fit_history']
            if len(retained): discrepancy(checks,prefix+'_fit_history',saved[-len(retained):],retained)
    meta=json.loads(path.with_suffix('.json').read_text())
    if meta['rng_state']!=state['rng'].bit_generator.state: raise AssertionError(prefix+' RNG mismatch')
    checks[prefix+'_rng']=True


def concatenate(parts): return {k:np.concatenate(v) for k,v in parts.items() if v}
def collect(parts,key,value): parts.setdefault(key,[]).append(np.asarray(value))


def metrics(parts,cov,cfg,interval):
    if not parts: return None
    d=concatenate(parts); n=len(d['time']); low,high=interval
    if not n:return None
    e=d['field']; total=d['sensor_total']; bias=d['sensor_bias']; noise=d['sensor_noise']
    mt,mb,mn=(float(np.mean(x*x)) for x in (total,bias,noise)); cross=float(np.mean(bias*noise))
    out=dict(interval=list(interval),samples=n,spatial_points=cfg['grid_points'],
        expected_samples=round((high-low)/cfg['dt']),full_interval_observed=n==round((high-low)/cfg['dt']),
        first_observed_time=float(d['time'][0]),last_observed_time=float(d['time'][-1]),right_endpoint_excluded=True,
        field=summary(e),field_p99=float(np.quantile(e,.99)),field_worst_site_mean=float(np.max(np.mean(e,axis=0))),
        field_worst_site_index=int(np.argmax(np.mean(e,axis=0))),
        field_per_site_mean=np.mean(e,axis=0).tolist(),initial_shadow_field_mean=float(np.mean(d['initial_field'])),
        learned_to_initial_error_ratio=float(np.mean(e)/np.mean(d['initial_field'])),
        task_global_rms=rms(d['task_error']),observed_task_global_rms=rms(d['observed_task_error']),
        guard_fraction=float(np.mean(d['guard'])),clipping_fraction=float(np.mean(d['clip'])),
        guard_count=int(np.sum(d['guard'])),clipping_count=int(np.sum(d['clip'])),query_count=int(e.size),
        input_partial_abs_min=float(np.min(d['input_partial_abs'])),input_global_rms=rms(d['input']),input_rate_global_rms=rms(d['input_rate']),
        observed_secant_global_rms=rms(d['secant']),incompatibility_half_mean_square=float(.5*np.mean(d['incompat']**2)),
        physical_row_secant_half_mean_square=float(.5*np.mean(d['physical_incompat']**2)),
        rate_sensor=dict(total_rms=rms(total),clean_fit_bias_rms=rms(bias),noise_contribution_rms=rms(noise),
          total_mean=float(np.mean(total)),bias_mean=float(np.mean(bias)),noise_mean=float(np.mean(noise)),
          total_variance=float(np.var(total)),bias_variance=float(np.var(bias)),noise_variance=float(np.var(noise)),
          mse_total=mt,mse_bias=mb,mse_noise=mn,cross_second_moment=cross,
          noise_theoretical_variance=float(cfg['position_noise_std']**2*np.sum(weights(cfg)**2)) if cfg['sensing_mode']=='causal' else 0.,
          mse_identity_error=abs(mt-mb-mn-2*cross)))
    out['covariance']={k:summary(np.concatenate(v)) for k,v in cov.items() if v}
    out['covariance']['per_site_means']={k:np.mean(np.concatenate(v).reshape(-1,cfg['grid_points']),axis=0).tolist() for k,v in cov.items() if v}
    out['covariance']['snapshot_interval']=.05
    if 'readout_field' in d:
        accepted=~d['guard']; fe=d['readout_field'][accepted];se=d['readout_sensing'][accepted]
        out['readout']=dict(accepted_count=int(np.sum(accepted)),query_count=int(accepted.size),
            field_error_rms=rms(fe),sensing_error_rms=rms(se),total_raw_error_rms=rms((d['readout_field']+d['readout_sensing'])[accepted]),
            total_applied_error_rms=rms(d['readout_applied'][accepted]),total_applied_error_rms_all=rms(d['readout_applied']),
            field_sensing_cross_second_moment=float(np.mean(fe*se)) if len(fe) else None,
            decomposition_identity_error=maximum(d['readout_total']-d['readout_field']-d['readout_sensing']))
    return out


def audit_phase(casepath,label,cfg,state,ode_enabled):
    directory=casepath/label; chunks=sorted((directory/'chunks').glob('chunk_*.npz'))
    if not chunks: chunks=sorted(directory.glob('chunk_*.npz'))
    phase_summary=json.loads((directory/'summary.json').read_text()) if (directory/'summary.json').exists() else {}
    checks={}; accum={k:{} for k in WINDOWS}; covacc={k:{} for k in WINDOWS}; compact=[]; ode=[]
    coef=weights(cfg); beta=np.exp(-cfg['dt']/cfg['memory']); alpha=cfg['memory']*(-np.expm1(-cfg['dt']/cfg['memory']))
    first=None; count=0; applied_count=0; max_subdiv=0; worst_ode=None
    for path in chunks:
        with np.load(path) as saved: d={k:saved[k] for k in saved.files}
        if not len(d['time']):continue
        t=d['time']; idx=d['index']; q=d['q_pre']; z=d['context']; n=len(t);valid=d['rate_valid']; app=d['command_applied']
        if first is None:first={k:d[k][0].copy() for k in ('q_pre','context','udot','sensor_innovation')}
        exact(checks,'contiguous_indices',idx,np.arange(state['index'],state['index']+n))
        discrepancy(checks,'time',t,idx*cfg['dt'])
        innovation=state['rng'].standard_normal((n,cfg['grid_points']))
        discrepancy(checks,'sensor_innovation',d['sensor_innovation'],innovation)
        discrepancy(checks,'sensor_noise',d['sensor_noise'],cfg['position_noise_std']*innovation)
        discrepancy(checks,'measured_w',d['measured_w'],d['true_w']+d['sensor_noise'])
        discrepancy(checks,'initial_w',d['true_w'][0],state['w']);discrepancy(checks,'initial_u',z[0,:,3],state['u'])
        discrepancy(checks,'initial_q',q[0],state['q'])
        if n>1:discrepancy(checks,'q_continuity',q[1:],d['q_post'][:-1])
        discrepancy(checks,'q_unit',np.linalg.norm(q,axis=-1),np.ones(q.shape[:-1]))
        discrepancy(checks,'q_post_unit',np.linalg.norm(d['q_post'],axis=-1),np.ones(q.shape[:-1]))
        if np.any(~app[:-1]):raise AssertionError('unapplied command occurs before final phase record')
        if label=='frozen':exact(checks,'frozen_q',d['q_post'],np.broadcast_to(state['q'],q.shape))
        ws,wss=spectral(d['true_w']);wt=rhs(d['true_w'],z[:,:,3],cfg)
        clean=np.stack((d['true_w'],ws,wss,z[:,:,3],wt),axis=-1)
        discrepancy(checks,'clean_context',d['clean_context'],clean);discrepancy(checks,'actual_wt',d['actual_wt'],wt)
        obsws,obswss=spectral(d['measured_w'])
        if cfg['sensing_mode']=='causal':
            fit,state['fit_history']=fitted(d['measured_w'],state['fit_history'],coef)
            cleanfit,state['clean_fit_history']=fitted(d['true_w'],state['clean_fit_history'],coef)
            expectedvalid=idx>=round(cfg['rate_fit_window']/cfg['dt'])
        else:fit=wt;cleanfit=wt;expectedvalid=np.ones(n,bool)
        exact(checks,'rate_valid',valid,expectedvalid);discrepancy(checks,'observed_rate',d['observed_rate'],fit)
        expectedz=np.stack((d['measured_w'],obsws,obswss,z[:,:,3],fit),axis=-1)
        discrepancy(checks,'observed_context',z,expectedz)
        v,state['history']=context_secants(z,valid,state['history'],cfg)
        vc,state['clean_history']=context_secants(clean,valid,state['clean_history'],cfg)
        sv=np.all(np.isfinite(v),axis=(1,2));exact(checks,'secant_valid',d['secant_valid'],sv)
        discrepancy(checks,'secant',d['secant'],v)
        ma=sv&(t<cfg['duration']-1e-10);exact(checks,'memory_applied',d['memory_applied'],ma)
        exact(checks,'learning_applied',d['learning_applied'],ma&(label!='frozen'))
        F=Fphysical(clean,cfg['plant_coefficients']);e=distance(q,F);initiale=distance(state['initial'],F)
        discrepancy(checks,'physical_F',d['physical_F'],F);discrepancy(checks,'field_error',d['F_error'],e)
        discrepancy(checks,'initial_field_error',d['initial_F_error'],initiale)
        incompat=np.sum(q*v,axis=-1);discrepancy(checks,'incompatibility',d['incompatibility_pre'],incompat)
        ref,refwt,refwtt=reference(t,cfg['grid_points']); target=refwtt-cfg['kp']*(d['measured_w']-ref)-cfg['kd']*(fit-refwt)
        wts,wtss=spectral(fit); supplied=np.stack((fit,wts,wtss,target),axis=-1)
        discrepancy(checks,'target_w',d['target_w'],ref);discrepancy(checks,'target_wt',d['target_wt'],refwt)
        discrepancy(checks,'target_wtt',d['target_wtt'],target);discrepancy(checks,'supplied_rates',d['supplied_rates'],supplied)
        if label=='training':
            s=2*np.pi*np.arange(cfg['grid_points'])/cfg['grid_points'];raw=cfg['probe_amplitude']/2*sum(om*np.sin((k+1)*s+.37*k)[None,:]*np.cos(om*t[:,None]) for k,om in enumerate((3.,5.,7.,11.)))
            guard=np.zeros(raw.shape,bool)
        else:
            guard=np.abs(q[:,:,3])<cfg['input_guard'];raw=-z[:,:,3].copy();accepted=~guard
            raw[accepted]=-np.sum(q[:,:,[0,1,2,4]]*supplied,axis=-1)[accepted]/q[:,:,3][accepted]
        command=np.clip(raw,-cfg['input_rate_limit'],cfg['input_rate_limit'])
        exact(checks,'guard',d['guard'],guard);exact(checks,'clip',d['clip'],np.abs(raw)>cfg['input_rate_limit'])
        discrepancy(checks,'preupdate_raw_command',d['udot_raw'],raw);discrepancy(checks,'applied_command',d['udot'],command)
        predictedw,predictedu=advance(d['true_w'],z[:,:,3],command,cfg)
        if n>1:
            discrepancy(checks,'plant_transition',d['true_w'][1:],predictedw[:-1]);discrepancy(checks,'input_transition',z[1:,:,3],predictedu[:-1])
        sparse={int(k):j for j,k in enumerate(d['memory_index'])}
        for j,k in enumerate(idx):
            if int(k) in sparse:
                p=sparse[int(k)];discrepancy(checks,'R_pre',d['memory_R_pre'][p],state['R'])
                eligible=bool(app[j])
                win=[name for name,(a,b) in WINDOWS.items() if eligible and a<=t[j]<b]
                if win:
                    eo,vo=np.linalg.eigh(state['R']);ec,vec=np.linalg.eigh(state['Rc'])
                    vals=dict(q_to_observed_bottom=distance(q[j],vo[:,:,0]),observed_bottom_to_physical=distance(vo[:,:,0],F[j]),
                        clean_bottom_to_physical=distance(vec[:,:,0],F[j]),observed_gap=eo[:,1]-eo[:,0],clean_gap=ec[:,1]-ec[:,0],
                        observed_min_to_max_eigenvalue_ratio=eo[:,0]/np.maximum(eo[:,-1],1e-30),
                        clean_min_to_max_eigenvalue_ratio=ec[:,0]/np.maximum(ec[:,-1],1e-30))
                    for name in win:
                        for key,value in vals.items():collect(covacc[name],key,value)
            if ma[j]:
                Rbefore=state['R']; state['R']=beta*state['R']+alpha*v[j,:, :,None]*v[j,:,None,:]
                state['Rc']=beta*state['Rc']+alpha*vc[j,:,:,None]*vc[j,:,None,:]
                if label!='frozen':
                    bound=max(float(np.max(np.trace(Rbefore,axis1=-2,axis2=-1))),float(np.max(np.trace(state['R'],axis1=-2,axis2=-1))))
                    expected_sub=max(4,int(np.ceil(cfg['gain']*cfg['dt']*bound/.25)))
                    exact(checks,'subdivisions',d['subdivisions'][j],expected_sub)
                    if ode_enabled and t[j] in (79.,80.,99.):
                        result=math.joint_ode_check(q[j],Rbefore,v[j],d['q_post'][j],state['R'],cfg)
                        result.update(time=float(t[j]),index=int(k),subdivisions=int(d['subdivisions'][j]));ode.append(result)
                    if ode_enabled and (worst_ode is None or d['subdivisions'][j]>worst_ode[0]):
                        worst_ode=(int(d['subdivisions'][j]),float(t[j]),q[j].copy(),Rbefore.copy(),v[j].copy(),d['q_post'][j].copy(),state['R'].copy())
                else:exact(checks,'frozen_subdivisions',d['subdivisions'][j],0)
            else:exact(checks,'no_update_q',d['q_post'][j],q[j])
            if int(k) in sparse:discrepancy(checks,'R_post',d['memory_R_post'][sparse[int(k)]],state['R'])
        base=dict(time=t,field=e,initial_field=initiale,task_error=d['true_w']-ref,observed_task_error=d['measured_w']-ref,
            guard=guard,clip=d['clip'],input_partial_abs=np.abs(q[:,:,3]),input=z[:,:,3],input_rate=command,
            secant=v,incompat=incompat,physical_incompat=np.sum(math.normalize(F)*v,axis=-1),
            sensor_total=fit-wt,sensor_bias=cleanfit-wt,sensor_noise=fit-cleanfit)
        if label!='training':
            physicalobs=-np.sum(F[:,:,[0,1,2,4]]*supplied,axis=-1)/F[:,:,3]
            clean_target=refwtt-cfg['kp']*(d['true_w']-ref)-cfg['kd']*(wt-refwt)
            cws,cwss=spectral(wt); clean_supplied=np.stack((wt,cws,cwss,clean_target),axis=-1)
            physicalclean=-np.sum(F[:,:,[0,1,2,4]]*clean_supplied,axis=-1)/F[:,:,3]
            base.update(readout_field=raw-physicalobs,readout_sensing=physicalobs-physicalclean,
                readout_total=raw-physicalclean,readout_applied=command-physicalclean)
        for name,(a,b) in WINDOWS.items():
            chosen=app&(t>=a-1e-10)&(t<b-1e-10)
            if np.any(chosen):
                for key,value in base.items():collect(accum[name],key,value[chosen])
        compactmask=app&(idx%round(.05/cfg['dt'])==0)
        if cfg['initial_seed']==732 and np.any(compactmask):
            compact.append(dict(time=t[compactmask],w=d['true_w'][compactmask],u=z[compactmask,:,3],q=q[compactmask]))
        count+=n;applied_count+=int(np.sum(app));max_subdiv=max(max_subdiv,int(np.max(d['subdivisions'])))
        state['q']=d['q_post'][-1].copy();state['w']=(predictedw[-1] if app[-1] else d['true_w'][-1]).copy()
        state['u']=(predictedu[-1] if app[-1] else z[-1,:,3]).copy();state['index']=int(idx[-1]+int(app[-1]))
        periodic=directory/f'checkpoint_{state["index"]:07d}'
        if app[-1] and periodic.with_suffix('.npz').exists():
            checkpoint_checks(periodic,state,checks,'periodic',cfg['sensing_mode']=='causal')
    if worst_ode is not None and worst_ode[0]>4 and worst_ode[1] not in [x['time'] for x in ode]:
        sub,t,q,R,v,qn,Rn=worst_ode;result=math.joint_ode_check(q,R,v,qn,Rn,cfg);result.update(time=t,subdivisions=sub,selection='maximum observed subdivisions, diagnostic only');ode.append(result)
    # Complete records include all state changes on a field stop. Other failure
    # stages may have an extra unrecorded observation, so compare the pre-attempt.
    if phase_summary.get('status')=='completed':checkpoint_checks(directory/'final_state',state,checks,'final',cfg['sensing_mode']=='causal')
    elif (directory/'failure_pre_attempt.npz').exists():
        with np.load(directory/'failure_pre_attempt.npz') as p: failindex=int(p['index'])
        if failindex==state['index'] and (not count or app[-1]):
            checkpoint_checks(directory/'failure_pre_attempt',state,checks,'failure_pre',cfg['sensing_mode']=='causal')
        elif count and not app[-1]:checkpoint_checks(directory/'final_state',state,checks,'failed_final',cfg['sensing_mode']=='causal')
    output=dict(outcome=phase_summary,records=count,applied_records=applied_count,checks=checks,
        max_subdivisions=max_subdiv,independent_joint_ode_checks=ode)
    for name,interval in WINDOWS.items():output[name]=metrics(accum[name],covacc[name],cfg,interval)
    if compact:
        compactdir=HERE/'audit'/'common_grid';compactdir.mkdir(parents=True,exist_ok=True)
        np.savez_compressed(compactdir/f'{cfg["case_id"]}__{label}.npz',**{k:np.concatenate([p[k] for p in compact]) for k in compact[0]})
    return output,state,first


def paired_common_horizon(casepath,phases,cfg):
    a=phases.get('adaptive',{}).get('control');b=phases.get('frozen',{}).get('control')
    if not a or not b:return None
    low=max(a['first_observed_time'],b['first_observed_time'])
    high=min(a['last_observed_time'],b['last_observed_time'])+cfg['dt']
    result=dict(interval=[low,high],right_endpoint_excluded=True,arms={})
    for arm in ('adaptive','frozen'):
        fields=[];tasks=[];guards=[];clips=[];times=[]
        files=sorted((casepath/arm/'chunks').glob('chunk_*.npz')) or sorted((casepath/arm).glob('chunk_*.npz'))
        for path in files:
            with np.load(path) as d:
                selected=d['command_applied']&(d['time']>=low-1e-10)&(d['time']<high-1e-10)
                if not np.any(selected):continue
                t=d['time'][selected];clean=d['clean_context'][selected]
                fields.append(distance(d['q_pre'][selected],Fphysical(clean,cfg['plant_coefficients'])))
                tasks.append(d['true_w'][selected]-reference(t,cfg['grid_points'])[0])
                guards.append(d['guard'][selected]);clips.append(d['clip'][selected]);times.append(t)
        e=np.concatenate(fields);t=np.concatenate(times)
        result['arms'][arm]=dict(samples=len(t),first_observed_time=float(t[0]),last_observed_time=float(t[-1]),
            field=summary(e),field_worst_site_mean=float(np.max(np.mean(e,axis=0))),
            task_global_rms=rms(np.concatenate(tasks)),guard_fraction=float(np.mean(np.concatenate(guards))),
            clipping_fraction=float(np.mean(np.concatenate(clips))))
    result['adaptive_to_frozen_task_rms_ratio']=result['arms']['adaptive']['task_global_rms']/result['arms']['frozen']['task_global_rms']
    result['adaptive_to_frozen_field_mean_ratio']=result['arms']['adaptive']['field']['mean']/result['arms']['frozen']['field']['mean']
    return result


def complete_window(phase,name):
    x=phase.get(name);return x if x and x['full_interval_observed'] else None


def assessment(phases):
    tr=complete_window(phases.get('training',{}),'training_evaluation');late=complete_window(phases.get('adaptive',{}),'late')
    result={}
    if tr:
        criteria=dict(field_mean=tr['field']['mean']<=.05,field_p95=tr['field']['p95']<=.1,field_max=tr['field']['maximum']<=.25,
            worst_site_mean=tr['field_worst_site_mean']<=.1,initial_shadow_ratio=tr['learned_to_initial_error_ratio']<=.25)
        result.update(training_criteria=criteria,near_identified_training=all(criteria.values()),inherited_training_mean=tr['field']['mean']<=.1,
            inherited_training_ratio=tr['learned_to_initial_error_ratio']<=.25)
    else:result.update(training_criteria=None,near_identified_training=None,inherited_training_mean=None,inherited_training_ratio=None)
    completed=phases.get('adaptive',{}).get('outcome',{}).get('status')=='completed'
    if late:
        criteria=dict(field_mean=late['field']['mean']<=.1,field_p95=late['field']['p95']<=.25,task_global_rms=late['task_global_rms']<=.001,
            guard_fraction=late['guard_fraction']<.01,clipping_fraction=late['clipping_fraction']<.01,completion=completed)
        result.update(adaptive_late_criteria=criteria,adaptive_late_pass=all(criteria.values()))
    else:result.update(adaptive_late_criteria=None,adaptive_late_pass=None)
    result['adaptive_completion']=completed
    return result


def numerical_verdict(result):
    tolerances=dict(default_absolute=1e-9,unit_norm_absolute=1e-12,
        projector_distance_absolute=2e-7,joint_ode_q_absolute=1e-7,joint_ode_R_absolute=1e-9)
    violations=[];worst=[]
    groups=[('initial',result.get('initial_checks',{})),('branch',result.get('branch_checks',{}))]
    groups += [(name,phase['checks']) for name,phase in result.get('phases',{}).items()]
    for name,checks in groups:
        for key,value in checks.items():
            if isinstance(value,bool):continue
            tol=2e-7 if 'field_error' in key else (1e-12 if key in ('q_unit','q_post_unit') else 1e-9)
            worst.append(dict(phase=name,check=key,absolute_error=value,tolerance=tol))
            if not np.isfinite(value) or value>tol:violations.append(worst[-1])
    for name,phase in result.get('phases',{}).items():
        for ode in phase['independent_joint_ode_checks']:
            if not ode['success'] or ode['q_max_abs_error']>1e-7 or ode['R_max_abs_error']>1e-9:
                violations.append(dict(phase=name,check='joint_ode',details=ode))
    return dict(passed=not violations,tolerances=tolerances,violations=violations,
        largest_absolute_residuals=sorted(worst,key=lambda x:x['absolute_error'],reverse=True)[:12],
        explanation='These are audit reconstruction tolerances, not scientific success thresholds. Projector-distance comparison permits near-zero square-root cancellation; ODE tolerance independently bounds numerical update error.')


def audit_case(casepath,protocol):
    cfg=json.loads((casepath/'config.json').read_text()); result={k:cfg[k] for k in ('case_id','condition','initial_seed','sensor_seed','dt','role')}
    hashes=json.loads((casepath/'artifact_hashes.json').read_text());bad=[]
    for name,expected in hashes.items():
        if sha(casepath/name)!=expected:bad.append(name)
    if bad:raise AssertionError('artifact hash mismatch: '+str(bad))
    result['integrity']=dict(files_checked=len(hashes),all_hashes_match=True,protocol_match=sha(casepath/'protocol.json')==sha(HERE/'protocol.json'))
    original=json.loads((casepath/'summary.json').read_text());result['outcomes']=original.get('outcomes',{});result['status']=original.get('status','attempted')
    if result['status']=='not_attempted':result.update(phases={},assessment=assessment({}));return result
    state=initial_state(cfg);checks={};checkpoint_checks(casepath/'initial',state,checks,'initial',cfg['sensing_mode']=='causal')
    result['initial_checks']=checks; phases={}
    ode_enabled=cfg['initial_seed']==732
    phases['training'],state,_=audit_phase(casepath,'training',cfg,state,ode_enabled)
    if (casepath/'branch.npz').exists():
        checks={};checkpoint_checks(casepath/'branch',state,checks,'branch',cfg['sensing_mode']=='causal');result['branch_checks']=checks
        firsts={}
        for name in ('adaptive','frozen'):
            phases[name],_,firsts[name]=audit_phase(casepath,name,cfg,copy.deepcopy(state),ode_enabled and name=='adaptive')
        if firsts['adaptive'] is not None and firsts['frozen'] is not None:
            for key in firsts['adaptive']:exact(checks,'first_shared_'+key,firsts['adaptive'][key],firsts['frozen'][key])
    result['phases']=phases;result['assessment']=assessment(phases)
    result['paired_common_horizon']=paired_common_horizon(casepath,phases,cfg)
    result['numerical_verdict']=numerical_verdict(result)
    return result


def refinement(cases):
    out={}
    for condition in ('ideal','causal_clean'):
        out[condition]={}
        ids=[f'{condition}_s732_dt{x}' for x in ('0p005','0p0025','0p00125')]
        for a,b in zip(ids[:-1],ids[1:]):
            if a not in cases or b not in cases:continue
            result=dict(coarse_case=a,fine_case=b,common_grid={},window_metrics={})
            for phase in ('training','adaptive','frozen'):
                pa,pb=cases[a].get('phases',{}).get(phase,{}),cases[b].get('phases',{}).get(phase,{})
                for win in WINDOWS:
                    wa,wb=complete_window(pa,win),complete_window(pb,win)
                    if not wa or not wb:continue
                    vals={}
                    for key,va,vb,threshold in [('field_mean',wa['field']['mean'],wb['field']['mean'],1e-5),('task_rms',wa['task_global_rms'],wb['task_global_rms'],1e-4)]:
                        change=abs(va-vb);rel=change/max(abs(vb),1e-30)
                        vals[key]=dict(coarse=va,fine=vb,absolute_change=change,relative_change=rel,absolute_flag=change>threshold,
                            relative_flag=rel>.1 if max(abs(va),abs(vb))>1e-6 else None)
                    result['window_metrics'][phase+'_'+win]=vals
                ga,gb=(HERE/'audit'/'common_grid'/f'{x}__{phase}.npz' for x in (a,b))
                if ga.exists() and gb.exists():
                    with np.load(ga) as da,np.load(gb) as db:
                        ka=np.rint(da['time']/.05).astype(int);kb=np.rint(db['time']/.05).astype(int)
                        common,ia,ib=np.intersect1d(ka,kb,return_indices=True)
                        if len(common):result['common_grid'][phase]=dict(samples=len(common),interval=[float(common[0]*.05),float(common[-1]*.05)],
                            sample_interval=.05,w_global_rms_difference=rms(da['w'][ia]-db['w'][ib]),u_global_rms_difference=rms(da['u'][ia]-db['u'][ib]),
                            q_projector_distance=summary(distance(da['q'][ia],db['q'][ib])))
            out[condition][a.split('_dt')[-1]+'_to_'+b.split('_dt')[-1]]=result
    return out


def aggregates(cases,protocol):
    out={}
    for condition in ('ideal','causal_clean','causal_noisy'):
        planned=[c for c in protocol['cases'] if c['condition']==condition and c['role']=='main_five_seed_study']
        available=[cases[c['case_id']] for c in planned if c['case_id'] in cases]
        out[condition]=dict(planned_count=5,audited_count=len(available),started_count=sum(x.get('status')!='not_attempted' for x in available),
            training_complete_count=sum(x.get('outcomes',{}).get('training',{}).get('status')=='completed' for x in available),
            adaptive_complete_count=sum(x.get('outcomes',{}).get('adaptive',{}).get('status')=='completed' for x in available),
            frozen_complete_count=sum(x.get('outcomes',{}).get('frozen',{}).get('status')=='completed' for x in available),
            near_identified_training_count=sum(x.get('assessment',{}).get('near_identified_training') is True for x in available),
            adaptive_late_pass_count=sum(x.get('assessment',{}).get('adaptive_late_pass') is True for x in available))
        out[condition]['training_per_criterion_pass_counts']={key:sum(x.get('assessment',{}).get('training_criteria',{}).get(key) is True for x in available if x.get('assessment',{}).get('training_criteria')) for key in ('field_mean','field_p95','field_max','worst_site_mean','initial_shadow_ratio')}
        out[condition]['windows']={}
        for phase in ('training','adaptive','frozen'):
            for win in WINDOWS:
                full=[complete_window(x.get('phases',{}).get(phase,{}),win) for x in available]
                full=[x for x in full if x]
                if not full:continue
                vals={}
                for key,f in [('field_mean',lambda x:x['field']['mean']),('field_p95',lambda x:x['field']['p95']),('task_global_rms',lambda x:x['task_global_rms'])]:
                    a=np.array([f(x) for x in full]);vals[key]=dict(count=len(a),mean=float(np.mean(a)),standard_deviation=float(np.std(a,ddof=1)) if len(a)>1 else None,minimum=float(np.min(a)),maximum=float(np.max(a)))
                out[condition]['windows'][phase+'_'+win]=dict(full_window_count=len(full),planned_count=5,metrics=vals)
    return out


def main():
    global HERE
    parser=argparse.ArgumentParser();parser.add_argument('--case');parser.add_argument('--runs',type=Path,required=True);parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    HERE=args.output
    HERE.mkdir(parents=True,exist_ok=False)
    (HERE/'protocol.json').write_bytes((PACKAGE/'protocol.json').read_bytes())
    protocol=json.loads((HERE/'protocol.json').read_text());casefiles=HERE/'audit'/'cases';casefiles.mkdir(parents=True,exist_ok=True)
    sources=HERE/'audit'/'sources';sources.mkdir(parents=True,exist_ok=True)
    (sources/f'audit_source_{AUDIT_SOURCE_HASH}.py').write_bytes(AUDIT_SOURCE_BYTES)
    (sources/f'independent_math_{sha(PRIOR)}.py').write_bytes(PRIOR.read_bytes())
    for c in protocol['cases']:
        if args.case and c['case_id']!=args.case:continue
        dest=casefiles/(c['case_id']+'.json');path=args.runs/c['case_id']
        if dest.exists() or not (path/'artifact_hashes.json').exists():continue
        print('Auditing '+c['case_id'],flush=True)
        result=audit_case(path,protocol);result['audit_source_sha256']=AUDIT_SOURCE_HASH;result['math_source_sha256']=sha(PRIOR)
        dump(dest,result);print('Passed reconstruction '+c['case_id'],flush=True)
    cases={p.stem:json.loads(p.read_text()) for p in sorted(casefiles.glob('*.json'))}
    result=dict(protocol_sha256=sha(HERE/'protocol.json'),audit_source_sha256=sha(__file__),math_source_sha256=sha(PRIOR),
        planned_cases=[c['case_id'] for c in protocol['cases']],cases=cases,aggregates=aggregates(cases,protocol),refinement=refinement(cases),
        methodology='Independent QR sensor weights, raw RNG/contexts/commands/plant transitions, literal exact held-secant R and selected joint DOP853 q/R. All entries remain free; true F is evaluator only. Sparse covariance diagnostics use pre-update R every 0.05 s; other window metrics use all applied records. Unobserved or incomplete full-window criteria are null, not failed or successful. Common-grid refinement uses applied observations every 0.05 s on actual shared coverage.')
    dump(HERE/'independent_audit.json',result)
if __name__=='__main__':main()
