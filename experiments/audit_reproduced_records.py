"""Read-only numerical audit of a reproduce_records.py output.

Recompute metrics independently from recorded traces, compare every historical
summary, compare the originally available raw records, and hash every new file.
Only audit files inside the reproduction output are written.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

METHODS = {'contact': ('learned', 'frozen', 'oracle'),
           'fourier': ('learned', 'K0', 'frozen', 'oracle'),
           'burgers': ('learned', 'frozen', 'oracle'),
           'contact_excitation': ('task_only', 'bounded_probe')}


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--historical-root', type=Path,
                        default=Path(__file__).resolve().parents[1] / 'results')
    args = parser.parse_args()
    output, original = args.output.resolve(), args.historical_root.resolve()
    provenance = json.loads((output / 'provenance.json').read_text())
    if provenance['status'] != 'complete' or provenance['completed_runs'] != 120:
        raise RuntimeError('All 120 planned runs must complete before this audit')
    checks, failures, comparisons = [], [], []

    def close(actual, expected, label, group='raw_metric'):
        a, e = np.asarray(actual, dtype=float), np.asarray(expected, dtype=float)
        ok = a.shape == e.shape and np.allclose(a, e, rtol=1e-9, atol=1e-11, equal_nan=True)
        finite = np.isfinite(a) & np.isfinite(e) if a.shape == e.shape else np.zeros(0, bool)
        difference = float(np.max(np.abs(a[finite] - e[finite]))) if np.any(finite) else 0.
        checks.append(dict(label=label, group=group, passed=bool(ok), max_absolute_difference=difference))
        if not ok:
            failures.append(checks[-1])

    def condition(ok, label):
        checks.append(dict(label=label, group='invariant', passed=bool(ok)))
        if not ok:
            failures.append(checks[-1])

    def compare_tree(a, b, label):
        if isinstance(a, dict):
            condition(a.keys() == b.keys(), label + '/keys')
            for key in a.keys() & b.keys():
                compare_tree(a[key], b[key], label + '/' + key)
        elif isinstance(a, (float, int, list)):
            close(a, b, label, 'historical_summary')
        else:
            condition(a == b, label)

    def mean(t, name, sel):
        return float(np.nanmean(t[name][sel]))

    def rms(t, name, sel):
        return float(np.sqrt(np.mean(t[name][sel] ** 2)))

    def contact_partials(t, cfg, true=False):
        x = t['x']
        b1 = np.column_stack((2*np.cos(2*x), -5*np.sin(5*x)))
        b2 = np.column_stack((-4*np.sin(2*x), -25*np.cos(5*x)))
        names = ('s1','s2','a','r1','r2','b')
        c = np.column_stack([t[n + '_true' if true else n] for n in names])
        late = t['time'] >= cfg['change_time']
        Btruth = np.where(late[:,None,None], np.array(cfg['B_after'])[None], np.array(cfg['B_before'])[None])
        B = Btruth if true else np.array([[t['B11'],t['B12']],[t['B21'],t['B22']]]).transpose(2,0,1)
        u = np.column_stack((t['u1'], t['u2']))
        vx = np.einsum('nij,nj->ni', Btruth, u)[:,0]
        F = np.zeros((len(t),3,8))
        F[:,0,0] = -np.sum(c[:,:2]*b1,axis=1)-np.sum(c[:,3:5]*b2,axis=1)*vx
        F[:,0,1] = c[:,2]
        F[:,0,2] = cfg['rho']
        F[:,0,3] = -np.sum(c[:,3:5]*b1,axis=1)
        F[:,0,4] = c[:,5]
        F[:,0,5] = 1.
        F[:,1,3] = F[:,2,4] = 1.
        F[:,1:,-2:] = -B
        return F

    def audit_contact(folder, t, cfg, seed, metrics, label, transfer):
        windows = [('training',40.,60.),('transfer',70.,80.)] if transfer else [
            ('early',0.,20.),('before_change',40.,60.),('change',60.,80.),('recovered',100.,120.)]
        for name,lo,hi in windows:
            sel = (t['time']>=lo)&(t['time']<hi)
            m = metrics[name]
            pairs = [('force_error','force_rmse'),('x_error','motion_rmse'),('control_norm','control_rms')]
            for col,key in pairs:
                close(rms(t,col,sel),m[key],label+'/'+name+'/'+key)
            pairs = [('F_error','F_error_mean'),('preupdate_energy','energy_mean' if transfer else 'incompatibility_mean'),
                     ('same_stream_oracle_energy','oracle_energy_mean'),('normalized_energy','normalized_energy_mean')]
            if transfer:
                close(np.min(t['readout_sigma_min'][sel]),m['readout_sigma_min'],label+'/'+name+'/readout_sigma')
            else:
                pairs += [('projector_error','projector_error_mean'),('clean_energy','clean_energy_mean'),('feature_power','feature_power_mean')]
            for col,key in pairs:
                close(mean(t,col,sel),m[key],label+'/'+name+'/'+key)
        close(np.min(t['indentation']),metrics['minimum_indentation'],label+'/minimum_indentation')
        close(np.mean(t['saturated'][:-1]),metrics['saturation_fraction'],label+'/saturation_fraction')
        condition(np.min(t['indentation'])>0,label+'/contact_maintained')
        truth, estimate = contact_partials(t,cfg,True), contact_partials(t,cfg)
        scale = np.array(cfg['row_scales'])[:,None]*np.array(cfg['context_scales'])
        error = np.linalg.norm((estimate-truth)*scale,axis=(1,2))/np.linalg.norm(truth*scale,axis=(1,2))
        close(error,t['F_error'],label+'/reconstructed_partials_error')
        with np.load(folder/'partials.npz') as p:
            close(truth[::10],p['F_true'],label+'/saved_true_partials')
            close(estimate[::10],p['F_est'],label+'/saved_estimated_partials')
        # The original sensor noise is regenerated only for verification, never
        # fed into a rerun or used to change the recorded values.
        n = len(t)
        noise = np.random.default_rng(seed).normal(size=(n,4))
        x = t['x']; g = np.asarray(cfg['geometry'])
        hp = 2*g[0]*np.cos(2*x)-5*g[1]*np.sin(5*x)
        lam = t['normal_force']/np.sqrt(1+hp*hp)
        y = g[0]*np.sin(2*x)+g[1]*np.cos(5*x)-t['indentation']
        clean = np.column_stack((np.sin(2*x),np.cos(5*x),y,lam))
        measured = clean.copy(); measured[:,-1] += cfg['force_noise']*noise[:,1]
        lag = round(cfg['window']/cfg['dt'])
        weights = np.r_[.5,np.ones(lag-1),.5]*cfg['dt']
        c = np.column_stack([t[k] for k in ('s1','s2','a','r1','r2','b')])
        ctruth = np.column_stack([t[k+'_true'] for k in ('s1','s2','a','r1','r2','b')])
        for data,column,coeff in ((measured,'preupdate_energy',c),(measured,'same_stream_oracle_energy',ctruth),(clean,'clean_energy',c)):
            integral = np.column_stack([np.convolve(data[:,j],weights,'valid') for j in range(4)])
            diff = data[lag:]-data[:-lag]
            free = np.column_stack((-integral[:,:2],integral[:,2],-diff[:,:2],diff[:,2]))/cfg['window']
            supplied = (diff[:,3]+cfg['rho']*integral[:,3])/cfg['window']
            E = .5*(np.sum(free*coeff[lag:],axis=1)+supplied)**2
            close(E,t[column][lag:],label+'/reconstructed_'+column)
        if not transfer:
            close(c[-1],metrics['final_material'],label+'/final_material')
            return
        mask=t['time']>=60.
        allc=np.column_stack([t[k] for k in ('s1','s2','a','r1','r2','b','B11','B12','B21','B22')])
        condition(np.array_equal(allc[mask],np.broadcast_to(allc[mask][0],allc[mask].shape)),label+'/frozen_coefficients')
        condition(np.all(t['learning_enabled'][mask]==0),label+'/learning_disabled_after_transfer')
        before=np.flatnonzero(t['time']<60)[-1]
        close([t[f'clean_information_eigenvalue_{i}'][before] for i in range(1,7)],
              metrics['clean_information_eigenvalues'],label+'/information_spectrum')
        # Independent vectorized evaluation at the declared common query points.
        rng=np.random.default_rng(2026);qx=rng.uniform(-.5,.5,1000);qv=rng.uniform(-.5,.5,1000)
        beta1=np.column_stack((2*np.cos(2*qx),-5*np.sin(5*qx)))
        beta2=np.column_stack((-4*np.sin(2*qx),-25*np.cos(5*qx)))
        idx=np.flatnonzero(mask)[0]
        arrays=[]
        for coefficient,B in ((ctruth[idx],np.asarray(cfg['B_before'])),
                (c[idx],np.array([[t['B11'][idx],t['B12'][idx]],[t['B21'][idx],t['B22'][idx]]]))):
            f=np.zeros((1000,3,8));f[:,0,0]=-beta1@coefficient[:2]-(beta2@coefficient[3:5])*qv
            f[:,0,1]=coefficient[2];f[:,0,2]=cfg['rho'];f[:,0,3]=-beta1@coefficient[3:5]
            f[:,0,4]=coefficient[5];f[:,0,5]=1;f[:,1,3]=f[:,2,4]=1;f[:,1:,-2:]=-B
            arrays.append(f*scale)
        close(np.linalg.norm(arrays[1]-arrays[0])/np.linalg.norm(arrays[0]),metrics['heldout_F_error'],label+'/heldout_F_error')

    def audit_fourier(t,cfg,seed,metrics,label):
        sel=(t['time']>=60)&(t['time']<80)
        for col,key in [('error','tracking_rmse'),('control','control_rms')]:
            close(rms(t,col,sel),metrics[key],label+'/'+key)
        for col,key in [('F_error','F_error_mean'),('projector_error','projector_error_mean'),
            ('preupdate_energy','energy_mean'),('same_stream_oracle_energy','oracle_energy_mean'),
            ('clean_energy','clean_energy_mean'),('K0_same_stream_energy','K0_same_stream_energy_mean'),
            ('normalized_energy','normalized_energy_mean'),('feature_power','feature_power_mean')]:
            close(mean(t,col,sel),metrics[key],label+'/'+key)
        phase=cfg['phase_rate']*t['time'];h=np.column_stack((np.ones(len(t)),np.cos(phase),np.sin(phase)))
        dh=np.column_stack((np.zeros(len(t)),-np.sin(phase),np.cos(phase)))
        c=np.column_stack([t[f'c{i}'] for i in range(6)]);ct=np.asarray(cfg['true_coefficients'])
        rho=h@ct[:3];kappa=h@ct[3:];rate=(t['control']-kappa*t['state'])/rho
        ptrue=(dh@ct[:3])*rate+(dh@ct[3:])*t['state']
        pest=np.sum(dh*c[:,:3],axis=1)*rate+np.sum(dh*c[:,3:],axis=1)*t['state']
        close(ptrue,t['phase_partial_true'],label+'/phase_partial_true')
        close(pest,t['phase_partial_est'],label+'/phase_partial_est')
        F=np.column_stack((kappa,rho,-np.ones(len(t)),ptrue))
        G=np.column_stack((np.sum(h*c[:,3:],axis=1),np.sum(h*c[:,:3],axis=1),-np.ones(len(t)),pest))
        scale=np.asarray(cfg['context_scales'])
        close(np.linalg.norm((G-F)*scale,axis=1)/np.linalg.norm(F*scale,axis=1),t['F_error'],label+'/full_differential_error')
        close(c[-1],metrics['final_coefficients'],label+'/final_coefficients')
        noise=np.random.default_rng(seed).normal(0,cfg['rate_noise'],len(t))
        lag=round(cfg['difference_window']/cfg['dt'])
        for observed,col,coeff in ((rate+noise,'preupdate_energy',c),(rate+noise,'same_stream_oracle_energy',np.broadcast_to(ct,c.shape)),(rate,'clean_energy',c)):
            feat=np.column_stack((h*observed[:,None],h*t['state'][:,None],-t['control']))
            v=(feat[lag:]-feat[:-lag])/cfg['difference_window']
            q=np.column_stack((coeff[lag:],np.ones(len(v))))
            close(.5*np.sum(v*q,axis=1)**2,t[col][lag:],label+'/reconstructed_'+col)

    def audit_burgers(folder,t,cfg,metrics,label):
        sel=(t['time']>=18)&(t['time']<24)
        for key in ('tracking_rms','rate_rmse_clean_preupdate','rate_rmse_measured_preupdate','control_rms'):
            close(rms(t,key,sel),metrics[key],label+'/'+key)
        close(np.max(t['control_peak'][sel]),metrics['control_peak'],label+'/control_peak')
        close(cfg['dt']*np.sum(t['control_rms'][:-1]**2),metrics['control_energy_full_run'],label+'/control_energy_full_run')
        close(np.mean(t['F_relative_error_preupdate'][sel]),metrics['F_relative_error_mean'],label+'/F_relative_error_mean')
        close(np.min(t['F_min_row_norm_preupdate']),metrics['F_min_row_norm'],label+'/F_min_row_norm')
        for k,truth in zip(('ca','nu','b'),cfg['true_coefficients']):
            close(np.sqrt(np.mean(((t[k+'_pre'][sel]-truth)/truth)**2)),metrics[k+'_relative_rms_error'],label+'/'+k+'_relative_rms_error')
        for col,key in [('sample_energy_noisy_preupdate','sample_energy_noisy_mean'),('sample_energy_clean_preupdate','sample_energy_clean_mean')]:
            close(mean(t,col,sel),metrics[key],label+'/'+key)
        for key in ('largest_inner_energy_increase','largest_descent_bound_violation'):
            close(np.max(t[key]),metrics[key],label+'/'+key)
            condition(np.max(t[key])<=1e-10,label+'/'+key+'_bound')
        close(np.max(np.abs(t['field_mean'])),metrics['maximum_absolute_field_mean'],label+'/maximum_absolute_field_mean')
        with np.load(folder/'partials.npz') as p:
            F,G=p['F_true'],p['F_est'];idx=np.rint(p['time']/cfg['dt']).astype(int)
            close(np.linalg.norm(G-F,axis=(1,2))/np.linalg.norm(F,axis=(1,2)),t['F_relative_error_preupdate'][idx],label+'/saved_partial_error')

    for study,methods in METHODS.items():
        cfg=json.loads((output/'records'/study/'config.json').read_text())
        condition(cfg==json.loads((original/study/'config.json').read_text()),study+'/unchanged_config')
        for seed in range(10,20):
            for method in methods:
                label=f'{study}/{seed}/{method}';folder=output/'records'/study/f'seed_{seed}'/method
                historical=original/study/f'seed_{seed}'/method
                s=json.loads((folder/'summary.json').read_text());old=json.loads((historical/'summary.json').read_text())
                compare_tree(s,old,label+'/historical')
                t=np.genfromtxt(folder/'trace.csv',delimiter=',',names=True)
                expected_count=round((cfg['plant'] if study=='contact_excitation' else cfg)['duration']/(cfg['plant'] if study=='contact_excitation' else cfg)['dt'])+1
                condition(len(t)==expected_count,label+'/sample_count')
                condition(np.all(np.diff(t['time'])>0),label+'/time_order')
                if study in ('contact','contact_excitation'):
                    audit_contact(folder,t,cfg['plant'] if study=='contact_excitation' else cfg,seed,s['metrics'],label,study=='contact_excitation')
                elif study=='fourier':
                    audit_fourier(t,cfg,seed,s['metrics'],label)
                else:
                    audit_burgers(folder,t,cfg,s['metrics'],label)
                if method=='frozen':
                    if study=='contact':
                        c=np.column_stack([t[k] for k in ('s1','s2','a','r1','r2','b','B11','B12','B21','B22')])
                    elif study=='fourier':
                        c=np.column_stack([t[f'c{k}'] for k in range(6)])
                    else:
                        c=np.column_stack([t[k+'_pre'] for k in ('ca','nu','b')])
                    condition(np.array_equal(c,np.broadcast_to(c[0],c.shape)),label+'/frozen_baseline_unchanged')
                if (historical/'trace.csv').exists():
                    for name in ('trace.csv','fields.csv'):
                        if (historical/name).exists():
                            a=np.genfromtxt(folder/name,delimiter=',',skip_header=1)
                            b=np.genfromtxt(historical/name,delimiter=',',skip_header=1)
                            close(a,b,label+'/historical_raw/'+name,'historical_raw')
                    if (historical/'partials.npz').exists():
                        with np.load(folder/'partials.npz') as a,np.load(historical/'partials.npz') as b:
                            condition(set(a.files)==set(b.files),label+'/historical_partials_keys')
                            for key in a.files:
                                close(a[key],b[key],label+'/historical_raw/partials/'+key,'historical_raw')
                    comparisons.append(label)
    # Independently aggregate the complete regenerated summaries. No repository
    # summarizer is invoked because it writes to tracked historical outputs.
    aggregate={}
    def stat(values):
        a=np.asarray(values,float)
        condition(len(a)==10 and np.all(np.isfinite(a)),'aggregate/complete_finite_seed_set')
        return dict(mean=float(a.mean()),sd=float(a.std(ddof=1)),n=len(a))
    for study in ('contact','fourier','burgers'):
        aggregate[study]={}
        for method in METHODS[study]:
            entries=[json.loads((output/'records'/study/f'seed_{seed}'/method/'summary.json').read_text())['metrics'] for seed in range(10,20)]
            if study=='contact':
                aggregate[study][method]={k:stat([e['recovered'][k] for e in entries]) for k in entries[0]['recovered']}
                aggregate[study][method].update(minimum_indentation=min(e['minimum_indentation'] for e in entries),
                    max_saturation_fraction=max(e['saturation_fraction'] for e in entries),
                    change_force_rmse=stat([e['change']['force_rmse'] for e in entries]))
            else:
                aggregate[study][method]={k:stat([e[k] for e in entries]) for k in entries[0] if np.isscalar(entries[0][k])}
    compare_tree(aggregate,json.loads((original/'aggregate.json').read_text()),'historical_main_aggregate')
    (output/'records'/'aggregate.json').write_text(json.dumps(aggregate,indent=2)+'\n')
    transfer_aggregate=json.loads((output/'records'/'contact_excitation'/'aggregate.json').read_text())
    compare_tree(transfer_aggregate,json.loads((original/'contact_excitation'/'aggregate.json').read_text()),'historical_transfer_aggregate')
    summary_comparisons=[c for c in checks if c['group']=='historical_summary']
    raw_comparisons=[c for c in checks if c['group']=='historical_raw']
    result=dict(status='PASS' if not failures else 'FAIL',runs_audited=120,
        main_runs=100,transfer_runs=20,historical_raw_runs_compared=len(comparisons),
        checks=len(checks),failures=failures,rtol=1e-9,atol=1e-11,
        maximum_summary_absolute_difference=max(c['max_absolute_difference'] for c in summary_comparisons),
        maximum_available_raw_absolute_difference=max(c['max_absolute_difference'] for c in raw_comparisons),
        historical_raw_run_labels=comparisons,
        limitation='Regenerated original diagnostic outputs. Burgers field snapshots are every 0.1 s; per-step noisy field observations were not originally logged. New records do not recover old archives.',
        check_details=checks)
    (output/'audit.json').write_text(json.dumps(result,indent=2)+'\n')
    manifest={str(p.relative_to(output)):dict(bytes=p.stat().st_size,sha256=digest(p))
        for p in sorted(output.rglob('*')) if p.is_file() and p.name!='file_manifest.json'}
    (output/'file_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='check_details'},indent=2))
    if failures:
        raise SystemExit(1)


if __name__=='__main__':
    main()
