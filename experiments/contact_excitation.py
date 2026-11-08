"""Supplementary paired contact study: regulation alone versus bounded probing.
Training ends at 60 s; coefficients are then frozen for a 20-s new task.
The main-paper protocol is unchanged. Evaluator-only clean information and
held-out partial queries never enter either learner or controller.
"""
from __future__ import annotations
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
import numpy as np
import contact
from common import dump_json, save_trace, rms
ROOT=Path(__file__).resolve().parents[1]
TRANSFER=60.
PROTOCOLS=('task_only','bounded_probe')

def reference(t):
    return (0.,0.,4.,0.) if t<TRANSFER else contact.refs(t-TRANSFER)

def probing(t):
    return np.array([.12*(np.sin(.7*t)+.5*np.sin(1.9*t+.3))/1.5,
                     .02*(np.sin(1.1*t+.5)+.5*np.sin(2.7*t))/1.5])

def heldout_error(row,cfg):
    rng=np.random.default_rng(2026)
    true_c,true_B=contact.truth(0.,cfg)
    fitted=np.array([row[n] for n in ['s1','s2','a','r1','r2','b']])
    fitted_B=np.array([[row['B11'],row['B12']],[row['B21'],row['B22']]])
    z=np.zeros((1000,8));z[:,0]=rng.uniform(-.5,.5,1000);z[:,3]=rng.uniform(-.5,.5,1000)
    scale=np.asarray(cfg['row_scales'])[:,None]*np.asarray(cfg['context_scales'])
    num=den=0.
    for point in z:
        F=contact.differential(point,true_c,true_B,cfg['rho'])
        H=contact.differential(point,fitted,fitted_B,cfg['rho'])
        num+=float(np.sum(((H-F)*scale)**2));den+=float(np.sum((F*scale)**2))
    return float(np.sqrt(num/den))

def run_case(args):
    seed,protocol,output=args
    cfg=contact.CONFIG.copy();cfg.update(duration=80.,change_time=1.e9)
    rows,partials=contact.simulate(seed,'learned',cfg,reference=reference,
        probe=probing if protocol=='bounded_probe' else None,
        freeze_after=TRANSFER,extra_diagnostics=True)
    dest=Path(output)/f'seed_{seed}'/protocol
    save_trace(dest/'trace.csv',rows);np.savez_compressed(dest/'partials.npz',**partials)
    metrics={}
    for name,lo,hi in [('training',40.,60.),('transfer',70.,80.)]:
        data=[x for x in rows if lo<=x['time']<hi]
        metrics[name]={
          'force_rmse':rms([x['force_error'] for x in data]),
          'motion_rmse':rms([x['x_error'] for x in data]),
          'F_error_mean':float(np.mean([x['F_error'] for x in data])),
          'energy_mean':float(np.mean([x['preupdate_energy'] for x in data])),
          'oracle_energy_mean':float(np.mean([x['same_stream_oracle_energy'] for x in data])),
          'normalized_energy_mean':float(np.mean([x['normalized_energy'] for x in data])),
          'control_rms':rms([x['control_norm'] for x in data]),
          'readout_sigma_min':min(x['readout_sigma_min'] for x in data)}
    transfer_row=rows[round(TRANSFER/cfg['dt'])]
    previous_row=rows[round(TRANSFER/cfg['dt'])-1]
    metrics.update(heldout_F_error=heldout_error(transfer_row,cfg),
       clean_information_eigenvalues=[previous_row[f'clean_information_eigenvalue_{k+1}'] for k in range(6)],
       minimum_indentation=min(x['indentation'] for x in rows),
       saturation_fraction=float(np.mean([x['saturated'] for x in rows[:-1]])))
    coeff_names=['s1','s2','a','r1','r2','b','B11','B12','B21','B22']
    frozen=np.array([[x[k] for k in coeff_names] for x in rows if x['time']>=TRANSFER])
    if not np.array_equal(frozen,np.broadcast_to(frozen[0],frozen.shape)):
        raise AssertionError('Coefficients changed after the frozen-transfer boundary')
    info=dict(seed=seed,protocol=protocol,metrics=metrics,freeze_check=True)
    dump_json(dest/'summary.json',info)
    print(json.dumps({'seed':seed,'protocol':protocol,'training_force':metrics['training']['force_rmse'],
        'transfer_force':metrics['transfer']['force_rmse'],'heldout_F':metrics['heldout_F_error']}),flush=True)
    return info

def aggregate(output,seeds):
    result={}
    for protocol in PROTOCOLS:
        entries=[json.loads((output/f'seed_{s}'/protocol/'summary.json').read_text())['metrics'] for s in seeds]
        def stat(values):
            a=np.array(values,float)
            return dict(mean=float(a.mean()),sd=float(a.std(ddof=1)) if len(a)>1 else 0.,n=len(a))
        a={name:{k:stat([x[name][k] for x in entries]) for k in entries[0][name]} for name in ('training','transfer')}
        a['heldout_F_error']=stat([x['heldout_F_error'] for x in entries])
        a['minimum_indentation']=min(x['minimum_indentation'] for x in entries)
        a['maximum_saturation_fraction']=max(x['saturation_fraction'] for x in entries)
        a['clean_information_eigenvalues_mean']=np.mean([x['clean_information_eigenvalues'] for x in entries],axis=0).tolist()
        result[protocol]=a
    dump_json(output/'aggregate.json',result)
    return result

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,default=ROOT/'results/contact_excitation')
    p.add_argument('--seeds',nargs='+',type=int,default=list(range(10,20)))
    p.add_argument('--workers',type=int,default=1)
    args=p.parse_args()
    if args.workers<1:raise ValueError('workers must be positive')
    cfg=contact.CONFIG.copy();cfg.update(duration=80.,change_time=1.e9)
    dump_json(args.output/'config.json',dict(plant=cfg,transfer_time=TRANSFER,seeds=args.seeds,
        training_task='x=0, L=4; online adaptation',transfer_task='original oscillatory task with t reset; coefficients frozen',
        probe='u1=.12*(sin(.7t)+.5sin(1.9t+.3))/1.5; u2=.02*(sin(1.1t+.5)+.5sin(2.7t))/1.5',
        evaluation_windows={'training':[40,60],'transfer':[70,80]},
        heldout_queries='1000 common evaluator-only points; RNG 2026; x,vx independently uniform [-.5,.5]',
        status='New supplementary protocol, separate from the 100 main-protocol runs'))
    jobs=[(s,m,str(args.output)) for s in args.seeds for m in PROTOCOLS]
    if args.workers==1:
        for job in jobs:run_case(job)
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:list(pool.map(run_case,jobs))
    aggregate(args.output,args.seeds)
if __name__=='__main__':main()
