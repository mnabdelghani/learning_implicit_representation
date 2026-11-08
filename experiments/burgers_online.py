"""Run the supplied Burgers reconstruction and add evaluated-partial diagnostics.
The plant, learner and control law are unchanged from examples/legacy_suite.
"""
from __future__ import annotations
import argparse,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'examples'/'legacy_suite'/'simulations'))
import burgers
from common import dump_json

def additional_partials(ep,truth,length):
    times=[];exact=[];estimated=[];rho=[]
    names=ep['trace_columns'];tr=ep['trace'];grid=ep['grid']
    dt=tr[1,0]-tr[0,0]
    for t,w,wd,u,wt,pred in ep['snapshots']:
        index=round(t/dt);c=tr[index,[names.index(n+'_pre') for n in ['ca','nu','b']]]
        ws,wss=burgers.spectral_derivatives(w,length)
        z=np.column_stack((w,ws,wss,u,wt))
        F=burgers.implicit_jacobian(z,truth);G=burgers.implicit_jacobian(z,c)
        times.append(t);exact.append(F);estimated.append(G)
        dot=np.sum(F*G,axis=1)/np.sqrt(np.sum(F*F,axis=1)*np.sum(G*G,axis=1))
        rho.append(float(np.sqrt(np.mean(np.maximum(0,2*(1-dot**2))))))
    return dict(time=np.array(times),s=grid,F_true=np.array(exact),F_est=np.array(estimated),projector_error=np.array(rho))

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--seeds',nargs='+',type=int,default=[10]);p.add_argument('--methods',nargs='+',default=['learned','frozen','oracle']);a=p.parse_args()
    cfg=burgers.load_config(ROOT/'examples'/'legacy_suite'/'configs'/'burgers.json');dump_json(a.output/'config.json',cfg)
    summaries=[]
    for seed in a.seeds:
        for method in a.methods:
            ep=burgers.simulate_episode(cfg,method,seed);dest=a.output/f'seed_{seed}'/method
            short=burgers.save_episode(ep,dest);np.savez_compressed(dest/'partials.npz',**additional_partials(ep,cfg['true_coefficients'],cfg['domain_length']));summaries.append(short)
            print(seed,method,short['metrics']['tracking_rms'],short['metrics']['F_relative_error_mean'],flush=True)
    dump_json(a.output/'summary.json',dict(status='Executed extension of supplied reconstruction',runs=summaries))
if __name__=='__main__':main()
