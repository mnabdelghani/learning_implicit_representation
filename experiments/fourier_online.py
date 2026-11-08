"""NEW closed-loop Fourier Memory experiment from the manuscript's periodic plant.
The original offline-identification/independent-readout protocol is not reproduced.
K=0 shadow sees precisely the K=1 observations; separate K=0 control has its own state.
"""
from __future__ import annotations
import argparse
from collections import deque
from pathlib import Path
import numpy as np
from common import ImplicitLearner,rms,save_trace,dump_json

CONFIG=dict(dt=.01,duration=80.,difference_window=.05,memory=8.,steps=8,phase_rate=4.,
    true_coefficients=[1.20,.27,-.18,1.50,.34,.24],initial=[1.,0,0,1.,0,0],
    lower=[1.,-.3,-.3,.8,-.5,-.5],upper=[1.6,.3,.3,2.2,.5,.5],scales=[1.,.3,.3,1.,.4,.4],
    rate_noise=.001,feedback_gain=2.,initial_state=0.2,context_scales=[.5,1.,1.,1.],
    new_choices=['Simultaneous control and identification, not the earlier offline episode',
    'Reference x_d=0.45 sin(0.7t)+0.2 sin(1.3t), gain 2, x(0)=0.2',
    'Held input integrated by RK4; input unbounded; constant/periodic blocks as in manuscript',
    'K=0 shadow observes K=1 data; K=0 control branch runs separately',
    'Specified scales/boxes; all energies and partials evaluated before the current update'])

def harmonics(phi):return np.array([1.,np.cos(phi),np.sin(phi)])
def harmonic_rates(phi):return np.array([0.,-np.sin(phi),np.cos(phi)])
def evaluated(c,phi):
    h=harmonics(phi);return c[:3]@h,c[3:]@h

def differential(x,rate,phi,c):
    rho,kappa=evaluated(c,phi);hp=harmonic_rates(phi)
    return np.array([kappa,rho,-1.,c[:3]@hp*rate+c[3:]@hp*x])

def residual(z,c):
    x,rate,u,phi=z;rho,kappa=evaluated(c,phi);return rho*rate+kappa*x-u

def refs(t):return .45*np.sin(.7*t)+.2*np.sin(1.3*t),.315*np.cos(.7*t)+.26*np.cos(1.3*t)

def rk4(x,u,t,dt,c,omega):
    def rhs(x,t):
        rho,kappa=evaluated(c,omega*t);return (u-kappa*x)/rho
    k1=rhs(x,t);k2=rhs(x+dt*k1/2,t+dt/2);k3=rhs(x+dt*k2/2,t+dt/2);k4=rhs(x+dt*k3,t+dt)
    return x+dt*(k1+2*k2+2*k3+k4)/6

def make_learner(cfg,K):
    ix=np.arange(6) if K==1 else np.array([0,3])
    args=[np.asarray(cfg[name])[ix] for name in ['initial','scales','lower','upper']]
    return ImplicitLearner(*args,cfg['dt'],cfg['memory'],int(cfg['steps']))

def expanded(c,K):
    if K==1:return c.copy()
    out=np.zeros(6);out[[0,3]]=c;return out

def simulate(seed=10,method='learned',cfg=CONFIG):
    if method not in ['learned', 'K0', 'frozen', 'oracle']: raise ValueError('Unknown simulation method')
    rng=np.random.default_rng(seed);dt=cfg['dt'];lag=round(cfg['difference_window']/dt);N=round(cfg['duration']/dt)
    truth=np.array(cfg['true_coefficients']);K=0 if method=='K0' else 1
    learner=make_learner(cfg,K);shadow=make_learner(cfg,0)
    measured=deque(maxlen=lag+1);clean=deque(maxlen=lag+1);rows=[];x=cfg['initial_state']
    for n in range(N+1):
        t=n*dt;phi=cfg['phase_rate']*t;h=harmonics(phi)
        c=expanded(learner.coefficients,K) if method!='oracle' else truth.copy()
        rho,kappa=evaluated(c,phi);rt,kt=evaluated(truth,phi);xd,xdot=refs(t)
        u=rho*(xdot-cfg['feedback_gain']*(x-xd))+kappa*x
        actual_rate=(u-kt*x)/rt;observed_rate=actual_rate+rng.normal(0,cfg['rate_noise'])
        F=differential(x,actual_rate,phi,truth);Fest=differential(x,actual_rate,phi,c)
        sc=np.array(cfg['context_scales']);ferr=float(np.linalg.norm((Fest-F)*sc)/np.linalg.norm(F*sc))
        f1=F*sc/np.linalg.norm(F*sc);f2=Fest*sc/np.linalg.norm(Fest*sc)
        projector_error=float(np.sqrt(max(0,2*(1-(f1@f2)**2))))
        feat=np.r_[h*observed_rate,h*x,-u];fc=np.r_[h*actual_rate,h*x,-u]
        measured.append(feat);clean.append(fc);E=Ec=Eo=En=eig=power=Eshadow=np.nan
        sh=expanded(shadow.coefficients,0)
        if len(measured)==lag+1:
            v=(measured[-1]-measured[0])/cfg['difference_window'];vc=(clean[-1]-clean[0])/cfg['difference_window']
            E=float((v@np.r_[c,1.])**2/2);Ec=float((vc@np.r_[c,1.])**2/2)
            Eo=float((v@np.r_[truth,1.])**2/2);Eshadow=float((v@np.r_[sh,1.])**2/2)
            ix=np.arange(6) if K==1 else np.array([0,3]);free=v[ix]
            d=learner.observe(free,v[-1],adapt=method in ('learned','K0') and n<N)
            channels=learner.channels(free,v[-1])[0];q=np.r_[c[ix]/learner.scales,1.]
            En=float((channels@q)**2/max(float((q@q)*(channels@channels)),1e-20))
            eig=d['memory_min_eigenvalue'];power=d['feature_power']
            shadow.observe(v[[0,3]],v[-1],adapt=n<N)
        row=dict(time=t,state=x,desired=xd,error=x-xd,control=u,F_error=ferr,projector_error=projector_error,
            rho_true=rt,rho_est=rho,kappa_true=kt,kappa_est=kappa,
            phase_partial_true=F[-1],phase_partial_est=Fest[-1],preupdate_energy=E,clean_energy=Ec,
            same_stream_oracle_energy=Eo,normalized_energy=En,feature_power=power,information_min_eigenvalue=eig,
            K0_same_stream_energy=Eshadow,coefficient_relative_error=float(np.linalg.norm(c-truth)/np.linalg.norm(truth)))
        for j in range(6):row[f'c{j}']=float(c[j])
        rows.append(row)
        if n<N:x=rk4(x,u,t,dt,truth,cfg['phase_rate'])
    return rows

def summarize(rows,cfg=CONFIG):
    lo=cfg['duration']-20
    late=[r for r in rows if lo<=r['time']<cfg['duration']]
    return dict(tracking_rmse=rms([r['error'] for r in late]),control_rms=rms([r['control'] for r in late]),
        F_error_mean=float(np.mean([r['F_error'] for r in late])),projector_error_mean=float(np.mean([r['projector_error'] for r in late])),
        energy_mean=float(np.nanmean([r['preupdate_energy'] for r in late])),
        oracle_energy_mean=float(np.nanmean([r['same_stream_oracle_energy'] for r in late])),
        clean_energy_mean=float(np.nanmean([r['clean_energy'] for r in late])),
        K0_same_stream_energy_mean=float(np.nanmean([r['K0_same_stream_energy'] for r in late])),
        normalized_energy_mean=float(np.nanmean([r['normalized_energy'] for r in late])),
        feature_power_mean=float(np.nanmean([r['feature_power'] for r in late])),
        final_coefficients=[rows[-1][f'c{j}'] for j in range(6)])

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--seeds',nargs='+',type=int,default=[10]);p.add_argument('--methods',nargs='+',default=['learned','K0','frozen','oracle'])
    p.add_argument('--duration',type=float,default=80.);a=p.parse_args();cfg=CONFIG.copy();cfg['duration']=a.duration
    dump_json(a.output/'config.json',cfg);summaries=[]
    for seed in a.seeds:
        for method in a.methods:
            rows=simulate(seed,method,cfg);s=dict(seed=seed,method=method,metrics=summarize(rows,cfg));dest=a.output/f'seed_{seed}'/method
            save_trace(dest/'trace.csv',rows);dump_json(dest/'summary.json',s);summaries.append(s);print(seed,method,s['metrics'],flush=True)
    dump_json(a.output/'summary.json',dict(status='NEW online-control protocol',runs=summaries))
if __name__=='__main__':main()
