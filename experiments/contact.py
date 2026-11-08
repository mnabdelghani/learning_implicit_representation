"""NEW contact reconstruction: online integral learning during force/motion control.
Original implementation was not supplied. Geometry, plant, noise, references and
change are from the manuscript; initialization, scales, boxes, readout details,
integration and diagnostic definitions are newly explicit choices (see config).
"""
from __future__ import annotations
import argparse
from collections import deque
from pathlib import Path
import numpy as np
from common import ImplicitLearner,rms,save_trace,dump_json,rowspace_error

CONFIG=dict(dt=.01,duration=120.,change_time=60.,memory=4.,window=.1,steps=48,
    rho=4.,geometry=[.035,.020],material_before=[150.,60.],material_after=[250.,90.],
    B_before=[[1.,.15],[-.08,.85]],B_after=[[.8,.22],[-.12,.65]],
    initial_material=[8.,4.,400.,3.2,1.6,160.],
    material_scales=[20.,20.,1000.,10.,10.,300.],
    material_lower=[-100.,-100.,200.,-50.,-50.,60.],
    material_upper=[100.,100.,1600.,50.,50.,700.],
    initial_B=[[1.,0.],[0.,1.]],force_noise=.01,velocity_noise=.001,
    motion_gain=4.,force_gain=20.,control_limit=.8,
    context_scales=[.5,.05,5.,.3,.3,10.,.5,.5],row_scales=[.1,1/.3,1/.3],
    new_choices=['Initial material and B; explicit coefficient boxes and optimization scales',
    'B box: diagonal [0.5,1.4], off-diagonal [-0.3,0.3], determinant at least 0.16',
    'Both material and B use projected steps at every inner iteration',
    'Normal-force conversion uses measured force direction and fitted curvature s/a',
    'Held-command RK4; event at 60 s; integral features use trapezoids',
    'Same-stream true-model integral energy is evaluator-only; change-crossing windows retained',
    'Unweighted time means/RMS; right endpoint excluded; no seed selection'])

def beta(x): return np.array([np.sin(2*x),np.cos(5*x)])
def beta1(x): return np.array([2*np.cos(2*x),-5*np.sin(5*x)])
def beta2(x): return np.array([-4*np.sin(2*x),-25*np.cos(5*x)])

def truth(t,cfg=CONFIG):
    late=t>=cfg['change_time'];k,eta=cfg['material_after' if late else 'material_before']
    a,b=cfg['rho']*k,cfg['rho']*eta;g=np.array(cfg['geometry'])
    return np.r_[a*g,a,b*g,b],np.array(cfg['B_after' if late else 'B_before'])

def refs(t):
    return (.45*np.sin(.6*t)+.08*np.sin(1.7*t),.27*np.cos(.6*t)+.136*np.cos(1.7*t),
            4+.7*np.sin(1.1*t)+.4*np.sin(2.3*t),.77*np.cos(1.1*t)+.92*np.cos(2.3*t))

def material_rate(x,y,lam,v,c,rho=4.):
    return float(c[:2]@beta(x)-c[2]*y-rho*lam+c[3:5]@beta1(x)*v[0]-c[5]*v[1])

def physical_rhs(state,u,c,B,rho=4.):
    x,y,lam=state;v=B@u
    return np.r_[v,material_rate(x,y,lam,v,c,rho)]

def residual(z,c,B,rho=4.):
    x,y,lam,vx,vy,ld,u1,u2=np.asarray(z)
    return np.r_[ld-material_rate(x,y,lam,[vx,vy],c,rho),np.array([vx,vy])-B@np.array([u1,u2])]

def differential(z,c,B,rho=4.):
    x,y,lam,vx,vy,ld,u1,u2=np.asarray(z)
    F=np.zeros((3,8))
    F[0,:6]=[-c[:2]@beta1(x)-c[3:5]@beta2(x)*vx,c[2],rho,-c[3:5]@beta1(x),c[5],1.]
    F[1,3]=1.;F[2,4]=1.;F[1:,-2:]=-B
    return F

def rk4(state,u,dt,c,B,rho):
    k1=physical_rhs(state,u,c,B,rho);k2=physical_rhs(state+dt*k1/2,u,c,B,rho)
    k3=physical_rhs(state+dt*k2/2,u,c,B,rho);k4=physical_rhs(state+dt*k3,u,c,B,rho)
    return state+dt*(k1+2*k2+2*k3+k4)/6

def simulate(seed=10,method='learned',cfg=None,*,reference=None,probe=None,freeze_after=float('inf'),extra_diagnostics=False):
    if method not in ['learned', 'frozen', 'oracle']: raise ValueError('Unknown simulation method')
    cfg=CONFIG.copy() if cfg is None else cfg
    reference=refs if reference is None else reference
    clean_R=np.zeros((6,6))
    dt=cfg['dt'];N=round(cfg['duration']/dt);lag=round(cfg['window']/dt)
    rng=np.random.default_rng(seed);g=np.array(cfg['geometry']);rho=cfg['rho']
    mlearner=ImplicitLearner(cfg['initial_material'],cfg['material_scales'],cfg['material_lower'],
        cfg['material_upper'],dt,cfg['memory'],cfg['steps'])
    blearners=[ImplicitLearner(cfg['initial_B'][i],[1.,1.],[.5,-.3] if i==0 else [-.3,.5],
         [1.4,.3] if i==0 else [.3,1.4],dt,cfg['memory'],cfg['steps']) for i in range(2)]
    x0=0.;lam0=4/np.sqrt(1+(g@beta1(x0))**2)
    state=np.array([x0,g@beta(x0)-lam0/cfg['material_before'][0],lam0])
    hist=deque(maxlen=lag+1);clean_hist=deque(maxlen=lag+1);rows=[];Fpreds=[];Ftrues=[]
    weights=np.r_[.5,np.ones(lag-1),.5]*dt
    for i in range(N+1):
        t=i*dt;c,B=truth(t,cfg);chat=mlearner.coefficients.copy();Bhat=np.array([l.coefficients for l in blearners])
        if method=='oracle': chat=c.copy();Bhat=B.copy()
        x,y,lam=state;hp=float(g@beta1(x));h=float(g@beta(x))
        force=lam*np.array([-hp,1.]);measured_force=force+rng.normal(0,cfg['force_noise'],2)
        lm=measured_force[1];Lm=float(np.linalg.norm(measured_force));L=float(np.linalg.norm(force))
        xd,xdotd,Ld,Ldotd=reference(t)
        vxcmd=xdotd-cfg['motion_gain']*(x-xd)
        measured_slope=-measured_force[0]/max(lm,.2)
        mhat=np.sqrt(1+measured_slope**2)
        hpphat=float((chat[:2]/chat[2])@beta2(x))
        mdothat=measured_slope*hpphat*vxcmd/mhat
        ldotcmd=(Ldotd-cfg['force_gain']*(Lm-Ld)-lm*mdothat)/mhat
        vycmd=(chat[:2]@beta(x)-chat[2]*y-rho*lm+chat[3:5]@beta1(x)*vxcmd-ldotcmd)/chat[5]
        raw_u=np.linalg.solve(Bhat,np.array([vxcmd,vycmd]))
        if probe is not None and t<freeze_after:
            increment=np.asarray(probe(t),dtype=float)
            if increment.shape!=(2,) or not np.all(np.isfinite(increment)):raise ValueError('Probe must be a finite 2-vector')
            raw_u=raw_u+increment
        u=np.clip(raw_u,-cfg['control_limit'],cfg['control_limit'])
        vel=B@u;vm=vel+rng.normal(0,cfg['velocity_noise'],2)
        ldot=material_rate(x,y,lam,vel,c,rho)
        z=np.r_[x,y,lam,vel,ldot,u]
        F=differential(z,c,B,rho);Fhat=differential(z,chat,Bhat,rho)
        scaled=F*np.asarray(cfg['context_scales'])*np.asarray(cfg['row_scales'])[:,None]
        scaledhat=Fhat*np.asarray(cfg['context_scales'])*np.asarray(cfg['row_scales'])[:,None]
        ferr=float(np.linalg.norm(scaledhat-scaled)/np.linalg.norm(scaled))
        projerr=rowspace_error(F,Fhat,cfg['context_scales'])
        hist.append(np.r_[beta(x),y,lm]);clean_hist.append(np.r_[beta(x),y,lam])
        E=Ec=Eoracle=En=power=eig=np.nan
        jump_window=t>=cfg['change_time'] and t-cfg['window']<cfg['change_time']
        if len(hist)==lag+1:
            H=np.asarray(hist);C=np.asarray(clean_hist);I=weights@H;Ic=weights@C
            d=H[-1]-H[0];dc=C[-1]-C[0]
            free=np.r_[-I[:2],I[2],-d[:2],d[2]]/cfg['window']
            supplied=(d[3]+rho*I[3])/cfg['window']
            vclean=np.r_[-Ic[:2],Ic[2],-dc[:2],dc[2]]/cfg['window']
            bclean=(dc[3]+rho*Ic[3])/cfg['window']
            E=float((free@chat+supplied)**2/2);Ec=float((vclean@chat+bclean)**2/2)
            Eoracle=float((free@c+supplied)**2/2)
            diag=mlearner.observe(free,supplied,adapt=method=='learned' and i<N and t<freeze_after)
            channels=mlearner.channels(free,supplied)[0];q=np.r_[chat/mlearner.scales,1.]
            En=float((channels@q)**2/max(float((q@q)*(channels@channels)),1e-20))
            power=diag['feature_power'];eig=diag['memory_min_eigenvalue']
        for j,l in enumerate(blearners):l.observe(-u,vm[j],adapt=method=='learned' and i<N and t<freeze_after)
        row=dict(time=t,x=x,desired_x=xd,normal_force=L,desired_force=Ld,x_error=x-xd,force_error=L-Ld,
            u1=u[0],u2=u[1],control_norm=float(np.linalg.norm(u)),saturated=float(np.max(np.abs(raw_u))>cfg['control_limit']),
            indentation=h-y,preupdate_energy=E,clean_energy=Ec,same_stream_oracle_energy=Eoracle,
            normalized_energy=En,feature_power=power,information_min_eigenvalue=eig,
            F_error=ferr,projector_error=projerr,Fx_true=F[0,0],Fx_est=Fhat[0,0],
            Fvx_true=F[0,3],Fvx_est=Fhat[0,3],force_rate_true=ldot,
            force_rate_pred=material_rate(x,y,lam,vel,chat,rho),jump_window=float(jump_window))
        for j,name in enumerate(['s1','s2','a','r1','r2','b']):row[name]=float(chat[j]);row[name+'_true']=float(c[j])
        for j in range(2):
            for k in range(2):row[f'B{j+1}{k+1}']=float(Bhat[j,k])
        if extra_diagnostics:
            if np.isfinite(E):
                vf=vclean*mlearner.scales
                clean_R=mlearner.beta*clean_R+(1-mlearner.beta)*np.outer(vf,vf)
            for k,val in enumerate(np.linalg.eigvalsh(clean_R)):
                row[f'clean_information_eigenvalue_{k+1}']=float(val)
            task_matrix=np.vstack((Bhat[0],float(chat[3:5]@beta1(x))*Bhat[0]-chat[5]*Bhat[1]))
            task_scaled=np.diag([1/.3,1/10])@task_matrix*.5
            row['readout_sigma_min']=float(np.linalg.svd(task_scaled,compute_uv=False)[-1])
            row['learning_enabled']=float(method=='learned' and i<N and t<freeze_after)
        rows.append(row)
        if i%10==0:Fpreds.append(Fhat);Ftrues.append(F)
        if i<N:
            state=rk4(state,u,dt,c,B,rho)
            if not np.all(np.isfinite(state)): raise FloatingPointError(f'Contact nonfinite: {seed=} {method=} {t=}')
    return rows,dict(F_est=np.array(Fpreds),F_true=np.array(Ftrues),time=np.arange(0,N+1,10)*dt)

def summarize(rows,cfg=CONFIG):
    def vals(name,lo,hi):return np.array([r[name] for r in rows if lo<=r['time']<hi])
    out={}
    for name,lo,hi in [('early',0,20),('before_change',40,60),('change',60,80),('recovered',100,120)]:
        if lo>=cfg['duration']:continue
        hi=min(hi,cfg['duration'])
        out[name]=dict(force_rmse=rms(vals('force_error',lo,hi)),motion_rmse=rms(vals('x_error',lo,hi)),
            F_error_mean=float(np.mean(vals('F_error',lo,hi))),projector_error_mean=float(np.mean(vals('projector_error',lo,hi))),
            incompatibility_mean=float(np.nanmean(vals('preupdate_energy',lo,hi))),
            oracle_energy_mean=float(np.nanmean(vals('same_stream_oracle_energy',lo,hi))),
            clean_energy_mean=float(np.nanmean(vals('clean_energy',lo,hi))),
            normalized_energy_mean=float(np.nanmean(vals('normalized_energy',lo,hi))),
            feature_power_mean=float(np.nanmean(vals('feature_power',lo,hi))),
            control_rms=rms(vals('control_norm',lo,hi)))
    out['minimum_indentation']=min(r['indentation'] for r in rows)
    out['saturation_fraction']=float(np.mean([r['saturated'] for r in rows[:-1]]))
    out['final_material']=[rows[-1][n] for n in ['s1','s2','a','r1','r2','b']]
    return out

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--seeds',nargs='+',type=int,default=[10]);p.add_argument('--methods',nargs='+',default=['learned','frozen','oracle'])
    p.add_argument('--duration',type=float,default=120.);args=p.parse_args();cfg=CONFIG.copy();cfg['duration']=args.duration
    dump_json(args.output/'config.json',cfg)
    summaries=[]
    for seed in args.seeds:
        for method in args.methods:
            rows,F=simulate(seed,method,cfg);dest=args.output/f'seed_{seed}'/method
            save_trace(dest/'trace.csv',rows);np.savez_compressed(dest/'partials.npz',**F)
            s=dict(seed=seed,method=method,metrics=summarize(rows,cfg));dump_json(dest/'summary.json',s);summaries.append(s)
            print(seed,method,s['metrics'].get('recovered',s['metrics'].get('early')),flush=True)
    dump_json(args.output/'summary.json',dict(status='NEW RECONSTRUCTION; not original results',runs=summaries))
if __name__=='__main__':main()
