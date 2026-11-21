"""New direct local-field contact study; all physical formulas are in Simulator.
Learner consumes only observed context rates. Controller receives only measured
context, generic task references, and a learned field. No residual fitting.
"""
from __future__ import annotations
import argparse, hashlib, json, platform, shutil, sys, time
from pathlib import Path
import numpy as np
import scipy
from direct_field import coupled_step, complete_rates, random_frame
D=np.array([.3,.03,5.,.15,.03,5.,.15,.03])
S=np.array([0,1,2,3,5]); C=np.array([4,6,7])

def dump(p,x): p.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
def digest(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def basis(F): return np.linalg.svd(F,full_matrices=False)[2][:3]
def probe(t,phase=0.):
    return np.array([.06*np.sin(1.1*t+phase)+.03*np.sin(2.3*t+.4+phase)+.015*np.sin(3.7*t+.2-phase), .002*np.sin(.9*t+.8+phase)+.0015*np.sin(1.9*t-phase)+.001*np.sin(3.1*t+.1)])

def refs(t):
    wx=np.array([.35,2.3,3.7]); ax=np.array([.03,.01,.005]); wl=np.array([.5,1.9,3.1]); al=np.array([.2,.08,.04])
    return np.array([ax@np.sin(wx*t),(ax*wx)@np.cos(wx*t),-(ax*wx**2)@np.sin(wx*t),4+al@np.sin(wl*t),(al*wl)@np.cos(wl*t),-(al*wl**2)@np.sin(wl*t)])

class Simulator:
    """Physical coefficients are used only for plant advance, sensing, evaluation."""
    def __init__(self, change):
        self.change=change; self.g=np.array([.035,.020]);self.y=np.array([0.,.020-4/150.,4.])
    def coefficients(self,t):
        k,eta=(250.,90.) if t>=self.change else (150.,60.)
        B=np.array([[.8,.22],[-.12,.65]]) if t>=self.change else np.array([[1.,.15],[-.08,.85]])
        return 4*k,4*eta,B
    def geometry(self,x):
        return (float(self.g@np.array([np.sin(2*x),np.cos(5*x)])),float(self.g@np.array([2*np.cos(2*x),-5*np.sin(5*x)])),float(self.g@np.array([-4*np.sin(2*x),-25*np.cos(5*x)])))
    def rhs(self,t,state,u):
        a,b,B=self.coefficients(t);x,y,lam=state;h,hp,_=self.geometry(x);vx,vy=B@u
        return np.array([vx,vy,a*(h-y)-4*lam+b*(hp*vx-vy)])
    def ideal_sensor(self,t,u,udot):
        # Ideal noise-free measurements of state, velocity, force rate and their
        # time derivatives. This SENSOR simulation supplies no partials to Q.
        x,y,lam=self.y;a,b,B=self.coefficients(t);h,hp,hpp=self.geometry(x)
        vx,vy=B@u;ax,ay=B@udot;ldot=a*(h-y)-4*lam+b*(hp*vx-vy)
        lddot=a*(hp*vx-vy)-4*ldot+b*(hpp*vx*vx+hp*ax-ay)
        return np.r_[x,y,lam,vx,vy,ldot,u],np.r_[vx,vy,ldot,ax,ay,lddot,udot]
    def evaluation(self,t,z):
        a,b,B=self.coefficients(t);h,hp,hpp=self.geometry(z[0]);F=np.zeros((3,8))
        F[0,:6]=[-a*hp-b*hpp*z[3],a,4.,-b*hp,b,1.]
        F[1,3]=1.;F[2,4]=1.;F[1:,-2:]=-B
        return F,h-z[1]
    def advance(self,t,u,udot,dt):
        # Piecewise linear command; physics fixed on [t,t+dt), right endpoint
        # uses old material when an exactly scheduled change occurs there.
        event=t<self.change<=t+dt; end=np.nextafter(t+dt,t) if event else t+dt
        rhs=lambda s,y:self.rhs(end if s==dt else t+s,y,u+s*udot)
        a=rhs(0.,self.y);b=rhs(dt/2,self.y+dt*a/2);c=rhs(dt/2,self.y+dt*b/2);d=rhs(dt,self.y+dt*c)
        self.y=self.y+dt*(a+2*b+2*c+d)/6

def command(q,z,target,cfg):
    """Generic rate feedback and learned completion; no simulator/truth access."""
    xd,vd,ad,ld,ldd,lddd=target
    ax=ad-4*(z[3]-vd)-4*(z[0]-xd)
    al=lddd-6*(z[5]-ldd)-9*(z[2]-ld)
    supplied=np.array([z[3],z[4],z[5],ax,al])
    out=complete_rates(q,S,supplied/D[S],C,minimum_singular_value=cfg['guard'])*D[C]
    return out[1:],supplied

def run(dest,cfg):
    dest.mkdir(parents=True,exist_ok=False);dump(dest/'config.json',cfg)
    for name in ['contact_direct.py','direct_field.py']: shutil.copy2(Path(__file__).parent/name,dest/name)
    dump(dest/'environment.json',dict(python=sys.version,numpy=np.__version__,scipy=scipy.__version__,platform=platform.platform(),command=sys.argv,source_hashes={n:digest(dest/n) for n in ['contact_direct.py','direct_field.py']}))
    plant=Simulator(cfg['change_time']);q=random_frame(3,8,cfg['seed']+10000);q0=q.copy();R=np.zeros((8,8));dt=cfg['dt']
    u=probe(0.,cfg['phase']);past_udot=(probe(dt,cfg['phase'])-u)/dt
    rows=[];Qs=[];Rs=[];Zs=[];Vs=[];Fs=[];Sup=[];nfevs=[];status='complete';error=None;start=time.perf_counter();correction=0.
    for n in range(round(cfg['duration']/dt)+1):
        t=n*dt;z,obs=plant.ideal_sensor(t,u,past_udot);v=obs/D;target=refs(max(t-cfg['warmup'],0))
        sigma=float(np.linalg.svd(q[:,C],compute_uv=False)[-1]);guard=False;raw=np.zeros(2);supplied=np.full(5,np.nan)
        if t<cfg['warmup']: raw=(probe(t+dt,cfg['phase'])-u)/dt
        else:
            try: raw,supplied=command(q,z,target,cfg)
            except ValueError as exc:guard=True;status='readout_failure';error=str(exc)
        # No physical fallback. Guard failure is recorded and terminates the run.
        limited=np.clip(raw,-cfg['input_rate_limit'],cfg['input_rate_limit'])
        unext=np.clip(u+dt*limited,-cfg['input_limit'],cfg['input_limit']);udot=(unext-u)/dt
        saturation=bool(np.max(np.abs(raw-limited))>1e-12 or np.max(np.abs(unext-(u+dt*limited)))>1e-12)
        F,indent=plant.evaluation(t,z);truth=basis(F*D);P=truth.T@truth;ev,U=np.linalg.eigh(R);bottom=U[:,:3].T
        # Evaluator-only completion error for the same requested query.
        qe=qf=np.nan
        if np.isfinite(supplied).all():
            truthout=np.linalg.solve(truth[:,C],-truth[:,S]@(supplied/D[S]))*D[C]
            try:learnout=complete_rates(q,S,supplied/D[S],C,minimum_singular_value=cfg['guard'])*D[C];qe=float(np.linalg.norm(learnout-truthout))
            except ValueError:pass
            try:frout=complete_rates(q0,S,supplied/D[S],C,minimum_singular_value=cfg['guard'])*D[C];qf=float(np.linalg.norm(frout-truthout))
            except ValueError:pass
        rows.append([t,*z[:3],target[0],target[3],np.linalg.norm(q.T@q-P),np.linalg.norm(q0.T@q0-P),np.linalg.norm(q@v),np.linalg.norm(q0@v),np.linalg.norm(truth@v),sigma,float(guard),float(saturation),indent,qe,qf,np.linalg.norm(q.T@q-bottom.T@bottom),np.linalg.norm(bottom.T@bottom-P),ev[3]-ev[2],*u,*udot])
        Qs.append(q.copy());Rs.append(R.copy());Zs.append(z);Vs.append(obs);Fs.append(F);Sup.append(supplied)
        if guard:break
        if not np.isfinite(z).all() or abs(z[0])>1 or abs(z[1])>.3 or abs(z[2])>100 or indent<=0 or z[2]<=0:
            status='domain_failure';error=f't={t}, state={z[:3].tolist()}, indentation={indent}';break
        if n<round(cfg['duration']/dt):
            try:q,R,dg=coupled_step(q,R,v,dt,1/cfg['tau'],cfg['gain']);correction=max(correction,dg['polar_correction']);nfevs.append(dg['function_evaluations'])
            except Exception as exc:status='integration_failure';error=repr(exc);break
            plant.advance(t,u,udot,dt);u=unext;past_udot=udot
    cols='time,x,y,lambda,desired_x,desired_lambda,projector_error,frozen_projector_error,incompatibility,frozen_incompatibility,true_incompatibility,readout_sigma,guarded,saturated,indentation,query_error,frozen_query_error,learner_memory_distance,memory_bias,eigengap,u1,u2,udot1,udot2'.split(',')
    arr=np.array(rows);np.savetxt(dest/'trace.csv',arr,delimiter=',',header=','.join(cols),comments='')
    np.savez_compressed(dest/'states.npz',q_pre=Qs,R_pre=Rs,z=Zs,v_observed=Vs,F_evaluator=Fs,supplied_query=Sup,q_initial=q0,scales=D,nfev=nfevs)
    summary=dict(status=status,error=error,samples=len(rows),runtime_seconds=time.perf_counter()-start,max_correction=correction)
    for name,lo,hi in [('end_training',cfg['warmup']-10,cfg['warmup']),('control',cfg['warmup'],cfg['duration']),('late',cfg['duration']-10,cfg['duration'])]:
        m=(arr[:,0]>=lo)&(arr[:,0]<hi)
        if not np.any(m):continue
        result={c:float(np.nanmean(arr[m,j])) if np.isfinite(arr[m,j]).any() else None for j,c in enumerate(cols) if c in ['projector_error','frozen_projector_error','incompatibility','frozen_incompatibility','guarded','saturated','learner_memory_distance','memory_bias','eigengap']}
        result.update(x_rmse=float(np.sqrt(np.mean((arr[m,1]-arr[m,4])**2))),lambda_rmse=float(np.sqrt(np.mean((arr[m,3]-arr[m,5])**2))),min_sigma=float(np.min(arr[m,11])),min_indentation=float(np.min(arr[m,14])))
        for c in ['query_error','frozen_query_error']:
            vv=arr[m,cols.index(c)];good=np.isfinite(vv);result[c+'_rms']=float(np.sqrt(np.mean(vv[good]**2))) if good.any() else None;result[c+'_count']=int(good.sum())
        summary[name]=result
    dump(dest/'summary.json',summary);dump(dest/'manifest.json',{p.name:digest(p) for p in dest.iterdir() if p.is_file()});print(json.dumps(summary),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--seed',type=int,default=0);p.add_argument('--phase',type=float,default=0.);p.add_argument('--dt',type=float,default=.01);p.add_argument('--tau',type=float,default=2.);p.add_argument('--gain',type=float,default=100.);p.add_argument('--duration',type=float,default=60.);p.add_argument('--warmup',type=float,default=40.);p.add_argument('--change-time',type=float,default=1e9);a=p.parse_args()
    cfg=dict(identity='contact_direct_ideal_temporal_20261120_v2_local_excited',seed=a.seed,phase=a.phase,dt=a.dt,tau=a.tau,gain=a.gain,duration=a.duration,warmup=a.warmup,change_time=a.change_time,guard=.005,input_limit=.8,input_rate_limit=5.,scales=D.tolist(),observations='ideal measured z and physical time derivative; no sensor differentiation/noise robustness claimed',initialization='generic Gaussian QR seed+10000',physical_partial_entries_supplied=0,adaptive_entries=24,control_task='x and vertical force lambda',integrator='DOP853 coupled normalized flow with exact evolving held-rate memory; plant RK4 piecewise-linear input',frozen_baseline='same-stream initial generic Q, not separately controlling')
    run(a.output,cfg)
