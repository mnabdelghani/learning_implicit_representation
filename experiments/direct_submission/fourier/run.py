"""Direct Fourier field benchmark; all scientific inputs are in sealed protocol.json.

Run outputs are immutable identities. Learner entry point `update` accepts only
coefficient/covariance state and observed rates/known phase. Plant and truth
functions are not accessed there or in `readout`. Raw information streams and
all coefficient/covariance states are retained for independent reconstruction.
"""
from pathlib import Path
import argparse, hashlib, json, platform, sys, time
import numpy as np
import scipy
from scipy.integrate import solve_ivp

HERE=Path(__file__).resolve().parent

def save(path,obj):
    path.write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n')

def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()

def coupled_step(c,R,w,dt,memory,gain,rtol,atol):
    outer=np.outer(w,w)
    def local(s):
        return np.exp(-s/memory)*R-memory*np.expm1(-s/memory)*outer
    sol=solve_ivp(lambda s,y:-gain*(local(s)@y),(0.,dt),c,
                  method='DOP853',rtol=rtol,atol=atol)
    if not sol.success: raise RuntimeError(sol.message)
    y=sol.y[:,-1];length=np.linalg.norm(y)
    if not np.isfinite(length) or length<np.finfo(float).tiny:
        raise FloatingPointError('Unnormalizable coefficient state')
    return y/length,local(dt)

def harmonic(phase):
    return np.array([1.,np.cos(phase),np.sin(phase)])

def update(c,R,observed_velocity,phase,cfg,K):
    w=np.kron(harmonic(phase),observed_velocity) if K else observed_velocity
    cpost,Rpost=coupled_step(c,R,w,cfg['dt'],cfg['memory'],cfg['gain'],cfg['rtol'],cfg['atol'])
    return cpost,Rpost

def evaluate(c,phase,K):
    return harmonic(phase)@c.reshape(3,2) if K else c.copy()

def readout(c,phase,xdot,guard,K):
    h=evaluate(c,phase,K)
    good=abs(h[1])>=guard
    return (-h[0]*xdot/h[1] if good else np.nan),good

def simulator(x,xdot,p):
    y=p['offset']+p['a']*x+p['b']/p['omega']*np.sin(p['omega']*x)
    slope=p['a']+p['b']*np.cos(p['omega']*x)
    return y,slope*xdot

def projector_error(h,f):
    h=h/np.linalg.norm(h,axis=1)[:,None]
    f=f/np.linalg.norm(f,axis=1)[:,None]
    return np.sqrt(np.maximum(0.,2*(1-np.sum(h*f,axis=1)**2)))

def rmse(a):
    return float(np.sqrt(np.mean(a*a))) if np.all(np.isfinite(a)) else None

def variable_frequency(a,dt):
    # Derivative RMS divided by centered signal RMS is an empirical rate scale.
    # A ramp has a duration-dependent value; this is not its spectral frequency.
    a=np.asarray(a);d=np.gradient(a,dt,axis=0)
    den=np.mean((a-np.mean(a,axis=0))**2)
    return float(np.sqrt(np.mean(d*d)/den)) if den>0 else None

def run(output,seed,cfg):
    started=time.monotonic();output.mkdir(exist_ok=False)
    save(output/'config.json',cfg)
    (output/'run.py').write_bytes(Path(__file__).read_bytes())
    rng=np.random.default_rng(seed);save(output/'rng_initial.json',rng.bit_generator.state)
    c=rng.normal(size=6);c/=np.linalg.norm(c);initial=c.copy()
    c0=c[:2]/np.linalg.norm(c[:2]);initial0=c0.copy()
    x0=rng.uniform(0.,2*np.pi/cfg['known_phase_scale'])
    dt=cfg['dt'];t=np.arange(round(cfg['duration']/dt)+1)*dt
    x=x0+.15*t+.05*np.sin(.3*t);dx=.15+.015*np.cos(.3*t)
    y,dy=simulator(x,dx,cfg['plant_only'])
    # Sensor boundary: learner only accesses measured_v and phase below.
    measured_v=np.column_stack((dx,dy+rng.normal(0,cfg['rate_noise'],len(t))))
    phase=cfg['known_phase_scale']*x
    Cs=np.empty((len(t),6));Rs=np.empty((len(t),6,6))
    C0s=np.empty((len(t),2));R0s=np.empty((len(t),2,2))
    R=np.zeros((6,6));R0=np.zeros((2,2))
    for n in range(len(t)):
        Cs[n]=c;Rs[n]=R;C0s[n]=c0;R0s[n]=R0
        if n==len(t)-1:break
        c,R=update(c,R,measured_v[n],phase[n],cfg,1)
        c0,R0=update(c0,R0,measured_v[n],phase[n],cfg,0)
    np.savez_compressed(output/'training.npz',time=t,x=x,observed_v=measured_v,
                        phase=phase,C_pre=Cs,R_pre=Rs,C0_pre=C0s,R0_pre=R0s,
                        evaluator_y=y,evaluator_ydot=dy)
    # All subsequent physical rowspace/target diagnostics are evaluator only.
    h=np.column_stack((np.ones(len(t)),np.cos(phase),np.sin(phase)))
    H=np.einsum('ni,nij->nj',h,Cs.reshape(-1,3,2))
    W=np.einsum('ni,nj->nij',h,measured_v).reshape(-1,6)
    clean_v=np.column_stack((dx,dy));Wclean=np.einsum('ni,nj->nij',h,clean_v).reshape(-1,6)
    truth=np.column_stack((-dy/dx,np.ones(len(t))))
    error=projector_error(H,truth);error0=projector_error(C0s,truth)
    frozenH=h@initial.reshape(3,2)
    energy=.5*np.sum(H*measured_v,axis=1)**2
    energy0=.5*np.sum(C0s*measured_v,axis=1)**2
    frozen_energy=.5*np.sum(frozenH*measured_v,axis=1)**2
    phasebound=np.abs(Cs[:,1])-np.hypot(Cs[:,3],Cs[:,5])
    late=(t>=cfg['late_window'][0])&(t<cfg['late_window'][1])
    test_t=np.arange(round(cfg['test_duration']/dt)+1)*dt
    test_x=x[-1]+.12*test_t+.03/.47*(1-np.cos(.47*test_t))
    test_dx=.12+.03*np.sin(.47*test_t)
    test_phase=cfg['known_phase_scale']*test_x
    truth_y,truth_dy=simulator(test_x,test_dx,cfg['plant_only'])
    forward={};test_arrays=dict(time=test_t,x=test_x,xdot=test_dx,phase=test_phase,
                               evaluator_y=truth_y,evaluator_ydot=truth_dy)
    for name,C,K in [('learned',c,1),('K0',c0,0),('frozen_prior',initial,1)]:
        vals=[readout(C,ph,d,cfg['guard'],K) for ph,d in zip(test_phase,test_dx)]
        predicted=np.asarray([z[0] for z in vals]);valid=np.asarray([z[1] for z in vals])
        # One measured initial y, then integrate predicted rates without assimilation.
        yp=np.r_[truth_y[0],truth_y[0]+np.cumsum(dt*(predicted[:-1]+predicted[1:])/2)]
        forward[name]=dict(invalid_queries=int(np.sum(~valid)),rate_rmse=rmse(predicted-truth_dy),
                           valid_only_rate_rmse=rmse(predicted[valid]-truth_dy[valid]),
                           y_rmse=rmse(yp-truth_y),final_y_error=float(yp[-1]-truth_y[-1]) if np.isfinite(yp[-1]) else None)
        test_arrays[name+'_rate']=predicted;test_arrays[name+'_valid']=valid;test_arrays[name+'_y']=yp
    np.savez_compressed(output/'heldout.npz',**test_arrays)
    # Clean covariance and noisy covariance are separate diagnostics.
    Rclean=np.zeros((6,6));decay=np.exp(-dt/cfg['memory']);weight=-cfg['memory']*np.expm1(-dt/cfg['memory'])
    for w in Wclean[:-1]: Rclean=decay*Rclean+weight*np.outer(w,w)
    target=np.array([-cfg['plant_only']['a'],1.,-cfg['plant_only']['b'],0.,0.,0.]);target/=np.linalg.norm(target)
    ev,vec=np.linalg.eigh(R);evclean=np.linalg.eigvalsh(Rclean)
    cerror=lambda cc:float(np.sqrt(max(0.,2*(1-(cc@target)**2))))
    criteria=cfg['confirmation_criteria'];fq=forward['learned']
    pass_checks=dict(late_projector=float(np.mean(error[late]))<criteria['late_projector_error_mean_max'],
                     heldout_rate=fq['rate_rmse'] is not None and fq['rate_rmse']<criteria['heldout_rate_rmse_max'],
                     heldout_y=fq['y_rmse'] is not None and fq['y_rmse']<criteria['heldout_y_rmse_max'],
                     heldout_guard=fq['invalid_queries']<=criteria['heldout_invalid_queries_max'])
    slope=-truth[:,0]
    summary=dict(identity=cfg['identity'],seed=seed,initial_C=initial.tolist(),final_C=c.tolist(),
                 initial_K0=initial0.tolist(),final_K0=c0.tolist(),
                 coefficient_total_variation=np.sum(np.abs(np.diff(Cs,axis=0)),axis=0).tolist(),
                 coefficient_net_changes=(c-initial).tolist(),max_norm_error=float(np.max(np.abs(np.linalg.norm(Cs,axis=1)-1))),
                 late_projector_mean=float(np.mean(error[late])),late_K0_projector_mean=float(np.mean(error0[late])),
                 late_energy_mean=float(np.mean(energy[late])),late_K0_energy_mean=float(np.mean(energy0[late])),
                 late_frozen_energy_mean=float(np.mean(frozen_energy[late])),
                 training_query_guard_count=int(np.sum(np.abs(H[:,1])<cfg['guard'])),
                 final_all_phase_Hy_lower_bound=float(phasebound[-1]),minimum_training_all_phase_bound=float(np.min(phasebound)),
                 coefficient_projector_error=cerror(c),bottom_eigenvector_target_error=cerror(vec[:,0]),
                 covariance_eigenvalues=ev.tolist(),clean_covariance_eigenvalues=evclean.tolist(),
                 clean_target_memory_bias=float(np.linalg.norm(Rclean@target)),
                 observed_target_memory_bias=float(np.linalg.norm(R@target-(target@R@target)*target)),
                 coefficient_gap=float(ev[1]-ev[0]),clean_coefficient_gap=float(evclean[1]-evclean[0]),
                 forward=forward,criteria_checks=pass_checks,qualifies=all(pass_checks.values()),
                 time_scales=dict(train_partial_phase_rate_range=[float(np.min(20*dx)),float(np.max(20*dx))],
                    test_partial_phase_rate_range=[float(np.min(20*test_dx)),float(np.max(20*test_dx))],
                    train_xdot_modulation_rad_s=.3,test_xdot_modulation_rad_s=.47,
                    train_x_variation_rate_scale=variable_frequency(x,dt),train_xdot_variation_rate_scale=variable_frequency(dx,dt),
                    train_physical_slope_variation_rate_scale=variable_frequency(slope,dt),
                    train_output_rate_variation_rate_scale=variable_frequency(dy,dt),
                    test_x_variation_rate_scale=variable_frequency(test_x,dt),test_xdot_variation_rate_scale=variable_frequency(test_dx,dt),
                    caveat='x is a monotone ramp plus slow modulation, so its centered derivative/RMS statistic is window dependent, not a spectral frequency. The phase is derived from supplied x and is not a separate manipulated input. Output rate is fast.'),
                 elapsed_seconds=time.monotonic()-started)
    save(output/'summary.json',summary);save(output/'rng_final.json',rng.bit_generator.state)
    save(output/'environment.json',dict(python=platform.python_version(),numpy=np.__version__,scipy=scipy.__version__,
                                      argv=sys.argv,source_sha256=sha(Path(__file__)),protocol_sha256=sha(HERE/'protocol.json')))
    save(output/'manifest.json',{p.name:sha(p) for p in sorted(output.iterdir()) if p.is_file()})
    return summary

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--seed',type=int,required=True);ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--dt',type=float);ap.add_argument('--noise',type=float)
    args=ap.parse_args();cfg=json.loads((HERE/'protocol.json').read_text())
    if args.dt is not None:cfg['dt']=args.dt
    if args.noise is not None:cfg['rate_noise']=args.noise
    try:
        result=run(args.output,args.seed,cfg)
        print(json.dumps({k:result[k] for k in ['seed','late_projector_mean','forward','qualifies','elapsed_seconds']},indent=2))
    except Exception as exc:
        if args.output.exists():save(args.output/'failure.json',dict(type=type(exc).__name__,message=str(exc)))
        raise
