"""Independent raw-record audit; does not import the experiment implementation."""
from pathlib import Path
import hashlib,json
import numpy as np
from scipy.integrate import solve_ivp
HERE=Path(__file__).resolve().parent

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,x):p.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
def p_error(H,F):
    U=H[:,:,None]*H[:,None,:]/np.sum(H*H,axis=1)[:,None,None]
    V=F[:,:,None]*F[:,None,:]/np.sum(F*F,axis=1)[:,None,None]
    return np.linalg.norm(U-V,axis=(1,2))
def r2(x):return float(np.sqrt(np.mean(x*x)))
def check(path):
    cfg=json.loads((path/'config.json').read_text());s=json.loads((path/'summary.json').read_text())
    manifest=json.loads((path/'manifest.json').read_text())
    assert all(sha(path/n)==v for n,v in manifest.items()),'file hash mismatch'
    assert sha(path/'run.py')==sha(HERE/'run.py'),'source differs'
    d=np.load(path/'training.npz');f=np.load(path/'heldout.npz');dt=cfg['dt'];N=len(d['time'])
    t=d['time'];phase=d['phase'];v=d['observed_v'];C=d['C_pre'];C0=d['C0_pre'];R=d['R_pre'];R0=d['R0_pre']
    h=np.stack((np.ones(N),np.cos(phase),np.sin(phase)),axis=1)
    W=(h[:,:,None]*v[:,None,:]).reshape(N,6)
    H=(h[:,:,None]*C.reshape(N,3,2)).sum(axis=1)
    F=np.stack((-d['evaluator_ydot']/v[:,0],np.ones(N)),axis=1)
    late=(t>=cfg['late_window'][0])&(t<cfg['late_window'][1])
    observed_metrics=dict(late_projector_mean=float(p_error(H,F)[late].mean()),
                          late_K0_projector_mean=float(p_error(C0,F)[late].mean()),
                          late_energy_mean=float((.5*(H*v).sum(axis=1)**2)[late].mean()),
                          late_K0_energy_mean=float((.5*(C0*v).sum(axis=1)**2)[late].mean()))
    metric_max=max(abs(value-s[key]) for key,value in observed_metrics.items())
    # The saved sqrt(2*(1-cos^2)) formula loses precision at exact recovery.
    # Preserve original clean summaries and permit only its diagnosed roundoff.
    assert metric_max < (1e-8 if cfg['rate_noise']==0 else 1e-9)
    decay=np.exp(-dt/cfg['memory']);weight=cfg['memory']*(1-decay)
    Rexpected=decay*R[:-1]+weight*(W[:-1,:,None]*W[:-1,None,:])
    R0expected=decay*R0[:-1]+weight*(v[:-1,:,None]*v[:-1,None,:])
    memory_max=max(np.max(abs(R[1:]-Rexpected)),np.max(abs(R0[1:]-R0expected)))
    assert memory_max<1e-12
    coupled=[]
    for n in [0,5,50,500,5000,N-2]:
        for coeff,cov,w,name in [(C,R,W,'K1'),(C0,R0,v,'K0')]:
            q=coeff[n];M=cov[n];size=len(q)
            def rhs(time,state):
                qq=state[:size];MM=state[size:].reshape(size,size)
                return np.r_[-cfg['gain']*(MM@qq-(qq@MM@qq)*qq),
                             (np.outer(w[n],w[n])-MM/cfg['memory']).ravel()]
            sol=solve_ivp(rhs,(0,dt),np.r_[q,M.ravel()],method='DOP853',rtol=1e-12,atol=1e-14)
            assert sol.success
            err=float(np.max(abs(sol.y[:size,-1]-coeff[n+1])))
            coupled.append({'sample':n,'representation':name,'C_max_error':err})
            assert err<1e-8
    rng=np.random.default_rng(s['seed']);q=rng.normal(size=6);q/=np.linalg.norm(q)
    x0=rng.uniform(0.,2*np.pi/cfg['known_phase_scale'])
    noise=rng.normal(0,cfg['rate_noise'],N)
    assert np.array_equal(q,C[0]);assert np.max(abs(v[:,1]-d['evaluator_ydot']-noise))<1e-15
    assert np.max(abs(d['x']-(x0+.15*t+.05*np.sin(.3*t))))<1e-13
    assert np.max(abs(phase-cfg['known_phase_scale']*d['x']))<1e-12
    assert np.min(np.sum(abs(np.diff(C,axis=0)),axis=0))>1e-5
    assert np.max(abs(np.linalg.norm(C,axis=1)-1))<1e-13
    forward_errors={}
    for name,cc,K in [('learned',C[-1],1),('K0',C0[-1],0),('frozen_prior',C[0],1)]:
        HH=np.tile(cc,(len(f['time']),1)) if K==0 else np.column_stack((np.ones(len(f['time'])),np.cos(f['phase']),np.sin(f['phase'])))@cc.reshape(3,2)
        valid=np.abs(HH[:,1])>=cfg['guard'];prediction=np.full(len(valid),np.nan)
        prediction[valid]=-HH[valid,0]/HH[valid,1]*f['xdot'][valid]
        assert np.array_equal(valid,f[name+'_valid'])
        rate_err=float(np.nanmax(abs(prediction-f[name+'_rate'])))
        yp=np.empty(len(valid));yp[0]=f['evaluator_y'][0]
        for k in range(len(valid)-1):yp[k+1]=yp[k]+dt/2*(prediction[k]+prediction[k+1])
        yi=float(np.nanmax(abs(yp-f[name+'_y'])))
        assert max(rate_err,yi)<1e-10
        if np.all(valid):
            assert abs(r2(prediction-f['evaluator_ydot'])-s['forward'][name]['rate_rmse'])<1e-11
            assert abs(r2(yp-f['evaluator_y'])-s['forward'][name]['y_rmse'])<1e-10
        else:assert s['forward'][name]['rate_rmse'] is None and s['forward'][name]['y_rmse'] is None
        forward_errors[name]=dict(invalid=int(np.sum(~valid)),rate_max_reconstruction_error=rate_err,y_max_reconstruction_error=yi)
    p=cfg['plant_only'];xx=d['x'][::113];eps=1e-6
    yy=lambda z:p['offset']+p['a']*z+p['b']/p['omega']*np.sin(p['omega']*z)
    fd=(yy(xx+eps)-yy(xx-eps))/(2*eps)
    analytic=p['a']+p['b']*np.cos(p['omega']*xx)
    finite_difference_max=float(np.max(abs(fd-analytic)));assert finite_difference_max<1e-7
    return dict(seed=s['seed'],run=path.name,hashes_verified=len(manifest),n_observations=N,
                all_memory_transition_max_error=float(memory_max),metric_reconstruction_max_error=metric_max,
                selected_coupled_checks=coupled,forward=forward_errors,truth_finite_difference_max_error=finite_difference_max,
                final_minimum_covariance_eigenvalue=float(np.linalg.eigvalsh(R[-1])[0]),passed=True)

if __name__=='__main__':
    paths=[HERE/'pilot_9001']+[HERE/f'seed_{s}' for s in range(10,20)]+[HERE/'refine_clean_dt02',HERE/'refine_clean_dt01']
    results=[check(p) for p in paths]
    clean=[json.loads((p/'summary.json').read_text()) for p in paths[-2:]]
    refine=dict(coarse_rate_rmse=clean[0]['forward']['learned']['rate_rmse'],fine_rate_rmse=clean[1]['forward']['learned']['rate_rmse'],
                coarse_y_rmse=clean[0]['forward']['learned']['y_rmse'],fine_y_rmse=clean[1]['forward']['learned']['y_rmse'],
                final_C_max_difference=float(np.max(abs(np.array(clean[0]['final_C'])-clean[1]['final_C']))))
    write(HERE/'audit.json',dict(passed=True,runs=results,refinement=refine,source_sha256=sha(Path(__file__))))
    print(json.dumps(dict(passed=True,runs=len(results),refinement=refine),indent=2))
