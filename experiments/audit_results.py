"""Independent checks against stored trajectories, not manuscript prose."""
from pathlib import Path
import json
import numpy as np
ROOT=Path(__file__).resolve().parents[1]

def read(path):
    if path.exists():return np.genfromtxt(path,names=True,delimiter=',')
    return np.load(path.with_suffix('.npz'))['data']

def close(a,b,label,atol=1e-11):
    if not np.allclose(a,b,rtol=1e-9,atol=atol,equal_nan=True):
        raise AssertionError(f'{label}: disagreement {np.max(np.abs(np.asarray(a)-np.asarray(b)))}')

def main():
    checked=[];max_partial_disagreement=0.
    for ex,methods in [('contact',['learned','frozen','oracle']),('fourier',['learned','K0','frozen','oracle']),('burgers',['learned','frozen','oracle'])]:
        for seed in range(10,20):
            for method in methods:
                folder=ROOT/'results'/ex/f'seed_{seed}'/method
                tr=read(folder/'trace.csv');summary=json.loads((folder/'summary.json').read_text())['metrics']
                lo,hi={'contact':(100.,120.),'fourier':(60.,80.),'burgers':(18.,24.)}[ex]
                sel=(tr['time']>=lo)&(tr['time']<hi)
                if ex=='contact':
                    m=summary['recovered'];close(np.sqrt(np.mean(tr['force_error'][sel]**2)),m['force_rmse'],folder)
                    close(np.sqrt(np.mean(tr['x_error'][sel]**2)),m['motion_rmse'],folder)
                    close(np.mean(tr['preupdate_energy'][sel]),m['incompatibility_mean'],folder)
                    f=np.load(folder/'partials.npz');scalez=np.array([.5,.05,5.,.3,.3,10.,.5,.5]);scaler=np.array([.1,1/.3,1/.3])
                    A=f['F_true']*scalez*scaler[None,:,None];B=f['F_est']*scalez*scaler[None,:,None]
                    err=np.linalg.norm(B-A,axis=(1,2))/np.linalg.norm(A,axis=(1,2))
                    close(err,tr['F_error'][::10],str(folder)+' raw partial error')
                    close(np.mean(tr['F_error'][sel]),m['F_error_mean'],folder)
                    if np.min(tr['indentation'])<=0:raise AssertionError('Contact lost')
                elif ex=='fourier':
                    close(np.sqrt(np.mean(tr['error'][sel]**2)),summary['tracking_rmse'],folder)
                    close(np.mean(tr['preupdate_energy'][sel]),summary['energy_mean'],folder)
                    t=tr['time'];h=np.column_stack((np.ones(len(t)),np.cos(4*t),np.sin(4*t)))
                    dh=np.column_stack((np.zeros(len(t)),-np.sin(4*t),np.cos(4*t)))
                    c=np.column_stack([tr[f'c{i}'] for i in range(6)])
                    rho=h@np.array([1.2,.27,-.18]);kappa=h@np.array([1.5,.34,.24]);rate=(tr['control']-kappa*tr['state'])/rho
                    phtrue=(dh@np.array([1.2,.27,-.18]))*rate+(dh@np.array([1.5,.34,.24]))*tr['state']
                    phest=np.sum(dh*c[:,:3],axis=1)*rate+np.sum(dh*c[:,3:],axis=1)*tr['state']
                    close(phtrue,tr['phase_partial_true'],folder);close(phest,tr['phase_partial_est'],folder)
                    F=np.column_stack((kappa,rho,-np.ones(len(t)),phtrue))
                    G=np.column_stack((np.sum(h*c[:,3:],axis=1),np.sum(h*c[:,:3],axis=1),-np.ones(len(t)),phest))
                    scales=np.array([.5,1,1,1]);err=np.linalg.norm((G-F)*scales,axis=1)/np.linalg.norm(F*scales,axis=1)
                    close(err,tr['F_error'],folder)
                    close(np.mean(err[sel]),summary['F_error_mean'],folder)
                else:
                    close(np.sqrt(np.mean(tr['tracking_rms'][sel]**2)),summary['tracking_rms'],folder)
                    close(np.mean(tr['sample_energy_noisy_preupdate'][sel]),summary['sample_energy_noisy_mean'],folder)
                    close(np.mean(tr['F_relative_error_preupdate'][sel]),summary['F_relative_error_mean'],folder)
                    f=np.load(folder/'partials.npz');A=f['F_true'];B=f['F_est']
                    err=np.linalg.norm(B-A,axis=(1,2))/np.linalg.norm(A,axis=(1,2))
                    ids=np.rint(f['time']/.005).astype(int)
                    close(err,tr['F_relative_error_preupdate'][ids],folder)
                checked.append(dict(study=ex,seed=seed,method=method,verified=True))
    out=dict(checked_runs=len(checked),status='PASS',checks=['Half-open evaluation windows and task RMS recomputed from recorded traces',
        'Incompatibility means independently recomputed', 'Raw saved contact/Burgers differentials reproduce trace errors',
        'Fourier full clock column independently reconstructed from saved coefficients and physical context',
        'All declared seeds present; no contact loss in any recorded branch'],runs=checked)
    (ROOT/'docs/result_audit.json').write_text(json.dumps(out,indent=2)+'\n')
    print(f'Independent recorded-data audit: {len(checked)} / 100 runs passed.')
if __name__=='__main__':main()
