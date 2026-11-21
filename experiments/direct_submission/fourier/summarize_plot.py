from pathlib import Path
import json,hashlib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
HERE=Path(__file__).resolve().parent
S=[json.loads((HERE/f'seed_{i}'/'summary.json').read_text()) for i in range(10,20)]
D=[np.load(HERE/f'seed_{i}'/'training.npz') for i in range(10,20)]
A=json.loads((HERE/'audit.json').read_text())
def stats(values):
    a=np.asarray(values,float)
    return dict(mean=float(a.mean()),sample_sd=float(a.std(ddof=1)),minimum=float(a.min()),maximum=float(a.max()),values=a.tolist())
def proj(H,F):
    P=H[:,:,None]*H[:,None,:]/np.sum(H*H,axis=1)[:,None,None]
    Q=F[:,:,None]*F[:,None,:]/np.sum(F*F,axis=1)[:,None,None]
    return np.linalg.norm(P-Q,axis=(1,2))
rows=[];errors=[];errors0=[]
for s,d in zip(S,D):
    h=np.column_stack((np.ones(len(d['time'])),np.cos(d['phase']),np.sin(d['phase'])))
    H=np.einsum('ni,nij->nj',h,d['C_pre'].reshape(-1,3,2))
    F=np.column_stack((-d['evaluator_ydot']/d['observed_v'][:,0],np.ones(len(h))))
    e=proj(H,F);e0=proj(d['C0_pre'],F);errors.append(e);errors0.append(e0)
    late=(d['time']>=200)&(d['time']<240)
    rows.append(dict(seed=s['seed'],late_projector_mean_stable=float(e[late].mean()),late_K0_projector_mean_stable=float(e0[late].mean()),
                     qualifies=s['qualifies'],criteria_checks=s['criteria_checks'],forward=s['forward'],final_Hy_bound=s['final_all_phase_Hy_lower_bound']))
aggregate=dict(identity=S[0]['identity'],seeds=list(range(10,20)),number_of_seeds=10,qualified=sum(x['qualifies'] for x in S),
               late_projector=stats([r['late_projector_mean_stable'] for r in rows]),late_K0_projector=stats([r['late_K0_projector_mean_stable'] for r in rows]),
               late_preupdate_energy=stats([s['late_energy_mean'] for s in S]),late_K0_preupdate_energy=stats([s['late_K0_energy_mean'] for s in S]),
               rate_rmse=stats([s['forward']['learned']['rate_rmse'] for s in S]),y_rmse=stats([s['forward']['learned']['y_rmse'] for s in S]),
               K0_rate_rmse=stats([s['forward']['K0']['rate_rmse'] for s in S]),K0_y_rmse=stats([s['forward']['K0']['y_rmse'] for s in S]),
               learned_heldout_invalid_queries_total=sum(s['forward']['learned']['invalid_queries'] for s in S),
               frozen_prior_invalid_seeds=sum(s['forward']['frozen_prior']['invalid_queries']>0 for s in S),
               frozen_prior_invalid_queries_total=sum(s['forward']['frozen_prior']['invalid_queries'] for s in S),
               minimum_final_all_phase_Hy_bound=min(s['final_all_phase_Hy_lower_bound'] for s in S),
               maximum_normalization_error=max(s['max_norm_error'] for s in S),
               clean_covariance_gap=stats([s['clean_coefficient_gap'] for s in S]),
               observed_covariance_gap=stats([s['coefficient_gap'] for s in S]),
               clean_target_bias_max=max(s['clean_target_memory_bias'] for s in S),
               time_scales=S[0]['time_scales'],runs=rows,audit_passed=A['passed'],refinement=A['refinement'],
               pilot_excluded=True,all_failed_queries_retained=True)
(HERE/'aggregate.json').write_text(json.dumps(aggregate,indent=2)+'\n')
plt.rcParams.update({'font.size':8,'axes.labelsize':8,'axes.titlesize':8,'legend.fontsize':6.5,'xtick.labelsize':7,'ytick.labelsize':7,'pdf.fonttype':42,'ps.fonttype':42,'lines.linewidth':1.1,'axes.spines.top':False,'axes.spines.right':False})
d=D[0];s=S[0];f=np.load(HERE/'seed_10/heldout.npz');t=d['time']
h=np.column_stack((np.ones(len(t)),np.cos(d['phase']),np.sin(d['phase'])))
H=np.einsum('ni,nij->nj',h,d['C_pre'].reshape(-1,3,2));F=np.column_stack((-d['evaluator_ydot']/d['observed_v'][:,0],np.ones(len(t))))
blue='#087eaf';orange='#d56524';green='#3c9160';gray='#575e67'
figdir=HERE/'figures';figdir.mkdir(exist_ok=False)
def smooth(z):
    return np.convolve(z,np.ones(50)/50,mode='valid')
def energy(ax):
    v=d['observed_v'];prior=h@d['C_pre'][0].reshape(3,2)
    for z,label,color in [(H,'Direct K=1',blue),(d['C0_pre'],'Direct K=0',orange),(prior,'Frozen prior',gray)]:
        ax.semilogy(t[49::10],np.maximum(smooth(.5*np.sum(z*v,axis=1)**2)[::10],1e-12),label=label,color=color)
    ax.set(xlabel='Training time (s)',ylabel='Pre-update energy',title='(a) Observed incompatibility');ax.legend(frameon=False,loc='upper right')
def recovery(ax):
    for E,label,color in [(np.array(errors),'K=1',blue),(np.array(errors0),'K=0',orange)]:
        ax.fill_between(t[::20],E.min(0)[::20],E.max(0)[::20],color=color,alpha=.12,lw=0)
        ax.semilogy(t[::20],np.median(E,axis=0)[::20],label=label,color=color)
    ax.set(xlabel='Training time (s)',ylabel='Row-projector error',title='(b) Recovery: ten-seed range');ax.legend(frameon=False)
def partials(ax):
    sel=t>=234;hh=H[sel]/np.linalg.norm(H[sel],axis=1)[:,None];hh*=np.sign(hh[:,1])[:,None]
    ff=F[sel]/np.linalg.norm(F[sel],axis=1)[:,None]
    ax.plot(t[sel],ff[:,0],color=gray,lw=2,label='True normalized entries');ax.plot(t[sel],ff[:,1],color=gray,lw=2)
    ax.plot(t[sel],hh[:,0],color=blue,ls='--',label='Learned x entry');ax.plot(t[sel],hh[:,1],color=green,ls='--',label='Learned y entry')
    ax.set(xlabel='Training time (s)',ylabel='Unit-row entries',title='(c) Independently learned partials');ax.legend(frameon=False,loc='center',ncol=1)
def timescales(ax):
    sel=(t>=220)&(t<=240)
    ax.plot(t[sel],(d['observed_v'][sel,0]-.15)/.015,color=orange,label='Actual input rate, scaled')
    ax.plot(t[sel],(-F[sel,0]-1.2)/.4,color=blue,label='Physical partial, scaled')
    ax.set(xlabel='Training time (s)',ylabel='Centered / amplitude',title='(d) Fast partial, slow supplied rate');ax.legend(frameon=False,loc='upper left')
def prediction(ax):
    tt=f['time'];y0=f['evaluator_y'][0]
    ax.plot(tt,f['evaluator_y']-y0,color=gray,lw=2,label='Plant output')
    ax.plot(tt,f['learned_y']-y0,color=blue,ls='--',label='K=1 prediction')
    ax.plot(tt,f['K0_y']-y0,color=orange,ls=':',label='K=0 prediction')
    ax.set(xlabel='New-input episode (s)',ylabel='Output change',title='(e) Independent integrated readout');ax.legend(frameon=False)
def prediction_error(ax):
    for name,label,color in [('learned','K=1',blue),('K0','K=0',orange)]:
        ax.semilogy(f['time'][1:],np.maximum(abs(f[name+'_y'][1:]-f['evaluator_y'][1:]),1e-9),label=label,color=color)
    ax.set(xlabel='New-input episode (s)',ylabel='Absolute output error',title='(f) No output assimilation');ax.legend(frameon=False)
plots=[('fourier_energy',energy),('fourier_F_error',recovery),('fourier_partials',partials),('fourier_timescales',timescales),('fourier_prediction',prediction),('fourier_prediction_error',prediction_error)]
fig,axes=plt.subplots(2,3,figsize=(7.3,4.25),layout='constrained')
for ax,(_,func) in zip(axes.flat,plots):func(ax);ax.grid(alpha=.14)
fig.savefig(figdir/'fourier_direct_summary.pdf');fig.savefig(figdir/'fourier_direct_summary.png',dpi=220);plt.close(fig)
for name,func in plots:
    fig,ax=plt.subplots(figsize=(2.45,1.95),layout='constrained');func(ax);ax.grid(alpha=.14)
    fig.savefig(figdir/(name+'.pdf'));fig.savefig(figdir/(name+'.png'),dpi=200);plt.close(fig)
(HERE/'figure_manifest.json').write_text(json.dumps({p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(figdir.iterdir())},indent=2)+'\n')
print(json.dumps({key:aggregate[key] for key in ['qualified','late_projector','rate_rmse','y_rmse','K0_rate_rmse','K0_y_rmse','refinement']},indent=2))
