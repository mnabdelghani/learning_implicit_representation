"""Regenerate individual data charts. LaTeX assembles them into multi-panel figures.
No synthetic trajectories, hand-edited results, custom color assignments, or
multi-axis subplots are used. Figures use seed 10, chosen before execution.
"""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'supplement'/'figures';OUT.mkdir(parents=True,exist_ok=True)

def load(path):
    path=Path(path)
    if path.exists():return np.genfromtxt(path,names=True,delimiter=',')
    packed=np.load(path.with_suffix('.npz'))
    return packed['data']

def rolling(y,n):
    y=np.asarray(y,float);valid=np.isfinite(y)
    s=np.r_[0.,np.cumsum(np.where(valid,y,0.))];v=np.r_[0,np.cumsum(valid)]
    end=np.arange(1,len(y)+1);start=np.maximum(0,end-n);count=v[end]-v[start]
    return np.divide(s[end]-s[start],count,out=np.full(len(y),np.nan),where=count>0)

def canvas(title,ylabel):
    fig=plt.figure(figsize=(2.30,1.65));ax=fig.add_subplot(111)
    ax.set_title(title,fontsize=8,loc='left',pad=3)
    ax.set_xlabel('Time (s)',fontsize=7);ax.set_ylabel(ylabel,fontsize=7,labelpad=2)
    ax.tick_params(labelsize=6.5,pad=2);ax.xaxis.set_major_locator(MaxNLocator(4))
    ax.grid(alpha=.18,linewidth=.4)
    return fig,ax

def finish(fig,ax,name,change=None,legend=True,ncol=1):
    if change is not None:ax.axvline(change,ls='--',linewidth=.6)
    if legend:ax.legend(fontsize=6,frameon=False,ncol=ncol,handlelength=1.8,labelspacing=.18,borderpad=.1)
    fig.tight_layout(pad=.35)
    fig.savefig(OUT/(name+'.pdf'))
    fig.savefig(OUT/(name+'.png'),dpi=180)
    plt.close(fig)

def line(ax,t,y,label,style='-',log=False,stride=1):
    if log:ax.semilogy(t[::stride],np.maximum(y[::stride],1e-18),style,lw=.8,label=label)
    else:ax.plot(t[::stride],y[::stride],style,lw=.8,label=label)

def contact_plots():
    a=load(ROOT/'results/contact/seed_10/learned/trace.csv');t=a['time']
    for j,key,label in [('a','Fx',r'$f_x$'),('b','Fvx',r'$f_{v_x}$')]:
        fig,ax=canvas(f'({j}) Evaluated nonlinear partial',label)
        line(ax,t,a[key+'_true'],'True','--',stride=3);line(ax,t,a[key+'_est'],'Estimated',stride=3)
        finish(fig,ax,'contact_partial_'+('x' if key=='Fx' else 'vx'),60.)
    fig,ax=canvas('(c) Full differential error','Error')
    line(ax,t,a['F_error'],r'$e_F$',log=True,stride=3)
    line(ax,t,a['projector_error'],'Row-projector distance',log=True,stride=3)
    finish(fig,ax,'contact_F_error',60.)
    fig,ax=canvas('(d) Pre-update incompatibility',r'Energy ($N^2/s^2$)')
    for key,label in [('preupdate_energy','Measured'),('same_stream_oracle_energy','True-model reference'),('clean_energy','Clean diagnostic')]:
        line(ax,t,rolling(a[key],100),label,log=True,stride=4)
    finish(fig,ax,'contact_energy',60.)
    fig,ax=canvas('(e) Normal-force task','Force (N)')
    line(ax,t,a['desired_force'],'Desired','--',stride=3);line(ax,t,a['normal_force'],'Actual',stride=3)
    finish(fig,ax,'contact_task',60.)
    fig,ax=canvas('(f) Applied commands','Command')
    for key in ['u1','u2']:line(ax,t,a[key],key,stride=3)
    finish(fig,ax,'contact_control',60.)
    fig,ax=canvas('Contact: motion task','Position (m)')
    line(ax,t,a['desired_x'],'Desired','--',stride=3);line(ax,t,a['x'],'Actual',stride=3)
    finish(fig,ax,'contact_motion',60.)
    fig,ax=canvas('Contact: compatibility and activity','Normalized quantity')
    line(ax,t,rolling(a['normalized_energy'],100),'Normalized energy',log=True,stride=4)
    line(ax,t,rolling(a['feature_power'],100),'Feature power',log=True,stride=4)
    finish(fig,ax,'contact_activity',60.)
    fig,ax=canvas('Contact: retained excitation','Smallest free eigenvalue')
    line(ax,t,a['information_min_eigenvalue'],'Measured information',log=True,stride=4)
    finish(fig,ax,'contact_information',60.)

def fourier_plots():
    a=load(ROOT/'results/fourier/seed_10/learned/trace.csv');t=a['time'];mask=t<=16
    for name,title,ylabel,key in [('rate','(a) Rate partial: initial 16 s',r'$f_{\dot{x}}=\rho$','rho'),
        ('phase','(b) Clock partial: initial 16 s',r'$f_\phi$','phase_partial')]:
        fig,ax=canvas(title,ylabel)
        line(ax,t[mask],a[key+'_true'][mask],'True','--');line(ax,t[mask],a[key+'_est'][mask],'Estimated')
        finish(fig,ax,'fourier_partial_'+name)
    fig,ax=canvas('(c) Differential vs. memory','Relative error')
    line(ax,t,a['F_error'],r'$e_F$',log=True,stride=3)
    line(ax,t,a['coefficient_relative_error'],'Stored coefficients',log=True,stride=3)
    finish(fig,ax,'fourier_F_error')
    fig,ax=canvas('(d) Same-stream energy','Energy')
    for key,label in [('preupdate_energy',r'$K=1$'),('K0_same_stream_energy',r'$K=0$ shadow'),('same_stream_oracle_energy','True-model reference')]:
        line(ax,t,rolling(a[key],100),label,log=True,stride=3)
    finish(fig,ax,'fourier_energy')
    fig,ax=canvas('(e) State tracking','State')
    line(ax,t,a['desired'],'Desired','--',stride=3);line(ax,t,a['state'],'Actual',stride=3)
    finish(fig,ax,'fourier_task')
    fig,ax=canvas('(f) Applied input','Input')
    line(ax,t,a['control'],'Learned readout',stride=3);finish(fig,ax,'fourier_control')
    fig,ax=canvas('Fourier: rate partial, complete run',r'$f_{\dot{x}}$')
    line(ax,t,a['rho_true'],'True','--',stride=3);line(ax,t,a['rho_est'],'Estimated',stride=3)
    finish(fig,ax,'fourier_full_partial')
    fig,ax=canvas('Fourier: activity and compatibility','Normalized quantity')
    line(ax,t,rolling(a['feature_power'],100),'Feature power',log=True,stride=3)
    line(ax,t,rolling(a['normalized_energy'],100),'Normalized energy',log=True,stride=3)
    finish(fig,ax,'fourier_activity')
    fig,ax=canvas('Fourier: retained excitation','Smallest free eigenvalue')
    line(ax,t,a['information_min_eigenvalue'],'Measured information',log=True,stride=3)
    finish(fig,ax,'fourier_information')

def burgers_plots():
    base=ROOT/'results/burgers/seed_10';a=load(base/'learned/trace.csv');t=a['time']
    p=np.load(base/'learned/partials.npz');pt=p['time'];site=16
    fig,ax=canvas(r'(a) Partials at $s=\pi/2$','Evaluated partial')
    for j,key in [(0,r'$f_w$'),(1,r'$f_{w_s}$')]:
        line(ax,pt,p['F_true'][:,site,j],key+' true','--')
        line(ax,pt,p['F_est'][:,site,j],key+' estimated')
    finish(fig,ax,'burgers_partials',ncol=2)
    fig,ax=canvas('(b) Distributed differential error','Error')
    line(ax,t,a['F_relative_error_preupdate'],r'$e_F$',log=True,stride=3)
    line(ax,pt,p['projector_error'],'RMS projector distance',log=True)
    finish(fig,ax,'burgers_F_error')
    fig,ax=canvas('(c) Pre-update incompatibility','Energy')
    line(ax,t,rolling(a['sample_energy_noisy_preupdate'],50),'Measured',log=True,stride=4)
    line(ax,t,rolling(a['sample_energy_clean_preupdate'],50),'Clean diagnostic',log=True,stride=4)
    ax.axhline(.0016,ls='--',lw=.7,label='True-model expectation')
    finish(fig,ax,'burgers_energy')
    f=load(base/'learned/fields.csv');svals=np.unique(f['s']);sel=np.isclose(f['s'],svals[site])
    fig,ax=canvas(r'(d) Field task at $s=\pi/2$','Field value')
    line(ax,f['time'][sel],f['desired'][sel],'Desired','--');line(ax,f['time'][sel],f['w'][sel],'Actual')
    finish(fig,ax,'burgers_task')
    fig,ax=canvas('(e) Distributed task error','Spatial RMS')
    for method,label in [('learned','Learned'),('frozen','Frozen'),('oracle','Oracle')]:
        b=load(base/method/'trace.csv');line(ax,b['time'],b['tracking_rms'],label,log=True,stride=3)
    finish(fig,ax,'burgers_tracking')
    fig,ax=canvas('(f) Distributed actuation','Spatial RMS input')
    line(ax,t,a['control_rms'],'Learned readout',stride=3);finish(fig,ax,'burgers_control')
    fig,ax=canvas('Burgers: retained excitation','Smallest free eigenvalue')
    line(ax,t,a['history_min_eigenvalue_scaled'],'Measured information',log=True,stride=3)
    finish(fig,ax,'burgers_information')

def burgers_compact():
    base=ROOT/'results/burgers/seed_10';a=load(base/'learned/trace.csv');t=a['time']
    p=np.load(base/'learned/partials.npz');pt=p['time'];site=16
    fig,ax=canvas(r'(a) Partials at $s=\pi/2$','Evaluated partial')
    for j,key in [(0,r'$f_w$'),(1,r'$f_{w_s}$')]:
        line(ax,pt,p['F_true'][:,site,j],key+' true','--')
        line(ax,pt,p['F_est'][:,site,j],key+' estimated')
    finish(fig,ax,'burgers_main_partials',ncol=2)
    fig,ax=canvas('(b) Pre-update incompatibility','Energy')
    line(ax,t,rolling(a['sample_energy_noisy_preupdate'],50),'Measured',log=True,stride=4)
    line(ax,t,rolling(a['sample_energy_clean_preupdate'],50),'Clean diagnostic',log=True,stride=4)
    ax.axhline(.0016,ls='--',lw=.7,label='True-model expectation')
    finish(fig,ax,'burgers_main_energy')
    fig,ax=canvas('(c) Distributed task error','Spatial RMS')
    for method,label in [('learned','Learned'),('frozen','Frozen'),('oracle','Oracle')]:
        b=load(base/method/'trace.csv');line(ax,b['time'],b['tracking_rms'],label,log=True,stride=3)
    finish(fig,ax,'burgers_main_tracking')

def main():
    contact_plots();fourier_plots();burgers_plots();burgers_compact()
    print(f'Generated {len(list(OUT.glob("*.pdf")))} data charts in {OUT}')
if __name__=='__main__':main()
