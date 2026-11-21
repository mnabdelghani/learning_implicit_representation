"""Plot archived compact data, no simulation or refitting. All five ideal seeds."""
from pathlib import Path
import argparse,json,hashlib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
HERE=Path(__file__).resolve().parent
plt.rcParams.update({'font.size':7,'axes.labelsize':7,'axes.titlesize':7,'legend.fontsize':6,
 'xtick.labelsize':6,'ytick.labelsize':6,'lines.linewidth':.9,'pdf.fonttype':42,'ps.fonttype':42})
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    args.output.mkdir(exist_ok=True,parents=True)
    data=np.load(HERE/'data/ideal_plot_data.npz');colors={'training':'#383838','adaptive':'#0072B2','frozen':'#D55E00'}
    titles={'training':'Training','adaptive':'Adaptive','frozen':'Frozen'}
    def clean(ax):
        ax.spines[['top','right']].set_visible(False);ax.grid(alpha=.15);ax.tick_params(pad=1)
    def save(fig,name):
        fig.savefig(args.output/(name+'.pdf'));fig.savefig(args.output/(name+'.png'),dpi=190);plt.close(fig)
    # Site 0 was declared before outcomes. Sign/gauge alignment is evaluator-only.
    fig,axes=plt.subplots(5,1,figsize=(2.45,3.15),sharex=True,layout='constrained')
    for phase in ['training','adaptive']:
        t=data[f'site0_{phase}_time'];sl=np.unique(np.r_[np.arange(0,len(t),5),len(t)-1]);q=data[f'site0_{phase}_q_site0'];F=data[f'site0_{phase}_physical_site0']
        for j,ax in enumerate(axes):
            ax.plot(t[sl],q[sl,j],color=colors[phase],label=titles[phase]);ax.plot(t[sl],F[sl,j],color='#AA3377',ls='--',lw=.8,label='Physical')
    labels=[r'$F_w$',r'$F_{w_s}$',r'$F_{w_{ss}}$',r'$F_u$',r'$F_{w_t}$']
    for j,ax in enumerate(axes):
        clean(ax);ax.set_ylabel(labels[j],labelpad=1);ax.axvline(80,color='.6',ls=':',lw=.6);ax.set_xlim(0,100)
    axes[0].set_title('All five learned entries, site 0',pad=2)
    handles=[plt.Line2D([],[],color=colors['training']),plt.Line2D([],[],color=colors['adaptive']),plt.Line2D([],[],color='#AA3377',ls='--')]
    axes[0].legend(handles,['Train','Adapt','Physical'],loc='upper right',ncol=1,frameon=False,fontsize=5)
    axes[-1].set_xlabel('Time (s)',labelpad=1);save(fig,'burgers_main_partials')
    # Cohort ranges include all five seeds; no smoothing, per-site aggregation first.
    fig,axes=plt.subplots(2,1,figsize=(2.45,3.15),sharex=True,layout='constrained')
    for phase in ['training','adaptive','frozen']:
        t=data[f'732_{phase}_time'];sl=np.unique(np.r_[np.arange(0,len(t),5),len(t)-1])
        for ax,key in zip(axes,['field_mean','energy']):
            values=np.stack([data[f'{s}_{phase}_{key}'] for s in range(732,737)])
            ax.fill_between(t[sl],np.maximum(values.min(axis=0)[sl],1e-16),np.maximum(values.max(axis=0)[sl],1e-16),color=colors[phase],alpha=.14,lw=0)
            ax.semilogy(t[sl],np.maximum(values.mean(axis=0)[sl],1e-16),color=colors[phase],label=titles[phase])
        physical=data[f'732_{phase}_physical_energy']
        axes[1].semilogy(t[sl],np.maximum(physical[sl],1e-16),color=colors[phase],ls=':',lw=.7)
    axes[0].set(title='Whole-field recovery: five seeds',ylabel='Mean projector error');axes[1].set(ylabel='Pre-update energy',xlabel='Time (s)')
    axes[0].legend(loc='upper right',frameon=False,fontsize=5.5)
    for ax in axes:clean(ax);ax.axvline(80,color='.6',ls=':',lw=.6);ax.set_xlim(0,100)
    axes[1].text(.02,.03,'Dotted: physical-row reference',transform=axes[1].transAxes,fontsize=5.3)
    save(fig,'burgers_main_energy')
    fig,axes=plt.subplots(2,1,figsize=(2.45,3.15),sharex=True,layout='constrained')
    for phase in ['adaptive','frozen']:
        t=data[f'732_{phase}_time'];sl=np.unique(np.r_[np.arange(0,len(t),5),len(t)-1]);values=np.stack([np.sqrt(data[f'{s}_{phase}_task_mse']) for s in range(732,737)])
        axes[0].fill_between(t[sl],values.min(axis=0)[sl],values.max(axis=0)[sl],color=colors[phase],alpha=.14,lw=0)
        axes[0].semilogy(t[sl],values.mean(axis=0)[sl],color=colors[phase],label=titles[phase])
        axes[1].plot(t[sl],data[f'site0_{phase}_w_site0'][sl],color=colors[phase],label=titles[phase])
    axes[1].plot(t[sl],data['site0_adaptive_target_site0'][sl],color='#222222',ls='--',label='Reference')
    axes[0].set(title='Same-field distributed readout',ylabel='Spatial RMS task error')
    axes[0].text(.03,.05,'All 5 seeds finish; no guards or clips',transform=axes[0].transAxes,fontsize=5.2)
    axes[1].set(ylabel=r'$w(s=0,t)$',xlabel='Time (s)')
    axes[0].legend(frameon=False,loc='upper right',fontsize=5.5);axes[1].legend(frameon=False,loc='lower right',fontsize=5.5)
    for ax in axes:clean(ax);ax.set_xlim(80,100)
    save(fig,'burgers_main_tracking')
    (args.output/'plot_provenance.json').write_text(json.dumps({'data_sha256':hashlib.sha256((HERE/'data/ideal_plot_data.npz').read_bytes()).hexdigest(),
      'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'representative_seed':732,'representative_site':0,
      'cohort_seeds':list(range(732,737)),'lines':'arithmetic mean across all 5 seeds; shaded full min/max range',
      'site0':'unit-normalized rows with evaluator-only sign alignment to physical normal','render_stride':5,'sample_dt':.005,
      'smoothing':False,'metric_sampling':'all applied samples; excludes t=100 terminal commands','log_display_floor':1e-16},indent=2)+'\n')
if __name__=='__main__':main()
