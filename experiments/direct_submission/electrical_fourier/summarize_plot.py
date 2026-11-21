"""Plot the prespecified electrical cohort from immutable saved records."""
from pathlib import Path
import argparse,hashlib,json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def load(p):return dict(np.load(p))
def stats(values):
    a=np.asarray(values,float)
    return dict(mean=float(np.mean(a)),sample_sd=float(np.std(a,ddof=1)),min=float(np.min(a)),max=float(np.max(a)))
def causal(a,n):
    cumulative=np.r_[0.,np.cumsum(a)]
    lo=np.maximum(0,np.arange(len(a))+1-n)
    return (cumulative[1:]-cumulative[lo])/(np.arange(len(a))+1-lo)

def main(records,output):
    output.mkdir(exist_ok=False)
    seeds=list(range(10,20));data=[];inputs={};rows=[]
    for seed in seeds:
        case=records/f'seed_{seed}'
        summary=json.loads((case/'summary.json').read_text())
        tr=load(case/'training.npz');ev=load(case/'training_evaluator.npz')
        cal=json.loads((case/'period_estimate.json').read_text())
        late=(tr['time_relative_s']>=100)&(tr['time_relative_s']<120)
        row=dict(seed=seed,late_projector_mean=float(np.mean(ev['projector_error_fourier'][late])),late_K0_projector_mean=float(np.mean(ev['projector_error_K0'][late])),late_energy_mean=float(np.mean(tr['energy_fourier_pre'][late])),frequency_hz=cal['frequency_hz'],tasks=summary['tasks'])
        row.update(prediction_rmse_a=summary['tasks']['fourier']['prediction']['rmse_a'],control_full_rmse_a=summary['tasks']['fourier']['control']['full_rmse_a'],control_late_rmse_a=summary['tasks']['fourier']['control']['late_rmse_a'],K0_control_late_rmse_a=summary['tasks']['K0']['control']['late_rmse_a'],frequency_error_hz=summary['evaluator_frequency_error_hz'],criteria_pass=summary['criteria_pass'],full_cycle=summary['full_cycle'])
        rows.append(row);data.append((tr,ev,summary))
        for p in case.rglob('*'):
            if p.is_file():inputs[str(p.relative_to(records))]=sha(p)
    aggregate=dict(seeds=seeds,illustration_seed=10,statistics={key:stats([r[key] for r in rows]) for key in ['late_projector_mean','late_K0_projector_mean','late_energy_mean','frequency_hz','prediction_rmse_a','control_full_rmse_a','control_late_rmse_a','K0_control_late_rmse_a','frequency_error_hz']},rows=rows)
    (output/'aggregate.json').write_text(json.dumps(aggregate,indent=2,allow_nan=False)+'\n')
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':7.2,'axes.titlesize':8,'axes.labelsize':7.3,'legend.fontsize':6.1,'xtick.labelsize':6.6,'ytick.labelsize':6.6,'pdf.fonttype':42,'axes.spines.top':False,'axes.spines.right':False,'lines.linewidth':1.0})
    fig,axs=plt.subplots(2,3,figsize=(7.3,4.25),layout='constrained')
    blue='#1468a3';orange='#d05d18';green='#258265';purple='#8e4ba8'
    tr,ev,s=data[0];t=tr['time_relative_s'];step=float(np.median(np.diff(t)));smooth=round(1/step)
    ax=axs[0,0]
    ax.semilogy(t,np.maximum(causal(tr['energy_fourier_pre'],smooth),1e-16),color=blue,label='Fourier')
    ax.semilogy(t,np.maximum(causal(tr['energy_K0_pre'],smooth),1e-16),color=orange,label='$K=0$')
    ax.set(title='(a) Fit to observed rates',xlabel='Learning time (s)',ylabel='Pre-update energy');ax.legend(loc='lower left',frameon=False)
    ax=axs[0,1]
    for field,color,label in [('projector_error_fourier',blue,'Fourier'),('projector_error_K0',orange,'$K=0$')]:
        values=np.stack([d[1][field] for d in data]);ax.fill_between(t,np.min(values,axis=0),np.max(values,axis=0),color=color,alpha=.18,lw=0);ax.semilogy(t,np.median(values,axis=0),color=color,label=label)
    ax.set(title='(b) Learned relation accuracy',xlabel='Learning time (s)',ylabel='Row-projector error');ax.legend(frameon=False,loc='lower left')
    ax=axs[0,2]
    freq=rows[0]['frequency_hz'];time=tr['time_s'];phase=2*np.pi*freq*time
    h=np.column_stack((np.ones(len(time)),np.cos(phase),np.sin(phase),np.cos(2*phase),np.sin(2*phase)))
    estimate=np.einsum('ni,nij->nj',h,tr['C_pre'].reshape(-1,5,4));truth=ev['physical_differential']*np.array([20,2,5,1])
    estimate/=np.linalg.norm(estimate,axis=1)[:,None];truth/=np.linalg.norm(truth,axis=1)[:,None];estimate*=np.where(np.sum(estimate*truth,axis=1)>=0,1.,-1.)[:,None]
    keep=t>=119
    for j,(color,label) in enumerate(zip([blue,orange,green,purple],['$F_i$','$F_q$','$F_\\Phi$','$F_t$'])):
        ax.plot(t[keep]-119,estimate[keep,j],color=color,label=label);ax.plot(t[keep]-119,truth[keep,j],color=color,linestyle=':',linewidth=1.8,alpha=.65)
    ax.set(title='(c) All four learned entries',xlabel='Last learning second (s)',ylabel='Unit/sign-aligned entry');ax.legend(ncols=4,loc='lower center',frameon=False,columnspacing=.6,handlelength=1)
    ax=axs[1,0];keep=t>=116
    ax.plot(t[keep]-116,tr['observed_rates'][keep,2],color=blue,label='Applied voltage')
    ax.plot(t[keep]-116,ev['physical_differential'][keep,3],color=purple,label='Clock partial $F_t$')
    ax.set(title='(d) Slow input, fast clock partial',xlabel='Last four learning seconds (s)',ylabel='Voltage (V)');ax.legend(frameon=False,loc='lower left')
    case=records/'seed_10';pred=load(case/'prediction_fourier.npz');pe=load(case/'prediction_evaluator.npz');ctrl=load(case/'control_fourier.npz');ctrl0=load(case/'control_K0.npz')
    # Saved keys are explicit; do not substitute or fabricate a failed trajectory.
    ax=axs[1,1];tt=pred['time_s']-pred['time_s'][0]
    ax.plot(pe['time_s']-pe['time_s'][0],pe['current_a'],color='0.45',linewidth=1.7,label='Actual current');ax.plot(tt,pred['predicted_current_a'],color=blue,linewidth=.9,label='Prediction')
    ax.set(title='(e) Independent current prediction',xlabel='Prediction time (s)',ylabel='Current (A)');ax.legend(frameon=False,loc='lower left')
    ax=axs[1,2];tc=ctrl['time_s']-ctrl['time_s'][0]
    ax.plot(tc,ctrl['reference_current_a'],color='0.2',linestyle='--',label='Reference');ax.plot(tc,ctrl['current_a'],color=blue,label='Fourier');ax.plot(ctrl0['time_s']-ctrl0['time_s'][0],ctrl0['current_a'],color=orange,alpha=.75,label='$K=0$')
    ax.set(title='(f) Current-reference control',xlabel='Control time (s)',ylabel='Current (A)');ax.legend(frameon=False,loc='lower left')
    for ax in axs.flat:ax.grid(alpha=.15);ax.margins(x=.01)
    for suffix in ['pdf','png']:fig.savefig(output/f'direct_fourier_summary.{suffix}',dpi=180)
    plt.close(fig)
    manifest=dict(input_sha256=inputs,script_sha256=sha(Path(__file__)),seed_selection='Seeds 10–19, illustration seed 10, fixed in protocol before confirmation',panels={'a':'seed10 pre-update energy; causal trailing1s mean, startup shortened; log display floor1e-16','b':'median/full range of all ten seeds at every pre-update time, unsmoothed','c':'all4 scaled physical entries, unit/sign aligned evaluator-only; final1s; dotted truth','d':'actual applied training voltage and evaluator clock partial, both volts; final4s','e':'independent prediction from one initial current, no subsequent assimilation','f':'frozen learned and K0 fields on independent own-trajectory feedback branches; complete transient shown'},outputs={p.name:sha(p) for p in output.iterdir() if p.is_file()})
    (output/'figure_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--records',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();main(a.records,a.output)
