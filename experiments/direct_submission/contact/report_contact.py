"""Plot only saved records; no learning, filtering for selection, or simulations."""
from pathlib import Path
import argparse,json,hashlib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--records',type=Path,default=Path(__file__).resolve().parent);args=parser.parse_args();HERE=args.records;OUT=HERE/'figures';OUT.mkdir(exist_ok=True)
plt.rcParams.update({'font.size':10.2,'axes.labelsize':10.2,'legend.fontsize':9.,'xtick.labelsize':9.,'ytick.labelsize':9.,'pdf.fonttype':42,'ps.fonttype':42,'axes.spines.top':False,'axes.spines.right':False,'lines.linewidth':1.15})
raw=np.load(HERE/'confirm_seed101/states.npz');tr=np.genfromtxt(HERE/'confirm_seed101/trace.csv',delimiter=',',names=True);t=tr['time'];q=raw['q_pre'];D=raw['scales'];F=raw['F_evaluator'];A=F*D
# Physical orthonormal frame via symmetric polar map (evaluator only).
values,vectors=np.linalg.eigh(A@A.transpose(0,2,1));W=(vectors*(1/np.sqrt(values))[:,None,:])@vectors.transpose(0,2,1);truth=W@A
U,sv,Vh=np.linalg.svd(truth@q.transpose(0,2,1));O=U@Vh;aligned=O@q
assert np.max(np.linalg.norm(aligned@aligned.transpose(0,2,1)-np.eye(3),axis=(1,2)))<1e-10
mask=t>=40;thin=np.arange(0,len(t),5)
colors=['#006bb6','#d34b28','#138a55']
def decorate(ax,xlo=0):
 ax.axvline(40,color='.5',lw=.8,ls=':');ax.axvline(60,color='.35',lw=.8,ls='--');ax.set_xlim(xlo,120);ax.set_xlabel('Time (s)');ax.grid(alpha=.15)
def save(fig,name):
 fig.tight_layout(pad=.4);fig.savefig(OUT/(name+'.pdf'));fig.savefig(OUT/(name+'.png'),dpi=180);plt.close(fig)
fig,axs=plt.subplots(2,1,figsize=(3.2,2.35),sharex=True)
for ax,j,label in zip(axs,[0,3],[r'$x$ entry',r'$v_x$ entry']):
 ax.plot(t[mask][::5],truth[mask,0,j][::5],color='k',ls='--',label='Physical');ax.plot(t[mask][::5],aligned[mask,0,j][::5],color=colors[0],label='Aligned');ax.set_ylabel(label);decorate(ax,40)
axs[0].legend(loc='best',ncol=2);axs[0].set_title('(a) Aligned normalized entries',loc='left',fontsize=9.5);axs[0].set_xlabel('');save(fig,'direct_contact_partials')
fig,ax=plt.subplots(figsize=(3.2,2.35));ax.plot(t[thin],tr['projector_error'][thin],color=colors[0],label='Learned');ax.plot(t[thin],tr['frozen_projector_error'][thin],color=colors[1],label='Frozen initial');ax.set_ylabel(r'$\|\widehat P-P\|_F$');ax.set_ylim(0,2.5);ax.legend();ax.set_title('(b) Whole-field row-space error',loc='left',fontsize=9.5);decorate(ax);save(fig,'direct_contact_field')
def mean_causal(v,n=50):
 cs=np.r_[0,np.cumsum(v)];ii=np.arange(len(v));lo=np.maximum(0,ii-n+1);return (cs[ii+1]-cs[lo])/(ii-lo+1)
fig,ax=plt.subplots(figsize=(3.2,2.35))
for name,label,col in [('incompatibility','Learned',colors[0]),('frozen_incompatibility','Frozen initial',colors[1])]:
 y=mean_causal(.5*tr[name]**2);ax.semilogy(t[thin],y[thin],label=label,color=col)
ax.set_ylabel(r'$\frac{1}{2}\|Qv\|^2$');ax.legend();ax.set_title('(c) Pre-update energy (0.5-s mean)',loc='left',fontsize=9.5);decorate(ax);save(fig,'direct_contact_energy')
fig,ax=plt.subplots(figsize=(3.2,2.35));ax.plot(t[thin],tr['lambda'][thin],color=colors[0],label='Force');ax.plot(t[mask][::5],tr['desired_lambda'][mask][::5],color='k',ls='--',label='Reference');ax.set_ylabel(r'Vertical force $\lambda$ (N)');ax.set_title('(d) Learned-field force readout',loc='left',fontsize=9.5);ax.legend();decorate(ax);save(fig,'direct_contact_force')
fig,ax=plt.subplots(figsize=(3.2,2.35));ax.plot(t[thin],tr['x'][thin],color=colors[0],label='Position');ax.plot(t[mask][::5],tr['desired_x'][mask][::5],color='k',ls='--',label='Reference');ax.set_ylabel(r'Horizontal position $x$ (m)');ax.legend();ax.set_title('(e) Learned-field motion readout',loc='left',fontsize=9.5);decorate(ax);save(fig,'direct_contact_motion')
fig,axs=plt.subplots(2,1,figsize=(3.2,2.35),sharex=True)
for ax,key in zip(axs,['u1','u2']):ax.plot(t[thin],tr[key][thin],color=colors[0]);ax.set_ylabel('$u_'+key[1]+'$');decorate(ax)
axs[0].set_xlabel('');axs[0].set_title('(f) Applied inputs; no saturation',loc='left',fontsize=9.5);save(fig,'direct_contact_inputs')
# Full all-entry normalized field, including training, is supplementary evidence.
fig,axs=plt.subplots(6,4,figsize=(7.0,8.1),sharex=True)
labels=['x','y',r'\lambda','v_x','v_y',r'\dot\lambda','u_1','u_2']
for i in range(3):
 for j in range(8):
  ax=axs[2*i+j//4,j%4];ax.plot(t[thin],truth[thin,i,j],color='k',ls='--',lw=.8);ax.plot(t[thin],aligned[thin,i,j],color=colors[0],lw=.9);ax.axvline(40,color='.5',lw=.5,ls=':');ax.axvline(60,color='.4',lw=.5,ls='--');ax.set_title(f'Row {i+1}, $'+labels[j]+'$',fontsize=8);ax.tick_params(labelsize=7.);ax.set_xlim(0,120);ax.set_xticks([0,60,120])
  if i==2 and j>=4:ax.set_xlabel('Time (s)',fontsize=7.5)
fig.suptitle('All 24 adaptive entries in an evaluator-only orthonormal gauge (black: physical; blue: learned)',fontsize=8.5);fig.tight_layout(rect=(0,0,1,.96));fig.savefig(OUT/'direct_contact_all24.pdf');fig.savefig(OUT/'direct_contact_all24.png',dpi=150);plt.close(fig)
# Retain canonical force-rate normalization failure; no quiet gauge repair.
physical_est=q/D;block=physical_est[:,:,[5,6,7]];canonical=np.linalg.solve(block,physical_est);sig=np.linalg.svd(block,compute_uv=False)[:,-1]
fig,axs=plt.subplots(2,1,figsize=(3.2,2.8),sharex=True)
for j,label in [(0,'x'),(3,'v_x')]:axs[0].semilogy(t[mask],np.maximum(np.abs(canonical[mask,0,j]-F[mask,0,j]),1e-10),label=label)
axs[0].set_ylabel('Entry error');axs[0].legend();axs[0].set_title('Canonical normalization is ill-conditioned',fontsize=8.5)
axs[1].semilogy(t[mask],sig[mask]);axs[1].set_ylabel(r'Block $\sigma_{\min}$');decorate(axs[1],40);save(fig,'direct_contact_canonical_gauge_limitation')
truth_sigma=np.linalg.svd(truth[:,:,[4,6,7]],compute_uv=False)[:,-1]
np.savez_compressed(HERE/'display_gauge_diagnostics.npz',time=t,truth_frame=truth,aligned_Q=aligned,orthogonal_alignment=O,canonical_field=canonical,canonical_sigma=sig,physical_control_sigma=truth_sigma)
report=dict(illustration_seed=101,gauge='T=(FDD^T F^T)^(-1/2)FD; O=UV^T from SVD(TQ^T); Q_display=OQ. O is orthogonal and evaluator-only, never learner/controller.',orthogonal_alignment_drift=float(np.max(np.linalg.norm(O@O.transpose(0,2,1)-np.eye(3),axis=(1,2)))),minimum_overlap_singular_value_control=float(sv[mask,-1].min()),physical_control_min_sigma=float(truth_sigma[mask].min()),learned_control_min_sigma=float(tr['readout_sigma'][mask].min()),canonical_min_sigma=float(sig[mask].min()),canonical_limit='Unit force-rate normalization is unstable; physical magnitudes are not recovered precisely. Full canonical diagnostics retained.',metrics={})
summaries=[json.loads((HERE/f'confirm_seed{s}/summary.json').read_text()) for s in [101,102,103,104,105]]
for key in ['x_rmse','lambda_rmse','projector_error','frozen_projector_error','incompatibility','frozen_incompatibility','query_error_rms','frozen_query_error_rms','min_sigma']:
 a=np.array([d['late'][key] for d in summaries]);report['metrics'][key]=dict(mean=float(a.mean()),sd=float(a.std(ddof=1)),min=float(a.min()),max=float(a.max()))
(HERE/'report_metrics.json').write_text(json.dumps(report,indent=2)+'\n')
macros={'DirectContactForce':'lambda_rmse','DirectContactMotion':'x_rmse','DirectContactProjector':'projector_error','DirectContactFrozenProjector':'frozen_projector_error','DirectContactQuery':'query_error_rms','DirectContactFrozenQuery':'frozen_query_error_rms'}
(HERE/'contact_metrics.tex').write_text('\n'.join('\\newcommand{\\'+name+'}{'+f"{report['metrics'][key]['mean']:.4g}"+'}' for name,key in macros.items())+'\n')
(HERE/'figure_manifest.json').write_text(json.dumps({p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in OUT.iterdir() if p.is_file()},indent=2)+'\n')
print(json.dumps(report,indent=2))

import subprocess
names=['partials','field','energy','force','motion','inputs']
rows=[' & '.join(r'\includegraphics[width=3.2in]{direct_contact_'+name+'.pdf}' for name in names[k:k+3]) for k in [0,3]]
tex=r'\documentclass[border=0pt]{standalone}'+r'\usepackage{graphicx}'+r'\begin{document}'+r'\begin{tabular}{@{}c@{}c@{}c@{}}'+((chr(92)*2+chr(10)).join(rows))+r'\end{tabular}'+r'\end{document}'
(OUT/'direct_contact_summary.tex').write_text(tex)
with (OUT/'combine_execution.log').open('w') as log:
    subprocess.run(['pdflatex','-interaction=nonstopmode','-halt-on-error','direct_contact_summary.tex'],cwd=OUT,stdout=log,stderr=subprocess.STDOUT,check=True)
subprocess.run(['pdftoppm','-singlefile','-png','-r','140',str(OUT/'direct_contact_summary.pdf'),str(OUT/'direct_contact_summary')],check=True)
(HERE/'figure_manifest.json').write_text(json.dumps({p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in OUT.iterdir() if p.is_file() and p.suffix in ['.pdf','.png']},indent=2)+'\n')
