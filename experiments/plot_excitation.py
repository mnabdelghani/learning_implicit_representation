"""Replot the declared paired-contact supplementary study from saved records."""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'supplement'/'figures'
OUT.mkdir(parents=True,exist_ok=True)

def read(protocol):
 p=ROOT/'results/contact_excitation/seed_10'/protocol/'trace.csv'
 if p.exists():return np.genfromtxt(p,names=True,delimiter=',')
 return np.load(p.with_suffix('.npz'))['data']

def make():
 a=read('task_only');b=read('bounded_probe')
 for name,key,ylabel,log in [('force','force_error','Absolute force error (N)',True),
 ('partial','F_error','Relative differential error',True),
 ('energy','preupdate_energy','Pre-update incompatibility',True),
 ('control','control_norm','Control norm',False)]:
  fig,ax=plt.subplots(figsize=(3.3,2.1))
  for data,label in [(a,'Task only'),(b,'Bounded probing')]:
   y=np.abs(data[key]) if key=='force_error' else data[key]
   if log:y=np.maximum(y,1e-12)
   ax.plot(data['time'][::8],y[::8],label=label,lw=.9)
  ax.axvline(60,ls='--',lw=.7,color='gray')
  if log:ax.set_yscale('log')
  ax.set_xlabel('Time (s)');ax.set_ylabel(ylabel)
  ax.legend(frameon=False,fontsize=7);ax.grid(alpha=.2)
  fig.tight_layout();fig.savefig(OUT/f'excitation_{name}.pdf');plt.close(fig)
 data=json.loads((ROOT/'results/contact_excitation/aggregate.json').read_text())
 fig,ax=plt.subplots(figsize=(3.3,2.1))
 for label,proto in [('Task only','task_only'),('Bounded probing','bounded_probe')]:
  values=np.maximum(data[proto]['clean_information_eigenvalues_mean'],1e-19)
  ax.semilogy(np.arange(1,7),values,'o-',label=label,lw=.9)
 ax.set_xlabel('Clean free-feature covariance eigenvalue index')
 ax.set_ylabel('Eigenvalue');ax.legend(frameon=False,fontsize=7);ax.grid(alpha=.2)
 fig.tight_layout();fig.savefig(OUT/'excitation_information.pdf');plt.close(fig)
 print('Generated five paired-excitation diagnostics')
if __name__=='__main__':make()
