"""Render only the completed predeclared comparison records, without rerunning."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--input', required=True, type=Path)
parser.add_argument('--figures-output', required=True, type=Path)
parser.add_argument('--tables-output', required=True, type=Path)
args = parser.parse_args()
DATA = args.input
OUT = args.figures_output
OUT.mkdir(parents=True, exist_ok=True)
args.tables_output.parent.mkdir(parents=True, exist_ok=True)
METHODS = ('memory_pg', 'memory_ewls', 'instant_nlms')
LABELS = ('Memory PG', 'Memory EWLS', 'Instantaneous NLMS')
COLORS = ('#2166ac', '#d95f02', '#1b9e77')
aggregate = json.loads((DATA / 'aggregate.json').read_text())['groups']
plt.rcParams.update({'font.size':9, 'axes.labelsize':9, 'legend.fontsize':8,
                     'pdf.fonttype':42, 'ps.fonttype':42})


def trace(seed, method):
    return np.genfromtxt(DATA / 'common_stream' / f'seed_{seed}' / method / 'trace.csv',
                         delimiter=',', names=True)


fig, axes = plt.subplots(1, 2, figsize=(7.15, 2.7), constrained_layout=True)
for method, label, color in zip(METHODS, LABELS, COLORS):
    records = [trace(seed, method) for seed in range(10, 20)]
    for ax, key in zip(axes, ('learned_partial_error', 'preupdate_energy')):
        # Non-overlapping one-second means per seed, then across-seed mean/SD.
        binned = np.stack([np.nanmean(record[key][:-1].reshape(80,100), axis=1)
                           for record in records])
        mean, sd = binned.mean(axis=0), binned.std(axis=0, ddof=1)
        t = np.arange(80)+.5
        ax.plot(t, mean, label=label, color=color, lw=1.4)
        ax.fill_between(t, np.maximum(mean-sd,1e-12), mean+sd, color=color, alpha=.18, linewidth=0)
        ax.set_xlabel('Time (s)')
        ax.grid(alpha=.2)
axes[0].set_ylabel('Learned-column\nrelative differential error')
axes[1].set_ylabel('Pre-update sampled energy')
axes[1].set_yscale('log')
axes[0].legend(loc='upper right', frameon=False)
fig.savefig(OUT / 'estimator_common_stream.pdf')
plt.close(fig)

records = [trace(10, method) for method in METHODS]
t = records[0]['time']
selected = (t >= 60) & (t < 65)
fig, axes = plt.subplots(3, 1, figsize=(7.15, 4.6), sharex=True, constrained_layout=True)
for ax, j, label in zip(axes, (0, 1, 3), (r'$\partial f/\partial x$',
                                       r'$\partial f/\partial \dot{x}$',
                                       r'$\partial f/\partial \phi$')):
    for record, method_label, color in zip(records, LABELS, COLORS):
        ax.plot(t[selected], record[f'F_est_{j}'][selected], label=method_label,
                color=color, lw=1.1, alpha=.9)
    ax.plot(t[selected], records[0][f'F_true_{j}'][selected], color='black',
            linestyle='--', lw=1.3, label='Evaluator truth')
    ax.set_ylabel(label)
    ax.grid(alpha=.2)
axes[0].legend(loc='upper center', ncol=4, frameon=False, bbox_to_anchor=(.5,1.42))
axes[-1].set_xlabel('Time (s)')
fig.savefig(OUT / 'estimator_partial_tracking.pdf')
plt.close(fig)


def entry(group, metric, scale=1., digits=5):
    data = aggregate[group]['metrics'][metric]
    return f"${data['mean']*scale:.{digits}f}\\pm{data['sample_sd']*scale:.{digits}f}$"


lines = [r'\begin{table*}[t]', r'\centering',
 r'\caption{Same-stream Fourier estimator comparison, mean $\pm$ sample SD over ten seeds on $[60,80)$~s. All three estimators observe the original memory-PG controller trajectory. Differential errors include the phase column; the learned-column error excludes the supplied exact input partial.}',
 r'\begin{tabular}{lcccc}', r'\toprule',
 r'Estimator & Full $e_F$ (\%) & Learned columns (\%) & Phase RMSE ($10^{-3}$) & Energy ($10^{-4}$)\\', r'\midrule']
for method, label in zip(METHODS, LABELS):
    group = 'common_stream/'+method
    values = [entry(group,'F_error_mean',100),entry(group,'learned_partial_error_mean',100),
              entry(group,'phase_partial_rmse',1000),entry(group,'preupdate_energy_mean',10000)]
    lines.append(label+' & '+' & '.join(values)+r'\\')
lines += [r'\bottomrule',r'\end{tabular}',r'\label{tab:estimator_same_stream}',r'\end{table*}', '',
 r'\begin{table}[t]',r'\centering',
 r'\caption{Separate closed-loop Fourier trajectories, mean $\pm$ sample SD over ten seeds on $[60,80)$~s. The memory-PG branch is the primary generating trajectory; EWLS and NLMS each control their own trajectory.}',
 r'\setlength{\tabcolsep}{4pt}',r'\begin{tabular}{lcc}',r'\toprule',
 r'Controller & Tracking RMSE ($10^{-3}$) & Input RMS\\',r'\midrule']
for method, label in zip(METHODS, LABELS):
    group = ('common_stream/' if method=='memory_pg' else 'own_trajectory/')+method
    lines.append(label+' & '+entry(group,'tracking_rmse',1000)+' & '+entry(group,'control_rms',1,6)+r'\\')
lines += [r'\bottomrule',r'\end{tabular}',r'\label{tab:estimator_closed_loop}',r'\end{table}', '']
args.tables_output.write_text('\n'.join(lines))
print('Wrote two figures and comparison tables from all completed seeds.')
