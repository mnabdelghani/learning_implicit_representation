"""Compact four-panel Burgers display from immutable archived plotting data.

No simulation, fitting, smoothing, or truth-based change to learned records.
The input rows already carry the original evaluator-only unit/sign convention.
"""
from pathlib import Path
import argparse
import hashlib
import json
import shutil
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

SEEDS = np.arange(732, 737)
BLUE = '#0072B2'
ORANGE = '#D55E00'
TRUTH = '#252525'
COLORS = [BLUE, ORANGE, '#009E73', '#CC79A7', '#6A3D9A']


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def sample_indices(size, stride):
    return np.unique(np.r_[np.arange(0, size, stride), size - 1]).astype(int)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--stride', type=int, default=5)
    args = parser.parse_args()
    if args.stride < 1:
        parser.error('--stride must be positive')
    if args.output.exists():
        parser.error('--output must be a new directory')
    args.output.mkdir(parents=True)
    data = np.load(args.data, allow_pickle=False)
    arrays = {}
    diagnostics = {'seeds': SEEDS.tolist(), 'finite_energy_start': {},
                   'unapplied_terminal_samples': {}, 'guard_samples': {},
                   'clip_samples': {}, 'training_energy_nan_samples': {}}

    # Retain the archived full time windows, including the displayed terminal
    # observation at t=100; summary metrics below exclude unapplied commands.
    for phase in ['training', 'adaptive', 'frozen']:
        time = data[f'732_{phase}_time']
        for seed in SEEDS:
            assert np.array_equal(time, data[f'{seed}_{phase}_time'])
        arrays[f'{phase}_time'] = time
        arrays[f'{phase}_render_indices'] = sample_indices(len(time), args.stride)
        energy = np.stack([data[f'{seed}_{phase}_energy'] for seed in SEEDS])
        physical = np.stack([data[f'{seed}_{phase}_physical_energy'] for seed in SEEDS])
        task = np.stack([np.sqrt(data[f'{seed}_{phase}_task_mse']) for seed in SEEDS])
        field = np.stack([data[f'{seed}_{phase}_field_mean'] for seed in SEEDS])
        for name, values in [('energy', energy), ('physical_energy', physical), ('task_rms', task), ('field_mean', field)]:
            arrays[f'{phase}_{name}_all_seeds'] = values
            # Plain mean deliberately propagates unobserved startup NaNs.
            arrays[f'{phase}_{name}_mean'] = np.mean(values, axis=0)
            arrays[f'{phase}_{name}_min'] = np.min(values, axis=0)
            arrays[f'{phase}_{name}_max'] = np.max(values, axis=0)
        for seed in SEEDS:
            key = f'{seed}_{phase}'
            applied = data[f'{key}_applied'].astype(bool)
            diagnostics['unapplied_terminal_samples'][key] = int(np.count_nonzero(~applied))
            diagnostics['guard_samples'][key] = int(np.count_nonzero(data[f'{key}_guard'][applied]))
            diagnostics['clip_samples'][key] = int(np.count_nonzero(data[f'{key}_clip'][applied]))
            valid = np.flatnonzero(np.isfinite(data[f'{key}_energy']))
            diagnostics['finite_energy_start'][key] = float(time[valid[0]])
            if phase == 'training':
                diagnostics['training_energy_nan_samples'][str(seed)] = int(np.count_nonzero(~np.isfinite(data[f'{key}_energy'])))
        for key in ['q_site0', 'physical_site0', 'w_site0', 'target_site0']:
            arrays[f'site0_{phase}_{key}'] = data[f'site0_{phase}_{key}']
        assert np.array_equal(time, data[f'site0_{phase}_time'])

    # Reproduce rather than overwrite original metric definitions. Instantaneous
    # task lines are mean spatial RMS; window summaries are temporal-spatial RMS.
    metrics = {}
    for phase in ['adaptive', 'frozen']:
        per_seed = []
        for seed in SEEDS:
            time = data[f'{seed}_{phase}_time']
            applied = data[f'{seed}_{phase}_applied'].astype(bool)
            late = applied & (time >= 96) & (time < 100)
            full = applied & (time >= 80) & (time < 100)
            per_seed.append({'seed': int(seed),
                             'late_task_rms': float(np.sqrt(np.mean(data[f'{seed}_{phase}_task_mse'][late]))),
                             'full_control_task_rms': float(np.sqrt(np.mean(data[f'{seed}_{phase}_task_mse'][full])))})
        metrics[phase] = per_seed

    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 8.3,
                         'axes.titlesize': 9, 'axes.labelsize': 8.3,
                         'xtick.labelsize': 8, 'ytick.labelsize': 8,
                         'legend.fontsize': 8, 'axes.linewidth': .65,
                         'pdf.fonttype': 42, 'ps.fonttype': 42,
                         'savefig.facecolor': 'white'})
    fig, axes = plt.subplots(2, 2, figsize=(7.3, 3.8), layout='constrained')
    a, b, c, d = axes.ravel()
    for ax in axes.ravel():
        ax.spines[['top', 'right']].set_visible(False)
        ax.grid(axis='y', color='#D8DDE2', linewidth=.45, alpha=.7)
        ax.tick_params(length=3, width=.65, pad=2)
        ax.set_axisbelow(True)

    # (a): one actively learning path, training then own-trajectory control.
    for phase in ['training', 'adaptive']:
        sl = arrays[f'{phase}_render_indices']; t = arrays[f'{phase}_time'][sl]
        a.fill_between(t, arrays[f'{phase}_energy_min'][sl], arrays[f'{phase}_energy_max'][sl],
                       color=BLUE, alpha=.12, linewidth=0)
        a.semilogy(t, arrays[f'{phase}_physical_energy_mean'][sl], color=TRUTH,
                   ls=':', lw=1.7, label='Physical row' if phase == 'training' else None)
        a.semilogy(t, arrays[f'{phase}_energy_mean'][sl], color=BLUE, lw=1.05,
                   label='Learned row' if phase == 'training' else None)
    a.set(title='(a) Incompatibility during learning', ylabel='Pre-update secant energy',
          xlabel='Time (s)', xlim=(0, 100))
    a.set_yticks([1e-3, 1e-7, 1e-11, 1e-15])
    a.set_ylim(top=1e-1)
    a.axvline(80, color='#777777', ls='--', lw=.7)
    handles, labels = a.get_legend_handles_labels()
    a.legend(handles[::-1], labels[::-1], loc='upper right', frameon=False)

    # (b): all five estimates and their evaluated true entries, same units/gauge.
    labels = [r'$F_w$', r'$F_{w_s}$', r'$F_{w_{ss}}$', r'$F_u$', r'$F_{w_t}$']
    for phase in ['training', 'adaptive']:
        sl = arrays[f'{phase}_render_indices']; t = arrays[f'{phase}_time'][sl]
        q = arrays[f'site0_{phase}_q_site0']; physical = arrays[f'site0_{phase}_physical_site0']
        for j in range(5):
            b.plot(t, physical[sl, j], color=TRUTH, ls=':', lw=1.65, zorder=2)
            b.plot(t, q[sl, j], color=COLORS[j], lw=1.0, zorder=3)
    b.set(title='(b) All five learned partials', ylabel='Unit/sign-aligned entry',
          xlabel='Time (s)', xlim=(0, 100), ylim=(-1.02, 1.20))
    b.set_yticks([-1, -.5, 0, .5, 1])
    b.axvline(80, color='#777777', ls='--', lw=.7)
    handles = [Line2D([], [], color=color, lw=1.2, label=label) for color, label in zip(COLORS, labels)]
    handles.append(Line2D([], [], color=TRUTH, ls=':', lw=1.7, label='True'))
    b.legend(handles=handles, loc='upper center', ncol=6, frameon=False,
             fontsize=8, handlelength=1.1, columnspacing=.65, handletextpad=.3)

    # (c): whole-field task error; all five seeds and the full shared transient.
    for phase, color, label in [('adaptive', BLUE, 'Adaptive'), ('frozen', ORANGE, 'Frozen at 80 s')]:
        sl = arrays[f'{phase}_render_indices']; t = arrays[f'{phase}_time'][sl]
        c.fill_between(t, arrays[f'{phase}_task_rms_min'][sl], arrays[f'{phase}_task_rms_max'][sl],
                       color=color, alpha=.12, linewidth=0)
        c.semilogy(t, arrays[f'{phase}_task_rms_mean'][sl], color=color, lw=1.15, label=label)
    c.set(title='(c) Whole-field control error', ylabel='Spatial RMS error',
          xlabel='Time (s)', xlim=(80, 100))
    c.set_xticks([80, 85, 90, 95, 100])
    c.legend(frameon=False, loc='upper right')

    # (d): one preselected location on the adaptive branch; no favorable crop.
    sl = arrays['adaptive_render_indices']; t = arrays['adaptive_time'][sl]
    d.plot(t, arrays['site0_adaptive_target_site0'][sl], color=TRUTH,
           ls=':', lw=1.9, label='Desired', zorder=2)
    d.plot(t, arrays['site0_adaptive_w_site0'][sl], color=BLUE,
           lw=1.05, label='Actual', zorder=3)
    d.set(title='(d) Controlled field at site 0', ylabel=r'$w(s=0,t)$',
          xlabel='Time (s)', xlim=(80, 100))
    d.set_xticks([80, 85, 90, 95, 100])
    handles, labels = d.get_legend_handles_labels()
    d.legend(handles[::-1], labels[::-1], loc='lower right', frameon=False)

    pdf = args.output/'direct_burgers_focus.pdf'
    png = args.output/'direct_burgers_focus.png'
    fig.savefig(pdf)
    fig.savefig(png, dpi=220)
    main_axis_limits = [list(ax.get_ylim()) for ax in axes.ravel()]
    plt.close(fig)

    # Additional supplement diagnostic: the whole-field row-space comparison.
    fig_diag, ax = plt.subplots(figsize=(3.5, 2.2), layout='constrained')
    for phase, color in [('training', BLUE), ('adaptive', BLUE), ('frozen', ORANGE)]:
        sl = arrays[f'{phase}_render_indices']; t = arrays[f'{phase}_time'][sl]
        ax.fill_between(t, arrays[f'{phase}_field_mean_min'][sl], arrays[f'{phase}_field_mean_max'][sl],
                        color=color, alpha=.12, linewidth=0)
        label = 'Learning' if phase == 'training' else ('Frozen at 80 s' if phase == 'frozen' else None)
        ax.semilogy(t, arrays[f'{phase}_field_mean_mean'][sl], color=color, lw=1.1, label=label)
    ax.spines[['top', 'right']].set_visible(False)
    ax.grid(axis='y', color='#D8DDE2', linewidth=.45, alpha=.7)
    ax.tick_params(length=3, width=.65, pad=2)
    ax.axvline(80, color='#777777', ls='--', lw=.7)
    ax.set(title='Whole-field representation error', ylabel='Mean row-projector error',
           xlabel='Time (s)', xlim=(0, 100))
    ax.legend(frameon=False, loc='upper right')
    diagnostic_pdf = args.output/'direct_burgers_diagnostics.pdf'
    diagnostic_png = args.output/'direct_burgers_diagnostics.png'
    fig_diag.savefig(diagnostic_pdf); fig_diag.savefig(diagnostic_png, dpi=220)
    plt.close(fig_diag)
    np.savez_compressed(args.output/'display_data.npz', **arrays)
    source_copy = args.output/'plot_figure_focus_source.py'
    shutil.copyfile(Path(__file__), source_copy)
    provenance = {
        'display_identity': 'burgers_figure_focus_20261121_v1',
        'scientific_protocol': 'burgers_direct_field_long_training_multiseed_causal_sensing_20261120_v1',
        'input_file': args.data.name, 'input_sha256': sha256(args.data),
        'source_sha256': sha256(Path(__file__)),
        'representative_seed': 732, 'representative_site': 0,
        'cohort_seeds': SEEDS.tolist(), 'figure_size_inches': [7.3, 3.8], 'main_axis_y_limits': main_axis_limits,
        'diagnostic_figure_size_inches': [3.5, 2.2],
        'panels': {
            'a': 'Arithmetic mean learned-row secant energy and full min/max band across all five seeds, training then adaptive branch; dotted physical-row reference is now the matched five-seed arithmetic mean, rather than only seed732. No frozen-energy curve.',
            'b': 'All five archived site0 entries for preselected seed732, training then adaptive; input arrays retain their original evaluator-only unit normalization and sign alignment. No new alignment.',
            'c': 'Arithmetic mean of instantaneous spatial task RMS across all five seeds, with full min/max bands; adaptive and frozen own-trajectory branches over the full80–100s control window.',
            'd': 'Actual adaptive-branch field versus desired field at preselected seed732/site0, full80–100s control window.'},
        'render_stride': args.stride, 'sample_interval_seconds': .005,
        'last_sample_included': True,
        'terminal_observation_note': 'The saved t100 observation is displayed, as in the original figure; terminal unapplied commands are excluded from reported window metrics.',
        'smoothing': False, 'log_floor': None,
        'startup_missing_energy': 'The first10 training samples (t<.05s) have no complete causal secant and remain NaN gaps; no values are fabricated or filled.',
        'display_changes_only': True,
        'supplement_diagnostic': 'The separate single-axis figure shows whole-field row-projector error, five-seed mean and full range, training and adaptive/frozen control; no metric or time window changed.',
        'retained_elsewhere': 'Original detailed panels remain archived unchanged; the manuscript retains quantitative row-space metrics.',
        'environment': {'python': sys.version, 'numpy': np.__version__, 'matplotlib': matplotlib.__version__},
        'diagnostics': diagnostics, 'task_window_metrics': metrics,
        'artifacts': {x.name: sha256(x) for x in [pdf, png, diagnostic_pdf, diagnostic_png, args.output/'display_data.npz', source_copy]}}
    (args.output/'figure_manifest.json').write_text(json.dumps(provenance, indent=2)+'\n')
    print(json.dumps({'output': str(args.output), 'input_sha256': provenance['input_sha256'],
                      'figure_size_inches': provenance['figure_size_inches'],
                      'control_guards': sum(v for k, v in diagnostics['guard_samples'].items() if '_training' not in k),
                      'control_clips': sum(v for k, v in diagnostics['clip_samples'].items() if '_training' not in k)}, indent=2))


if __name__ == '__main__':
    main()
