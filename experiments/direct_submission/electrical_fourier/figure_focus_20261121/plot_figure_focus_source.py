"""Four-part electrical figure and separate diagnostics from immutable records.

The displayed case and final-one-second interval were declared before this
presentation change. No learner/simulator imports or experimental reruns occur.
"""
from pathlib import Path
import argparse
import hashlib
import json
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def causal_mean(values, count):
    total = np.r_[0., np.cumsum(values)]
    end = np.arange(len(values)) + 1
    start = np.maximum(0, end-count)
    return (total[end]-total[start])/(end-start)


def outside_text(fig):
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    box = fig.bbox
    outside = []
    inactive_ticks = set()
    for ax in fig.axes:
        for axis in [ax.xaxis, ax.yaxis]:
            low, high = sorted(axis.get_view_interval())
            for tick in axis.get_major_ticks()+axis.get_minor_ticks():
                if not low <= tick.get_loc() <= high:
                    inactive_ticks.update([id(tick.label1), id(tick.label2)])
    for item in fig.findobj(match=matplotlib.text.Text):
        if item.get_visible() and item.get_text() and id(item) not in inactive_ticks:
            b = item.get_window_extent(renderer)
            if b.x0 < box.x0-.5 or b.y0 < box.y0-.5 or b.x1 > box.x1+.5 or b.y1 > box.y1+.5:
                outside.append(dict(text=item.get_text(), bounds=list(b.bounds)))
    return outside


def main(records, output):
    output.mkdir(parents=True, exist_ok=False)
    (output/'plot_figure_focus.py').write_bytes(Path(__file__).read_bytes())
    input_hashes = {}

    def read(case, name):
        path = records/f'seed_{case}'/name
        input_hashes[str(path.relative_to(records))] = sha(path)
        if path.suffix == '.npz':
            with np.load(path) as source:
                return {key: source[key] for key in source.files}
        return json.loads(path.read_text())

    training = read(10, 'training.npz')
    config = read(10, 'effective_protocol.json')
    assert config['identity'] == 'electrical_fourier_direct_20261121_v1'
    assert config['seed'] == config['illustration_seed'] == 10
    assert config['confirmation_seeds'] == list(range(10, 20))
    assert config['harmonic_order'] == 2
    assert config['learner_dt_s'] == .005 and config['training_duration_s'] == 120.
    evaluator = read(10, 'training_evaluator.npz')
    period = read(10, 'period_estimate.json')
    summary = read(10, 'summary.json')
    prediction = read(10, 'prediction_fourier.npz')
    prediction_truth = read(10, 'prediction_evaluator.npz')
    control = read(10, 'control_fourier.npz')
    constant_control = read(10, 'control_K0.npz')
    assert summary['tasks']['fourier']['prediction']['status'] == 'complete'
    assert summary['tasks']['fourier']['control']['status'] == 'complete'
    assert summary['tasks']['K0']['control']['status'] == 'complete'
    t = training['time_s']
    relative = training['time_relative_s']
    dt = float(np.median(np.diff(t)))
    assert np.allclose(np.diff(t), dt)
    scales = np.array(config['scales'])
    phase = 2*np.pi*period['frequency_hz']*t
    basis = np.column_stack((np.ones(len(t)), np.cos(phase), np.sin(phase),
                             np.cos(2*phase), np.sin(2*phase)))
    rows = np.einsum('nk,nkj->nj', basis, training['C_pre'].reshape(-1, 5, 4))
    truth = evaluator['physical_differential']*scales
    energy_raw = training['energy_fourier_pre']
    energy_check = .5*np.sum(rows*(training['observed_rates']/scales), axis=1)**2
    energy_max_error = float(np.max(np.abs(energy_check-energy_raw)))
    assert np.allclose(energy_check, energy_raw, rtol=1e-8, atol=1e-12)
    smooth_count = round(1./dt)
    energy_mean = causal_mean(energy_raw, smooth_count)
    entry_mask = relative >= 119.
    assert len(t[entry_mask]) == 201
    # A SINGLE evaluator-only least-squares scalar/sign for the entire fixed
    # display interval. No pointwise unit normalization or anchor entry is used.
    # All four learned traces retain their relative amplitudes and variation.
    numerator = float(np.sum(rows[entry_mask]*truth[entry_mask]))
    denominator = float(np.sum(rows[entry_mask]**2))
    if not np.isfinite(denominator) or denominator <= 1e-12:
        raise ValueError('Degenerate display-alignment denominator')
    alpha = numerator/denominator
    display_estimate = alpha*rows[entry_mask]
    display_truth = truth[entry_mask]
    error_after = display_estimate-display_truth
    field_errors = []
    constant_errors = []
    for seed in range(10, 20):
        seed_cfg = config if seed == 10 else read(seed, 'effective_protocol.json')
        assert seed_cfg['seed'] == seed
        for setting in ['identity', 'scales', 'learner_dt_s', 'training_duration_s', 'harmonic_order']:
            assert seed_cfg[setting] == config[setting]
        seed_ev = evaluator if seed == 10 else read(seed, 'training_evaluator.npz')
        assert seed_ev['projector_error_fourier'].shape == relative.shape
        field_errors.append(seed_ev['projector_error_fourier'])
        constant_errors.append(seed_ev['projector_error_K0'])
    field_errors = np.stack(field_errors)
    constant_errors = np.stack(constant_errors)

    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 8.2,
        'axes.titlesize': 9.2, 'axes.labelsize': 8.3, 'legend.fontsize': 7.5,
        'xtick.labelsize': 7.7, 'ytick.labelsize': 7.7,
        'pdf.fonttype': 42, 'ps.fonttype': 42,
        'axes.spines.top': False, 'axes.spines.right': False,
        'lines.linewidth': 1.0})
    blue = '#006bb6'
    orange = '#ca650d'
    green = '#16845a'
    purple = '#8547a1'
    dark = '#262626'
    truth_style = dict(linestyle=':', linewidth=2.1)
    learned_style = dict(linestyle='-', linewidth=.85)
    fig, axes = plt.subplots(2, 2, figsize=(7.3, 3.8))
    fig.subplots_adjust(left=.098, right=.985, bottom=.115, top=.93,
                        wspace=.30, hspace=.67)
    a, b, c, d = axes.flat
    a.semilogy(relative, np.maximum(energy_mean, 1e-16), color=blue)
    a.set(title='(a) Incompatibility energy', xlabel='Learning time (s)',
          ylabel=r'$\frac{1}{2}|\widehat F\dot z|^2$', xlim=(0, 120), xticks=[0, 40, 80, 120])
    a.text(.98, .96, '1-s causal mean', transform=a.transAxes,
           ha='right', va='top', fontsize=7.5)
    entry_time = relative[entry_mask]-119.
    colors = [blue, orange, green, purple]
    names = [r'$F_i$', r'$F_q$', r'$F_\Phi$', r'$F_t$']
    for j, color in enumerate(colors):
        b.plot(entry_time, display_truth[:, j], color=color, **truth_style)
        b.plot(entry_time, display_estimate[:, j], color=color, **learned_style)
    b.set(title='(b) Learned partials',
          xlabel='Last learning second (s)', ylabel='Scaled partial entry',
          xlim=(0, 1), ylim=(-5.8, 6.3), xticks=[0, .5, 1], yticks=[-5, 0, 5])
    b.legend([Line2D([], [], color=color, lw=1.5) for color in colors], names,
             loc='upper center', ncol=4, frameon=False, columnspacing=.8,
             handlelength=1.1, handletextpad=.35, borderpad=.1)
    prediction_t = prediction['time_relative_s']
    prediction_truth_t = prediction_truth['time_s']-prediction_truth['time_s'][0]
    c.plot(prediction_truth_t, prediction_truth['current_a'], color=dark,
           label='True current', **truth_style)
    c.plot(prediction_t, prediction['predicted_current_a'], color=blue,
           label='Prediction', **learned_style)
    c.set(title='(c) Current prediction', xlabel='Prediction time (s)',
          ylabel='Current (A)', xlim=(0, 12), ylim=(-1.65, 2.05), xticks=[0, 4, 8, 12])
    c.legend(frameon=False, loc='lower left', ncol=2, handlelength=1.7,
             columnspacing=1.0, borderpad=.1)
    control_t = control['time_relative_s']
    d.plot(control_t, control['reference_current_a'], color=dark,
           label='Reference', **truth_style)
    d.plot(control_t, control['current_a'], color=blue,
           label='Controlled current', **learned_style)
    d.set(title='(d) Current control', xlabel='Control time (s)',
          ylabel='Current (A)', xlim=(0, 12), ylim=(-.4, 1.), xticks=[0, 4, 8, 12])
    d.legend(frameon=False, loc='lower left', ncol=2, handlelength=1.7,
             columnspacing=.8, borderpad=.1)
    for ax in axes.flat:
        ax.grid(alpha=.17)
    main_outside = outside_text(fig)
    for extension in ['pdf', 'png']:
        fig.savefig(output/f'direct_fourier_focus.{extension}', dpi=240)
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(7.3, 2.5))
    fig.subplots_adjust(left=.075, right=.99, bottom=.35, top=.84, wspace=.43)
    a, b, c = axes
    for values, color, name in [(field_errors, blue, 'Fourier'),
                                (constant_errors, orange, 'Constant-only')]:
        a.fill_between(relative, values.min(axis=0), values.max(axis=0),
                        color=color, alpha=.17, lw=0)
        a.semilogy(relative, np.median(values, axis=0), color=color, label=name)
    a.set(title='(a) All-seed row-space error', xlabel='Learning time (s)',
          ylabel='Projector error', xlim=(0, 120), xticks=[0, 60, 120])
    voltage_mask = relative >= 116.
    b.plot(relative[voltage_mask]-116., training['observed_rates'][voltage_mask, 2],
           color=blue, label='Applied voltage')
    b.plot(relative[voltage_mask]-116., evaluator['physical_differential'][voltage_mask, 3],
           color=purple, label=r'Clock partial $F_t$')
    b.set(title='(b) Voltage and clock partial', xlabel='Last four learning seconds (s)',
          ylabel='Voltage (V)', xlim=(0, 4), xticks=[0, 2, 4])
    c.plot(control_t, control['reference_current_a'], color=dark, label='Reference', **truth_style)
    c.plot(control_t, control['current_a'], color=blue, label='Fourier', **learned_style)
    c.plot(constant_control['time_relative_s'], constant_control['current_a'],
           color=orange, label='Constant-only', linewidth=.75)
    c.set(title='(c) Control comparison', xlabel='Control time (s)',
          ylabel='Current (A)', xlim=(0, 12), xticks=[0, 6, 12])
    for ax in axes:
        ax.grid(alpha=.17)
        ax.title.set_fontsize(8.4)
        ax.legend(frameon=False, loc='upper center', bbox_to_anchor=(.5, -.40),
                  ncol=2, fontsize=7., handlelength=1.4, columnspacing=.8,
                  handletextpad=.4, borderpad=.1)
    diagnostic_outside = outside_text(fig)
    for extension in ['pdf', 'png']:
        fig.savefig(output/f'direct_fourier_diagnostics.{extension}', dpi=240)
    plt.close(fig)

    np.savez_compressed(output/'display_data.npz',
        training_time_s=t, training_time_relative_s=relative,
        preupdate_energy=energy_raw, energy_causal_mean=energy_mean,
        evaluated_scaled_estimate=rows, scaled_true_field=truth,
        display_time_relative_s=entry_time, display_scalar=np.array(alpha),
        displayed_estimate=display_estimate, displayed_truth=display_truth,
        displayed_difference=error_after,
        projector_errors_fourier_all_seeds=field_errors,
        projector_errors_constant_all_seeds=constant_errors,
        applied_training_voltage=training['observed_rates'][:, 2],
        physical_clock_partial=evaluator['physical_differential'][:, 3],
        prediction_time_s=prediction_t, predicted_current=prediction['predicted_current_a'],
        prediction_truth_time_s=prediction_truth_t, true_prediction_current=prediction_truth['current_a'],
        control_time_s=control_t, controlled_current=control['current_a'],
        reference_current=control['reference_current_a'],
        constant_control_time_s=constant_control['time_relative_s'],
        constant_control_current=constant_control['current_a'])
    manifest = dict(identity='electrical_four_panel_presentation_20261121_v1',
        illustration_seed=10, scientific_runs=0, record_mutations=0,
        command_example='python plot_figure_focus.py --records PATH_TO_RECORDS --output NEW_DIRECTORY',
        environment=dict(python=sys.version, numpy=np.__version__, matplotlib=matplotlib.__version__),
        input_sha256=input_hashes, source_sha256=sha(Path(__file__)),
        alignment=dict(method='One least-squares scalar/sign across all four entries and all saved times in the original final1s display window; evaluator-only',
            formula='Hhat=C_pre evaluated at observed calibrated frequency; A=F_evaluator D; alpha=sum(Hhat*A)/sum(Hhat**2); display alpha*Hhat versus A',
            context_scales=scales.tolist(), window_relative_s=[119., 120.], endpoints_included=True,
            samples=int(entry_mask.sum()), scalar=alpha, numerator=numerator, denominator=denominator,
            no_pointwise_normalization=True, no_entry_forced_to_truth=True,
            raw_entry_rms_before_alignment=np.sqrt(np.mean((rows[entry_mask]-display_truth)**2, axis=0)).tolist(),
            scaled_entry_rms_after_alignment=np.sqrt(np.mean(error_after**2, axis=0)).tolist()),
        preprocessing=dict(energy='Actual saved pre-update energy, reconstructed from evaluated C_pre and observed rates',
            energy_causal_mean_seconds=1., energy_causal_mean_samples=smooth_count,
            energy_startup='Shortened trailing window', energy_log_display_floor=1e-16,
            curve_decimation=1, task_transients='Full saved12s including initial transient',
            projector_band='Pointwise min–max over all ten declared seeds; unsmoothed median line'),
        panels=dict(main_a='Seed10 incompatibility energy; no constant-only comparison',
            main_b='All four estimated/true scaled partial entries in single fixed scalar/sign display convention',
            main_c='Independent current prediction versus evaluator current; no future output assimilation',
            main_d='Learned-field current control versus task reference; weights remain frozen',
            supplement_a='All ten seeds Fourier versus constant-only projector error',
            supplement_b='Applied voltage and true clock partial; both in volts',
            supplement_c='Fourier and constant-only own-trajectory control versus shared reference'),
        checks=dict(max_energy_reconstruction_error=energy_max_error,
            main_text_outside_canvas=main_outside, diagnostics_text_outside_canvas=diagnostic_outside),
        figures=dict(main_inches=[7.3, 3.8], diagnostics_inches=[7.3, 2.5]),
        outputs={p.name:dict(bytes=p.stat().st_size, sha256=sha(p)) for p in sorted(output.iterdir()) if p.is_file()})
    (output/'figure_manifest.json').write_text(json.dumps(manifest, indent=2, allow_nan=False)+'\n')
    print(json.dumps(dict(output=str(output), scalar=alpha,
        entry_rms=manifest['alignment']['scaled_entry_rms_after_alignment'],
        checks=manifest['checks']), indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--records', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    main(args.records, args.output)
