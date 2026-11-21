"""Observation-only period calibration from a constant-voltage prefix.

This module imports no plant model/configuration. Inputs are measured clock
times and currents. It estimates a persistent observable fundamental over a
declared broad band, not arbitrary unknown dynamics or changing frequency.
"""
import numpy as np
from scipy.optimize import minimize_scalar
from scipy.signal import find_peaks, periodogram


def _harmonic_fit(time, current, frequency_hz, harmonics):
    phase = 2 * np.pi * frequency_hz * time
    columns = [np.ones(len(time)), time / max(np.ptp(time), 1e-12)]
    for k in range(1, harmonics + 1):
        columns.extend((np.cos(k*phase), np.sin(k*phase)))
    design = np.column_stack(columns)
    coefficients, _, _, _ = np.linalg.lstsq(design, current, rcond=None)
    residual = current - design @ coefficients
    return float(residual @ residual / len(time)), coefficients, residual


def estimate_period(time_s, observed_current_a, *, search_hz=(0.5, 15.0),
                    discard_seconds=2.0, harmonics=4, max_fit_samples=2400,
                    spectral_peak_count=8):
    """Estimate frequency without simulator truth, known phase, or rate targets.

    Only the supplied prefix is read. A fixed settling interval is discarded.
    Candidate frequencies come from Hann-window current-spectrum peaks and
    their first four subharmonics, followed by harmonic least-squares refinement.
    A candidate must have a resolved first harmonic (four residual-noise
    standard errors and 1% of fitted periodic amplitude) to avoid relabeling
    a period as an arbitrary integer multiple. The fitted offset/trend is a
    nuisance term and is not passed to the direct learner as a physical model.
    """
    t, y = np.asarray(time_s, float), np.asarray(observed_current_a, float)
    if t.ndim != 1 or y.shape != t.shape or len(t) < 32:
        raise ValueError('Need matching one-dimensional time/current arrays')
    if not np.all(np.isfinite(t)) or not np.all(np.isfinite(y)):
        raise ValueError('Time/current observations must be finite')
    differences = np.diff(t)
    dt = float(np.median(differences))
    if dt <= 0 or not np.allclose(differences, dt, rtol=1e-5, atol=1e-10):
        raise ValueError('This prototype requires uniformly spaced time observations')
    low, high = map(float, search_hz)
    if low <= 0 or high <= low or high * harmonics >= .5 / dt:
        raise ValueError('Search band and fitted harmonics must be below Nyquist')
    keep = t >= t[0] + discard_seconds
    if np.count_nonzero(keep) < 64:
        raise ValueError('Insufficient retained prefix after settling interval')
    t, y = t[keep], y[keep]
    t = t - t[0]
    span = float(t[-1] - t[0])
    nfft = 1 << int(np.ceil(np.log2(8 * len(t))))
    frequencies, power = periodogram(y, fs=1/dt, window='hann', detrend='linear', nfft=nfft)
    eligible = (frequencies >= low) & (frequencies <= high)
    peaks, _ = find_peaks(power)
    peaks = peaks[eligible[peaks]]
    if len(peaks) == 0:
        raise ValueError('No interior spectral peak in declared frequency band')
    peaks = peaks[np.argsort(power[peaks])[-spectral_peak_count:]][::-1]
    fft_peak = float(frequencies[peaks[0]])
    seeds = []
    for peak in peaks:
        for divisor in range(1, harmonics + 1):
            candidate = float(frequencies[peak] / divisor)
            if low <= candidate <= high and all(abs(candidate-s) > .25/span for s in seeds):
                seeds.append(candidate)
    requested_stride = max(1, int(np.ceil(len(t)/max_fit_samples)))
    # A sample-count cap must not alias any searched harmonic. Preserve at
    # least the original Nyquist margin, and use a 0.45-cycle/sample margin
    # when additional decimation is possible. This bound depends only on
    # declared search limits, observation dt and harmonic count, never truth.
    maximum_safe_stride = max(1, int(np.floor(.45/(high*harmonics*dt))))
    stride = min(requested_stride, maximum_safe_stride)
    fit_t, fit_y = t[::stride], y[::stride]
    candidates = []
    for initial in seeds:
        bounds = (max(low, initial - .75/span), min(high, initial + .75/span))
        fit = minimize_scalar(lambda f: _harmonic_fit(fit_t, fit_y, f, harmonics)[0],
                              method='bounded', bounds=bounds,
                              options=dict(xatol=1e-10, maxiter=80))
        mse, coefficients, residual = _harmonic_fit(fit_t, fit_y, fit.x, harmonics)
        amplitudes = np.hypot(coefficients[2::2], coefficients[3::2])
        threshold = max(.01 * float(np.linalg.norm(amplitudes)),
                        4 * np.sqrt(2*mse/len(fit_t)))
        candidates.append(dict(frequency_hz=float(fit.x), mse_a2=mse,
                               first_harmonic_amplitude_a=float(amplitudes[0]),
                               harmonic_amplitudes_a=amplitudes.tolist(),
                               first_harmonic_threshold_a=float(threshold),
                               resolved_first_harmonic=bool(amplitudes[0] > threshold),
                               optimizer_success=bool(fit.success)))
    resolved = [item for item in candidates if item['resolved_first_harmonic'] and item['optimizer_success']]
    if not resolved:
        return dict(status='unresolved', frequency_hz=None, fft_peak_hz=fft_peak,
                    search_hz=[low, high], candidates=candidates,
                    reason='No candidate has a statistically and relatively resolved first harmonic')
    chosen = min(resolved, key=lambda x: x['mse_a2'])
    frequency = chosen['frequency_hz']
    mse, coefficients, residual = _harmonic_fit(t, y, frequency, harmonics)
    distinct = [item for item in resolved if abs(item['frequency_hz']-frequency) > .5/span]
    runner_up = min(distinct, key=lambda x: x['mse_a2']) if distinct else None
    ambiguity_ratio = None if runner_up is None else runner_up['mse_a2']/max(chosen['mse_a2'], 1e-30)
    return dict(status='estimated', frequency_hz=frequency, period_s=1/frequency,
                fft_peak_hz=fft_peak, search_hz=[low, high],
                retained_duration_s=span, retained_samples=int(len(t)),
                fit_samples=int(len(fit_t)), fitted_harmonics=int(harmonics),
                fit_stride=int(stride), fit_sample_interval_s=float(stride*dt),
                fit_nyquist_hz=float(.5/(stride*dt)),
                maximum_searched_harmonic_hz=float(high*harmonics),
                fit_residual_rms_a=float(np.sqrt(mse)),
                fit_explained_fraction=float(1-mse/max(np.var(y), 1e-30)),
                distinct_runner_up_mse_ratio=ambiguity_ratio,
                ambiguous=bool(ambiguity_ratio is not None and ambiguity_ratio < 1.1),
                candidate_count=len(candidates), candidates=candidates,
                phase_convention='Use 2*pi*f_hat*t; unknown phase is absorbed by sine/cosine weights',
                limitations=['Constant-voltage calibration prefix and stationary periodic response required',
                             'A weak or absent observable fundamental can remain ambiguous',
                             'Frequency frozen after calibration; no online frequency-drift tracking demonstrated'])
