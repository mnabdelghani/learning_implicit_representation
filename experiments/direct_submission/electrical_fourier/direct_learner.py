"""Direct differential-entry learning. No plant or simulator imports.

The independently adaptive row contains (2K+1)*4 weights in harmonic-major
order. Its evaluated entries multiply normalized observed rates. The only
time representation input is an observation-calibrated scalar frequency.
"""
import numpy as np
from scipy.integrate import solve_ivp


def basis(time, frequency_hz, order):
    phase = 2*np.pi*frequency_hz*np.asarray(time)
    result = [np.ones_like(phase)]
    for k in range(1, order+1):
        result.extend((np.cos(k*phase), np.sin(k*phase)))
    return np.stack(result, axis=-1)


def evaluate(weights, time, frequency_hz, order):
    return basis(time, frequency_hz, order) @ weights.reshape(2*order+1, 4)


def memory_step(weights, memory, observed_rate, time, frequency_hz,
                *, order, dt, tau, gain, scales, rtol=1e-9, atol=1e-11):
    """Integrate the coupled normalized law over one held observation.

    With R(s) known analytically for the held lifted rate, the solution of
    adot=-gain*R(s)*a, normalized, exactly solves the unit-row tangent law.
    R is evaluated at each ODE stage, not frozen at an interval endpoint.
    """
    w = np.kron(basis(time, frequency_hz, order), observed_rate/scales)
    outer = np.outer(w, w)
    def local(s):
        return np.exp(-s/tau)*memory - tau*np.expm1(-s/tau)*outer
    solution = solve_ivp(lambda s,a: -gain*(local(s)@a), (0.,dt), weights,
                         method='DOP853', rtol=rtol, atol=atol)
    if not solution.success:
        raise RuntimeError(solution.message)
    a = solution.y[:,-1]
    length = np.linalg.norm(a)
    if not np.isfinite(length) or length < np.finfo(float).tiny:
        raise FloatingPointError('Unnormalizable learned row')
    return a/length, local(dt)


def physical_field(weights, time, frequency_hz, order, scales):
    return evaluate(weights, time, frequency_hz, order)/scales


def complete(weights, time, frequency_hz, order, scales, supplied_rates,
             computed_index, guard):
    """One-channel homogeneous completion, with normalized-field guard.

    supplied_rates includes zero in the computed slot. No true field, model
    parameters, or future plant observations are accessible here.
    """
    row = evaluate(weights, time, frequency_hz, order)
    norm = np.linalg.norm(row)
    conditioned = np.isfinite(norm) and norm > 1e-12
    if not conditioned or abs(row[computed_index])/norm < guard:
        return np.nan, False
    scaled_rates = np.asarray(supplied_rates)/scales
    value = -np.dot(row, scaled_rates)/row[computed_index]*scales[computed_index]
    return float(value), bool(np.isfinite(value))
