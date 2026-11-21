"""Electrical-actuator simulator with a separate observation interface.

An external drive maintains rotor motion. Only the electrical voltage balance
is simulated: u = L*idot + R*i + e(t), with constant positive L,R. Its integrated
relation has context z=(i,q,U,t), qdot=i, Udot=u, and true partials (L,R,-1,e(t)).
Mechanical dynamics, torque control,
saturation, switching electronics, temperature, and magnetic saturation are
outside this model. Parameters and phase are simulator/evaluator-only.
"""
from dataclasses import asdict, dataclass
from typing import Callable

import numpy as np
from scipy.integrate import solve_ivp


@dataclass(frozen=True)
class ElectricalParameters:
    resistance_ohm: float = 2.0
    inductance_h: float = 0.15
    frequency_hz: float = 3.73
    initial_phase_rad: float = 0.41
    emf_first_v: float = 1.5
    emf_second_v: float = 0.45
    emf_first_phase_rad: float = 0.6
    emf_second_phase_rad: float = -0.3


def physical_terms(time, parameters=ElectricalParameters()):
    """Simulator-only constant L, periodic EMF and its derivatives/primitive."""
    p = parameters
    t = np.asarray(time, dtype=float)
    omega = 2 * np.pi * p.frequency_hz
    phase = omega * t + p.initial_phase_rad
    L = np.full_like(t, p.inductance_h)
    Ldot = np.zeros_like(t)
    Lddot = np.zeros_like(t)
    emf = (p.emf_first_v * np.sin(phase + p.emf_first_phase_rad)
           + p.emf_second_v * np.sin(2 * phase + p.emf_second_phase_rad))
    emfdot = omega * (
        p.emf_first_v * np.cos(phase + p.emf_first_phase_rad)
        + 2 * p.emf_second_v * np.cos(2 * phase + p.emf_second_phase_rad))
    emfddot = -omega**2 * (
        p.emf_first_v * np.sin(phase + p.emf_first_phase_rad)
        + 4 * p.emf_second_v * np.sin(2 * phase + p.emf_second_phase_rad))
    emf_integral = (-p.emf_first_v / omega * np.cos(phase + p.emf_first_phase_rad)
                    -p.emf_second_v / (2*omega) * np.cos(2*phase + p.emf_second_phase_rad))
    emf_integral_zero = (-p.emf_first_v / omega * np.cos(p.initial_phase_rad + p.emf_first_phase_rad)
                         -p.emf_second_v / (2*omega) * np.cos(2*p.initial_phase_rad + p.emf_second_phase_rad))
    return dict(inductance_h=L, inductance_rate_h_per_s=Ldot,
                inductance_acceleration_h_per_s2=Lddot, back_emf_v=emf,
                back_emf_rate_v_per_s=emfdot,
                back_emf_acceleration_v_per_s2=emfddot,
                back_emf_integral_from_zero_vs=emf_integral-emf_integral_zero,
                physical_phase_rad=phase)


def current_rate(current, time, voltage, parameters=ElectricalParameters()):
    """Simulator-only right-hand side; not a learner/readout function."""
    p=parameters
    phase=2*np.pi*p.frequency_hz*time+p.initial_phase_rad
    emf=(p.emf_first_v*np.sin(phase+p.emf_first_phase_rad)
         +p.emf_second_v*np.sin(2*phase+p.emf_second_phase_rad))
    return (voltage-emf-p.resistance_ohm*current)/p.inductance_h


def step_current(current, time, dt, applied_voltage,
                 parameters=ElectricalParameters(), max_substep=0.0005):
    """Causal RK4 plant step for voltage held over [time,time+dt).

    All evaluations use the SAME already selected applied_voltage. No future
    controller observation or model estimate enters the simulator step.
    """
    if dt <= 0 or max_substep <= 0:
        raise ValueError('Positive step sizes required')
    count = max(1, int(np.ceil(dt / max_substep)))
    h = dt / count
    value = float(current)
    for k in range(count):
        t = time + k * h
        rhs = lambda x, s: current_rate(x, s, applied_voltage, parameters)
        k1 = rhs(value, t)
        k2 = rhs(value + h * k1 / 2, t + h / 2)
        k3 = rhs(value + h * k2 / 2, t + h / 2)
        k4 = rhs(value + h * k3, t + h)
        value += h * (k1 + 2*k2 + 2*k3 + k4) / 6
    return float(value)


def generate_stream(dt: float, duration: float, seed: int,
                    voltage_fn: Callable[[float], float], *,
                    parameters=ElectricalParameters(), initial_current=0.0, start_time=0.0,
                    current_noise_std=0.0, current_rate_noise_std=0.0,
                    rtol=1e-10, atol=1e-12):
    """Return (observations, simulator_truth) as separate dictionaries.

    Observation keys: time_s, current_a, applied_voltage_v, current_rate_a_per_s.
    The ideal-rate baseline observes current_rate directly, not a numerical
    derivative of noisy current. Optional rate noise models a separate rate
    sensor; it is not described as a causal differentiation algorithm.

    voltage_fn is an exogenous smooth/open-loop command for this generator.
    A feedback controller should instead call step_current with its selected
    held command. Piecewise voltage jumps should be split into separate solver
    intervals; the present generator is used with constant voltage in the pilot.
    A fixed call is deterministic. No simulator parameters are in observations.
    start_time continues the physical clock; voltage_fn receives global time.
    """
    if dt <= 0 or duration <= 0:
        raise ValueError('Positive sampling interval and duration required')
    steps = int(round(duration / dt))
    if not np.isclose(steps * dt, duration, rtol=0, atol=1e-10):
        raise ValueError('Duration must be a whole number of sampling intervals')
    if current_noise_std < 0 or current_rate_noise_std < 0:
        raise ValueError('Sensor standard deviations must be nonnegative')
    p = parameters
    if p.inductance_h <= 0 or p.resistance_ohm <= 0 or p.frequency_hz <= 0:
        raise ValueError('Positive resistance, frequency and inductance required')
    time = float(start_time)+np.arange(steps + 1, dtype=float) * dt
    solution = solve_ivp(
        lambda t, y: [current_rate(y[0], t, float(voltage_fn(t)), p),
                      y[0], float(voltage_fn(t))],
        (float(start_time), float(start_time)+duration), [float(initial_current),0.,0.], t_eval=time, method='DOP853',
        rtol=rtol, atol=atol, max_step=4*dt)
    if not solution.success:
        raise RuntimeError(solution.message)
    current = solution.y[0]
    voltage = np.array([float(voltage_fn(t)) for t in time])
    rate = current_rate(current, time, voltage, p)
    rng = np.random.default_rng(seed)
    observations = dict(time_s=time.copy(),
                        current_a=current + rng.normal(0., current_noise_std, len(time)),
                        applied_voltage_v=voltage.copy(),
                        current_rate_a_per_s=rate + rng.normal(0., current_rate_noise_std, len(time)))
    terms = physical_terms(time, p)
    # True integrated-residual partials for z=(i,q,U,t), evaluator only.
    # f=L*(i-i0)+R*q-U+integral_start^t e(s)ds, qdot=i and Udot=u.
    field = np.column_stack((terms['inductance_h'],
                             np.full(len(time),p.resistance_ohm),
                             -np.ones(len(time)), terms['back_emf_v']))
    truth = dict(time_s=time.copy(), current_a=current, current_rate_a_per_s=rate,
                 applied_voltage_v=voltage, physical_differential=field,
                 integrated_current_c=solution.y[1],
                 integrated_voltage_vs=solution.y[2],
                 back_emf_integral_from_start_vs=(terms['back_emf_integral_from_zero_vs']
                     -physical_terms(start_time,p)['back_emf_integral_from_zero_vs']),
                 resistance_plus_inductance_rate_ohm=(p.resistance_ohm
                         + terms['inductance_rate_h_per_s']),
                 parameters=asdict(p), solver_nfev=int(solution.nfev), **terms)
    return observations, truth
