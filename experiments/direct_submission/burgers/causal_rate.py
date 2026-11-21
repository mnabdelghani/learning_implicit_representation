"""Fixed causal endpoint-quadratic rate sensor, using measured positions only.

At an available endpoint t, fit b0 + b1*r + b2*r**2 to measured samples with
r=(t_i-t)/H in [-1,0], and return b1/H. The first result requires every sample
in the full H-second window; no startup estimate or true-rate fill is supplied.

For chronological time offsets tau_j <= 0, the fixed derivative weights a_j
satisfy sum(a)=0, sum(a*tau)=1 and sum(a*tau**2)=0 (up to roundoff). Therefore
for a sufficiently smooth scalar component,
  estimate - w'(t) = mu3*w'''(t)/6 + mu4*w''''(t)/24 + ...,
  muk = sum_j a_j*tau_j**k.
With bounded third derivative, a rigorous Taylor remainder bound is
  abs(error) <= sup|w'''| * sum_j abs(a_j)*abs(tau_j)**3 / 6.
The complex derivative response is D(omega)=sum_j a_j*exp(i*omega*tau_j).
Relative to the exact derivative, D/(i*omega) =
  1 - mu3*omega**2/6 - i*mu4*omega**3/24 + O(omega**4).
Thus this endpoint estimator has no universal half-window delay. Its response
and equivalent phase delay depend on frequency. The leading nonzero smooth
bias is generally quadratic in H, not zero for arbitrary smooth signals.

Independent measurement noise of variance sigma**2 at each sample produces
rate variance sigma**2 * sum(a_j**2). Overlapping output windows are correlated;
correlated input noise instead gives a.T @ covariance @ a. This sensor does not
remove measurement noise or certify the learner's physical identification.

Checkpoint state is exactly dt, window, and the chronological history deque.
There is no counter, RNG, plant model, truth channel, or adaptive tuning state.
"""
from __future__ import annotations
from collections import deque
import numpy as np


class CausalQuadraticRate:
    """Full-window, equally weighted quadratic least-squares endpoint slope."""

    def __init__(self, dt, window=.1):
        self.dt, self.window = float(dt), float(window)
        if not (np.isfinite(self.dt) and self.dt > 0 and
                np.isfinite(self.window) and self.window > 0):
            raise ValueError('dt and window must be finite and positive')
        ratio = self.window/self.dt
        intervals = round(ratio)
        if intervals < 2 or abs(ratio-intervals) > 1e-10*max(1., abs(ratio)):
            raise ValueError('window must contain an integer number of at least two dt intervals')
        self.sample_offsets = np.arange(-intervals, 1, dtype=float)*self.dt
        scaled_time = self.sample_offsets/self.window
        design = np.column_stack((np.ones(intervals+1), scaled_time, scaled_time**2))
        self.weights = np.linalg.pinv(design)[1]/self.window
        self.weights.setflags(write=False)
        self.sample_offsets.setflags(write=False)
        self.noise_variance_factor = float(self.weights @ self.weights)
        self.history = deque(maxlen=intervals+1)

    def update(self, measured_w):
        """Append one new chronological sample; return rate or None at startup.

        The caller supplies consecutive measurements exactly dt apart. Returned
        arrays and stored samples do not alias the caller's observation array.
        A restored history must be chronological with this same maxlen.
        """
        sample = np.array(measured_w, dtype=float, copy=True)
        if sample.ndim != 1 or sample.size == 0 or not np.all(np.isfinite(sample)):
            raise ValueError('measured_w must be a finite nonempty one-dimensional array')
        if self.history and sample.shape != self.history[-1].shape:
            raise ValueError('measurement shape changed within the causal history')
        self.history.append(sample)
        if len(self.history) < self.history.maxlen:
            return None
        return self.weights @ np.asarray(self.history)

    def diagnostics(self, noise_sigma=1e-5, angular_frequencies=(3., 11.)):
        """Analytic fixed-filter diagnostics; never used to tune the sensor."""
        tau, a = self.sample_offsets, self.weights
        moments = [float(a @ tau**k) for k in range(5)]
        frequency = []
        for omega in angular_frequencies:
            omega = float(omega)
            if not np.isfinite(omega) or omega <= 0:
                raise ValueError('diagnostic angular frequencies must be positive and finite')
            response = np.sum(a*np.exp(1j*omega*tau))/(1j*omega)
            phase = float(np.angle(response))
            frequency.append(dict(angular_frequency=omega,
                relative_amplitude=float(abs(response)), relative_phase_radians=phase,
                equivalent_phase_delay_seconds=-phase/omega))
        return dict(dt=self.dt, window=self.window, samples=self.history.maxlen,
            moments= moments, cubic_bias_coefficient=moments[3]/6,
            quartic_bias_coefficient=moments[4]/24,
            bounded_third_derivative_error_coefficient=float(np.sum(abs(a)*abs(tau)**3)/6),
            noise_variance_factor=self.noise_variance_factor,
            noise_sigma=float(noise_sigma),
            rate_noise_standard_deviation=float(abs(noise_sigma)*np.sqrt(self.noise_variance_factor)),
            frequency_response=frequency,
            assumptions='Independent per-sample measurement noise; successive rate errors are correlated.')
