"""Simulator and evaluator only; no physical coefficients enter the learner."""
import numpy as np

def spatial_grid(n, length=2 * np.pi):
    return np.arange(n, dtype=float) * length / n


def spectral_derivatives(w, length=2 * np.pi):
    """First and second derivatives on an evenly spaced periodic real grid."""
    w = np.asarray(w, dtype=float)
    k = 2 * np.pi * np.fft.fftfreq(w.size, d=length / w.size)
    wh = np.fft.fft(w)
    return np.fft.ifft(1j * k * wh).real, np.fft.ifft(-k * k * wh).real


def plant_rhs(w, u, coefficients, length=2 * np.pi):
    ca, nu, b = coefficients
    ws, wss = spectral_derivatives(w, length)
    return -ca * w * ws + nu * wss + b * u


def implicit_jacobian(z, coefficients):
    """Return F = partial f / partial z in the manuscript coordinate convention."""
    ca, nu, b = np.asarray(coefficients)
    z = np.asarray(z)
    out = np.empty_like(z, dtype=float)
    out[..., 0] = ca * z[..., 1]
    out[..., 1] = ca * z[..., 0]
    out[..., 2] = -nu
    out[..., 3] = -b
    out[..., 4] = 1.0
    return out

