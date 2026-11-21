"""Direct matrix learning from measured context rates, without a residual model.

The caller chooses observable context channels and fixed units. This module has
no physical coefficients, residual features, prescribed partial entries, plant
model or truth channel. Every entry of Q is adaptive. A local matrix requires
the identification/locality assumptions; executing this law does not establish
them. Independent harmonic weights can use the same integrator on lifted rates.
"""
from __future__ import annotations

import numpy as np
from scipy.integrate import solve_ivp


def random_frame(rows: int, columns: int, seed: int) -> np.ndarray:
    """A generic orthonormal-row initialization independent of any plant."""
    if not 0 < rows < columns:
        raise ValueError("Require 0 < rows < columns")
    basis, _ = np.linalg.qr(np.random.default_rng(seed).normal(size=(columns, rows)))
    return basis.T


def coupled_step(Q, R, observed_rate, dt, forgetting, gain, *,
                 rtol=1e-10, atol=1e-12):
    """Integrate Qdot=-gain*(QR-QRQ^T Q), Rdot=vv^T-forgetting*R.

    The measured rate v is held over this observation interval; memory evolves
    throughout the Q integration. R has its analytic held-rate solution. A
    final polar correction removes only numerical orthogonality drift, which
    is returned in the diagnostics. This is not an eigensolver replacement for
    learning and does not hold R fixed during the Q update.
    """
    Q, R, v = (np.asarray(a, dtype=float) for a in (Q, R, observed_rate))
    if Q.ndim != 2:
        raise ValueError("Q must be a matrix")
    m, d = Q.shape
    if not 0 < m < d or R.shape != (d, d) or v.shape != (d,):
        raise ValueError("Inconsistent field, memory or observed-rate dimensions")
    if not all(np.all(np.isfinite(a)) for a in (Q, R, v)):
        raise ValueError("Inputs must be finite")
    if not np.isfinite([dt, forgetting, gain]).all() or dt <= 0 or forgetting < 0 or gain <= 0:
        raise ValueError("Require dt>0, forgetting>=0 and gain>0")
    if np.linalg.norm(Q @ Q.T - np.eye(m)) > 1e-8:
        raise ValueError("Q must have orthonormal rows")
    if np.linalg.norm(R - R.T) > 1e-10 * max(1., np.linalg.norm(R)):
        raise ValueError("Memory must be symmetric")
    R0 = (R + R.T) / 2
    if np.linalg.eigvalsh(R0)[0] < -1e-10 * max(1., np.linalg.norm(R0)):
        raise ValueError("Memory must be positive semidefinite")
    outer = np.outer(v, v)

    def memory(t):
        if forgetting == 0:
            return R0 + t * outer
        return np.exp(-forgetting * t) * R0 + (-np.expm1(-forgetting * t) / forgetting) * outer

    def rhs(t, flat):
        q = flat.reshape(m, d)
        qr = q @ memory(t)
        return (-gain * (qr - (qr @ q.T) @ q)).ravel()

    sol = solve_ivp(rhs, (0., float(dt)), Q.ravel(), method="DOP853",
                    rtol=rtol, atol=atol, t_eval=[float(dt)])
    if not sol.success:
        raise RuntimeError(sol.message)
    raw = sol.y[:, -1].reshape(m, d)
    drift = float(np.linalg.norm(raw @ raw.T - np.eye(m)))
    if drift > 1e-7:
        raise RuntimeError("Integration orthogonality error exceeds roundoff-correction allowance")
    values, vectors = np.linalg.eigh(raw @ raw.T)
    if values[0] <= 0:
        raise RuntimeError("Numerical integration lost row rank")
    normalized = (vectors * (1 / np.sqrt(values))) @ vectors.T @ raw
    diagnostics = dict(preupdate_incompatibility=(Q @ v).copy(),
                       preupdate_energy=float(np.linalg.norm(Q @ v) ** 2 / 2),
                       raw_orthogonality_error=drift,
                       polar_correction=float(np.linalg.norm(normalized - raw)),
                       function_evaluations=int(sol.nfev))
    return normalized, memory(dt), diagnostics


def complete_rates(Q, supplied_columns, supplied_rates, computed_columns,
                   *, minimum_singular_value=1e-4):
    """Unique regular least-squares rate completion using the learned field.

    No physical residual is inverted here. If the result is an input derivative,
    the caller must integrate it from the measured current input. This function
    rejects an ill-conditioned query rather than substituting an oracle model.
    """
    Q = np.asarray(Q, dtype=float)
    S, C = list(supplied_columns), list(computed_columns)
    rates = np.asarray(supplied_rates, dtype=float)
    if not np.isfinite(minimum_singular_value) or minimum_singular_value <= 0:
        raise ValueError("The singular-value guard must be positive and finite")
    if Q.ndim != 2 or min(Q.shape) == 0 or not np.all(np.isfinite(Q)) or not np.all(np.isfinite(rates)):
        raise ValueError("Field must be a nonempty finite matrix; supplied rates must be finite")
    if sorted(S + C) != list(range(Q.shape[1])) or not C or rates.shape != (len(S),):
        raise ValueError("Supplied/computed columns must partition the context")
    block = Q[:, C]
    sigma = np.linalg.svd(block, compute_uv=False)
    if len(C) > Q.shape[0] or sigma[-1] < minimum_singular_value:
        raise ValueError("Computed-rate block is not regularly invertible")
    return np.linalg.lstsq(block, -(Q[:, S] @ rates), rcond=None)[0]
