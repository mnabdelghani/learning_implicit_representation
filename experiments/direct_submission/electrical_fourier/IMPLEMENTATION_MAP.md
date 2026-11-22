# Electrical experiment: implementation map

The experiment learns a temporal differential row from observed rates and then
freezes it for independent prediction and current control. The simulator's
parameters, phase, and analytic partials are confined to simulation and
evaluation. `protocol.json` fixes the settings, seed cohort, metric windows,
guards, and two numerical refinements. See `REPRODUCE.txt` for commands and
the exact saved-record inventory.

| Source and function | Inputs | Outputs and role |
|---|---|---|
| `electrical_simulator.generate_stream` | Sampling interval, duration, sensor seed, prescribed voltage, simulator-only physical parameters and initial state | Separate observation and truth dictionaries; integrates physical current and its current/voltage integrals. |
| `electrical_simulator.step_current` | Current, clock, already chosen held voltage, physical parameters, substep limit | Next physical current by RK4; used only to advance the control plant. |
| `electrical_simulator.physical_terms` | Clock and physical parameters | Analytic EMF, derivatives, integrals, phase; simulator/evaluator only. |
| `period_estimator.estimate_period` | Calibration clock/current only and fixed search settings | Frequency estimate, candidate/ambiguity diagnostics; only the scalar frequency enters learning. |
| `direct_learner.basis` / `evaluate` | Clock, calibrated frequency, harmonic order; learned weights for `evaluate` | Harmonic basis / four independently represented differential entries in scaled coordinates. |
| `direct_learner.memory_step` | Current row and memory, one observed rate vector and clock, frequency, scales, gain, memory time, interval, tolerances | Next normalized row and memory; no simulator parameters or target partials. |
| `direct_learner.complete` | Frozen weights/frequency, clock, supplied rates, computed-channel index, guard | Conditioned homogeneous readout and validity flag; invalid queries cause task abortion in the runner. |
| `run.predict` | Frozen learned row, observation clock, prescribed input and initial predicted current | Independent predicted-current prefix and solver/guard status; its API receives no truth trajectory. |
| `run.control` | Frozen learned row, reference protocol, own current observations; separate simulator parameters for plant propagation | Own-trajectory currents, requested/applied voltages, guards, saturation and task status. |
| `run.stable_projector_error` / `phase_diagnostics` | Learned/evaluator rows / frozen learned coefficients | Evaluator-only projector error / grid and conservative conditioning bounds. |
| `run.run` / `execute` | New output directory, seed, declared numerical options | Source snapshots, raw records, final metrics and hashes; orchestration keeps learner and evaluator inputs separate. |
| `audit.audit` | Saved run; optional evaluator-only oracle diagnostic | Independent hashes, physical/readout/metric checks, full memory reconstruction and selected nonlinear learning-step checks. |
| `summarize_plot.main` | Saved seed 10--19 directories, new output directory | Cohort aggregate, fixed-seed illustration, figure/source-data hashes; no new simulations. |
| `plot_figure_focus.main` | The same saved seed 10--19 directories, new output directory | Four main evidence panels and three supplementary diagnostics; all four entries share one fixed evaluator-only scalar/sign across the original final-second window. Original plotter and raw records remain unchanged. |

## What is learned

For the physical voltage balance
\(u=L\dot i+Ri+e(t)\), introduce \(\dot q=i\) and \(\dot U=u\).
The integrated physical relation has context \(z=(i,q,U,t)\) and true row
\(F=(L,R,-1,e(t))\). Its observed rate vector is
\(v=(\dot i,i,u,1)\): the explicit clock derivative is retained.

Let \(D=\operatorname{diag}(20,2,5,1)\), \(r=D^{-1}v\), and
\(h=(1,\cos\hat\omega t,\sin\hat\omega t,
\cos2\hat\omega t,\sin2\hat\omega t)^T\).
The vector \(c\in\mathbb R^{20}\) is stored in harmonic-major order, with
four entries per harmonic. The evaluated row is
\(H_j(t)=\sum_k c_{k,j}h_k(t)\), so
\(\xi=H(t)r=c^T(h\otimes r)\). The physical-coordinate estimate is
\(\widehat F=HD^{-1}\). All twenty weights adapt; normalization couples
their norm but no physical entry or harmonic coefficient is fixed to truth.
The matched \(K=0\) learner has four adaptive entries and receives the same
observations. Neither learner differentiates a fitted physical residual.

The estimated scalar frequency comes from a constant-voltage current prefix.
Harmonic amplitudes fitted during this calibration are nuisance fit quantities,
not initial partials or learning targets. The true frequency and phase remain
simulator/evaluator quantities. Fourier functions carry temporal variation of
the entries; they do not replace the clock-rate input. No derivative of the
basis is added here because it represents differential entries themselves.

The target \(F\) is an integrable physical Jacobian. Arbitrary candidate
\(HD^{-1}\) rows need not be ambient Jacobians: the learning rule does not
enforce cross-partial identities. Row-space accuracy and conditioned readouts
are assessed up to nonzero scalar gauge. The representation is learned from
ideal rate observations, not true partial-derivative targets.

## Why the numerical update implements the stated memory law

Write \(w=h\otimes r\) and use the unit column \(c\). The implemented law is

\[
\dot R=ww^T-R/\tau,\qquad
\dot c=-\eta\{Rc-(c^TRc)c\},\qquad \|c\|=1.
\]

For held observation \(w_n\), the exact memory within an interval is

\[
R_n(s)=e^{-s/\tau}R_n+
\tau(1-e^{-s/\tau})w_nw_n^T,\qquad 0\le s\le\Delta t.
\]

`memory_step` solves \(\dot a=-\eta R_n(s)a\) from the current unit row and
returns \(c=a/\|a\|\). Differentiating the normalization gives

\[
\dot c=\frac{\dot a}{\|a\|}
-\frac{a(a^T\dot a)}{\|a\|^3}
=-\eta Rc+\eta(c^TRc)c.
\]

Thus the auxiliary linear solve and normalization implement the same nonlinear
unit-row tangent flow, to the stated numerical solver tolerance. The memory is
evaluated at every DOP853 stage. This is the correlation-memory experiment
variant; equivalence of its numerical integration does not transfer every
instantaneous-data convergence theorem to the changing memory.

Every interval-end memory is reconstructable from zero initialization and the
saved times/rates using the exact recurrence above. `training.npz` stores
`C_pre`, `C0_pre`, final coefficients, and `R_final`/`R0_final`; it does **not**
store a complete memory trajectory or internal solver stages. Here `R0_final`
means the final \(K=0\) memory. `C_pre[n]` precedes assimilation of sample
\(n\); the last saved sample is not assimilated. Pre-update energy is
\(\frac12(c_n^Tw_n)^2\). Failure update counts and final states preserve
which updates completed. The audit reconstructs all memory steps and checks
selected coefficient steps by independently integrating the projected law.

## Conditioned reuse and scope

For supplied rates with computed slot \(j\) set to zero, `complete` evaluates
\(v_j=-D_{jj}\sum_{k\ne j}H_kv_k/D_{kk}\,/H_j\), provided
\(|H_j|/\|H\|\ge0.05\) and the row norm is finite and nonzero. Prediction
computes current rate; control computes voltage. A failed query aborts the
task, preserves its prefix, and leaves its full-horizon RMSE null. Independent
prediction never assimilates later true current. Control uses its own current
trajectory and advances the plant only after selecting a held command.

This is finite-horizon electrical simulation under externally maintained
periodic motion with ideal sensing. The prescribed voltage is slower than the
periodic clock partial; current/rate observations can still contain fast
responses. The experiment does not establish hardware robustness, changing
frequency tracking, or mechanical motion control. Historical spatial-Fourier
and coefficient-estimator records remain separate and do not contribute to
this cohort. These documents make no claim of published artifact availability.
