---
name: implicit-experiment-audit
description: "Audit or implement approved implicit-learning simulations and reproducibility checks for contact, Fourier Memory, Burgers, and supplementary examples. Use to trace data, partials, causal scoring, baselines, and metrics; do not invent results or rerun all experiments for prose-only edits."
---

# Numerical experiment integrity

## Understand before running

Read applicable AGENTS.md, `docs/REPRODUCE.md`, the relevant source/configuration,
and data manifest. Discover the actual CLI from source and `--help`; do not
invent flags. Confirm which results are available as raw traces versus summary
files, and which archived records are missing.

Identify whether the task is a read-only audit, a smoke test, a reproduction, a
new protocol, or an implementation repair. These have different acceptance
criteria. Numerical work must not be silently folded into an editorial change.

## Map information access

Write a small table listing simulator-only truth, learner observations, controller
observations, references, and evaluator-only signals. Trace their actual code
paths. Oracle branches may use truth; adaptive branches may not inherit it.

Reconstruct command, sensing, pre-update scoring, memory update, parameter update,
and plant-integration timing from the implementation. Do not assume every model
uses the same physical sensing order. "Pre-update" means scoring the specified
observation with the model before assimilating that observation, not before all
sensor access. Check loops and the unused final command separately.

## Check mathematics against code

- Derive the residual and actual context partials at fixed candidate parameters.
  A constant coefficient error is not an evaluated differential error.
- Test Jacobian actions against finite differences at multiple admissible states,
  including state/phase product terms and the clock column.
- Distinguish instantaneous differential incompatibility, complete-feature
  secants, and integral-window residuals. Check units and the factor one half.
- Use one current candidate over the retained history or sufficient statistics
  that genuinely represent it. Do not accumulate stale fitted residuals as the
  gradient of a new candidate.
- Check fixed physical coefficients, scales, feasible bounds, conditioning,
  saturation, maintained contact, numerical step size, and model-change windows.
- Separate row-space gauge error from entrywise error in a fixed convention.
- Fourier: same-stream shadow versus independent closed-loop K=0 branch.
- Contact: integral quadrature, kinematic/material updates, slope observations,
  and jump-crossing windows. Fitted s and r need not share a geometric ratio.
- Burgers: spatial operators, boundary conditions, aliasing choices, full-field
  sensing/actuation, held inputs, and noise used in feature differences.

## Execute without damaging the baseline

Inspect each script's output paths before execution. Some existing summarizers
and plotting scripts overwrite tracked files. Start a unique scratch directory
under ignored `runs/` (or outside the repository) and retain the immutable
baseline. Confirm ignore status before large outputs.

Record commit plus dirty diff/hash, invocation, Python/dependency versions,
configuration, seeds, durations, integration steps, observation intervals,
memory, bounds, and evaluation windows. A starter schema is available in
[RUN_RECORD_TEMPLATE.json](references/RUN_RECORD_TEMPLATE.json).

Run focused tests before expensive runs. Use the main and legacy suites when
appropriate:

```bash
python -m unittest discover -s tests -v
(cd examples/legacy_suite && python -m unittest discover -s tests -v)
```

Do not assert the expected number of tests; report what actually ran. Ask before
large/unbounded compute or changing the tested environment. Never run on a shared
server with unbounded parallelism.

## Audit metrics independently

Compute reported errors from raw records, independently of online summaries.
Check axes of aggregation, RMS versus mean norm, unit scales, endpoints, discarded
warm-up, sample standard deviation (`ddof=1`), seed inclusion, and failures.
Report lack of raw records as an audit limitation; summaries do not replace them.

Record incompatibility, feature activity/excitation diagnostics, differential
accuracy, task error, and control effort separately. Low activity can make energy
small without identification. Noisy covariance positivity is not physical PE.
A true-model residual on the same noisy samples is a reference, not necessarily
a universal lower bound.

Changed parameters/probing/noise/integrators/targets define a new protocol.
Rerun affected cases and preserve old records; never selectively replace favorable
seeds. Mark exploratory results as exploratory. Changed plots alone cannot change
what an experiment establishes.

Deliver the claim-to-code-to-data map, failures and assumptions, executed commands,
actual test/run counts, independently checked metrics, and whether any paper
numbers need an approved update. Use `$implicit-figure-audit` for presentation.
