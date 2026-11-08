# Experiments: additional rules

Read the repository `AGENTS.md`. Preserve the plant/learner/controller/evaluator
boundary in every implementation and test.

- Inspect source, configuration, and saved records before a rerun. Use separate
  scratch outputs and record commit, dirty diff, Python/dependency versions,
  invocation, seeds, integrator, sampling, memory, bounds, and scoring windows.
- Test analytic partials by directional finite differences at several admissible
  states, including clock coordinates. Respect feature/state constraints.
- Distinguish differential, integral-window, and complete-feature-secant residuals;
  do not label them interchangeable measurements of the raw-context `F dot(z)`.
- Keep current candidate parameters common across all retained observations.
  Use an explicit causal command/sensing/scoring/update/plant-advance ordering.
- Check that every plotted pre-update partial and energy uses the same intended
  parameter snapshot. Control observations must not leak future measurements.
- Verify rank/conditioning before inversions, report saturation and domain exits,
  and record all planned seeds and failures. Never discard hard cases silently.
- Protect frozen baselines from adaptation. A same-stream shadow ablation must
  use the same observations; a separate closed-loop baseline has its own trajectory.
- Keep clean diagnostic channels evaluator-only. No training on true coefficients,
  true partials, desired control targets, or oracle-only information.
- Do not run summarizers or plotters without checking their hard-coded output paths.
  The existing documentation contains commands that overwrite `results/`.
- Report exact-vs-discretized equations, held-input effects, noisy regressors,
  phase/clock assumptions, boundary conditions, and jump-crossing windows.
- Preserve material-change and phase protocols unless the user requests a new
  experiment. Give any changed protocol a new run identifier.
- Smoke tests do not reproduce the paper. Synthetic checks do not prove theorems.
