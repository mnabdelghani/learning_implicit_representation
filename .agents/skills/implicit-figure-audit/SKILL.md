---
name: implicit-figure-audit
description: "Create or audit scientific figures for the implicit-learning paper using executed saved records. Check true and estimated partials, gauge conventions, incompatibility, control outcomes, preprocessing, and final PDF readability. Do not use illustrative or generated curves as experimental evidence."
---

# Scientific figures from recorded evidence

## Establish the figure's claim

Read applicable AGENTS.md, the caption, relevant manuscript discussion, plotting
script, configuration, and raw records. Use `docs/FIGURE_CODE_MAP.md` as an index,
not as proof that mappings are current. Inventory inputs and output paths before
running a plotter, especially scripts with hard-coded tracked destinations.

Classify each curve: truth, pre-update estimate, reference, independent baseline,
same-stream shadow, noisy observation, clean evaluator diagnostic, aggregate
metric, or theoretical expectation. Check units, sign, dimensions, and data timing.
No synthetic curves, image generators, manual path editing, or invented data may
be presented as numerical results.

## Ensure the three experimental questions remain separate

1. **Differential estimation:** compare evaluated partials at matched contexts and
   declared scales/gauge, not merely learned coefficients. Show the clock partial
   when it matters. A selected-site/selected-entry plot is not a whole-field audit.
2. **Compatibility:** show the actual pre-update residual definition and energy,
   relevant noise/discretization reference, and activity/excitation context.
   Distinguish integral observations from differential and secant observations.
3. **Task performance:** compare desired and realized outputs and report control
   effort/saturation. A successful output plot is not proof of internal stability.

## Guard against misleading presentation

- Verify the displayed seed was preselected and all declared seeds enter statistics.
- Match true and estimated partials at the same specified physical/evaluator context.
- Check plotted entries are informative; a nearly zero component can hide a defect.
  Changing sites or windows after viewing outcomes must be disclosed and justified.
- Do not confuse constant stored coefficients with rapidly varying evaluated partials.
- A same-stream Fourier K=0 shadow must share data with the K=1 learner; an
  independently controlled plant is a different comparison.
- Mark material/actuator changes consistently across panels and describe transients.
- Apply only declared preprocessing. Record window length in samples and seconds,
  causal versus centered smoothing, decimation, masks, startup behavior, and
  logarithmic rendering floors. Metrics must use the declared unsmoothed data.
- Never drop NaNs, failures, initial transients, or clipped commands invisibly.
- Use consistent axes where comparisons require them. Explain any log scale,
  normalization, selected time interval, or separately scaled panel.
- Preserve raw traces. Generate to scratch outputs for review before replacing
  published figures. Do not recompute simulations for a layout-only task.

## Generate and inspect

Prefer editable plotting code and vector PDFs with embedded readable labels.
Keep one data chart per figure file when following the current project convention;
LaTeX can assemble the multi-panel layout. Reuse the current visual conventions
unless explicitly redesigning them. Do not create a new dependency for styling.

Inspect both the standalone chart and its size in the compiled paper. Check
legend occlusion, true/estimated line distinction, clipping, fonts, margins, panel
labels, units, and caption references. A plot readable at full screen can fail
at one-column size. The supplement can use more space for legacy panels.

For each panel record source file/hash, seed/config, columns and transformations,
plot command, output file, and claim supported. Existing summaries alone cannot
reconstruct a missing time series; report missing data rather than inventing it.

Deliver a panel-by-panel audit, regenerated outputs only if requested, accurate
captions, and unresolved limitations. Run `$implicit-manuscript-edit` or
`$implicit-sync-review` for the document build; do not commit or push.
