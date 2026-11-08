# Public repository: implicit representation research

This repository contains the supplement, experiments, results, tests, and
reproduction guidance for *Implicit Representation Estimation, Prediction, and
Control*. It is intended for public GitHub publication. The unpublished main
paper lives in the local sibling `../manuscript_private/` when available and
**must not be copied here**. A public clone must work without that sibling.

Read `docs/CODEX_WORKFLOW.md` for the workflow. Consult `docs/REPRODUCE.md`,
`docs/FIGURE_CODE_MAP.md`, and `docs/DATA_MANIFEST.json` for the particular task.
When available, `../notes/RESEARCH_CONTEXT.md` carries local scientific decisions.

## Scientific and publication agreements

- Develop *Implicit Representation Estimation, Prediction, and Control*. Preserve
  the author's concepts and notation unless a change is explicitly requested or
  a demonstrated error requires a clearly explained correction.
- The main manuscript targets **10 IEEEtran pages including references**, with
  normal class typography and margins. All formal theorem, proposition, and lemma
  statements and their complete proofs belong in the **main manuscript**. Do not
  remove assumptions, move proofs to the supplement, or shrink typography to meet
  the target. If correctness needs more space, report the conflict rather than
  silently weakening the mathematics.
- Keep the private manuscript, its PDFs, source bundles, reviews quoting unpublished
  text, and build artifacts out of the public repository, releases, issues, and
  pull-request descriptions. Do not initialize Git in the common workspace root.
- No commit, push, tag, release, external upload, dependency installation, destructive
  Git operation, or alteration of authentication/permissions without explicit
  approval. A request to edit files is not approval to publish them. Use targeted
  file staging only after approval; do not use `git add .` or force-push.
- A changing source file, a PDF, and a Git commit are different evidence. Establish
  which source produced a PDF before claiming they are synchronized. Preserve
  unrelated author edits and do not revert or stash them without approval.
- No separate affine-form presentation of the learning method. Retain supplied
  physical terms inside the implicit feature relation. This terminology preference
  does not permit hiding assumptions or changing mathematically equivalent code.
- Learning is incompatibility regulation through adjustment of the estimate.
  Do not reintroduce `dot(xi)=dot(Q)v+Q dot(v)` as a new learning law or an
  unexplained regulation rule.
- Distinguish `F(z)=D_z f(z)` from a direct rate-product field `H(z)` that need not
  be a Jacobian. Integrating a well-posed completion is not the same as proving
  existence of underlying level sets.
- Distinguish a coefficient row in features from the evaluated partials:
  `f_theta(z)=Q_theta zeta(z)` and `F_theta(z)=Q_theta D_z zeta(z)` at fixed
  parameters. Include every product and clock chain-rule term.
- Distinguish compatibility, identification up to gauge, readout accuracy, and
  closed-loop performance. Low incompatibility alone proves none of the latter
  three without additional assumptions. Numerical tests do not replace proofs.
- Nonsingular left transformations preserve exact compatible constraints, not
  generally inconsistent least-squares weighting. Compare physical entries only
  in a declared convention or alignment; use fixed context units for row-space
  comparisons. State rank and conditioning assumptions.
- Use `v_C` for channels **to be computed for a query**, not permanently unknown
  history. Available history through time t is distinct from predicted future
  states and new control commands. State the data-availability/update order.
- Fourier Memory is central. Separate an empirical bottom-eigenspace from true
  physical identification; retain compatibility and excitation hypotheses.
  Coefficient normalization alone is not a pointwise rank certificate. A harmonic
  lift of rates is not differentiation of complete harmonic-state products.
- Main experiments: nonlinear viscoelastic contact, online Fourier Memory, and
  controlled Burgers. Acrobot, active flow, and archival geometry are supporting
  examples. Do not silently alter this selection or their provenance.
- Every numerical claim must trace to executed code, configuration, seeds, saved
  records, and metric calculation. Never invent data, erase failed seeds, or alter
  records to match prose. New protocols create new experiment identities.
- Report estimated versus true evaluated partials, pre-update incompatibility,
  and task performance separately. Separate evaluator-only clean signals from
  learner/controller inputs. True-model energy is a reference, not a universal
  lower bound. An eigengap in noisy features is not proof of physical excitation.
- Do not claim formal verification, global stability, unrestricted equation
  discovery, flight safety, aerodynamic validation, a DOI, a license, or journal
  acceptance unless supported by specific evidence and author authorization.

## Working method

1. Read the applicable guidance and actual files. Establish task scope, current
   Git status, and the authoritative source/PDF pair. Read linked documents only
   as needed; do not ingest the entire repository by default.
2. For reviews, give findings before edits. For requested fixes, make a small,
   coherent patch and explain any changed claim or assumption. Separate proof
   defects, exposition problems, implementation bugs, and optional extensions.
3. Verify at the appropriate level: analytic proof checking, focused tests,
   numerical reruns, metric audit, LaTeX build, and visual inspection are different
   checks. Record commands, working directories, environment, and actual outcomes.
4. Use a new scratch run directory for exploratory simulations. Inspect scripts
   before executing them: current summarizers/plotters may overwrite tracked
   outputs. Never rerun all experiments merely to fix text or formatting.
5. For a build, use a separate output directory. Check both page count and layout.
   A successful compile does not establish scientific validity.
6. End with changed files, verification performed, unresolved limitations, and Git
   status. State explicitly whether anything was committed/pushed. Do not claim
   actions that were not executed.

## Skills

Invoke a relevant skill with `$implicit-theory-review`,
`$implicit-manuscript-edit`, `$implicit-experiment-audit`,
`$implicit-figure-audit`, `$implicit-sync-review`, or
`$implicit-research-handoff`. These are workflows, not permissions to publish.
If discovery fails, read the corresponding `SKILL.md` explicitly.

## Repository map and checks

- `experiments/`: current contact, Fourier, Burgers, excitation, plotting, and audit code.
- `examples/legacy_suite/`: separately documented earlier implementations and tests.
- `supplement/`: public supplementary document and its figures.
- `results/`: recorded data and summaries, not scratch outputs.
- `tests/`: regression tests; test counts are discovered by execution, not hard-coded.
- `runs/`: preferred scratch outputs after confirming they are ignored locally.

From the repository root, after inspecting the active environment:

```bash
python -m unittest discover -s tests -v
(cd examples/legacy_suite && python -m unittest discover -s tests -v)
git diff --check
```

Use existing dependencies; do not assume the Ubuntu interpreter matches the
recorded run environment. Read `requirements-tested.txt` before reproducing data.
Never change its versions simply to make installation succeed without recording
that environment change and evaluating its consequences.

Build the supplement outside the tracked directory:

```bash
mkdir -p ../.research_build/supplement
(cd supplement && latexmk -pdf -interaction=nonstopmode -halt-on-error \
  -file-line-error -outdir=../../.research_build/supplement supplement.tex)
```

The repository may track `supplement/supplement.pdf`; rebuilding that path changes
Git status. Do not restore/stash/overwrite it without inspecting why it changed.

## Code Review Rules

Flag evaluator-to-learner leakage, post-update scores labeled pre-update, omitted
clock derivatives, re-use of stale residuals, silent overwrites of published data,
inconsistent feature scaling, gauge-dependent comparisons labeled invariant, and
claims stronger than their assumptions. A green test suite does not excuse these.
Before any approved commit inspect staged **contents**, not only filenames, for
private manuscript material, credentials, unreviewed binaries, and large traces.
`AGENTS.md` and `.gitignore` are guidance, not technical prevention of disclosure.
