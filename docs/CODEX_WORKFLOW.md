# Research development with Codex

## Boundaries and sources of truth

`AGENTS.md` states durable agreements. `.agents/skills/` contains reusable task
workflows. Neither replaces source code, complete proofs, raw numerical records,
or a freshly compiled PDF. This public repository must remain usable without the
unpublished main paper; private local notes are not a dependency of public builds.

Main-paper source/PDF files stay in a sibling directory outside this repository.
Public instructions and skill definitions can be versioned here after review.
Codex must not assume that it automatically receives a ChatGPT conversation.
Transfer accepted decisions through an explicit handoff, then verify the files.

## Recommended sequence

1. Read-only baseline: status, sources, PDF freshness, pending edits.
2. Choose one claim or one experimental change, not an unrestricted rewrite.
3. Review or implement the smallest coherent change.
4. Verify analytically and computationally as appropriate.
5. Build and visually inspect the relevant PDFs.
6. Prepare a handoff/diff; publish only after explicit approval.

A proof review and a supplement synchronization are separate tasks. Do not alter
scientific claims just to eliminate a textual mismatch without identifying why.

## Skill invocation examples

- `$implicit-theory-review`: audit a theorem and dependent claims, first read-only.
- `$implicit-manuscript-edit`: apply an agreed manuscript edit and rebuild privately.
- `$implicit-experiment-audit`: trace learner inputs, partials, residuals, and metrics.
- `$implicit-figure-audit`: audit plots against recorded inputs and final PDF layout.
- `$implicit-sync-review`: compare paper/supplement/code and assess public boundaries.
- `$implicit-research-handoff`: record actual work and the next bounded task.

Skills are instructions, not a formal theorem prover or a numerical execution
service. An unavailable tool/data archive must be reported, not simulated in prose.

## Scratch output and builds

Use ignored `runs/` directories for exploratory numerical work, verifying ignore
status with `git check-ignore`. Do not overwrite recorded `results/` in a trial.
Review plotting/auditing script output paths before execution.
Build the supplement to a sibling `.research_build/supplement/` directory while
editing; copying its checked PDF into the tracked location is a separate step.
Do not change `.gitignore`, hooks, package versions, or VS Code settings silently.

## Tests and reporting

From repository root:

```bash
python -m unittest discover -s tests -v
(cd examples/legacy_suite && python -m unittest discover -s tests -v)
git diff --check
```

Actual pass/fail counts come from the current run. Record environment differences
from `requirements-tested.txt`. A smoke run is not a ten-seed reproduction. No
new release or DOI exists merely because documentation mentions one.

## Commit preparation

Review staged content, generated binaries, source/data hashes, and public/private
separation. Use explicit paths and approval before stage/commit/push. Do not
publish private reviews, file excerpts, or access tokens. Preserve all unrelated
work and avoid force operations.

## Keeping skills available across local projects

The kit stores canonical skills here under `.agents/skills/`. On Linux its optional
installer creates local symlinks from the sibling private/workspace skill folders
back to these public instruction folders, never in the reverse direction. Public
repository content therefore has no dependency on private files. Do not also
install duplicate global copies with identical skill names.
