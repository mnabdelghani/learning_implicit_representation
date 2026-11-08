---
name: implicit-sync-review
description: "Synchronize the private main paper, public supplement, code, figures, and reproducibility documentation after an accepted change. Check proof placement, source/PDF freshness, data availability, and publication boundaries. Prepare a reviewable patch without automatically staging, committing, or pushing."
---

# Cross-document consistency and publication review

## Inventory the actual versions

Read applicable instructions in both projects. Resolve the private main source,
public supplement source, included files, PDFs, and relevant experiment records.
Read source headers/includes and hash/timestamp information before deciding which
PDF is current. Inspect both staged and unstaged Git diffs and untracked files.
Do not overwrite pending author changes or reapply an old update archive blindly.

If the private manuscript is unavailable, public supplement maintenance can
continue, but main-paper consistency remains explicitly unverified. Do not create
a public build dependency on private files or copy them into the repository.

## Compare meaning, not just text

Make a consistency table: subject; authoritative source/label; supplement/code
location; mismatch; required correction; verification.

Check in particular:

- Every main theorem/proposition/lemma has a complete proof in the main paper.
  Remove obsolete claims that those proofs were moved to the supplement, but
  only after checking that the complete proofs really exist in the current source.
- Mathematical definitions, gauge conventions, units, computed-channel notation,
  fixed-parameter derivatives, clock terms, rank/excitation/locality assumptions,
  and claims about stability agree across prose, examples, and tests.
- The three main experiments, metrics, scoring windows, seeds, noise, model
  changes, baselines, and displayed panels match actual configurations/records.
- Distinguish current runs, reconstructed protocols, historical examples, and
  missing source data. Do not relabel archives as newly executed experiments.
- Cross-references use current section/figure labels and S-numbering as applicable.
  Updating section order can invalidate README and figure-map references too.
- Every claimed public archive/release/DOI is actually available at the named
  location and version. A local data ZIP is not a published release asset.
- Captions do not imply that a supplement contains a proof that is only private,
  or that public code requires an unpublished manuscript to run.

Keep useful extra explanatory examples; do not preserve duplicated proofs merely
to keep an old section number. Where stable external numbering is necessary,
update all dependent references deliberately.

## Make a bounded patch

For an authorized synchronization edit, preserve existing numerical data and
protocols. Separate editorial corrections from changes requiring a mathematical
review or a new experiment. Use `$implicit-theory-review` or
`$implicit-experiment-audit` rather than quietly changing a scientific conclusion.

Build both documents to isolated output directories. For the public supplement,
from repository root:

```bash
mkdir -p ../.research_build/supplement
(cd supplement && latexmk -pdf -interaction=nonstopmode -halt-on-error \
  -file-line-error -outdir=../../.research_build/supplement supplement.tex)
```

For the paper use `$implicit-manuscript-edit`. Inspect generated PDFs, references,
figure layout, and main page count. Run relevant tests for code or assertion
changes; do not claim rerun results for unchanged simulations.

## Publication boundary review

Review locally before any approval to publish:

```bash
git status --short
git diff --stat
git diff --check
git diff --cached --stat
git diff --cached --name-only
```

These are diagnostics, not a complete leak detector. Inspect contents as well.
No main-paper LaTeX/PDF/source bundle, private review, external private symlink,
credential, or unapproved large record belongs in the public change set. A
tracked `.gitignore` rule does not remove an already tracked sensitive file.

Do not automatically stage, commit, push, tag, or upload. Present explicit proposed
paths and wait for approval. If an exposure is discovered, report it; do not
force-rewrite history or manipulate credentials as an unsolicited cleanup.

Deliver the consistency table, changed files, test/build outcomes, which PDF
matches which source, remaining unverified availability claims, and actual Git
publication status. Record private details in the local handoff, not public docs.
