---
name: implicit-manuscript-edit
description: "Edit the private implicit-learning LaTeX paper, improve its conceptual flow, keep every formal proof in the main text, compile it, and inspect the ten-page result. Use for requested paper edits; do not rewrite simulations or publish files."
---

# Main-paper editing and build

## Establish the current document

Read private AGENTS.md and the user's requested changes. Resolve the private
root from the actual workspace. Normally it contains `manuscript.tex`,
`experiments.tex`, `results_macros.tex`, `references.bib`, and `figures/`.
Read include directives and build configuration before editing.

Inspect Git status where applicable; a private directory may have its own local
history or none. Do not initialize repositories, create remotes, reset changes,
or overwrite unrelated edits without permission.

Record source/PDF timestamps or hashes. A precompiled PDF is not proof of the
current source's page count. Never rebuild LaTeX by OCR or by copying PDF text.

## Edit by scientific purpose

- Identify the claim or clarity problem. Preserve terminology and scope unless
  the user asks to revise them or the mathematics demonstrably requires it.
- Preserve the sequence: context/representation -> nondegenerate learning ->
  identification/tracking -> structured features -> query readouts -> Fourier
  Memory -> experiments. Improve transitions, not by silently rewriting the thesis.
- Use computed channels `v_C`; explain known history versus new prediction/control
  queries. Keep learning-as-regulation distinct from physical control.
- Keep all formal proofs complete in the main manuscript. An application of a
  result already proved in the main paper is acceptable when its hypotheses and
  substitution are explicit. A missing argument deferred to the supplement is not.
- Retain labels; use LaTeX references rather than hard-coded numbering. Update
  abstract/conclusion/captions when a conclusion changes.
- Do not change metric macros, recorded values, feature families, or experimental
  settings for an editorial task. Label any proposed new experiment separately.
- Check bibliography keys against actual sources. No fabricated reference entries.

Use `$implicit-theory-review` before changing a nontrivial theorem or proof.

## Ten-page compression priorities

First remove genuine repetition, shorten transitions, group related equations,
and improve figure placement/captions. Move only nonessential experimental detail
or supporting examples to the supplement with accurate cross-references.
Never move required proofs or assumptions; never silently reduce font size,
margins, or figure-label readability. Report a page overrun rather than corrupting
scientific content. The limit includes references.

## Isolated build

Inspect `build.sh`, local latexmk configuration, and external-command use. Prefer
an isolated output directory outside public Git. From the private source root:

```bash
mkdir -p ../.research_build/private
latexmk -pdf -interaction=nonstopmode -halt-on-error -file-line-error \
  -outdir=../.research_build/private manuscript.tex
pdfinfo ../.research_build/private/manuscript.pdf
```

A local dependency failure is not permission to install packages or enable
unrestricted shell escape. Report the missing requirement and ask for approval.
After a failed build, do not present an older PDF as newly generated.

Review the final log for undefined citations/references, missing files, font
substitutions, overfull boxes, and structural errors. Underfull warnings require
visual judgment rather than indiscriminate suppression.

## Inspect and report

Render pages with available tools, for example:

```bash
mkdir -p ../.research_build/private/render
pdftoppm -png -r 130 ../.research_build/private/manuscript.pdf \
  ../.research_build/private/render/page
```

Inspect all changed pages and their neighbors; inspect all pages before a delivery
claimed to be fully checked. Check proof completeness, equation boundaries,
clipped figures, small legends, broken glyphs, blank pages, and page count. If
image viewing is unavailable, say that visual inspection was not performed.

Provide the actual source/output paths, changed sections, page count, warnings,
and checks run. Keep all outputs private. Use `$implicit-research-handoff` to
record accepted work without exporting the manuscript to public Git.
