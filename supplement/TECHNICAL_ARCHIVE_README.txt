Technical Archive — 20 November 2026

The primary submission companion is supplement.pdf: a focused guide, the
complete current direct Fourier/contact/Burgers protocols, and the discussion
of time derivatives versus variations of complete trajectories (S1–S3).

technical_archive.pdf preserves the complete preceding 44-page supplement,
with an archival cover and the original S1–S20, equation, figure, and table
numbering. Its body is retained as historical documentation; older structured
studies do not demonstrate the current all-entry direct learning mechanism.
The current direct evidence remains fully present in focused Section S2,
including every sensing boundary, failed trial, guard, seed result, refinement,
and provenance qualification. Focused S3 preserves Archive S19 verbatim.

Selected reading map:
- Archive S2–S5: structured application updates and original protocols.
- Archive S9/S11: supporting constructions and separately documented examples.
- Archive S13–S18: historical learning-rule audits, comparisons and provenance.
- Archive S19 -> Focused S3: temporal derivatives and whole-trajectory variations.
- Archive S20 -> Focused S2: current direct-partial protocols and all limitations.
- Archive Figures S16–S19 -> Focused Figures S1–S4.
- Archive Tables S11–S15 -> Focused Tables S1–S5.

technical_archive.tex freezes the old local text includes into one source file.
It uses the existing public figure files and references.bib in this repository;
it has no dependency on the private main manuscript. All formal results and
complete proofs used by the main paper remain in that paper.

From the repository root, build independently with:
  bash scripts/build_supplement.sh
  bash scripts/build_supplement.sh technical_archive

Both commands update their corresponding PDF and auxiliary files under supplement/.
The delivered PDFs were built and reviewed separately in isolated directories. The focused supplement
has no bibliography; the archive uses the retained reference bibliography.
The PDF files in this directory are reviewed local outputs. Their presence
is not a claim of external publication, journal submission, or a data release.
