# Supplement: additional rules

Read the repository `AGENTS.md`. This is a public supporting document, not a
second authoritative manuscript.

- Keep protocols, extra diagnostics, data provenance, and additional examples here.
  Every formal theorem/proposition/lemma proof used by the main paper must be
  complete in that paper, never deferred to this supplement.
- Before deleting old supporting proofs, verify the complete corresponding proof
  in the actual private source/PDF. If the private paper is unavailable, report
  the verification gap rather than assuming a matching version.
- Remove obsolete statements that proofs were relocated from the main paper.
  Check the abstract, section introductions, README, captions, and cross-references.
- Preserve useful explanatory counterexamples or computations that do not replace
  a missing main-paper proof. Do not move a new essential formal result here to
  save main-paper pages.
- Build independently of the private manuscript. Never require a private `.aux`,
  figure path, source include, or symlink to compile a public clone.
- Keep S-prefixed numbering and verify references after reorganization. Do not
  hard-code new main-paper theorem numbers without checking the matching version.
- Preserve provenance: contact is a disclosed reconstruction, Fourier uses an
  online protocol, the active-flow example is not validated aerodynamic evidence,
  and archival ellipse material lacks original executable records.
- Do not say a data archive, release, DOI, or license is available without checking
  the referenced artifact. A local ZIP is not a published GitHub release.
- Favor readable figures over compactness; this supplement has no fixed ten-page
  limit. Prevent empty headings, trailing float-only pages with wasted space, and
  tiny text in legacy multi-panel figures.
- Compile to the isolated build directory first. Replace the tracked supplement
  PDF only as an explicitly intended local output; publication still needs approval.
