---
name: implicit-theory-review
description: "Audit definitions, theorem assumptions, proofs, gauge conventions, Fourier Memory, and readout claims in the implicit-learning project. Use for mathematical review or requested proof repairs, not routine formatting or numerical reruns."
---

# Mathematical review

## Start with evidence and scope

Read the applicable AGENTS.md. Locate the actual manuscript source and its
included files. Read the requested statement together with definitions, proof,
referenced results, and later uses. Do not treat a historical PDF or a handoff as
authoritative merely because its filename says "latest".

Default to a read-only audit when the request is to review. When the user asks
for a repair, explain the defect and make the smallest defensible correction.
Do not move any main-paper proof into the supplement.

## Produce a claim ledger

For each requested result record:

| Item | Required information |
|---|---|
| Identity | Source path, stable LaTeX label, current PDF page |
| Objects | Domains, shapes, regularity, rank, units, norms |
| Assumptions | Fixed or varying data; excitation; initialization; locality |
| Conclusion | Exact, approximate, asymptotic, finite-time, or empirical |
| Dependencies | Earlier lemmas/propositions and external primary sources |
| Proof status | Checked argument, gap, counterexample, or unresolved step |

Do not call a result "verified" solely because tests pass. Give an ordinary,
checkable mathematical derivation of the key steps and identify uncertainty.

## Audit in dependency order

1. Verify the definitions of context, admissible variation, residual, differential,
   parameters, and readout roles. A field may annihilate an observed trajectory
   without being the Jacobian of an underlying residual.
2. Check every dimension, transpose, norm, restriction to a subspace, and
   derivative. Specify which variables are held fixed. Include the parameter
   derivative when differentiating a residual whose parameters adapt in time.
3. Derive the projected flow independently. Separate fixed-sample descent from
   moving-data behavior, fixed covariance convergence from drift tracking, and
   empirical minimizers from physical identification.
4. Check equality and exceptional cases: zero data, a sample in the learned row
   space, orthogonal initialization, boundary eigenvalue ties, empty memory,
   rank deficiency, and degenerate readout blocks.
5. Check spectral ordering and the actual separating gap. A positive eigenvalue
   alone need not separate a noisy near-nullspace. Exact compatibility over a
   weighted history is stronger than approximate constancy over an informal window.
6. For moving targets, verify differentiability of the spectral projector,
   constants in the projector-error derivative, the local invariant-neighborhood
   condition, and behavior at zero error. Do not infer global contraction.
7. Check gauge claims against exact compatibility and inconsistent least squares
   separately. Declare context scaling, row weighting, and any alignment.
8. Check Fourier claims: fixed lifted coefficient space, complete product
   derivatives, phase-dependent gauge factors, pointwise rank, truncation, drift,
   and excitation. Empirical coefficient identifiability is not automatically
   physical differential recovery.
9. Check readout causality, solvability, calibration, and state consistency.
   Instantaneous completion bounds do not establish finite-horizon accuracy,
   closed-loop stability, or internal stability without further analysis.
10. Reconcile the abstract, introduction, conclusion, captions, and experiment
    interpretations with the actual proved conclusion.

See [the detailed checklist](references/CLAIM_CHECKLIST.md) when appropriate.

## Counterexamples and computational support

Search first for low-dimensional stationary or rank-deficient exceptions. Use
symbolic algebra or numerical integration to challenge a proposed identity when
useful, recording the setup. Such calculations can falsify a universal claim;
passing examples do not establish it. Keep scripts and detailed reviews private
unless their publication is separately authorized.

For cited external results, verify the exact hypothesis from a primary source.
Do not invent citations or replace the user's framework with a different method
without explaining the distinction.

## Repair and completion

For a genuine gap choose the narrowest justified remedy: repair an algebraic
step, supply an omitted hypothesis, weaken the conclusion transparently, or give
a counterexample and leave the result unresolved. Preserve a complete main-paper
proof and update dependent statements. Never bury a lost claim in an editorial edit.

Use `$implicit-manuscript-edit` to compile and inspect an approved patch and
`$implicit-sync-review` to identify dependent supplement updates.

Deliver findings ordered by scientific severity, with source labels, derivation
or counterexample, proposed correction, and downstream impact. End with exactly
what was checked analytically and computationally and what remains unverified.
