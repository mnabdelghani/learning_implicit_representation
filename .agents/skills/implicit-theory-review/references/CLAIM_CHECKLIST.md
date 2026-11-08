# Focused mathematical checks

These are review questions, not substitute proofs or preapproved conclusions.
Use the current manuscript's notation and hypotheses.

## Representation and gauge

- Are variations tangents of admissible curves or arbitrary ambient perturbations?
- Does the context include derivative coordinates whose rates are higher derivatives?
- Is the family Jacobian-derived, a general rate-product field, or a feature row?
- Is the gauge pointwise, constant, or constrained by the selected feature family?
- Does a coordinate transformation change units or the row metric?
- Is rank fixed where a projector or pseudoinverse is differentiated?
- Is level calibration independent of differential identification?
- Does adaptation change the residual level even if the instantaneous differential
  is compatible? Is an extra parameter-derivative term being dropped?

## Learning and subspaces

- Is the Euclidean gradient projected using the stated metric and constraint?
- Does the flow preserve nondegeneracy/row orthogonality, or only decrease energy?
- Is descent claimed only for frozen data/objective?
- Are stationary eigen-subspaces that are not minimizing excluded by initialization?
- Are eigenvalues ordered consistently and tied boundary eigenspaces handled?
- Does the excitation restriction refer to the true tangent subspace, not merely
  positive eigenvalues in a noise-contaminated covariance?
- Does exact physical identification require compatibility for every sample with
  positive memory weight, including old exponentially weighted observations?
- Is empirical tracking error distinguished from physical-subspace bias?
- Are finite-step, constrained, sampled implementations analyzed on their own terms?

## Readouts

- Are supplied versus computed channels roles of the current query?
- Do the proposed channels respect kinematic dependencies and measurement timing?
- Is a readout exact/unique/nonunique/inconsistent, and under which rank condition?
- Does the metric change under a general nonsingular row transformation?
- Is the perturbation small enough to preserve the relevant singular-value bound?
- Does an alleged gauge-invariant estimate actually use an aligned orthonormal
  representation and fixed context units?
- Does a local rate estimate require integration, initial data, or level calibration?
- Are unmodeled internal dynamics or actuator saturation outside the conclusion?

## Fourier Memory

- Are stored coefficients constant/slow under a defined basis, or only claimed so?
- Is lifting a rate being confused with differentiating a harmonic-state product?
- Is the clock contribution included in each actual partial and feature derivative?
- Can phase-dependent gauge factors enlarge the coefficient covariance kernel?
- Does a row normalization guarantee only a coefficient norm, not evaluated rank?
- Is a finite phase-grid check being mislabeled a continuum guarantee?
- Are coefficient identification, physical row-space identification, and readout
  usefulness separate statements?
- Is a gradient-forcing bound being mislabeled a convergence/tracking theorem?
- Are truncation, memory drift, and initialization handled explicitly?
