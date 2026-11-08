# Offline initialization and online adaptation

The motivating workflow is to learn a useful representation from controlled prior interaction, verify its task readouts, and then allow continued adaptation during operation. Offline training does not, by itself, certify a safety-critical controller.

## 1. Choose and document the observation context

Decide which sensor channels, clocks, derivatives, integrals, and feature products are available. Store their units, scales, timestamps, calibration, and causal differentiation/integration windows. An unknown or drifting clock and an unknown offset are not repaired automatically by storing more correlations.

The representation may be underdetermined by a particular data stream. Record excitation diagnostics and readout-block conditioning, not just a small residual. Preserve the distinction between an evaluated context differential and its coefficient row in selected feature coordinates.

## 2. Learn on recorded data

The common learner accepts observed free feature channels and the supplied physical contribution. The joint implicit coefficient row is `(theta, 1)`. Samples may be read from a log at any computational speed; `dt` remains their physical sampling interval.

A minimal interface, with application-specific observations and declared constants, is:

```python
import sys
import numpy as np
sys.path.insert(0, "experiments")
from common import ImplicitLearner

learner = ImplicitLearner(
    initial=initial_coefficients,
    scales=coefficient_scales,
    lower=lower_bounds,
    upper=upper_bounds,
    dt=physical_sample_interval,
    memory=memory_time,
    steps=inner_steps,
)
for free_channels, supplied_channel in recorded_observations:
    diagnostic = learner.observe(free_channels, supplied_channel, adapt=True)

np.savez(
    "offline_model.npz",
    theta=learner.theta,
    scales=learner.scales,
    lower=learner.lower,
    upper=learner.upper,
    R=learner.R,
)
```

`recorded_observations` must be formed from the sensor protocol and feature definition; neither true physical coefficients nor partial labels belong in that input. This is an interface example, not an additional aircraft experiment.

For a genuinely fixed offline objective, accumulate the desired correlations once and then optimize against that frozen matrix. Repeatedly replaying samples through exponential forgetting changes their weighting and therefore changes the objective. Do not confuse solver iterations with new physical-time observations.

## 3. Test independent task readouts

Use held-out trajectories or queries. Verify initial conditions and residual-level calibration. A forward task and an inverse task may require different blocks to be well conditioned. For underactuation, only feasible inverse commands should be queried. Bounds and regularization should be explicit.

The retained Acrobot example is concrete: its five learned physical coefficients are used for forward acceleration, passive-joint-compatible inverse torque queries, and a local controller. The control episode starts close to upright; it is not swing-up. Its test queries are separate from its training episode.

## 4. Initialize the online model deliberately

Restore the saved coefficient coordinates into a learner configured with the same feature ordering, units, and scales. Decide whether to retain the old covariance. Retaining it is sensible only when the old observations still describe the intended model; a long memory can delay adaptation after a change. Clearing the covariance while keeping coefficients preserves the prior estimate but discards its old information statistic.

```python
saved = np.load("offline_model.npz")
if not np.array_equal(saved["scales"], learner.scales):
    raise ValueError("Feature scales differ from the saved representation")
learner.theta = saved["theta"].copy()
learner.R.fill(0.0)  # deliberate new information episode, retaining coefficients
```

The main paper instead starts with empty memory and imperfect coefficients so that learning during control is visible. It does not imply that this is the preferred deployment workflow.

## 5. Monitor adaptation without overclaiming safety

Score incompatibility before each update. Separately monitor signal activity, readout conditioning, model/partial diagnostics when ground truth is available in simulation, task error, command limits, and estimator feasibility. A controller can make a data stream uninformative; near-zero residual does not eliminate this issue.

Safety-critical operation needs additional evidence about operating envelopes, model mismatch, sensing failures, actuator limitations, transient behavior, and supervisory logic. No such certification is supplied by the manuscript's local readout bounds or illustrative simulations.
