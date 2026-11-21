# Supplementary excitation-versus-task-sufficiency experiment

This retained structured-coefficient study is separate from the 100 historical runs and is documented in the [technical archive](../supplement/technical_archive.pdf), which preserves its preceding section and figure numbers. It tests whether a learner that fits observations generated during nearly stationary regulation can transfer its learned differential to a new task. The control protocol is fully specified in `experiments/contact_excitation.py`. The current main paper uses the direct-field experiments in `experiments/direct_submission/`, documented in [focused Supplement S2](../supplement/supplement.pdf).

**Shared conditions.** Same initial physical parameters, parameter prior, sensor-noise seeds 10–19, sample interval, learning rule, box bounds, controller gains and saturation limits. No material/actuator change. Training occurs on a stationary reference `x=0, L=4 N` for 60 seconds. The comparison is (i) task-only regulation and (ii) bounded deterministic command probing during training. At 60 seconds probing and adaptation stop, and the frozen model controls a new oscillatory motion/force reference for 20 seconds.

**Evaluation.** Training `[40,60)` s; transfer `[70,80)` s. The saved diagnostic includes the true and estimated full differential at the current simulator context, pre-update incompatibility, a clean evaluator-only information spectrum, and an independently sampled 1000-point differential test. No true partials or clean information are supplied to the learner.

**Observed outcome.** Task-only regulation achieves very low force error and measured incompatibility near the same-stream true-model value, but the differential remains inaccurate and the frozen model transfers poorly. Bounded probing improves differential recovery and transfer, at a cost to training-task regulation. The numerical results are descriptive for this model, not a general guarantee.

| Metric, mean across ten seeds | Task only | Bounded probing |
|---|---:|---:|
| Training force RMSE (N) | 0.002609 | 0.164071 |
| Training relative differential error | 0.192523 | 0.01904 |
| Transfer force RMSE (N) | 0.985581 | 0.065014 |
| Held-out relative differential error | 0.195960 | 0.042322 |

See `results/contact_excitation/aggregate.json` for exact means, sample SDs and all measured diagnostics. The study has a deterministic fixed probing signal; it is not a general safe-exploration policy. No flight or safety-critical claim is made.

```bash
python experiments/contact_excitation.py --seeds 10 11 12 13 14 15 16 17 18 19 --workers 2
python experiments/plot_excitation.py
```
