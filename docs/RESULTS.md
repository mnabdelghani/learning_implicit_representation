# Executed results

## Current direct-field main evidence

The current studies learn all field entries directly and reuse the same learned field for readout. All specified seeds are included; numerical results are scoped to the declared sensors, inputs, and protocols. Supplementary S20 and `experiments/direct_submission/` retain raw-record mappings and audits. No public raw-data download is implied by this local package.

| Study | Cohort and evaluation | Direct-field result | Limitation |
|---|---|---|---|
| Fourier monotone graph | Seeds 10–19; training `[200,240)` s and separate 30-s prediction | Row-projector error `(6.3912 ± 1.3368)e-5`; rate RMSE `(1.60965 ± .46632)e-5`; integrated-output RMSE `(9.24323 ± 6.70758)e-5` | Known harmonic phase/order, exact input/rate and initial output; generic intermediate field need not itself be a Jacobian |
| Contact | Seeds 101–105; late `[110,120)` s | Mean force RMSE `.121638` N; motion RMSE `6.81102e-5` m; row-projector error `.298169` | Ideal rates and maintained contact; useful regulation does not establish precise canonical material magnitudes |
| Burgers | Ideal seeds 732–736; late `[96,100)` s | Row-projector error `.0136576 ± .0006381`; adaptive task RMS `(8.57217 ± .18563)e-6`, frozen `(2.48110 ± 1.52371)e-5` | Only two seeds pass every strict training criterion; causal/noisy conditions fail; full-horizon error dominated by the shared transition |

Fourier’s same-stream K=0 rate RMSE is `.0354751 ± .0001978`; the frozen random K=1 prior has invalid queries in seven of ten heldout runs, retained as failures. All ten learned prediction trajectories complete. The actual supplied rate modulates at `.3` rad/s during training, while partial phase rate is `2.7–3.3` rad/s. This is a new graph benchmark and not the earlier periodic dynamic plant.

Contact uses all 24 entries and maintains adaptation through the change. Burgers learns all five entries at each spatial site and continues learning during control; ideal observations are distinguished from the failed causal/noisy cases. These results do not establish general noise robustness, identification without excitation, or stability.

## Retained coefficient-family results

The preceding revision's main Fourier experiment uses a coupled normalized
learner within a prescribed residual family; it does not meet the author's
subsequent requirement for unknown-partial-structure direct field learning. Over
`[60,80)` s on its ten own controlled trajectories, tracking RMSE is
0.002311168 ± 0.000028630 and input-aligned relative differential error is
0.005450814 ± 0.000158912. All seven homogeneous coefficients adapt.
See [`NORMALIZED_FOURIER_MAIN_20261119.json`](NORMALIZED_FOURIER_MAIN_20261119.json)
for exact aggregates and provenance, and
[`FIGURE_CODE_MAP.md`](FIGURE_CODE_MAP.md) for current figure assignments.

The table below preserves the **original structured-study protocol set**.
Its bounded Fourier, contact, and Burgers entries are retained historical studies. They do not supply the current direct main evidence. The Fourier protocols are distinct and their results must not be interchanged.

Ten noise seeds (10–19), all included. Standard deviations are sample standard deviations.
Contact uses [100,120) s, Fourier [60,80) s, Burgers [18,24) s.

| Study | Method | Task RMS (mean ± SD) | Mean scaled differential error |
|---|---|---:|---:|
| contact | learned | 0.003424863 ± 0.000181 | 0.000895904 ± 0.000433 |
| contact | frozen | 0.8554918 ± 0.000275 | 0.583578 ± 3.2e-08 |
| contact | oracle | 0.003234699 ± 0.000133 | 0 ± 0 |
| fourier | learned | 0.002298958 ± 3.12e-05 | 0.00687125 ± 0.000188 |
| fourier | K0 | 0.04792815 ± 1.13e-05 | 0.184933 ± 8.61e-06 |
| fourier | frozen | 0.06727334 ± 0 | 0.214235 ± 2.93e-17 |
| fourier | oracle | 0.001228617 ± 0 | 0 ± 0 |
| burgers | learned | 0.0001974733 ± 1.94e-06 | 0.000107351 ± 1.08e-05 |
| burgers | frozen | 0.01658457 ± 3.66e-18 | 0.158437 ± 2.93e-17 |
| burgers | oracle | 0.0001976558 ± 2.86e-20 | 0 ± 0 |
