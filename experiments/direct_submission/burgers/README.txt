Direct five-entry Burgers learning and readout

This portable source package reuses the immutable study
burgers_direct_field_long_training_multiseed_causal_sensing_20261120_v1.
It is a packaging revision, not a newly executed scientific experiment.
The old coefficient-family Burgers example is a different algorithm.

Required installed tools: Python 3, NumPy, SciPy, Matplotlib, and /usr/bin/g++.
No network access or installation is performed by these scripts.
Run from this directory:

  OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python verify_package.py
  python plot_main.py --output newly_generated_figures
  OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python burgers_direct.py --case ideal_s732_dt0p005 --output reproduced_runs/ideal_s732_dt0p005
  python audit_runs.py --runs reproduced_runs --output newly_generated_audit

To reproduce the complete 19-case study, use --all instead of --case and a new
output directory. The study originally took approximately 23 minutes and saved
6.63 GB of records. Each run preserves every attempted sample, failure, guard,
clip, source, compiler/environment identity, and original fixed protocol. A new
build records its own native binary identity; it must not claim the historical
binary identity. Existing output directories are never overwritten.

The focused verification replays 600 complete archived samples (training,
adaptive, frozen), then compares the native update to the literal coupled ODE
and two successive step halvings. This numerical test does not replace the
historical whole-loop dt sensitivity study. The independent audit utility
imports no simulation runner or learner. Its scientific analysis functions
are unchanged from the historical audit; only file locations/CLI were adapted.

Mechanism: at each of 32 sites, z=(w,w_s,w_ss,u,w_t). Every entry of the unit
row q is independently learned; no physical coefficients or fixed partials
are supplied. Rdot=v v^T-R/4 and qdot=-10000(qR-(qRq^T)q), with v the causal
0.05 s context secant, sampled every 0.005 s. The same pre-update q produces
the control input-rate completion. The controller never calls physical F.
The declared ideal sensor supplies exact w and w_t; spatial derivatives are
spectral. The learner receives secants, not an exact w_tt sensor. Training lasts
80 s, then adaptive/frozen own-trajectory control lasts 20 s. The input guard
is |q_u|<0.1 (generic input decay), input-rate clipping is +/-5, and the physical
field-magnitude stop is 5. All guards/failures count as such.

The five ideal seeds (732--736) all meet late readout/recovery thresholds, but
only two meet every near-identification threshold at the training split.
Therefore these results demonstrate continued direct learning and readout,
not uniformly qualified tracking after convergence. Five causal-clean and
five causal-noisy trials fail the combined late criteria. Those results and
all four dt-refinement cases remain in data/all_conditions_audit_summary.json
and all_conditions_metrics.tsv. The causal cases do not establish robustness.

Data boundaries: ideal_plot_data.npz contains full-sample scalar trajectories
for every ideal seed and full-sample site-0 entry traces for preselected seed
732. It is a derived compact plotting dataset, not the complete original raw
archive. raw_subset and replay_seed732 contain named complete original chunks
and checkpoints; their roles are documented in source hashes. Full original
archives remain separate. All main metrics use all applied samples, no render
subsampling, and exclude terminal unapplied commands. Figure lines are cohort
means, bands are min/max across all five seeds; only rendering uses stride 5.
Entry curves have evaluator-only unit/sign alignment; no alignment enters
learning/control. Physical-row secant energy is nonzero because finite causal
secants need not lie exactly in the instantaneous physical tangent plane.

source_mapping.json maps extracted scientific symbols to the original source
hashes. data/raw_source_hashes.json binds the 610 raw ideal chunks used in the
independent final-pass metric reconstruction. package_sources.json binds the
portable sources. Historical records and their hashes have not been changed.
