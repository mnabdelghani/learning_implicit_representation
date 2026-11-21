Direct representation-learning reproducibility package

This directory is self-contained scientific code and compact evidence for the
three revised main examples: Fourier graph prediction, nonlinear Contact
regulation, and Burgers distributed readout. It contains no manuscript source
or manuscript PDF. Included figure PDFs are standalone outputs from these code
studies. The historical residual-coefficient experiments are separate protocols.

Layout
  fourier/: exact executed runner, independent raw audit, plotter, protocol,
            aggregate summaries and included figure.
  contact/: exact executed runner and direct matrix learner, cohort driver,
            raw audit, plotter, fixed protocol, compact summaries and figures.
  burgers/: portable extraction of the executed direct-row code, independent
            raw audit, compact full-sample plotting data, selected original
            chunks/checkpoints, protocol, metrics and figures.
  source_identity.json: executed-source identity checks; Burgers has its own
            detailed scientific-symbol and packaging-change map.
  manifest.json: SHA-256 and byte count for every included source/data/figure.
  verify_bundle.py: hash verification and optional focused numerical checks.

Requirements and environment
The studies ran with Python 3.12.3, NumPy 2.3.5, SciPy 1.17.0 and Matplotlib 3.10.8, as recorded in the
per-run environment files. Use a prepared environment,
not an unrecorded dependency upgrade. No script installs dependencies or uses
network services. Burgers' optional replay/full execution compiles its unchanged
C++ kernel with /usr/bin/g++, -O3 -std=c++17 -shared -fPIC -ffp-contract=off;
no prebuilt architecture-specific binary is distributed. Contact's combined
figure also requires pdflatex with the standalone class and pdftoppm. Individual
numerical checks need Python/NumPy/SciPy and the Burgers compiler only.

Quick integrity and portability check, from a fresh writable copy:
  python direct_submission/verify_bundle.py
  OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python direct_submission/verify_bundle.py --smoke
The smoke check does not run a full experiment. It checks normalized coupled
ODEs and same-field readouts, then exactly replays 600 archived Burgers samples.
Burgers builds a local binary/cache in that copy. Rechecking original hashes is
still valid: generated cache/build files are not original package members.

Raw records
Full records are provided separately as sibling files in raw_data/:
  fourier_raw.tar.gz, contact_raw.tar.gz, burgers_raw.tar
These are separate
artifacts, not contained in the code ZIP. Verify their supplied archive checksum
before extraction using raw_data/raw_data_manifest.json, which also indexes
every archived member. Extract into a NEW writable working directory and retain
the original archives unchanged. The fixed top-level directory names are:
  fourier/seed_10...seed_19, pilot_9001 and both refine_clean_* cases (13 total);
  contact/confirm_seed101...105, refine_seed101 and retained pilots (9 total);
  burgers/ideal_s732_dt0p005 and the other 18 declared cases (no runs/ level).
For example, from the directory holding direct_submission/ and raw_data/:
  mkdir reproduced_from_archives
  tar -xf raw_data/fourier_raw.tar.gz -C reproduced_from_archives
  tar -xf raw_data/contact_raw.tar.gz -C reproduced_from_archives
  tar -xf raw_data/burgers_raw.tar -C reproduced_from_archives
The WORK_CONTACT and WORK_BURGERS paths below are respectively
reproduced_from_archives/contact and reproduced_from_archives/burgers.
The source folder names in old environment/traceback metadata are historical
provenance, not runtime dependencies. No private workspace path is required.

Regenerate figures and audit existing records (no retraining)
Fourier's unchanged postprocessors expect records beside their source. In a new
working copy of the extracted Fourier case-directory parent, copy run.py,
protocol.json, audit.py and summarize_plot.py from direct_submission/fourier/.
From that working directory run:
  python audit.py
  python summarize_plot.py
The plotter requires a NEW figures/ directory; do not copy the included_figures
folder to that name. Audit/aggregate files are regenerated in the working copy.

For Contact, use the extracted Contact case-directory parent:
  python direct_submission/contact/audit_contact.py --records WORK_CONTACT
  python direct_submission/contact/report_contact.py --records WORK_CONTACT
These write audit/report/figure outputs into that working copy. They must not
be pointed at sealed originals whose postprocessing artifacts should remain
unchanged. The independent review summaries included in contact/data/ describe
the broader full-transition and literal joint-ODE audit performed for submission.

For Burgers, use the extracted directory containing the 19 case directories:
  python direct_submission/burgers/audit_runs.py --runs WORK_BURGERS --output NEW_BURGERS_AUDIT
  python direct_submission/burgers/plot_main.py --output NEW_BURGERS_FIGURES
The plotter uses the packaged compact data and needs no full raw archive. The
raw audit reads all attempted observations and writes to a new destination.

Reproduce the full simulation records
Each study's README/REPRODUCE file lists exact commands. All simulation output
destinations must be new. Fourier runs seeds 10--19 plus the separate pilot and
two clean refinements. Contact's reproduce_contact.py runs the five confirmatory
seeds and one refinement (its three pilots remain separate archived evidence).
Burgers' burgers_direct.py --all runs all 19 cases, including causal sensing
failures and refinements, with its declared computational limits. For example:
  python direct_submission/contact/reproduce_contact.py --output NEW_CONTACT --workers 2
  OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python direct_submission/burgers/burgers_direct.py --all --output NEW_BURGERS
Never replace a failed seed, extend training conditionally, or substitute a
refined run for its original main-cohort identity. Reruns create new records;
archive hashes attest original bytes, not equivalence across arbitrary software
versions or physical validation.

Scientific boundaries retained in this package
Every field entry is adapted from observed incompatibility; no physical residual
coefficient family is fitted and differentiated. Fourier supplies phase/order and
independent harmonic weights, and assesses a separate prediction trajectory.
Contact and Burgers assume ideal rate sensing in their successful main cases.
Contact shows approximate row-space recovery and uses a same-stream frozen
initial shadow, not a separate frozen controller. Burgers includes only 2/5
strict pre-control qualifications but 5/5 successful late adaptive cases;
causal/noisy failures and timestep sensitivities remain reported. Evaluator-only
physical fields or gauge alignment never supply learning/controller targets.
Successful task readout is not a global identification or stability theorem.

The Fourier row-norm diagnostic is included in fourier/.

Fourier guard qualification
Training normalizes the coefficient matrix only. No all-phase row-norm guard
was enforced, and no readout task was performed during training. The final
frozen fields used for heldout readout have a certified all-phase output-entry
lower bound H_y > 0.619 for every confirmatory seed (minimum lower bound
0.6195546476735543). This final readout certificate does not certify training
row norms across all phases. row_norm_guard_diagnostic.json records a post-hoc
analysis of archived data; its training-phase numerical searches provide
feasible-phase upper bounds, not certified continuum minima. It changes neither
the executed source nor the predeclared protocol.

No upload, DOI, public release or journal submission is asserted by this bundle.
