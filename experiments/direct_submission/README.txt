Direct representation-learning reproducibility package

This directory contains the current electrical Fourier, nonlinear Contact and
Burgers protocols, plus the retained historical spatial Fourier graph. The new
electrical identity electrical_fourier_direct_20261121_v1 awaits independent
validation; no completed electrical outcome or public data release is asserted
here. Contact/Burgers retain their existing record identities. No manuscript
source or manuscript PDF is included. Current bindings belong to repository
files docs/DIRECT_MAIN_20261121.json and docs/ELECTRICAL_FOURIER_20261121.json.
The historical residual-coefficient experiments remain separate protocols.

Layout
  electrical_fourier/: new temporal winding runner, independent audit, plotter
            and protocol; validation/status are recorded in the current manifest.
  fourier/: retained historical spatial-graph runner, independent raw audit,
            plotter, protocol, aggregate summaries and original figure.
  contact/: exact executed runner and direct matrix learner, cohort driver,
            raw audit, plotter, fixed protocol, compact summaries and figures.
  burgers/: portable extraction of the executed direct-row code, independent
            raw audit, compact full-sample plotting data, selected original
            chunks/checkpoints, protocol, metrics and figures.
  source_identity.json: source identity checks for the matched package version;
            Burgers also has a scientific-symbol and packaging-change map.
  manifest.json: member hashes/byte counts for that package version. The new
            electrical files require the rebuilt package's own binding.
  verify_bundle.py: hash verification and optional focused numerical checks.

Requirements and environment
The retained November 20 studies recorded Python 3.12.3, NumPy 2.3.5, SciPy 1.17.0
and Matplotlib 3.10.8. Electrical runs must record their own actual environment.
Use a prepared environment,
not an unrecorded dependency upgrade. No script installs dependencies or uses
network services. Burgers' optional replay/full execution compiles its unchanged
C++ kernel with /usr/bin/g++, -O3 -std=c++17 -shared -fPIC -ffp-contract=off;
no prebuilt architecture-specific binary is distributed. Contact's combined
figure also requires pdflatex with the standalone class and pdftoppm. Individual
numerical checks need Python/NumPy/SciPy and the Burgers compiler only.

Package integrity and portability check, from a fresh writable copy with its matching manifest:
  python direct_submission/verify_bundle.py
  OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python direct_submission/verify_bundle.py --smoke
The existing smoke check does not validate the new electrical experiment. It
checks normalized coupled ODEs and same-field readouts, then exactly replays
600 archived Burgers samples. Electrical validation needs its separate audit.
Burgers builds a local binary/cache in that copy. Rechecking original hashes is
still valid: generated cache/build files are not original package members.

Current electrical protocol and commands
The learner estimates frequency using current observations in a 20-s constant-
voltage prefix; true frequency and phase remain simulator-only. It independently
adapts all 20 weights of a K=2 temporal field from ideal current-rate observations.
The integrated context is (i, integral i, integral u, t), with no fixed partial
entry. Frozen weights support separate 12-s current-prediction and current-control
tasks. Only the clock partial varies faster than applied training voltage;
current and its rate contain fast ripple. K=0 uses the same training stream and
its own task trajectories. Readout guard failure aborts the affected branch;
full-episode errors are undefined for aborted tasks, without fallback results.

From the repository root (all destinations NEW):
  python experiments/direct_submission/electrical_fourier/run.py --seed 10 --output NEW_ELECTRICAL/seed_10
Repeat for every seed 10--19, retaining illustration seed 10. Seed-10 numerical
refinements and independent postprocessing use:
  python experiments/direct_submission/electrical_fourier/run.py --seed 10 --learner-dt 0.0025 --output NEW_ELECTRICAL/refine_seed10_dt0025
  python experiments/direct_submission/electrical_fourier/run.py --seed 10 --plant-substep 0.00025 --output NEW_ELECTRICAL/refine_seed10_substep00025
  mkdir NEW_ELECTRICAL_AUDIT
  python experiments/direct_submission/electrical_fourier/audit.py --run NEW_ELECTRICAL/seed_10 --output NEW_ELECTRICAL_AUDIT/seed_10.json
  python experiments/direct_submission/electrical_fourier/summarize_plot.py --records NEW_ELECTRICAL --output NEW_ELECTRICAL_FIGURES
The auditor takes one case with --run and writes one JSON file with --output.
Repeat for all ten seeds and both refinements with distinct filenames; audit
the separate pilot under its own identity. Plotting still takes the records
root with --records. These commands do not claim completed cohort validation.
Use the sealed protocol for the full case list; pilot_9002 is separate from the
main cohort. The new electrical archive has top-level electrical_fourier/ with
seed_10...seed_19, pilot_9002, refine_seed10_dt0025 and
refine_seed10_substep00025. Earlier development pilots, calibration challenges
and their source bindings remain under electrical_fourier/exploratory/.
Use the current manifest for actual archive names/checksums and validation
status; do not infer a public download or a successful audit from these paths.

Retained November 20 raw records
The preceding package's full records are separate sibling files in raw_data/:
  fourier_raw.tar.gz, contact_raw.tar.gz, burgers_raw.tar
Here fourier_raw.tar.gz is the historical spatial graph, not electrical data;
contact/Burgers records remain current and unchanged. Repository manifests
DIRECT_MAIN_20261120.json and DIRECT_RAW_DATA_20261120.json retain their original
identities and checksums; they do not bind the new electrical study.
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
The historical graph's unchanged postprocessors expect records beside their
source. In a new working copy of its extracted case-directory parent, copy run.py,
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
destinations must be new. Historical graph Fourier runs seeds 10--19 plus its
separate pilot and two clean refinements, using its unchanged REPRODUCE.txt.
The electrical commands/case identities above are separate. Contact's
reproduce_contact.py runs the five confirmatory
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
Every field entry in these direct studies is adapted from observed
incompatibility; no physical residual coefficient family is fitted and
differentiated. The electrical protocol estimates period from prefix current
observations and supplies order K=2; its validation is pending. The historical
graph supplies phase 20x and order K=1 and evaluates a separate prediction
trajectory. These two evidence identities must not be interchanged.
Contact and Burgers assume ideal rate sensing in their successful main cases.
Contact shows approximate row-space recovery and uses a same-stream frozen
initial shadow, not a separate frozen controller. Burgers includes only 2/5
strict pre-control qualifications but 5/5 successful late adaptive cases;
causal/noisy failures and timestep sensitivities remain reported. Evaluator-only
physical fields or gauge alignment never supply learning/controller targets.
Successful task readout is not a global identification or stability theorem.

The historical graph row-norm diagnostic remains in fourier/. Its retained
figure is historical_direct_graph_fourier_summary.pdf in supplement/figures/.

Historical spatial-graph guard qualification (not electrical evidence)
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
