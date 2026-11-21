# Reproduction instructions

The repository deliberately omits both the PDF and LaTeX of the definitive 10-page manuscript. The [focused supplement](../supplement/supplement.pdf) and separate [technical archive](../supplement/technical_archive.pdf) are self-contained and do not require the main-paper source or auxiliary files. The focused supplement uses S1 for its guide, S2 for current direct-study protocols, and S3 for trajectory variations. The technical archive preserves the preceding S1–S20 numbering, including the historical constructions, Acrobot/flow examples, and flow Figures S8/S9.

## Current direct-field main studies

The current sources are in `experiments/direct_submission/{fourier,contact,burgers}/`. Focused Supplement S2 gives their complete protocols, failed cases, information boundaries, and numerical refinements. Each study has a separate identity; old `results/` traces and old coefficient-family commands below do not reproduce these main figures.

For Fourier, copy `run.py`, `protocol.json`, `audit.py`, and `summarize_plot.py` from `experiments/direct_submission/fourier/` into a new working directory and follow its `REPRODUCE.txt`. It lists all ten confirmation seeds, the excluded pilot, and two clean refinements, followed by the independent audit and plotting commands. Every output directory must be new. The full raw arrays include pre-update coefficient/covariance states and heldout predictions; `aggregate.json` is a summary, not a replacement for them.

For contact, follow the portable protocol and reproduction instructions in `experiments/direct_submission/contact/`. Keep all five seeds 101–105 and the declared pilots/refinement separate. The main figure uses seed 101. Ideal derivative sensing and continuing excitation are part of the protocol; the old `experiments/contact.py` is a different coefficient learner.

For Burgers, run from `experiments/direct_submission/burgers/`:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python verify_package.py
python plot_main.py --output newly_generated_figures
```

The first command performs the packaged replay/kernel checks; it is not a new 19-case scientific run. `README.txt` lists single-case and complete-study reproduction commands, runtime/storage expectations, and required installed compiler. All ideal, causal-clean, causal-noisy, and refinement conditions remain included. Compact plotting records, full raw subsets, and the separate complete raw archive have different roles. A new native build records its own binary identity.

Keep current local source/record hashes with the accompanying package. A locally assembled bundle is not a published release, and no remote download is asserted here. Record the actual environment for every reproduction; do not overwrite immutable records.

## Retained historical protocols

The protocols in this section support the technical archive; their historical section and figure references use that document's preserved numbering.

The historical runs recorded Python 3.13.5. The November 18 reproduction used Python 3.12.3 and the same NumPy, SciPy, and Matplotlib versions in `requirements-tested.txt`; all historical summaries and available raw arrays agreed numerically. Record the interpreter and package versions for any new execution. Run the main and legacy checks:

```bash
python -m unittest discover -s tests -v
(cd examples/legacy_suite && python -m unittest discover -s tests -v)
```

Reproduce all 100 original structured-study and 20 paired-transfer runs in a fresh directory, then independently audit every regenerated record against the historical summaries:

```bash
python experiments/reproduce_records.py --output runs/reproduced_20261118 --workers 2
python experiments/audit_reproduced_records.py --output runs/reproduced_20261118
python experiments/derivative_tracking_audit.py --records runs/reproduced_20261118/records --output runs/derivative_tracking_20261118
```

The runner refuses an existing output directory, snapshots the unchanged simulation modules/configurations, fixes numerical-library threading, and records all jobs and failures. It does not overwrite `results/`. The independent audit regenerates aggregate statistics without invoking the historical summarizers. The learned-partial diagnostic excludes exactly supplied derivative entries from the error denominator. Contact and Burgers diagnostics use saved partial snapshots; Fourier uses every recorded time step.

The additional estimator comparison has its own fixed protocol and output identity. It is not part of the historical reproduction:

```bash
python experiments/estimator_comparison.py --output runs/estimator_comparison_20261118 --protocol experiments/configs/estimator_comparison_20261118.json
python experiments/audit_estimator_comparison.py --input runs/estimator_comparison_20261118
python experiments/plot_estimator_comparison.py --input runs/estimator_comparison_20261118 --figures-output runs/comparison_figures --tables-output runs/comparison_tables.tex
```

The direct lifted-rate experiment also has a separate identity. Its postprocessor checks hashes, rebuilds all memory matrices, and independently checks selected exact-flow transitions using matrix exponentials:

```bash
python experiments/direct_lift_audit.py --output runs/direct_lift_20261118
python experiments/plot_direct_lift_audit.py --input runs/direct_lift_20261118 --output runs/direct_lift_analysis_20261118
```

The exact-Jacobian companion learns all four coefficients of an integrable family, so every candidate is a context Jacobian. It preserves the six-coefficient experiment as a separate protocol:

```bash
python experiments/direct_jacobian_audit.py --output runs/direct_jacobian_20261118
python experiments/audit_direct_jacobian.py --input runs/direct_jacobian_20261118 --output runs/direct_jacobian_analysis_20261118
```

Both direct experiments use a prescribed harmonic family and exact clock, with noisy second-rate measurements. They test derivative learning and recovery after a jump, without a controller or a claim of unknown-basis discovery.

The separate normalized Fourier plant experiment uses all seven adaptive coefficients and numerically integrates the coupled normalized learning and memory equations. It first replays the comparison's saved PG observation streams, then controls its own ten plant trajectories. Its bottom-eigenvector and frozen-prior traces are same-stream shadows. The physical family, zero-level calibration, phase, and sensor assumptions remain supplied. See `docs/NORMALIZED_FOURIER_20261118.txt` for the exact equations, scaling, timing, and readout guard.

```bash
python experiments/normalized_fourier.py --replay-root runs/estimator_comparison_20261118/common_stream --output runs/normalized_fourier_20261118
python experiments/audit_normalized_fourier.py --input runs/normalized_fourier_20261118 --output runs/normalized_fourier_audit_20261118
python experiments/report_normalized_fourier.py --input runs/normalized_fourier_20261118 --output runs/normalized_fourier_analysis_20261118
python experiments/table_normalized_fourier.py --audit runs/normalized_fourier_audit_20261118/audit.json --output runs/normalized_fourier_table.tex
```

The retained normalized Fourier protocol supplied a preceding periodic-plant figure; old contact and Burgers use structured coefficient updates. These are historical supporting studies. To regenerate the six historical Fourier panels and their current-rate consistency check **without rerunning any simulation**:

```bash
python experiments/plot_normalized_fourier_main.py --records runs/normalized_fourier_20261118 --output runs/normalized_fourier_main_20261119 --figures runs/normalized_fourier_main_figures
```

The analysis output must be new. Every panel uses the preselected seed-10 own trajectory; summary metrics use all ten seeds on `[60,80)` seconds. The script checks saved pre-update rows, all partials, secant energies, and inverse/forward consistency against retained raw records. It writes per-sample rate queries, source hashes, metric summaries, generated macros, and the six panels. The query accepts state, input, phase and learned coefficients, not measured or true rate. True rate is used for scoring. For the inverse-generated command, the forward completion equals the requested rate; this is achieved-rate/readout consistency, not independent prediction or a new control experiment. See `docs/NORMALIZED_FOURIER_MAIN_20261119.json` for the executed postprocessing identity.

The technical archive's unanchored Fourier figure is an exact copy of `examples/legacy_suite/results/fourier/fourier.pdf`, generated by its retained `simulations/fourier.py` reconstruction. It does not recover the old historical plot's unknown lineage. The old ellipse and unanchored assets remain archival files but are included in neither PDF; the archive's ellipse text is analytic only.

For the normalized simulation commands above, each output directory must be new. The auditor independently checks every normalized-flow transition and memory, with selected tighter coupled-ODE checks. The reporter replays the full seed-10 stream with tighter integration tolerances, calculates all-seed statistics, and generates review figures. This new protocol does not retroactively change the methods or statistics in the original structured-study records.
The executed local evidence archive, its verified checksum, and audit/refinement outcomes are described in `docs/NORMALIZED_FOURIER_20261118.json`. No public download or external data deposit is implied.

The estimator-comparison, direct-learning and learned-partial studies have a separate local archive described in `docs/SUPPLEMENTARY_STUDIES_20261118.json`; it does not contain the normalized Fourier records. Learned-partial postprocessing also requires the regenerated original-record archive. To reproduce the three presentation-only figure-label repairs, run `python experiments/repair_supplement_layout.py --output runs/layout_repair_20261118`; the script checks that plotted data and axis limits remain unchanged.

Compile the focused supplement using `bash scripts/build_supplement.sh` (equivalently, `bash scripts/build_supplement.sh supplement`); this updates `supplement/supplement.pdf`. Run `bash scripts/build_supplement.sh technical_archive` to build `supplement/technical_archive.tex` into its separate PDF and auxiliary files. The script accepts only those two target names and requires TeX Live including IEEEtran and AMS packages. It runs BibTeX only when the generated auxiliary file declares bibliography data: the archive requires BibTeX, and the focused supplement does not. Both builds use public repository files only. Figures and tables under `supplement/` serve both documents and were generated from the separately audited studies. The historical scripts `experiments/summarize.py`, `experiments/make_figures.py`, and `experiments/plot_excitation.py` contain fixed output paths; inspect them before use because they can replace tracked artifacts.

The complete-record archives named in `docs/DATA_MANIFEST.json` are not included in this repository, and the manifest does not provide download locations. If an archive is available, verify its SHA-256 checksum before extraction. The default source bundle includes selected seed-10 traces and summary JSON files for every original run; these support the original structured-study plots and aggregate tables, but independent auditing of every run from raw records requires the corresponding archive or a documented rerun. The historical periodic-plant Fourier panels require the separate normalized records; current direct main panels use the separately identified direct-study records described above. A rerun creates new records and should first use a separate output directory for comparison with the saved summaries.

A command or trace generated after changing initial conditions, probing, noise, integrator settings or feature families is a new experiment and must not be substituted for the paper's reported results.

The completed reproduction is described in `docs/REPRODUCTION_20261118.json`. Its new archive names contain `Reproduced_20261118`; their checksums do not replace the checksums of the missing historical archives. Diagnostic CSVs are not complete historical sensor logs: configurations, source, seeds, and PCG64 states specify the sensing sequence, and Burgers retains field snapshots every 0.1 s. No publication or external download location is implied by the local archive manifest.
