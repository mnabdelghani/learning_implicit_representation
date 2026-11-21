# Reproduction instructions

The repository deliberately omits both the PDF and LaTeX of the definitive 10-page manuscript. The [focused supplement](../supplement/supplement.pdf) and separate [technical archive](../supplement/technical_archive.pdf) are self-contained and do not require the main-paper source or auxiliary files. The focused supplement uses S1 for its guide, S2 for current direct-study protocols, and S3 for trajectory variations. The technical archive preserves the preceding S1–S20 numbering, including the historical constructions, Acrobot/flow examples, and flow Figures S8/S9.

## Current direct-field main studies

The current study directories are `experiments/direct_submission/{electrical_fourier,contact,burgers}/`. The new electrical identity is `electrical_fourier_direct_20261121_v1`; independent validation and publication are pending. Its current source/record bindings belong to `docs/DIRECT_MAIN_20261121.json` and `docs/ELECTRICAL_FOURIER_20261121.json`. Match the focused supplement and figure to those bindings before treating them as the electrical protocol. Contact and Burgers keep their existing record identities. Old `results/` traces and the historical graph commands below do not reproduce the electrical study.

The electrical learner estimates frequency from current observations during a 20-s constant-voltage prefix. True frequency/phase remain simulator-only; phase is represented by learned sine/cosine weights. It adapts all 20 independent weights of a `K=2` temporal field using ideal current-rate sensing, then freezes them for separate 12-s current-prediction and current-control tasks. Only the clock partial is fast relative to the supplied training voltage; current and its rate have fast ripple.

From the repository root, use new output directories:

```bash
python experiments/direct_submission/electrical_fourier/run.py --seed 10 --output NEW_ELECTRICAL/seed_10
```

Repeat for every specified seed 10–19, changing both seed and destination. Seed 10 is the illustration seed. Keep the separate pilot `pilot_9002` and the two seed-10 refinements distinct from the main cohort:

```bash
python experiments/direct_submission/electrical_fourier/run.py --seed 10 --learner-dt 0.0025 --output NEW_ELECTRICAL/refine_seed10_dt0025
python experiments/direct_submission/electrical_fourier/run.py --seed 10 --plant-substep 0.00025 --output NEW_ELECTRICAL/refine_seed10_substep00025
mkdir NEW_ELECTRICAL_AUDIT
python experiments/direct_submission/electrical_fourier/audit.py --run NEW_ELECTRICAL/seed_10 --output NEW_ELECTRICAL_AUDIT/seed_10.json
python experiments/direct_submission/electrical_fourier/summarize_plot.py --records NEW_ELECTRICAL --output NEW_ELECTRICAL_FIGURES
```

The auditor takes one case directory with `--run` and writes one JSON file with `--output`. Repeat it for all ten seeds and both refinements, using a distinct output filename each time; audit the separate pilot under its own identity. The plotter instead takes the full records root with `--records`. The refinement flags are for the specified seed-10 numerical checks, not replacements for main-cohort runs. These commands describe the intended interfaces; no successful run or audit is claimed here. Follow the sealed protocol for the complete case list. Retain every aborted query branch; its full-episode RMSE is invalid, rather than a mean over selected valid samples. The `K=0` comparator uses the same training observations but its own task trajectories. Complete electrical records retain calibration observations/candidates, pre-update weights/memory, evaluator-only truth, task traces, commands, failures and source/environment identities. Earlier development pilots and calibration challenges are retained separately under `electrical_fourier/exploratory/` in the electrical raw archive; they are not confirmation cases.

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

The preceding spatial-graph Fourier protocol remains in `experiments/direct_submission/fourier/` with identity `direct_fourier_monotone_20261120_v1`. It supplies phase `20x`, learns six weights and predicts a separate 30-s graph trajectory; it is not temporal electrical evidence. Copy its unchanged `run.py`, `protocol.json`, `audit.py`, and `summarize_plot.py` into a new working directory and follow its `REPRODUCE.txt`. That file retains all ten graph confirmation seeds, the excluded pilot, two clean refinements, and the original reproduction commands. Its auditors/postprocessors expect records beside their copied sources:

```bash
python audit.py
python summarize_plot.py
```

These two commands run only from that historical graph working copy. The plotter needs a new `figures/` directory. Full graph arrays retain pre-update coefficient/covariance states and heldout predictions; `aggregate.json` is not a substitute for them. The retained graph figure is `historical_direct_graph_fourier_summary.pdf`. `docs/DIRECT_MAIN_20261120.json` and `docs/DIRECT_RAW_DATA_20261120.json` remain immutable bindings for this graph plus the unchanged contact/Burgers records, not for the new electrical data.

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

Compile the focused supplement using `bash scripts/build_supplement.sh` (equivalently, `bash scripts/build_supplement.sh supplement`); this updates `supplement/supplement.pdf`. Run `bash scripts/build_supplement.sh technical_archive` to build `supplement/technical_archive.tex` into its separate PDF and auxiliary files. The script accepts only those two target names and requires TeX Live including IEEEtran and AMS packages. It runs BibTeX only when the generated auxiliary file declares bibliography data: the archive requires BibTeX, and the focused supplement does not. Both builds use public repository files only. Figures and tables under `supplement/` serve both documents. Historical figures retain their audited identities; the electrical replacement requires its own completed audit and current manifest binding. The historical scripts `experiments/summarize.py`, `experiments/make_figures.py`, and `experiments/plot_excitation.py` contain fixed output paths; inspect them before use because they can replace tracked artifacts.

The complete-record archives named in `docs/DATA_MANIFEST.json` are not included in this repository, and the manifest does not provide download locations. If an archive is available, verify its SHA-256 checksum before extraction. The default source bundle includes selected seed-10 traces and summary JSON files for every original run; these support the original structured-study plots and aggregate tables, but independent auditing of every run from raw records requires the corresponding archive or a documented rerun. The historical periodic-plant Fourier panels require the separate normalized records; current direct main panels use the separately identified direct-study records described above. A rerun creates new records and should first use a separate output directory for comparison with the saved summaries.

A command or trace generated after changing initial conditions, probing, noise, integrator settings or feature families is a new experiment and must not be substituted for the paper's reported results.

The completed reproduction is described in `docs/REPRODUCTION_20261118.json`. Its new archive names contain `Reproduced_20261118`; their checksums do not replace the checksums of the missing historical archives. Diagnostic CSVs are not complete historical sensor logs: configurations, source, seeds, and PCG64 states specify the sensing sequence, and Burgers retains field snapshots every 0.1 s. No publication or external download location is implied by the local archive manifest.
