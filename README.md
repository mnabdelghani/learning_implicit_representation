# Learning Implicit Representation

**Research code and supplementary material for _Implicit Representation Estimation, Prediction, and Control_.**

**Publication boundary:** The definitive manuscript source and PDF are deliberately excluded. This repository contains supporting protocols, code, figures, and reproducibility documentation. The supplement is not a replacement for the manuscript. A local accompanying package does not imply a published release or a public raw-data download.

The current study set learns differential entries from observed context-rate incompatibility and reuses the learned field for prediction or control. The portable sources are in [`experiments/direct_submission/`](experiments/direct_submission/). The temporal electrical Fourier study, `electrical_fourier_direct_20261121_v1`, passed independent numerical audits and declared criteria for all ten confirmation seeds and both refinements: 2,064 checks, no failures. It is a self-contained simulation with ideal current-rate sensing, not hardware validation. Contact and Burgers retain their existing evidence identities. Section S2 of the [focused supplement](supplement/supplement.pdf) gives the protocols and limitations. The preceding spatial Fourier graph and physical-coefficient studies remain separately identified historical evidence.

The focused supplement contains a guide in S1, the three current direct-study protocols in S2, and the time-derivative/whole-trajectory-variation discussion in S3. The separate [technical archive](supplement/technical_archive.pdf) preserves the preceding S1–S20 numbering, supporting constructions, historical comparisons, and provenance. Its original flow Figures S8 and S9 remain there.

## Current direct-field studies

- **Electrical Fourier:** an actuator winding uses the integrated context `(i, integral i, integral u, t)`. All 20 weights of a `K=2` temporal basis adapt independently across four partial entries. Frequency is estimated from current observations in a 20-s constant-voltage prefix; the learner receives neither true frequency nor true phase, and sine/cosine weights absorb phase. Ideal current-rate observations are supplied for learning. Frozen weights are tested on separate 12-s current-prediction and current-control trajectories, with a matched `K=0` comparison. Only the clock partial varies faster than the applied training voltage; current and its rate contain fast ripple. Seeds 10–19 and illustration seed 10 are specified in the protocol. Every learned prediction/control task completes without guard failure; all ten `K=0` predictions abort and retain undefined full-episode errors.
- **Contact:** all 24 entries of a three-row field adapt; five initializations perform force/motion readout through a material/actuation change. Ideal rate sensing and maintained contact are supplied. Useful regulation accompanies approximate row-space recovery; precise canonical material magnitudes are not established.
- **Burgers:** each of 32 spatial sites learns all five entries from causal context secants, with ideal state/rate sensing. Five seeds branch into adaptive and frozen own-trajectory control. Continued adaptation supports late readout; only two seeds satisfy every strict training criterion. Failed causal/noisy conditions and timestep sensitivity remain reported.

Compatibility, estimated partial relations and task performance are assessed separately. The electrical study's observation-based calibration assumes an observable stationary period within its search band; it is not unrestricted frequency discovery. These scoped protocols do not establish general integrability, broad noise robustness, global stability or physical safety.

## Included material

- [`supplement/supplement.pdf`](supplement/supplement.pdf): focused current-study protocols, limitations, and trajectory-variation discussion. All formal results used in the main paper retain their complete proofs in the main paper.
- [`supplement/technical_archive.pdf`](supplement/technical_archive.pdf): retained supporting constructions, Acrobot/flow examples, historical studies, and the implementation audit, with the preceding section/figure numbering preserved. Both documents use assets under `supplement/`.
- `experiments/direct_submission/{electrical_fourier,contact,burgers}/`: current study sources, protocols, audits and reproduction instructions. Electrical records include the independently audited ten-seed cohort and two separate refinements. Full raw archives are separately identified artifacts.
- `experiments/direct_submission/fourier/`: retained historical spatial-graph study with six independent weights and supplied phase `20x`; its original code, commands and records remain unchanged.
- Other `experiments/` scripts: retained contact, bounded/normalized periodic Fourier, Burgers, excitation, and estimator-comparison protocols. `direct_field.py` is the generic all-entry matrix learner; a focused unit test is not an application-validation result.
- `examples/legacy_suite/`: retained Acrobot, immersed-airfoil prototype, Burgers, Fourier, and historical records. The active flow is an unvalidated numerical prototype. The ellipse discussion is analytic; its untraceable empirical figure is excluded.
- `tests/`: differential, gauge, learning, and numerical checks.
- `results/`: historical summaries and selected seed-10 traces; these are not the current direct-study raw records.
- `docs/`: interpretation, limitations, figure/code mappings, and reproducibility guidance.

## Reproduction

Use [`docs/REPRODUCE.md`](docs/REPRODUCE.md) to distinguish current direct studies from historical reproductions. Every changed scientific protocol has a separate identity; reruns belong in new output directories. The current records specify their Python, numerical-library, and compiler versions. Historical runs used Python 3.13.5; the November 18 reproduction used Python 3.12.3 with the packages in `requirements-tested.txt`.

```bash
python -m unittest discover -s tests -v
(cd examples/legacy_suite && python -m unittest discover -s tests -v)
```

The portable Burgers package has a focused compiled-kernel/replay check. Electrical reproduction, audit and plotting commands are documented separately from the retained graph commands; audit outcomes are bound to the recorded source and data identities. Follow the study-specific instructions. With TeX Live and IEEEtran installed, `bash scripts/build_supplement.sh` builds the focused supplement by default; `bash scripts/build_supplement.sh technical_archive` builds the separate archive. Each command updates the corresponding local PDF and auxiliary files under `supplement/`. Bibliography processing runs only when the document declares bibliography data; the archive requires BibTeX, while the focused supplement has no bibliography.

## Retained historical evidence

The preceding spatial graph has identity `direct_fourier_monotone_20261120_v1`: six independent harmonic weights, supplied phase `20x`, and a separate 30-s prediction episode. Its source remains in `experiments/direct_submission/fourier/`, and its figure is retained as `historical_direct_graph_fourier_summary.pdf`. These records are not evidence for the new electrical protocol.

The [technical archive](supplement/technical_archive.pdf) retains the older supporting studies. The old normalized Fourier experiment learns seven coefficients of a prescribed physical residual with the coupled normalized law. Its same-inverse-command current-rate completion is a consistency diagnostic, not independent prediction. Old contact and Burgers studies fit physical coefficients and evaluate induced partials. Their additional excitation, matched-estimator, and learned-entry diagnostics retain that mechanism.

The earlier six-entry lifted field and four-coefficient exact-Jacobian companion are separate known-basis diagnostics. The exact-Jacobian family still imposes a physical/algebraic restriction; it is not interchangeable with independently learned field entries. The current electrical replacement has a new identity; its validation status must not be inferred from these historical audits.

## Data, provenance and citation

The original `Implicit_Learning_All_Seed_Records.zip` is absent; its filename/checksum remain in [`docs/DATA_MANIFEST.json`](docs/DATA_MANIFEST.json). All original structured summaries and selected traces remain. On November 18, all 100 original structured and 20 paired-transfer protocols were reproduced and audited: saved summaries and the 12 available historical raw-run arrays matched with zero maximum absolute difference. [`docs/REPRODUCTION_20261118.json`](docs/REPRODUCTION_20261118.json) identifies the regenerated records, not a recovered historical archive.

Separate retained local archives are described in [`docs/SUPPLEMENTARY_STUDIES_20261118.json`](docs/SUPPLEMENTARY_STUDIES_20261118.json) and [`docs/NORMALIZED_FOURIER_20261118.json`](docs/NORMALIZED_FOURIER_20261118.json). Historical periodic-plant figure/query postprocessing is in [`docs/NORMALIZED_FOURIER_MAIN_20261119.json`](docs/NORMALIZED_FOURIER_MAIN_20261119.json).

The November 21 revision uses [`docs/DIRECT_MAIN_20261121.json`](docs/DIRECT_MAIN_20261121.json) for current source/record bindings and [`docs/ELECTRICAL_FOURIER_20261121.json`](docs/ELECTRICAL_FOURIER_20261121.json) for electrical provenance, independent audits and refinements. These identify local evidence, not a published release or public raw-data download. The unchanged [`docs/DIRECT_MAIN_20261120.json`](docs/DIRECT_MAIN_20261120.json) and [`docs/DIRECT_RAW_DATA_20261120.json`](docs/DIRECT_RAW_DATA_20261120.json) identify the preceding spatial graph together with the unchanged contact/Burgers records; the latter retains its 3,853 archived file hashes. Match each archive to its own manifest rather than relabeling the older archive as electrical evidence. No remote availability, journal acceptance, peer review, DOI, software license, or frozen publication release is asserted. See `CITATION.cff` and `NOTICE.md` for citation metadata and reuse terms pending the author's license decision.
