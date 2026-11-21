# Learning Implicit Representation

**Research code and supplementary material for _Implicit Representation Estimation, Prediction, and Control_.**

**Publication boundary:** The definitive manuscript source and PDF are deliberately excluded. This repository contains supporting protocols, code, figures, and reproducibility documentation. The supplement is not a replacement for the manuscript. A local accompanying package does not imply a published release or a public raw-data download.

The current main evidence directly learns every field entry from observed context-rate incompatibility and uses that same learned field for prediction or control. The portable implementations are in [`experiments/direct_submission/`](experiments/direct_submission/), and complete protocols and limitations are in Supplementary Section S20. The earlier physical-coefficient studies remain separately identified historical evidence; their algorithms and records have not been relabeled as direct partial learning.

## Current direct-field studies

- **Fourier:** six independent harmonic field weights, ten seeds, a nonlinear monotone graph, known coordinate phase/order, exact input position/rate and noisy output-rate sensing. The actual supplied rate modulates at 0.3 rad/s while the physical partial phase rate is 2.7–3.3 rad/s. A frozen learned field predicts a separate 30-s input episode from one exact initial output measurement. This is a changed benchmark, not the old periodic plant's control experiment.
- **Contact:** all 24 entries of a three-row field adapt; five initializations perform force/motion readout through a material/actuation change. Ideal rate sensing and maintained contact are supplied. Useful regulation accompanies approximate row-space recovery; precise canonical material magnitudes are not established.
- **Burgers:** each of 32 spatial sites learns all five entries from causal context secants, with ideal state/rate sensing. Five seeds branch into adaptive and frozen own-trajectory control. Continued adaptation supports late readout; only two seeds satisfy every strict training criterion. Failed causal/noisy conditions and timestep sensitivity remain reported.

These are scoped simulation demonstrations. Low incompatibility, field recovery, and task performance are reported separately. They do not establish unrestricted equation discovery, general integrability, unknown-frequency discovery, broad noise robustness, global stability, or physical safety.

## Included material

- `supplement/`: standalone protocols, original supporting studies, the implementation audit, direct main-study records and limitations in S20, and supporting figures. Formal results used in the main paper retain their complete proofs there.
- `experiments/direct_submission/{fourier,contact,burgers}/`: portable current implementations, protocols, audits, reproduction instructions, and compact evidence where provided. Full raw archives are separate accompanying artifacts.
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

The portable Burgers package also has a focused compiled-kernel/replay check; the Fourier package independently audits complete covariance records and readouts. Follow their included instructions. To compile the standalone supplement, run `bash scripts/build_supplement.sh` with TeX Live, IEEEtran, and BibTeX installed.

## Retained historical evidence

The old normalized Fourier experiment learns seven coefficients of a prescribed physical residual with the coupled normalized law. Its same-inverse-command current-rate completion is a consistency diagnostic, not independent prediction. Old contact and Burgers studies fit physical coefficients and evaluate induced partials. Their additional excitation, matched-estimator, and learned-entry diagnostics retain that mechanism.

The earlier six-entry lifted field and four-coefficient exact-Jacobian companion are separate known-basis diagnostics. The exact-Jacobian family still imposes a physical/algebraic restriction; it is not interchangeable with independently learned field entries. The current main studies have new records or separately audited direct records and different scopes.

## Data, provenance and citation

The original `Implicit_Learning_All_Seed_Records.zip` is absent; its filename/checksum remain in [`docs/DATA_MANIFEST.json`](docs/DATA_MANIFEST.json). All original structured summaries and selected traces remain. On November 18, all 100 original structured and 20 paired-transfer protocols were reproduced and audited: saved summaries and the 12 available historical raw-run arrays matched with zero maximum absolute difference. [`docs/REPRODUCTION_20261118.json`](docs/REPRODUCTION_20261118.json) identifies the regenerated records, not a recovered historical archive.

Separate retained local archives are described in [`docs/SUPPLEMENTARY_STUDIES_20261118.json`](docs/SUPPLEMENTARY_STUDIES_20261118.json) and [`docs/NORMALIZED_FOURIER_20261118.json`](docs/NORMALIZED_FOURIER_20261118.json). Historical periodic-plant figure/query postprocessing is in [`docs/NORMALIZED_FOURIER_MAIN_20261119.json`](docs/NORMALIZED_FOURIER_MAIN_20261119.json).

The current direct package supplies study-specific source/configuration hashes, audits, and data mappings. [`docs/DIRECT_MAIN_20261120.json`](docs/DIRECT_MAIN_20261120.json) identifies its code and full raw archives; [`docs/DIRECT_RAW_DATA_20261120.json`](docs/DIRECT_RAW_DATA_20261120.json) lists all 3,853 archived file hashes. Accompanying raw archives must be matched to these manifests. No remote availability, journal acceptance, peer review, DOI, software license, or frozen publication release is asserted. See `CITATION.cff` and `NOTICE.md` for citation metadata and reuse terms pending the author's license decision.
