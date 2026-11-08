# Learning Implicit Representation

**Research code and supplementary material for _Implicit Representation Estimation, Prediction, and Control_.**

**Publication boundary:** The definitive 10-page manuscript (both LaTeX and PDF) is deliberately **not included** in this repository at the author's request. This repository is intended to host the supporting material, numerical experiments, tests, and reproduction documentation. The supplement is not a replacement for the manuscript.

## Included material

- `supplement/supplement.tex` and `supplement/supplement.pdf`: standalone experimental protocols, extended diagnostics, explanatory counterexamples, and additional examples, including the paired contact-excitation study. All formal proofs remain in the private main paper.
- `experiments/`: nonlinear contact, online Fourier Memory, nonlinear Burgers, result auditing, plots, and the supplementary excitation study.
- `examples/legacy_suite/`: original accompanying Acrobot, immersed-airfoil prototype, Burgers and Fourier implementations and existing records. The ellipse example is archival only; its executable source was not supplied.
- `tests/`: differential, gauge, learning, and numerical tests.
- `results/`: numerical summaries and selected seed-10 traces. All-seed complete traces are distributed separately as a large data archive, not as Git objects.
- `docs/`: results, limitations, provenance, offline-to-online learning guidance, and reproducibility map.

## Quick start

Python 3.13 is the recorded interpreter; `requirements-tested.txt` records the numerical dependencies.

```bash
python -m pip install -r requirements-tested.txt
python -m unittest discover -s tests -v
(cd examples/legacy_suite && python -m unittest discover -s tests -v)
```

A short contact run, separate from the paper's numerical results:

```bash
python experiments/contact.py --duration 2 --seeds 10 --output runs/contact_smoke
```

To reproduce the ten-seed results and regenerate plots, see [`docs/REPRODUCE.md`](docs/REPRODUCE.md). To compile the supplement independently of the withheld manuscript, run `bash scripts/build_supplement.sh` (requires a TeX Live installation with IEEEtran and BibTeX).

## Experimental interpretation

The three retained studies are nonlinear contact control, Fourier Memory with simultaneous learning/control, and distributed Burgers learning/control. All report pre-update incompatibility and evaluated differential diagnostics. Contact was reconstructed because the original code was unavailable; Fourier uses a newly designed closed-loop protocol; Burgers uses the previously supplied reconstruction with additional diagnostics. None of these simulations establishes identification without excitation, global stability, or safety certification.

The additional **paired contact-excitation experiment** compares regulation without probing to regulation with bounded deterministic probing. Both learn during an initial stationary-reference task, then freeze their estimates and perform a different task. It examines the difference between low training incompatibility and transferable differential information. The experiment is independent of the original 100-run protocol; see `docs/EXCITATION_STUDY.md` and Supplement S8.

## Data, provenance and citation

Full records from the original 100-run study were previously generated and are distributed as a separate archive (`Implicit_Learning_All_Seed_Records.zip`). The repository includes a data manifest with its SHA-256 checksum, so the archive can be verified before extraction. The new paired-excitation study has its own configurations, summary statistics, and selected diagnostic traces. See `docs/DATA_MANIFEST.json`.

No journal acceptance, peer review, software license, DOI, or frozen publication release is implied by the repository. See `CITATION.cff` for citation metadata and `NOTICE.md` for reuse terms pending the author's license decision. Before citing numerical results in a paper, identify the exact commit or tagged release and confirm the archived records match it.
