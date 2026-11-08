# Reproduction instructions

The repository deliberately omits both the PDF and LaTeX of the definitive 10-page manuscript. The supplement is self-contained and does not require the main-paper source or auxiliary files.

Install Python 3.13 and the versions in `requirements-tested.txt`, then run:

```bash
python -m unittest discover -s tests -v
(cd examples/legacy_suite && python -m unittest discover -s tests -v)
```

Re-run the main experiments with their original declared ten seeds (this overwrites existing records):

```bash
python experiments/contact.py --output results/contact --seeds 10 11 12 13 14 15 16 17 18 19
python experiments/fourier_online.py --output results/fourier --seeds 10 11 12 13 14 15 16 17 18 19
python experiments/burgers_online.py --output results/burgers --seeds 10 11 12 13 14 15 16 17 18 19
python experiments/summarize.py
python experiments/audit_results.py
python experiments/make_figures.py
```

For the additional excitation study:

```bash
python experiments/contact_excitation.py --workers 2
python experiments/plot_excitation.py
```

Compile the supplement using `bash scripts/build_supplement.sh`; requires TeX Live including IEEEtran, AMS packages and BibTeX. The generated plots are separate from the manuscript and can be regenerated from seed-10 records.

To audit all ten seeds without rerunning the original study, obtain the separately distributed complete-record archive, verify its SHA-256 checksum against `docs/DATA_MANIFEST.json`, and extract into the repository root. The default source bundle includes selected seed-10 traces and summary JSON files for every original run; this is sufficient to reproduce the plots and aggregate tables, but not to independently audit all ten original traces without the separate archive.

A command or trace generated after changing initial conditions, probing, noise, integrator settings or feature families is a new experiment and must not be substituted for the paper's reported results.
