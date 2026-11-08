# Figure and code provenance

| Figure family | Source data | Generator | Scope |
|---|---|---|---|
| Contact partials, energy, task, command | `results/contact/seed_10/learned/trace.csv` and `partials.npz` | `experiments/make_figures.py` | New ten-seed contact reconstruction |
| Fourier rate/clock partials, energy, task | `results/fourier/seed_10/learned/trace.csv` | `experiments/make_figures.py` | New simultaneous learning/control protocol |
| Burgers partials, energy, field tracking | `results/burgers/seed_10/learned/` | `experiments/make_figures.py` | Supplied reconstruction with added diagnostics |
| Task-only versus probing | `results/contact_excitation/seed_10/` and `aggregate.json` | `experiments/plot_excitation.py` | New supplementary study |
| Acrobot identification and local LQR | `examples/legacy_suite/results/acrobot/seed_10/` | Original legacy implementation | Archived reconstruction |
| Immersed-airfoil control and vorticity | `examples/legacy_suite/results/airfoil/` | Original legacy implementation | Prototype only, not validated aerodynamic data |
| Ellipse archived figure | `supplement/figures/ellipse_archived.pdf` | Not supplied | Archival, no original implementation |

All main study summary values are aggregated from noise seeds 10–19; plotted seed 10 was preselected. True partials are evaluator-only. The displayed incompatibility is computed before assimilation of the current sample. Noise and discretization prevent interpreting a small energy as a guarantee of complete identification.
