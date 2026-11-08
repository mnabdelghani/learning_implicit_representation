IMPLICITLEARNING -- SIMULATION CODE AND ACTIVE-FLOW-CONTROL LATEX
Prepared 7 November 2026

START HERE
This package provides executable Fourier, Acrobot, Burgers, and airfoil
flow-control experiments. The active-flow-control text and bibliography are
separate from the full manuscript. The revised full manuscript is delivered
separately from this code package.

NOTATION CLEANUP -- 7 November 2026
The Acrobot and airfoil implementation notes now describe learning through
one implicit feature relation, with supplied physical contributions retained
as channels with fixed coefficients. The standalone flow-control write-up
uses the corresponding implicit output-rate constraint. This is a notation
and documentation revision only: every Python source, test, configuration,
and saved numerical result is unchanged. No experiments were rerun to replace
the recorded results. The manifest has been regenerated for this package.

Provenance matters:
  Fourier: the same deterministic diagnostic generated for the revised paper,
           with a command-line interface added.
  Acrobot: a new implementation of the manuscript's described protocol.
  Burgers: a new implementation of the manuscript's described protocol.
  Airfoil: a new computational-flow prototype, not an original paper result.

The original Acrobot/Burgers simulation records were absent from the supplied
archives. Newly chosen settings are disclosed in configs/ and docs/. The
included numerical output must not be substituted silently for the original
paper's numerical table. No values are manufactured to match that table.

INSTALL
Python 3.12 was used for the saved runs. NumPy, SciPy, and Matplotlib are the
only third-party runtime dependencies. No GPU is required.

  python -m pip install -r requirements.txt

For the exact package versions used here:

  python -m pip install -r requirements-tested.txt

An optional virtual environment avoids changing your main Python installation:

  python -m venv .venv

On Windows, use .venv\Scripts\python.exe in place of python in the commands.
On macOS/Linux, use .venv/bin/python after creating the environment.

RUN
Short execution check of all four experiments:

  python run_all.py --quick --output runs/quick

Default experiments, one deterministic noise seed:

  python run_all.py --seed 10 --output runs/default_seed10

Run one experiment:

  python simulations/fourier.py --output runs/fourier
  python simulations/acrobot.py --seed 10 --output runs/acrobot
  python simulations/burgers.py --seed 10 --output runs/burgers
  python simulations/airfoil.py --seed 10 --output runs/airfoil

For ten independent noise seeds in the two reconstructed paper experiments:

  python simulations/acrobot.py --seeds 10 11 12 13 14 15 16 17 18 19 --output runs/acrobot_ten_seeds
  python simulations/burgers.py --seeds 10 11 12 13 14 15 16 17 18 19 --output runs/burgers_ten_seeds

Only seed 10 has been executed for the included default physical results.
The single-seed summaries contain no estimated multi-seed standard deviation.

For all script-specific switches:

  python simulations/acrobot.py --help
  python simulations/burgers.py --help
  python simulations/airfoil.py --help

Numerical consistency checks:

  python -m unittest discover -s tests -v

Quick mode checks execution and produces short trajectories. It is not a
publication experiment. Default runs use the settings recorded by each script.
Changing a grid, time step, horizon, seed, or noise level changes the experiment.

WHAT IS SAVED
Each experiment writes its own data, figure(s), and a machine-readable JSON
summary under the selected output directory. The results/ directory contains
the runs actually performed while preparing this package. Configurations and
seed information are included with the outputs. run_all.py also saves a
run_manifest.json recording commands and elapsed times.

The Fourier energy plot uses a display floor of 1e-32; the CSV retains the
computed values. At roundoff, a tiny energy value is not a scientific claim
of exact recovery or of a general convergence theorem.

IMPLICIT LEARNING
The learning quantity is the differential incompatibility F_theta(z) dot z.
For the dynamic experiments, complete nonlinear features are differenced
causally; the learner does not receive true partial derivatives as labels.
Ground-truth quantities, where available from the simulator, are reserved for
evaluation. Readouts and controllers use the learned relation and measured
state. Nonzero constraints or explicitly stated model structure prevent the
trivial zero representation. The implementation notes specify which is used.

ACTIVE FLOW CONTROL TEXT
  latex/active_flow_control.tex -- standalone experiment write-up
  latex/active_flow_control_insert.tex -- compact section for an existing paper
  latex/active_flow_control.bib -- five verified primary references
  docs/active_flow_reference_notes.txt -- what each reference supports

The LaTeX describes the actual packaged prototype, its implicit residual,
partials, learning, readout controller, and validation status. It distinguishes
the supplied numerical demonstration from a validated surface-jet experiment.
No full-paper LaTeX or paper PDF is included in this code package.

Compile the active-flow-control text from its directory:

  pdflatex active_flow_control.tex
  bibtex active_flow_control
  pdflatex active_flow_control.tex
  pdflatex active_flow_control.tex

The compact LaTeX insert and .bib file are also provided separately for reuse.
For a short insertion into the manuscript, use active_flow_control_insert.tex
and merge the five BibTeX entries into your bibliography. The insert needs
amsmath, amssymb, and booktabs. Its equation labels are prefixed afc_short.
An inline bibliography alternative, active_flow_control_references.tex, is
also included. If your paper already uses thebibliography, merge its bibitem
entries into that environment instead of nesting another bibliography.
Deduplicate any entries already present, such as the Montala preprint. Only the
active-flow-control experiment is covered by these new LaTeX files.

FILES
  run_all.py                 Portable runner for selected experiments
  requirements*.txt          Dependencies and tested versions
  simulations/              Runnable experiment implementations
  configs/                  Declared numerical choices
  tests/                    Physics, derivative, and constraint checks
  docs/                     Protocol assumptions and reference notes
  latex/                    Active-flow-control text and bibliography only
  results/                  Actual newly generated data and figures

Read docs/verification_summary.txt for the final executed checks and the
distinction between computational correctness and physical validation.
