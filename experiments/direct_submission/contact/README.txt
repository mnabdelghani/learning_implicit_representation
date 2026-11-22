Direct temporal contact replacement, fixed November 20, 2026

Study status: five heldout generic initializations pass the predeclared criteria.
The same pre-update learned Q computes every control command. All 24 entries
adapt without physical residual coefficients. See confirmatory_protocol.json,
contact_direct.py and data/independent_burgers_agent_audit.json.

Portable source files expected together:
  contact_direct.py, direct_field.py, confirmatory_protocol.json,
  reproduce_contact.py, audit_contact.py, report_contact.py, plot_contact_main.py,
  plot_figure_focus.py.
No private manuscript input is needed. Use the recorded Python/NumPy/SciPy
versions. Plotting also uses Matplotlib, and the combined vector PDF uses the
project's existing pdflatex (standalone class) and pdftoppm installation.

Reproduce the exact five confirmation seeds and one dt-halving refinement in a
new destination (refuses an existing output directory):
  python reproduce_contact.py --output /path/to/new/contact_records --workers 2
Then independently reconstruct metrics and generate the presentation:
  python audit_contact.py --records /path/to/new/contact_records
  python report_contact.py --records /path/to/new/contact_records
These postprocessors use saved records only; they do not retrain the field.
To regenerate the current focused main presentation from an extracted original
case (or the corresponding newly reproduced case), use a new output directory:
  python plot_figure_focus.py --case /path/to/contact_records/confirm_seed101 --output /path/to/new/contact_main_figure
This produces the four-panel partial-error/energy/force/position figure and its
input/source/output hashes. It uses NumPy and Matplotlib, without LaTeX tools.
The preceding plot_contact_main.py still produces the five-panel presentation.
The prior report_contact.py remains unchanged and produces the earlier combined
presentation plus the all-24 traces, canonical-gauge diagnostic, and statistics.
The reproduction driver verifies the protocol's frozen source hashes before
running. It preserves every case and log. The original declaration producer
is declare_and_run.py; it is historical provenance, not the recommended rerun CLI.

Cases preserved in this study:
 pilot_ideal_seed0: first reference, guard failure at 58.58 s;
 pilot_local_seed0: revised local rich reference, complete 60 s, no parameter jump;
 pilot_local_change_seed0: same local reference, complete 120 s with jump 60 s;
 confirm_seed101--105: five heldout Q initializations, same 120-s protocol;
 refine_seed101: .005-s step rather than .01, remaining settings unchanged.
Each case includes its actually executed source snapshot, config, environment,
raw every-sample Q/R/context/rates/physical-evaluator field/query, CSV, summary,
and per-file SHA256 manifest. The pilots are not part of confirmatory statistics.

Scope and unavoidable limitations:
- Ideal noise-free context and physical time-rate sensors, including force
  acceleration. This does not solve differentiation-noise robustness.
- Measured temporal trajectories only, no oracle tangent labels or targets.
- Vertical force lambda and horizontal position, not normal-force magnitude.
- Approximate local row-space identification: primary mean projector error
  .298169 versus 1.93389 for same-stream frozen initial Q. This residual remains.
- Orthogonal evaluator-only gauge alignment is used for normalized entry plots.
  Forced unit force-rate normalization is ill-conditioned and physical material
  amplitudes are not recovered accurately; the failing diagnostic is retained.
- Frozen initial model is a same-stream shadow, not a separate closed-loop trial.
- Completion error compares the same requested query against physical truth;
  it is not independent future prediction. Actual task tracking is separate.
- Step halving preserves acceptance but changes force RMSE .121637 to .109237;
  no claim of converged task precision. No global integrability/stability proof.
- At the change, finite one-sided new-branch rates are observed; jump impulses
  are not treated as ordinary continuous-time derivative samples.

Presentation:
  included_figures/direct_contact_focus.pdf: current four-panel main figure.
  figure_focus_20261121/: source snapshot, derived arrays, figure/analysis
    manifests for contact_focused_figure_20261121_v1. All 24 signed entry errors
    and the complete 0--120 s window remain visible; energy has its unchanged
    causal 0.5-s mean. Thick dotted task references and thin solid actual
    trajectories distinguish the curves. Projector error remains in text and
    prior diagnostics; its late mean is 0.298169.
  included_figures/direct_contact_summary.pdf: retained five-panel presentation.
  included_figures/historical_direct_contact_summary_20261120.pdf: prior combined
    six-panel presentation, preserved under the original data/figure_manifest.json.
  analysis_20261121/analysis_manifest.json: contact_all24_exposition_20261121_v1,
    a new analysis identity bound to the unchanged seed101 raw archive members.
  analysis_20261121/figure_manifest.json, display_data.npz, plot_contact_source.py:
    exact display gauge, transformations, all 24 signed entry errors, and source.
  Both all-entry heatmaps show OQ-T with one unclipped linear scale; the orthogonal
    display gauge does not recover canonical material amplitudes. Learning
    continues after control starts at 40 s; the plant changes at 60 s.
  The original report_contact.py outputs remain:
  figures/direct_contact_{partials,field,energy,force,motion,inputs}.pdf: panels.
  figures/direct_contact_all24.pdf: all normalized entries, evaluator gauge.
  figures/direct_contact_canonical_gauge_limitation.pdf: retained failed gauge.
  report_metrics.json/contact_metrics.tex: all-five primary-window statistics.
  display_gauge_diagnostics.npz: exact plotted display transformations/data.
Packaging note: this folder includes source, compact data/audit summaries,
included_figures/ PDFs, and the new derived main-figure display arrays. Full raw
Contact records and original report display arrays remain separate in the
unchanged raw_data/contact_raw.tar.gz (or .tar), not in this code package.
Original scientific source bytes are unchanged; see ../source_identity.json.
No private manuscript input or publication is implied.
