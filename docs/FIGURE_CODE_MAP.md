# Figure and code provenance

The current study set uses direct all-entry learning and same-field readout. The electrical Fourier replacement, `electrical_fourier_direct_20261121_v1`, awaits independent validation and a bound figure; its row below is the intended assignment, not a claim of completed results. Current bindings belong to `DIRECT_MAIN_20261121.json` and `ELECTRICAL_FOURIER_20261121.json`. The preceding spatial graph and coefficient-family figures remain historical. Stable LaTeX labels identify current assignments; original record identities retain their meanings. Match [Focused Supplement S2](../supplement/supplement.pdf) and its electrical figure to the current manifest before use. The [technical archive](../supplement/technical_archive.pdf) preserves the preceding S1–S20 numbering, including flow Figures S8/S9.

Paths are repository-relative unless prefixed by **NF** or **ST**. NF denotes
the extracted normalized records described in
[`NORMALIZED_FOURIER_20261118.json`](NORMALIZED_FOURIER_20261118.json). ST denotes
the separate supplementary-study records described in
[`SUPPLEMENTARY_STUDIES_20261118.json`](SUPPLEMENTARY_STUDIES_20261118.json).
These archives exist locally; these metadata files do not assert a public
download. The original 100-run reproduction is a separate historical protocol
set, described in [`REPRODUCTION_20261118.json`](REPRODUCTION_20261118.json).

## Current main and focused-supplement figures

| Figure / stable label | Records and actual learning mechanism | Generator and included assets |
|---|---|---|
| Current electrical Fourier — validation pending, `fig:fourier_online`; supplementary `fig:direct_fourier_new` | `electrical_fourier_direct_20261121_v1`, seeds 10–19, illustration seed 10; 20 independent `K=2` temporal weights, frequency estimated from a 20-s current-only calibration prefix, ideal current-rate sensing; matched `K=0`, separate 12-s frozen-weight prediction and current-control tasks | `experiments/direct_submission/electrical_fourier/run.py`, `audit.py`, `summarize_plot.py`; new electrical `direct_fourier_summary.pdf` only after its source/record binding is validated |
| Current direct contact, `fig:contact_online`; focused-supplement diagnostics `fig:direct_contact_all24`, `fig:direct_contact_canonical` | Five seeds 101–105, preselected seed 101; all 24 matrix entries adapt from ideal measured rates, then supply the rate-level force/motion readout | `experiments/direct_submission/contact/`; `direct_contact_summary.pdf`, `direct_contact_all24.pdf`, `direct_contact_canonical_gauge_limitation.pdf` |
| Current direct Burgers, `fig:burgers_online`; focused-supplement `fig:direct_burgers_protocol` | Ideal-rate cohort seeds 732–736; preselected seed 732, site 0; five independent entries per spatial site, causal secants and own-trajectory adaptive/frozen control | `experiments/direct_submission/burgers/burgers_direct.py`, `plot_main.py`; `direct_burgers_partials.pdf`, `direct_burgers_energy.pdf`, `direct_burgers_tracking.pdf` |

## Retained technical-archive figures and historical assets

The following historical studies and supporting constructions retain their original identities. Technical-archive section and figure numbers keep their preceding meanings; they are not the focused-supplement numbering. The retained direct graph has a separate historical asset so replacing the current electrical figure cannot relabel graph data. `DIRECT_MAIN_20261120.json` and `DIRECT_RAW_DATA_20261120.json` remain unchanged bindings for the graph plus the existing contact/Burgers records.

| Figure / stable label | Records and actual learning mechanism | Generator and included assets |
|---|---|---|
| Retained direct spatial Fourier graph (former main assignment) | `direct_fourier_monotone_20261120_v1`, seeds 10–19, illustration seed 10; six independent harmonic weights, supplied phase `20x`, same-stream `K=0` and separate 30-s prediction | Unchanged `experiments/direct_submission/fourier/run.py`, `summarize_plot.py`; preserved figure `historical_direct_graph_fourier_summary.pdf` |
| Retained coefficient-based contact (former main assignment) | `results/contact/seed_10/learned/`; integral material and velocity-level actuation fits, 48 projected-gradient steps | `experiments/contact.py`, `experiments/make_figures.py`; historical contact panels |
| Retained coefficient-based Burgers (former main assignment) | `results/burgers/seed_10/`; three bounded coefficients, supplied time-derivative coefficient, eight projected-gradient steps | `experiments/burgers_online.py`, `examples/legacy_suite/simulations/burgers.py`, `experiments/make_figures.py`; historical Burgers panels |
| Original bounded Fourier, `fig:fourier_bounded` | `results/fourier/seed_10/`: bounded harmonic coefficients and same-stream K=0 shadow; eight projected-gradient steps | `experiments/fourier_online.py`; `experiments/make_figures.py`; six original `fourier_*.pdf` panels |
| Contact/Fourier diagnostics, `fig:contact_fourier_diagnostics` | Same original contact and bounded Fourier records | `experiments/make_figures.py`; contact motion/activity/information and Fourier full-partial/activity/information panels |
| Burgers diagnostics, `fig:burgers_diagnostics` | Same original Burgers records | `experiments/make_figures.py`; `burgers_F_error.pdf`, `burgers_task.pdf`, `burgers_control.pdf` |
| Excitation and frozen-model transfer, `fig:excitation` | `results/contact_excitation/`: seed-10 trajectories; information spectrum averages seeds 10–19 | `experiments/contact_excitation.py`, `experiments/plot_excitation.py`; five `excitation_*.pdf` panels |
| Reconstructed unanchored Fourier, `fig:unanchored_fourier` | `examples/legacy_suite/results/fourier/`; fixed-batch covariance and retracted row update; the direct field need not be a Jacobian | `examples/legacy_suite/simulations/fourier.py`; `unanchored_fourier_reconstructed.pdf`, a byte-identical copy of the checked legacy `fourier.pdf` |
| Acrobot, `fig:acrobot_archived` | `examples/legacy_suite/results/acrobot/seed_10/`; proximal coefficient updates during identification, then frozen-model local LQR | `examples/legacy_suite/simulations/acrobot.py`; that directory's identification/control PDFs |
| Active flow, archive Figures S7–S9: `fig:flow_results`, `fig:flow_fields`, `fig:flow_streamlines` | `examples/legacy_suite/results/airfoil/`; ridge-regularized box solve with 25% relaxation. Field views use the saved seed-10 `fields.npz:final_adaptive` at control time 6; no new simulation | `examples/legacy_suite/simulations/airfoil.py` for retained results/parameters; `experiments/plot_active_flow_fields.py` for enlarged full-range `active_flow_vorticity.pdf` and instantaneous `active_flow_streamlines.pdf` (PNG previews also included). Input/output hashes and transformations: `supplement/figures/active_flow_fields_provenance.json`. Legacy numerical prototype, not direct-partial or validated aerodynamic evidence |
| Direct field, `fig:direct_lift_diagnostics` | ST direct-lift records; six-entry field, frozen-memory normalized substeps | `experiments/direct_lift_audit.py`, `experiments/plot_direct_lift_audit.py`; `direct_lift_diagnostics.pdf` |
| Estimator comparisons, `fig:estimator_common_stream`, `fig:estimator_partial_tracking` | ST comparison records; common streams, seeds 10–19; bounded PG, EWLS and projected NLMS | `experiments/estimator_comparison.py`, `experiments/plot_estimator_comparison.py`; two estimator PDFs |
| Learned-only partial errors, `fig:unknown_partial_tracking` | Postprocessing of the original 100-run protocol set; supplied entries excluded from the error mask; no new learning rule | `experiments/derivative_tracking_audit.py`; `unknown_partial_tracking.pdf` |
| Extended normalized Fourier, `fig:normalized_fourier_partials`, `fig:normalized_fourier_summary` | NF replay and own-trajectory streams, all seeds 10–19, eigenvector/frozen shadows; partial figure explicitly combines replay partials with own-trajectory task/control | `experiments/normalized_fourier.py`, `experiments/report_normalized_fourier.py`; two `normalized_fourier_*.pdf` overview figures |

Current direct assets and retained historical assets live under `supplement/figures/`, except the Acrobot source PDFs. The electrical figure must have its own validated source/record binding before replacing the current assignment; the historical graph copy keeps its original bytes. The direct Burgers generator names its outputs `burgers_main_{partials,energy,tracking}.pdf`; the included copies use the `direct_burgers_` prefix without changing their bytes. Raw direct-study archives accompany the package separately; compact plotting arrays and raw subsets are not described as complete archives.
`experiments/repair_supplement_layout.py` repairs labels/layout in three older
supplementary panels and checks their plotted data. The four-coefficient
exact-Jacobian companion to the direct-field study has table-only results;
its runner and independent checker are `experiments/direct_jacobian_audit.py`
and `experiments/audit_direct_jacobian.py`.

The circle, mechanical and ellipse constructions are analytic examples retained in the technical archive. The
ellipse empirical figure was removed because its generator/records could not
be established. The old ellipse and unanchored figure files remain archival
assets on disk but are not included in the main paper, focused supplement, or technical archive. The
checked unanchored reconstruction is not claimed to recover the historical
figure's exact provenance.

The six historical periodic-plant Fourier panels use the same normalized-controller seed-10
trajectory. Rate/clock partials show the first 16 seconds without smoothing;
the energy panel uses a causal one-second mean. Aligned partials divide by the
learned input coefficient, which is adaptive. Reported late statistics use all
ten seeds and `60 <= t < 80 s`. Exact input hashes, plot transformations, code
hash and output hashes are recorded in
[`NORMALIZED_FOURIER_MAIN_20261119.json`](NORMALIZED_FOURIER_MAIN_20261119.json).

The added current-rate query uses the same learned coefficients and recorded
state/input/phase; actual rate is evaluator-only. Because the input was obtained
by the same relation's inverse controller, this query returns the requested
rate algebraically. Its error against the realized plant checks relation/readout
accuracy; it is not independent predictive validation. The frozen comparator
is queried on the same observations, not run as a separate controller.

Technical Archive S18 inventories the retained historical implementations; focused Supplement S2 specifies the current direct examples. The archive retains the preceding copy of those protocols as S20. The electrical protocol specifies a new-input prediction with no later current/rate assimilation and a separate causal current-control trajectory using the same frozen weights. Frequency comes from prefix current observations, not the true plant frequency/phase; only the clock partial is fast relative to applied training voltage. These tasks await their own validation and must not inherit outcomes from the graph or inverse-command consistency check above. Contact uses a frozen initial row only as a same-stream diagnostic; Burgers runs separate adaptive/frozen control trajectories. Small incompatibility alone does not establish physical identification, and retained historical implementations do not all execute the central normalized law.
