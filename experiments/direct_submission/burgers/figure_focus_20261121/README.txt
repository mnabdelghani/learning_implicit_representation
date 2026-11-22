Burgers focused figure display, 2026-11-21

This is a presentation revision of the immutable ideal-sensing study
burgers_direct_field_long_training_multiseed_causal_sensing_20261120_v1.
No simulation, learning rule, physical data, scientific protocol, or recorded
metric changed. The original detailed plots and plot_main.py remain unchanged.

From the Burgers study directory, using the existing NumPy/Matplotlib environment:

  python plot_figure_focus.py --data data/ideal_plot_data.npz --output NEW_FIGURES

The output directory must not already exist. The command produces the four-panel
main figure, the one-panel additional representation diagnostic, exact derived
arrays, a source snapshot, and a SHA-256 manifest. Default render stride is 5
samples (0.025s); the final saved sample is included. No smoothing or log floor is
used. Metrics use every applied sample and exclude terminal unapplied commands.
The first 10 training energies have no complete 0.05s secant and remain NaN gaps.

Main figure, 7.3 by 3.8 inches:
(a) Pre-update secant incompatibility: learned-row mean and full range over all
five ideal seeds, through training and adaptive control. Dotted physical-row
energy is the matched five-seed mean; this averages the same saved physical
reference arrays instead of drawing only seed 732 as the older display did.
(b) All five partials at preselected seed 732/site0. Original unit/sign alignment
is used unchanged, only for display. Learned entries are colored; evaluated true
entries are dotted. The whole 0–100s window remains visible.
(c) Mean instantaneous spatial task RMS and full seed range for adaptive and
frozen own-trajectory control. The entire 80–100s window, including the dominant
shared initial transient, remains visible.
(d) Adaptive actual versus desired field at seed 732/site0 for the same full
control window. No independent prediction claim is made.

Additional supplement figure, 3.5 by 2.2 inches:
Whole-field mean row-projector error, with all-five-seed mean and range during
training and adaptive/frozen control. This is distinct from the selected-site
entry plot. Both figures mark the 80 s control onset; adaptation continues in the
adaptive branch. There are no guards/clips in these ideal control records.
Only two of five seeds met all strict training criteria; the figure does not
claim exact physical identification or robustness to failed sensing conditions.
Those results remain in the full protocol and unchanged original records.

figure_manifest.json binds input, source, transformations, diagnostic counts,
window metrics and artifacts. display_data.npz contains exact input slices and
computed means/ranges at full sample rate; render_indices specifies the display
subsampling. This local package does not itself establish public publication.
