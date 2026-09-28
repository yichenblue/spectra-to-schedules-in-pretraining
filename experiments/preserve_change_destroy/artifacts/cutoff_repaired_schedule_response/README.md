# Cutoff-repaired schedule response

This directory is the immutable paper-facing snapshot of the independently
reviewed co-mathematician workstream
`ws_567_rv_preserve_change_destroy_computation_verification`.

The fixed-spectrum calculation uses numerical cutoffs `10^12` and `10^14` on
the same time grid and with the same logarithmic bin width.  All 106
cutoff-only comparisons pass the predeclared `0.002` curve and exponent
tolerances.  The response table deliberately preserves 88 stable fits, 14
structurally censored points, and four inconclusive fits.

The compact paper-facing figure plots all non-structural finite-window fits
with the same observable-specific markers, without a separate inconclusive
legend entry.  Their audit statuses remain unchanged in
`layer_b_response_summary.csv`.

The ten finite-width rows compare true Gaussian online-SGD means with the exact
conditional finite-`W` Volterra recursion at `m=128`, `d=256`, using 1024
trajectories per anchor.  They are a finite-window authenticity diagnostic and
are not used for exponent fitting.

Regenerate the paper-facing figure from the repository root with:

```sh
MPLBACKEND=Agg \
MPLCONFIGDIR=/private/tmp/spectra-joint-schedules-schedule-response-mpl \
./.venv/bin/python -m \
  experiments.preserve_change_destroy.plot_cutoff_repaired_schedule_response
```

The full numerical generator, formula trace, independent review, and 17-test
gate remain in the completed workstream.  The copied `summary.json` retains
those original provenance paths by design.
