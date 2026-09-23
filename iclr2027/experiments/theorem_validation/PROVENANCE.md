# Provenance and error controls

## Reused implementation ideas

- Head-preserving logarithmic spectral bins follow
  `ws_443_noisy_plrf_de_volterra_validation` and
  `preserve_change_destroy/de_quadrature.py`: retain every leading mode and
  compress only the smooth index tail.
- The positive modal second-moment recursion follows
  `preserve_change_destroy/dynamics.py`; the noise gap is evolved directly,
  never obtained by subtracting noisy and clean losses.
- Frozen log-uniform fits, local-slope variation, separate pointwise/row
  stability, and curve/slope resolution gates follow the existing
  `preserve_change_destroy` experiment contract.
- The ratio experiment audits base, spectrum-only refined, time-only refined,
  and jointly refined routes separately for centered clean risk, the directly
  evolved noise gap, and total risk; this prevents cancellation between either
  numerical axes or loss components.
- The ratio optimizer and equal-sample quantile construction implement
  `thm:td_lrs_ratio_control` and `lem:td_lrs_integer_reachability` directly.

## Historical errors explicitly excluded

- A ratio path does not determine exact finite-step dynamics.  Every integer
  schedule is therefore rerun through its own modal propagators.
- Pointwise modal contraction and Volterra row stability are checked
  separately.
- The continuum proxy is not relabeled as the exact conditional risk, and no
  absolute-prefactor gate compares those two objects.
- Fixed finite spectra eventually decay exponentially.  The spectral theorem
  fit is restricted to a pre-saturation joint window.
- Analytic effective-spectrum quadrature is not described as an empirical
  Gaussian random-feature draw.
- No per-curve fitted vertical offset is used in the spectral Gamma collapse.
- Every summary stores the base/refined grid contract and SHA-256 fingerprints
  of the generating code and theorem source files.  Unit tests include the
  memory prefactor, a hand-computed one-step direct-gap recurrence, and JSON
  boolean-schema checks.

## Promotion status

Artifacts remain `PILOT_UNPROMOTED`: the repository-wide co-mathematician
health check is blocked by pre-existing guarded-source drift in another paper
directory.  These scripts do not edit guarded manuscript sources.
