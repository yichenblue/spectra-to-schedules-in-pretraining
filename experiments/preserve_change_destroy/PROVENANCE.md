# Historical-code provenance and exclusions

This experiment was written against the current manuscript equations.  The
older co-mathematician workstreams were used as implementation references,
not as a source of automatically paper-ready claims.

## Reused patterns

| Current component | Historical reference | What was retained |
| --- | --- | --- |
| `spectrum.py` | `ws_443_noisy_plrf_de_volterra_validation/src/noisy_plrf_de_volterra_validation.py:finite_w_modes` | smaller-Gram eigendecomposition, empirical teacher overlaps, and null-space energy |
| `dynamics.py` validation routes | `ws_454_..._computation_verification/src/plrf_verification.py` | independent modal, original-coordinate covariance, and Volterra comparisons |
| `monte_carlo.py` | `ws_443_noisy_plrf_de_volterra_validation/src/validate_noise_forcing_with_paired_sgd.py` | common-input antithetic label noise and direct gap estimation |
| `slopes.py` | `ws_443_noisy_plrf_de_volterra_validation/src/fit_compute_optimal_exponents.py` | local log-slope computation and a quantitative slope-variation diagnostic |
| varying-schedule formula | approved `ws_511_prove_the_exact_variable_batch_variable_averaged_learning_rate_mode_recursion_an` | exact averaged-gradient modal polynomial, injection coefficient, and separate stability conditions |
| exact small-case test | `ws_505_low_regularity_phase_ib_time_dependent_lrs_extension_computation_verification` | dense/spectral two-time kernel and forward-solve cross-check pattern |
| `de_quadrature.py` | `ws_443_noisy_plrf_de_volterra_validation/src/fit_nontrace_presaturation_spectral_de.py` | head-preserving leading modes, logarithmic compression of only the smooth spectral tail, chunked profile evaluation, and resolution doubling |
| `resolvent_de.py`, `stieltjes_exact_density.c` | audited support-adaptive zero-regularization solver from `ws_443_noisy_plrf_de_volterra_validation` | minimal local extraction used by Experiment 0b; checked-in densities are optional caches and cache misses are recomputed by the bundled C backend |
| `bridge_validation.py` | current exact modal equations plus Gaussian rotational invariance | distributionally exact mini-batch-gradient sampling, with explicit small-width moment tests against the discrete recursion |

The old workstream notation used `v` for the ambient dimension and `d` for
feature width.  The current code uses the paper's `d` and `m`, respectively.
Consequently the old numerical scaling `N(0,1/d_old)` maps to the current
`N(0,1/m)`; no extra rescaling is applied.

Experiment 0b has no runtime dependency on the historical workstream tree.
The local Python wrapper and C backend, together with the optional density
caches, are sufficient to reproduce its deterministic-equivalent curves.

## Deliberately rewritten

- The main one-step polynomial is the current averaged-gradient expression
  with stepwise integer `B_t`.  The old stationary summed-gradient `q_gamma`
  appears only in a unit test of the fixed-batch substitution
  `eta = B * gamma`.
- The clean modes, noisy--clean gap, and row mass are propagated directly.
  No noisy/clean terminal subtraction is used.
- One global learning rate is selected before any curve is fitted.  It is not
  adapted to feature seeds or widths.
- Fits use declared physical windows.  Theory values never select a window.

## Historical failures converted into guards

- `ws_395` treated an SGD forcing curve as full-batch GD.  This experiment
  makes no GD comparison.
- `ws_396` used realization-dependent step sizes and produced misaligned
  intrinsic-time grids.  `aggregate_curves` rejects unequal grids.
- Early `ws_443` deterministic-equivalent runs averaged spectral-edge atoms,
  subtracted poorly resolved plateaus, lost precision in `1-mu`, or missed
  narrow support.  The finite experiment uses the raw empirical spectrum and
  no contour inversion.  The scalable backend retains every leading mode and
  rejects the failed equal-mass compression route; it never computes a small
  gap by subtracting a near-unit eigenvalue.
- Several `ws_443` exponent fits were pre-asymptotic.  `slopes.py` returns
  `INCONCLUSIVE` when the declared decade span or local-slope stability gate
  fails.
- `ws_511` supplied counterexamples showing that pointwise contraction and
  Volterra row stability are distinct.  Both are recorded and checked here.

`summary.json` repeats these inclusions and exclusions so generated artifacts
retain their implementation contract outside the source tree.

The scalable backend additionally fixes `sigma2` and the intrinsic inverse
ratio amplitude across every effective width, fits only on declared
triangular windows, and requires the refined spectral/time grid to reproduce
both curves and slopes.  It is labeled a head-preserving continuum spectral
proxy rather than a finite-random-feature deterministic equivalent.

The authenticity bridge deliberately uses the deterministic analytic
power-law spectrum in all three routes.  It tests the discrete-to-continuum
and stochastic-to-expectation links without mixing in empirical random-feature
spectrum fluctuations.  It does not claim finite-dataset sample reuse,
nonlinear representation learning, or empirical-spectrum equivalence.

The finite-size experiment-one protocol adds no historical correction fit.
Widths through `4096` validate the stochastic/discrete/continuum bridge;
`m=10000` is a locked holdout; and the smooth-continuum transition is gated on
all three directly evolved observables before extrapolation.  Small-width
fixed-window slopes that fail the preregistered decade-span audit remain
explicitly descriptive and are never promoted to claim-bearing evidence.

The label-noise sweep reuses the exact linear-response structure already
checked in the historical noisy-PLRF workstream, but independently verifies it
under the current bridge normalization and common random numbers.  It never
selects a variance to improve a phase-law fit.  Preserve crossovers beyond the
finite `m=10000` horizon are reported as censored there and are resolved only
with the separately validated smooth-continuum proxy.

The dedicated `sigma2=100, m=10000` experiment-one artifact is a post-sweep
exploratory view.  It uses the same fixed window and strict local-slope audit
as the original experiment and records point-error agreement separately from
plateau stability.  It is not promoted to preregistered phase-law evidence.
