# Preserve--change--destroy experiment

This package implements the first paper-facing experiment for the current
manuscript.  It tests the schedule phase law on the frozen-feature PLRF model
at

\[
\alpha=0.4,\qquad \beta=0.3,\qquad
q_0=0.5,\qquad q_{\mathcal K}=0.75.
\]

With fixed averaged-gradient learning rate and

\[
B_t=\left\lceil B_0(1+T_t)^\vartheta\right\rceil,
\qquad T_t=\eta t,
\]

the three representative schedules are

- `theta=0.25`: destroy positive-power decay at the left boundary;
- `theta=0.50`: change the total exponent to `0.25`;
- `theta=0.90`: preserve the clean exponent `0.50`.

## Source-faithful implementation

The empirical matrix has the current-paper normalization

\[
W_{jk}\sim\mathcal N(0,1/m),\qquad
\widehat H=\Lambda^{1/2}WW^\top\Lambda^{1/2}.
\]

The exact solver uses

\[
q_t(z)=1-2\eta z+\left(1+\frac1{B_t}\right)\eta^2z^2,
\qquad c_t=\frac{\eta^2}{B_t}.
\]

It evolves clean modes, the noisy--clean gap, and the Volterra row mass
directly.  In particular, it never obtains a centered curve by subtracting two
nearly equal terminal risks.

One global `eta` is selected from the worst spectrum and trace over all widths
and feature seeds.  This keeps every realization on exactly the same intrinsic
time grid.  Both pointwise spectral stability and the empirical row-mass gate
are checked independently.

## Reused historical ideas

The code deliberately reuses only the following reviewed or independently
validated patterns from the co-mathematician workstreams:

- the smaller-Gram empirical spectrum and teacher-overlap construction;
- modal/covariance/Volterra cross-validation;
- the paired antithetic true-SGD estimator for the noise gap;
- local-slope stability diagnostics.

The varying-batch dynamics were rewritten from the current paper equations.
The package does **not** reuse the stationary summed-gradient polynomial as its
main solver, per-realization learning rates, forcing as a proxy for full-batch
GD, contour-DE centering, or a theory-guided best-window search.

## Commands

Run all tests from the repository root:

```sh
./.venv/bin/python -m unittest discover \
  -s experiments/preserve_change_destroy/tests -v
```

Run the three-layer finite-width diagnostic used in the paper appendix:

```sh
MPLCONFIGDIR=/private/tmp/paper-experiments-exp0-mpl \
./.venv/bin/python -m \
  experiments.preserve_change_destroy.run_experiment_0_finite_rf_sgd_bridge

MPLCONFIGDIR=/private/tmp/paper-experiments-exp0b-mpl \
./.venv/bin/python -m \
  experiments.preserve_change_destroy.run_experiment_0b_finite_w_de_bridge

MPLCONFIGDIR=/private/tmp/paper-experiments-exp0c-mpl \
./.venv/bin/python -m \
  experiments.preserve_change_destroy.plot_experiment_0c_three_layer_bridge
```

Experiment 0 compares true online-SGD means with the exact conditional
finite-`W` Volterra recursion.  Experiment 0b compares that recursion with the
support-adaptive resolvent deterministic equivalent on the same discrete
schedules.  Experiment 0c verifies the shared-object contract, merges the two
bridges at `m=512`, and writes the paper-facing vector figure to
`paper_figures/volterra_three_layer_bridge.pdf`.
The Experiment 0b solver and its C backend are bundled in this package; its
checked-in density files are optional caches, so no historical workstream is
needed at runtime.

Run the separate finite-bulk three-layer bridge:

```sh
MPLCONFIGDIR=/private/tmp/paper-experiments-fb-bridge-mpl \
./.venv/bin/python -m \
  experiments.preserve_change_destroy.run_fb_three_layer_bridge \
  --profile full
```

This bridge uses representative `FB1=(0.20,0.40)` and
`FB2=(0.12,0.65)` settings, the finite-bulk learning-rate scaling
`eta_m=0.05*B0*m^(2*alpha-1)`, and the triangular horizon
`T_max(m)=2*m^(2*alpha)/log(m)`.  Its three batch ramps have
`theta={0,1,2}` and represent linear, logarithmic, and summable cumulative
injection.  They are not the LM/IM boundary/change/preserve labels.  Results
and the paper-ready vector figure are written under
`artifacts/fb_three_layer_bridge_full` without changing the existing LM/IM
bridge or manuscript sources.

Plot the independently reviewed LM/IM schedule-response diagnostic:

```sh
MPLBACKEND=Agg \
MPLCONFIGDIR=/private/tmp/paper-experiments-schedule-response-mpl \
./.venv/bin/python -m \
  experiments.preserve_change_destroy.plot_cutoff_repaired_schedule_response
```

Its exponent panels use a fixed-infinite-spectrum Volterra quadrature with the
unresolved numerical tail included in the centered observable.  The primary
spectral cutoff is `10^12`, and an independent cutoff-only rerun at `10^14`
must pass the fixed curve and exponent tolerances.  The third panel summarizes
ten same-feature, finite-window comparisons between true Gaussian online SGD
and the exact conditional finite-`W` Volterra recursion; it is an authenticity
check and is not used to fit exponents.  The reviewed CSV/JSON snapshot and
paper-facing figure inputs live under
`artifacts/cutoff_repaired_schedule_response`.

Run the fast end-to-end profile:

```sh
MPLCONFIGDIR=/private/tmp/paper-experiments-pcd-mpl \
./.venv/bin/python -m experiments.preserve_change_destroy.run \
  --profile smoke
```

Run the preregistered main sweep explicitly:

```sh
MPLCONFIGDIR=/private/tmp/paper-experiments-pcd-mpl \
./.venv/bin/python -m experiments.preserve_change_destroy.run \
  --profile main
```

Run the scalable head-preserving spectral-quadrature backend:

```sh
MPLCONFIGDIR=/private/tmp/paper-experiments-pcd-de-mpl \
./.venv/bin/python -m experiments.preserve_change_destroy.run_de \
  --profile smoke

MPLCONFIGDIR=/private/tmp/paper-experiments-pcd-de-mpl \
./.venv/bin/python -m experiments.preserve_change_destroy.run_de \
  --profile main
```

Run the matched-spectrum authenticity bridge:

```sh
MPLCONFIGDIR=/private/tmp/paper-experiments-pcd-bridge-mpl \
./.venv/bin/python -m experiments.preserve_change_destroy.run_bridge \
  --profile smoke

MPLCONFIGDIR=/private/tmp/paper-experiments-pcd-bridge-mpl \
./.venv/bin/python -m experiments.preserve_change_destroy.run_bridge \
  --profile main

MPLCONFIGDIR=/private/tmp/paper-experiments-pcd-bridge-mpl \
./.venv/bin/python -m experiments.preserve_change_destroy.run_bridge \
  --profile large
```

Run the full eight-schedule experiment-one diagnostic at `m=10000`:

```sh
MPLCONFIGDIR=/private/tmp/paper-experiments-pcd-exp1-m10000-mpl \
./.venv/bin/python \
  -m experiments.preserve_change_destroy.run_experiment_one_m10000
```

Run the locked finite-width validation and fixed-equation extrapolation:

```sh
MPLCONFIGDIR=/private/tmp/paper-experiments-pcd-finite-size-mpl \
./.venv/bin/python \
  -m experiments.preserve_change_destroy.run_finite_size_experiment_one
```

This protocol uses `m=512,1024,2048,4096` for bridge validation and reserves
the unchanged eight-schedule `m=10000` artifact as a locked holdout.  It then
checks the smooth-schedule continuum against the integer-schedule continuum
before running the same equations at finite effective widths from `10^4` to
`10^48`.  No theory-guided window choice or fitted finite-width correction is
allowed.  At `10^4` and `10^6` the nominal window spans fewer than the required
two decades, so their fixed-window slopes are marked descriptive only; they
show finite-size drift but cannot pass the theory audit.

Run the preregistered label-noise crossover sweep:

```sh
MPLCONFIGDIR=/private/tmp/paper-experiments-pcd-noise-sweep-mpl \
./.venv/bin/python \
  -m experiments.preserve_change_destroy.run_noise_sweep
```

The sweep freezes `sigma2={1,10,100,1000}` and
`theta={0.25,0.50,0.90}`.  It defines the destroy/change crossover as the
first upcrossing of `gap/clean=1`, and the preserve crossover as the final
downcrossing after the gap/clean peak.  Unit-noise and variance-ten true-SGD
runs use common random numbers at `m=10000`; their directly checked pathwise
homogeneity supplies the two larger variances.  The previously validated
`m_eff=10^30` continuum trajectory resolves crossover times that are censored
by the finite `m=10000`, `T<=100` bridge.  Censored events remain explicit.

Generate the post-sweep exploratory experiment-one phase plot at
`m=10000, sigma2=100`:

```sh
MPLCONFIGDIR=/private/tmp/paper-experiments-pcd-exp1-sigma100-mpl \
./.venv/bin/python \
  -m experiments.preserve_change_destroy.run_experiment_one_sigma100
```

This runner retains all eight schedules, the original fixed window, and the
original local-slope gate.  It reconstructs variance 100 from the locked
variance-ten bridge using the independently checked pathwise noise-response
homogeneity.  Because variance 100 was selected after viewing the noise sweep,
the artifact is labeled exploratory even when its point estimates are closer
to the theory curve.

The DE backend is deliberately separate from the finite-`W` experiment.  It
uses the head-preserving PLRF spectral proxy

\[
\lambda_j=j^{-2\alpha},\qquad
w_j^{\rm clean}=j^{-2(\alpha+\beta)},
\]

retains every leading mode, and logarithmically compresses only the smooth
tail.  Its nominal triangular window is

\[
m^{0.2}\leq T\leq m^{0.5},
\]

so both `T -> infinity` and `T / m^(2 alpha) -> 0`.  The ratio amplitude and
`sigma2` are fixed across effective widths.  The main profile reaches
effective width `10^48` without constructing a dense random-feature matrix.

Every DE run repeats the largest-width calculation with half the logarithmic
spectral-bin width and twice the time resolution.  Curve and fitted-slope
differences must pass explicit gates.  The backend is a continuum spectral
proxy; it is not presented as a contour-DE replacement for a finite random
feature realization.

The smoke profile exercises every route but is explicitly not paper-ready
exponent evidence.  The main profile still fails closed unless the
preregistered triangular windows and local-slope stability checks pass.

The bridge fixes the same analytic spectrum, label-noise variance, integer
mini-batch schedule, and intrinsic grid across three routes: true Gaussian
online SGD, the exact discrete second-moment recursion, and the scheduled
continuum modal solver.  Gaussian rotational invariance samples each full
mini-batch update exactly in distribution at `O(m)` rather than `O(B_t*m)`
cost.  The main bridge uses widths `256,512,1024,2048` and reports clean,
noisy--clean gap, and noisy-total curve and effective-slope discrepancies.

## Artifacts

Each run writes:

- `summary.json`: source scope, calibration, stability gates, fit status, and
  reuse/exclusion contract;
- `curves.csv`: seed-mean trajectories and standard errors;
- `slope_audits.csv`: all fixed-window and window-sensitivity fits;
- `paired_sgd_validation.csv`: conditional true-SGD checks;
- `preserve_change_destroy.{png,pdf}`: the three-panel figure.

The DE backend writes the analogous files under `artifacts/de_*`, plus
`spectral_profiles.csv` and `resolution_checks.csv`.

The authenticity bridge writes `bridge_curves.csv`, `bridge_metrics.csv`, a
machine-readable `summary.json`, and one 3-by-4 overlay figure per schedule
exponent under `artifacts/bridge_*`.

The `large` bridge extends the unchanged contract to widths `4096` and
`10000`.  It uses 128 stochastic trajectories per width and schedule, keeps
all 10,000 spectral modes as singletons, and runs 800 actual stochastic
updates at the largest width.  Reduced trajectory count is never hidden:
standard errors and standardized residuals remain in the curve and metric
artifacts.

The dedicated `m=10000` experiment-one run keeps the authenticity and phase
claims separate.  It first requires true SGD, the discrete exact recursion,
and the continuum route to agree.  It then audits the centered-risk slopes
against the joint-limit phase curve on the fixed window
`m^0.2 <= T <= m^0.45`.  A successful authenticity bridge does not override
an inconclusive finite-width phase-law audit.

The finite-size protocol writes its combined three-panel figure and compact
gate summary under `artifacts/finite_size_experiment_one`, with the training
bridge under `artifacts/bridge_finite_size_train` and the continuum sweep
under `artifacts/de_finite_size_extrapolation`.

The noise sweep writes `noise_sweep_curves.csv`, `crossover_metrics.csv`,
`response_checks.csv`, a compact gate summary, and its figure under
`artifacts/noise_sweep`.

The exploratory variance-100 phase plot and its curve, slope, and bridge
tables are written under `artifacts/experiment_one_m10000_sigma100`.
