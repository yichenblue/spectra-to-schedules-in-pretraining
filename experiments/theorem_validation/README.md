# Theorem-validation experiments

These deterministic experiments complement the existing
`preserve_change_destroy` figure.  Together they follow the manuscript's
`spectral origin -> schedule transformation -> schedule design` narrative.

## Experiment C: SGD spectral-loss classification pilot

`spectral_loss_sgd.py` trains an actual linear predictor with plain SGD on
fresh Gaussian covariates under six prescribed spectrum/target mechanisms:
canonical polynomial PLRF, a mass-compensated sparse target, stretched-
exponential eigenvalues, a spectral gap, geometrically spaced eigenvalues,
and a polynomial spectrum with a rapidly decaying target.  Population excess
risk is evaluated exactly from the parameter error.  The checked-in run is a
pilot and the `--smoke` route is implementation-only; neither is confirmatory
without frozen classification and fit-window gates.

The completed pilot exposed two predeclared failure modes of a global power
fit: finite-width cutoff in polynomial spectra and an apparently excellent
power fit for a geometrically spaced spectrum with log-periodic local-slope
oscillation.  The follow-up configuration
`spectral_loss_sgd_confirmation_v002.json` therefore freezes the primary fit
window at intrinsic time `[8, 128]`, uses widths 1024 and 4096 with five new
paired seeds, and treats local-slope and width stability as primary evidence.
The canonical and sparse polynomial targets must agree within 0.08 in fitted
power exponent at width 4096; the spectral-gap case must favor an exponential
fit by at least 0.05 in R-squared.  Stretched-exponential and geometric cases
are diagnosed through structured slope drift or oscillation, not global
R-squared alone.  The rapidly decaying target remains exploratory.

Confirmation v001 is retained as a failed numerical preflight: its all-width
gapped covariance violated the Gaussian-SGD mean-square stability condition
through the trace-of-covariance term at width 4096.  V002 changes only that
counterexample to a width-independent rank-64 gapped covariance and adds the
missing trace term to the preflight stability check; it does not reuse v001's
response values.

## Experiment D: polynomial width--time extension

`spectral_width_time.py` evaluates the exact discrete modal forcing for the
canonical polynomial and mass-compensated sparse targets.  It co-scales the
terminal clock with width according to `T <= 0.0025 d^0.8`, which bounds the
analytic omitted-tail fraction by five percent.  The primary width is
`d=2^22`, the frozen primary fit window is `T in [16, 256]`, and the predicted
forcing exponent is one half.  This experiment deliberately isolates the
paper's forcing estimand; it does not relabel an unseparated minibatch-SGD
total-risk curve as forcing.

`spectral_width_time_full_sgd.py` is the corresponding complete-risk
extension.  For Gaussian diagonal features it solves the exact expected clean
minibatch-SGD Volterra recursion, including sampling-noise feedback.  Modal
sums use doubled geometric spectral quadrature at the primary width, with a
frozen convergence gate; forcing remains visible only as a diagnostic overlay.

`spectral_loss_sgd_width_time.py` is the literal sampled counterpart over all
six prior constructions. Every update draws a fresh dense Gaussian minibatch,
forms its empirical gradient, and updates the complete parameter vector. It
uses neither an analytic recurrence nor spectral binning; consequently its
feasible widths are much smaller than those of the expectation calculation.

## Experiment A: spectral criterion

`spectral_criterion.py` evaluates the two exact conditional objects in
`iclr:thm:spectral_iff`: low-spectrum cutoff mass and its discrete SGD modal
transform.  The plotted curves use the theorem's Gamma constants; no curve is
vertically fitted.  Forcing and memory are gated separately over a frozen
pre-saturation window, and the spectral quadrature is doubled.

This is a prescribed effective-spectrum diagnostic conditional on frozen
eigenpairs.  It does not prove a PLRF population-to-empirical random-feature
bridge and it does not plot the full loss.

Suggested paper caption: "Prescribed effective-spectrum diagnostic for the
spectral criterion, conditional on a frozen representation.  Panels (a)--(b)
compare the cutoff spectral tails with the Gamma-scaled exact modal dynamics;
there are no fitted vertical offsets.  Panel (c) compares their exponents over
the common frozen pre-saturation window.  The truncation scale is a quadrature
control, not a neural-network parameter count."

## Experiment B: ratio control

`ratio_control.py` evaluates the exact fixed-horizon continuum objective

    J(r) = integral w(u) / r(u) du,  integral r(u) du = D,

and its square-root minimizer.  Constant and time-reversed paths have the same
sample budget.  Equal-sample quantiles then give a deterministic unit-batch
schedule, which is run through the discrete modal second-moment recursion over
a co-scaled Phase-IIIa data sweep.

The unconstrained square-root profile is an exact optimizer of the continuum
proxy (equivalently, the nonbinding-bounds case), not an exact finite-step risk
optimizer.  The data sweep supports one representative achievability branch;
it does not validate the lower bound over every admissible schedule or all six
rate classes.

Suggested paper caption: "Same-budget ratio control in one representative
Phase-IIIa cell ($\alpha=3/4$, $\beta=1/4$, $p=1$, and $\sigma^2=1$).
Panel (a) verifies the square-root minimizer of the phase-level objective
against two pre-fixed controls.  Panel (b) realizes it with exactly $D$ unit
batches.  Panel (c) runs each realized schedule through its own exact modal
recursion and recovers the predicted $D^{-1/2}$ risk law.  This is an integer
achievability diagnostic, not a finite-step global-optimality claim or a test
of all phase classes.  The short black segment is a visually offset slope
guide, not a prefactor prediction."

## Reproduce

From the repository root, use the repository interpreter:

```bash
./.venv/bin/python -m experiments.theorem_validation.run_all
./.venv/bin/python -m unittest discover \
  -s experiments/theorem_validation/tests -v
```

Artifacts are written beneath `artifacts/spectral_criterion` and
`artifacts/ratio_control_phase_iiia`.  The earlier deterministic Phase-IIa
development run is retained at `artifacts/ratio_control`; it failed only the
predeclared exact-risk exponent gate (measured 0.342 versus theory 2/7, error
0.0566) and is not claim-bearing.
