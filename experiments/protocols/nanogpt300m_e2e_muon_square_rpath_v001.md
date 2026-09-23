# 300M Muon WSD/8-1-1 square-ratio tails v001

## Purpose

This is an `A2_EXTERNAL` follow-up to the completed 300M hybrid-Muon
factorization calibration.  The calibration selected nominal
`B/eta^2` matching over `B/eta` matching on one mature prefix and one future
tape.  This run asks whether that practical matching rule continues to align
the full WSD 80/20 and 8-1-1 validation-loss trajectories.

The experiment is not a theorem-facing plain-SGD test.  `B/eta^2` and
`sum(eta^2)` are empirical Muon clocks selected by the calibration; they are
not substituted into the paper's SGD claims.

## Frozen square-ratio construction

Let `B0=256`, `eta0=1e-4`, and let one integer macro have source width `h`
canonical batches.  The two factorizations are:

- fixed batch / LR schedule: `h` optimizer updates at `B=B0` and
  `eta=eta0/sqrt(h)`;
- fixed LR / batch schedule: one optimizer update at `B=B0*h` and
  `eta=eta0`.

Within every macro, both arms consume exactly `B0*h` contexts, have the same
pointwise `B/eta^2=(B0/eta0^2)h`, and add exactly `eta0^2` to
`sum(eta^2)`.  They deliberately do not match `sum(eta)`.  Consequently the
primary comparison axis is `sum(eta^2)`, with tokens since fork as the main
operational axis and `sum(eta)` retained only as a diagnostic.

The continuous source LR fraction is the original WSD 80/20 exponential or
8-1-1 path.  Integer macro widths are obtained by inverse-clock rounding of
the squared LR fraction.  The frozen result is 4,265 WSD macros with
`1 <= h <= 99`, and 1,091 8-1-1 macros with `10 <= h <= 100`.  The largest
fixed-LR batch is 25,600 sequences and is executed with micro-batch 32;
there is no single-tensor batch allocation of that size.

## Shared state and measurements

All four tails restore the same completed 79,346-update 300M Muon prefix,
including the complete Muon and auxiliary AdamW optimizer state.  They reuse
the existing shared 6.5B corpus, training tape, fixed validation probe, and
uploaded checkpoint without re-uploading them.  Every arm consumes the same
5,078,272 future contexts (1,300,037,632 tokens) in the same order.

Fixed-probe validation CE is recorded at the fork and after every macro:
4,266 values for each WSD arm and 1,092 values for each 8-1-1 arm.  Per-update
training CE, learning rate, batch, both clocks, clipping, and timing are also
retained for diagnostics.  Submission remains unauthorized until explicitly
requested.
