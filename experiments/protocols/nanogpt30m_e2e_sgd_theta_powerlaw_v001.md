# 30M intrinsic-time ratio-exponent experiment v001

## Question

At fixed model, corpus, global batch, optimizer, initialization, future data
order, and validation probe, does changing only the exponent of the prescribed
ratio path

\[
r_\vartheta(\tau)=\frac{B}{\eta(\tau)}
=r_0\left(1+\frac{\tau}{\tau_s}\right)^\vartheta
\]

change the decay rate of held-out cross-entropy?

This is an end-to-end LLM external-validity experiment (`A2_EXTERNAL`).  It is
not a direct theorem-phase or memory-kernel measurement.

## Frozen training controls

- nanoGPT: 6 layers, 6 heads, width 384, 30,036,864 parameters.
- GPT-2 vocabulary 50,304; block size 256; dropout and bias disabled.
- Plain SGD in FP32 with no momentum, weight decay, clipping, AMP, or TF32.
- Global batch size \(B=8\) in every arm.
- Base learning rate \(\eta_0=0.005941406469904266\).
- Initialization seed 2026090801 and tape seed 2026090802.
- OpenWebText 2.5B-token expanded corpus and the existing fixed 1,024-context
  document-disjoint validation probe.  Its first 700M tokens are byte-identical
  to the original corpus.

## Shared mature prefix

Train the first 196,608 rows of the existing aligned, without-replacement
PCG64 tape once at constant \(B=8\) and \(\eta_0\).  The resulting state must
reproduce the prior mature-snapshot state hash
`53ba7f6928e2dcb5690b2b2e11208ffb86f44fd49202118c0c21da77811a305d`.
Persist this checkpoint and fork every tail from it with a fresh, empty SGD
state.

## Five tail paths

Use exactly

\[
\vartheta\in\{0,\;0.5,\;0.75,\;1.0,\;1.5\}.
\]

For tail intrinsic time \(\tau_0=0\), set

\[
\tau_s=1024\eta_0,\qquad
X_n=1+\frac{\tau_n}{\tau_s},\qquad
\eta_n=\eta_0X_n^{-\vartheta},\qquad
\tau_{n+1}=\tau_n+\eta_n.
\]

All arms stop at the common endpoint \(\tau_{\max}=14\tau_s\), hence
\(X:1\to15\).  Only the terminal update may be shortened to land exactly on
the common intrinsic-time endpoint.  The five tails consume prefixes of the
same future tape; unequal update and token counts are part of the design.

Expected discrete tail update counts are 14,336, 38,977, 66,314, 114,687, and
356,523 in increasing \(\vartheta\) order.  The longest complete trajectory,
including the shared prefix, processes 1,132,812,288 tokens and fits inside the
2.5B corpus without repeating an aligned context.  Counting the prefix once
and all five tails, the campaign executes 787,445 optimizer updates and
processes 1,612,687,360 tokens in aggregate; repeated tokens across arms are
the intentional paired future-tape reuse, not within-arm resampling.

The original 700M-corpus PCG64 tape is retained exactly through \(X=10\).
Rows needed only for \(10<X\le15\) are appended from aligned contexts beyond
the original 700M-token boundary with extension seed 2026090803.  Therefore
the mature checkpoint and every previously established training example are
reused without alteration.

## Measurements

- Primary: mean CE on the unchanged fixed validation probe at 410 common,
  logarithmically spaced target values of \(X\) from 1 to 15.
- Retain all 1,024 per-context validation CEs.
- Retain per-update training CE, learning rate, intrinsic time, \(X\), update,
  and token coordinates for update/token-axis diagnostics.
- Record the checkpoint state hash, validation anchor hash, probe hash, and
  future-tape identity in every arm.

## Frozen analysis

Fit all five validation trajectories jointly on the full common window:

\[
L_\vartheta(X)=L_\infty+A_\vartheta X^{-q_\vartheta},
\]

using one shared \(L_\infty\).  Compare this arm-specific-exponent model with
a common-\(q\) null by RSS and BIC.  Report fixed-floor sensitivity and
63-point rolling local slopes, which preserve an approximately 0.15-decade
window on the 410-point grid.  If the fitted floor is boundary-dominated, the loss fails to
decrease, the common interval is shorter than one decade, or local/floor
sensitivity prevents stable exponents, classify exponent recovery as
inconclusive rather than forcing a phase label.

The primary figures after collection are validation CE and floor-centered CE
against intrinsic time.  Update and token axes, plus raw/EMA training CE, are
diagnostics only.

## Execution order

1. Run CPU preflight.
2. Run the single prefix job and collect its checkpoint.
3. Verify the checkpoint state hash against the prior mature snapshot.
4. Launch the five independent one-A100 tail jobs concurrently, one arm per GPU;
   no tail uses DDP.
5. Replay validation and run the frozen joint analysis offline.

No GPU submission is authorized by this protocol file.
