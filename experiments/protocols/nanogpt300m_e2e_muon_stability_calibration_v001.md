# 300M hybrid-Muon learning-rate/clipping gate v001

This is one `A2_EXTERNAL` H200 short-window calibration, not the 6.5B formal
prefix. It uses the same 303,473,664-parameter model, 256 × 256-token global
batch, BF16/Flash runtime, and fixed 1,024-context validation probe as the
300M campaign. Data windows are extracted byte-for-byte from the first 2,304
updates of the pinned 6.5B source tape. Only those windows are transferred;
the full 6.5B dataset remains a separately authorized upload.

1. From initialization, run 128 updates at nominal LR `1.5e-4`, with a
   256-update linear LR warmup and clip 1.0. Record the **preclip** global
   gradient norm each update. This diagnostic is discarded as a training
   state. Set the second clip candidate to `1.25 × p95(preclip norm)`, rounded
   to three decimals; it must exceed 1.0.
2. Restart from the same initialization and train one shared 2,048-update
   maturity state at `1.5e-4` with that second clip. The first 256 updates
   linearly warm up LR. Save fixed-probe CE at updates 0, 256, 1,024, and
   2,048 and retain every-update training CE, preclip norm, and clipping bit.
3. From **identical model and complete hybrid-optimizer state**, fork six
   256-update branches: LR `7.5e-5`, `1.5e-4`, or `3e-4`, crossed with clip
   1.0 or the measured second threshold. All branches consume the same future
   256-update tape. Evaluate at branch updates 0, 128, and 256 and retain
   per-update diagnostics.

The default decision is to retain center LR `1.5e-4` and the measured second
clip only if shared-maturity and center branch fixed-probe CE both decrease,
the hybrid optimizer states are finite FP32, and clipping affects at most
10% of updates in both segments. The two outer LRs are stability guardrails,
not a CE-minimization search. If the gate fails, report `short_window_no_go`
and do not freeze formal settings. Even `short_window_go` does not authorize
formal prefix GPU submission; it only provides nominal LR/clipping inputs
for that separately gated run.

The B0 H200 throughput was approximately 180k tokens/s. The pilot executes
128 + 2,048 + 6 × 256 = 3,712 optimizer updates, or about 243M training
tokens across diagnostic and branch-reused tapes. Training-time projection is
about 22.5 minutes before transfer and checkpoint overhead; predicted use is
0.42 H200-hours, with a hard one-H200-hour job duration. These are resource
planning figures, not a completed-job measurement.
