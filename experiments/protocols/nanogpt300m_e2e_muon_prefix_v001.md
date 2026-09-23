# 300M hybrid-Muon shared prefix: offline submission preparation v001

This run is an `A2_EXTERNAL` end-to-end language-model trajectory, not a
theorem-facing plain-SGD experiment. The checked-in prefix configuration and
the generated attempt are launch-false. No shared-data upload or GPU job is
authorized by this preparation.

## Frozen starting settings

The 300M B0 job `6202317.0` measured a safe micro-batch of 32, about
180k training tokens/s, and 0.46984 seconds for one fixed 1,024-context probe
evaluation. The two prospective from-initialization v002 jobs `6209720.0`
and `6209721.0` each completed 8,192 updates with finite FP32 hybrid-Muon
state and zero clipping. Their last-four fixed-probe means were 3.67418 and
3.60725 CE. The gap exceeds the preregistered 0.01 near-tie band, so the
provisional nominal LR is `1e-4`. Keep the measured nearly non-clipping
control `33.504` rather than expand a clipping grid. This is an operational
prefix setting, not a claim that the LR or clip is globally optimal.

Use the same 303,473,664-parameter 20-layer/16-head/width-1,024 model,
256-token context, global batch 256, micro-batch 32, BF16 autocast, FP32
parameters and optimizer states, TF32, Flash-only SDPA, eager execution,
256-update linear warmup, initialization seed, and exact original 6.5B
training tape and validation probe as the calibrated jobs.

## Trajectory and observations

Train one Muon prefix from initialization for 79,346 canonical updates,
consuming 5,200,019,456 tokens. The existing v002 8,192-update checkpoint
cannot substitute for this formal trajectory: it has only 18 fixed-probe
evaluation ticks, whereas the frozen 6,000-point prefix grid requires early
coverage as well as fork-side density. Do not stitch that checkpoint into
the formal run and silently claim the missing evaluations were measured.

Freeze the nominal tail regular validation stride at one macro, based on
the measured sub-second fixed-probe cost. The prefix grid defined by
`prefix_validation_updates(1)` has exactly 6,000 unique ticks including
initialization and the fork. Its final 4,096 updates have 4,097 observations,
one at every update, matching the future tail's nominal intrinsic-time
sampling density. The earlier 1,903 observations are spread evenly. Save
fixed-probe validation CE as the primary metric and every-update training CE,
learning rate, nominal intrinsic time, nominal `B/eta`, preclip gradient
norm, clipping bit, and step duration as diagnostics. Muon's internal LR
adjustment makes nominal `T` and `B/eta` diagnostic, not SGD-theorem clocks.

At the fork, save model parameters, complete Muon and auxiliary AdamW
state, CPU/CUDA RNG states, data identity, source-tape cursor, update/token
ledger, nominal intrinsic time, and validation CE anchor. Four future tails
must resume this **same** checkpoint and ordered future tape. The full 6.5B
objects are SHA-addressed in one OSDF shared cache for later reuse; the
generated cache manifest remains `not_uploaded` until separately authorized
upload and remote byte/hash verification.

## Measured resource envelope and release gate

At approximately 176k training tokens/s, 5.2B prefix tokens imply about
8.21 H200 GPU-hours of training. Six thousand 0.46984-second evaluations
add about 0.78 H200 GPU-hours; the subtotal is approximately 9.0 hours,
before initial input transfer and final checkpoint/output transfer. The
single-H200 submit template requests 12 hours and 32 GB of worker disk,
leaving a measured-compute margin. The validated output is packed into one
tar file and transferred directly to OSDF staging, not the access point's
home directory. The cluster's willingness to match this
long H200 job is **not** established by offline materialization; confirm
policy and the staged files before submission. If the long-job request is
unavailable, split into resumable chunks rather than let a 9-hour job hit a
short duration cap.

Before any GPU launch: (1) verify the completed local 6.5B corpus, tape,
probe, B0, and v002 source/result hashes; (2) upload the four data objects
once to OSDF and verify remote sizes and SHA-256; (3) stage the code bundle,
config, wrapper, and JDL remotely and check the 12-hour H200 request; and
(4) obtain explicit authorization for this formal prefix run. A zero-exit
job alone is insufficient: replay the output validator over raw trace,
6,000 fixed-probe evaluations, and the complete checkpoint.
