# 300M Muon factorization calibration v001

## Scope

This is a short external-validity calibration for the repository's current hybrid Muon optimizer. It is not theorem-facing and does not assert that a pure-Muon scaling law has been identified.

## Shared start and future tape

All arms load the same completed 300M prefix checkpoint, including model parameters, optimizer states, and RNG states. They consume the same next 524,288 ordered training contexts (134,217,728 tokens) beginning at checkpoint tape cursor 20,312,576. The fixed 1,024-context validation probe is the primary metric; per-update training cross-entropy is diagnostic.

## Arms

| Arm | Batch (B) | Learning rate \(\eta\) | Updates | Matched to anchor |
|---|---:|---:|---:|---|
| anchor | 256 | \(10^{-4}\) | 2,048 | reference |
| linear | 128 | \(5\times10^{-5}\) | 4,096 | \(B/\eta\), equivalently endpoint \(\sum\eta\) |
| square | 128 | \(10^{-4}/\sqrt{2}\) | 4,096 | \(B/\eta^2\), equivalently endpoint \(\sum\eta^2\) |

All other optimizer and runtime settings remain frozen. Validation is evaluated every 8,192 consumed contexts, including the common start and endpoint, for 65 samples per arm.

## Decision rule

On the common source-fraction grid, compute fixed-probe CE RMSE for anchor versus linear and anchor versus square. Also report endpoint gaps and RMSE normalized by the anchor's absolute loss decrease. A candidate is a clear winner only if its RMSE is at most half the competing RMSE. Otherwise the calibration is inconclusive.

Because this is one checkpoint and one future tape, it can select the more promising factorization rule for the next experiment but cannot establish uncertainty, universality, or a paper-level scaling law.

## Compute and launch boundary

Each arm is an independent single-H200 job; no DDP is used. The shared dataset cache is reused without upload. The 2.5 GB prefix checkpoint is uploaded once under its SHA-256 identity and reused by all three jobs. Materialization is launch-false; cluster submission requires separate user authorization.
