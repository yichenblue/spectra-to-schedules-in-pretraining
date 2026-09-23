# Language-model data snapshot

This directory contains the smallest frozen result set needed to redraw the four language-model figures in the manuscript.

The 30M and 124M SGD NPZ files were publication-slimmed by removing the per-context `context_cross_entropies`/`context_cross_entropy` arrays. All aggregate axes and validation-cross-entropy arrays used by the paper plots are unchanged. The removed arrays are useful for re-running uncertainty and probe-level validation gates, but they are not read by the paper plotting code.

The 300M Muon validation files were already compact and are retained without dropping fields. The surrogate directory retains the fitted parameters, accepted summaries, observed/predicted CSV rows, and the no-refit WSD transfer rows.

Training data, optimizer/model checkpoints, full per-update traces, and HTCondor logs are excluded. Fresh training is driven by the code, frozen configurations, protocols, and launch material in `experiments/`.
