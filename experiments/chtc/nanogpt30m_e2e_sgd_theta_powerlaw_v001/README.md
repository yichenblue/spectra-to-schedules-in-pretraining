# nanoGPT 30M theta-powerlaw campaign

This package stages the minimal two-phase GPU experiment specified in
`experiments/protocols/nanogpt30m_e2e_sgd_theta_powerlaw_v001.md`.

The execution order is deliberately split:

1. Reuse the collected historical 196,608-update mature
   `prefix_checkpoint.pt`.
2. Require its tensor-state hash to equal the historical mature snapshot.
3. Start five tail jobs from that one checkpoint for
   `theta0`, `theta0p5`, `theta0p75`, `theta1`, and `theta1p5`.
4. Analyze the five collected tail directories offline.

The tail submit file queues all five independent one-A100 jobs without a
concurrency throttle, so all five may run simultaneously when the pool has five
matching GPUs.  Each tail remains a single-GPU run; there is no DDP.

All GPU jobs reuse the existing immutable OSDF objects for the expanded 2.5B
training stream, validation stream, and validation probe.  Dataset files must
not be uploaded separately for each tail.  The original tape is preserved
through \(X=10\), and only the new \(10<X\le15\) rows use the expanded region.

The core training module has no network or submission capability.  This
directory will contain only launch materialization and shell entry points;
creating those files does not authorize a paid GPU submission.

Local commands must use the repository interpreter:

```bash
./.venv/bin/python -m experiments.nanogpt_local.e2e_sgd_theta_powerlaw_30m_v001 \
  --preflight \
  --config experiments/configs/nanogpt30m_e2e_sgd_theta_powerlaw_v001.json
```

The prefix materializer remains available for audit only; the X=15 extension
does not require rerunning it:

```bash
./.venv/bin/python \
  experiments/chtc/nanogpt30m_e2e_sgd_theta_powerlaw_v001/materialize_campaign.py \
  prepare-prefix \
  --output-dir /tmp/nanogpt30m-theta-prefix-a001 \
  --attempt a001
```

After the prefix output has been collected and its state hash has passed,
prepare all five tail jobs around that one checkpoint:

```bash
./.venv/bin/python \
  experiments/chtc/nanogpt30m_e2e_sgd_theta_powerlaw_v001/materialize_campaign.py \
  prepare-tails \
  --checkpoint /path/to/prefix_output/prefix_checkpoint.pt \
  --output-dir /tmp/nanogpt30m-theta-tails-a001 \
  --attempt a001
```

Both materialization modes refuse to replace an existing output directory.
They create submit descriptions but never run `ssh`, upload, or call
`condor_submit`.

After all tails are collected:

```bash
./.venv/bin/python -m experiments.nanogpt_local.e2e_sgd_theta_powerlaw_30m_analysis_v001 \
  --campaign-root PATH_TO_COLLECTED_TAILS \
  --output-dir experiments/results/nanogpt30m-e2e-sgd-theta-powerlaw-v001
```

No submission is authorized until the user explicitly authorizes the named
attempt.
