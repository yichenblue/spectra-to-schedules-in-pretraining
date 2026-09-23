# Three-arm theta extension

This launch-false package adds the three exploratory arms `theta1p25`,
`theta1p75`, and `theta2` to the completed 30M campaign.  It reuses the
collected mature checkpoint and the already-uploaded immutable 5B-token OSDF
data object.  The 5B corpus begins with the exact bytes of the prior 2.5B
corpus.  The deterministic future tape preserves every established 2.5B row,
then extends without replacement into the 5B-only portion for `theta2`.  Model,
optimizer, batch size, terminal coordinate `X=15`, and the 410-point validation
contract are unchanged.

The exact tail-update counts are 201,066, 638,210, and 1,151,656.  Measured
throughput from the completed lower-theta arms predicts approximately 1.73,
5.21, and 9.30 A100 GPU-hours, respectively.  The common hard runtime cap is
10.5 GPU-hours per arm.

Attempt `a006` replaces the infrastructure-failed `a005`.  The only execution
change is `request_disk=16384MB`, increased from 8192MB so the approximately
10 GB shared training object fits in the worker scratch directory.  No
scientific setting, input hash, schedule, or checkpoint changes.

Prepare locally with the repository interpreter:

```bash
./.venv/bin/python \
  experiments/chtc/nanogpt30m_e2e_sgd_theta_powerlaw_high_v001/materialize_campaign.py \
  --output-dir /tmp/nanogpt30m-theta-high-a006 \
  --checkpoint /path/to/prefix_checkpoint.pt \
  --attempt a006
```

Materialization does not authorize or perform GPU submission.
