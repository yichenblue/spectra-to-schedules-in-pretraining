# Dense theta extension on 0 < theta < 0.5

This launch-false package adds the three exploratory arms
`theta0p125`, `theta0p25`, and `theta0p375` to the completed 30M campaign.
It reuses the exact collected mature checkpoint, immutable OSDF data objects,
future tape, optimizer, terminal coordinate `X=15`, and 410-point validation
probe contract.  It does not rerun the shared prefix or re-upload the dataset.

Prepare locally with the repository interpreter:

```bash
./.venv/bin/python \
  experiments/chtc/nanogpt30m_e2e_sgd_theta_powerlaw_dense01_v001/materialize_campaign.py \
  --output-dir /tmp/nanogpt30m-theta-dense01-a003 \
  --checkpoint /path/to/prefix_checkpoint.pt \
  --attempt a003
```

Materialization does not authorize or perform GPU submission.
