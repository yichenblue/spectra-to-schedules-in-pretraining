# nanoGPT 124M 2.5B accelerated ratio-path campaign v002

This directory contains launch-false local materialization and worker files
for the BF16-autocast, FP32-parameter, TF32, Flash-only revision of the 124M
plain-SGD campaign. The model, data, tape, schedules, optimizer, token count,
and training-time grid are unchanged from v001. The measurement-only
validation grid is expanded to match the 300M point counts and relative
density. A fresh v002 B0, prefix, and four tails are required; a v001 FP32
checkpoint is intentionally incompatible.

From the experiment repository root:

```bash
./.venv/bin/python -m experiments.nanogpt_local.e2e_sgd_rpath_factorization_124m_2p5b_v002 --preflight
./.venv/bin/python experiments/chtc/nanogpt124m_e2e_sgd_rpath_factorization_2p5b_v002/materialize_campaign.py plan
```

The immutable corpus and tape are not uploaded again. `freeze-data` verifies
the local bytes and emits v002 records whose OSDF URIs point to the existing
SHA-addressed v001 cache:

```bash
./.venv/bin/python experiments/chtc/nanogpt124m_e2e_sgd_rpath_factorization_2p5b_v002/materialize_campaign.py freeze-data \
  --data-dir data/openwebtext-cooldown-2p5b-v001 \
  --source-manifest experiments/configs/openwebtext_cooldown_2p5b_source_manifest_complete_v001.json \
  --output-dir /tmp/nanogpt124m-rpath-v002-data-freeze
```

After `freeze-data`, materialize the one-job, launch-false H200 B0 package:

```bash
./.venv/bin/python experiments/chtc/nanogpt124m_e2e_sgd_rpath_factorization_2p5b_v002/materialize_b0.py \
  --data-freeze-dir /path/to/v002/data-freeze-v001 \
  --output-dir /new/absent/b0-launch-a001 \
  --remote-dir /home/wang3587/e2e-b0-transfer/experiments/chtc/nanogpt124m_e2e_sgd_rpath_factorization_2p5b_v002/b0-a001
```

The package reuses the six SHA-addressed v001 OSDF objects. It contains one
exact-H200 Condor job with a 0.25 GPU-hour hard ceiling; materialization alone
does not authorize submission.

Run and collect the response-free B0 before freezing the prefix budget:

```bash
./.venv/bin/python experiments/chtc/nanogpt124m_e2e_sgd_rpath_factorization_2p5b_v002/materialize_campaign.py freeze-prefix \
  --data-manifest /path/to/v002/data_manifest.json \
  --b0-result /path/to/v002/b0/result.json \
  --output-dir /tmp/nanogpt124m-rpath-v002-prefix-freeze
```

After the B0 result passes offline replay and `freeze-prefix` succeeds,
materialize the launch-false shared-prefix package:

```bash
./.venv/bin/python experiments/chtc/nanogpt124m_e2e_sgd_rpath_factorization_2p5b_v002/materialize_prefix.py \
  --output-dir /new/absent/prefix-launch-a001 \
  --remote-dir /home/wang3587/e2e-b0-transfer/experiments/chtc/nanogpt124m_e2e_sgd_rpath_factorization_2p5b_v002/prefix-a001
```

This creates one exact-H200 job. Its predicted and hard wall-time budgets are
derived from the collected B0 rather than copied from v001.

Only a collected and fully validated v002 prefix may seed the four tails:

```bash
./.venv/bin/python experiments/chtc/nanogpt124m_e2e_sgd_rpath_factorization_2p5b_v002/materialize_campaign.py freeze-tails \
  --data-manifest /path/to/v002/data_manifest.json \
  --checkpoint /path/to/v002/prefix_checkpoint.pt \
  --prefix-result /path/to/v002/result.json \
  --output-dir /tmp/nanogpt124m-rpath-v002-tail-freeze
```

Every worker code bundle must contain:

- `experiments/nanogpt_local/e2e_sgd_rpath_factorization_124m_2p5b_v002.py`;
- `experiments/nanogpt_local/model.py`;
- `experiments/configs/nanogpt124m_e2e_sgd_rpath_factorization_2p5b_v002.json`;
- the package `__init__.py` files.

None of these utilities uploads data, logs in to CHTC, or submits work.
`materialize_b0.py` and `materialize_prefix.py` create launch-false submit
files for separate, user-authorized staging and submission. All emitted
configs retain `cluster_submission_authorized=false`.
