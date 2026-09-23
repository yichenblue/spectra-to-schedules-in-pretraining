#!/usr/bin/env bash
set -euo pipefail

mkdir -p inputs runtime
for input_name in pretrain_train.bin training_tape.npy pretrain_validation.bin validation_probe.npy
do
  mv "${input_name}" "inputs/${input_name}"
done

tar -xf code_bundle.tar -C runtime
export PYTHONPATH="${PWD}/runtime"
export PYTHONUNBUFFERED=1
PYTHON_BIN="$(command -v python3 || command -v python || true)"
test -n "${PYTHON_BIN}"

"${PYTHON_BIN}" -m experiments.nanogpt_local.e2e_300m_b0_v001 \
  --plan runtime/experiments/configs/nanogpt300m_e2e_6p5b_v001.json \
  --input-dir inputs --input-manifest input_manifest.json --output-dir b0_output

test -s b0_output/result.json
