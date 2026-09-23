#!/usr/bin/env bash
set -euo pipefail

mkdir -p inputs
for input_name in pretrain_train.bin training_tape.npy pretrain_validation.bin validation_probe.npy
do
  test -f "${input_name}"
  mv "${input_name}" "inputs/${input_name}"
done

mkdir -p runtime
tar -xf code_bundle.tar -C runtime
export PYTHONPATH="${PWD}/runtime"
export PYTHONUNBUFFERED=1
PYTHON_BIN="$(command -v python3 || command -v python || true)"
test -n "${PYTHON_BIN}"

"${PYTHON_BIN}" -m experiments.nanogpt_local.e2e_300m_muon_prefix_v001 \
  --run-prefix --config prefix_config.json --output-dir prefix_output
"${PYTHON_BIN}" -m experiments.nanogpt_local.e2e_300m_muon_prefix_v001 \
  --validate-result --config prefix_config.json --output-dir prefix_output \
  > prefix_output/offline_validation.json

test -s prefix_output/result.json
test -s prefix_output/training_trace.npz
test -s prefix_output/validation_evaluations.npz
test -s prefix_output/prefix_checkpoint.pt
test -s prefix_output/offline_validation.json
tar -cf prefix_output.tar prefix_output
test -s prefix_output.tar
