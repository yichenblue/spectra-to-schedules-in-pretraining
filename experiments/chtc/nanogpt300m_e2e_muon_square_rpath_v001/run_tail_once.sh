#!/usr/bin/env bash
set -euo pipefail

ARM="${1:?missing tail arm}"
mkdir -p inputs
for input_name in pretrain_train.bin training_tape.npy pretrain_validation.bin validation_probe.npy
do
  test -f "${input_name}"
  mv "${input_name}" "inputs/${input_name}"
done

test -f prefix_checkpoint.pt
mkdir -p runtime
tar -xf code_bundle.tar -C runtime
export PYTHONPATH="${PWD}/runtime"
export PYTHONUNBUFFERED=1
PYTHON_BIN="$(command -v python3 || command -v python || true)"
test -n "${PYTHON_BIN}"

"${PYTHON_BIN}" -m experiments.nanogpt_local.e2e_300m_muon_square_rpath_v001 \
  --run-arm --config tail_config.json --arm "${ARM}" \
  --checkpoint prefix_checkpoint.pt --output-dir tail_output
"${PYTHON_BIN}" -m experiments.nanogpt_local.e2e_300m_muon_square_rpath_v001 \
  --validate-result --config tail_config.json --arm "${ARM}" \
  --output-dir tail_output > tail_output/offline_validation.json

test -s tail_output/result.json
test -s tail_output/training_trace.npz
test -s tail_output/validation_evaluations.npz
test -s tail_output/offline_validation.json
tar -cf tail_output.tar tail_output
test -s tail_output.tar
