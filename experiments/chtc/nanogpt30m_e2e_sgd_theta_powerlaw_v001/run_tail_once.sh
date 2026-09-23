#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -ne 1 ]]; then
  echo "usage: run_tail_once.sh THETA" >&2
  exit 2
fi
THETA="$1"

mkdir -p runtime/data/openwebtext-cooldown-2p5b-v001
tar -xf code_bundle.tar -C runtime

stage_input() {
  local source="$1"
  local destination="$2"
  if [[ ! -f "${destination}" ]]; then
    mv "${source}" "${destination}"
  fi
  test -s "${destination}"
}

stage_input \
  pretrain_train.bin \
  runtime/data/openwebtext-cooldown-2p5b-v001/pretrain_train.bin
stage_input \
  pretrain_validation.bin \
  runtime/data/openwebtext-cooldown-2p5b-v001/pretrain_validation.bin
stage_input \
  validation_probe.npy \
  runtime/data/openwebtext-cooldown-2p5b-v001/validation_probe.npy

mkdir -p runtime/experiments/configs
mv theta_config.json \
  runtime/experiments/configs/nanogpt30m_e2e_sgd_theta_powerlaw_v001.json
test -s prefix_checkpoint.pt

export PYTHONPATH="${PWD}/runtime"
export CUBLAS_WORKSPACE_CONFIG=":4096:8"
export PYTHONUNBUFFERED=1

PYTHON_BIN="$(command -v python3 || command -v python || true)"
if [[ -z "${PYTHON_BIN}" ]]; then
  echo "no Python interpreter found in the container PATH" >&2
  exit 127
fi

"${PYTHON_BIN}" - <<'PY'
import torch

if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
    raise RuntimeError("expected exactly one visible CUDA GPU")
if "A100" not in torch.cuda.get_device_name(0):
    raise RuntimeError(f"expected A100, got {torch.cuda.get_device_name(0)!r}")
print({"torch": torch.__version__, "cuda": torch.version.cuda,
       "device": torch.cuda.get_device_name(0)}, flush=True)
PY

"${PYTHON_BIN}" -m \
  experiments.nanogpt_local.e2e_sgd_theta_powerlaw_30m_v001 \
  --run-tail \
  --theta "${THETA}" \
  --checkpoint prefix_checkpoint.pt \
  --config runtime/experiments/configs/nanogpt30m_e2e_sgd_theta_powerlaw_v001.json \
  --output-dir tail_output

test -s tail_output/result.json
test -s tail_output/training_trace.npz
test -s tail_output/validation_evaluations.npz
