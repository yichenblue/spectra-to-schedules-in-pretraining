#!/usr/bin/env bash
set -euo pipefail

# HTCondor downloads immutable OSDF inputs by basename. Recreate the paths
# frozen in prefix_config.json before starting the one shared prefix.
mkdir -p inputs
for input_name in \
  metadata.json \
  openwebtext_cooldown_2p5b_source_manifest_complete.json \
  pretrain_train.bin \
  pretrain_validation.bin \
  validation_probe.npy \
  training_tape.npy
do
  if [[ -f "${input_name}" ]]; then
    mv "${input_name}" "inputs/${input_name}"
  fi
  test -f "inputs/${input_name}"
done

mkdir -p runtime
tar -xf code_bundle.tar -C runtime

export PYTHONPATH="${PWD}/runtime"
export PYTHONUNBUFFERED=1

PYTHON_BIN="$(command -v python3 || command -v python || true)"
if [[ -z "${PYTHON_BIN}" ]]; then
  echo "no Python interpreter found in the container PATH" >&2
  exit 127
fi

"${PYTHON_BIN}" - <<'PY'
import numpy
import torch

if not torch.cuda.is_available():
    raise RuntimeError("CUDA is unavailable")
if torch.cuda.device_count() != 1:
    raise RuntimeError(f"expected one visible GPU, got {torch.cuda.device_count()}")
if "H200" not in torch.cuda.get_device_name(0):
    raise RuntimeError(f"expected H200, got {torch.cuda.get_device_name(0)!r}")
if not torch.cuda.is_bf16_supported():
    raise RuntimeError("visible H200 does not report BF16 support")
print(
    {
        "numpy": numpy.__version__,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "device": torch.cuda.get_device_name(0),
    },
    flush=True,
)
PY

"${PYTHON_BIN}" -m \
  experiments.nanogpt_local.e2e_sgd_rpath_factorization_124m_2p5b_v002 \
  --run-prefix \
  --config prefix_config.json \
  --output-dir prefix_output

test -s prefix_output/result.json
test -s prefix_output/prefix_checkpoint.pt
test -s prefix_output/training_trace.npz
test -s prefix_output/validation_evaluations.npz
