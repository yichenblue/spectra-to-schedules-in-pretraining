#!/usr/bin/env bash
set -euo pipefail

arm_lr="$1"
phase="$2"
arm_clip="$3"

mkdir -p inputs runtime
for input_name in pretrain_train.bin training_tape.npy pretrain_validation.bin validation_probe.npy
do
  mv "${input_name}" "inputs/${input_name}"
done

tar -xf code_bundle.tar -C runtime
export PYTHONPATH="${PWD}/runtime"
export PYTHONUNBUFFERED=1
python_bin="$(command -v python3 || command -v python || true)"
test -n "${python_bin}"

start_update=0
resume_args=()
if [[ "${phase}" == "extension" ]]
then
  start_update=8192
  resume_args=(--resume-checkpoint checkpoint.pt)
else
  test "${phase}" == "initial"
fi

"${python_bin}" -m experiments.nanogpt_local.e2e_300m_muon_calibration_v002 \
  --plan runtime/experiments/configs/nanogpt300m_e2e_6p5b_v001.json \
  --input-dir inputs --input-manifest input_manifest.json \
  --output-dir calibration_output \
  --learning-rate "${arm_lr}" --gradient-clip "${arm_clip}" \
  --start-update "${start_update}" "${resume_args[@]}"

test -s calibration_output/result.json
test -s calibration_output/trace.npz
test -s calibration_output/checkpoint.pt
