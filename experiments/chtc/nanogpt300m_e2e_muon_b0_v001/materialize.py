#!/usr/bin/env python3
"""Pack only the 300M H200 B0 windows; never upload the 6.5B stream."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path
import sys
import tarfile

import numpy as np


PACKAGE = Path(__file__).resolve().parent
ROOT = PACKAGE.parents[2]
EXPERIMENTS = PACKAGE.parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.nanogpt_local import e2e_300m_6p5b_plan_v001 as scale  # noqa: E402


PLAN = EXPERIMENTS / "configs/nanogpt300m_e2e_6p5b_v001.json"
RUN_ID = "nanogpt300m-e2e-muon-b0-v001"
ATTEMPT = "a001"
REMOTE = "/home/wang3587/e2e-b0-transfer/experiments/chtc/nanogpt300m_e2e_muon_b0_v001/a001"
CONTAINER = (
    "docker.io/pytorch/pytorch:2.13.0-cuda12.6-cudnn9-runtime@sha256:"
    "6acf597eeb8e376a96580dde4952f37cc017fef732bb40bfc73f28f25e3f64b4"
)
WINDOW = 257
TRAIN_CONTEXTS = 4 * 256
PROBE_CONTEXTS = 1_024


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical(value: dict) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode("utf-8")


def _windows(stream: np.ndarray, offsets: np.ndarray) -> np.ndarray:
    if offsets.ndim != 1 or int(np.min(offsets)) < 0 or int(np.max(offsets)) + WINDOW > len(stream):
        raise RuntimeError("B0 source window escapes the frozen token stream")
    positions = np.asarray(offsets, dtype=np.int64)[:, None] + np.arange(WINDOW, dtype=np.int64)[None, :]
    return np.ascontiguousarray(stream[positions], dtype="<u2").reshape(-1)


def _bundle(output: Path) -> Path:
    members = (
        "experiments/__init__.py",
        "experiments/nanogpt_local/__init__.py",
        "experiments/nanogpt_local/model.py",
        "experiments/nanogpt_local/build_openwebtext_cooldown_corpus_v001.py",
        "experiments/nanogpt_local/build_openwebtext_cooldown_6p5b_append_v001.py",
        "experiments/nanogpt_local/muon_124m_v001.py",
        "experiments/nanogpt_local/muon_300m_v001.py",
        "experiments/nanogpt_local/e2e_300m_6p5b_plan_v001.py",
        "experiments/nanogpt_local/e2e_300m_b0_v001.py",
        "experiments/configs/nanogpt300m_e2e_6p5b_v001.json",
        "experiments/configs/openwebtext_cooldown_6p5b_expansion_source_plan_v001.json",
    )
    target = output / "code_bundle.tar"
    with tarfile.open(target, "w") as archive:
        for name in members:
            source = ROOT / name
            payload = source.read_bytes()
            info = tarfile.TarInfo(name)
            info.size = len(payload)
            info.mode = 0o644
            info.mtime = 0
            info.uid = info.gid = 0
            info.uname = info.gname = ""
            archive.addfile(info, io.BytesIO(payload))
    return target


def materialize(output: Path) -> dict:
    if output.exists() or output.is_symlink():
        raise RuntimeError(f"refusing to overwrite existing B0 attempt: {output}")
    plan = json.loads(PLAN.read_text(encoding="utf-8"))
    scale.validate_plan(plan, ROOT, verify_data=True)
    if plan["campaign_id"] != "nanogpt300m-e2e-6p5b-v001":
        raise RuntimeError("300M campaign identity changed")
    folder = ROOT / plan["data"]["expanded_corpus"]
    train = np.memmap(folder / "pretrain_train.bin", dtype="<u2", mode="r")
    tape = np.load(folder / "training_tape.npy", mmap_mode="r", allow_pickle=False)
    validation = np.memmap(folder / "pretrain_validation.bin", dtype="<u2", mode="r")
    probe = np.load(folder / "validation_probe.npy", allow_pickle=False)
    if len(tape) < TRAIN_CONTEXTS or len(probe) != PROBE_CONTEXTS:
        raise RuntimeError("B0 source tape or fixed validation probe changed")
    output.mkdir(parents=True)
    inputs = output / "inputs"
    inputs.mkdir()
    _windows(train, tape[:TRAIN_CONTEXTS]).tofile(inputs / "pretrain_train.bin")
    _windows(validation, probe).tofile(inputs / "pretrain_validation.bin")
    np.save(inputs / "training_tape.npy", np.arange(TRAIN_CONTEXTS, dtype="<i8") * WINDOW, allow_pickle=False)
    np.save(inputs / "validation_probe.npy", np.arange(PROBE_CONTEXTS, dtype="<i8") * WINDOW, allow_pickle=False)
    names = ("pretrain_train.bin", "training_tape.npy", "pretrain_validation.bin", "validation_probe.npy")
    manifest = {
        "schema_version": "nanogpt300m_e2e_muon_b0_inputs_v001",
        "campaign_id": plan["campaign_id"],
        "source_data_sha256": {
            key: plan["data"][key] for key in (
                "train_sha256", "training_tape_file_sha256",
                "validation_sha256", "validation_probe_sha256",
            )
        },
        "files": {
            name: {"sha256": _sha256(inputs / name), "size_bytes": (inputs / name).stat().st_size}
            for name in names
        },
    }
    (output / "input_manifest.json").write_bytes(_canonical(manifest))
    bundle = _bundle(output)
    wrapper = output / "run_once.sh"
    wrapper.write_bytes((PACKAGE / "run_once.sh").read_bytes())
    wrapper.chmod(0o755)
    jdl = output / "job.sub"
    jdl.write_text("\n".join((
        "# One 300M hybrid-Muon runtime B0; packed windows only, no formal training.",
        "universe = container",
        f"container_image = docker://{CONTAINER}",
        "docker_override_entrypoint = true",
        f"initialdir = {REMOTE}",
        f"executable = {REMOTE}/run_once.sh",
        "transfer_executable = true",
        "getenv = false",
        "should_transfer_files = YES",
        "when_to_transfer_output = ON_EXIT",
        "transfer_input_files = " + ",".join((
            f"{REMOTE}/code_bundle.tar", f"{REMOTE}/input_manifest.json",
            *(f"{REMOTE}/inputs/{name}" for name in names),
        )),
        "transfer_output_files = b0_output",
        "requirements = (TARGET.HasDocker =?= true)",
        "request_gpus = 1",
        'require_gpus = (DeviceName == "NVIDIA H200")',
        "gpus_minimum_capability = 9.0",
        "gpus_minimum_memory = 140000MB",
        "gpus_minimum_runtime = 12.6",
        "request_cpus = 4",
        "request_memory = 32768MB",
        "request_disk = 24576MB",
        "allowed_job_duration = 1800",
        "+WantGPULab = true",
        '+GPUJobLength = "short"',
        f'+ExperimentRunId = "{RUN_ID}"',
        f'+ExperimentAttemptId = "{RUN_ID}-{ATTEMPT}"',
        "log = logs/a001_$(Cluster)_$(Process).log",
        "output = logs/a001_$(Cluster)_$(Process).out",
        "error = logs/a001_$(Cluster)_$(Process).err",
        "queue 1",
        "",
    )), encoding="utf-8")
    files = (bundle, wrapper, jdl, output / "input_manifest.json", *(inputs / name for name in names))
    report = {
        "schema_version": "nanogpt300m_e2e_muon_b0_materialization_v001",
        "run_id": RUN_ID,
        "attempt_id": ATTEMPT,
        "status": "prepared_launch_false",
        "remote_directory": REMOTE,
        "maximum_gpu_hours": 0.5,
        "full_6p5b_dataset_uploaded": False,
        "files": {str(path.relative_to(output)): {"sha256": _sha256(path), "size_bytes": path.stat().st_size} for path in files},
    }
    (output / "materialization.json").write_bytes(_canonical(report))
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(materialize(args.output_dir), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
