#!/usr/bin/env python3
"""Prepare two shared-input H200 LR arms, conditional extension, or clip control."""

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
from experiments.nanogpt_local import e2e_300m_muon_calibration_v002 as core  # noqa: E402


PLAN = EXPERIMENTS / "configs/nanogpt300m_e2e_6p5b_v001.json"
REMOTE_ROOT = "/home/wang3587/e2e-b0-transfer/experiments/chtc/nanogpt300m_e2e_muon_calibration_v002"
OSDF_ROOT = "osdf:///chtc/staging/w/wang3587/nanogpt300m-e2e-muon-calibration-v002"
CONTAINER = (
    "docker.io/pytorch/pytorch:2.13.0-cuda12.6-cudnn9-runtime@sha256:"
    "6acf597eeb8e376a96580dde4952f37cc017fef732bb40bfc73f28f25e3f64b4"
)
WINDOW = 257
CHUNK_CONTEXTS = 8_192
INPUT_NAMES = ("pretrain_train.bin", "training_tape.npy", "pretrain_validation.bin", "validation_probe.npy")
ARM_TAGS = {core.LR_CANDIDATES[0]: "lr7p5e05", core.LR_CANDIDATES[1]: "lr1e04"}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n", encoding="utf-8")


def _pack_windows(stream: np.ndarray, offsets: np.ndarray, target: Path) -> None:
    if offsets.ndim != 1 or len(offsets) == 0 or int(np.min(offsets)) < 0 or int(np.max(offsets)) + WINDOW > len(stream):
        raise RuntimeError("v002 source window escapes pinned token stream")
    with target.open("xb") as writer:
        for begin in range(0, len(offsets), CHUNK_CONTEXTS):
            block = np.asarray(offsets[begin:begin + CHUNK_CONTEXTS], dtype=np.int64)
            positions = block[:, None] + np.arange(WINDOW, dtype=np.int64)[None, :]
            np.ascontiguousarray(stream[positions], dtype="<u2").tofile(writer)


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
        "experiments/nanogpt_local/e2e_300m_muon_calibration_v001.py",
        "experiments/nanogpt_local/e2e_300m_muon_calibration_v002.py",
        "experiments/configs/nanogpt300m_e2e_6p5b_v001.json",
        "experiments/configs/nanogpt300m_e2e_muon_calibration_v002.json",
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


def _pack_shared_inputs(output: Path, plan: dict, source_begin: int) -> dict:
    folder = ROOT / plan["data"]["expanded_corpus"]
    train = np.memmap(folder / "pretrain_train.bin", dtype="<u2", mode="r")
    tape = np.load(folder / "training_tape.npy", mmap_mode="r", allow_pickle=False)
    validation = np.memmap(folder / "pretrain_validation.bin", dtype="<u2", mode="r")
    probe = np.load(folder / "validation_probe.npy", allow_pickle=False)
    first = source_begin * core.prior.GLOBAL_BATCH
    last = (source_begin + core.STAGE_UPDATES) * core.prior.GLOBAL_BATCH
    if len(tape) < last or len(probe) != 1_024:
        raise RuntimeError("v002 tape/probe ended before planned stage")
    inputs = output / "inputs"
    inputs.mkdir()
    _pack_windows(train, tape[first:last], inputs / INPUT_NAMES[0])
    _pack_windows(validation, probe, inputs / INPUT_NAMES[2])
    contexts = last - first
    np.save(inputs / INPUT_NAMES[1], np.arange(contexts, dtype="<i8") * WINDOW, allow_pickle=False)
    np.save(inputs / INPUT_NAMES[3], np.arange(1_024, dtype="<i8") * WINDOW, allow_pickle=False)
    return {
        "schema_version": core.INPUT_SCHEMA,
        "run_id": core.RUN_ID,
        "campaign_id": plan["campaign_id"],
        "source_update_begin": source_begin,
        "packed_tape_updates": core.STAGE_UPDATES,
        "source_data_sha256": core._source_identity(plan),
        "files": {name: {"sha256": _sha256(inputs / name), "size_bytes": (inputs / name).stat().st_size} for name in INPUT_NAMES},
    }


def _input_uris(manifest: dict) -> dict[str, str]:
    if set(manifest.get("files", {})) != set(INPUT_NAMES):
        raise RuntimeError("v002 OSDF cache inventory changed")
    return {
        name: f"{OSDF_ROOT}/{manifest['files'][name]['sha256']}/{name}"
        for name in INPUT_NAMES
    }


def _job_text(
    remote_stage: str, tag: str, lr: float, clip: float,
    phase: str, input_uris: dict[str, str], remote_bundle: str,
    remote_wrapper: str, remote_manifest: str,
    remote_checkpoint: str | None = None,
) -> str:
    remote_arm = f"{remote_stage}/{tag}"
    transferred = [remote_bundle, remote_manifest]
    transferred.extend(input_uris[name] for name in INPUT_NAMES)
    if remote_checkpoint is not None:
        transferred.append(remote_checkpoint)
    return "\n".join((
        "# One independent 300M hybrid-Muon prospective calibration arm.",
        "universe = container",
        f"container_image = docker://{CONTAINER}",
        "docker_override_entrypoint = true",
        f"initialdir = {remote_arm}",
        f"executable = {remote_wrapper}",
        "transfer_executable = true",
        f"arguments = {lr:.6g} {phase} {clip:.3f}",
        "getenv = false",
        "should_transfer_files = YES",
        "when_to_transfer_output = ON_EXIT",
        "transfer_input_files = " + ",".join(transferred),
        "transfer_output_files = calibration_output",
        "requirements = (TARGET.HasDocker =?= true)",
        "request_gpus = 1",
        'require_gpus = (DeviceName == "NVIDIA H200")',
        "gpus_minimum_capability = 9.0",
        "gpus_minimum_memory = 140000MB",
        "gpus_minimum_runtime = 12.6",
        "request_cpus = 4",
        "request_memory = 32768MB",
        "request_disk = 16384MB",
        "allowed_job_duration = 5400",
        "+WantGPULab = true",
        '+GPUJobLength = "short"',
        f'+ExperimentRunId = "{core.RUN_ID}"',
        f'+ExperimentAttemptId = "{core.RUN_ID}-{phase}-{tag}"',
        "log = logs/$(Cluster)_$(Process).log",
        "output = logs/$(Cluster)_$(Process).out",
        "error = logs/$(Cluster)_$(Process).err",
        "queue 1",
        "",
    ))


def _arm_result(folder: Path, learning_rate: float, completed: int) -> dict:
    value = json.loads((folder / "result.json").read_text(encoding="utf-8"))
    if (
        value.get("schema_version") != core.RESULT_SCHEMA
        or value.get("run_id") != core.RUN_ID
        or value.get("arm", {}).get("learning_rate") != learning_rate
        or value["arm"].get("gradient_clip") != core.CONTROL_CLIP
        or value["arm"].get("completed_updates") != completed
        or not isinstance(value.get("checkpoint_sha256"), str)
    ):
        raise RuntimeError("v002 previous arm result cannot support continuation")
    return value


def materialize(
    output: Path, phase: str = "initial", result_dirs: tuple[Path, Path] | None = None,
) -> dict:
    if output.exists() or output.is_symlink():
        raise RuntimeError(f"refusing to overwrite v002 attempt: {output}")
    if phase not in ("initial", "extension", "clip_control"):
        raise RuntimeError("unknown prospective calibration phase")
    if phase != "initial" and result_dirs is None:
        raise RuntimeError("conditional v002 phase requires both prior result directories")
    plan = json.loads(PLAN.read_text(encoding="utf-8"))
    scale.validate_plan(plan, ROOT, verify_data=phase != "clip_control")
    core.validate_config(ROOT, plan)
    decision = core.decide(result_dirs) if result_dirs is not None else None
    if phase == "extension" and decision["status"] != "extend_both_to_16384":
        raise RuntimeError("extension was not triggered by the frozen tie rule")
    if phase == "clip_control" and decision["status"] not in (
        "provisional_lr_candidate_clip_unresolved", "provisional_lr_near_tie_clip_unresolved",
    ):
        raise RuntimeError("clipping control requires a prospective LR candidate")
    output.mkdir(parents=True)
    remote_base = f"{REMOTE_ROOT}/a001"
    remote_stage = f"{REMOTE_ROOT}/" + {"initial": "a001", "extension": "a002", "clip_control": "a003"}[phase]
    files = []
    if phase in ("initial", "extension"):
        manifest = _pack_shared_inputs(output, plan, 0 if phase == "initial" else core.STAGE_UPDATES)
        files.extend(output / "inputs" / name for name in INPUT_NAMES)
    else:
        manifest = None
    if phase == "clip_control":
        initial_manifest_path = ROOT / "experiments/generated" / core.RUN_ID / "a001/input_manifest.json"
        initial_manifest = json.loads(initial_manifest_path.read_text(encoding="utf-8"))
        if initial_manifest.get("source_update_begin") != 0:
            raise RuntimeError("conditional clip control lost initial shared input identity")
        input_uris = _input_uris(initial_manifest)
    else:
        input_uris = _input_uris(manifest)
    if phase == "initial":
        files.append(_bundle(output))
        wrapper = output / "run_once.sh"
        wrapper.write_bytes((PACKAGE / "run_once.sh").read_bytes())
        wrapper.chmod(0o755)
        files.append(wrapper)
    selected = (
        core.LR_CANDIDATES if phase != "clip_control"
        else (decision["provisional_learning_rate"],)
    )
    for lr in selected:
        tag = ARM_TAGS[lr] if phase != "clip_control" else f"{ARM_TAGS[lr]}-clip10"
        arm = output / tag
        arm.mkdir()
        remote_manifest = f"{remote_stage}/{tag}/input_manifest.json"
        remote_checkpoint = None
        if phase == "initial":
            remote_manifest = f"{remote_stage}/input_manifest.json"
        elif phase == "extension":
            previous = _arm_result(result_dirs[core.LR_CANDIDATES.index(lr)], lr, core.STAGE_UPDATES)
            extension_manifest = dict(manifest)
            extension_manifest["resume_checkpoint_sha256"] = previous["checkpoint_sha256"]
            local_manifest = arm / "input_manifest.json"
            _write_json(local_manifest, extension_manifest)
            files.append(local_manifest)
            remote_checkpoint = f"{remote_base}/{ARM_TAGS[lr]}/calibration_output/checkpoint.pt"
        elif phase == "clip_control":
            remote_manifest = f"{remote_base}/input_manifest.json"
        jdl = arm / "job.sub"
        runtime_stage = remote_stage if phase == "initial" else remote_base
        jdl.write_text(_job_text(
            remote_stage, tag, lr,
            core.STARTUP_CLIP_CONTROL if phase == "clip_control" else core.CONTROL_CLIP,
            "initial" if phase != "extension" else "extension",
            input_uris, f"{runtime_stage}/code_bundle.tar",
            f"{runtime_stage}/run_once.sh", remote_manifest, remote_checkpoint,
        ), encoding="utf-8")
        files.append(jdl)
    if phase == "initial":
        manifest_path = output / "input_manifest.json"
        _write_json(manifest_path, manifest)
        files.append(manifest_path)
    if phase in ("initial", "extension"):
        cache = {
            "schema_version": "nanogpt300m_e2e_muon_calibration_shared_cache_v002",
            "run_id": core.RUN_ID,
            "status": "not_uploaded",
            "upload_each_sha256_once": True,
            "inputs": {
                name: {**manifest["files"][name], "local_path": str((output / "inputs" / name).resolve()),
                       "osdf_uri": input_uris[name], "upload_status": "not_uploaded"}
                for name in INPUT_NAMES
            },
        }
        cache_path = output / "shared_cache_manifest.json"
        _write_json(cache_path, cache)
        files.append(cache_path)
    report = {
        "schema_version": "nanogpt300m_e2e_muon_calibration_materialization_v002",
        "run_id": core.RUN_ID,
        "phase": phase,
        "status": "prepared_launch_false",
        "remote_stage": remote_stage,
        "arm_count": len(selected),
        "predicted_gpu_hours_training_only": len(selected) * .85,
        "maximum_gpu_hours": len(selected) * 1.5,
        "reuses_early_shared_remote_inputs": phase == "clip_control",
        "osdf_inputs": input_uris,
        "shared_cache_upload_status": "not_uploaded" if phase != "clip_control" else "reuse_initial_upload_state_to_verify",
        "full_6p5b_dataset_uploaded": False,
        "files": {str(path.relative_to(output)): {"sha256": _sha256(path), "size_bytes": path.stat().st_size} for path in files},
    }
    _write_json(output / "materialization.json", report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--phase", choices=("initial", "extension", "clip_control"), default="initial")
    parser.add_argument("--low-result-dir", type=Path)
    parser.add_argument("--mid-result-dir", type=Path)
    args = parser.parse_args()
    results = None if args.low_result_dir is None and args.mid_result_dir is None else (args.low_result_dir, args.mid_result_dir)
    if results is not None and any(path is None for path in results):
        parser.error("both previous arm result directories are required")
    print(json.dumps(materialize(args.output_dir, args.phase, results), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
