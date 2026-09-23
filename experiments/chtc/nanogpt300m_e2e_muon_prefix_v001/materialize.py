#!/usr/bin/env python3
"""Build a launch-false shared-cache and one-H200 formal prefix package."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path
import sys
import tarfile


PACKAGE = Path(__file__).resolve().parent
ROOT = PACKAGE.parents[2]
EXPERIMENTS = PACKAGE.parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.nanogpt_local import e2e_300m_muon_prefix_v001 as core  # noqa: E402


CONFIG = EXPERIMENTS / "configs/nanogpt300m_e2e_muon_prefix_v001.json"
PROTOCOL = EXPERIMENTS / "protocols/nanogpt300m_e2e_muon_prefix_v001.md"
REMOTE_ROOT = "/home/wang3587/e2e-b0-transfer/experiments/chtc/nanogpt300m_e2e_muon_prefix_v001"
OSDF_ROOT = "osdf:///chtc/staging/w/wang3587/nanogpt300m-e2e-6p5b-v001"
CONTAINER = (
    "docker.io/pytorch/pytorch:2.13.0-cuda12.6-cudnn9-runtime@sha256:"
    "6acf597eeb8e376a96580dde4952f37cc017fef732bb40bfc73f28f25e3f64b4"
)
DATA_FILES = {
    "train": "pretrain_train.bin",
    "training_tape": "training_tape.npy",
    "validation": "pretrain_validation.bin",
    "validation_probe": "validation_probe.npy",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, value: dict) -> None:
    if path.exists() or path.is_symlink():
        raise RuntimeError(f"refusing to overwrite {path}")
    path.write_text(json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _bundle(output: Path) -> Path:
    names = (
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
        "experiments/nanogpt_local/e2e_300m_muon_prefix_v001.py",
        "experiments/configs/nanogpt300m_e2e_6p5b_v001.json",
        "experiments/configs/nanogpt300m_e2e_muon_prefix_v001.json",
        "experiments/configs/openwebtext_cooldown_6p5b_expansion_source_plan_v001.json",
    )
    target = output / "code_bundle.tar"
    with tarfile.open(target, "w") as archive:
        for name in names:
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


def _cache(config: dict) -> dict:
    folder = ROOT / "data/openwebtext-cooldown-6p5b-v001"
    artifacts = {}
    for role, name in DATA_FILES.items():
        record = config["data_files"][role]
        path = folder / name
        if not path.is_file() or path.is_symlink() or path.stat().st_size != record["size_bytes"]:
            raise RuntimeError(f"full 6.5B {role} source is missing or changed")
        artifacts[role] = {
            "local_path": str(path.resolve()),
            "filename": name,
            "sha256": record["sha256"],
            "size_bytes": record["size_bytes"],
            "osdf_uri": f"{OSDF_ROOT}/{record['sha256']}/{name}",
            "upload_status": "not_uploaded",
        }
    return {
        "schema_version": "nanogpt300m_e2e_muon_shared_cache_v001",
        "campaign_id": config["campaign_id"],
        "status": "not_uploaded",
        "upload_each_sha256_once": True,
        "reusable_by_prefix_and_four_future_tails": True,
        "artifacts": artifacts,
    }


def _jdl(cache: dict, remote: str, run_id: str, duration: int,
         attempt_id: str = "a001") -> str:
    uris = [cache["artifacts"][role]["osdf_uri"] for role in DATA_FILES]
    inputs = [f"{remote}/code_bundle.tar", f"{remote}/prefix_config.json", *uris]
    return "\n".join((
        "# Launch-false one-H200 300M hybrid-Muon formal shared prefix.",
        "universe = container",
        f"container_image = docker://{CONTAINER}",
        "docker_override_entrypoint = true",
        f"initialdir = {remote}",
        f"executable = {remote}/run_prefix_once.sh",
        "transfer_executable = true",
        "getenv = false",
        "should_transfer_files = YES",
        "when_to_transfer_output = ON_EXIT",
        "transfer_input_files = " + ",".join(inputs),
        "transfer_output_files = prefix_output.tar",
        'transfer_output_remaps = "prefix_output.tar = '
        f'{OSDF_ROOT}/results/{attempt_id}/prefix_output.tar"',
        "requirements = (TARGET.HasDocker =?= true)",
        "request_gpus = 1",
        'require_gpus = (DeviceName == "NVIDIA H200")',
        "gpus_minimum_capability = 9.0",
        "gpus_minimum_memory = 140000MB",
        "gpus_minimum_runtime = 12.6",
        "request_cpus = 4",
        "request_memory = 32768MB",
        "request_disk = 32768MB",
        f"allowed_job_duration = {duration}",
        "+WantGPULab = true",
        '+GPUJobLength = "long"',
        f'+ExperimentRunId = "{run_id}"',
        f'+ExperimentAttemptId = "{run_id}-{attempt_id}"',
        "log = logs/$(Cluster)_$(Process).log",
        "output = logs/$(Cluster)_$(Process).out",
        "error = logs/$(Cluster)_$(Process).err",
        "queue 1",
        "",
    ))


def materialize(output: Path, attempt_id: str = "a001") -> dict:
    if not attempt_id.startswith("a") or not attempt_id[1:].isdigit():
        raise RuntimeError(f"invalid attempt id: {attempt_id}")
    if output.exists() or output.is_symlink():
        raise RuntimeError(f"refusing to overwrite formal prefix attempt: {output}")
    preflight = core.preflight(CONFIG, verify_data=True)
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    cache = _cache(config)
    output.mkdir(parents=True)
    config_path = output / "prefix_config.json"
    config_path.write_bytes(CONFIG.read_bytes())
    bundle = _bundle(output)
    wrapper = output / "run_prefix_once.sh"
    wrapper.write_bytes((PACKAGE / "run_prefix_once.sh").read_bytes())
    wrapper.chmod(0o755)
    preflight_path = output / "offline_preflight.json"
    _write_json(preflight_path, preflight)
    cache_path = output / "shared_cache_manifest.json"
    _write_json(cache_path, cache)
    remote = f"{REMOTE_ROOT}/{attempt_id}"
    jdl_path = output / "prefix.sub"
    jdl_path.write_text(
        _jdl(cache, remote, core.RUN_ID, config["hard_job_duration_seconds"], attempt_id),
        encoding="utf-8",
    )
    report = {
        "schema_version": "nanogpt300m_e2e_muon_prefix_materialization_v001",
        "run_id": core.RUN_ID,
        "attempt_id": attempt_id,
        "status": "offline_prepared_upload_and_cluster_policy_check_pending",
        "remote_stage": remote,
        "source_data_verified": preflight["data_gate"] == "passed_data_verified",
        "calibration_provenance_verified": True,
        "validation_grid_sha256": preflight["validation_grid_sha256"],
        "validation_samples": preflight["validation_samples"],
        "near_fork_validation_samples": preflight["near_fork_validation_samples"],
        "predicted_gpu_hours_before_transfer_and_checkpoint": preflight["estimated_gpu_hours_before_transfer_and_checkpoint"],
        "hard_job_duration_seconds": config["hard_job_duration_seconds"],
        "shared_cache_status": cache["status"],
        "full_6p5b_dataset_uploaded": False,
        "cluster_submission_authorized": False,
        "files": {
            path.name: {"sha256": _sha256(path), "size_bytes": path.stat().st_size}
            for path in (config_path, bundle, wrapper, preflight_path, cache_path, jdl_path)
        },
        "protocol": {"path": str(PROTOCOL), "sha256": _sha256(PROTOCOL)},
    }
    _write_json(output / "materialization.json", report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--attempt-id", default="a001")
    args = parser.parse_args()
    print(json.dumps(materialize(args.output_dir, args.attempt_id), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
