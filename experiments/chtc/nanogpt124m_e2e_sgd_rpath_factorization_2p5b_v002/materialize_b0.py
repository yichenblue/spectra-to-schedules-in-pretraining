#!/usr/bin/env python3
"""Materialize one launch-false H200 B0 package for the v002 campaign."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import shutil
import subprocess
import sys
import tarfile
from typing import Any, Mapping, Sequence


PACKAGE = Path(__file__).resolve().parent
REPOSITORY = Path(__file__).resolve().parents[3]
if str(REPOSITORY) not in sys.path:
    sys.path.insert(0, str(REPOSITORY))

from experiments.nanogpt_local import (  # noqa: E402
    e2e_sgd_rpath_factorization_124m_2p5b_v002 as core,
)


EXPERIMENTS = REPOSITORY / "experiments"
DEFAULT_DATA_FREEZE = (
    EXPERIMENTS
    / "generated/nanogpt124m-e2e-sgd-rpath-factorization-2p5b-v002/"
    "data-freeze-v001"
)
DEFAULT_REMOTE_DIR = (
    "/home/wang3587/e2e-b0-transfer/experiments/chtc/"
    "nanogpt124m_e2e_sgd_rpath_factorization_2p5b_v002/b0-a001"
)
CONTAINER_IMAGE = (
    "docker.io/pytorch/pytorch:2.13.0-cuda12.6-cudnn9-runtime@sha256:"
    "6acf597eeb8e376a96580dde4952f37cc017fef732bb40bfc73f28f25e3f64b4"
)
ATTEMPT_ID = "b0-a001"


class MaterializationError(RuntimeError):
    """A frozen input or generated launch artifact is inconsistent."""


def _canonical(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _load(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise MaterializationError(f"cannot read JSON {path}: {error}") from error
    if type(value) is not dict:
        raise MaterializationError(f"JSON root is not an object: {path}")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write(path: Path, payload: bytes) -> None:
    if path.exists() or path.is_symlink():
        raise MaterializationError(f"refusing to overwrite {path}")
    path.write_bytes(payload)


def _safe_remote(path: str) -> str:
    value = PurePosixPath(path)
    if not value.is_absolute() or ".." in value.parts or not path.startswith("/home/"):
        raise MaterializationError("remote directory must be a safe absolute /home path")
    return value.as_posix()


def _bundle(output: Path) -> tuple[Path, list[str]]:
    members = (
        EXPERIMENTS / "__init__.py",
        EXPERIMENTS / "nanogpt_local/__init__.py",
        EXPERIMENTS / "nanogpt_local/model.py",
        EXPERIMENTS
        / "nanogpt_local/e2e_sgd_rpath_factorization_124m_2p5b_v002.py",
        EXPERIMENTS
        / "configs/nanogpt124m_e2e_sgd_rpath_factorization_2p5b_v002.json",
    )
    target = output / "code_bundle.tar"
    names: list[str] = []
    with tarfile.open(target, "w") as archive:
        for source in members:
            if not source.is_file() or source.is_symlink():
                raise MaterializationError(f"bundle source is absent or linked: {source}")
            relative = source.relative_to(REPOSITORY).as_posix()
            payload = source.read_bytes()
            info = tarfile.TarInfo(relative)
            info.size = len(payload)
            info.mode = 0o644
            info.mtime = 0
            info.uid = info.gid = 0
            info.uname = info.gname = ""
            archive.addfile(info, io.BytesIO(payload))
            names.append(relative)
    return target, names


def _validate_data_freeze(
    data_freeze: Path,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    manifest = _load(data_freeze / "data_manifest.json")
    cache = _load(data_freeze / "shared_cache_manifest.json")
    config = _load(data_freeze / "b0_config.json")
    if (
        manifest.get("campaign_id") != core.CAMPAIGN_ID
        or cache.get("campaign_id") != core.CAMPAIGN_ID
        or cache.get("status") != "reuse_existing_sha_addressed_v001_objects"
        or cache.get("planned_upload_rounds") != 0
        or config.get("campaign_id") != core.CAMPAIGN_ID
        or config.get("phase") != "b0"
        or config.get("launch_control")
        != {"cluster_submission_authorized": False}
    ):
        raise MaterializationError("v002 data-freeze identity or launch gate changed")
    core.validate_config(config)
    data = manifest.get("data")
    artifacts = cache.get("artifacts")
    if type(data) is not dict or type(artifacts) is not dict or config.get("data") != data:
        raise MaterializationError("data manifest, cache, and B0 config disagree")
    expected_roles = {
        "metadata",
        "source_manifest",
        "train",
        "training_tape",
        "validation",
        "validation_probe",
    }
    if set(data) != expected_roles or set(artifacts) != expected_roles:
        raise MaterializationError("shared-data inventory changed")
    for role in sorted(expected_roles):
        record = data[role]
        cached = artifacts[role]
        if (
            cached.get("sha256") != record.get("sha256")
            or cached.get("size_bytes") != record.get("size_bytes")
            or cached.get("upload_status") != "reuse_existing"
            or not str(cached.get("osdf_uri", "")).startswith("osdf:///")
        ):
            raise MaterializationError(f"shared-cache record changed for {role}")
    return manifest, cache, config


def _jdl(remote: str, cache: Mapping[str, Any]) -> str:
    artifacts = cache["artifacts"]
    order = (
        "metadata",
        "source_manifest",
        "train",
        "training_tape",
        "validation",
        "validation_probe",
    )
    inputs = [
        f"{remote}/code_bundle.tar",
        f"{remote}/b0_config.json",
        *(str(artifacts[role]["osdf_uri"]) for role in order),
    ]
    return "\n".join(
        [
            "# One response-free H200 BF16/TF32/Flash calibration; no data reupload.",
            "universe = container",
            f"container_image = docker://{CONTAINER_IMAGE}",
            "docker_override_entrypoint = true",
            f"initialdir = {remote}",
            f"executable = {remote}/run_b0_once.sh",
            "transfer_executable = true",
            "getenv = false",
            "should_transfer_files = YES",
            "when_to_transfer_output = ON_EXIT",
            "preserve_relative_paths = false",
            "transfer_input_files = " + ",".join(inputs),
            "transfer_output_files = b0_output",
            "requirements = (TARGET.HasDocker =?= true)",
            "request_gpus = 1",
            'require_gpus = (DeviceName == "NVIDIA H200")',
            "gpus_minimum_capability = 9.0",
            "gpus_minimum_memory = 140000MB",
            "gpus_minimum_runtime = 12.6",
            "request_cpus = 4",
            "request_memory = 16384MB",
            "request_disk = 20480MB",
            "allowed_job_duration = 900",
            "+WantGPULab = true",
            '+GPUJobLength = "short"',
            f'+ExperimentRunId = "{core.CAMPAIGN_ID}-b0"',
            f'+ExperimentAttemptId = "{core.CAMPAIGN_ID}-b0-a001"',
            "log = logs/b0-a001_$(Cluster)_$(Process).log",
            "output = logs/b0-a001_$(Cluster)_$(Process).out",
            "error = logs/b0-a001_$(Cluster)_$(Process).err",
            "queue 1",
            "",
        ]
    )


def materialize(data_freeze_dir: Path, output_dir: Path, remote_dir: str) -> dict[str, Any]:
    data_freeze = data_freeze_dir.expanduser().resolve()
    output = output_dir.expanduser().resolve()
    remote = _safe_remote(remote_dir)
    if output.exists() or output.is_symlink():
        raise MaterializationError(f"output directory already exists: {output}")
    _, cache, config = _validate_data_freeze(data_freeze)
    output.mkdir(parents=True)
    _write(output / "b0_config.json", _canonical(config))
    bundle, bundle_members = _bundle(output)
    wrapper = output / "run_b0_once.sh"
    shutil.copyfile(PACKAGE / "run_b0_once.sh", wrapper)
    wrapper.chmod(0o755)
    syntax = subprocess.run(
        ["/bin/bash", "-n", str(wrapper)],
        check=False,
        capture_output=True,
        text=True,
    )
    if syntax.returncode != 0:
        raise MaterializationError(f"worker shell syntax failed: {syntax.stderr}")
    submit = output / "b0.sub"
    _write(submit, _jdl(remote, cache).encode("utf-8"))
    tracked = (output / "b0_config.json", bundle, wrapper, submit)
    record = {
        "schema_version": (
            "nanogpt124m_e2e_sgd_rpath_factorization_2p5b_b0_materialization_v002"
        ),
        "campaign_id": core.CAMPAIGN_ID,
        "attempt_id": ATTEMPT_ID,
        "status": "prepared_launch_false",
        "launch_authorized": False,
        "remote_directory": remote,
        "target": {
            "gpu_count": 1,
            "device_name": "NVIDIA H200",
            "maximum_gpu_hours": core.B0_MAXIMUM_GPU_HOURS,
        },
        "shared_dataset_cache": {
            "policy": "reuse_existing_SHA_addressed_v001_objects",
            "artifact_count": len(cache["artifacts"]),
            "reupload_required": False,
        },
        "base_git_revision": subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=REPOSITORY,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip(),
        "campaign_contract_sha256": core.campaign_contract_sha256(),
        "schedule_manifest_sha256": core.schedule_manifest()["sha256"],
        "bundle_members": bundle_members,
        "files": {
            path.name: {"sha256": _sha256(path), "size_bytes": path.stat().st_size}
            for path in tracked
        },
    }
    _write(output / "materialization.json", _canonical(record))
    return record


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-freeze-dir", type=Path, default=DEFAULT_DATA_FREEZE)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--remote-dir", default=DEFAULT_REMOTE_DIR)
    args = parser.parse_args(argv)
    print(
        json.dumps(
            materialize(args.data_freeze_dir, args.output_dir, args.remote_dir),
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
