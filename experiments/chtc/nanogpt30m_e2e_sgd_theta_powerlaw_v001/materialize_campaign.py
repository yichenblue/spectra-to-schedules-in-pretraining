#!/usr/bin/env python3
"""Create launch-false CHTC packages for the 30M theta campaign.

This utility performs local validation and packaging only.  It has no SSH,
upload, credential, Condor-submit, retry, or resume capability.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import tarfile
from typing import Any, Mapping, Sequence

import torch


PACKAGE = Path(__file__).resolve().parent
REPOSITORY = Path(__file__).resolve().parents[3]
if str(REPOSITORY) not in sys.path:
    sys.path.insert(0, str(REPOSITORY))
CONFIG = REPOSITORY / "experiments/configs/nanogpt30m_e2e_sgd_theta_powerlaw_v001.json"
REMOTE_ROOT = Path("/home/wang3587/e2e-b0-transfer/experiments/chtc")
REMOTE_CAMPAIGN = REMOTE_ROOT / "nanogpt30m_e2e_sgd_theta_powerlaw_v001"
OSDF_ROOT = (
    "osdf:///chtc/staging/w/wang3587/"
    "nanogpt124m-e2e-sgd-rpath-factorization-2p5b-v001"
)
CONTAINER_IMAGE = (
    "docker.io/pytorch/pytorch:2.13.0-cuda12.6-cudnn9-runtime@sha256:"
    "6acf597eeb8e376a96580dde4952f37cc017fef732bb40bfc73f28f25e3f64b4"
)
ALLOWED_JOB_DURATION = 15_300
EXPECTED_PREFIX_STATE_SHA256 = (
    "53ba7f6928e2dcb5690b2b2e11208ffb86f44fd49202118c0c21da77811a305d"
)
DATA = {
    "train": (
        "d718aa14fa180720cf27afe9457b9c2c0b9cc4bed3fbe4839de98b167dc61668",
        5_000_000_512,
        "pretrain_train.bin",
    ),
    "validation": (
        "52085210f2da47797efd93401d8b890c6efdb41055c266d499649e1d82ee20be",
        33_554_432,
        "pretrain_validation.bin",
    ),
    "validation_probe": (
        "0295b25ce763f30570d4a268491d5a829e244ade33c602c32423e66aea278155",
        8_320,
        "validation_probe.npy",
    ),
}
THETAS = (
    ("theta0", "0"),
    ("theta0p5", "0.5"),
    ("theta0p75", "0.75"),
    ("theta1", "1"),
    ("theta1p5", "1.5"),
)
CODE_FILES = (
    "experiments/__init__.py",
    "experiments/nanogpt_local/__init__.py",
    "experiments/nanogpt_local/common.py",
    "experiments/nanogpt_local/model.py",
    "experiments/nanogpt_local/pretrain.py",
    "experiments/nanogpt_local/e2e_sgd_b0_v001.py",
    "experiments/nanogpt_local/e2e_sgd_cooldown_calibration_v001.py",
    "experiments/nanogpt_local/e2e_sgd_theta_powerlaw_30m_v001.py",
)


class MaterializationError(RuntimeError):
    """A local frozen-input or package invariant failed."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        value = json.load(stream)
    if type(value) is not dict:
        raise MaterializationError(f"JSON root is not an object: {path}")
    return value


def _write(path: Path, value: Mapping[str, Any]) -> None:
    raw = (
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False)
        + "\n"
    ).encode("utf-8")
    with path.open("xb") as stream:
        stream.write(raw)


def _new_output(path: Path) -> Path:
    result = path.expanduser().resolve()
    if result.exists() or result.is_symlink():
        raise MaterializationError(f"refusing to replace output: {result}")
    result.mkdir(parents=True)
    return result


def _verify_local_inputs() -> dict[str, Any]:
    config = _json(CONFIG)
    for role, (digest, size, _) in DATA.items():
        path = REPOSITORY / config["data"][role]["path"]
        if (
            not path.is_file()
            or path.is_symlink()
            or path.stat().st_size != size
            or _sha256(path) != digest
        ):
            raise MaterializationError(f"local {role} input is absent or changed")
    for relative in CODE_FILES:
        path = REPOSITORY / relative
        if not path.is_file() or path.is_symlink():
            raise MaterializationError(f"code input is absent or linked: {relative}")
    return config


def _add_to_tar(archive: tarfile.TarFile, source: Path, relative: str) -> None:
    info = archive.gettarinfo(str(source), arcname=relative)
    info.uid = info.gid = 0
    info.uname = info.gname = ""
    info.mtime = 0
    info.mode = 0o644
    with source.open("rb") as stream:
        archive.addfile(info, stream)


def _bundle(path: Path) -> dict[str, Any]:
    with tarfile.open(path, "x") as archive:
        for relative in CODE_FILES:
            _add_to_tar(archive, REPOSITORY / relative, relative)
    return {"path": path.name, "size_bytes": path.stat().st_size, "sha256": _sha256(path)}


def _common_submit(
    initial: Path,
    executable: Path,
    output_name: str,
    extra_transfers: Sequence[Path] = (),
) -> list[str]:
    cache_files = [
        f"{OSDF_ROOT}/{digest}/{filename}"
        for digest, _, filename in DATA.values()
    ]
    transfers = [
        initial / "code_bundle.tar",
        initial / "theta_config.json",
        *cache_files,
        *extra_transfers,
    ]
    return [
        "universe = container",
        f"container_image = docker://{CONTAINER_IMAGE}",
        "docker_override_entrypoint = true",
        f"initialdir = {initial}",
        f"executable = {executable}",
        "transfer_executable = true",
        "getenv = false",
        "should_transfer_files = YES",
        "when_to_transfer_output = ON_EXIT",
        "preserve_relative_paths = false",
        "transfer_input_files = " + ",".join(str(path) for path in transfers),
        f"transfer_output_files = {output_name}",
        "requirements = (TARGET.HasDocker =?= true)",
        "request_gpus = 1",
        'require_gpus = (DeviceName == "NVIDIA A100-SXM4-80GB")',
        "gpus_minimum_capability = 8.0",
        "gpus_minimum_memory = 40000MB",
        "gpus_minimum_runtime = 12.6",
        "request_cpus = 4",
        "request_memory = 8192MB",
        "request_disk = 8192MB",
        # Leave fifteen minutes for wrapper validation and output transfer after
        # the core's frozen 14,400-second execution deadline.
        f"allowed_job_duration = {ALLOWED_JOB_DURATION}",
        "+WantGPULab = true",
        '+GPUJobLength = "medium"',
    ]


def prepare_prefix(output_dir: Path, attempt: str) -> dict[str, Any]:
    _verify_local_inputs()
    output = _new_output(output_dir)
    (output / "logs").mkdir()
    shutil.copyfile(CONFIG, output / "theta_config.json")
    shutil.copyfile(PACKAGE / "run_prefix_once.sh", output / "run_prefix_once.sh")
    os.chmod(output / "run_prefix_once.sh", 0o755)
    bundle = _bundle(output / "code_bundle.tar")
    remote = REMOTE_CAMPAIGN / f"prefix-{attempt}"
    lines = _common_submit(remote, remote / "run_prefix_once.sh", "prefix_output")
    lines.extend(
        [
            f'+ExperimentRunId = "nanogpt30m-e2e-sgd-theta-powerlaw-v001-prefix"',
            f'+ExperimentAttemptId = "nanogpt30m-e2e-sgd-theta-powerlaw-v001-prefix-{attempt}"',
            f"log = logs/prefix-{attempt}_$(Cluster)_$(Process).log",
            f"output = logs/prefix-{attempt}_$(Cluster)_$(Process).out",
            f"error = logs/prefix-{attempt}_$(Cluster)_$(Process).err",
            "queue 1",
        ]
    )
    (output / "prefix.sub").write_text("\n".join(lines) + "\n", encoding="utf-8")
    manifest = {
        "classification": "PREFIX_PACKAGE_PREPARED_LAUNCH_FALSE",
        "campaign_id": "nanogpt30m-e2e-sgd-theta-powerlaw-v001",
        "attempt": attempt,
        "launch_authorized": False,
        "expected_prefix_state_sha256": EXPECTED_PREFIX_STATE_SHA256,
        "data_remote_cache": {
            role: {
                "sha256": digest,
                "size_bytes": size,
                "path": f"{OSDF_ROOT}/{digest}/{filename}",
            }
            for role, (digest, size, filename) in DATA.items()
        },
        "code_bundle": bundle,
        "config_sha256": _sha256(output / "theta_config.json"),
    }
    _write(output / "prepared_manifest.json", manifest)
    return manifest


def _validate_checkpoint(path: Path) -> dict[str, Any]:
    checkpoint = path.expanduser().resolve()
    if not checkpoint.is_file() or checkpoint.is_symlink():
        raise MaterializationError("prefix checkpoint is absent or linked")
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    from experiments.nanogpt_local import e2e_sgd_theta_powerlaw_30m_v001 as core

    data_identity = core._canonical_sha256(
        {
            "train": {
                "sha256": DATA["train"][0],
                "token_count": 2_500_000_256,
                "bytes": DATA["train"][1],
            },
            "validation": {
                "sha256": DATA["validation"][0],
                "token_count": 16_777_216,
                "bytes": DATA["validation"][1],
            },
            "validation_probe": {"sha256": DATA["validation_probe"][0]},
        }
    )
    try:
        core._validate_checkpoint_payload(
            payload,
            data_identity_sha256=data_identity,
            validation_probe_sha256=DATA["validation_probe"][0],
        )
    except (RuntimeError, TypeError, ValueError) as error:
        raise MaterializationError(
            f"prefix checkpoint failed the full frozen contract: {error}"
        ) from error
    return {
        "path": str(checkpoint),
        "size_bytes": checkpoint.stat().st_size,
        "sha256": _sha256(checkpoint),
        "model_state_sha256": EXPECTED_PREFIX_STATE_SHA256,
    }


def prepare_tails(output_dir: Path, checkpoint: Path, attempt: str) -> dict[str, Any]:
    _verify_local_inputs()
    checkpoint_record = _validate_checkpoint(checkpoint)
    output = _new_output(output_dir)
    (output / "tail_outputs").mkdir()
    (output / "logs").mkdir()
    shutil.copyfile(CONFIG, output / "theta_config.json")
    shutil.copyfile(PACKAGE / "run_tail_once.sh", output / "run_tail_once.sh")
    os.chmod(output / "run_tail_once.sh", 0o755)
    shutil.copyfile(checkpoint.expanduser().resolve(), output / "prefix_checkpoint.pt")
    bundle = _bundle(output / "code_bundle.tar")
    remote = REMOTE_CAMPAIGN / f"tails-{attempt}"
    lines = _common_submit(
        remote,
        remote / "run_tail_once.sh",
        "tail_output",
        (remote / "prefix_checkpoint.pt",),
    )
    lines.extend(
        [
            "arguments = $(theta)",
            'transfer_output_remaps = "tail_output=tail_outputs/$(tag)"',
            '+ExperimentRunId = "nanogpt30m-e2e-sgd-theta-powerlaw-v001-$(tag)"',
            f'+ExperimentAttemptId = "nanogpt30m-e2e-sgd-theta-powerlaw-v001-$(tag)-{attempt}"',
            f"log = logs/$(tag)-{attempt}_$(Cluster)_$(Process).log",
            f"output = logs/$(tag)-{attempt}_$(Cluster)_$(Process).out",
            f"error = logs/$(tag)-{attempt}_$(Cluster)_$(Process).err",
            "queue tag, theta from jobs.tsv",
        ]
    )
    (output / "tails.sub").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (output / "jobs.tsv").write_text(
        "\n".join(f"{tag}\t{theta}" for tag, theta in THETAS) + "\n",
        encoding="utf-8",
    )
    manifest = {
        "classification": "TAIL_PACKAGE_PREPARED_LAUNCH_FALSE",
        "campaign_id": "nanogpt30m-e2e-sgd-theta-powerlaw-v001",
        "attempt": attempt,
        "launch_authorized": False,
        "execution_plan": {
            "tail_jobs": len(THETAS),
            "maximum_concurrent_tail_jobs": len(THETAS),
            "gpus_per_tail_job": 1,
            "distributed_training": False,
        },
        "theta_jobs": [{"tag": tag, "theta": float(theta)} for tag, theta in THETAS],
        "checkpoint": checkpoint_record,
        "code_bundle": bundle,
        "config_sha256": _sha256(output / "theta_config.json"),
    }
    _write(output / "prepared_manifest.json", manifest)
    return manifest


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    prefix = subparsers.add_parser("prepare-prefix")
    prefix.add_argument("--output-dir", type=Path, required=True)
    prefix.add_argument("--attempt", default="a001")
    tails = subparsers.add_parser("prepare-tails")
    tails.add_argument("--output-dir", type=Path, required=True)
    tails.add_argument("--checkpoint", type=Path, required=True)
    tails.add_argument("--attempt", default="a001")
    args = parser.parse_args(argv)
    result = (
        prepare_prefix(args.output_dir, args.attempt)
        if args.command == "prepare-prefix"
        else prepare_tails(args.output_dir, args.checkpoint, args.attempt)
    )
    print(json.dumps(result, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
