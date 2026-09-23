#!/usr/bin/env python3
"""Materialize one launch-false shared-prefix H200 package for v002."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Any, Mapping, Sequence


PACKAGE = Path(__file__).resolve().parent
REPOSITORY = Path(__file__).resolve().parents[3]
if str(REPOSITORY) not in sys.path:
    sys.path.insert(0, str(REPOSITORY))

from experiments.chtc.nanogpt124m_e2e_sgd_rpath_factorization_2p5b_v002 import (  # noqa: E402
    materialize_b0 as packaging,
)
from experiments.nanogpt_local import (  # noqa: E402
    e2e_sgd_rpath_factorization_124m_2p5b_v002 as core,
)


EXPERIMENTS = REPOSITORY / "experiments"
GENERATED = (
    EXPERIMENTS
    / "generated/nanogpt124m-e2e-sgd-rpath-factorization-2p5b-v002"
)
DEFAULT_PREFIX_FREEZE = GENERATED / "prefix-freeze-v001"
DEFAULT_DATA_FREEZE = GENERATED / "data-freeze-v001"
DEFAULT_B0_COMPLETION = GENERATED / "b0-launch-a001/completion_record.json"
DEFAULT_REMOTE_DIR = (
    "/home/wang3587/e2e-b0-transfer/experiments/chtc/"
    "nanogpt124m_e2e_sgd_rpath_factorization_2p5b_v002/prefix-a001"
)
ATTEMPT_ID = "prefix-a001"


def _validate_inputs(
    prefix_freeze: Path,
    data_freeze: Path,
    b0_completion_path: Path,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    prefix = packaging._load(prefix_freeze / "prefix_config.json")
    calibration = packaging._load(
        prefix_freeze / "H200_BF16_resource_calibration.json"
    )
    cache = packaging._load(data_freeze / "shared_cache_manifest.json")
    completion = packaging._load(b0_completion_path)
    if (
        prefix.get("campaign_id") != core.CAMPAIGN_ID
        or prefix.get("phase") != "prefix"
        or prefix.get("arm_id") is not None
        or prefix.get("launch_control")
        != {"cluster_submission_authorized": False}
        or prefix.get("resource_calibration") != calibration
        or completion.get("campaign_id") != core.CAMPAIGN_ID
        or completion.get("attempt_id") != "b0-a001"
        or completion.get("status") != "passed"
        or completion.get("offline_replay_validation_passed") is not True
        or completion.get("all_b0_gates_passed") is not True
        or completion.get("artifacts", {}).get("result_sha256")
        != calibration.get("b0_result_sha256")
        or completion.get("artifacts", {}).get("training_trace_sha256")
        != calibration.get("b0_trace_sha256")
    ):
        raise packaging.MaterializationError(
            "prefix freeze is not bound to the passed v002 B0"
        )
    core.validate_config(prefix)
    budget = prefix.get("runtime", {}).get(
        "formal_wall_time_and_gpu_hour_budget"
    )
    if (
        type(budget) is not dict
        or budget.get("phase") != "prefix"
        or budget.get("arm_id") is not None
        or budget.get("source_b0_result_sha256")
        != calibration.get("b0_result_sha256")
        or type(budget.get("predicted_gpu_hours")) not in (int, float)
        or type(budget.get("maximum_gpu_hours")) not in (int, float)
        or not 0 < float(budget["predicted_gpu_hours"]) <= float(
            budget["maximum_gpu_hours"]
        )
        or budget.get("training_updates") != core.CONSTANT_SOURCE_UPDATES
        or budget.get("validation_evaluations")
        != core.REFERENCE_300M_PREFIX_EVALUATIONS
    ):
        raise packaging.MaterializationError("prefix resource budget changed")
    artifacts = cache.get("artifacts")
    if (
        cache.get("campaign_id") != core.CAMPAIGN_ID
        or cache.get("status") != "reuse_existing_sha_addressed_v001_objects"
        or cache.get("planned_upload_rounds") != 0
        or type(artifacts) is not dict
        or set(artifacts) != set(prefix.get("data", {}))
    ):
        raise packaging.MaterializationError("shared-cache contract changed")
    for role, record in prefix["data"].items():
        cached = artifacts[role]
        if (
            cached.get("sha256") != record.get("sha256")
            or cached.get("size_bytes") != record.get("size_bytes")
            or cached.get("upload_status") != "reuse_existing"
            or not str(cached.get("osdf_uri", "")).startswith("osdf:///")
        ):
            raise packaging.MaterializationError(
                f"shared-cache record changed for {role}"
            )
    return prefix, cache, completion


def _jdl(
    remote: str,
    cache: Mapping[str, Any],
    maximum_gpu_hours: float,
) -> tuple[str, int]:
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
        f"{remote}/prefix_config.json",
        *(str(artifacts[role]["osdf_uri"]) for role in order),
    ]
    allowed_seconds = int(math.ceil(maximum_gpu_hours * 3_600.0))
    lines = [
        "# One shared constant-r v002 prefix on one H200; no data reupload.",
        "universe = container",
        f"container_image = docker://{packaging.CONTAINER_IMAGE}",
        "docker_override_entrypoint = true",
        f"initialdir = {remote}",
        f"executable = {remote}/run_prefix_once.sh",
        "transfer_executable = true",
        "getenv = false",
        "should_transfer_files = YES",
        "when_to_transfer_output = ON_EXIT",
        "preserve_relative_paths = false",
        "transfer_input_files = " + ",".join(inputs),
        "transfer_output_files = prefix_output",
        "requirements = (TARGET.HasDocker =?= true)",
        "request_gpus = 1",
        'require_gpus = (DeviceName == "NVIDIA H200")',
        "gpus_minimum_capability = 9.0",
        "gpus_minimum_memory = 140000MB",
        "gpus_minimum_runtime = 12.6",
        "request_cpus = 4",
        "request_memory = 16384MB",
        "request_disk = 20480MB",
        f"allowed_job_duration = {allowed_seconds}",
        "+WantGPULab = true",
        '+GPUJobLength = "medium"',
        f'+ExperimentRunId = "{core.CAMPAIGN_ID}-prefix"',
        f'+ExperimentAttemptId = "{core.CAMPAIGN_ID}-prefix-a001"',
        "log = logs/prefix-a001_$(Cluster)_$(Process).log",
        "output = logs/prefix-a001_$(Cluster)_$(Process).out",
        "error = logs/prefix-a001_$(Cluster)_$(Process).err",
        "queue 1",
        "",
    ]
    return "\n".join(lines), allowed_seconds


def materialize(
    prefix_freeze_dir: Path,
    data_freeze_dir: Path,
    b0_completion_path: Path,
    output_dir: Path,
    remote_dir: str,
) -> dict[str, Any]:
    prefix_freeze = prefix_freeze_dir.expanduser().resolve()
    data_freeze = data_freeze_dir.expanduser().resolve()
    b0_completion = b0_completion_path.expanduser().resolve()
    output = output_dir.expanduser().resolve()
    remote = packaging._safe_remote(remote_dir)
    if output.exists() or output.is_symlink():
        raise packaging.MaterializationError(
            f"output directory already exists: {output}"
        )
    prefix, cache, completion = _validate_inputs(
        prefix_freeze, data_freeze, b0_completion
    )
    budget = prefix["runtime"]["formal_wall_time_and_gpu_hour_budget"]
    output.mkdir(parents=True)
    packaging._write(output / "prefix_config.json", packaging._canonical(prefix))
    bundle, bundle_members = packaging._bundle(output)
    wrapper = output / "run_prefix_once.sh"
    shutil.copyfile(PACKAGE / "run_prefix_once.sh", wrapper)
    wrapper.chmod(0o755)
    syntax = subprocess.run(
        ["/bin/bash", "-n", str(wrapper)],
        check=False,
        capture_output=True,
        text=True,
    )
    if syntax.returncode != 0:
        raise packaging.MaterializationError(
            f"worker shell syntax failed: {syntax.stderr}"
        )
    jdl, allowed_seconds = _jdl(
        remote, cache, float(budget["maximum_gpu_hours"])
    )
    submit = output / "prefix.sub"
    packaging._write(submit, jdl.encode("utf-8"))
    tracked = (output / "prefix_config.json", bundle, wrapper, submit)
    record = {
        "schema_version": (
            "nanogpt124m_e2e_sgd_rpath_factorization_2p5b_prefix_materialization_v002"
        ),
        "campaign_id": core.CAMPAIGN_ID,
        "attempt_id": ATTEMPT_ID,
        "status": "prepared_launch_false",
        "launch_authorized": False,
        "remote_directory": remote,
        "source_b0": {
            "attempt_id": completion["attempt_id"],
            "job_id": completion["job_id"],
            "result_sha256": completion["artifacts"]["result_sha256"],
            "trace_sha256": completion["artifacts"]["training_trace_sha256"],
            "offline_replay_validation_passed": True,
        },
        "training": {
            "updates": core.CONSTANT_SOURCE_UPDATES,
            "tokens": core.PREFIX_TOKENS,
            "batch_size": core.ATOMIC_BATCH_SIZE,
            "learning_rate": core.CANONICAL_BASE_LEARNING_RATE,
            "ratio_B_over_eta": core.TARGET_INITIAL_RATIO,
            "validation_evaluations": core.REFERENCE_300M_PREFIX_EVALUATIONS,
        },
        "target": {
            "gpu_count": 1,
            "device_name": "NVIDIA H200",
            "predicted_gpu_hours": budget["predicted_gpu_hours"],
            "maximum_gpu_hours": budget["maximum_gpu_hours"],
            "allowed_job_duration_seconds": allowed_seconds,
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
            path.name: {
                "sha256": packaging._sha256(path),
                "size_bytes": path.stat().st_size,
            }
            for path in tracked
        },
    }
    packaging._write(output / "materialization.json", packaging._canonical(record))
    return record


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prefix-freeze-dir", type=Path, default=DEFAULT_PREFIX_FREEZE)
    parser.add_argument("--data-freeze-dir", type=Path, default=DEFAULT_DATA_FREEZE)
    parser.add_argument("--b0-completion", type=Path, default=DEFAULT_B0_COMPLETION)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--remote-dir", default=DEFAULT_REMOTE_DIR)
    args = parser.parse_args(argv)
    print(
        json.dumps(
            materialize(
                args.prefix_freeze_dir,
                args.data_freeze_dir,
                args.b0_completion,
                args.output_dir,
                args.remote_dir,
            ),
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
