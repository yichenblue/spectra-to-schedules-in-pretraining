#!/usr/bin/env python3
"""Materialize the launch-false four-tail H200 package for v002."""

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
DEFAULT_TAIL_FREEZE = GENERATED / "tail-freeze-v001"
DEFAULT_DATA_FREEZE = GENERATED / "data-freeze-v001"
DEFAULT_PREFIX_COMPLETION = GENERATED / "prefix-launch-a001/completion_record.json"
DEFAULT_REMOTE_DIR = (
    "/home/wang3587/e2e-b0-transfer/experiments/chtc/"
    "nanogpt124m_e2e_sgd_rpath_factorization_2p5b_v002/tails-a001"
)
DEFAULT_REMOTE_PREFIX_CHECKPOINT = (
    "/home/wang3587/e2e-b0-transfer/experiments/chtc/"
    "nanogpt124m_e2e_sgd_rpath_factorization_2p5b_v002/"
    "prefix-a001/prefix_output/prefix_checkpoint.pt"
)
ATTEMPT_ID = "tails-a001"

QUEUE_ROWS = (
    (
        "811-fblr",
        "tail_811__fblr_config.json",
        "tail_output_811_fblr",
        "eight_one_one__fixed_batch_lr",
    ),
    (
        "811-flrbs",
        "tail_811__flrbs_config.json",
        "tail_output_811_flrbs",
        "eight_one_one__fixed_lr_batch",
    ),
    (
        "wsd-fblr",
        "tail_wsd__fblr_config.json",
        "tail_output_wsd_fblr",
        "wsd_exp_80_20__fixed_batch_lr",
    ),
    (
        "wsd-flrbs",
        "tail_wsd__flrbs_config.json",
        "tail_output_wsd_flrbs",
        "wsd_exp_80_20__fixed_lr_batch",
    ),
)


def _validate_inputs(
    tail_freeze: Path,
    data_freeze: Path,
    prefix_completion_path: Path,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, dict[str, Any]]]:
    manifest = packaging._load(tail_freeze / "tail_manifest.json")
    cache = packaging._load(data_freeze / "shared_cache_manifest.json")
    completion = packaging._load(prefix_completion_path)
    expected_configs = {row[1] for row in QUEUE_ROWS}
    if (
        manifest.get("schema_version")
        != "nanogpt124m_e2e_sgd_rpath_tail_materialization_v002"
        or manifest.get("campaign_id") != core.CAMPAIGN_ID
        or manifest.get("campaign_contract_sha256")
        != core.campaign_contract_sha256()
        or manifest.get("numerics_id") != core.NUMERICS_ID
        or manifest.get("launch_authorized") is not False
        or set(manifest.get("tail_configs", [])) != expected_configs
        or manifest.get("topology")
        != "four_independent_single_H200_jobs_no_DDP"
    ):
        raise packaging.MaterializationError("tail-freeze identity changed")
    checkpoint = manifest.get("prefix_checkpoint")
    completion_checkpoint = completion.get("artifacts", {}).get(
        core.PREFIX_CHECKPOINT_FILENAME
    )
    if (
        type(checkpoint) is not dict
        or type(completion_checkpoint) is not dict
        or completion.get("campaign_id") != core.CAMPAIGN_ID
        or completion.get("attempt_id") != "prefix-a001"
        or completion.get("status") != "passed"
        or completion.get("tail_materialization_ready") is not True
        or completion.get("offline_validation", {}).get("status") != "passed"
        or completion.get("offline_validation", {}).get("all_gates_passed")
        is not True
        or checkpoint.get("sha256") != completion_checkpoint.get("sha256")
        or checkpoint.get("size_bytes") != completion_checkpoint.get("size_bytes")
        or checkpoint.get("model_state_sha256")
        != completion.get("checkpoint_model_state_sha256")
    ):
        raise packaging.MaterializationError(
            "tail freeze is not bound to a passed v002 prefix"
        )
    artifacts = cache.get("artifacts")
    if (
        cache.get("campaign_id") != core.CAMPAIGN_ID
        or cache.get("status") != "reuse_existing_sha_addressed_v001_objects"
        or cache.get("planned_upload_rounds") != 0
        or type(artifacts) is not dict
    ):
        raise packaging.MaterializationError("shared-cache contract changed")

    configs: dict[str, dict[str, Any]] = {}
    budgets = manifest.get("tail_resource_budgets")
    if type(budgets) is not dict or set(budgets) != set(core.ARM_IDS):
        raise packaging.MaterializationError("tail budget inventory changed")
    for _, filename, _, arm_id in QUEUE_ROWS:
        config = packaging._load(tail_freeze / filename)
        core.validate_config(config)
        if (
            config.get("campaign_id") != core.CAMPAIGN_ID
            or config.get("phase") != "tail"
            or config.get("arm_id") != arm_id
            or config.get("launch_control")
            != {"cluster_submission_authorized": False}
            or config.get("prefix_checkpoint") != checkpoint
            or config.get("runtime", {}).get(
                "formal_wall_time_and_gpu_hour_budget"
            )
            != budgets[arm_id]
        ):
            raise packaging.MaterializationError(f"tail config changed: {filename}")
        if set(config.get("data", {})) != set(artifacts):
            raise packaging.MaterializationError("tail/cache inventory changed")
        for role, record in config["data"].items():
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
        configs[filename] = config
    return manifest, cache, completion, configs


def _jdl(
    remote: str,
    remote_prefix_checkpoint: str,
    cache: Mapping[str, Any],
    manifest: Mapping[str, Any],
) -> tuple[str, list[dict[str, Any]]]:
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
        f"{remote}/$(config)",
        remote_prefix_checkpoint,
        *(str(artifacts[role]["osdf_uri"]) for role in order),
    ]
    rows: list[dict[str, Any]] = []
    for arm_slug, config, output_dir, arm_id in QUEUE_ROWS:
        budget = manifest["tail_resource_budgets"][arm_id]
        rows.append(
            {
                "arm_slug": arm_slug,
                "config": config,
                "output_dir": output_dir,
                "arm_id": arm_id,
                "allowed_seconds": int(
                    math.ceil(float(budget["maximum_gpu_hours"]) * 3_600.0)
                ),
                "predicted_gpu_hours": float(budget["predicted_gpu_hours"]),
                "maximum_gpu_hours": float(budget["maximum_gpu_hours"]),
            }
        )
    lines = [
        "# Four independent v002 tails on four H200s; no DDP or data reupload.",
        "universe = container",
        f"container_image = docker://{packaging.CONTAINER_IMAGE}",
        "docker_override_entrypoint = true",
        f"initialdir = {remote}",
        f"executable = {remote}/run_tail_once.sh",
        "transfer_executable = true",
        "arguments = $(config) $(output_dir)",
        "getenv = false",
        "should_transfer_files = YES",
        "when_to_transfer_output = ON_EXIT",
        "preserve_relative_paths = false",
        "transfer_input_files = " + ",".join(inputs),
        "transfer_output_files = $(output_dir)",
        "requirements = (TARGET.HasDocker =?= true)",
        "request_gpus = 1",
        'require_gpus = (DeviceName == "NVIDIA H200")',
        "gpus_minimum_capability = 9.0",
        "gpus_minimum_memory = 140000MB",
        "gpus_minimum_runtime = 12.6",
        "request_cpus = 4",
        "request_memory = 16384MB",
        "request_disk = 20480MB",
        "allowed_job_duration = $(allowed_seconds)",
        "+WantGPULab = true",
        '+GPUJobLength = "medium"',
        f'+ExperimentRunId = "{core.CAMPAIGN_ID}-$(arm_slug)"',
        f'+ExperimentAttemptId = "{core.CAMPAIGN_ID}-$(arm_slug)-a001"',
        "log = logs/tails-a001_$(arm_slug)_$(Cluster)_$(Process).log",
        "output = logs/tails-a001_$(arm_slug)_$(Cluster)_$(Process).out",
        "error = logs/tails-a001_$(arm_slug)_$(Cluster)_$(Process).err",
        "",
        "queue arm_slug, config, output_dir, allowed_seconds from (",
        *(
            f"{row['arm_slug']} {row['config']} {row['output_dir']} "
            f"{row['allowed_seconds']}"
            for row in rows
        ),
        ")",
        "",
    ]
    return "\n".join(lines), rows


def materialize(
    tail_freeze_dir: Path,
    data_freeze_dir: Path,
    prefix_completion_path: Path,
    output_dir: Path,
    remote_dir: str,
    remote_prefix_checkpoint: str,
) -> dict[str, Any]:
    tail_freeze = tail_freeze_dir.expanduser().resolve()
    data_freeze = data_freeze_dir.expanduser().resolve()
    prefix_completion = prefix_completion_path.expanduser().resolve()
    output = output_dir.expanduser().resolve()
    remote = packaging._safe_remote(remote_dir)
    remote_checkpoint = packaging._safe_remote(remote_prefix_checkpoint)
    if output.exists() or output.is_symlink():
        raise packaging.MaterializationError(
            f"output directory already exists: {output}"
        )
    manifest, cache, completion, configs = _validate_inputs(
        tail_freeze, data_freeze, prefix_completion
    )
    output.mkdir(parents=True)
    for filename, config in configs.items():
        packaging._write(output / filename, packaging._canonical(config))
    bundle, bundle_members = packaging._bundle(output)
    wrapper = output / "run_tail_once.sh"
    shutil.copyfile(PACKAGE / "run_tail_once.sh", wrapper)
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
    jdl, rows = _jdl(remote, remote_checkpoint, cache, manifest)
    submit = output / "tails.sub"
    packaging._write(submit, jdl.encode("utf-8"))
    tracked = tuple(output / row[1] for row in QUEUE_ROWS) + (
        bundle,
        wrapper,
        submit,
    )
    predicted = sum(row["predicted_gpu_hours"] for row in rows)
    maximum = sum(row["maximum_gpu_hours"] for row in rows)
    record = {
        "schema_version": (
            "nanogpt124m_e2e_sgd_rpath_factorization_2p5b_tails_materialization_v002"
        ),
        "campaign_id": core.CAMPAIGN_ID,
        "attempt_id": ATTEMPT_ID,
        "status": "prepared_launch_false",
        "launch_authorized": False,
        "remote_directory": remote,
        "remote_prefix_checkpoint": remote_checkpoint,
        "source_prefix": {
            "attempt_id": completion["attempt_id"],
            "job_id": completion["job_id"],
            "checkpoint_sha256": manifest["prefix_checkpoint"]["sha256"],
            "checkpoint_model_state_sha256": manifest["prefix_checkpoint"][
                "model_state_sha256"
            ],
            "offline_validation_passed": True,
        },
        "topology": "four_independent_single_H200_jobs_no_DDP",
        "arms": rows,
        "aggregate_budget": {
            "predicted_gpu_hours": predicted,
            "maximum_gpu_hours": maximum,
            "predicted_parallel_wall_hours": max(
                row["predicted_gpu_hours"] for row in rows
            ),
            "maximum_parallel_wall_hours": max(
                row["maximum_gpu_hours"] for row in rows
            ),
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
    parser.add_argument("--tail-freeze-dir", type=Path, default=DEFAULT_TAIL_FREEZE)
    parser.add_argument("--data-freeze-dir", type=Path, default=DEFAULT_DATA_FREEZE)
    parser.add_argument(
        "--prefix-completion", type=Path, default=DEFAULT_PREFIX_COMPLETION
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--remote-dir", default=DEFAULT_REMOTE_DIR)
    parser.add_argument(
        "--remote-prefix-checkpoint", default=DEFAULT_REMOTE_PREFIX_CHECKPOINT
    )
    args = parser.parse_args(argv)
    print(
        json.dumps(
            materialize(
                args.tail_freeze_dir,
                args.data_freeze_dir,
                args.prefix_completion,
                args.output_dir,
                args.remote_dir,
                args.remote_prefix_checkpoint,
            ),
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
