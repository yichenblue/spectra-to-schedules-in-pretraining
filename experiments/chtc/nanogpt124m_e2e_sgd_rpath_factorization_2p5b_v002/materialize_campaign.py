#!/usr/bin/env python3
"""Materialize launch-false records for the accelerated 124M 2.5B campaign.

This utility only reads local artifacts and writes a new local output
directory.  It cannot upload, SSH, create a Condor run specification, or
submit work.  Tail configs are intentionally impossible to materialize until
the shared prefix checkpoint has been collected and hash-frozen.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import statistics
import sys
from typing import Any, Mapping, Sequence

import numpy as np


PACKAGE = Path(__file__).resolve().parent
REPOSITORY = Path(__file__).resolve().parents[3]
if str(REPOSITORY) not in sys.path:
    sys.path.insert(0, str(REPOSITORY))

from experiments.nanogpt_local import (  # noqa: E402
    build_openwebtext_cooldown_2p5b_v001 as corpus,
)
from experiments.nanogpt_local import (  # noqa: E402
    e2e_sgd_rpath_factorization_124m_2p5b_analysis_v002 as analysis,
)
from experiments.nanogpt_local import (  # noqa: E402
    e2e_sgd_rpath_factorization_124m_2p5b_v002 as core,
)


DATA_FILENAMES = {
    "metadata": "metadata.json",
    "train": "pretrain_train.bin",
    "validation": "pretrain_validation.bin",
    "validation_probe": "validation_probe.npy",
    "training_tape": "training_tape.npy",
}
DATA_SHAPES = {
    "train": (2 * core.CORPUS_TOKENS, core.CORPUS_TOKENS, "uint16"),
    "validation": (33_554_432, 16_777_216, "uint16"),
}
CONTAINER_IMAGE = (
    "docker.io/pytorch/pytorch:2.13.0-cuda12.6-cudnn9-runtime@sha256:"
    "6acf597eeb8e376a96580dde4952f37cc017fef732bb40bfc73f28f25e3f64b4"
)
# The numerical campaign is new, but the dataset bytes are not.  Keep the
# already-uploaded SHA-addressed v001 namespace so v002 never uploads a
# duplicate ~5 GB corpus or tape.
SHARED_DATA_OSDF_ROOT = (
    "osdf:///chtc/staging/w/wang3587/"
    "nanogpt124m-e2e-sgd-rpath-factorization-2p5b-v001"
)
DATA_MANIFEST_SCHEMA = (
    "nanogpt124m_e2e_sgd_rpath_data_manifest_v002"
)
SHARED_CACHE_SCHEMA = (
    "nanogpt124m_e2e_sgd_rpath_shared_cache_v002"
)
TAIL_MATERIALIZATION_SCHEMA = (
    "nanogpt124m_e2e_sgd_rpath_tail_materialization_v002"
)


class MaterializationError(RuntimeError):
    """A local artifact is absent or violates the frozen campaign shape."""


def _json_bytes(value: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(
            value,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    raw = _json_bytes(value)
    temporary = path.with_name(path.name + ".part")
    if path.exists() or path.is_symlink() or temporary.exists() or temporary.is_symlink():
        raise MaterializationError(f"refusing to replace output: {path}")
    with temporary.open("xb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        value = json.load(stream)
    if type(value) is not dict:
        raise MaterializationError(f"JSON root must be an object: {path}")
    return value


def _regular(path: Path, label: str) -> Path:
    result = path.expanduser().resolve()
    if not result.is_file() or result.is_symlink():
        raise MaterializationError(f"{label} is absent or linked: {result}")
    return result


def _record(path: Path, staged_path: str) -> dict[str, Any]:
    return {
        "path": staged_path,
        "sha256": core._sha256_file(path),
        "size_bytes": path.stat().st_size,
    }


def _materialize_tape(data_dir: Path) -> Path:
    path = data_dir / DATA_FILENAMES["training_tape"]
    if path.is_file() and not path.is_symlink():
        return path
    if path.exists() or path.is_symlink():
        raise MaterializationError(f"training tape target is not absent: {path}")
    temporary = path.with_name(path.name + ".part")
    if temporary.exists() or temporary.is_symlink():
        raise MaterializationError(f"training tape partial exists: {temporary}")
    tape = core.make_training_tape()
    with temporary.open("xb") as stream:
        np.save(stream, tape, allow_pickle=False)
        stream.flush()
        os.fsync(stream.fileno())
    checked = np.load(temporary, mmap_mode="r", allow_pickle=False)
    if (
        checked.shape != (core.TAPE_CONTEXTS,)
        or checked.dtype.str != "<i8"
        or core._array_sha256(checked) != core.TAPE_OFFSETS_SHA256
    ):
        raise MaterializationError("training tape round trip changed")
    os.replace(temporary, path)
    return path


def _validate_expanded_corpus_metadata(
    metadata: Mapping[str, Any],
    paths: Mapping[str, Path],
    records: Mapping[str, Mapping[str, Any]],
    source_manifest: Mapping[str, Any],
) -> None:
    """Bind the frozen 2.5B builder lineage to the bytes being cached."""

    indices = metadata.get("dataset", {}).get("source_shard_indices")
    sources = metadata.get("sources")
    if (
        metadata.get("schema_version") != corpus.MATERIALIZED_SCHEMA
        or metadata.get("completed") is not True
        or metadata.get("dataset", {}).get("id") != corpus.DATASET_ID
        or metadata.get("dataset", {}).get("revision")
        != corpus.DATASET_REVISION
        or type(indices) is not list
        or not indices
        or indices != list(range(len(indices)))
        or type(sources) is not list
        or len(sources) != len(indices)
        or [record.get("shard_index") for record in sources] != indices
        or metadata.get("construction", {}).get("targets")
        != {
            "train_tokens": corpus.TRAIN_TOKEN_TARGET,
            "validation_tokens": corpus.VALIDATION_TOKEN_TARGET,
        }
        or metadata.get("splits", {}).get("train", {}).get("tokens_written")
        != corpus.TRAIN_TOKEN_TARGET
        or metadata.get("splits", {}).get("validation", {}).get("tokens_written")
        != corpus.VALIDATION_TOKEN_TARGET
        or metadata.get("validation_probe", {}).get("rows")
        != corpus.VALIDATION_PROBE_DOCUMENTS
        or metadata.get("validation_probe", {}).get("all_documents_distinct")
        is not True
        or metadata.get("validation_probe", {}).get(
            "all_documents_complete_in_validation_stream"
        )
        is not True
    ):
        raise MaterializationError("expanded corpus metadata contract changed")

    metadata_artifacts = metadata.get("artifacts")
    artifact_roles = {
        "train": "pretrain_train",
        "validation": "pretrain_validation",
        "validation_probe": "validation_probe",
    }
    if type(metadata_artifacts) is not dict:
        raise MaterializationError("expanded corpus artifact inventory is absent")
    for role, metadata_role in artifact_roles.items():
        observed = metadata_artifacts.get(metadata_role)
        if (
            type(observed) is not dict
            or observed.get("path") != paths[role].name
            or observed.get("bytes") != records[role]["size_bytes"]
            or observed.get("sha256") != records[role]["sha256"]
        ):
            raise MaterializationError(
                f"expanded corpus metadata does not bind {role} bytes"
            )
    if (
        records["validation"]["sha256"] != corpus.LEGACY_VALIDATION_SHA256
        or records["validation_probe"]["sha256"]
        != corpus.LEGACY_PROBE_SHA256
        or corpus._sha256_prefix(
            paths["train"], corpus.LEGACY_TRAIN_PREFIX_TOKENS * 2
        )
        != corpus.LEGACY_TRAIN_PREFIX_SHA256
    ):
        raise MaterializationError(
            "expanded corpus is not continuous with the established 700M corpus"
        )
    try:
        corpus._validate_materialized_metadata(metadata, source_manifest)
    except corpus.LargeCorpusContractError as error:
        raise MaterializationError(
            f"expanded corpus does not match its complete source manifest: {error}"
        ) from error


def freeze_data(
    data_dir: Path,
    source_manifest_path: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    root = data_dir.expanduser().resolve()
    if not root.is_dir() or root.is_symlink():
        raise MaterializationError(f"data directory is absent or linked: {root}")
    source_path = _regular(source_manifest_path, "complete source manifest")
    try:
        source_manifest = corpus.load_source_manifest(source_path)
        source_report = corpus.validate_source_manifest(
            source_manifest, require_complete=True
        )
    except corpus.LargeCorpusContractError as error:
        raise MaterializationError(
            f"source manifest is not complete and capacity-certified: {error}"
        ) from error
    paths = {
        role: _regular(root / filename, role)
        for role, filename in DATA_FILENAMES.items()
        if role != "training_tape"
    }
    paths["source_manifest"] = source_path
    for role, (size, tokens, _) in DATA_SHAPES.items():
        if paths[role].stat().st_size != size:
            raise MaterializationError(
                f"{role} must contain exactly {tokens} uint16 tokens"
            )
    probe = np.load(paths["validation_probe"], allow_pickle=False)
    if (
        probe.shape != (core.VALIDATION_CONTEXTS,)
        or probe.dtype.str != "<i8"
        or len(set(int(value) for value in probe)) != core.VALIDATION_CONTEXTS
        or int(np.min(probe)) < 0
        or int(np.max(probe)) + core.BLOCK_SIZE >= 16_777_216
    ):
        raise MaterializationError("validation probe shape/range changed")
    metadata = _load_json(paths["metadata"])
    records = {
        role: _record(paths[role], f"inputs/{paths[role].name}")
        for role in ("metadata", "train", "validation", "validation_probe")
    }
    records["source_manifest"] = _record(
        source_path,
        "inputs/openwebtext_cooldown_2p5b_source_manifest_complete.json",
    )
    _validate_expanded_corpus_metadata(
        metadata, paths, records, source_manifest
    )

    paths["training_tape"] = _materialize_tape(root)
    records["training_tape"] = _record(
        paths["training_tape"], "inputs/training_tape.npy"
    )
    tape = np.load(paths["training_tape"], mmap_mode="r", allow_pickle=False)
    if (
        tape.shape != (core.TAPE_CONTEXTS,)
        or tape.dtype.str != "<i8"
        or core._array_sha256(tape) != core.TAPE_OFFSETS_SHA256
    ):
        raise MaterializationError("training tape identity changed")
    data = {
        "source_manifest": {
            **records["source_manifest"],
            "schema_version": corpus.MANIFEST_SCHEMA,
            "status": "complete",
            "manifest_sha256": source_manifest["manifest_sha256"],
            "capacity_certificate_sha256": source_manifest[
                "capacity_audit"
            ]["certificate_sha256"],
            "source_prefix_stop_exclusive": source_report[
                "source_prefix_stop_exclusive"
            ],
        },
        "metadata": records["metadata"],
        "train": {
            **records["train"],
            "token_count": core.CORPUS_TOKENS,
            "dtype": "uint16",
        },
        "validation": {
            **records["validation"],
            "token_count": 16_777_216,
            "dtype": "uint16",
            "document_disjoint_from_train": True,
        },
        "validation_probe": {
            **records["validation_probe"],
            "context_count": core.VALIDATION_CONTEXTS,
            "dtype": "int64",
            "one_context_per_document": True,
        },
        "training_tape": {
            **records["training_tape"],
            "context_count": core.TAPE_CONTEXTS,
            "dtype": "int64",
            "offsets_sha256": core.TAPE_OFFSETS_SHA256,
        },
    }
    cache = {
        "schema_version": SHARED_CACHE_SCHEMA,
        "campaign_id": core.CAMPAIGN_ID,
        "status": "reuse_existing_sha_addressed_v001_objects",
        "upload_policy": "reuse_existing_sha256_objects_no_v002_reupload",
        "planned_upload_rounds": 0,
        "maximum_uploads_per_artifact": 1,
        "source_cache_campaign_id": (
            "nanogpt124m-e2e-sgd-rpath-factorization-2p5b-v001"
        ),
        "artifacts": {
            role: {
                "source_path": str(paths[role]),
                "sha256": record["sha256"],
                "size_bytes": record["size_bytes"],
                "osdf_uri": (
                    f"{SHARED_DATA_OSDF_ROOT}/{record['sha256']}/"
                    f"{Path(record['path']).name}"
                ),
                "upload_status": "reuse_existing",
            }
            for role, record in data.items()
        },
    }
    return data, cache


def _base_config(
    phase: str,
    data: Mapping[str, Any],
    *,
    arm_id: str | None = None,
    resource_calibration: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if phase == "b0":
        if arm_id is not None or resource_calibration is not None:
            raise MaterializationError("B0 cannot carry an arm or calibration")
        resource_budget: Any = {
            "schema_version": core.RESOURCE_BUDGET_SCHEMA,
            "phase": "b0",
            "maximum_gpu_hours": core.B0_MAXIMUM_GPU_HOURS,
            "purpose": "response_free_H200_measurement_only",
        }
    else:
        if phase == "prefix" and arm_id is not None:
            raise MaterializationError("prefix cannot select a tail arm")
        if phase == "tail" and arm_id not in core.ARM_IDS:
            raise MaterializationError("tail must select one frozen arm")
        if resource_calibration is None:
            raise MaterializationError(
                f"{phase} requires a completed H200 B0 calibration"
            )
        resource_budget = core.derive_resource_budget(
            phase, arm_id, resource_calibration
        )
    return {
        "schema_version": core.CONFIG_SCHEMA,
        "campaign_id": core.CAMPAIGN_ID,
        "campaign_contract_sha256": core.campaign_contract_sha256(),
        "phase": phase,
        "arm_id": arm_id,
        "primary_prediction_id": core.PRIMARY_PREDICTION_ID,
        "theorem_facing": False,
        "launch_control": {"cluster_submission_authorized": False},
        "resource_calibration": (
            None if resource_calibration is None else dict(resource_calibration)
        ),
        "model": core.MODEL_SPEC,
        "optimizer": {
            "name": "sgd",
            "momentum": 0.0,
            "dampening": 0.0,
            "nesterov": False,
            "weight_decay": 0.0,
            "gradient_clip": 0.0,
            "foreach": False,
        },
        "schedule_manifest_sha256": core.schedule_manifest()["sha256"],
        "data": dict(data),
        "prefix_checkpoint": None,
        "runtime": {
            **core.RUNTIME_SPEC,
            "container_image": CONTAINER_IMAGE,
            "formal_wall_time_and_gpu_hour_budget": resource_budget,
        },
        "measurements": {
            "primary": "fixed_validation_probe_cross_entropy",
            "retain_per_update_training_cross_entropy": True,
            "retain_raw_validation_context_cross_entropy": True,
            "lei_style": "fixed_probe_validation_CE_on_T_or_update_axis",
            "blake_style": "training_CE_with_16_and_64_update_mean_lines",
            "numerical_runtime": core.NUMERICS_ID,
        },
        "artifacts": {
            "result": core.RESULT_FILENAME,
            "training_trace": core.TRACE_FILENAME,
            "validation_evaluations": core.EVALUATION_FILENAME,
            "prefix_checkpoint": (
                core.PREFIX_CHECKPOINT_FILENAME if phase == "prefix" else None
            ),
        },
    }


def _data_identity(data: Mapping[str, Any]) -> str:
    return core._canonical_sha256(
        {
            role: {
                key: record[key]
                for key in ("sha256", "size_bytes")
            }
            for role, record in data.items()
        }
    )


def materialize_data_phase(
    data_dir: Path,
    source_manifest_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    output = output_dir.expanduser().resolve()
    if output.exists() or output.is_symlink():
        raise MaterializationError(f"output directory already exists: {output}")
    data, cache = freeze_data(data_dir, source_manifest_path)
    manifest = {
        "schema_version": DATA_MANIFEST_SCHEMA,
        "campaign_id": core.CAMPAIGN_ID,
        "data": data,
        "data_identity_sha256": _data_identity(data),
    }
    b0 = _base_config("b0", data)
    core.validate_config(b0)
    if output.exists() or output.is_symlink():
        raise MaterializationError(f"output directory appeared during validation: {output}")
    output.mkdir(parents=True)
    _atomic_json(output / "data_manifest.json", manifest)
    _atomic_json(output / "shared_cache_manifest.json", cache)
    _atomic_json(output / "b0_config.json", b0)
    return {
        "classification": "DATA_AND_PREPREFIX_CONFIGS_FROZEN_LAUNCH_FALSE",
        "output_dir": str(output),
        "data_identity_sha256": manifest["data_identity_sha256"],
        "files": sorted(path.name for path in output.iterdir()),
        "launch_authorized": False,
    }


def _recompute_b0_summary(
    *,
    label: str,
    batch_size: int,
    updates: int,
    training: np.ndarray,
    validation: np.ndarray,
    step_times: np.ndarray,
    evaluation_times: np.ndarray,
    optimizer_state_entries: np.ndarray,
    parameters_finite: np.ndarray,
    parameters_fp32: np.ndarray,
    gradients_fp32: np.ndarray,
) -> dict[str, Any]:
    if (
        training.dtype != np.float32
        or training.shape != (updates,)
        or validation.dtype != np.float32
        or validation.shape != (2, core.VALIDATION_CONTEXTS)
        or step_times.dtype != np.float64
        or step_times.shape != (updates,)
        or evaluation_times.dtype != np.float64
        or evaluation_times.shape != (2,)
        or optimizer_state_entries.dtype != np.int64
        or optimizer_state_entries.shape != ()
        or parameters_finite.dtype != np.bool_
        or parameters_finite.shape != ()
        or parameters_fp32.dtype != np.bool_
        or parameters_fp32.shape != ()
        or gradients_fp32.dtype != np.bool_
        or gradients_fp32.shape != ()
        or not np.all(np.isfinite(training))
        or not np.all(np.isfinite(validation))
        or not np.all(np.isfinite(step_times))
        or not np.all(np.isfinite(evaluation_times))
        or np.any(training < 0.0)
        or np.any(validation < 0.0)
        or np.any(step_times <= 0.0)
        or np.any(evaluation_times <= 0.0)
    ):
        raise MaterializationError(f"B0 {label} raw metrics changed")
    optimizer_entries = int(optimizer_state_entries.item())
    finite_parameters = bool(parameters_finite.item())
    fp32_parameters = bool(parameters_fp32.item())
    fp32_gradients = bool(gradients_fp32.item())
    timed = [float(value) for value in step_times[8:]]
    median = statistics.median(timed)
    initial = float(np.mean(validation[0], dtype=np.float64))
    final = float(np.mean(validation[1], dtype=np.float64))
    return {
        "label": label,
        "batch_size": batch_size,
        "learning_rate": core.CANONICAL_BASE_LEARNING_RATE,
        "ratio_B_over_eta": batch_size / core.CANONICAL_BASE_LEARNING_RATE,
        "updates": updates,
        "contexts": batch_size * updates,
        "tokens": batch_size * updates * core.BLOCK_SIZE,
        "initial_validation_cross_entropy": initial,
        "final_validation_cross_entropy": final,
        "maximum_training_cross_entropy": float(np.max(training)),
        "minimum_training_cross_entropy": float(np.min(training)),
        "step_time_median_seconds": median,
        "step_time_p90_seconds": float(np.percentile(step_times[8:], 90)),
        "tokens_per_second": batch_size * core.BLOCK_SIZE / median,
        "validation_evaluation_time_max_seconds": float(
            np.max(evaluation_times)
        ),
        "optimizer_state_entries": optimizer_entries,
        "parameters_finite": finite_parameters,
        "parameters_fp32": fp32_parameters,
        "gradients_fp32": fp32_gradients,
        "all_values_and_parameters_finite": bool(
            finite_parameters
            and fp32_parameters
            and fp32_gradients
            and optimizer_entries == 0
        ),
        "validation_increase_below_abort_margin": bool(final <= initial + 0.25),
    }


def _b0_summary_matches(observed: Any, expected: Mapping[str, Any]) -> bool:
    if type(observed) is not dict or set(observed) != set(expected):
        return False
    for key, expected_value in expected.items():
        observed_value = observed[key]
        if type(expected_value) is bool:
            if type(observed_value) is not bool or observed_value != expected_value:
                return False
        elif type(expected_value) is int:
            if type(observed_value) is not int or observed_value != expected_value:
                return False
        elif type(expected_value) is float:
            if (
                type(observed_value) not in (int, float)
                or type(observed_value) is bool
                or not math.isfinite(float(observed_value))
                or not math.isclose(
                    float(observed_value),
                    expected_value,
                    rel_tol=1e-12,
                    abs_tol=1e-12,
                )
            ):
                return False
        elif observed_value != expected_value:
            return False
    return True


def _validate_b0_result(
    result_path: Path,
    data_identity_sha256: str,
) -> dict[str, Any]:
    result_file = _regular(result_path, "B0 result")
    if result_file.name != core.RESULT_FILENAME:
        raise MaterializationError("B0 result must retain its result.json name")
    result = _load_json(result_file)
    expected_gates = {
        "base_path_finite",
        "maximum_batch_path_finite",
        "base_validation_abort_margin",
        "maximum_batch_validation_abort_margin",
        "plain_sgd_state_empty",
        "fp32_parameters_and_gradients",
        "bf16_tf32_flash_only_runtime",
        "peak_reserved_below_80_percent_H200",
        "wall_time_below_0p25_H200_hours",
    }
    if (
        result.get("schema_version") != core.RESULT_SCHEMA
        or result.get("campaign_id") != core.CAMPAIGN_ID
        or result.get("campaign_contract_sha256")
        != core.campaign_contract_sha256()
        or result.get("numerics_id") != core.NUMERICS_ID
        or result.get("phase") != "b0"
        or result.get("arm_id") is not None
        or result.get("status") != "passed"
        or result.get("contains_prediction_response") is not False
        or not core._is_hex64(result.get("initialization_sha256"))
        or result.get("schedule_manifest_sha256")
        != core.schedule_manifest()["sha256"]
        or result.get("data_identity_sha256") != data_identity_sha256
        or result.get("resource_calibration") is not None
        or result.get("resource_budget")
        != _base_config("b0", {})["runtime"][
            "formal_wall_time_and_gpu_hour_budget"
        ]
        or type(result.get("gates")) is not dict
        or set(result["gates"]) != expected_gates
        or any(type(value) is not bool for value in result["gates"].values())
        or not all(result["gates"].values())
    ):
        raise MaterializationError("B0 result identity or gate changed")
    measurements = result.get("measurements")
    measurement_contract = {
        "base": ("base_B4_eta0p005", 4, core.B0_BASE_UPDATES),
        "maximum_batch": (
            "maximum_batch_B40_eta0p005",
            40,
            core.B0_MAX_BATCH_UPDATES,
        ),
    }
    if type(measurements) is not dict or set(measurements) != set(
        measurement_contract
    ):
        raise MaterializationError("B0 measurements are absent")
    runtime = result.get("runtime")
    if (
        type(runtime) is not dict
        or runtime.get("device") != "cuda"
        or type(runtime.get("device_name")) is not str
        or "H200" not in runtime["device_name"]
        or not core._runtime_record_matches_contract(runtime)
        or not core._positive_finite(runtime.get("elapsed_seconds"))
        or not core._positive_finite(runtime.get("gpu_hours"))
        or not math.isclose(
            float(runtime["gpu_hours"]),
            float(runtime["elapsed_seconds"]) / 3_600.0,
            rel_tol=1e-12,
            abs_tol=0.0,
        )
        or float(runtime["gpu_hours"]) > core.B0_MAXIMUM_GPU_HOURS
        or type(runtime.get("physical_gpu_memory_bytes")) is not int
        or runtime["physical_gpu_memory_bytes"] <= 0
        or type(runtime.get("peak_allocated_bytes")) is not int
        or runtime["peak_allocated_bytes"] < 0
        or type(runtime.get("peak_reserved_bytes")) is not int
        or runtime["peak_reserved_bytes"] < runtime["peak_allocated_bytes"]
    ):
        raise MaterializationError("B0 H200 runtime contract changed")
    artifact = result.get("artifacts", {}).get("diagnostic_trace")
    if type(artifact) is not dict or artifact.get("path") != core.TRACE_FILENAME:
        raise MaterializationError("B0 diagnostic trace record changed")
    trace_path = result_file.parent / core.TRACE_FILENAME
    if (
        not trace_path.is_file()
        or trace_path.is_symlink()
        or trace_path.stat().st_size != artifact.get("size_bytes")
        or core._sha256_file(trace_path) != artifact.get("sha256")
    ):
        raise MaterializationError("B0 diagnostic trace bytes changed")
    with np.load(trace_path, allow_pickle=False) as payload:
        members = {
            "schema_version",
            "phase",
            "base_training_cross_entropy",
            "maximum_batch_training_cross_entropy",
            "base_validation_context_cross_entropy",
            "maximum_batch_validation_context_cross_entropy",
            "base_step_time_seconds",
            "maximum_batch_step_time_seconds",
            "base_validation_evaluation_time_seconds",
            "maximum_batch_validation_evaluation_time_seconds",
            "base_optimizer_state_entries",
            "maximum_batch_optimizer_state_entries",
            "base_parameters_finite",
            "maximum_batch_parameters_finite",
            "base_parameters_fp32",
            "maximum_batch_parameters_fp32",
            "base_gradients_fp32",
            "maximum_batch_gradients_fp32",
        }
        if set(payload.files) != members:
            raise MaterializationError("B0 diagnostic trace inventory changed")
        if (
            payload["schema_version"].shape != ()
            or payload["schema_version"].dtype.kind not in {"U", "S"}
            or payload["phase"].shape != ()
            or payload["phase"].dtype.kind not in {"U", "S"}
            or str(payload["schema_version"].item()) != core.TRACE_SCHEMA
            or str(payload["phase"].item()) != "b0"
        ):
            raise MaterializationError("B0 diagnostic trace identity changed")
        recomputed = {}
        for role, (label, batch_size, updates) in measurement_contract.items():
            prefix = "base" if role == "base" else "maximum_batch"
            recomputed[role] = _recompute_b0_summary(
                label=label,
                batch_size=batch_size,
                updates=updates,
                training=payload[f"{prefix}_training_cross_entropy"],
                validation=payload[
                    f"{prefix}_validation_context_cross_entropy"
                ],
                step_times=payload[f"{prefix}_step_time_seconds"],
                evaluation_times=payload[
                    f"{prefix}_validation_evaluation_time_seconds"
                ],
                optimizer_state_entries=payload[
                    f"{prefix}_optimizer_state_entries"
                ],
                parameters_finite=payload[f"{prefix}_parameters_finite"],
                parameters_fp32=payload[f"{prefix}_parameters_fp32"],
                gradients_fp32=payload[f"{prefix}_gradients_fp32"],
            )
    for role, expected in recomputed.items():
        if not _b0_summary_matches(measurements[role], expected):
            raise MaterializationError(
                f"B0 {role} summary does not replay from raw metrics"
            )
    recomputed_gates = {
        "base_path_finite": recomputed["base"][
            "all_values_and_parameters_finite"
        ],
        "maximum_batch_path_finite": recomputed["maximum_batch"][
            "all_values_and_parameters_finite"
        ],
        "base_validation_abort_margin": recomputed["base"][
            "validation_increase_below_abort_margin"
        ],
        "maximum_batch_validation_abort_margin": recomputed["maximum_batch"][
            "validation_increase_below_abort_margin"
        ],
        "plain_sgd_state_empty": bool(
            recomputed["base"]["optimizer_state_entries"] == 0
            and recomputed["maximum_batch"]["optimizer_state_entries"] == 0
        ),
        "fp32_parameters_and_gradients": bool(
            recomputed["base"]["parameters_fp32"]
            and recomputed["maximum_batch"]["parameters_fp32"]
            and recomputed["base"]["gradients_fp32"]
            and recomputed["maximum_batch"]["gradients_fp32"]
        ),
        "bf16_tf32_flash_only_runtime": (
            core._runtime_record_matches_contract(runtime)
        ),
        "peak_reserved_below_80_percent_H200": bool(
            runtime["peak_reserved_bytes"]
            <= 0.8 * runtime["physical_gpu_memory_bytes"]
        ),
        "wall_time_below_0p25_H200_hours": bool(
            runtime["gpu_hours"] <= core.B0_MAXIMUM_GPU_HOURS
        ),
    }
    if result["gates"] != recomputed_gates or not all(
        recomputed_gates.values()
    ):
        raise MaterializationError("B0 gates do not replay from raw metrics")
    return {
        "schema_version": core.RESOURCE_CALIBRATION_SCHEMA,
        "b0_result_sha256": core._sha256_file(result_file),
        "b0_trace_sha256": core._sha256_file(trace_path),
        "initialization_sha256": result["initialization_sha256"],
        "data_identity_sha256": data_identity_sha256,
        "schedule_manifest_sha256": core.schedule_manifest()["sha256"],
        "device_name": runtime["device_name"],
        "base_step_time_p90_seconds": recomputed["base"][
            "step_time_p90_seconds"
        ],
        "maximum_batch_step_time_p90_seconds": recomputed[
            "maximum_batch"
        ]["step_time_p90_seconds"],
        "validation_evaluation_time_max_seconds": max(
            recomputed["base"]["validation_evaluation_time_max_seconds"],
            recomputed["maximum_batch"][
                "validation_evaluation_time_max_seconds"
            ],
        ),
    }


def materialize_prefix_phase(
    data_manifest_path: Path,
    b0_result_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    data_manifest = _load_json(_regular(data_manifest_path, "data manifest"))
    if (
        data_manifest.get("schema_version") != DATA_MANIFEST_SCHEMA
        or data_manifest.get("campaign_id") != core.CAMPAIGN_ID
        or type(data_manifest.get("data")) is not dict
    ):
        raise MaterializationError("data manifest identity changed")
    data = data_manifest["data"]
    identity = _data_identity(data)
    if data_manifest.get("data_identity_sha256") != identity:
        raise MaterializationError("data manifest digest changed")
    core.validate_config(_base_config("b0", data))
    calibration = _validate_b0_result(b0_result_path, identity)
    prefix = _base_config(
        "prefix", data, resource_calibration=calibration
    )
    core.validate_config(prefix)
    output = output_dir.expanduser().resolve()
    if output.exists() or output.is_symlink():
        raise MaterializationError(f"output directory already exists: {output}")
    output.mkdir(parents=True)
    _atomic_json(output / "H200_BF16_resource_calibration.json", calibration)
    _atomic_json(output / "prefix_config.json", prefix)
    return {
        "classification": (
            "PREFIX_CONFIG_FROZEN_FROM_PASSED_H200_BF16_B0_LAUNCH_FALSE"
        ),
        "output_dir": str(output),
        "b0_result_sha256": calibration["b0_result_sha256"],
        "predicted_gpu_hours": prefix["runtime"][
            "formal_wall_time_and_gpu_hour_budget"
        ]["predicted_gpu_hours"],
        "maximum_gpu_hours": prefix["runtime"][
            "formal_wall_time_and_gpu_hour_budget"
        ]["maximum_gpu_hours"],
        "files": sorted(path.name for path in output.iterdir()),
        "launch_authorized": False,
    }


def materialize_tail_phase(
    data_manifest_path: Path,
    checkpoint_path: Path,
    prefix_result_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    data_manifest = _load_json(_regular(data_manifest_path, "data manifest"))
    if (
        data_manifest.get("schema_version") != DATA_MANIFEST_SCHEMA
        or data_manifest.get("campaign_id") != core.CAMPAIGN_ID
        or type(data_manifest.get("data")) is not dict
    ):
        raise MaterializationError("data manifest identity changed")
    data = data_manifest["data"]
    data_identity_sha256 = _data_identity(data)
    if data_manifest.get("data_identity_sha256") != data_identity_sha256:
        raise MaterializationError("data manifest digest changed")
    checkpoint = _regular(checkpoint_path, "prefix checkpoint")
    prefix_result_file = _regular(prefix_result_path, "prefix result")
    if (
        prefix_result_file.name != core.RESULT_FILENAME
        or checkpoint
        != (prefix_result_file.parent / core.PREFIX_CHECKPOINT_FILENAME).resolve()
    ):
        raise MaterializationError(
            "prefix result and checkpoint must be one intact result directory"
        )
    prefix_result = _load_json(prefix_result_file)
    if (
        prefix_result.get("schema_version") != core.RESULT_SCHEMA
        or prefix_result.get("campaign_id") != core.CAMPAIGN_ID
        or prefix_result.get("campaign_contract_sha256")
        != core.campaign_contract_sha256()
        or prefix_result.get("numerics_id") != core.NUMERICS_ID
        or prefix_result.get("phase") != "prefix"
        or prefix_result.get("status") != "completed"
        or prefix_result.get("contains_prediction_response") is not False
        or prefix_result.get("schedule_manifest_sha256")
        != core.schedule_manifest()["sha256"]
        or prefix_result.get("data_identity_sha256") != data_identity_sha256
        or prefix_result.get("completion_gates")
        != core._expected_completion_gates()
        or prefix_result.get("counts")
        != {
            "updates": core.CONSTANT_SOURCE_UPDATES,
            "contexts": core.PREFIX_CONTEXTS,
            "tokens": core.PREFIX_TOKENS,
            "evaluation_rows": len(core._prefix_evaluation_macros()),
        }
    ):
        raise MaterializationError("prefix result is not a completed campaign prefix")
    calibration = prefix_result.get("resource_calibration")
    expected_prefix = _base_config(
        "prefix", data, resource_calibration=calibration
    )
    core.validate_config(expected_prefix)
    if prefix_result.get("resource_budget") != expected_prefix["runtime"][
        "formal_wall_time_and_gpu_hour_budget"
    ]:
        raise MaterializationError("prefix resource calibration changed")
    checkpoint_artifact = prefix_result.get("artifacts", {}).get("checkpoint", {})
    checkpoint_sha = core._sha256_file(checkpoint)
    if (
        checkpoint_artifact.get("sha256") != checkpoint_sha
        or checkpoint_artifact.get("size_bytes") != checkpoint.stat().st_size
    ):
        raise MaterializationError("prefix result does not bind the checkpoint bytes")
    try:
        prefix_report = analysis.validate_result_directory(
            prefix_result_file.parent
        )
    except (analysis.AnalysisError, core.CampaignError) as error:
        raise MaterializationError(
            f"prefix result-directory validation failed: {error}"
        ) from error
    if (
        prefix_report.get("status") != "passed"
        or prefix_report.get("phase") != "prefix"
        or prefix_report.get("campaign_contract_sha256")
        != core.campaign_contract_sha256()
        or prefix_report.get("numerics_id") != core.NUMERICS_ID
        or prefix_report.get("data_identity_sha256") != data_identity_sha256
        or prefix_report.get("schedule_manifest_sha256")
        != core.schedule_manifest()["sha256"]
        or prefix_report.get("prefix_checkpoint_sha256") != checkpoint_sha
        or prefix_report.get("prefix_model_state_sha256")
        != prefix_result.get("checkpoint_model_state_sha256")
    ):
        raise MaterializationError("prefix checkpoint state identity changed")
    output = output_dir.expanduser().resolve()
    if output.exists() or output.is_symlink():
        raise MaterializationError(f"output directory already exists: {output}")
    output.mkdir(parents=True)
    checkpoint_record = {
        "path": "inputs/prefix_checkpoint.pt",
        "sha256": checkpoint_sha,
        "size_bytes": checkpoint.stat().st_size,
        "model_state_sha256": prefix_report["prefix_model_state_sha256"],
    }
    written = []
    budgets = {}
    for arm_id in core.ARM_IDS:
        config = _base_config(
            "tail",
            data,
            arm_id=arm_id,
            resource_calibration=calibration,
        )
        config["prefix_checkpoint"] = checkpoint_record
        core.validate_config(config)
        name = arm_id.replace("eight_one_one", "811").replace(
            "wsd_exp_80_20", "wsd"
        ).replace("fixed_batch_lr", "fblr").replace("fixed_lr_batch", "flrbs")
        filename = f"tail_{name}_config.json"
        _atomic_json(output / filename, config)
        written.append(filename)
        budgets[arm_id] = config["runtime"][
            "formal_wall_time_and_gpu_hour_budget"
        ]
    manifest = {
        "schema_version": TAIL_MATERIALIZATION_SCHEMA,
        "campaign_id": core.CAMPAIGN_ID,
        "campaign_contract_sha256": core.campaign_contract_sha256(),
        "numerics_id": core.NUMERICS_ID,
        "prefix_checkpoint": checkpoint_record,
        "resource_calibration_sha256": core._canonical_sha256(calibration),
        "tail_resource_budgets": budgets,
        "tail_configs": sorted(written),
        "topology": "four_independent_single_H200_jobs_no_DDP",
        "launch_authorized": False,
    }
    _atomic_json(output / "tail_manifest.json", manifest)
    return {
        "classification": "FOUR_TAIL_CONFIGS_FROZEN_LAUNCH_FALSE",
        "output_dir": str(output),
        "files": sorted(path.name for path in output.iterdir()),
        "launch_authorized": False,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    plan = subparsers.add_parser("plan")
    plan.add_argument("--skip-tape-hash", action="store_true")
    data = subparsers.add_parser("freeze-data")
    data.add_argument("--data-dir", type=Path, required=True)
    data.add_argument("--source-manifest", type=Path, required=True)
    data.add_argument("--output-dir", type=Path, required=True)
    prefix = subparsers.add_parser("freeze-prefix")
    prefix.add_argument("--data-manifest", type=Path, required=True)
    prefix.add_argument("--b0-result", type=Path, required=True)
    prefix.add_argument("--output-dir", type=Path, required=True)
    tails = subparsers.add_parser("freeze-tails")
    tails.add_argument("--data-manifest", type=Path, required=True)
    tails.add_argument("--checkpoint", type=Path, required=True)
    tails.add_argument("--prefix-result", type=Path, required=True)
    tails.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "plan":
        report = core.preflight(verify_tape=not args.skip_tape_hash)
    elif args.command == "freeze-data":
        report = materialize_data_phase(
            args.data_dir, args.source_manifest, args.output_dir
        )
    elif args.command == "freeze-prefix":
        report = materialize_prefix_phase(
            args.data_manifest, args.b0_result, args.output_dir
        )
    else:
        report = materialize_tail_phase(
            args.data_manifest,
            args.checkpoint,
            args.prefix_result,
            args.output_dir,
        )
    print(json.dumps(report, sort_keys=True, ensure_ascii=False, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
