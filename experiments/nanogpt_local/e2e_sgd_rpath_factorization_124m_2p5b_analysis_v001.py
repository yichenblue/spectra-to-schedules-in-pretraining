"""Offline validator/collector for the 124M 2.5B ratio-path campaign."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from . import e2e_sgd_rpath_factorization_124m_2p5b_v001 as core


ANALYSIS_SCHEMA = "nanogpt124m_e2e_sgd_rpath_factorization_2p5b_analysis_v001"
CAMPAIGN_SCHEMA = "nanogpt124m_e2e_sgd_rpath_factorization_2p5b_campaign_validation_v001"
Json = dict[str, Any]


class AnalysisError(RuntimeError):
    """A retained artifact is incomplete, malformed, or inconsistent."""


def _load_json(path: Path) -> Json:
    with path.open("r", encoding="utf-8") as stream:
        value = json.load(stream)
    if type(value) is not dict:
        raise AnalysisError(f"JSON root must be an object: {path}")
    return value


def _scalar_text(value: np.ndarray, label: str) -> str:
    if value.shape != () or value.dtype.kind not in {"U", "S"}:
        raise AnalysisError(f"{label} must be a scalar string")
    return str(value.item())


def _artifact_path(root: Path, record: Any, expected_name: str) -> Path:
    if type(record) is not dict or record.get("path") != expected_name:
        raise AnalysisError(f"artifact record changed: {expected_name}")
    path = root / expected_name
    if (
        not path.is_file()
        or path.is_symlink()
        or path.stat().st_size != record.get("size_bytes")
        or core._sha256_file(path) != record.get("sha256")
    ):
        raise AnalysisError(f"artifact bytes changed: {path}")
    return path


def _load_trace(path: Path, phase: str, arm_id: str | None, rows: int) -> np.ndarray:
    with np.load(path, allow_pickle=False) as payload:
        expected = {"schema_version", "phase", "training_cross_entropy"}
        if phase == "tail":
            expected.add("arm_id")
        if set(payload.files) != expected:
            raise AnalysisError("training trace member inventory changed")
        if (
            _scalar_text(payload["schema_version"], "trace schema")
            != core.TRACE_SCHEMA
            or _scalar_text(payload["phase"], "trace phase") != phase
        ):
            raise AnalysisError("training trace identity changed")
        if phase == "tail" and _scalar_text(payload["arm_id"], "trace arm") != arm_id:
            raise AnalysisError("training trace arm changed")
        values = np.ascontiguousarray(payload["training_cross_entropy"])
    if (
        values.dtype != np.float32
        or values.shape != (rows,)
        or not np.all(np.isfinite(values))
        or np.any(values < 0.0)
    ):
        raise AnalysisError("training trace values changed")
    return values


def _expected_tail_evaluations(arm_id: str) -> dict[str, np.ndarray]:
    plan = core._arm_plan(arm_id)
    selected = set(int(value) for value in plan["evaluation_macros"])
    macros = [core.CONSTANT_MACROS]
    updates = [core.CONSTANT_SOURCE_UPDATES]
    cursors = [core.PREFIX_CONTEXTS]
    intrinsic = [core.PREFIX_INTRINSIC_TIME]
    for step in core._iter_tail_steps(arm_id):
        if not step["macro_completed"]:
            continue
        macro = int(step["macro_index"]) + 1
        if macro in selected and macro != core.CONSTANT_MACROS:
            macros.append(macro)
            updates.append(int(step["global_update"]))
            cursors.append(int(step["tape_context_stop"]))
            intrinsic.append(float(step["intrinsic_time_after"]))
    return {
        "macro_index": np.asarray(macros, dtype=np.int64),
        "global_update": np.asarray(updates, dtype=np.int64),
        "tape_context_cursor": np.asarray(cursors, dtype=np.int64),
        "tokens_seen": np.asarray(cursors, dtype=np.int64) * core.BLOCK_SIZE,
        "intrinsic_time": np.asarray(intrinsic, dtype=np.float64),
    }


def _expected_prefix_evaluations() -> dict[str, np.ndarray]:
    macros = np.asarray(core._prefix_evaluation_macros(), dtype=np.int64)
    updates = 4 * macros
    cursors = 4 * updates
    return {
        "macro_index": macros,
        "global_update": updates,
        "tape_context_cursor": cursors,
        "tokens_seen": cursors * core.BLOCK_SIZE,
        "intrinsic_time": macros.astype(np.float64) * core.MACRO_INTRINSIC_TIME,
    }


def _load_evaluations(
    path: Path, expected: Mapping[str, np.ndarray]
) -> dict[str, np.ndarray]:
    members = {
        "schema_version",
        "macro_index",
        "global_update",
        "tape_context_cursor",
        "tokens_seen",
        "intrinsic_time",
        "validation_cross_entropy",
        "context_cross_entropy",
    }
    with np.load(path, allow_pickle=False) as payload:
        if set(payload.files) != members:
            raise AnalysisError("validation artifact member inventory changed")
        if (
            _scalar_text(payload["schema_version"], "evaluation schema")
            != core.EVALUATION_SCHEMA
        ):
            raise AnalysisError("validation artifact schema changed")
        arrays = {
            name: np.ascontiguousarray(payload[name])
            for name in members
            if name != "schema_version"
        }
    row_count = len(expected["macro_index"])
    for name, value in expected.items():
        if not np.array_equal(arrays[name], value):
            raise AnalysisError(f"validation coordinate changed: {name}")
    contexts = arrays["context_cross_entropy"]
    means = arrays["validation_cross_entropy"]
    if (
        contexts.dtype != np.float32
        or contexts.shape != (row_count, core.VALIDATION_CONTEXTS)
        or means.dtype != np.float64
        or means.shape != (row_count,)
        or not np.all(np.isfinite(contexts))
        or not np.all(np.isfinite(means))
        or not np.allclose(
            means,
            np.mean(contexts, axis=1, dtype=np.float64),
            rtol=0.0,
            atol=1e-12,
        )
    ):
        raise AnalysisError("fixed-probe validation values changed")
    return arrays


def _runtime_is_exact_h200_fp32(value: Any) -> bool:
    if type(value) is not dict:
        return False
    numeric = (
        value.get("elapsed_seconds"),
        value.get("gpu_hours"),
        value.get("sampled_step_time_median_seconds"),
        value.get("sampled_step_time_p90_seconds"),
    )
    if any(
        type(item) not in (int, float)
        or not math.isfinite(float(item))
        or float(item) <= 0.0
        for item in numeric
    ):
        return False
    elapsed = float(value["elapsed_seconds"])
    return bool(
        value.get("device") == "cuda"
        and type(value.get("device_name")) is str
        and "H200" in value["device_name"]
        and value.get("dtype") == "float32"
        and value.get("deterministic_algorithms") is True
        and value.get("deterministic_warn_only") is False
        and value.get("flash_sdpa_enabled") is False
        and value.get("memory_efficient_sdpa_enabled") is False
        and value.get("cudnn_sdpa_enabled") is False
        and value.get("math_sdpa_enabled") is True
        and math.isclose(
            float(value["gpu_hours"]), elapsed / 3_600.0,
            rel_tol=1e-12, abs_tol=0.0,
        )
        and type(value.get("peak_allocated_bytes")) is int
        and value["peak_allocated_bytes"] >= 0
        and type(value.get("peak_reserved_bytes")) is int
        and value["peak_reserved_bytes"] >= value["peak_allocated_bytes"]
    )


def _runtime_within_derived_hard_budget(
    runtime: Any, resource_budget: Any,
) -> bool:
    if type(runtime) is not dict or type(resource_budget) is not dict:
        return False
    consumed = runtime.get("gpu_hours")
    maximum = resource_budget.get("maximum_gpu_hours")
    return bool(
        type(consumed) in (int, float)
        and type(consumed) is not bool
        and math.isfinite(float(consumed))
        and float(consumed) > 0.0
        and type(maximum) in (int, float)
        and type(maximum) is not bool
        and math.isfinite(float(maximum))
        and float(maximum) > 0.0
        and float(consumed) <= float(maximum)
    )


def validate_result_directory(path: str | Path) -> Json:
    root = Path(path).expanduser().resolve()
    result_path = root / core.RESULT_FILENAME
    if not result_path.is_file() or result_path.is_symlink():
        raise AnalysisError(f"result is absent or linked: {result_path}")
    result = _load_json(result_path)
    if (
        result.get("schema_version") != core.RESULT_SCHEMA
        or result.get("campaign_id") != core.CAMPAIGN_ID
        or result.get("status") != "completed"
        or result.get("schedule_manifest_sha256")
        != core.schedule_manifest()["sha256"]
    ):
        raise AnalysisError("result identity or completion changed")
    phase = result.get("phase")
    if phase == "prefix":
        arm_id = None
        rows = core.CONSTANT_SOURCE_UPDATES
        expected_evaluations = _expected_prefix_evaluations()
        expected_counts = {
            "updates": core.CONSTANT_SOURCE_UPDATES,
            "contexts": core.PREFIX_CONTEXTS,
            "tokens": core.PREFIX_TOKENS,
            "evaluation_rows": len(expected_evaluations["macro_index"]),
        }
        if result.get("contains_prediction_response") is not False:
            raise AnalysisError("prefix response classification changed")
    elif phase == "tail":
        arm_id = result.get("arm_id")
        if arm_id not in core.ARM_IDS:
            raise AnalysisError("tail arm identity changed")
        rows = int(core._arm_plan(str(arm_id))["tail_updates"])
        expected_evaluations = _expected_tail_evaluations(str(arm_id))
        plan = core._arm_plan(str(arm_id))
        expected_counts = {
            "prefix_updates": core.CONSTANT_SOURCE_UPDATES,
            "tail_updates": rows,
            "total_updates": int(plan["total_updates"]),
            "prefix_tokens": core.PREFIX_TOKENS,
            "tail_tokens": core.TAIL_TOKENS,
            "total_tokens": core.EXECUTED_TOKENS_PER_TRAJECTORY,
            "tail_evaluation_rows_including_prefix_anchor": len(
                expected_evaluations["macro_index"]
            ),
        }
        if result.get("contains_prediction_response") is not True:
            raise AnalysisError("tail response classification changed")
        if (
            result.get("shape_id") != plan["shape_id"]
            or result.get("realization_id") != plan["realization_id"]
            or result.get("terminal_intrinsic_time")
            != plan["terminal_intrinsic_time"]
        ):
            raise AnalysisError("tail result schedule identity changed")
    else:
        raise AnalysisError("only completed prefix/tail results are campaign inputs")
    try:
        core._validate_resource_contract(
            {
                "runtime": {
                    "formal_wall_time_and_gpu_hour_budget": result.get(
                        "resource_budget"
                    )
                },
                "resource_calibration": result.get("resource_calibration"),
            },
            str(phase),
            arm_id,
            str(result.get("data_identity_sha256")),
        )
    except core.CampaignError as error:
        raise AnalysisError(f"resource calibration changed: {error}") from error
    artifacts = result.get("artifacts")
    if type(artifacts) is not dict:
        raise AnalysisError("result artifact inventory is absent")
    trace_path = _artifact_path(
        root, artifacts.get("training_trace"), core.TRACE_FILENAME
    )
    evaluation_path = _artifact_path(
        root,
        artifacts.get("validation_evaluations"),
        core.EVALUATION_FILENAME,
    )
    trace = _load_trace(trace_path, phase, arm_id, rows)
    evaluations = _load_evaluations(evaluation_path, expected_evaluations)
    checkpoint_sha = None
    checkpoint_state_sha = None
    validation_anchor_sha = core._array_sha256(
        evaluations["context_cross_entropy"][-1 if phase == "prefix" else 0]
    )
    if phase == "prefix":
        checkpoint_path = _artifact_path(
            root,
            artifacts.get("checkpoint"),
            core.PREFIX_CHECKPOINT_FILENAME,
        )
        checkpoint_sha = core._sha256_file(checkpoint_path)
        checkpoint = core._load_prefix_checkpoint(
            checkpoint_path,
            str(result["data_identity_sha256"]),
            str(result["resource_calibration"]["initialization_sha256"]),
        )
        checkpoint_state_sha = checkpoint["model_state_sha256"]
        if checkpoint_state_sha != result.get("checkpoint_model_state_sha256"):
            raise AnalysisError("prefix result checkpoint state binding changed")
        checkpoint_anchor = np.ascontiguousarray(
            checkpoint["final_validation_context_cross_entropy"].numpy(),
            dtype=np.float32,
        )
        if core._array_sha256(checkpoint_anchor) != validation_anchor_sha:
            raise AnalysisError("prefix checkpoint/evaluation anchor changed")
    counts = result.get("counts")
    gates = {
        "result_identity": True,
        "artifact_hashes": True,
        "exact_counts": counts == expected_counts,
        "complete_per_update_training_CE": len(trace) == rows,
        "complete_fixed_probe_grid": len(evaluations["macro_index"])
        == len(expected_evaluations["macro_index"]),
        "exact_H200_FP32_math_SDPA_runtime": _runtime_is_exact_h200_fp32(
            result.get("runtime")
        ),
        "runtime_within_derived_hard_budget": (
            _runtime_within_derived_hard_budget(
                result.get("runtime"), result.get("resource_budget")
            )
        ),
        "all_values_finite": bool(
            np.all(np.isfinite(trace))
            and np.all(np.isfinite(evaluations["context_cross_entropy"]))
        ),
    }
    return {
        "schema_version": ANALYSIS_SCHEMA,
        "campaign_id": core.CAMPAIGN_ID,
        "phase": phase,
        "arm_id": arm_id,
        "status": "passed" if all(gates.values()) else "failed",
        "gates": gates,
        "data_identity_sha256": result.get("data_identity_sha256"),
        "schedule_manifest_sha256": result.get("schedule_manifest_sha256"),
        "resource_calibration_sha256": core._canonical_sha256(
            result["resource_calibration"]
        ),
        "prefix_checkpoint_sha256": (
            checkpoint_sha
            if phase == "prefix"
            else result.get("prefix_checkpoint_sha256")
        ),
        "prefix_model_state_sha256": (
            checkpoint_state_sha
            if phase == "prefix"
            else result.get("prefix_model_state_sha256")
        ),
        "prefix_validation_anchor_sha256": validation_anchor_sha,
        "training_rows": len(trace),
        "validation_rows": len(evaluations["macro_index"]),
        "terminal_intrinsic_time": float(evaluations["intrinsic_time"][-1]),
        "final_validation_cross_entropy": float(
            evaluations["validation_cross_entropy"][-1]
        ),
    }


def validate_campaign(prefix: str | Path, tails: Sequence[str | Path]) -> Json:
    prefix_report = validate_result_directory(prefix)
    if prefix_report["phase"] != "prefix":
        raise AnalysisError("campaign prefix input is not a prefix result")
    tail_reports = [validate_result_directory(path) for path in tails]
    identities = [str(report["arm_id"]) for report in tail_reports]
    if sorted(identities) != sorted(core.ARM_IDS):
        raise AnalysisError("campaign requires exactly four distinct tail arms")
    linkage_fields = (
        "data_identity_sha256",
        "schedule_manifest_sha256",
        "resource_calibration_sha256",
        "prefix_checkpoint_sha256",
        "prefix_model_state_sha256",
        "prefix_validation_anchor_sha256",
    )
    linkage = {
        field: len({report[field] for report in [prefix_report, *tail_reports]}) == 1
        for field in linkage_fields
    }
    paired = {}
    for shape_id in core.SHAPE_IDS:
        selected = [
            report
            for report in tail_reports
            if str(report["arm_id"]).startswith(shape_id + "__")
        ]
        paired[shape_id] = bool(
            len(selected) == 2
            and selected[0]["validation_rows"] == selected[1]["validation_rows"]
            and math.isclose(
                float(selected[0]["terminal_intrinsic_time"]),
                float(selected[1]["terminal_intrinsic_time"]),
                rel_tol=0.0,
                abs_tol=1e-12,
            )
        )
    gates = {
        "prefix_valid": prefix_report["status"] == "passed",
        "four_distinct_tails_valid": all(
            report["status"] == "passed" for report in tail_reports
        ),
        "shared_prefix_and_data_lineage": all(linkage.values()),
        "same_shape_validation_T_grids": all(paired.values()),
        "single_seed_scope_retained": True,
    }
    return {
        "schema_version": CAMPAIGN_SCHEMA,
        "campaign_id": core.CAMPAIGN_ID,
        "status": "passed" if all(gates.values()) else "failed",
        "decision_scope": "single_seed_A2_EXTERNAL_ratio_path_factorization",
        "gates": gates,
        "linkage": linkage,
        "pairing": paired,
        "prefix": prefix_report,
        "tails": sorted(tail_reports, key=lambda row: str(row["arm_id"])),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prefix", type=Path, required=True)
    parser.add_argument("--tail", type=Path, action="append", required=True)
    args = parser.parse_args(argv)
    result = validate_campaign(args.prefix, args.tail)
    print(json.dumps(result, sort_keys=True, ensure_ascii=False, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
