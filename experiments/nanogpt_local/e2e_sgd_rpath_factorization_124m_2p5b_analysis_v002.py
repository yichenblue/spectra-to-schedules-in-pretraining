"""Offline validator/collector for the 124M BF16 ratio-path campaign v002.

The validator is intentionally version-strict.  It accepts only artifacts
produced under the v002 BF16-autocast, TF32, Flash-SDPA-only campaign
contract.  In particular, the completed v001 FP32/math-SDPA campaign is not
an input to this analysis and cannot be silently mixed with v002 results.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from . import e2e_sgd_rpath_factorization_124m_2p5b_v002 as core


ANALYSIS_SCHEMA = "nanogpt124m_e2e_sgd_rpath_factorization_2p5b_analysis_v002"
CAMPAIGN_SCHEMA = (
    "nanogpt124m_e2e_sgd_rpath_factorization_2p5b_campaign_validation_v002"
)
LEGACY_CAMPAIGN_ID = "nanogpt124m-e2e-sgd-rpath-factorization-2p5b-v001"
LEGACY_RESULT_SCHEMA = (
    "nanogpt124m_e2e_sgd_rpath_factorization_2p5b_result_v001"
)
Json = dict[str, Any]


class AnalysisError(RuntimeError):
    """A retained v002 artifact is incomplete, malformed, or inconsistent."""


def _load_json(path: Path) -> Json:
    with path.open("r", encoding="utf-8") as stream:
        value = json.load(stream)
    if type(value) is not dict:
        raise AnalysisError(f"JSON root must be an object: {path}")
    return value


def _is_hex64(value: Any) -> bool:
    return bool(
        type(value) is str
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _scalar_text(value: np.ndarray, label: str) -> str:
    if value.shape != () or value.dtype.kind not in {"U", "S"}:
        raise AnalysisError(f"{label} must be a scalar string")
    return str(value.item())


def _artifact_path(root: Path, record: Any, expected_name: str) -> Path:
    if (
        type(record) is not dict
        or set(record) != {"path", "sha256", "size_bytes"}
        or record.get("path") != expected_name
        or not _is_hex64(record.get("sha256"))
        or type(record.get("size_bytes")) is not int
        or record["size_bytes"] <= 0
    ):
        raise AnalysisError(f"artifact record changed: {expected_name}")
    path = root / expected_name
    if (
        not path.is_file()
        or path.is_symlink()
        or path.stat().st_size != record["size_bytes"]
        or core._sha256_file(path) != record["sha256"]
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


def _runtime_is_exact_h200_bf16_tf32_flash_only(value: Any) -> bool:
    if type(value) is not dict or not core._runtime_record_matches_contract(value):
        return False
    numeric = (
        value.get("elapsed_seconds"),
        value.get("gpu_hours"),
        value.get("sampled_step_time_median_seconds"),
        value.get("sampled_step_time_p90_seconds"),
    )
    if any(
        type(item) not in (int, float)
        or type(item) is bool
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
        and type(value.get("torch_version")) is str
        and type(value.get("cuda_version")) is str
        and type(value.get("python_version")) is str
        and math.isclose(
            float(value["gpu_hours"]),
            elapsed / 3_600.0,
            rel_tol=1e-12,
            abs_tol=0.0,
        )
        and type(value.get("peak_allocated_bytes")) is int
        and type(value.get("peak_allocated_bytes")) is not bool
        and value["peak_allocated_bytes"] >= 0
        and type(value.get("peak_reserved_bytes")) is int
        and type(value.get("peak_reserved_bytes")) is not bool
        and value["peak_reserved_bytes"] >= value["peak_allocated_bytes"]
    )


def _runtime_within_derived_hard_budget(runtime: Any, resource_budget: Any) -> bool:
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


def _validate_anchor_reload(value: Any, stored_anchor_sha256: str) -> Json:
    keys = {
        "comparison",
        "bitwise_equal_required",
        "stored_anchor_sha256",
        "reloaded_anchor_sha256",
        "max_abs_error",
        "mean_abs_error",
        "max_abs_tolerance",
        "mean_abs_tolerance",
        "passed",
    }
    if type(value) is not dict or set(value) != keys:
        raise AnalysisError("tail prefix-anchor reload diagnostic is malformed")
    maximum = value.get("max_abs_error")
    mean = value.get("mean_abs_error")
    if (
        value.get("comparison")
        != "stored_prefix_anchor_vs_reloaded_model_evaluation"
        or value.get("bitwise_equal_required") is not False
        or value.get("stored_anchor_sha256") != stored_anchor_sha256
        or not _is_hex64(value.get("reloaded_anchor_sha256"))
        or type(maximum) not in (int, float)
        or type(maximum) is bool
        or not math.isfinite(float(maximum))
        or float(maximum) < 0.0
        or type(mean) not in (int, float)
        or type(mean) is bool
        or not math.isfinite(float(mean))
        or float(mean) < 0.0
        or float(mean) > float(maximum)
        or value.get("max_abs_tolerance")
        != core.ANCHOR_RELOAD_POLICY["max_abs_tolerance"]
        or value.get("mean_abs_tolerance")
        != core.ANCHOR_RELOAD_POLICY["mean_abs_tolerance"]
    ):
        raise AnalysisError("tail prefix-anchor reload diagnostic changed")
    expected_pass = bool(
        float(maximum) <= float(core.ANCHOR_RELOAD_POLICY["max_abs_tolerance"])
        and float(mean)
        <= float(core.ANCHOR_RELOAD_POLICY["mean_abs_tolerance"])
    )
    same_hash = value["stored_anchor_sha256"] == value["reloaded_anchor_sha256"]
    zero_error = float(maximum) == 0.0 and float(mean) == 0.0
    if value.get("passed") is not expected_pass or not expected_pass:
        raise AnalysisError("tail prefix-anchor reload tolerance did not pass")
    if same_hash != zero_error:
        raise AnalysisError("tail prefix-anchor reload hash/error report is inconsistent")
    return dict(value)


def _reject_legacy_result(result: Mapping[str, Any]) -> None:
    if (
        result.get("campaign_id") == LEGACY_CAMPAIGN_ID
        or result.get("schema_version") == LEGACY_RESULT_SCHEMA
    ):
        raise AnalysisError(
            "v001 FP32/math-SDPA results are incompatible with the v002 "
            "BF16/TF32/Flash-only analysis"
        )


def validate_result_directory(path: str | Path) -> Json:
    root = Path(path).expanduser().resolve()
    result_path = root / core.RESULT_FILENAME
    if not result_path.is_file() or result_path.is_symlink():
        raise AnalysisError(f"result is absent or linked: {result_path}")
    result = _load_json(result_path)
    _reject_legacy_result(result)
    if (
        result.get("schema_version") != core.RESULT_SCHEMA
        or result.get("campaign_id") != core.CAMPAIGN_ID
        or result.get("campaign_contract_sha256")
        != core.campaign_contract_sha256()
        or result.get("numerics_id") != core.NUMERICS_ID
        or result.get("status") != "completed"
        or result.get("schedule_manifest_sha256")
        != core.schedule_manifest()["sha256"]
        or not _is_hex64(result.get("data_identity_sha256"))
    ):
        raise AnalysisError("v002 result identity, contract, or completion changed")

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
        expected_artifacts = {
            "checkpoint",
            "training_trace",
            "validation_evaluations",
        }
        if result.get("contains_prediction_response") is not False:
            raise AnalysisError("prefix response classification changed")
        if "prefix_validation_anchor_reload" in result:
            raise AnalysisError("prefix result cannot contain a tail reload diagnostic")
    elif phase == "tail":
        arm_id = result.get("arm_id")
        if arm_id not in core.ARM_IDS:
            raise AnalysisError("tail arm identity changed")
        plan = core._arm_plan(str(arm_id))
        rows = int(plan["tail_updates"])
        expected_evaluations = _expected_tail_evaluations(str(arm_id))
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
        expected_artifacts = {"training_trace", "validation_evaluations"}
        if result.get("contains_prediction_response") is not True:
            raise AnalysisError("tail response classification changed")
        if (
            result.get("shape_id") != plan["shape_id"]
            or result.get("realization_id") != plan["realization_id"]
            or result.get("terminal_intrinsic_time")
            != plan["terminal_intrinsic_time"]
            or not _is_hex64(result.get("prefix_checkpoint_sha256"))
            or not _is_hex64(result.get("prefix_model_state_sha256"))
        ):
            raise AnalysisError("tail result schedule or prefix identity changed")
    else:
        raise AnalysisError("only completed v002 prefix/tail results are inputs")

    completion_gates = result.get("completion_gates")

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
            str(result["data_identity_sha256"]),
        )
    except core.CampaignError as error:
        raise AnalysisError(f"resource calibration changed: {error}") from error

    artifacts = result.get("artifacts")
    if type(artifacts) is not dict or set(artifacts) != expected_artifacts:
        raise AnalysisError("result artifact inventory changed")
    trace_path = _artifact_path(
        root, artifacts.get("training_trace"), core.TRACE_FILENAME
    )
    evaluation_path = _artifact_path(
        root,
        artifacts.get("validation_evaluations"),
        core.EVALUATION_FILENAME,
    )
    trace = _load_trace(trace_path, str(phase), arm_id, rows)
    evaluations = _load_evaluations(evaluation_path, expected_evaluations)
    validation_anchor_sha256 = core._array_sha256(
        evaluations["context_cross_entropy"][-1 if phase == "prefix" else 0]
    )

    checkpoint_sha256 = None
    checkpoint_state_sha256 = None
    anchor_reload = None
    if phase == "prefix":
        checkpoint_path = _artifact_path(
            root,
            artifacts.get("checkpoint"),
            core.PREFIX_CHECKPOINT_FILENAME,
        )
        checkpoint_sha256 = core._sha256_file(checkpoint_path)
        try:
            checkpoint = core._load_prefix_checkpoint(
                checkpoint_path,
                str(result["data_identity_sha256"]),
                str(result["resource_calibration"]["initialization_sha256"]),
            )
        except core.CampaignError as error:
            raise AnalysisError(f"v002 prefix checkpoint changed: {error}") from error
        checkpoint_state_sha256 = checkpoint["model_state_sha256"]
        if checkpoint_state_sha256 != result.get("checkpoint_model_state_sha256"):
            raise AnalysisError("prefix result checkpoint state binding changed")
        checkpoint_anchor = np.ascontiguousarray(
            checkpoint["final_validation_context_cross_entropy"].numpy(),
            dtype=np.float32,
        )
        if core._array_sha256(checkpoint_anchor) != validation_anchor_sha256:
            raise AnalysisError("prefix checkpoint/evaluation anchor changed")
    else:
        anchor_reload = _validate_anchor_reload(
            result.get("prefix_validation_anchor_reload"),
            validation_anchor_sha256,
        )

    gates = {
        "v002_result_identity_and_campaign_contract": True,
        "artifact_hashes_and_schemas": True,
        "exact_counts": result.get("counts") == expected_counts,
        "complete_per_update_training_CE": len(trace) == rows,
        "complete_fixed_probe_grid": len(evaluations["macro_index"])
        == len(expected_evaluations["macro_index"]),
        "exact_H200_BF16_TF32_flash_only_runtime": (
            _runtime_is_exact_h200_bf16_tf32_flash_only(result.get("runtime"))
        ),
        "runtime_within_derived_hard_budget": _runtime_within_derived_hard_budget(
            result.get("runtime"), result.get("resource_budget")
        ),
        "all_values_finite": bool(
            np.all(np.isfinite(trace))
            and np.all(np.isfinite(evaluations["context_cross_entropy"]))
        ),
        "final_model_and_gradient_contract": (
            completion_gates == core._expected_completion_gates()
        ),
        "prefix_anchor_contract": bool(
            phase == "prefix" or (anchor_reload is not None and anchor_reload["passed"])
        ),
    }
    return {
        "schema_version": ANALYSIS_SCHEMA,
        "campaign_id": core.CAMPAIGN_ID,
        "campaign_contract_sha256": core.campaign_contract_sha256(),
        "numerics_id": core.NUMERICS_ID,
        "phase": phase,
        "arm_id": arm_id,
        "status": "passed" if all(gates.values()) else "failed",
        "gates": gates,
        "data_identity_sha256": result["data_identity_sha256"],
        "schedule_manifest_sha256": result["schedule_manifest_sha256"],
        "resource_calibration_sha256": core._canonical_sha256(
            result["resource_calibration"]
        ),
        "prefix_checkpoint_sha256": (
            checkpoint_sha256
            if phase == "prefix"
            else result["prefix_checkpoint_sha256"]
        ),
        "prefix_model_state_sha256": (
            checkpoint_state_sha256
            if phase == "prefix"
            else result["prefix_model_state_sha256"]
        ),
        "prefix_validation_anchor_sha256": validation_anchor_sha256,
        "prefix_validation_anchor_reload": anchor_reload,
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
        raise AnalysisError("campaign prefix input is not a v002 prefix result")
    tail_reports = [validate_result_directory(path) for path in tails]
    identities = [str(report["arm_id"]) for report in tail_reports]
    if sorted(identities) != sorted(core.ARM_IDS):
        raise AnalysisError("campaign requires exactly four distinct v002 tail arms")
    linkage_fields = (
        "campaign_contract_sha256",
        "numerics_id",
        "data_identity_sha256",
        "schedule_manifest_sha256",
        "resource_calibration_sha256",
        "prefix_checkpoint_sha256",
        "prefix_model_state_sha256",
        "prefix_validation_anchor_sha256",
    )
    reports = [prefix_report, *tail_reports]
    linkage = {
        field: len({report[field] for report in reports}) == 1
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
        "shared_v002_prefix_data_and_numerics_lineage": all(linkage.values()),
        "same_shape_validation_T_grids": all(paired.values()),
        "single_seed_scope_retained": True,
    }
    return {
        "schema_version": CAMPAIGN_SCHEMA,
        "campaign_id": core.CAMPAIGN_ID,
        "campaign_contract_sha256": core.campaign_contract_sha256(),
        "numerics_id": core.NUMERICS_ID,
        "status": "passed" if all(gates.values()) else "failed",
        "decision_scope": "single_seed_A2_EXTERNAL_ratio_path_factorization_v002",
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
    return 0 if result.get("status") == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
