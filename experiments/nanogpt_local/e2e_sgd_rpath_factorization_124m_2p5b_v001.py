"""124M plain-SGD WSD/8-1-1 ratio-path factorization campaign.

This module is the small production core for the long 124M experiment.  It
has three deliberately separate responsibilities:

* derive the exact integer schedule and fixed validation grid;
* train the one shared constant-r prefix and publish its checkpoint; and
* resume one of four tails from that checkpoint.

It contains no cluster, upload, retry, or submission code.  The checked-in
campaign package remains launch-false until a separate user-authorized turn.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
from functools import lru_cache
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import statistics
import time
from typing import Any, Iterator, Mapping, Sequence

import numpy as np
import torch
from torch.nn import functional as F

from .model import GPT, GPTConfig, _strip_compile_prefix


CAMPAIGN_ID = "nanogpt124m-e2e-sgd-rpath-factorization-2p5b-v001"
CONFIG_SCHEMA = "nanogpt124m_e2e_sgd_rpath_factorization_2p5b_config_v001"
SCHEDULE_SCHEMA = "nanogpt124m_e2e_sgd_rpath_factorization_2p5b_schedule_v001"
CHECKPOINT_SCHEMA = "nanogpt124m_e2e_sgd_rpath_factorization_2p5b_prefix_checkpoint_v001"
TRACE_SCHEMA = "nanogpt124m_e2e_sgd_rpath_factorization_2p5b_trace_v001"
EVALUATION_SCHEMA = "nanogpt124m_e2e_sgd_rpath_factorization_2p5b_evaluations_v001"
RESULT_SCHEMA = "nanogpt124m_e2e_sgd_rpath_factorization_2p5b_result_v001"
PREFLIGHT_SCHEMA = "nanogpt124m_e2e_sgd_rpath_factorization_2p5b_preflight_v001"
RESOURCE_CALIBRATION_SCHEMA = (
    "nanogpt124m_e2e_sgd_rpath_factorization_2p5b_H200_calibration_v001"
)
RESOURCE_BUDGET_SCHEMA = (
    "nanogpt124m_e2e_sgd_rpath_factorization_2p5b_resource_budget_v001"
)
SOURCE_MANIFEST_SCHEMA = "openwebtext_cooldown_2p5b_source_manifest_v001"
PRIMARY_PREDICTION_ID = "A2_EXTERNAL"

SHAPE_IDS = ("wsd_exp_80_20", "eight_one_one")
REALIZATION_IDS = ("fixed_batch_lr", "fixed_lr_batch")
ARM_IDS = tuple(
    f"{shape_id}__{realization_id}"
    for shape_id in SHAPE_IDS
    for realization_id in REALIZATION_IDS
)

BLOCK_SIZE = 256
VOCAB_SIZE = 50_304
MODEL_SPEC = {
    "family": "nanogpt",
    "block_size": BLOCK_SIZE,
    "vocab_size": VOCAB_SIZE,
    "n_layer": 12,
    "n_head": 12,
    "n_embd": 768,
    "dropout": 0.0,
    "bias": False,
    "expected_parameter_count": 123_783_936,
}

# 2.5B / 1024 is not an integer.  The nearest lower source horizon that is
# divisible by twenty makes both the 80/20 split and the four-update macro
# exact.  The 6,400-token difference from 2.5B is declared, not hidden.
NOMINAL_TOKENS_PER_TRAJECTORY = 2_500_000_000
CANONICAL_UPDATES = 2_441_400
ATOMIC_BATCH_SIZE = 4
EXECUTED_TOKENS_PER_TRAJECTORY = CANONICAL_UPDATES * ATOMIC_BATCH_SIZE * BLOCK_SIZE
TOKEN_SHORTFALL_FROM_NOMINAL = (
    NOMINAL_TOKENS_PER_TRAJECTORY - EXECUTED_TOKENS_PER_TRAJECTORY
)
CONSTANT_SOURCE_UPDATES = 4 * CANONICAL_UPDATES // 5
TAIL_SOURCE_UPDATES = CANONICAL_UPDATES - CONSTANT_SOURCE_UPDATES
TAPE_CONTEXTS = CANONICAL_UPDATES * ATOMIC_BATCH_SIZE
PREFIX_CONTEXTS = CONSTANT_SOURCE_UPDATES * ATOMIC_BATCH_SIZE
TAIL_CONTEXTS = TAPE_CONTEXTS - PREFIX_CONTEXTS
PREFIX_TOKENS = PREFIX_CONTEXTS * BLOCK_SIZE
TAIL_TOKENS = TAIL_CONTEXTS * BLOCK_SIZE

CANONICAL_BASE_LEARNING_RATE = 0.005
MACRO_INTRINSIC_TIME = 0.02
CONSTANT_MACROS = CONSTANT_SOURCE_UPDATES // 4
PREFIX_INTRINSIC_TIME = CONSTANT_SOURCE_UPDATES * CANONICAL_BASE_LEARNING_RATE
TARGET_INITIAL_RATIO = ATOMIC_BATCH_SIZE / CANONICAL_BASE_LEARNING_RATE
EXECUTED_RATIO_UNIT = ATOMIC_BATCH_SIZE / MACRO_INTRINSIC_TIME

# The corpus stores one extra aligned block so the nominal 2.5B-token domain
# has an explicit next-token lookahead at its right boundary.
CORPUS_TOKENS = 2_500_000_256
ALIGNED_STRIDE = BLOCK_SIZE
INITIALIZATION_SEED = 2_026_090_803
TRAINING_TAPE_SEED = 2_026_090_804
TAPE_OFFSETS_SHA256 = (
    "970c2fd6f126ceb7c2ce981cb785113ebf271e6316d21dd4bd5abe55d87bedac"
)
PREFIX_TAPE_SHA256 = (
    "fb278b5f502ea6033daa036186807e95783603655ad1d00f423e57dbdc6f4809"
)
TAIL_TAPE_SHA256 = (
    "2369fe8d7881dba094d03ccce1fb2a30d220647bb1c3ad71a431667f2dddd96b"
)
# PyTorch's seeded CPU normal initializer is reproducible within the pinned
# Linux container but its exact bytes are not a cross-architecture contract.
# B0 therefore measures the initialization twice in that container and freezes
# the common digest into the downstream resource calibration.

VALIDATION_CONTEXTS = 1_024
VALIDATION_BATCH_SIZE = 32
TRACE_FILENAME = "training_trace.npz"
EVALUATION_FILENAME = "validation_evaluations.npz"
PREFIX_CHECKPOINT_FILENAME = "prefix_checkpoint.pt"
RESULT_FILENAME = "result.json"
B0_BASE_UPDATES = 512
B0_MAX_BATCH_UPDATES = 64
B0_MAXIMUM_GPU_HOURS = 0.25
RESOURCE_FIXED_OVERHEAD_SECONDS = 600.0
RESOURCE_HARD_LIMIT_MULTIPLIER = 1.5

Json = dict[str, Any]


class CampaignError(RuntimeError):
    """A frozen campaign contract or numerical invariant failed."""


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


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _array_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    return hashlib.sha256(array.tobytes(order="C")).hexdigest()


def _model_config() -> GPTConfig:
    return GPTConfig(
        block_size=BLOCK_SIZE,
        vocab_size=VOCAB_SIZE,
        n_layer=12,
        n_head=12,
        n_embd=768,
        dropout=0.0,
        bias=False,
    )


def _analytic_parameter_count() -> int:
    width = int(MODEL_SPEC["n_embd"])
    layers = int(MODEL_SPEC["n_layer"])
    return (
        VOCAB_SIZE * width
        + BLOCK_SIZE * width
        + layers * (12 * width * width + 2 * width)
        + width
    )


def _raw_fractions(shape_id: str) -> np.ndarray:
    if shape_id not in SHAPE_IDS:
        raise CampaignError(f"unknown schedule shape: {shape_id!r}")
    indices = np.arange(CANONICAL_UPDATES, dtype=np.float64)
    coordinates = indices / float(CANONICAL_UPDATES - 1)
    result = np.ones(CANONICAL_UPDATES, dtype=np.float64)
    tail = coordinates >= 0.8
    if shape_id == "wsd_exp_80_20":
        result[tail] = np.exp(
            -math.log(10.0) * (coordinates[tail] - 0.8) / 0.2
        )
    else:
        result[tail & (coordinates < 0.9)] = 1.0 / math.sqrt(10.0)
        result[coordinates >= 0.9] = 0.1
    if not np.array_equal(
        result[:CONSTANT_SOURCE_UPDATES],
        np.ones(CONSTANT_SOURCE_UPDATES, dtype=np.float64),
    ):
        raise CampaignError("the exact 80% constant-r prefix changed")
    return result


@lru_cache(maxsize=None)
def _schedule_plan(shape_id: str) -> Json:
    raw = _raw_fractions(shape_id)
    raw_terminal_time = CANONICAL_BASE_LEARNING_RATE * float(np.sum(raw))
    # Use the first complete dT macro at or above the continuous source
    # clock.  Besides making the endpoint error one-sided and < dT, this
    # keeps the inverse-clock integerization within the declared B<=40 cap
    # at the discontinuous final 8-1-1 stage.
    macro_count = int(math.ceil(raw_terminal_time / MACRO_INTRINSIC_TIME))
    terminal_time = macro_count * MACRO_INTRINSIC_TIME
    tail_scale = (terminal_time - PREFIX_INTRINSIC_TIME) / (
        CANONICAL_BASE_LEARNING_RATE
        * float(np.sum(raw[CONSTANT_SOURCE_UPDATES:]))
    )
    fractions = raw
    fractions[CONSTANT_SOURCE_UPDATES:] *= tail_scale
    rates = fractions * CANONICAL_BASE_LEARNING_RATE
    boundaries = np.empty(CANONICAL_UPDATES + 1, dtype=np.float64)
    boundaries[0] = 0.0
    np.cumsum(rates, dtype=np.float64, out=boundaries[1:])
    boundaries[-1] = terminal_time
    targets = np.arange(macro_count + 1, dtype=np.float64) * MACRO_INTRINSIC_TIME
    cumulative = np.floor(
        np.interp(
            targets,
            boundaries,
            np.arange(CANONICAL_UPDATES + 1, dtype=np.float64),
        )
        + 0.5
    ).astype(np.int64)
    cumulative[: CONSTANT_MACROS + 1] = (
        np.arange(CONSTANT_MACROS + 1, dtype=np.int64) * 4
    )
    cumulative[0] = 0
    cumulative[-1] = CANONICAL_UPDATES
    g = np.ascontiguousarray(np.diff(cumulative), dtype=np.int16)
    if (
        g.shape != (macro_count,)
        or int(np.min(g)) < 1
        or int(np.max(g)) > 40
        or int(np.sum(g, dtype=np.int64)) != CANONICAL_UPDATES
        or not np.all(g[:CONSTANT_MACROS] == 4)
    ):
        raise CampaignError("balanced integer schedule invariants failed")
    changed = np.flatnonzero(g != 4)
    source_knots = (
        (CONSTANT_SOURCE_UPDATES,)
        if shape_id == "wsd_exp_80_20"
        else (CONSTANT_SOURCE_UPDATES, 9 * CANONICAL_UPDATES // 10)
    )
    knot_macros = tuple(
        int(np.searchsorted(cumulative, knot)) for knot in source_knots
    )
    return {
        "shape_id": shape_id,
        "macro_count": macro_count,
        "terminal_intrinsic_time": terminal_time,
        "raw_terminal_intrinsic_time": raw_terminal_time,
        "tail_rescale": tail_scale,
        "g": g,
        "g_sha256": _array_sha256(g),
        "g_minimum": int(np.min(g)),
        "g_maximum": int(np.max(g)),
        "g_sum": int(np.sum(g, dtype=np.int64)),
        "source_knots": source_knots,
        "knot_macros": knot_macros,
        "first_nonconstant_ratio_macro": int(changed[0]),
    }


def _prefix_evaluation_macros() -> tuple[int, ...]:
    # Sparse over the expensive mature prefix.  The tail grid below is much
    # denser, as requested, and both factorizations use identical T ticks.
    ticks = {0, CONSTANT_MACROS}
    ticks.update(2**power for power in range(4, 14))
    ticks.update(range(16_384, CONSTANT_MACROS, 32_768))
    for offset in (-128, -64, -32, -16, -8, -4, -2, -1):
        ticks.add(CONSTANT_MACROS + offset)
    return tuple(sorted(value for value in ticks if 0 <= value <= CONSTANT_MACROS))


def _tail_evaluation_macros(shape_id: str) -> tuple[int, ...]:
    plan = _schedule_plan(shape_id)
    stop = int(plan["macro_count"])
    ticks = set(range(CONSTANT_MACROS, stop + 1, 256))
    ticks.add(stop)
    anchors = {
        CONSTANT_MACROS,
        int(plan["first_nonconstant_ratio_macro"]),
        *[int(value) for value in plan["knot_macros"]],
        stop,
    }
    # WSD runs to a larger terminal T.  Also sample it at every dense 8-1-1
    # anchor so that the shorter schedule's complete validation grid has an
    # exact same-T WSD counterpart, rather than an interpolated one.
    shorter = _schedule_plan("eight_one_one")
    anchors.update(
        {
            int(shorter["macro_count"]),
            *[int(value) for value in shorter["knot_macros"]],
        }
    )
    offsets = (-128, -64, -32, -16, -8, -4, -2, -1, 0, 1, 2, 4, 8, 16, 32, 64, 128)
    for anchor in anchors:
        for offset in offsets:
            value = anchor + offset
            if CONSTANT_MACROS <= value <= stop:
                ticks.add(value)
    return tuple(sorted(ticks))


def _arm_plan(arm_id: str) -> Json:
    if arm_id not in ARM_IDS:
        raise CampaignError(f"unknown arm: {arm_id!r}")
    shape_id, realization_id = arm_id.split("__", 1)
    schedule = _schedule_plan(shape_id)
    tail_g = schedule["g"][CONSTANT_MACROS:]
    tail_updates = (
        int(np.sum(tail_g, dtype=np.int64))
        if realization_id == "fixed_batch_lr"
        else 4 * len(tail_g)
    )
    return {
        **schedule,
        "arm_id": arm_id,
        "realization_id": realization_id,
        "prefix_updates": CONSTANT_SOURCE_UPDATES,
        "tail_updates": tail_updates,
        "total_updates": CONSTANT_SOURCE_UPDATES + tail_updates,
        "prefix_contexts": PREFIX_CONTEXTS,
        "tail_contexts": TAIL_CONTEXTS,
        "total_contexts": TAPE_CONTEXTS,
        "prefix_tokens": PREFIX_TOKENS,
        "tail_tokens": TAIL_TOKENS,
        "total_tokens": EXECUTED_TOKENS_PER_TRAJECTORY,
        "evaluation_macros": _tail_evaluation_macros(shape_id),
    }


def _iter_tail_steps(arm_id: str) -> Iterator[Json]:
    plan = _arm_plan(arm_id)
    fixed_batch = plan["realization_id"] == "fixed_batch_lr"
    cursor = PREFIX_CONTEXTS
    tail_update = 0
    global_update = CONSTANT_SOURCE_UPDATES
    intrinsic = PREFIX_INTRINSIC_TIME
    g_values = plan["g"]
    for macro_index in range(CONSTANT_MACROS, int(plan["macro_count"])):
        g = int(g_values[macro_index])
        count = g if fixed_batch else 4
        for within_macro in range(count):
            batch_size = ATOMIC_BATCH_SIZE if fixed_batch else g
            learning_rate = (
                MACRO_INTRINSIC_TIME / g
                if fixed_batch
                else CANONICAL_BASE_LEARNING_RATE
            )
            before = intrinsic
            tail_update += 1
            global_update += 1
            start = cursor
            cursor += batch_size
            complete = within_macro == count - 1
            intrinsic = (
                (macro_index + 1) * MACRO_INTRINSIC_TIME
                if complete
                else intrinsic + learning_rate
            )
            yield {
                "tail_update": tail_update,
                "global_update": global_update,
                "macro_index": macro_index,
                "within_macro_update": within_macro,
                "g": g,
                "batch_size": batch_size,
                "learning_rate": learning_rate,
                "ratio_B_over_eta": EXECUTED_RATIO_UNIT * g,
                "intrinsic_time_before": before,
                "intrinsic_time_after": intrinsic,
                "tape_context_start": start,
                "tape_context_stop": cursor,
                "macro_completed": complete,
            }


def _eligible_aligned_starts(token_count: int) -> int:
    if token_count < BLOCK_SIZE + 1:
        return 0
    return (int(token_count) - (BLOCK_SIZE + 1)) // ALIGNED_STRIDE + 1


def make_training_tape(token_count: int = CORPUS_TOKENS) -> np.ndarray:
    eligible = _eligible_aligned_starts(token_count)
    if eligible < TAPE_CONTEXTS:
        raise CampaignError(
            f"corpus exposes {eligible} aligned contexts, need {TAPE_CONTEXTS}"
        )
    generator = np.random.Generator(np.random.PCG64(TRAINING_TAPE_SEED))
    tape = np.ascontiguousarray(
        generator.permutation(eligible)[:TAPE_CONTEXTS] * ALIGNED_STRIDE,
        dtype=np.int64,
    )
    if (
        _array_sha256(tape) != TAPE_OFFSETS_SHA256
        or _array_sha256(tape[:PREFIX_CONTEXTS]) != PREFIX_TAPE_SHA256
        or _array_sha256(tape[PREFIX_CONTEXTS:]) != TAIL_TAPE_SHA256
    ):
        raise CampaignError("frozen training tape hash changed")
    return tape


def schedule_manifest() -> Json:
    shapes: list[Json] = []
    for shape_id in SHAPE_IDS:
        plan = _schedule_plan(shape_id)
        arms = [_arm_plan(f"{shape_id}__{realization}") for realization in REALIZATION_IDS]
        shapes.append(
            {
                "shape_id": shape_id,
                "macro_count": plan["macro_count"],
                "constant_macros": CONSTANT_MACROS,
                "tail_macros": int(plan["macro_count"]) - CONSTANT_MACROS,
                "prefix_intrinsic_time": PREFIX_INTRINSIC_TIME,
                "terminal_intrinsic_time": plan["terminal_intrinsic_time"],
                "raw_terminal_intrinsic_time": plan["raw_terminal_intrinsic_time"],
                "tail_rescale": plan["tail_rescale"],
                "g_sha256": plan["g_sha256"],
                "g_minimum": plan["g_minimum"],
                "g_maximum": plan["g_maximum"],
                "g_sum": plan["g_sum"],
                "source_knots": list(plan["source_knots"]),
                "knot_macros": list(plan["knot_macros"]),
                "first_nonconstant_ratio_macro": plan[
                    "first_nonconstant_ratio_macro"
                ],
                "evaluation_macros": list(_tail_evaluation_macros(shape_id)),
                "arms": [
                    {
                        "arm_id": arm["arm_id"],
                        "realization_id": arm["realization_id"],
                        "prefix_updates": arm["prefix_updates"],
                        "tail_updates": arm["tail_updates"],
                        "total_updates": arm["total_updates"],
                        "tokens": arm["total_tokens"],
                    }
                    for arm in arms
                ],
            }
        )
    unsigned = {
        "schema_version": SCHEDULE_SCHEMA,
        "campaign_id": CAMPAIGN_ID,
        "model": MODEL_SPEC,
        "optimizer": {
            "name": "sgd",
            "momentum": 0.0,
            "weight_decay": 0.0,
            "gradient_clip": 0.0,
            "atomic_batch_size": ATOMIC_BATCH_SIZE,
            "canonical_base_learning_rate": CANONICAL_BASE_LEARNING_RATE,
            "initial_ratio_B_over_eta": TARGET_INITIAL_RATIO,
        },
        "accounting": {
            "nominal_tokens_per_trajectory": NOMINAL_TOKENS_PER_TRAJECTORY,
            "executed_tokens_per_trajectory": EXECUTED_TOKENS_PER_TRAJECTORY,
            "token_shortfall_from_nominal": TOKEN_SHORTFALL_FROM_NOMINAL,
            "canonical_updates": CANONICAL_UPDATES,
            "constant_source_updates": CONSTANT_SOURCE_UPDATES,
            "tail_source_updates": TAIL_SOURCE_UPDATES,
            "prefix_tokens": PREFIX_TOKENS,
            "tail_tokens": TAIL_TOKENS,
            "shared_prefix_executed_once": True,
            "total_executed_campaign_tokens": PREFIX_TOKENS + 4 * TAIL_TOKENS,
        },
        "clock": {
            "macro_intrinsic_time": MACRO_INTRINSIC_TIME,
            "prefix_intrinsic_time": PREFIX_INTRINSIC_TIME,
            "comparison_axis": "intrinsic_time_T",
            "equal_optimizer_updates_required": False,
        },
        "tape": {
            "algorithm": "PCG64(seed).permutation(eligible_aligned_starts)[:required]",
            "seed": TRAINING_TAPE_SEED,
            "corpus_tokens": CORPUS_TOKENS,
            "eligible_aligned_starts": _eligible_aligned_starts(CORPUS_TOKENS),
            "contexts": TAPE_CONTEXTS,
            "offsets_sha256": TAPE_OFFSETS_SHA256,
            "prefix_offsets_sha256": PREFIX_TAPE_SHA256,
            "tail_offsets_sha256": TAIL_TAPE_SHA256,
            "same_future_tape_for_all_four_tails": True,
        },
        "validation": {
            "fixed_probe_contexts": VALIDATION_CONTEXTS,
            "batch_size": VALIDATION_BATCH_SIZE,
            "primary_metric": "fixed_validation_probe_cross_entropy",
            "prefix_evaluation_macros": list(_prefix_evaluation_macros()),
            "tail_stride_macros": 256,
            "dense_at_schedule_knots": True,
        },
        "shapes": shapes,
        "evidence": {
            "primary_prediction_id": PRIMARY_PREDICTION_ID,
            "theorem_facing": False,
            "classification": "single_seed_external_validity",
        },
    }
    return {**unsigned, "sha256": _canonical_sha256(unsigned)}


def _verify_step_pairing(arm_id: str) -> Json:
    plan = _arm_plan(arm_id)
    steps = _iter_tail_steps(arm_id)
    updates = 0
    cursor = PREFIX_CONTEXTS
    final_time = PREFIX_INTRINSIC_TIME
    completed_macros = 0
    for step in steps:
        updates += 1
        if int(step["tape_context_start"]) != cursor:
            raise CampaignError("tail tape is not contiguous")
        cursor = int(step["tape_context_stop"])
        final_time = float(step["intrinsic_time_after"])
        completed_macros += int(bool(step["macro_completed"]))
    return {
        "arm_id": arm_id,
        "updates": updates,
        "contexts": cursor - PREFIX_CONTEXTS,
        "terminal_intrinsic_time": final_time,
        "completed_tail_macros": completed_macros,
        "passed": bool(
            updates == int(plan["tail_updates"])
            and cursor == TAPE_CONTEXTS
            and math.isclose(
                final_time,
                float(plan["terminal_intrinsic_time"]),
                rel_tol=0.0,
                abs_tol=1e-9,
            )
            and completed_macros == int(plan["macro_count"]) - CONSTANT_MACROS
        ),
    }


def preflight(*, verify_tape: bool = True) -> Json:
    manifest = schedule_manifest()
    arm_checks = [_verify_step_pairing(arm_id) for arm_id in ARM_IDS]
    tape_checks: Json = {
        "verified": False,
        "offsets_sha256": TAPE_OFFSETS_SHA256,
    }
    if verify_tape:
        tape = make_training_tape()
        tape_checks = {
            "verified": True,
            "shape": list(tape.shape),
            "dtype": tape.dtype.str,
            "offsets_sha256": _array_sha256(tape),
            "minimum_offset": int(np.min(tape)),
            "maximum_offset": int(np.max(tape)),
        }
    shape_pairing = {}
    for shape_id in SHAPE_IDS:
        left = _arm_plan(f"{shape_id}__fixed_batch_lr")
        right = _arm_plan(f"{shape_id}__fixed_lr_batch")
        shape_pairing[shape_id] = bool(
            left["g_sha256"] == right["g_sha256"]
            and left["evaluation_macros"] == right["evaluation_macros"]
            and left["tail_contexts"] == right["tail_contexts"]
            and left["terminal_intrinsic_time"] == right["terminal_intrinsic_time"]
        )
    checks = {
        "parameter_count_exact": (
            _analytic_parameter_count() == MODEL_SPEC["expected_parameter_count"]
        ),
        "nominal_token_integerization_declared": (
            EXECUTED_TOKENS_PER_TRAJECTORY == 2_499_993_600
            and TOKEN_SHORTFALL_FROM_NOMINAL == 6_400
        ),
        "prefix_is_exactly_eighty_percent": (
            5 * CONSTANT_SOURCE_UPDATES == 4 * CANONICAL_UPDATES
            and 5 * PREFIX_TOKENS == 4 * EXECUTED_TOKENS_PER_TRAJECTORY
        ),
        "shared_prefix_identity": bool(
            CONSTANT_MACROS * 4 == CONSTANT_SOURCE_UPDATES
            and PREFIX_INTRINSIC_TIME == CONSTANT_MACROS * MACRO_INTRINSIC_TIME
        ),
        "four_tail_step_invariants": all(row["passed"] for row in arm_checks),
        "same_shape_T_context_ratio_grids": all(shape_pairing.values()),
        "tape_identity": (not verify_tape) or tape_checks["verified"],
        "per_update_training_ce_retained": True,
        "fixed_probe_validation_is_primary": True,
        "launch_authorized": False,
    }
    passed = all(value for key, value in checks.items() if key != "launch_authorized")
    return {
        "schema_version": PREFLIGHT_SCHEMA,
        "campaign_id": CAMPAIGN_ID,
        "status": "passed" if passed else "failed",
        "launch_authorized": False,
        "schedule_manifest_sha256": manifest["sha256"],
        "checks": checks,
        "shape_pairing": shape_pairing,
        "arms": arm_checks,
        "tape": tape_checks,
        "readiness": {
            "code_and_schedule": "ready" if passed else "blocked",
            "expanded_corpus": "pending_materialization_and_hash_freeze",
            "h200_b0": "pending_response_free_gpu_measurement",
            "formal_gpu_submission": "not_authorized",
        },
    }


def _load_json(path: Path) -> Json:
    with path.open("r", encoding="utf-8") as stream:
        value = json.load(stream)
    if type(value) is not dict:
        raise CampaignError(f"JSON root must be an object: {path}")
    return value


def _artifact_record(path: Path) -> Json:
    return {
        "path": path.name,
        "sha256": _sha256_file(path),
        "size_bytes": path.stat().st_size,
    }


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    raw = json.dumps(
        value,
        indent=2,
        sort_keys=True,
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8") + b"\n"
    temporary = path.with_name(path.name + ".part")
    if path.exists() or path.is_symlink() or temporary.exists() or temporary.is_symlink():
        raise CampaignError(f"refusing to replace artifact: {path}")
    with temporary.open("xb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _atomic_npz(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    temporary = path.with_name(path.name + ".part")
    if path.exists() or path.is_symlink() or temporary.exists() or temporary.is_symlink():
        raise CampaignError(f"refusing to replace artifact: {path}")
    with temporary.open("xb") as stream:
        np.savez_compressed(stream, **arrays)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _state_sha256(state: Mapping[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode("utf-8") + b"\0")
        digest.update(str(value.dtype).encode("ascii") + b"\0")
        digest.update(np.asarray(value.shape, dtype="<i8").tobytes())
        digest.update(value.numpy().tobytes(order="C"))
    return digest.hexdigest()


@lru_cache(maxsize=1)
def _model_state_contract() -> tuple[tuple[str, tuple[int, ...], torch.dtype], ...]:
    # A meta-device instance exposes the exact state inventory without
    # allocating a second ~495 MB model while a collected checkpoint is being
    # validated offline.
    with torch.device("meta"):
        reference = GPT(_model_config())
    return tuple(
        (name, tuple(tensor.shape), tensor.dtype)
        for name, tensor in reference.state_dict().items()
    )


def _validate_model_state(state: Any) -> Mapping[str, torch.Tensor]:
    if type(state) is not dict:
        raise CampaignError("prefix checkpoint model state is not an object")
    expected = {
        name: (shape, dtype)
        for name, shape, dtype in _model_state_contract()
    }
    if set(state) != set(expected):
        raise CampaignError("prefix checkpoint model state keys changed")
    for name, tensor in state.items():
        shape, dtype = expected[name]
        if (
            not isinstance(tensor, torch.Tensor)
            or tensor.device.type != "cpu"
            or tuple(tensor.shape) != shape
            or tensor.dtype != dtype
            or not bool(torch.all(torch.isfinite(tensor)))
        ):
            raise CampaignError(
                f"prefix checkpoint tensor contract changed: {name}"
            )
    if not torch.equal(
        state["transformer.wte.weight"], state["lm_head.weight"]
    ):
        raise CampaignError("prefix checkpoint tied embedding weights diverged")
    return state


def _atomic_checkpoint(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_name(path.name + ".part")
    if path.exists() or path.is_symlink() or temporary.exists() or temporary.is_symlink():
        raise CampaignError(f"refusing to replace artifact: {path}")
    with temporary.open("xb") as stream:
        torch.save(dict(payload), stream)
        stream.flush()
        os.fsync(stream.fileno())
    checked = torch.load(temporary, map_location="cpu", weights_only=True)
    if type(checked) is not dict or checked.get("schema_version") != CHECKPOINT_SCHEMA:
        raise CampaignError("prefix checkpoint round trip failed")
    os.replace(temporary, path)


def _validate_data_record(record: Any, role: str) -> Json:
    if type(record) is not dict:
        raise CampaignError(f"data.{role} must be an object")
    required = {"path", "sha256", "size_bytes"}
    if role in {"train", "validation"}:
        required |= {"token_count", "dtype"}
    if role == "training_tape":
        required |= {"context_count", "dtype", "offsets_sha256"}
    if role == "validation_probe":
        required |= {"context_count", "dtype"}
    if role == "source_manifest":
        required |= {
            "schema_version",
            "status",
            "manifest_sha256",
            "capacity_certificate_sha256",
            "source_prefix_stop_exclusive",
        }
    if not required.issubset(record):
        raise CampaignError(f"data.{role} is missing required fields")
    digest = record.get("sha256")
    if (
        type(digest) is not str
        or len(digest) != 64
        or any(character not in "0123456789abcdef" for character in digest)
    ):
        raise CampaignError(f"data.{role}.sha256 is not frozen")
    if role == "source_manifest" and (
        record.get("schema_version") != SOURCE_MANIFEST_SCHEMA
        or record.get("status") != "complete"
        or not _is_hex64(record.get("manifest_sha256"))
        or not _is_hex64(record.get("capacity_certificate_sha256"))
        or type(record.get("source_prefix_stop_exclusive")) is not int
        or not 1 <= record["source_prefix_stop_exclusive"] <= 80
    ):
        raise CampaignError("complete source-manifest lineage is not frozen")
    return dict(record)


def _is_hex64(value: Any) -> bool:
    return bool(
        type(value) is str
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _positive_finite(value: Any) -> bool:
    return bool(
        type(value) in (int, float)
        and type(value) is not bool
        and math.isfinite(float(value))
        and float(value) > 0.0
    )


def _data_identity_sha256(data: Mapping[str, Mapping[str, Any]]) -> str:
    return _canonical_sha256(
        {
            role: {
                key: record[key]
                for key in ("sha256", "size_bytes")
            }
            for role, record in data.items()
        }
    )


def derive_resource_budget(
    phase: str,
    arm_id: str | None,
    calibration: Mapping[str, Any],
) -> Json:
    if phase == "prefix" and arm_id is None:
        updates = CONSTANT_SOURCE_UPDATES
        evaluations = len(_prefix_evaluation_macros())
        step_seconds = calibration.get("base_step_time_p90_seconds")
    elif phase == "tail" and arm_id in ARM_IDS:
        plan = _arm_plan(str(arm_id))
        updates = int(plan["tail_updates"])
        evaluations = len(plan["evaluation_macros"])
        if plan["realization_id"] == "fixed_batch_lr":
            step_seconds = calibration.get("base_step_time_p90_seconds")
        else:
            base_seconds = calibration.get("base_step_time_p90_seconds")
            maximum_seconds = calibration.get(
                "maximum_batch_step_time_p90_seconds"
            )
            if not _positive_finite(base_seconds) or not _positive_finite(
                maximum_seconds
            ):
                raise CampaignError("H200 calibration timings are invalid")
            # Variable-batch tails visit every integer B in [4, 40].  The B0
            # samples both endpoints; use the slower endpoint for every update
            # rather than assuming that the B40 endpoint must be slower.
            step_seconds = max(float(base_seconds), float(maximum_seconds))
    else:
        raise CampaignError("resource budget phase/arm is invalid")
    evaluation_seconds = calibration.get(
        "validation_evaluation_time_max_seconds"
    )
    if not _positive_finite(step_seconds) or not _positive_finite(
        evaluation_seconds
    ):
        raise CampaignError("H200 calibration timings are invalid")
    measured_work_seconds = (
        updates * float(step_seconds)
        + evaluations * float(evaluation_seconds)
    )
    predicted_seconds = measured_work_seconds + RESOURCE_FIXED_OVERHEAD_SECONDS
    hard_seconds = (
        RESOURCE_HARD_LIMIT_MULTIPLIER * measured_work_seconds
        + RESOURCE_FIXED_OVERHEAD_SECONDS
    )
    return {
        "schema_version": RESOURCE_BUDGET_SCHEMA,
        "phase": phase,
        "arm_id": arm_id,
        "source_b0_result_sha256": calibration.get("b0_result_sha256"),
        "update_cost_source": (
            "B4_step_p90"
            if phase == "prefix"
            or _arm_plan(str(arm_id))["realization_id"] == "fixed_batch_lr"
            else "max_B4_B40_step_p90_conservative_for_variable_batch_updates"
        ),
        "training_updates": updates,
        "validation_evaluations": evaluations,
        "fixed_overhead_seconds": RESOURCE_FIXED_OVERHEAD_SECONDS,
        "hard_limit_multiplier": RESOURCE_HARD_LIMIT_MULTIPLIER,
        "predicted_gpu_hours": predicted_seconds / 3_600.0,
        "maximum_gpu_hours": hard_seconds / 3_600.0,
    }


def _validate_resource_contract(
    config: Mapping[str, Any],
    phase: str,
    arm_id: str | None,
    data_identity_sha256: str,
) -> None:
    runtime = config["runtime"]
    calibration = config.get("resource_calibration")
    budget = runtime.get("formal_wall_time_and_gpu_hour_budget")
    if phase == "b0":
        if calibration is not None or budget != {
            "schema_version": RESOURCE_BUDGET_SCHEMA,
            "phase": "b0",
            "maximum_gpu_hours": B0_MAXIMUM_GPU_HOURS,
            "purpose": "response_free_H200_measurement_only",
        }:
            raise CampaignError("B0 resource contract changed")
        return
    expected_calibration_keys = {
        "schema_version",
        "b0_result_sha256",
        "b0_trace_sha256",
        "initialization_sha256",
        "data_identity_sha256",
        "schedule_manifest_sha256",
        "device_name",
        "base_step_time_p90_seconds",
        "maximum_batch_step_time_p90_seconds",
        "validation_evaluation_time_max_seconds",
    }
    if (
        type(calibration) is not dict
        or set(calibration) != expected_calibration_keys
        or calibration.get("schema_version") != RESOURCE_CALIBRATION_SCHEMA
        or not _is_hex64(calibration.get("b0_result_sha256"))
        or not _is_hex64(calibration.get("b0_trace_sha256"))
        or not _is_hex64(calibration.get("initialization_sha256"))
        or calibration.get("data_identity_sha256") != data_identity_sha256
        or calibration.get("schedule_manifest_sha256")
        != schedule_manifest()["sha256"]
        or type(calibration.get("device_name")) is not str
        or "H200" not in calibration["device_name"]
        or any(
            not _positive_finite(calibration.get(key))
            for key in (
                "base_step_time_p90_seconds",
                "maximum_batch_step_time_p90_seconds",
                "validation_evaluation_time_max_seconds",
            )
        )
    ):
        raise CampaignError("completed H200 B0 calibration is absent or invalid")
    expected_budget = derive_resource_budget(phase, arm_id, calibration)
    if budget != expected_budget:
        raise CampaignError("formal resource budget is not derived from H200 B0")


def validate_config(config: Mapping[str, Any]) -> Json:
    if (
        config.get("schema_version") != CONFIG_SCHEMA
        or config.get("campaign_id") != CAMPAIGN_ID
        or config.get("primary_prediction_id") != PRIMARY_PREDICTION_ID
        or config.get("theorem_facing") is not False
    ):
        raise CampaignError("config identity changed")
    phase = config.get("phase")
    arm_id = config.get("arm_id")
    if phase == "prefix":
        if arm_id is not None:
            raise CampaignError("prefix config cannot select a tail arm")
    elif phase == "tail":
        if arm_id not in ARM_IDS:
            raise CampaignError("tail config must select one frozen arm")
    elif phase == "b0":
        if arm_id is not None:
            raise CampaignError("B0 config cannot select a tail arm")
    else:
        raise CampaignError("phase must be b0, prefix, or tail")
    if config.get("launch_control") != {"cluster_submission_authorized": False}:
        raise CampaignError("checked-in config must remain launch-false")
    prefix_checkpoint = config.get("prefix_checkpoint")
    if phase == "tail":
        if type(prefix_checkpoint) is not dict or set(prefix_checkpoint) != {
            "path",
            "sha256",
            "size_bytes",
            "model_state_sha256",
        }:
            raise CampaignError("tail config must freeze the collected prefix checkpoint")
        _validate_data_record(prefix_checkpoint, "prefix_checkpoint")
        model_state_sha = prefix_checkpoint.get("model_state_sha256")
        if (
            type(model_state_sha) is not str
            or len(model_state_sha) != 64
            or any(character not in "0123456789abcdef" for character in model_state_sha)
        ):
            raise CampaignError("prefix checkpoint model-state hash is not frozen")
    elif prefix_checkpoint is not None:
        raise CampaignError("only tail configs may name a prefix checkpoint")
    if config.get("model") != MODEL_SPEC:
        raise CampaignError("124M block-256 model contract changed")
    expected_optimizer = {
        "name": "sgd",
        "momentum": 0.0,
        "dampening": 0.0,
        "nesterov": False,
        "weight_decay": 0.0,
        "gradient_clip": 0.0,
        "foreach": False,
    }
    if config.get("optimizer") != expected_optimizer:
        raise CampaignError("plain-SGD contract changed")
    manifest = schedule_manifest()
    if config.get("schedule_manifest_sha256") != manifest["sha256"]:
        raise CampaignError("schedule manifest hash changed")
    data = config.get("data")
    if type(data) is not dict:
        raise CampaignError("data contract is absent")
    checked_data = {
        role: _validate_data_record(data.get(role), role)
        for role in (
            "source_manifest",
            "metadata",
            "train",
            "validation",
            "validation_probe",
            "training_tape",
        )
    }
    if (
        checked_data["source_manifest"].get("schema_version")
        != SOURCE_MANIFEST_SCHEMA
        or checked_data["source_manifest"].get("status") != "complete"
        or checked_data["train"].get("token_count") != CORPUS_TOKENS
        or checked_data["train"].get("dtype") != "uint16"
        or checked_data["train"].get("size_bytes") != 2 * CORPUS_TOKENS
        or checked_data["validation"].get("token_count") != 16_777_216
        or checked_data["validation"].get("dtype") != "uint16"
        or checked_data["validation_probe"].get("context_count") != VALIDATION_CONTEXTS
        or checked_data["validation_probe"].get("dtype") != "int64"
        or checked_data["training_tape"].get("context_count") != TAPE_CONTEXTS
        or checked_data["training_tape"].get("dtype") != "int64"
        or checked_data["training_tape"].get("offsets_sha256") != TAPE_OFFSETS_SHA256
    ):
        raise CampaignError("data shape contract changed")
    runtime = config.get("runtime")
    if type(runtime) is not dict:
        raise CampaignError("runtime contract is absent")
    for key, expected in {
        "device": "cuda",
        "required_gpu_name_substring": "H200",
        "visible_gpu_count": 1,
        "dtype": "float32",
        "amp": False,
        "tf32": False,
        "compile": False,
        "deterministic": True,
        "math_sdpa_only": True,
        "validation_batch_size": VALIDATION_BATCH_SIZE,
    }.items():
        if runtime.get(key) != expected:
            raise CampaignError(f"runtime.{key} changed")
    data_identity_sha256 = _data_identity_sha256(checked_data)
    _validate_resource_contract(
        config, str(phase), arm_id, data_identity_sha256
    )
    return {
        "phase": phase,
        "arm_id": arm_id,
        "data": checked_data,
        "prefix_checkpoint": prefix_checkpoint,
        "data_identity_sha256": data_identity_sha256,
        "resource_calibration": config.get("resource_calibration"),
    }


def _resolved_regular_file(record: Mapping[str, Any], label: str) -> Path:
    path = Path(str(record["path"])).expanduser().resolve()
    if not path.is_file() or path.is_symlink():
        raise CampaignError(f"{label} is absent or linked: {path}")
    if (
        path.stat().st_size != int(record["size_bytes"])
        or _sha256_file(path) != record["sha256"]
    ):
        raise CampaignError(f"{label} bytes changed")
    return path


def _load_inputs(config: Mapping[str, Any]) -> tuple[np.memmap, np.memmap, np.ndarray, np.memmap, Json]:
    checked = validate_config(config)["data"]
    paths = {
        role: _resolved_regular_file(record, role)
        for role, record in checked.items()
    }
    source_manifest = _load_json(paths["source_manifest"])
    source_record = checked["source_manifest"]
    if (
        source_manifest.get("schema_version") != SOURCE_MANIFEST_SCHEMA
        or source_manifest.get("status") != "complete"
        or source_manifest.get("manifest_sha256")
        != source_record["manifest_sha256"]
        or source_manifest.get("capacity_audit", {}).get(
            "certificate_sha256"
        )
        != source_record["capacity_certificate_sha256"]
        or len(source_manifest.get("sources", []))
        != source_record["source_prefix_stop_exclusive"]
    ):
        raise CampaignError("loaded source-manifest lineage changed")
    train = np.memmap(paths["train"], dtype="<u2", mode="r")
    validation = np.memmap(paths["validation"], dtype="<u2", mode="r")
    probe = np.load(paths["validation_probe"], allow_pickle=False)
    tape = np.load(paths["training_tape"], allow_pickle=False, mmap_mode="r")
    if (
        train.shape != (CORPUS_TOKENS,)
        or validation.shape != (16_777_216,)
        or probe.shape != (VALIDATION_CONTEXTS,)
        or probe.dtype.str != "<i8"
        or tape.shape != (TAPE_CONTEXTS,)
        or tape.dtype.str != "<i8"
        or _array_sha256(tape) != TAPE_OFFSETS_SHA256
        or int(np.min(probe)) < 0
        or int(np.max(probe)) + BLOCK_SIZE >= len(validation)
        or int(np.max(train)) >= VOCAB_SIZE
        or int(np.max(validation)) >= VOCAB_SIZE
    ):
        raise CampaignError("loaded data failed shape/range/tape validation")
    identity = {
        role: {key: record[key] for key in ("sha256", "size_bytes")}
        for role, record in checked.items()
    }
    return train, validation, np.ascontiguousarray(probe), tape, identity


def _configure_runtime(config: Mapping[str, Any]) -> torch.device:
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8":
        raise CampaignError("CUBLAS_WORKSPACE_CONFIG must equal ':4096:8'")
    torch.manual_seed(INITIALIZATION_SEED)
    torch.cuda.manual_seed_all(INITIALIZATION_SEED)
    torch.use_deterministic_algorithms(True, warn_only=False)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_cudnn_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)
    if (
        not torch.are_deterministic_algorithms_enabled()
        or torch.is_deterministic_algorithms_warn_only_enabled()
        or torch.backends.cuda.flash_sdp_enabled()
        or torch.backends.cuda.mem_efficient_sdp_enabled()
        or torch.backends.cuda.cudnn_sdp_enabled()
        or not torch.backends.cuda.math_sdp_enabled()
    ):
        raise CampaignError("deterministic math-SDPA runtime setup failed")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise CampaignError("execution requires exactly one visible CUDA GPU")
    device = torch.device("cuda", 0)
    name = torch.cuda.get_device_name(device)
    required = str(config["runtime"]["required_gpu_name_substring"])
    if required not in name:
        raise CampaignError(f"GPU {name!r} does not contain {required!r}")
    return device


def _new_model(
    device: torch.device,
    *,
    expected_initialization_sha256: str | None = None,
) -> tuple[GPT, str]:
    torch.manual_seed(INITIALIZATION_SEED)
    torch.cuda.manual_seed_all(INITIALIZATION_SEED)
    model = GPT(_model_config())
    if sum(parameter.numel() for parameter in model.parameters()) != MODEL_SPEC[
        "expected_parameter_count"
    ]:
        raise CampaignError("model parameter count changed")
    initialization_sha256 = _state_sha256(model.state_dict())
    if (
        expected_initialization_sha256 is not None
        and initialization_sha256 != expected_initialization_sha256
    ):
        raise CampaignError("model initialization disagrees with H200 B0")
    return model.to(device), initialization_sha256


def _optimizer(model: torch.nn.Module, learning_rate: float) -> torch.optim.SGD:
    return torch.optim.SGD(
        list(model.parameters()),
        lr=float(learning_rate),
        momentum=0.0,
        dampening=0.0,
        weight_decay=0.0,
        nesterov=False,
        foreach=False,
    )


def _batch(values: np.ndarray, offsets: Sequence[int] | np.ndarray, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    starts = [int(value) for value in offsets]
    x = np.stack(
        [np.asarray(values[start : start + BLOCK_SIZE], dtype=np.int64) for start in starts]
    )
    y = np.stack(
        [
            np.asarray(values[start + 1 : start + BLOCK_SIZE + 1], dtype=np.int64)
            for start in starts
        ]
    )
    return torch.from_numpy(x).to(device), torch.from_numpy(y).to(device)


def _training_step(
    model: GPT,
    optimizer: torch.optim.SGD,
    train: np.ndarray,
    offsets: np.ndarray,
    learning_rate: float,
    device: torch.device,
) -> float:
    if not 1 <= len(offsets) <= 40 or not math.isfinite(learning_rate) or learning_rate <= 0:
        raise CampaignError("invalid update descriptor")
    for group in optimizer.param_groups:
        group["lr"] = float(learning_rate)
    model.train()
    optimizer.zero_grad(set_to_none=True)
    x, y = _batch(train, offsets, device)
    _, loss, _ = model(x, y)
    if loss is None or not bool(torch.isfinite(loss).detach().cpu()):
        raise CampaignError("nonfinite training cross-entropy")
    loss.backward()
    optimizer.step()
    # Stateless SGD keeps this mapping empty.  Inspect it directly instead of
    # rebuilding a serialized state_dict on every one of millions of updates.
    if optimizer.state:
        raise CampaignError("plain SGD acquired optimizer state")
    value = float(loss.detach().cpu())
    if value < 0.0 or not math.isfinite(value):
        raise CampaignError("invalid training cross-entropy")
    return value


@torch.inference_mode()
def _evaluate(
    model: GPT,
    validation: np.ndarray,
    probe: np.ndarray,
    device: torch.device,
) -> np.ndarray:
    was_training = model.training
    model.eval()
    rows: list[np.ndarray] = []
    for start in range(0, VALIDATION_CONTEXTS, VALIDATION_BATCH_SIZE):
        offsets = probe[start : start + VALIDATION_BATCH_SIZE]
        x, y = _batch(validation, offsets, device)
        logits, _, _ = model(x, None)
        losses = F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]),
            y.reshape(-1),
            reduction="none",
        ).reshape(len(offsets), BLOCK_SIZE)
        rows.append(losses.mean(dim=1).float().cpu().numpy())
    model.train(was_training)
    result = np.ascontiguousarray(np.concatenate(rows), dtype=np.float32)
    if result.shape != (VALIDATION_CONTEXTS,) or not np.all(np.isfinite(result)):
        raise CampaignError("fixed validation probe evaluation failed")
    return result


def _evaluation_arrays(rows: Sequence[Mapping[str, Any]], contexts: Sequence[np.ndarray]) -> Json:
    matrix = np.ascontiguousarray(np.stack(contexts), dtype=np.float32)
    return {
        "schema_version": np.asarray(EVALUATION_SCHEMA),
        "macro_index": np.asarray([row["macro_index"] for row in rows], dtype=np.int64),
        "global_update": np.asarray([row["global_update"] for row in rows], dtype=np.int64),
        "tape_context_cursor": np.asarray(
            [row["tape_context_cursor"] for row in rows], dtype=np.int64
        ),
        "tokens_seen": np.asarray([row["tokens_seen"] for row in rows], dtype=np.int64),
        "intrinsic_time": np.asarray(
            [row["intrinsic_time"] for row in rows], dtype=np.float64
        ),
        "validation_cross_entropy": np.asarray(
            [row["validation_cross_entropy"] for row in rows], dtype=np.float64
        ),
        "context_cross_entropy": matrix,
    }


def _evaluation_row(macro: int, update: int, cursor: int, values: np.ndarray) -> Json:
    return {
        "macro_index": int(macro),
        "global_update": int(update),
        "tape_context_cursor": int(cursor),
        "tokens_seen": int(cursor * BLOCK_SIZE),
        "intrinsic_time": float(macro * MACRO_INTRINSIC_TIME),
        "validation_cross_entropy": float(np.mean(values, dtype=np.float64)),
        "context_cross_entropy_sha256": _array_sha256(values),
    }


def _runtime_record(device: torch.device, started: float, step_times: Sequence[float]) -> Json:
    elapsed = time.monotonic() - started
    timed = list(step_times)
    return {
        "device": device.type,
        "device_name": torch.cuda.get_device_name(device),
        "dtype": "float32",
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "python_version": platform.python_version(),
        "deterministic_algorithms": bool(
            torch.are_deterministic_algorithms_enabled()
        ),
        "deterministic_warn_only": bool(
            torch.is_deterministic_algorithms_warn_only_enabled()
        ),
        "flash_sdpa_enabled": bool(torch.backends.cuda.flash_sdp_enabled()),
        "memory_efficient_sdpa_enabled": bool(
            torch.backends.cuda.mem_efficient_sdp_enabled()
        ),
        "cudnn_sdpa_enabled": bool(torch.backends.cuda.cudnn_sdp_enabled()),
        "math_sdpa_enabled": bool(torch.backends.cuda.math_sdp_enabled()),
        "elapsed_seconds": elapsed,
        "gpu_hours": elapsed / 3_600.0,
        "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
        "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
        "sampled_step_time_median_seconds": (
            statistics.median(timed) if timed else None
        ),
        "sampled_step_time_p90_seconds": (
            float(np.percentile(timed, 90)) if timed else None
        ),
    }


def _progress(phase: str, completed: int, total: int, started: float) -> None:
    print(
        json.dumps(
            {
                "event": "progress",
                "phase": phase,
                "completed_updates": completed,
                "total_updates": total,
                "elapsed_seconds": time.monotonic() - started,
            },
            sort_keys=True,
        ),
        flush=True,
    )


def _prepare_output(path: Path) -> Path:
    output = path.expanduser().resolve()
    if output.exists() or output.is_symlink():
        raise CampaignError(f"output directory already exists: {output}")
    output.mkdir(parents=True)
    return output


def _parameters_finite(model: torch.nn.Module) -> bool:
    return all(
        bool(torch.all(torch.isfinite(parameter)).detach().cpu())
        for parameter in model.parameters()
    )


def _run_b0_arm(
    *,
    label: str,
    batch_size: int,
    learning_rate: float,
    updates: int,
    train: np.ndarray,
    validation: np.ndarray,
    probe: np.ndarray,
    tape: np.ndarray,
    device: torch.device,
) -> tuple[Json, np.ndarray, list[np.ndarray], np.ndarray, np.ndarray, str]:
    model, initialization_sha256 = _new_model(device)
    optimizer = _optimizer(model, learning_rate)
    torch.cuda.synchronize(device)
    evaluation_started = time.monotonic()
    initial = _evaluate(model, validation, probe, device)
    torch.cuda.synchronize(device)
    evaluation_times = [time.monotonic() - evaluation_started]
    losses = np.empty(updates, dtype=np.float32)
    step_times: list[float] = []
    cursor = 0
    for update in range(updates):
        torch.cuda.synchronize(device)
        started = time.monotonic()
        losses[update] = _training_step(
            model,
            optimizer,
            train,
            tape[cursor : cursor + batch_size],
            learning_rate,
            device,
        )
        torch.cuda.synchronize(device)
        step_times.append(time.monotonic() - started)
        cursor += batch_size
    torch.cuda.synchronize(device)
    evaluation_started = time.monotonic()
    final = _evaluate(model, validation, probe, device)
    torch.cuda.synchronize(device)
    evaluation_times.append(time.monotonic() - evaluation_started)
    parameters_finite = _parameters_finite(model)
    optimizer_state_entries = len(optimizer.state)
    finite = bool(
        np.all(np.isfinite(losses))
        and np.all(np.isfinite(initial))
        and np.all(np.isfinite(final))
        and parameters_finite
        and optimizer_state_entries == 0
    )
    summary = {
        "label": label,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "ratio_B_over_eta": batch_size / learning_rate,
        "updates": updates,
        "contexts": cursor,
        "tokens": cursor * BLOCK_SIZE,
        "initial_validation_cross_entropy": float(np.mean(initial, dtype=np.float64)),
        "final_validation_cross_entropy": float(np.mean(final, dtype=np.float64)),
        "maximum_training_cross_entropy": float(np.max(losses)),
        "minimum_training_cross_entropy": float(np.min(losses)),
        "step_time_median_seconds": statistics.median(step_times[8:]),
        "step_time_p90_seconds": float(np.percentile(step_times[8:], 90)),
        "tokens_per_second": (
            batch_size * BLOCK_SIZE / statistics.median(step_times[8:])
        ),
        "validation_evaluation_time_max_seconds": max(evaluation_times),
        "optimizer_state_entries": optimizer_state_entries,
        "parameters_finite": parameters_finite,
        "all_values_and_parameters_finite": finite,
        "validation_increase_below_abort_margin": bool(
            float(np.mean(final, dtype=np.float64))
            <= float(np.mean(initial, dtype=np.float64)) + 0.25
        ),
    }
    del optimizer, model
    torch.cuda.empty_cache()
    return (
        summary,
        losses,
        [initial, final],
        np.asarray(step_times, dtype=np.float64),
        np.asarray(evaluation_times, dtype=np.float64),
        initialization_sha256,
    )


def run_b0(config: Mapping[str, Any], output_dir: Path) -> Json:
    """Measure only H200 stability, throughput, and the largest direct batch.

    This is explicitly response-free.  It does not estimate a schedule
    effect and cannot be promoted into the four-tail scientific result.
    """

    checked = validate_config(config)
    if checked["phase"] != "b0":
        raise CampaignError("run_b0 requires phase=b0")
    started = time.monotonic()
    train, validation, probe, tape, data_identity = _load_inputs(config)
    device = _configure_runtime(config)
    output = _prepare_output(output_dir)
    torch.cuda.reset_peak_memory_stats(device)
    (
        base,
        base_ce,
        base_eval,
        base_step_times,
        base_evaluation_times,
        base_initialization_sha256,
    ) = (
        _run_b0_arm(
            label="base_B4_eta0p005",
            batch_size=4,
            learning_rate=CANONICAL_BASE_LEARNING_RATE,
            updates=B0_BASE_UPDATES,
            train=train,
            validation=validation,
            probe=probe,
            tape=tape,
            device=device,
        )
    )
    (
        maximum,
        maximum_ce,
        maximum_eval,
        maximum_step_times,
        maximum_evaluation_times,
        maximum_initialization_sha256,
    ) = _run_b0_arm(
        label="maximum_batch_B40_eta0p005",
        batch_size=40,
        learning_rate=CANONICAL_BASE_LEARNING_RATE,
        updates=B0_MAX_BATCH_UPDATES,
        train=train,
        validation=validation,
        probe=probe,
        tape=tape,
        device=device,
    )
    if base_initialization_sha256 != maximum_initialization_sha256:
        raise CampaignError("repeated initialization changed within H200 B0")
    trace_path = output / TRACE_FILENAME
    _atomic_npz(
        trace_path,
        {
            "schema_version": np.asarray(TRACE_SCHEMA),
            "phase": np.asarray("b0"),
            "base_training_cross_entropy": base_ce,
            "maximum_batch_training_cross_entropy": maximum_ce,
            "base_validation_context_cross_entropy": np.stack(base_eval).astype(
                np.float32
            ),
            "maximum_batch_validation_context_cross_entropy": np.stack(
                maximum_eval
            ).astype(np.float32),
            "base_step_time_seconds": base_step_times,
            "maximum_batch_step_time_seconds": maximum_step_times,
            "base_validation_evaluation_time_seconds": base_evaluation_times,
            "maximum_batch_validation_evaluation_time_seconds": (
                maximum_evaluation_times
            ),
            "base_optimizer_state_entries": np.asarray(
                base["optimizer_state_entries"], dtype=np.int64
            ),
            "maximum_batch_optimizer_state_entries": np.asarray(
                maximum["optimizer_state_entries"], dtype=np.int64
            ),
            "base_parameters_finite": np.asarray(
                base["parameters_finite"], dtype=np.bool_
            ),
            "maximum_batch_parameters_finite": np.asarray(
                maximum["parameters_finite"], dtype=np.bool_
            ),
        },
    )
    runtime = _runtime_record(device, started, ())
    total_memory = int(torch.cuda.get_device_properties(device).total_memory)
    gates = {
        "base_path_finite": base["all_values_and_parameters_finite"],
        "maximum_batch_path_finite": maximum["all_values_and_parameters_finite"],
        "base_validation_abort_margin": base[
            "validation_increase_below_abort_margin"
        ],
        "maximum_batch_validation_abort_margin": maximum[
            "validation_increase_below_abort_margin"
        ],
        "plain_sgd_state_empty": bool(
            base["optimizer_state_entries"] == 0
            and maximum["optimizer_state_entries"] == 0
        ),
        "peak_reserved_below_80_percent_H200": bool(
            runtime["peak_reserved_bytes"] <= 0.8 * total_memory
        ),
        "wall_time_below_0p25_H200_hours": bool(runtime["gpu_hours"] <= 0.25),
    }
    result = {
        "schema_version": RESULT_SCHEMA,
        "campaign_id": CAMPAIGN_ID,
        "phase": "b0",
        "arm_id": None,
        "status": "passed" if all(gates.values()) else "failed",
        "contains_prediction_response": False,
        "scope": "resource_and_numerical_preflight_only",
        "initialization_sha256": base_initialization_sha256,
        "schedule_manifest_sha256": schedule_manifest()["sha256"],
        "data_identity_sha256": _canonical_sha256(data_identity),
        "resource_calibration": None,
        "resource_budget": config["runtime"][
            "formal_wall_time_and_gpu_hour_budget"
        ],
        "measurements": {"base": base, "maximum_batch": maximum},
        "gates": gates,
        "runtime": {**runtime, "physical_gpu_memory_bytes": total_memory},
        "artifacts": {"diagnostic_trace": _artifact_record(trace_path)},
    }
    _atomic_json(output / RESULT_FILENAME, result)
    return result


def run_prefix(config: Mapping[str, Any], output_dir: Path) -> Json:
    checked = validate_config(config)
    if checked["phase"] != "prefix":
        raise CampaignError("run_prefix requires phase=prefix")
    started = time.monotonic()
    train, validation, probe, tape, data_identity = _load_inputs(config)
    device = _configure_runtime(config)
    output = _prepare_output(output_dir)
    torch.cuda.reset_peak_memory_stats(device)
    expected_initialization_sha256 = str(
        checked["resource_calibration"]["initialization_sha256"]
    )
    model, initialization_sha256 = _new_model(
        device,
        expected_initialization_sha256=expected_initialization_sha256,
    )
    optimizer = _optimizer(model, CANONICAL_BASE_LEARNING_RATE)
    trace = np.empty(CONSTANT_SOURCE_UPDATES, dtype=np.float32)
    evaluation_set = set(_prefix_evaluation_macros())
    rows: list[Json] = []
    contexts: list[np.ndarray] = []
    initial = _evaluate(model, validation, probe, device)
    rows.append(_evaluation_row(0, 0, 0, initial))
    contexts.append(initial)
    step_times: list[float] = []
    for index in range(CONSTANT_SOURCE_UPDATES):
        begin = 4 * index
        if index % 1_024 == 0:
            torch.cuda.synchronize(device)
            tick = time.monotonic()
        trace[index] = _training_step(
            model,
            optimizer,
            train,
            tape[begin : begin + ATOMIC_BATCH_SIZE],
            CANONICAL_BASE_LEARNING_RATE,
            device,
        )
        completed = index + 1
        if index % 1_024 == 0:
            torch.cuda.synchronize(device)
            step_times.append(time.monotonic() - tick)
        if completed % 4 == 0:
            macro = completed // 4
            if macro in evaluation_set:
                values = _evaluate(model, validation, probe, device)
                rows.append(_evaluation_row(macro, completed, 4 * completed, values))
                contexts.append(values)
        if completed % 4_096 == 0:
            _progress("prefix", completed, CONSTANT_SOURCE_UPDATES, started)
    if not np.all(np.isfinite(trace)) or optimizer.state_dict()["state"]:
        raise CampaignError("prefix completion invariants failed")
    final_state = {
        name: tensor.detach().cpu().contiguous()
        for name, tensor in model.state_dict().items()
    }
    state_sha256 = _state_sha256(final_state)
    checkpoint = {
        "schema_version": CHECKPOINT_SCHEMA,
        "campaign_id": CAMPAIGN_ID,
        "schedule_manifest_sha256": schedule_manifest()["sha256"],
        "model_config": asdict(_model_config()),
        "model_state": final_state,
        "model_state_sha256": state_sha256,
        "optimizer_state_entries": 0,
        "initialization_seed": INITIALIZATION_SEED,
        "initialization_sha256": initialization_sha256,
        "training_tape_seed": TRAINING_TAPE_SEED,
        "training_tape_sha256": TAPE_OFFSETS_SHA256,
        "data_identity_sha256": _canonical_sha256(data_identity),
        "completed_updates": CONSTANT_SOURCE_UPDATES,
        "completed_macros": CONSTANT_MACROS,
        "tape_context_cursor": PREFIX_CONTEXTS,
        "tokens_seen": PREFIX_TOKENS,
        "intrinsic_time": PREFIX_INTRINSIC_TIME,
        "final_validation_context_cross_entropy": torch.from_numpy(
            np.ascontiguousarray(contexts[-1], dtype=np.float32)
        ),
    }
    checkpoint_path = output / PREFIX_CHECKPOINT_FILENAME
    _atomic_checkpoint(checkpoint_path, checkpoint)
    trace_path = output / TRACE_FILENAME
    _atomic_npz(
        trace_path,
        {
            "schema_version": np.asarray(TRACE_SCHEMA),
            "phase": np.asarray("prefix"),
            "training_cross_entropy": trace,
        },
    )
    evaluation_path = output / EVALUATION_FILENAME
    _atomic_npz(evaluation_path, _evaluation_arrays(rows, contexts))
    runtime = _runtime_record(device, started, step_times)
    result = {
        "schema_version": RESULT_SCHEMA,
        "campaign_id": CAMPAIGN_ID,
        "phase": "prefix",
        "arm_id": None,
        "status": "completed",
        "contains_prediction_response": False,
        "schedule_manifest_sha256": schedule_manifest()["sha256"],
        "data_identity_sha256": _canonical_sha256(data_identity),
        "resource_calibration": config["resource_calibration"],
        "resource_budget": config["runtime"][
            "formal_wall_time_and_gpu_hour_budget"
        ],
        "counts": {
            "updates": CONSTANT_SOURCE_UPDATES,
            "contexts": PREFIX_CONTEXTS,
            "tokens": PREFIX_TOKENS,
            "evaluation_rows": len(rows),
        },
        "checkpoint_model_state_sha256": state_sha256,
        "artifacts": {
            "checkpoint": _artifact_record(checkpoint_path),
            "training_trace": _artifact_record(trace_path),
            "validation_evaluations": _artifact_record(evaluation_path),
        },
        "runtime": runtime,
    }
    _atomic_json(output / RESULT_FILENAME, result)
    return result


def _load_prefix_checkpoint(
    path: Path,
    data_identity_sha256: str,
    initialization_sha256: str,
) -> Mapping[str, Any]:
    if not path.is_file() or path.is_symlink():
        raise CampaignError(f"prefix checkpoint is absent or linked: {path}")
    value = torch.load(path, map_location="cpu", weights_only=True)
    required = {
        "schema_version": CHECKPOINT_SCHEMA,
        "campaign_id": CAMPAIGN_ID,
        "schedule_manifest_sha256": schedule_manifest()["sha256"],
        "model_config": asdict(_model_config()),
        "optimizer_state_entries": 0,
        "initialization_seed": INITIALIZATION_SEED,
        "initialization_sha256": initialization_sha256,
        "training_tape_seed": TRAINING_TAPE_SEED,
        "training_tape_sha256": TAPE_OFFSETS_SHA256,
        "data_identity_sha256": data_identity_sha256,
        "completed_updates": CONSTANT_SOURCE_UPDATES,
        "completed_macros": CONSTANT_MACROS,
        "tape_context_cursor": PREFIX_CONTEXTS,
        "tokens_seen": PREFIX_TOKENS,
        "intrinsic_time": PREFIX_INTRINSIC_TIME,
    }
    if type(value) is not dict or any(value.get(key) != expected for key, expected in required.items()):
        raise CampaignError("prefix checkpoint metadata changed")
    state = _validate_model_state(value.get("model_state"))
    if _state_sha256(state) != value.get("model_state_sha256"):
        raise CampaignError("prefix checkpoint model state changed")
    final_validation = value.get("final_validation_context_cross_entropy")
    if (
        not isinstance(final_validation, torch.Tensor)
        or final_validation.dtype != torch.float32
        or tuple(final_validation.shape) != (VALIDATION_CONTEXTS,)
        or not bool(torch.all(torch.isfinite(final_validation)))
    ):
        raise CampaignError("prefix checkpoint validation anchor changed")
    return value


def run_tail(config: Mapping[str, Any], checkpoint_path: Path, output_dir: Path) -> Json:
    checked = validate_config(config)
    if checked["phase"] != "tail" or checked["arm_id"] not in ARM_IDS:
        raise CampaignError("run_tail requires one frozen tail arm")
    arm_id = str(checked["arm_id"])
    plan = _arm_plan(arm_id)
    started = time.monotonic()
    train, validation, probe, tape, data_identity = _load_inputs(config)
    data_identity_sha256 = _canonical_sha256(data_identity)
    checkpoint_path = checkpoint_path.expanduser().resolve()
    checkpoint_record = checked["prefix_checkpoint"]
    if (
        not checkpoint_path.is_file()
        or checkpoint_path.is_symlink()
        or checkpoint_path.stat().st_size != int(checkpoint_record["size_bytes"])
        or _sha256_file(checkpoint_path) != checkpoint_record["sha256"]
    ):
        raise CampaignError("prefix checkpoint file disagrees with tail config")
    checkpoint = _load_prefix_checkpoint(
        checkpoint_path,
        data_identity_sha256,
        str(checked["resource_calibration"]["initialization_sha256"]),
    )
    if checkpoint["model_state_sha256"] != checkpoint_record["model_state_sha256"]:
        raise CampaignError("prefix checkpoint state disagrees with tail config")
    device = _configure_runtime(config)
    output = _prepare_output(output_dir)
    torch.cuda.reset_peak_memory_stats(device)
    model = GPT(_model_config())
    model.load_state_dict(_strip_compile_prefix(dict(checkpoint["model_state"])), strict=True)
    model.to(device)
    optimizer = _optimizer(model, CANONICAL_BASE_LEARNING_RATE)
    trace = np.empty(int(plan["tail_updates"]), dtype=np.float32)
    evaluation_set = set(plan["evaluation_macros"])
    rows: list[Json] = []
    contexts: list[np.ndarray] = []
    anchor = checkpoint["final_validation_context_cross_entropy"].numpy().copy()
    reloaded_anchor = _evaluate(model, validation, probe, device)
    if _array_sha256(reloaded_anchor) != _array_sha256(anchor):
        raise CampaignError("tail load did not reproduce the prefix validation anchor")
    anchor = reloaded_anchor
    rows.append(
        _evaluation_row(
            CONSTANT_MACROS,
            CONSTANT_SOURCE_UPDATES,
            PREFIX_CONTEXTS,
            anchor,
        )
    )
    contexts.append(np.ascontiguousarray(anchor, dtype=np.float32))
    step_times: list[float] = []
    last_step: Mapping[str, Any] | None = None
    for index, step in enumerate(_iter_tail_steps(arm_id)):
        start = int(step["tape_context_start"])
        stop = int(step["tape_context_stop"])
        if index % 1_024 == 0:
            torch.cuda.synchronize(device)
            tick = time.monotonic()
        trace[index] = _training_step(
            model,
            optimizer,
            train,
            tape[start:stop],
            float(step["learning_rate"]),
            device,
        )
        if index % 1_024 == 0:
            torch.cuda.synchronize(device)
            step_times.append(time.monotonic() - tick)
        last_step = step
        if bool(step["macro_completed"]):
            macro = int(step["macro_index"]) + 1
            if macro in evaluation_set and macro != CONSTANT_MACROS:
                values = _evaluate(model, validation, probe, device)
                rows.append(
                    _evaluation_row(
                        macro,
                        int(step["global_update"]),
                        int(step["tape_context_stop"]),
                        values,
                    )
                )
                contexts.append(values)
        completed = index + 1
        if completed % 4_096 == 0:
            _progress(arm_id, completed, int(plan["tail_updates"]), started)
    if (
        last_step is None
        or int(last_step["tape_context_stop"]) != TAPE_CONTEXTS
        or not np.all(np.isfinite(trace))
        or optimizer.state_dict()["state"]
    ):
        raise CampaignError("tail completion invariants failed")
    trace_path = output / TRACE_FILENAME
    _atomic_npz(
        trace_path,
        {
            "schema_version": np.asarray(TRACE_SCHEMA),
            "phase": np.asarray("tail"),
            "arm_id": np.asarray(arm_id),
            "training_cross_entropy": trace,
        },
    )
    evaluation_path = output / EVALUATION_FILENAME
    _atomic_npz(evaluation_path, _evaluation_arrays(rows, contexts))
    runtime = _runtime_record(device, started, step_times)
    result = {
        "schema_version": RESULT_SCHEMA,
        "campaign_id": CAMPAIGN_ID,
        "phase": "tail",
        "arm_id": arm_id,
        "shape_id": plan["shape_id"],
        "realization_id": plan["realization_id"],
        "status": "completed",
        "contains_prediction_response": True,
        "schedule_manifest_sha256": schedule_manifest()["sha256"],
        "prefix_checkpoint_sha256": _sha256_file(checkpoint_path),
        "prefix_model_state_sha256": checkpoint["model_state_sha256"],
        "data_identity_sha256": data_identity_sha256,
        "resource_calibration": config["resource_calibration"],
        "resource_budget": config["runtime"][
            "formal_wall_time_and_gpu_hour_budget"
        ],
        "counts": {
            "prefix_updates": CONSTANT_SOURCE_UPDATES,
            "tail_updates": int(plan["tail_updates"]),
            "total_updates": int(plan["total_updates"]),
            "prefix_tokens": PREFIX_TOKENS,
            "tail_tokens": TAIL_TOKENS,
            "total_tokens": EXECUTED_TOKENS_PER_TRAJECTORY,
            "tail_evaluation_rows_including_prefix_anchor": len(rows),
        },
        "terminal_intrinsic_time": plan["terminal_intrinsic_time"],
        "artifacts": {
            "training_trace": _artifact_record(trace_path),
            "validation_evaluations": _artifact_record(evaluation_path),
        },
        "runtime": runtime,
    }
    _atomic_json(output / RESULT_FILENAME, result)
    return result


def _write_stdout(value: Mapping[str, Any]) -> None:
    print(json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False))


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--preflight", action="store_true")
    modes.add_argument("--run-b0", action="store_true")
    modes.add_argument("--run-prefix", action="store_true")
    modes.add_argument("--run-tail", action="store_true")
    parser.add_argument("--skip-tape-hash", action="store_true")
    parser.add_argument("--config", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args(argv)
    if args.preflight:
        _write_stdout(preflight(verify_tape=not args.skip_tape_hash))
        return 0
    if args.config is None or args.output_dir is None:
        parser.error("execution requires --config and --output-dir")
    config = _load_json(args.config)
    if args.run_b0:
        _write_stdout(run_b0(config, args.output_dir))
        return 0
    if args.run_prefix:
        _write_stdout(run_prefix(config, args.output_dir))
        return 0
    if args.checkpoint is None:
        parser.error("--run-tail requires --checkpoint")
    _write_stdout(run_tail(config, args.checkpoint, args.output_dir))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
