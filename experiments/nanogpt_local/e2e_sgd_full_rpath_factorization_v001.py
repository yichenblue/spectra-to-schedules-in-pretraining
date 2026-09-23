"""Full-training WSD/8-1-1 ratio-path factorization experiment.

This module is deliberately independent of the frozen short-horizon rpath
runner.  It reuses only stable nanoGPT/data primitives and executes one of
four arms.  It never submits work and has no checkpoint/resume path.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path
import statistics
import time
import traceback
from typing import Any, Iterator, Mapping, Sequence

import numpy as np
import torch

from . import e2e_sgd_b0_v001 as b0
from . import e2e_sgd_cooldown_calibration_v001 as cooldown


CAMPAIGN_ID = "nanogpt30m-e2e-sgd-full-rpath-factorization-v001"
CONFIG_SCHEMA = "nanogpt30m_e2e_sgd_full_rpath_factorization_config_v001"
RESULT_SCHEMA = "nanogpt30m_e2e_sgd_full_rpath_factorization_result_v001"
VALIDATION_SCHEMA = "nanogpt30m_e2e_sgd_full_rpath_factorization_validation_v001"
PRIMARY_PREDICTION_ID = "A2_EXTERNAL"
STAGE = "pilot_full_training_ratio_path_factorization"

SHAPE_IDS = ("wsd_exp_80_20", "eight_one_one")
REALIZATION_IDS = ("fixed_batch_lr", "fixed_lr_batch")
ARM_IDS = tuple(
    f"{shape_id}__{realization_id}"
    for shape_id in SHAPE_IDS for realization_id in REALIZATION_IDS
)
RUN_ID_BY_ARM = {
    arm_id: (
        CAMPAIGN_ID.removesuffix("-v001")
        + "-"
        + ("wsd" if arm_id.startswith("wsd_") else "811")
        + ("-fblr-v001" if arm_id.endswith("fixed_batch_lr") else "-flrbs-v001")
    )
    for arm_id in ARM_IDS
}

BLOCK_SIZE = 256
ALIGNED_STRIDE = 256
ATOMIC_BATCH_SIZE = 4
CANONICAL_UPDATES = 102_400
CONSTANT_SOURCE_UPDATES = 81_920
CANONICAL_BASE_LEARNING_RATE = 0.005
MACRO_INTRINSIC_TIME = 0.02
CONSTANT_MACROS = 20_480
TARGET_INITIAL_RATIO = 800.0
EXECUTED_RATIO_UNIT = ATOMIC_BATCH_SIZE / MACRO_INTRINSIC_TIME
MACRO_COUNTS = {"wsd_exp_80_20": 22_481, "eight_one_one": 21_546}
TERMINAL_INTRINSIC_TIMES = {
    shape_id: count * MACRO_INTRINSIC_TIME
    for shape_id, count in MACRO_COUNTS.items()
}
TAIL_RESCALES = {
    "wsd_exp_80_20": 0.9999265774962143,
    "eight_one_one": 1.0004288132549877,
}
VALIDATION_CONTEXTS = 1_024
VALIDATION_BATCH_SIZE = 32
INITIALIZATION_SEED = 2_026_090_803
TRAINING_TAPE_SEED = 2_026_090_804
TAPE_CONTEXTS = CANONICAL_UPDATES * ATOMIC_BATCH_SIZE
TAPE_OFFSETS_SHA256 = (
    "053c0d4099fee3e9d18f542dd966cec85a6885a5c60c3c7f344142463c969486"
)
EXPECTED_INITIALIZATION_SHA256 = (
    "3efbbb0af5bcd090846ea05aaeebf1ce67f6a24097e9d9d8311207e5f4899a81"
)
EXPECTED_TICK0_VALIDATION_SHA256 = (
    "78577e61ffba520902c3be853e41f4c947a109d5a5aee4784d4fc05f5b47c5af"
)
EXPECTED_PREFIX_END_VALIDATION_SHA256 = (
    "12641441202e40705a167bd15bab7a2fdbb7494b43ab6a3e3839f4d9cf58f4ba"
)
EXPECTED_PREFIX_END_VALIDATION_CE = 5.8282282571308315

EXPECTED_MODEL = {
    "family": "nanogpt", "expected_parameter_count": 30_036_864,
    "block_size": 256, "vocab_size": 50_304, "n_layer": 6,
    "n_head": 6, "n_embd": 384, "dropout": 0.0, "bias": False,
}
EXPECTED_DATA = {
    "metadata": {
        "path": "data/openwebtext-cooldown-700m-v001/metadata.json",
        "sha256": "ff0d8eefd80858aa07aa0e469996a106c40266ae1cfaf7c8979a85bfb6a569e1",
    },
    "train": {
        "path": "data/openwebtext-cooldown-700m-v001/pretrain_train.bin",
        "sha256": "9e2c31272706fe9a1bc43dc3e32ab24287892eb6e9de6302d8d815568af564c7",
        "size_bytes": 1_400_000_000, "token_count": 700_000_000,
        "dtype": "uint16",
    },
    "validation": {
        "path": "data/openwebtext-cooldown-700m-v001/pretrain_validation.bin",
        "sha256": "52085210f2da47797efd93401d8b890c6efdb41055c266d499649e1d82ee20be",
        "size_bytes": 33_554_432, "token_count": 16_777_216,
        "dtype": "uint16", "document_disjoint_from_train": True,
    },
    "validation_probe": {
        "path": "data/openwebtext-cooldown-700m-v001/validation_probe.npy",
        "sha256": "0295b25ce763f30570d4a268491d5a829e244ade33c602c32423e66aea278155",
        "context_count": 1_024, "dtype": "int64",
        "semantics": "absolute_validation_context_start_offsets",
        "one_context_per_document": True,
    },
}


class FullRPathConfigError(ValueError):
    pass


class FullRPathResultError(ValueError):
    pass


class FullRPathAbort(RuntimeError):
    pass


JsonDict = dict[str, Any]


def _canonical(value: Any) -> bytes:
    return (json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
        allow_nan=False,
    ) + "\n").encode("utf-8")


def _load_config(value: str | Path | Mapping[str, Any]) -> JsonDict:
    if isinstance(value, (str, Path)):
        with Path(value).expanduser().resolve().open("r", encoding="utf-8") as handle:
            loaded = json.load(handle)
    else:
        loaded = copy.deepcopy(dict(value))
    if type(loaded) is not dict:
        raise FullRPathConfigError("config must be a JSON object")
    loaded.pop("_config_hash", None)
    loaded["_config_hash"] = b0.canonical_hash(loaded)
    return loaded


def _raw_fraction(shape_id: str, source_update_index: int) -> float:
    if shape_id not in SHAPE_IDS:
        raise FullRPathConfigError(f"unknown shape {shape_id!r}")
    if type(source_update_index) is not int or not 0 <= source_update_index < CANONICAL_UPDATES:
        raise FullRPathConfigError("source update index is out of range")
    x = source_update_index / (CANONICAL_UPDATES - 1)
    if x < 0.8:
        return 1.0
    if shape_id == "wsd_exp_80_20":
        return math.exp(-math.log(10.0) * (x - 0.8) / 0.2)
    if x < 0.9:
        return 1.0 / math.sqrt(10.0)
    return 0.1


def _canonical_schedule(shape_id: str) -> JsonDict:
    """Build the source schedule, rescaling only its nonconstant tail."""

    raw = np.asarray(
        [_raw_fraction(shape_id, index) for index in range(CANONICAL_UPDATES)],
        dtype=np.float64,
    )
    if not np.array_equal(raw[:CONSTANT_SOURCE_UPDATES], np.ones(CONSTANT_SOURCE_UPDATES)):
        raise RuntimeError("canonical constant prefix changed")
    target_time = TERMINAL_INTRINSIC_TIMES[shape_id]
    constant_time = CONSTANT_SOURCE_UPDATES * CANONICAL_BASE_LEARNING_RATE
    derived_tail_scale = (target_time - constant_time) / (
        CANONICAL_BASE_LEARNING_RATE * float(np.sum(raw[CONSTANT_SOURCE_UPDATES:]))
    )
    tail_scale = TAIL_RESCALES[shape_id]
    if not math.isclose(derived_tail_scale, tail_scale, rel_tol=0.0, abs_tol=5e-15):
        raise RuntimeError("canonical tail normalization changed")
    fractions = raw.copy()
    fractions[CONSTANT_SOURCE_UPDATES:] *= tail_scale
    rates = CANONICAL_BASE_LEARNING_RATE * fractions
    boundaries = np.empty(CANONICAL_UPDATES + 1, dtype=np.float64)
    boundaries[: CONSTANT_SOURCE_UPDATES + 1] = (
        np.arange(CONSTANT_SOURCE_UPDATES + 1, dtype=np.float64)
        * CANONICAL_BASE_LEARNING_RATE
    )
    boundaries[CONSTANT_SOURCE_UPDATES + 1 :] = constant_time + np.cumsum(
        rates[CONSTANT_SOURCE_UPDATES:], dtype=np.float64
    )
    boundaries[-1] = target_time
    if not np.all(np.diff(boundaries) > 0.0):
        raise RuntimeError("canonical clock is not strictly increasing")
    return {
        "shape_id": shape_id, "tail_scale": float(tail_scale),
        "fractions": fractions, "learning_rates": rates,
        "intrinsic_boundaries": boundaries,
        "terminal_intrinsic_time": target_time,
    }


def _balanced_integer_path(shape_id: str) -> np.ndarray:
    """Invert the source clock on the 0.02 grid and round cumulative counts."""

    schedule = _canonical_schedule(shape_id)
    macro_count = MACRO_COUNTS[shape_id]
    boundaries = schedule["intrinsic_boundaries"]
    targets = np.arange(macro_count + 1, dtype=np.float64) * MACRO_INTRINSIC_TIME
    coordinates = np.interp(
        targets, boundaries, np.arange(CANONICAL_UPDATES + 1, dtype=np.float64)
    )
    cumulative = np.floor(coordinates + 0.5).astype(np.int64)
    # These values are exact consequences of the unscaled eta=.005 prefix.
    cumulative[: CONSTANT_MACROS + 1] = (
        np.arange(CONSTANT_MACROS + 1, dtype=np.int64) * 4
    )
    cumulative[0] = 0
    cumulative[-1] = CANONICAL_UPDATES
    g = np.diff(cumulative)
    if (
        g.shape != (macro_count,) or int(np.min(g)) < 1
        or int(np.sum(g)) != CANONICAL_UPDATES
        or not np.all(g[:CONSTANT_MACROS] == 4)
    ):
        raise RuntimeError("balanced integer schedule invariants failed")
    return np.ascontiguousarray(g, dtype=np.int64)


def _source_knots(shape_id: str) -> tuple[int, ...]:
    return (CONSTANT_SOURCE_UPDATES,) if shape_id == "wsd_exp_80_20" else (
        CONSTANT_SOURCE_UPDATES, 92_160,
    )


def _schedule_plan(shape_id: str) -> JsonDict:
    g = _balanced_integer_path(shape_id)
    cumulative = np.concatenate(
        [np.zeros(1, dtype=np.int64), np.cumsum(g, dtype=np.int64)]
    )
    source_knots = _source_knots(shape_id)
    knot_macros = [int(np.searchsorted(cumulative, knot)) for knot in source_knots]
    changed = np.flatnonzero(g != 4)
    schedule = _canonical_schedule(shape_id)
    return {
        "shape_id": shape_id,
        "macro_count": MACRO_COUNTS[shape_id],
        "terminal_intrinsic_time": TERMINAL_INTRINSIC_TIMES[shape_id],
        "tail_rescale": schedule["tail_scale"],
        "g": g,
        "g_sha256": cooldown._array_sha256(g),
        "g_minimum": int(np.min(g)), "g_maximum": int(np.max(g)),
        "g_sum": int(np.sum(g)),
        "cumulative_atomic_boundaries": cumulative,
        "source_knots": source_knots, "knot_macros": tuple(knot_macros),
        "first_nonconstant_ratio_macro": int(changed[0]),
    }


def schedule_manifest() -> JsonDict:
    shapes = []
    for shape_id in SHAPE_IDS:
        plan = _schedule_plan(shape_id)
        shapes.append({
            key: (
                value.tolist()
                if isinstance(value, np.ndarray)
                else list(value) if isinstance(value, tuple) else value
            )
            for key, value in plan.items()
            if key != "cumulative_atomic_boundaries"
        })
    unsigned = {
        "schema_version": "nanogpt30m_e2e_sgd_full_rpath_schedule_v001",
        "campaign_id": CAMPAIGN_ID,
        "canonical_updates": CANONICAL_UPDATES,
        "constant_source_updates": CONSTANT_SOURCE_UPDATES,
        "aligned_stride": ALIGNED_STRIDE,
        "atomic_batch_size": ATOMIC_BATCH_SIZE,
        "canonical_base_learning_rate": CANONICAL_BASE_LEARNING_RATE,
        "macro_intrinsic_time": MACRO_INTRINSIC_TIME,
        "shape_order": list(SHAPE_IDS), "shapes": shapes,
    }
    return {**unsigned, "sha256": b0.canonical_hash(unsigned)}


def _arm_plans() -> dict[str, JsonDict]:
    plans: dict[str, JsonDict] = {}
    for shape_id in SHAPE_IDS:
        schedule = _schedule_plan(shape_id)
        for realization_id in REALIZATION_IDS:
            arm_id = f"{shape_id}__{realization_id}"
            plans[arm_id] = {
                **schedule, "arm_id": arm_id, "realization_id": realization_id,
                "updates": (
                    CANONICAL_UPDATES if realization_id == "fixed_batch_lr"
                    else 4 * schedule["macro_count"]
                ),
                "contexts": TAPE_CONTEXTS,
                "tokens": TAPE_CONTEXTS * BLOCK_SIZE,
            }
    return plans


def _arm_plan(arm_id: str) -> JsonDict:
    try:
        return _arm_plans()[arm_id]
    except KeyError as error:
        raise FullRPathConfigError(f"unknown arm {arm_id!r}") from error


def _iter_arm_steps(plan: Mapping[str, Any]) -> Iterator[JsonDict]:
    """Yield updates using a flat context tape.

    ``fixed_batch_lr`` partitions a macro's ``4*g`` contexts into ``g``
    batches of four.  ``fixed_lr_batch`` partitions the identical flat block
    into four batches of ``g``.  Therefore both arms use the same contexts and
    execute the same r=200*g and dT=.02 in every macro.  For g=4 the update
    descriptors and tape slices are exactly identical.
    """

    cursor = 0
    update = 0
    intrinsic = 0.0
    fixed_batch = plan["realization_id"] == "fixed_batch_lr"
    for macro_index, raw_g in enumerate(plan["g"]):
        g = int(raw_g)
        count = g if fixed_batch else 4
        for within in range(count):
            batch_size = ATOMIC_BATCH_SIZE if fixed_batch else g
            learning_rate = (
                MACRO_INTRINSIC_TIME / g
                if fixed_batch else CANONICAL_BASE_LEARNING_RATE
            )
            update += 1
            start, stop = cursor, cursor + batch_size
            before = intrinsic
            macro_completed = within == count - 1
            intrinsic = (
                (macro_index + 1) * MACRO_INTRINSIC_TIME
                if macro_completed else intrinsic + learning_rate
            )
            cursor = stop
            yield {
                "update": update, "macro_index": macro_index,
                "within_macro_update": within, "g": g,
                "gradient_accumulation_steps": 1,
                "batch_size": batch_size,
                "learning_rate": learning_rate,
                "ratio_B_over_eta": EXECUTED_RATIO_UNIT * g,
                "intrinsic_time_before": before,
                "intrinsic_time_after": intrinsic,
                "tape_context_start": start, "tape_context_stop": stop,
                "tape_context_cursor": cursor,
                "macro_completed": macro_completed,
            }


def _evaluation_macros(shape_id: str) -> tuple[int, ...]:
    plan = _schedule_plan(shape_id)
    macro_count = int(plan["macro_count"])
    # Exact B4-r800 evaluation updates divided by four.  This preserves the
    # established validation grid without repeating its full evaluation cost.
    constant_r_macros = (
        0, 16, 32, 64, 128, 256, 512, 1_024, 2_048, 3_072,
        4_096, 5_120, 6_144, 7_168, 8_192, 9_216, 10_240, 11_264,
        12_288, 13_312, 14_336, 15_360, 16_384, 17_408, 18_432,
        19_456, 20_480, 21_504, 22_528, 23_552, 24_576, 25_600,
    )
    base = {value for value in constant_r_macros if value <= macro_count}
    base.update(range(CONSTANT_MACROS, macro_count + 1, 128))
    base.add(macro_count)
    anchors = {
        CONSTANT_MACROS, int(plan["first_nonconstant_ratio_macro"]),
        macro_count,
        *[int(value) for value in plan["knot_macros"]],
    }
    for anchor in anchors:
        for offset in (-64, -16, -4, -1, 0, 1, 4, 16, 64):
            if 0 <= anchor + offset <= macro_count:
                base.add(anchor + offset)
    return tuple(sorted(base))


def _make_training_tape(token_count: int, seed: int = TRAINING_TAPE_SEED) -> np.ndarray:
    eligible = cooldown._eligible_aligned_starts(int(token_count))
    if eligible < TAPE_CONTEXTS:
        raise FullRPathConfigError("training corpus is too small for the frozen tape")
    generator = np.random.Generator(np.random.PCG64(int(seed)))
    tape = np.ascontiguousarray(
        generator.permutation(eligible)[:TAPE_CONTEXTS] * ALIGNED_STRIDE,
        dtype=np.int64,
    )
    if np.unique(tape).size != TAPE_CONTEXTS:
        raise RuntimeError("training tape contains duplicate contexts")
    return tape


def _validate_config(config: Mapping[str, Any], arm_id: str | None = None) -> None:
    selected = config.get("arm_id")
    if selected not in ARM_IDS or (arm_id is not None and selected != arm_id):
        raise FullRPathConfigError("config arm identity changed")
    if (
        config.get("schema_version") != CONFIG_SCHEMA
        or config.get("campaign_id") != CAMPAIGN_ID
        or config.get("run_id") != RUN_ID_BY_ARM[selected]
        or config.get("primary_prediction_id") != PRIMARY_PREDICTION_ID
    ):
        raise FullRPathConfigError("config identity changed")
    if config.get("model") != EXPECTED_MODEL:
        raise FullRPathConfigError("model contract changed")
    optimizer = config.get("optimizer", {})
    expected_optimizer = {
        "name": "sgd", "momentum": 0.0, "dampening": 0.0,
        "nesterov": False, "weight_decay": 0.0, "gradient_clip": 0.0,
        "foreach": False,
    }
    if any(optimizer.get(key) != value for key, value in expected_optimizer.items()):
        raise FullRPathConfigError("plain-SGD contract changed")
    design = config.get("design", {})
    plan = _arm_plan(str(selected))
    expected_design = {
        "canonical_fixed_batch_updates": CANONICAL_UPDATES,
        "constant_source_updates": CONSTANT_SOURCE_UPDATES,
        "aligned_stride": ALIGNED_STRIDE,
        "atomic_batch_size": ATOMIC_BATCH_SIZE,
        "canonical_base_learning_rate": CANONICAL_BASE_LEARNING_RATE,
        "macro_intrinsic_time": MACRO_INTRINSIC_TIME,
        "macro_count": plan["macro_count"],
        "terminal_intrinsic_time": plan["terminal_intrinsic_time"],
        "initialization_seed": INITIALIZATION_SEED,
        "training_tape_seed": TRAINING_TAPE_SEED,
        "validation_batch_size": VALIDATION_BATCH_SIZE,
        "g_sha256": plan["g_sha256"],
    }
    if any(design.get(key) != value for key, value in expected_design.items()):
        raise FullRPathConfigError("schedule design contract changed")
    if tuple(design.get("evaluation_macros", ())) != _evaluation_macros(plan["shape_id"]):
        raise FullRPathConfigError("evaluation grid changed")
    data = config.get("data", {})
    for role, expected in EXPECTED_DATA.items():
        actual = data.get(role, {})
        if any(actual.get(key) != value for key, value in expected.items()):
            raise FullRPathConfigError(f"{role} data contract changed")
    allocation = data.get("training_context_allocation", {})
    if (
        allocation.get("contexts") != TAPE_CONTEXTS
        or allocation.get("aligned_stride") != ALIGNED_STRIDE
        or allocation.get("offsets_sha256") != TAPE_OFFSETS_SHA256
        or allocation.get("within_arm_context_reuse") is not False
    ):
        raise FullRPathConfigError("training tape contract changed")
    runtime = config.get("runtime", {})
    for key, value in {
        "training_device": "cuda", "dtype": "float32", "amp": False,
        "tf32": False, "compile": False, "deterministic": True,
    }.items():
        if runtime.get(key) != value:
            raise FullRPathConfigError(f"runtime.{key} changed")
    if config.get("artifacts", {}).get("checkpoint_policy") != "none":
        raise FullRPathConfigError("checkpoint/resume is outside this experiment")


def _prepare_inputs(config: Mapping[str, Any]):
    metadata_spec = config["data"]["metadata"]
    metadata_path = b0.resolve_path(str(metadata_spec["path"]))
    if not metadata_path.is_file():
        raise FileNotFoundError(f"dataset metadata is absent: {metadata_path}")
    metadata_sha256 = b0.sha256_file(metadata_path)
    if metadata_sha256 != metadata_spec["sha256"]:
        raise FullRPathConfigError("dataset metadata SHA256 changed")
    train, train_info = cooldown._load_uint16_stream(config["data"]["train"], "train")
    validation, validation_info = cooldown._load_uint16_stream(
        config["data"]["validation"], "validation"
    )
    probe, probe_info = cooldown._load_validation_probe(
        config["data"]["validation_probe"], len(validation)
    )
    if int(np.max(train)) >= EXPECTED_MODEL["vocab_size"]:
        raise FullRPathConfigError("training token exceeds vocabulary")
    if int(np.max(validation)) >= EXPECTED_MODEL["vocab_size"]:
        raise FullRPathConfigError("validation token exceeds vocabulary")
    tape = _make_training_tape(len(train), TRAINING_TAPE_SEED)
    if cooldown._array_sha256(tape) != TAPE_OFFSETS_SHA256:
        raise FullRPathConfigError("generated training tape hash changed")
    for info, role in (
        (train_info, "train"), (validation_info, "validation"),
        (probe_info, "validation_probe"),
    ):
        info["path"] = str(config["data"][role]["path"])
    return train, validation, probe, tape, {
        "metadata": {
            "path": str(metadata_spec["path"]), "sha256": metadata_sha256,
            "bytes": int(metadata_path.stat().st_size),
        },
        "train": train_info, "validation": validation_info,
        "validation_probe": probe_info,
    }


def _expected_trace_coordinates(plan: Mapping[str, Any]) -> dict[str, np.ndarray]:
    steps = list(_iter_arm_steps(plan))
    integer_columns = {
        "update": np.int32, "macro_index": np.int32,
        "within_macro_update": np.int16, "g": np.int16,
        "batch_size": np.int16, "tape_context_cursor": np.int32,
    }
    columns = {
        name: np.asarray([row[name] for row in steps], dtype=dtype)
        for name, dtype in integer_columns.items()
    }
    for name in ("learning_rate", "ratio_B_over_eta", "intrinsic_time_after"):
        columns[name] = np.asarray([row[name] for row in steps], dtype=np.float64)
    columns["examples_seen"] = columns["tape_context_cursor"].astype(np.int64)
    columns["tokens_seen"] = columns["examples_seen"] * BLOCK_SIZE
    return columns


def _trace_payload(arm_id: str, columns: Mapping[str, np.ndarray]) -> JsonDict:
    unsigned = {
        "schema_version": "nanogpt30m_e2e_sgd_full_rpath_training_trace_v001",
        "arm_id": arm_id, "row_count": len(next(iter(columns.values()), [])),
        "column_order": list(columns),
        "columns": {key: cooldown._encode_array(value) for key, value in columns.items()},
    }
    return {**unsigned, "sha256": b0.canonical_hash(unsigned)}


def _evaluation_row(plan: Mapping[str, Any], step: Mapping[str, Any] | None,
                    macro: int, losses: np.ndarray) -> JsonDict:
    values = np.asarray(losses, dtype=np.float64)
    cursor = 0 if step is None else int(step["tape_context_cursor"])
    update = 0 if step is None else int(step["update"])
    # The loss is measured after ``macro`` bins have completed, so label it
    # with the last executed bin (tick zero uses the first scheduled bin).
    g_index = min(max(macro - 1, 0), int(plan["macro_count"]) - 1)
    g = int(plan["g"][g_index])
    return {
        "arm_id": plan["arm_id"], "shape_id": plan["shape_id"],
        "realization_id": plan["realization_id"], "macro_index": macro,
        "update": update, "tape_context_cursor": cursor,
        "examples_seen": cursor,
        "tokens_seen": cursor * BLOCK_SIZE,
        "intrinsic_time": macro * MACRO_INTRINSIC_TIME,
        "g": g, "ratio_B_over_eta": EXECUTED_RATIO_UNIT * g,
        "validation_cross_entropy": float(np.mean(values)),
        "validation_standard_error": float(np.std(values, ddof=1) / math.sqrt(len(values))),
        "context_cross_entropies_sha256": cooldown._array_sha256(
            np.ascontiguousarray(losses, dtype=np.float32)
        ),
    }


def _evaluation_payload(arm_id: str, rows: Sequence[Mapping[str, Any]],
                        contexts: Sequence[np.ndarray]) -> JsonDict:
    matrix = np.ascontiguousarray(np.stack(contexts), dtype=np.float32)
    unsigned = {
        "schema_version": "nanogpt30m_e2e_sgd_full_rpath_evaluations_v001",
        "arm_id": arm_id, "row_count": len(rows),
        "context_count": VALIDATION_CONTEXTS, "rows": [dict(row) for row in rows],
        "context_cross_entropies": cooldown._encode_array(matrix),
    }
    return {**unsigned, "sha256": b0.canonical_hash(unsigned)}


def _training_step(model: torch.nn.Module, optimizer: torch.optim.Optimizer,
                   train: np.ndarray, offsets: np.ndarray, device: torch.device,
                   learning_rate: float) -> float:
    """One direct-batch plain-SGD update without per-step global norm scans."""

    if (
        offsets.ndim != 1 or not ATOMIC_BATCH_SIZE <= len(offsets) <= 40
        or not math.isfinite(learning_rate) or learning_rate <= 0.0
    ):
        raise FullRPathAbort("training step descriptor is invalid")
    cooldown._set_learning_rate(optimizer, float(learning_rate))
    model.train()
    optimizer.zero_grad(set_to_none=True)
    x, y = b0._batch_from_offsets(train, offsets, BLOCK_SIZE, device)
    _, loss, _ = model(x, y)
    if loss is None or not bool(torch.isfinite(loss).detach().cpu()):
        raise FullRPathAbort("nonfinite_training_loss")
    loss.backward()
    optimizer.step()
    if optimizer.state_dict()["state"]:
        raise FullRPathAbort("plain SGD unexpectedly acquired optimizer state")
    value = float(loss.detach().cpu())
    if value < 0.0:
        raise FullRPathAbort("negative_training_cross_entropy")
    return value


def _run_arm(config: Mapping[str, Any], plan: Mapping[str, Any], train: np.ndarray,
             validation: np.ndarray, probe: np.ndarray, tape: np.ndarray,
             device: torch.device, deadline: float, progress_callback=None):
    model = cooldown._new_model(config, device)
    initialization_sha256 = b0._model_initialization_sha256(model)
    if initialization_sha256 != EXPECTED_INITIALIZATION_SHA256:
        raise FullRPathAbort("frozen model initialization hash changed")
    optimizer = b0._build_optimizer(model, CANONICAL_BASE_LEARNING_RATE)
    expected = _expected_trace_coordinates(plan)
    losses = np.empty(int(plan["updates"]), dtype=np.float32)
    eval_set = set(_evaluation_macros(str(plan["shape_id"])))
    rows: list[JsonDict] = []
    context_losses: list[np.ndarray] = []
    initial = cooldown._evaluate_contexts(model, validation, probe, device, VALIDATION_BATCH_SIZE)
    rows.append(_evaluation_row(plan, None, 0, initial))
    context_losses.append(initial)
    if rows[0]["context_cross_entropies_sha256"] != EXPECTED_TICK0_VALIDATION_SHA256:
        raise FullRPathAbort("frozen tick-0 validation hash changed")
    step_times: list[float] = []
    completed = 0
    last_step: JsonDict | None = None
    for step in _iter_arm_steps(plan):
        cooldown._check_deadline(deadline)
        start, stop = int(step["tape_context_start"]), int(step["tape_context_stop"])
        offsets = tape[start:stop]
        b0._synchronize(device)
        started = time.monotonic()
        train_ce = _training_step(
            model, optimizer, train, offsets, device, float(step["learning_rate"]),
        )
        b0._synchronize(device)
        step_times.append(time.monotonic() - started)
        completed += 1
        losses[completed - 1] = train_ce
        last_step = step
        if bool(step["macro_completed"]):
            macro = int(step["macro_index"]) + 1
            if macro in eval_set:
                current = cooldown._evaluate_contexts(
                    model, validation, probe, device, VALIDATION_BATCH_SIZE
                )
                row = _evaluation_row(plan, step, macro, current)
                rows.append(row)
                context_losses.append(current)
                if (
                    macro == CONSTANT_MACROS
                    and (
                        row["context_cross_entropies_sha256"]
                        != EXPECTED_PREFIX_END_VALIDATION_SHA256
                        or row["validation_cross_entropy"]
                        != EXPECTED_PREFIX_END_VALIDATION_CE
                    )
                ):
                    raise FullRPathAbort("historical r800 prefix replay hash changed")
        if completed % 256 == 0 and progress_callback is not None:
            progress_callback({
                "kind": "progress", "arm_id": plan["arm_id"],
                "completed_updates": completed, "expected_updates": plan["updates"],
            })
    if completed != plan["updates"] or last_step is None:
        raise RuntimeError("arm did not complete its schedule")
    if int(last_step["tape_context_cursor"]) != TAPE_CONTEXTS:
        raise RuntimeError("arm did not consume the full paired tape")
    if not bool(np.all(np.isfinite(losses))):
        raise FullRPathAbort("training cross-entropy became nonfinite")
    if optimizer.state_dict()["state"]:
        raise FullRPathAbort("plain SGD unexpectedly acquired optimizer state")
    trace = {**expected, "training_cross_entropy": losses}
    diagnostics = {
        "initialization_sha256": initialization_sha256,
        "tick0_validation_context_cross_entropies_sha256": rows[0]["context_cross_entropies_sha256"],
        "prefix_end_validation_context_cross_entropies_sha256": next(
            row["context_cross_entropies_sha256"]
            for row in rows if row["macro_index"] == CONSTANT_MACROS
        ),
        "optimizer_state_entries": len(optimizer.state_dict()["state"]),
        "step_time_median_seconds": statistics.median(step_times),
        "step_time_p90_seconds": float(np.percentile(step_times, 90)),
        "all_training_cross_entropies_finite": bool(np.all(np.isfinite(losses))),
    }
    del optimizer, model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return trace, rows, context_losses, diagnostics


def preflight(config_or_path: str | Path | Mapping[str, Any], arm_id: str | None = None) -> JsonDict:
    config = _load_config(config_or_path)
    _validate_config(config, arm_id)
    selected = str(config["arm_id"])
    plan = _arm_plan(selected)
    tape = _make_training_tape(EXPECTED_DATA["train"]["token_count"])
    steps = _expected_trace_coordinates(plan)
    checks = {
        "schedule_invariants": (
            plan["g_sum"] == CANONICAL_UPDATES and plan["g_minimum"] >= 1
            and plan["g_maximum"] == 40
            and bool(np.all(plan["g"][:CONSTANT_MACROS] == 4))
        ),
        "paired_tape_hash_exact": cooldown._array_sha256(tape) == TAPE_OFFSETS_SHA256,
        "terminal_coordinates_exact": (
            int(steps["tape_context_cursor"][-1]) == TAPE_CONTEXTS
            and math.isclose(float(steps["intrinsic_time_after"][-1]),
                             plan["terminal_intrinsic_time"], abs_tol=1e-9)
        ),
        "no_checkpoint_policy": config["artifacts"]["checkpoint_policy"] == "none",
    }
    return {
        "schema_version": "nanogpt30m_e2e_sgd_full_rpath_preflight_v001",
        "campaign_id": CAMPAIGN_ID, "run_id": config["run_id"],
        "arm_id": selected, "status": "passed" if all(checks.values()) else "failed",
        "checks": checks, "scientific_training_executed": False,
        "gpu_submission_performed": False,
    }


def _artifact_path(config: Mapping[str, Any], output_dir: str | Path | None) -> Path:
    root = (Path(output_dir).expanduser().resolve() if output_dir is not None else
            b0.resolve_path(str(config["artifacts"]["output_dir"])))
    return root / str(config["artifacts"].get("result", "result.json"))


def _failure(error: BaseException) -> JsonDict:
    return {"type": type(error).__name__, "message": str(error),
            "traceback_tail": traceback.format_exception(error)[-8:]}


def run_experiment(config_or_path: str | Path | Mapping[str, Any],
                   output_dir: str | Path | None = None, arm_id: str | None = None,
                   progress_callback=None) -> JsonDict:
    if arm_id is None:
        raise FullRPathConfigError("one explicit arm_id is required")
    started = time.monotonic()
    config = _load_config(config_or_path)
    _validate_config(config, arm_id)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    cooldown._configure_deterministic_cuda()
    plan = _arm_plan(arm_id)
    train, validation, probe, tape, data_info = _prepare_inputs(config)
    device = b0.resolve_device(str(config["runtime"]["training_device"]))
    if device.type != "cuda":
        raise FullRPathConfigError("scientific execution requires CUDA")
    deadline = started + float(config["runtime"]["maximum_wall_seconds"])
    result: JsonDict = {
        "schema_version": RESULT_SCHEMA, "campaign_id": CAMPAIGN_ID,
        "run_id": config["run_id"], "arm_id": arm_id,
        "status": "incomplete", "decision": "INCOMPLETE",
        "gpu_submission_performed": False,
        "config_sha256": config["_config_hash"], "schedule": schedule_manifest(),
        "data": data_info, "model": EXPECTED_MODEL,
        "optimizer": {"name": "sgd", "momentum": 0.0, "weight_decay": 0.0},
        "counts": {"expected_updates": plan["updates"],
                   "expected_contexts": TAPE_CONTEXTS},
        "measurements": {"training_trace": None, "evaluations": None},
        "diagnostics": {}, "runtime": {}, "failure": None,
    }
    try:
        trace, rows, contexts, diagnostics = _run_arm(
            config, plan, train, validation, probe, tape, device, deadline,
            progress_callback,
        )
        result["measurements"] = {
            "training_trace": _trace_payload(arm_id, trace),
            "evaluations": _evaluation_payload(arm_id, rows, contexts),
        }
        result["diagnostics"] = diagnostics
        result["counts"].update({
            "actual_updates": len(trace["training_cross_entropy"]),
            "actual_contexts": int(trace["examples_seen"][-1]),
            "actual_evaluation_rows": len(rows),
        })
        result["status"], result["decision"] = "completed", "ARM_COMPLETED"
    except Exception as error:
        result["failure"] = _failure(error)
    elapsed = time.monotonic() - started
    result["runtime"] = {
        "device": str(device), "device_name": torch.cuda.get_device_name(device),
        "dtype": "float32", "tf32": False, "elapsed_seconds": elapsed,
        "actual_gpu_hours": elapsed / 3_600.0,
        "maximum_wall_seconds": config["runtime"]["maximum_wall_seconds"],
        "resource_budget_passed": (
            elapsed <= float(config["runtime"]["maximum_wall_seconds"])
            and elapsed / 3_600.0 <= float(config["runtime"]["maximum_a100_gpu_hours"])
        ),
    }
    path = _artifact_path(config, output_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    b0._atomic_strict_json(path, result)
    validate_result(config, result, require_complete=result["status"] == "completed")
    return result


def _validate_hashed(record: Mapping[str, Any], label: str) -> None:
    unsigned = dict(record)
    observed = unsigned.pop("sha256", None)
    if observed != b0.canonical_hash(unsigned):
        raise FullRPathResultError(f"{label} hash changed")


def validate_result(config_or_path: str | Path | Mapping[str, Any],
                    result_or_path: str | Path | Mapping[str, Any], *,
                    require_complete: bool = True) -> JsonDict:
    config = _load_config(config_or_path)
    _validate_config(config)
    if isinstance(result_or_path, (str, Path)):
        with Path(result_or_path).expanduser().resolve().open("r", encoding="utf-8") as handle:
            result = json.load(handle)
    else:
        result = copy.deepcopy(dict(result_or_path))
    arm_id = str(config["arm_id"])
    plan = _arm_plan(arm_id)
    if (
        result.get("schema_version") != RESULT_SCHEMA
        or result.get("campaign_id") != CAMPAIGN_ID
        or result.get("run_id") != RUN_ID_BY_ARM[arm_id]
        or result.get("arm_id") != arm_id
        or result.get("config_sha256") != config["_config_hash"]
        or result.get("gpu_submission_performed") is not False
    ):
        raise FullRPathResultError("result identity changed")
    if result.get("schedule") != schedule_manifest():
        raise FullRPathResultError("embedded schedule manifest changed")
    if result.get("model") != EXPECTED_MODEL or result.get("optimizer") != {
        "name": "sgd", "momentum": 0.0, "weight_decay": 0.0,
    }:
        raise FullRPathResultError("model or optimizer result contract changed")
    data = result.get("data")
    if not isinstance(data, Mapping):
        raise FullRPathResultError("data audit records are absent")
    for role, expected in EXPECTED_DATA.items():
        actual = data.get(role)
        if not isinstance(actual, Mapping) or actual.get("sha256") != expected["sha256"]:
            raise FullRPathResultError(f"{role} data hash changed")
        if role in ("train", "validation") and (
            actual.get("token_count") != expected["token_count"]
            or actual.get("bytes") != expected["size_bytes"]
        ):
            raise FullRPathResultError(f"{role} data accounting changed")
        if role == "validation_probe" and (
            actual.get("context_count") != VALIDATION_CONTEXTS
            or actual.get("dtype") != "int64"
        ):
            raise FullRPathResultError("validation probe accounting changed")
    status = result.get("status")
    if status not in ("completed", "incomplete"):
        raise FullRPathResultError("result status changed")
    if status != "completed":
        if require_complete:
            raise FullRPathResultError("result is incomplete")
        if result.get("decision") != "INCOMPLETE" or not isinstance(
            result.get("failure"), Mapping
        ):
            raise FullRPathResultError("incomplete result lacks a failure witness")
        return {"schema_version": VALIDATION_SCHEMA, "run_id": result.get("run_id"),
                "arm_id": arm_id, "status": "incomplete_validated"}
    if result.get("decision") != "ARM_COMPLETED" or result.get("failure") is not None:
        raise FullRPathResultError("completed result has an inconsistent outcome")
    measurements = result.get("measurements")
    if not isinstance(measurements, Mapping):
        raise FullRPathResultError("result measurements are absent")
    trace = measurements.get("training_trace")
    evaluations = measurements.get("evaluations")
    if not isinstance(trace, Mapping) or not isinstance(evaluations, Mapping):
        raise FullRPathResultError("embedded trace or evaluations are absent")
    _validate_hashed(trace, "training trace")
    _validate_hashed(evaluations, "evaluations")
    decoded = {
        key: cooldown._decode_array(value, key)
        for key, value in trace.get("columns", {}).items()
    }
    expected = _expected_trace_coordinates(plan)
    expected_order = [*expected, "training_cross_entropy"]
    if (
        trace.get("schema_version")
        != "nanogpt30m_e2e_sgd_full_rpath_training_trace_v001"
        or trace.get("arm_id") != arm_id
        or trace.get("row_count") != plan["updates"]
        or trace.get("column_order") != expected_order
    ):
        raise FullRPathResultError("training trace schema changed")
    if set(decoded) != {*expected, "training_cross_entropy"}:
        raise FullRPathResultError("training trace columns changed")
    for key, values in expected.items():
        if decoded[key].dtype != values.dtype or not np.array_equal(decoded[key], values):
            raise FullRPathResultError(f"training coordinate changed: {key}")
    training_ce = decoded["training_cross_entropy"]
    if (
        training_ce.dtype != np.dtype(np.float32)
        or training_ce.shape != (plan["updates"],)
        or not bool(np.all(np.isfinite(training_ce)))
        or bool(np.any(training_ce < 0.0))
    ):
        raise FullRPathResultError("training CE trace is invalid")
    rows = evaluations.get("rows")
    if not isinstance(rows, list):
        raise FullRPathResultError("evaluation rows must be a list")
    matrix = cooldown._decode_array(
        evaluations.get("context_cross_entropies", {}), "validation contexts"
    )
    expected_macros = _evaluation_macros(plan["shape_id"])
    if (
        evaluations.get("schema_version")
        != "nanogpt30m_e2e_sgd_full_rpath_evaluations_v001"
        or evaluations.get("arm_id") != arm_id
        or evaluations.get("row_count") != len(expected_macros)
        or evaluations.get("context_count") != VALIDATION_CONTEXTS
        or len(rows) != len(expected_macros)
        or matrix.dtype != np.dtype(np.float32)
        or matrix.shape != (len(expected_macros), VALIDATION_CONTEXTS)
        or not bool(np.all(np.isfinite(matrix)))
        or bool(np.any(matrix < 0.0))
    ):
        raise FullRPathResultError("validation payload changed")

    wanted = set(expected_macros)
    endpoint_steps: dict[int, JsonDict | None] = {0: None}
    for step in _iter_arm_steps(plan):
        macro = int(step["macro_index"]) + 1
        if step["macro_completed"] and macro in wanted:
            endpoint_steps[macro] = step
    for index, (row, macro) in enumerate(zip(rows, expected_macros, strict=True)):
        if not isinstance(row, Mapping) or row.get("macro_index") != macro:
            raise FullRPathResultError("validation macro grid changed")
        step = endpoint_steps[macro]
        cursor = 0 if step is None else int(step["tape_context_cursor"])
        update = 0 if step is None else int(step["update"])
        g_index = min(max(macro - 1, 0), int(plan["macro_count"]) - 1)
        g = int(plan["g"][g_index])
        expected_coordinates = {
            "arm_id": arm_id, "shape_id": plan["shape_id"],
            "realization_id": plan["realization_id"], "macro_index": macro,
            "update": update, "tape_context_cursor": cursor,
            "examples_seen": cursor, "tokens_seen": cursor * BLOCK_SIZE,
            "intrinsic_time": macro * MACRO_INTRINSIC_TIME,
            "g": g, "ratio_B_over_eta": EXECUTED_RATIO_UNIT * g,
        }
        if any(row.get(key) != value for key, value in expected_coordinates.items()):
            raise FullRPathResultError("validation operational coordinates changed")
        values = matrix[index].astype(np.float64)
        if row.get("context_cross_entropies_sha256") != cooldown._array_sha256(
            matrix[index]
        ):
            raise FullRPathResultError("per-context validation hash changed")
        if row.get("validation_cross_entropy") != float(np.mean(values)):
            raise FullRPathResultError("validation mean is not reproducible")
        expected_se = float(np.std(values, ddof=1) / math.sqrt(len(values)))
        if row.get("validation_standard_error") != expected_se:
            raise FullRPathResultError("validation standard error is not reproducible")
        if macro == CONSTANT_MACROS and (
            row.get("context_cross_entropies_sha256")
            != EXPECTED_PREFIX_END_VALIDATION_SHA256
            or row.get("validation_cross_entropy")
            != EXPECTED_PREFIX_END_VALIDATION_CE
        ):
            raise FullRPathResultError("historical r800 prefix replay identity changed")

    expected_counts = {
        "expected_updates": plan["updates"], "expected_contexts": TAPE_CONTEXTS,
        "actual_updates": plan["updates"], "actual_contexts": TAPE_CONTEXTS,
        "actual_evaluation_rows": len(expected_macros),
    }
    if result.get("counts") != expected_counts:
        raise FullRPathResultError("completed arm accounting changed")
    diagnostics = result.get("diagnostics")
    if not isinstance(diagnostics, Mapping):
        raise FullRPathResultError("stability diagnostics are absent")
    if (
        diagnostics.get("initialization_sha256") != EXPECTED_INITIALIZATION_SHA256
        or diagnostics.get("tick0_validation_context_cross_entropies_sha256")
        != EXPECTED_TICK0_VALIDATION_SHA256
        or diagnostics.get(
            "prefix_end_validation_context_cross_entropies_sha256"
        ) != EXPECTED_PREFIX_END_VALIDATION_SHA256
        or rows[0].get("context_cross_entropies_sha256")
        != EXPECTED_TICK0_VALIDATION_SHA256
        or diagnostics.get("optimizer_state_entries") != 0
        or diagnostics.get("all_training_cross_entropies_finite") is not True
    ):
        raise FullRPathResultError("completed arm lacks frozen stability identities")
    for key in ("step_time_median_seconds", "step_time_p90_seconds"):
        value = diagnostics.get(key)
        if (
            type(value) not in (int, float) or type(value) is bool
            or not math.isfinite(float(value)) or float(value) < 0.0
        ):
            raise FullRPathResultError("step timing diagnostic is invalid")
    runtime = result.get("runtime")
    if not isinstance(runtime, Mapping):
        raise FullRPathResultError("runtime accounting is absent")
    elapsed = runtime.get("elapsed_seconds")
    gpu_hours = runtime.get("actual_gpu_hours")
    if (
        runtime.get("device") != config["runtime"]["training_device"]
        or type(runtime.get("device_name")) is not str
        or not runtime.get("device_name")
        or runtime.get("dtype") != "float32"
        or runtime.get("tf32") is not False
        or type(elapsed) not in (int, float) or type(elapsed) is bool
        or type(gpu_hours) not in (int, float) or type(gpu_hours) is bool
        or not math.isfinite(float(elapsed)) or float(elapsed) < 0.0
        or not math.isclose(
            float(gpu_hours), float(elapsed) / 3_600.0,
            rel_tol=0.0, abs_tol=1e-12,
        )
        or runtime.get("maximum_wall_seconds")
        != config["runtime"]["maximum_wall_seconds"]
    ):
        raise FullRPathResultError("runtime accounting changed")
    expected_budget = bool(
        float(elapsed) <= float(config["runtime"]["maximum_wall_seconds"])
        and float(gpu_hours)
        <= float(config["runtime"]["maximum_a100_gpu_hours"])
    )
    if runtime.get("resource_budget_passed") is not expected_budget:
        raise FullRPathResultError("runtime budget diagnostic changed")
    return {"schema_version": VALIDATION_SCHEMA, "run_id": result["run_id"],
            "arm_id": arm_id, "status": "passed"}


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--arm-id", choices=ARM_IDS, required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--preflight", action="store_true")
    mode.add_argument("--run", action="store_true")
    return parser


def main() -> int:
    args = build_argument_parser().parse_args()
    value = (preflight(args.config, args.arm_id) if args.preflight else
             run_experiment(args.config, args.output_dir, args.arm_id))
    print(json.dumps(value, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "ARM_IDS", "CAMPAIGN_ID", "CONFIG_SCHEMA", "RUN_ID_BY_ARM",
    "SHAPE_IDS", "REALIZATION_IDS", "_arm_plan", "_arm_plans",
    "_balanced_integer_path", "_canonical_schedule", "_evaluation_macros",
    "_iter_arm_steps", "_make_training_tape", "preflight", "run_experiment",
    "schedule_manifest", "validate_result",
]
