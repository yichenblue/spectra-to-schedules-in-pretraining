"""300M hybrid-Muon WSD/8-1-1 tails matched by B/eta^2."""

from __future__ import annotations

import argparse
from functools import lru_cache
import hashlib
import json
import math
from pathlib import Path
import time
from typing import Any, Iterator, Mapping, Sequence

import numpy as np
import torch

from . import e2e_300m_b0_v001 as b0
from . import e2e_300m_6p5b_plan_v001 as scale
from . import e2e_300m_muon_prefix_v001 as prefix
from .muon_300m_v001 import HybridMuon, optimizer_state_fp32_and_finite


RUN_ID = "nanogpt300m-e2e-muon-square-rpath-v001"
CONFIG_SCHEMA = "nanogpt300m_e2e_muon_square_rpath_config_v001"
RESULT_SCHEMA = "nanogpt300m_e2e_muon_square_rpath_result_v001"
VALIDATION_SCHEMA = "nanogpt300m_e2e_muon_square_rpath_validation_v001"
PREFIX_CHECKPOINT_SCHEMA = "nanogpt300m_e2e_muon_prefix_checkpoint_v001"
SHAPE_IDS = ("wsd_exp_80_20", "eight_one_one")
REALIZATION_IDS = ("fixed_batch_lr", "fixed_lr_batch")
ARM_IDS = tuple(f"{shape}__{realization}" for shape in SHAPE_IDS for realization in REALIZATION_IDS)

BASE_BATCH = 256
MICRO_BATCH = 32
BASE_LR = 1.0e-4
GRADIENT_CLIP = 33.504
CONTEXT_TOKENS = 256
CANONICAL_UPDATES = 99_183
PREFIX_UPDATES = 79_346
TAIL_SOURCE_UPDATES = CANONICAL_UPDATES - PREFIX_UPDATES
PREFIX_CONTEXTS = PREFIX_UPDATES * BASE_BATCH
TAIL_CONTEXTS = TAIL_SOURCE_UPDATES * BASE_BATCH


class SquareRPathError(RuntimeError):
    """A frozen square-ratio schedule or artifact contract changed."""


def _root() -> Path:
    return Path(__file__).resolve().parents[2]


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if type(value) is not dict:
        raise SquareRPathError(f"JSON root is not an object: {path}")
    return value


def _array_sha256(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def _prefix_sum_eta_squared() -> float:
    warmup = np.arange(1, prefix.WARMUP_UPDATES + 1, dtype=np.float64) / prefix.WARMUP_UPDATES
    return BASE_LR * BASE_LR * (
        float(np.sum(np.square(warmup), dtype=np.float64))
        + PREFIX_UPDATES - prefix.WARMUP_UPDATES
    )


def _raw_lr_fractions(shape_id: str) -> np.ndarray:
    coordinates = np.arange(PREFIX_UPDATES, CANONICAL_UPDATES, dtype=np.float64) / (CANONICAL_UPDATES - 1)
    if shape_id == "wsd_exp_80_20":
        return np.exp(-math.log(10.0) * (coordinates - 0.8) / 0.2)
    if shape_id == "eight_one_one":
        return np.where(coordinates < 0.9, 1.0 / math.sqrt(10.0), 0.1)
    raise SquareRPathError(f"unknown schedule shape: {shape_id}")


@lru_cache(maxsize=None)
def schedule_plan(shape_id: str) -> dict[str, Any]:
    fractions = _raw_lr_fractions(shape_id)
    squared = np.square(fractions)
    macro_count = int(math.ceil(float(np.sum(squared, dtype=np.float64))))
    rescale = macro_count / float(np.sum(squared, dtype=np.float64))
    boundaries = np.empty(TAIL_SOURCE_UPDATES + 1, dtype=np.float64)
    boundaries[0] = 0.0
    np.cumsum(squared * rescale, dtype=np.float64, out=boundaries[1:])
    boundaries[-1] = float(macro_count)
    cumulative = np.floor(
        np.interp(
            np.arange(macro_count + 1, dtype=np.float64),
            boundaries,
            np.arange(TAIL_SOURCE_UPDATES + 1, dtype=np.float64),
        )
        + 0.5
    ).astype(np.int64)
    cumulative[0] = 0
    cumulative[-1] = TAIL_SOURCE_UPDATES
    h = np.ascontiguousarray(np.diff(cumulative), dtype=np.int16)
    if (
        h.shape != (macro_count,)
        or int(np.min(h)) < 1
        or int(np.max(h)) > 100
        or int(np.sum(h, dtype=np.int64)) != TAIL_SOURCE_UPDATES
    ):
        raise SquareRPathError("squared-clock inverse schedule integerization failed")
    if shape_id == "wsd_exp_80_20":
        source_knots = (0, TAIL_SOURCE_UPDATES)
    else:
        coordinates = np.arange(PREFIX_UPDATES, CANONICAL_UPDATES, dtype=np.float64) / (CANONICAL_UPDATES - 1)
        source_knots = (0, int(np.searchsorted(coordinates, 0.9, side="left")), TAIL_SOURCE_UPDATES)
    knot_macros = tuple(int(np.searchsorted(cumulative, knot, side="left")) for knot in source_knots)
    return {
        "shape_id": shape_id,
        "macro_count": macro_count,
        "squared_clock_rescale": rescale,
        "h": h,
        "h_sha256": _array_sha256(h),
        "h_minimum": int(np.min(h)),
        "h_maximum": int(np.max(h)),
        "h_sum": int(np.sum(h, dtype=np.int64)),
        "source_knots": source_knots,
        "knot_macros": knot_macros,
    }


def arm_plan(arm_id: str) -> dict[str, Any]:
    if arm_id not in ARM_IDS:
        raise SquareRPathError(f"unknown arm: {arm_id}")
    shape_id, realization_id = arm_id.split("__", 1)
    plan = schedule_plan(shape_id)
    updates = TAIL_SOURCE_UPDATES if realization_id == "fixed_batch_lr" else int(plan["macro_count"])
    serialized_plan = {key: value for key, value in plan.items() if key != "h"}
    serialized_plan["source_knots"] = list(plan["source_knots"])
    serialized_plan["knot_macros"] = list(plan["knot_macros"])
    return {
        **serialized_plan,
        "arm_id": arm_id,
        "realization_id": realization_id,
        "optimizer_updates": updates,
        "tail_contexts": TAIL_CONTEXTS,
        "tail_tokens": TAIL_CONTEXTS * CONTEXT_TOKENS,
        "validation_samples": int(plan["macro_count"]) + 1,
    }


def iter_tail_steps(arm_id: str) -> Iterator[dict[str, Any]]:
    plan = arm_plan(arm_id)
    h_values = schedule_plan(plan["shape_id"])["h"]
    fixed_batch = plan["realization_id"] == "fixed_batch_lr"
    cursor = PREFIX_CONTEXTS
    optimizer_update = PREFIX_UPDATES
    sum_eta = prefix._intrinsic_time()
    sum_eta_squared = _prefix_sum_eta_squared()
    for macro_index, raw_h in enumerate(h_values):
        h = int(raw_h)
        steps = h if fixed_batch else 1
        batch = BASE_BATCH if fixed_batch else BASE_BATCH * h
        eta = BASE_LR / math.sqrt(h) if fixed_batch else BASE_LR
        for within_macro in range(steps):
            start = cursor
            optimizer_update += 1
            cursor += batch
            sum_eta += eta
            sum_eta_squared += eta * eta
            yield {
                "macro_index": macro_index,
                "within_macro_update": within_macro,
                "optimizer_update": optimizer_update,
                "h": h,
                "batch_size": batch,
                "learning_rate": eta,
                "ratio_B_over_eta_squared": batch / (eta * eta),
                "sum_eta_after": sum_eta,
                "sum_eta_squared_after": sum_eta_squared,
                "tape_context_start": start,
                "tape_context_stop": cursor,
                "macro_completed": within_macro == steps - 1,
            }


def _verify_arm(arm_id: str) -> dict[str, Any]:
    plan = arm_plan(arm_id)
    cursor = PREFIX_CONTEXTS
    updates = 0
    macros = 0
    terminal_s2 = _prefix_sum_eta_squared()
    for step in iter_tail_steps(arm_id):
        if step["tape_context_start"] != cursor:
            raise SquareRPathError("tail tape is not contiguous")
        cursor = int(step["tape_context_stop"])
        terminal_s2 = float(step["sum_eta_squared_after"])
        updates += 1
        macros += int(step["macro_completed"])
    expected_s2 = _prefix_sum_eta_squared() + int(plan["macro_count"]) * BASE_LR * BASE_LR
    return {
        "arm_id": arm_id,
        "optimizer_updates": updates,
        "macros": macros,
        "contexts": cursor - PREFIX_CONTEXTS,
        "terminal_sum_eta_squared": terminal_s2,
        "passed": bool(
            updates == int(plan["optimizer_updates"])
            and macros == int(plan["macro_count"])
            and cursor == PREFIX_CONTEXTS + TAIL_CONTEXTS
            and math.isclose(terminal_s2, expected_s2, rel_tol=0.0, abs_tol=1e-12)
        ),
    }


def validate_config(config: Mapping[str, Any]) -> dict[str, Any]:
    root = _root()
    frozen = _read_json(root / "experiments/configs/nanogpt300m_e2e_muon_square_rpath_v001.json")
    if config != frozen:
        raise SquareRPathError("square-ratio config differs from its frozen copy")
    plan = _read_json(root / "experiments/configs/nanogpt300m_e2e_6p5b_v001.json")
    scale.validate_plan(plan, root, verify_data=False)
    schedule = config.get("schedule", {})
    if (
        config.get("schema_version") != CONFIG_SCHEMA
        or config.get("run_id") != RUN_ID
        or config.get("campaign_id") != plan["campaign_id"]
        or config.get("primary_prediction_id") != "A2_EXTERNAL"
        or config.get("theorem_facing") is not False
        or config.get("model_parameters") != 303_473_664
        or config.get("base_batch_sequences") != BASE_BATCH
        or config.get("micro_batch_sequences") != MICRO_BATCH
        or config.get("base_learning_rate") != BASE_LR
        or config.get("gradient_clip") != GRADIENT_CLIP
        or config.get("tail_source_updates") != TAIL_SOURCE_UPDATES
        or config.get("tail_contexts_per_arm") != TAIL_CONTEXTS
        or config.get("tail_tokens_per_arm") != TAIL_CONTEXTS * CONTEXT_TOKENS
        or schedule.get("matching_rule") != "B_over_eta_squared"
        or schedule.get("primary_alignment_clock") != "sum_eta_squared"
        or schedule.get("equal_sum_eta_required") is not False
        or schedule.get("wsd_tail_macros") != schedule_plan("wsd_exp_80_20")["macro_count"]
        or schedule.get("eight_one_one_tail_macros") != schedule_plan("eight_one_one")["macro_count"]
        or schedule.get("maximum_batch_sequences") != BASE_BATCH * schedule_plan("eight_one_one")["h_maximum"]
        or config.get("launch_control") != {
            "cluster_submission_authorized": False,
            "shared_data_reupload_required": False,
            "prefix_checkpoint_reupload_required": False,
        }
    ):
        raise SquareRPathError("square-ratio identity, schedule, or launch boundary changed")
    checkpoint = config["prefix_checkpoint"]
    if (
        checkpoint.get("schema_version") != PREFIX_CHECKPOINT_SCHEMA
        or checkpoint.get("optimizer_update") != PREFIX_UPDATES
        or checkpoint.get("tape_context_cursor") != PREFIX_CONTEXTS
        or checkpoint.get("sha256") != "06102c98743433bf2be68651d0e88573eebb943900b33b2b0b7dcae6561f8d1a"
        or checkpoint.get("size_bytes") != 2_635_175_211
    ):
        raise SquareRPathError("shared prefix checkpoint identity changed")
    return dict(config)


def preflight(config_path: Path, *, verify_checkpoint_hash: bool = True) -> dict[str, Any]:
    config = validate_config(_read_json(config_path))
    arms = [_verify_arm(arm_id) for arm_id in ARM_IDS]
    pair_checks: dict[str, bool] = {}
    for shape_id in SHAPE_IDS:
        left = list(iter_tail_steps(f"{shape_id}__fixed_batch_lr"))
        right = list(iter_tail_steps(f"{shape_id}__fixed_lr_batch"))
        left_macros = [row for row in left if row["macro_completed"]]
        pair_checks[shape_id] = bool(
            len(left_macros) == len(right)
            and all(a["tape_context_stop"] == b["tape_context_stop"] for a, b in zip(left_macros, right))
            and all(math.isclose(a["ratio_B_over_eta_squared"], b["ratio_B_over_eta_squared"], rel_tol=1e-14) for a, b in zip(left_macros, right))
            and all(math.isclose(a["sum_eta_squared_after"], b["sum_eta_squared_after"], rel_tol=0.0, abs_tol=1e-12) for a, b in zip(left_macros, right))
        )
    checkpoint = _root() / "experiments/results/nanogpt300m-e2e-muon-prefix-v001/a002/prefix_output/prefix_checkpoint.pt"
    checkpoint_status = "not_checked"
    if verify_checkpoint_hash:
        record = config["prefix_checkpoint"]
        if (
            not checkpoint.is_file() or checkpoint.is_symlink()
            or checkpoint.stat().st_size != record["size_bytes"]
            or b0._sha256(checkpoint) != record["sha256"]
        ):
            raise SquareRPathError("shared prefix checkpoint is missing or changed")
        checkpoint_status = "passed"
    passed = all(row["passed"] for row in arms) and all(pair_checks.values())
    return {
        "schema_version": "nanogpt300m_e2e_muon_square_rpath_preflight_v001",
        "run_id": RUN_ID,
        "status": "offline_ready_submission_authorization_pending" if passed else "failed",
        "matching_rule": "B_over_eta_squared",
        "primary_alignment_clock": "sum_eta_squared",
        "same_shape_pair_checks": pair_checks,
        "arms": arms,
        "validation_samples": {
            shape: schedule_plan(shape)["macro_count"] + 1 for shape in SHAPE_IDS
        },
        "maximum_batch_sequences": max(BASE_BATCH * schedule_plan(shape)["h_maximum"] for shape in SHAPE_IDS),
        "checkpoint_verification": checkpoint_status,
        "shared_data_reupload_required": False,
        "prefix_checkpoint_reupload_required": False,
        "estimated_total_h200_hours": config["estimated_total_h200_hours"],
        "cluster_submission_authorized": False,
    }


def _load_inputs(config: Mapping[str, Any], workdir: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    paths: dict[str, Path] = {}
    for role, record in config["data_files"].items():
        path = workdir / record["path"]
        if (
            not path.is_file() or path.is_symlink()
            or path.stat().st_size != record["size_bytes"]
            or b0._sha256(path) != record["sha256"]
        ):
            raise SquareRPathError(f"transferred {role} bytes changed")
        paths[role] = path
    train = np.memmap(paths["train"], dtype="<u2", mode="r")
    tape = np.load(paths["training_tape"], mmap_mode="r", allow_pickle=False)
    validation = np.memmap(paths["validation"], dtype="<u2", mode="r")
    probe = np.load(paths["validation_probe"], allow_pickle=False)
    if (
        len(train) != 6_500_057_344
        or tape.shape != (CANONICAL_UPDATES * BASE_BATCH,)
        or tape.dtype.str != "<i8"
        or len(validation) != 16_777_216
        or probe.shape != (1_024,)
        or probe.dtype.str != "<i8"
        or PREFIX_CONTEXTS + TAIL_CONTEXTS > len(tape)
    ):
        raise SquareRPathError("transferred source geometry changed")
    return train, tape, validation, probe


def _load_prefix(config: Mapping[str, Any], path: Path, model: torch.nn.Module, optimizer: HybridMuon) -> dict[str, Any]:
    record = config["prefix_checkpoint"]
    if (
        not path.is_file() or path.is_symlink()
        or path.stat().st_size != record["size_bytes"]
        or b0._sha256(path) != record["sha256"]
    ):
        raise SquareRPathError("transferred prefix checkpoint bytes changed")
    state = torch.load(path, map_location=next(model.parameters()).device, weights_only=False)
    if (
        type(state) is not dict
        or state.get("schema_version") != PREFIX_CHECKPOINT_SCHEMA
        or state.get("run_id") != prefix.RUN_ID
        or state.get("optimizer_update") != PREFIX_UPDATES
        or state.get("tape_context_cursor") != PREFIX_CONTEXTS
        or state.get("source_data_sha256") != config["source_data_sha256"]
        or state.get("optimizer_partition_sha256") != optimizer.partition["sha256"]
    ):
        raise SquareRPathError("prefix checkpoint cannot seed square-ratio tails")
    model.load_state_dict(state["model_state_dict"])
    optimizer.load_state_dict(state["optimizer_state_dict"])
    torch.set_rng_state(state["cpu_rng_state"].cpu())
    torch.cuda.set_rng_state_all([value.cpu() for value in state["cuda_rng_state_all"]])
    if not optimizer_state_fp32_and_finite(optimizer):
        raise SquareRPathError("restored optimizer state is not finite FP32")
    return state


def _step(
    model: torch.nn.Module,
    optimizer: HybridMuon,
    train: np.ndarray,
    contexts: np.ndarray,
    device: torch.device,
    learning_rate: float,
) -> tuple[float, float, bool, float]:
    batch = len(contexts)
    if batch <= 0 or batch % MICRO_BATCH:
        raise SquareRPathError("invalid dynamic global batch")
    for group in optimizer.param_groups:
        group["lr"] = learning_rate
    optimizer.zero_grad(set_to_none=True)
    torch.cuda.synchronize(device)
    began = time.monotonic()
    ce = 0.0
    for start in range(0, batch, MICRO_BATCH):
        x, y = b0._batch(train, contexts[start:start + MICRO_BATCH], device)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            _, loss, _ = model(x, y)
        if loss is None or not bool(torch.isfinite(loss).detach().cpu()):
            raise SquareRPathError("nonfinite training CE")
        weight = MICRO_BATCH / batch
        ce += float(loss.detach().cpu()) * weight
        (loss * weight).backward()
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIP, foreach=True)
    if not bool(torch.isfinite(norm).detach().cpu()):
        raise SquareRPathError("nonfinite preclip gradient norm")
    norm_value = float(norm.detach().cpu())
    optimizer.step()
    torch.cuda.synchronize(device)
    return ce, norm_value, norm_value > GRADIENT_CLIP, time.monotonic() - began


def run(config_path: Path, arm_id: str, checkpoint_path: Path, output_dir: Path) -> dict[str, Any]:
    config = validate_config(_read_json(config_path))
    plan = arm_plan(arm_id)
    if output_dir.exists() or output_dir.is_symlink():
        raise SquareRPathError(f"refusing to overwrite output: {output_dir}")
    train, tape, validation, probe = _load_inputs(config, Path.cwd())
    device = b0._runtime()
    model = b0._new_model(device)
    optimizer = HybridMuon(model, BASE_LR)
    checkpoint = _load_prefix(config, checkpoint_path, model, optimizer)
    output_dir.mkdir(parents=True)

    steps = list(iter_tail_steps(arm_id))
    updates = len(steps)
    macros = int(plan["macro_count"])
    training_ce = np.empty(updates, dtype=np.float32)
    norms = np.empty(updates, dtype=np.float32)
    clipped = np.empty(updates, dtype=np.bool_)
    seconds = np.empty(updates, dtype=np.float32)
    batch_sizes = np.empty(updates, dtype=np.int32)
    rates = np.empty(updates, dtype=np.float64)
    ratios = np.empty(updates, dtype=np.float64)
    sum_eta = np.empty(updates, dtype=np.float64)
    sum_eta_squared = np.empty(updates, dtype=np.float64)
    contexts_after = np.empty(updates, dtype=np.int64)
    macro_indices = np.empty(updates, dtype=np.int32)

    validation_ce = np.empty(macros + 1, dtype=np.float64)
    validation_seconds = np.empty(macros + 1, dtype=np.float64)
    validation_contexts = np.empty(macros + 1, dtype=np.int64)
    validation_sum_eta = np.empty(macros + 1, dtype=np.float64)
    validation_sum_eta_squared = np.empty(macros + 1, dtype=np.float64)
    validation_updates = np.empty(macros + 1, dtype=np.int64)

    torch.cuda.synchronize(device)
    began = time.monotonic()
    validation_ce[0] = b0._validation_ce(model, validation, probe, device)
    torch.cuda.synchronize(device)
    validation_seconds[0] = time.monotonic() - began
    validation_contexts[0] = 0
    validation_sum_eta[0] = prefix._intrinsic_time()
    validation_sum_eta_squared[0] = _prefix_sum_eta_squared()
    validation_updates[0] = PREFIX_UPDATES
    if abs(validation_ce[0] - checkpoint["validation_anchor_ce"]) > 1e-4:
        raise SquareRPathError("loaded model does not reproduce prefix validation anchor")

    validation_position = 1
    started = time.monotonic()
    for index, step in enumerate(steps):
        contexts = tape[int(step["tape_context_start"]):int(step["tape_context_stop"])]
        row = _step(model, optimizer, train, contexts, device, float(step["learning_rate"]))
        training_ce[index], norms[index], clipped[index], seconds[index] = row
        batch_sizes[index] = int(step["batch_size"])
        rates[index] = float(step["learning_rate"])
        ratios[index] = float(step["ratio_B_over_eta_squared"])
        sum_eta[index] = float(step["sum_eta_after"])
        sum_eta_squared[index] = float(step["sum_eta_squared_after"])
        contexts_after[index] = int(step["tape_context_stop"]) - PREFIX_CONTEXTS
        macro_indices[index] = int(step["macro_index"])
        if step["macro_completed"]:
            torch.cuda.synchronize(device)
            began = time.monotonic()
            validation_ce[validation_position] = b0._validation_ce(model, validation, probe, device)
            torch.cuda.synchronize(device)
            validation_seconds[validation_position] = time.monotonic() - began
            validation_contexts[validation_position] = contexts_after[index]
            validation_sum_eta[validation_position] = sum_eta[index]
            validation_sum_eta_squared[validation_position] = sum_eta_squared[index]
            validation_updates[validation_position] = int(step["optimizer_update"])
            validation_position += 1
        if (index + 1) % 512 == 0 or index + 1 == updates:
            print(json.dumps({"event": "progress", "arm": arm_id, "completed_updates": index + 1, "total_updates": updates, "completed_macros": validation_position - 1, "total_macros": macros}), flush=True)

    if (
        validation_position != macros + 1
        or not optimizer_state_fp32_and_finite(optimizer)
        or not all(bool(torch.all(torch.isfinite(parameter)).detach().cpu()) for parameter in model.parameters())
        or not np.all(np.isfinite(training_ce))
        or not np.all(np.isfinite(validation_ce))
    ):
        raise SquareRPathError("tail completed incompletely or nonfinitely")

    trace_path = output_dir / "training_trace.npz"
    np.savez_compressed(
        trace_path,
        optimizer_update=np.asarray([int(step["optimizer_update"]) for step in steps], dtype=np.int64),
        optimizer_update_since_fork=np.arange(1, updates + 1, dtype=np.int64),
        macro_index=macro_indices,
        contexts_consumed=contexts_after,
        tokens_consumed=contexts_after * CONTEXT_TOKENS,
        batch_size=batch_sizes,
        learning_rate=rates,
        ratio_B_over_eta_squared=ratios,
        sum_eta_after=sum_eta,
        sum_eta_squared_after=sum_eta_squared,
        training_cross_entropy=training_ce,
        preclip_gradient_norm=norms,
        gradient_clipped=clipped,
        step_time_seconds=seconds,
    )
    evaluations_path = output_dir / "validation_evaluations.npz"
    np.savez_compressed(
        evaluations_path,
        optimizer_update=validation_updates,
        macro_index=np.arange(0, macros + 1, dtype=np.int64),
        contexts_consumed=validation_contexts,
        tokens_consumed=validation_contexts * CONTEXT_TOKENS,
        source_fraction=validation_contexts.astype(np.float64) / TAIL_CONTEXTS,
        sum_eta=validation_sum_eta,
        sum_eta_squared=validation_sum_eta_squared,
        validation_cross_entropy=validation_ce,
        evaluation_time_seconds=validation_seconds,
    )
    result = {
        "schema_version": RESULT_SCHEMA,
        "run_id": RUN_ID,
        "campaign_id": config["campaign_id"],
        "status": "passed",
        "arm_id": arm_id,
        "arm_plan": plan,
        "matching_rule": "B_over_eta_squared",
        "primary_alignment_clock": "sum_eta_squared",
        "prefix_checkpoint_sha256": config["prefix_checkpoint"]["sha256"],
        "source_data_sha256": config["source_data_sha256"],
        "config_sha256": b0._sha256(config_path),
        "tail_contexts": TAIL_CONTEXTS,
        "tail_tokens": TAIL_CONTEXTS * CONTEXT_TOKENS,
        "optimizer_updates": updates,
        "validation_samples": macros + 1,
        "initial_validation_ce": float(validation_ce[0]),
        "final_validation_ce": float(validation_ce[-1]),
        "clipping_fraction": float(np.mean(clipped)),
        "model_and_optimizer_fp32_finite": True,
        "runtime": {
            "device_name": torch.cuda.get_device_name(device),
            "torch_version": torch.__version__,
            "elapsed_seconds": time.monotonic() - started,
            "training_tokens_per_second": TAIL_CONTEXTS * CONTEXT_TOKENS / float(np.sum(seconds)),
            "mean_fixed_probe_evaluation_seconds": float(np.mean(validation_seconds)),
        },
        "artifacts": {
            "training_trace": {"path": trace_path.name, "sha256": b0._sha256(trace_path)},
            "validation_evaluations": {"path": evaluations_path.name, "sha256": b0._sha256(evaluations_path)},
        },
    }
    (output_dir / "result.json").write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    return result


def validate_result(config_path: Path, arm_id: str, output_dir: Path) -> dict[str, Any]:
    config = validate_config(_read_json(config_path))
    plan = arm_plan(arm_id)
    result = _read_json(output_dir / "result.json")
    if (
        result.get("schema_version") != RESULT_SCHEMA
        or result.get("run_id") != RUN_ID
        or result.get("status") != "passed"
        or result.get("arm_id") != arm_id
        or result.get("arm_plan") != plan
        or result.get("matching_rule") != "B_over_eta_squared"
        or result.get("primary_alignment_clock") != "sum_eta_squared"
        or result.get("prefix_checkpoint_sha256") != config["prefix_checkpoint"]["sha256"]
        or result.get("config_sha256") != b0._sha256(config_path)
        or result.get("tail_contexts") != TAIL_CONTEXTS
        or result.get("tail_tokens") != TAIL_CONTEXTS * CONTEXT_TOKENS
        or result.get("optimizer_updates") != plan["optimizer_updates"]
        or result.get("validation_samples") != plan["validation_samples"]
        or result.get("model_and_optimizer_fp32_finite") is not True
    ):
        raise SquareRPathError("tail result ledger changed")
    for role, filename in (("training_trace", "training_trace.npz"), ("validation_evaluations", "validation_evaluations.npz")):
        path = output_dir / filename
        if result["artifacts"][role].get("path") != filename or b0._sha256(path) != result["artifacts"][role].get("sha256"):
            raise SquareRPathError(f"tail {role} bytes changed")
    with np.load(output_dir / "training_trace.npz", allow_pickle=False) as trace:
        if trace["training_cross_entropy"].shape != (plan["optimizer_updates"],) or not np.all(np.isfinite(trace["training_cross_entropy"])):
            raise SquareRPathError("training trace shape or finiteness changed")
    with np.load(output_dir / "validation_evaluations.npz", allow_pickle=False) as evaluations:
        if (
            evaluations["validation_cross_entropy"].shape != (plan["validation_samples"],)
            or int(evaluations["contexts_consumed"][-1]) != TAIL_CONTEXTS
            or not np.all(np.isfinite(evaluations["validation_cross_entropy"]))
            or not math.isclose(float(evaluations["validation_cross_entropy"][0]), result["initial_validation_ce"], abs_tol=1e-12)
            or not math.isclose(float(evaluations["validation_cross_entropy"][-1]), result["final_validation_ce"], abs_tol=1e-12)
        ):
            raise SquareRPathError("validation trace grid or values changed")
    return {"schema_version": VALIDATION_SCHEMA, "run_id": RUN_ID, "arm_id": arm_id, "status": "passed"}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--preflight", action="store_true")
    mode.add_argument("--run-arm", action="store_true")
    mode.add_argument("--validate-result", action="store_true")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--arm", choices=ARM_IDS)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--skip-checkpoint-hash", action="store_true")
    args = parser.parse_args(argv)
    if args.preflight:
        value = preflight(args.config, verify_checkpoint_hash=not args.skip_checkpoint_hash)
    elif args.run_arm:
        if None in (args.arm, args.checkpoint, args.output_dir):
            parser.error("--run-arm requires --arm, --checkpoint, and --output-dir")
        value = run(args.config, args.arm, args.checkpoint, args.output_dir)
    else:
        if None in (args.arm, args.output_dir):
            parser.error("--validate-result requires --arm and --output-dir")
        value = validate_result(args.config, args.arm, args.output_dir)
    print(json.dumps(value, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
