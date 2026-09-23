"""Minimal 30M end-to-end plain-SGD stability/measurability pilot.

The runner trains every nanoGPT parameter from a fresh random initialization.
It is deliberately isolated from the frozen legacy AdamW runners: it has no
submission code, checkpoint loading, scheduler, clipping, AMP, or resume path.

Two explicit CLI modes are available. ``--preflight`` performs only local CPU
mechanical checks. ``--run`` executes the configured learning-rate funnel and
writes metrics/result artifacts; it never submits an external job.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import platform
import statistics
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import torch

try:
    from .common import (
        atomic_write_json,
        canonical_hash,
        load_config,
        resolve_path,
        sha256_file,
    )
    from .model import GPT, GPTConfig
    from .pretrain import resolve_device, seed_everything
except ImportError:  # pragma: no cover - direct script execution
    from common import (
        atomic_write_json,
        canonical_hash,
        load_config,
        resolve_path,
        sha256_file,
    )
    from model import GPT, GPTConfig
    from pretrain import resolve_device, seed_everything


RUN_ID = "nanogpt30m-e2e-sgd-b0-v001"
CONFIG_SCHEMA = "nanogpt30m_e2e_sgd_b0_config_v001"
PREFLIGHT_SCHEMA = "nanogpt30m_e2e_sgd_b0_preflight_v001"
RESULT_SCHEMA = "nanogpt30m_e2e_sgd_b0_result_v001"
PRIMARY_PREDICTION_ID = "A2_EXTERNAL"
JsonDict = dict[str, Any]


class B0ConfigError(ValueError):
    """Raised when the compact checked-in B0 contract is inconsistent."""


class B0Abort(RuntimeError):
    """Raised when one trajectory encounters a hard numerical stop."""


def _seed_everything_strict(seed: int) -> None:
    seed_everything(seed, deterministic=True)
    torch.use_deterministic_algorithms(True)


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise B0ConfigError(f"{label} must be a JSON object")
    return value


def _positive_int(value: Any, label: str) -> int:
    if type(value) is not int or value <= 0:
        raise B0ConfigError(f"{label} must be a positive integer")
    return value


def _finite_number(value: Any, label: str, *, minimum: float | None = None) -> float:
    if type(value) not in (int, float) or type(value) is bool:
        raise B0ConfigError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result) or (minimum is not None and result < minimum):
        raise B0ConfigError(f"{label} is outside its allowed range")
    return result


def _validate_config(config: Mapping[str, Any]) -> None:
    """Validate only the controls needed by this B0, without a new framework."""

    identities = {
        "schema_version": CONFIG_SCHEMA,
        "run_id": RUN_ID,
        "stage": "pilot_stability_measurability",
        "primary_prediction_id": PRIMARY_PREDICTION_ID,
    }
    for key, expected in identities.items():
        if config.get(key) != expected:
            raise B0ConfigError(f"{key} must equal {expected!r}")

    launch = _mapping(config.get("launch_control"), "launch_control")
    if launch.get("cluster_submission_authorized") is not False:
        raise B0ConfigError("cluster submission must remain unauthorized")
    if launch.get("runner_has_no_submission_capability") is not True:
        raise B0ConfigError("runner must declare that it cannot submit jobs")

    scope = _mapping(config.get("scientific_scope"), "scientific_scope")
    if scope.get("theorem_facing") is not False or scope.get("b0_only") is not True:
        raise B0ConfigError("scientific scope must remain non-theorem-facing B0 only")

    data = _mapping(config.get("data"), "data")
    if data.get("sampling") != "seeded_random_offsets_with_replacement":
        raise B0ConfigError("B0 data sampling contract changed")
    if data.get("reuse_scope") != "B0_pilot_only":
        raise B0ConfigError("small OpenWebText data must remain B0-only")
    for role in ("metadata", "train", "validation"):
        spec = _mapping(data.get(role), f"data.{role}")
        if type(spec.get("path")) is not str or not spec["path"]:
            raise B0ConfigError(f"data.{role}.path must be nonempty")
        digest = spec.get("sha256")
        if (
            type(digest) is not str
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
        ):
            raise B0ConfigError(f"data.{role}.sha256 must be lowercase hex64")
    _positive_int(data["train"].get("token_count"), "data.train.token_count")
    _positive_int(data["validation"].get("token_count"), "data.validation.token_count")

    model = _mapping(config.get("model"), "model")
    expected_model = {
        "family": "nanogpt",
        "block_size": 256,
        "vocab_size": 50304,
        "n_layer": 6,
        "n_head": 6,
        "n_embd": 384,
        "dropout": 0.0,
        "bias": False,
        "expected_parameter_count": 30036864,
        "initialization": "fresh_random_per_seed",
        "checkpoint": None,
    }
    for key, expected in expected_model.items():
        if model.get(key) != expected or type(model.get(key)) is not type(expected):
            raise B0ConfigError(f"model.{key} must equal {expected!r}")

    optimizer = _mapping(config.get("optimizer"), "optimizer")
    expected_optimizer = {
        "name": "sgd",
        "momentum": 0.0,
        "dampening": 0.0,
        "nesterov": False,
        "weight_decay": 0.0,
        "gradient_clip": 0.0,
    }
    for key, expected in expected_optimizer.items():
        if optimizer.get(key) != expected or type(optimizer.get(key)) is not type(expected):
            raise B0ConfigError(f"optimizer.{key} must equal {expected!r}")

    runtime = _mapping(config.get("runtime"), "runtime")
    expected_runtime = {
        "training_device": "cuda",
        "preflight_device": "cpu",
        "dtype": "float32",
        "amp": False,
        "tf32": False,
        "compile": False,
        "deterministic": True,
    }
    for key, expected in expected_runtime.items():
        if runtime.get(key) != expected or type(runtime.get(key)) is not type(expected):
            raise B0ConfigError(f"runtime.{key} must equal {expected!r}")
    microbatch_size = _positive_int(runtime.get("microbatch_size"), "runtime.microbatch_size")
    accumulation = _positive_int(runtime.get("grad_accum_steps"), "runtime.grad_accum_steps")
    effective_batch = _positive_int(
        runtime.get("effective_batch_size"), "runtime.effective_batch_size"
    )
    if microbatch_size * accumulation != effective_batch:
        raise B0ConfigError("effective batch must equal microbatch_size * grad_accum_steps")
    tokens_per_update = effective_batch * int(model["block_size"])
    if runtime.get("tokens_per_update") != tokens_per_update:
        raise B0ConfigError("runtime.tokens_per_update is inconsistent")
    _positive_int(runtime.get("maximum_wall_seconds"), "runtime.maximum_wall_seconds")

    pilot = _mapping(config.get("pilot"), "pilot")
    learning_rates = pilot.get("learning_rates")
    if not isinstance(learning_rates, list) or len(learning_rates) < 2:
        raise B0ConfigError("pilot.learning_rates must contain at least two values")
    parsed_rates = [
        _finite_number(value, f"pilot.learning_rates[{index}]", minimum=0.0)
        for index, value in enumerate(learning_rates)
    ]
    if any(value <= 0 for value in parsed_rates) or parsed_rates != sorted(set(parsed_rates)):
        raise B0ConfigError("pilot.learning_rates must be positive, unique, and increasing")
    screen_updates = _positive_int(pilot.get("screen_updates"), "pilot.screen_updates")
    selection_updates = _positive_int(
        pilot.get("selection_updates"), "pilot.selection_updates"
    )
    confirmation_updates = _positive_int(
        pilot.get("confirmation_updates"), "pilot.confirmation_updates"
    )
    if not screen_updates < selection_updates < confirmation_updates:
        raise B0ConfigError("require screen_updates < selection_updates < confirmation_updates")
    seeds = pilot.get("initialization_seeds")
    if not isinstance(seeds, list) or len(seeds) != 2 or any(type(seed) is not int for seed in seeds):
        raise B0ConfigError("pilot.initialization_seeds must contain exactly two integers")
    if len(set(seeds)) != 2:
        raise B0ConfigError("pilot initialization seeds must differ")
    for key in ("training_offset_seed", "validation_probe_seed"):
        if type(pilot.get(key)) is not int:
            raise B0ConfigError(f"pilot.{key} must be an integer")
    probe_batches = _positive_int(
        pilot.get("fixed_validation_probe_batches"),
        "pilot.fixed_validation_probe_batches",
    )
    del probe_batches
    eval_updates = pilot.get("evaluation_updates")
    if (
        not isinstance(eval_updates, list)
        or any(type(update) is not int or update < 0 for update in eval_updates)
        or eval_updates != sorted(set(eval_updates))
    ):
        raise B0ConfigError("pilot.evaluation_updates must be sorted unique nonnegative integers")
    required_updates = {0, screen_updates, selection_updates, confirmation_updates}
    if not required_updates.issubset(set(eval_updates)):
        raise B0ConfigError("evaluation updates must include every stage boundary and zero")
    replay_updates = _positive_int(pilot.get("replay_updates"), "pilot.replay_updates")
    if replay_updates > selection_updates:
        raise B0ConfigError("pilot.replay_updates exceeds the selection trajectory")

    gates = _mapping(config.get("gates"), "gates")
    for key in (
        "abort_loss_increase_nats",
        "abort_gradient_multiple",
        "abort_relative_update",
        "stable_relative_update",
        "minimum_validation_improvement_nats",
        "minimum_validation_improvement_se_multiple",
        "maximum_step_time_p90_over_median",
        "maximum_peak_memory_fraction",
        "replay_absolute_tolerance",
        "replay_relative_tolerance",
    ):
        _finite_number(gates.get(key), f"gates.{key}", minimum=0.0)
    if float(gates["stable_relative_update"]) > float(gates["abort_relative_update"]):
        raise B0ConfigError("stable relative-update gate cannot exceed abort threshold")

    artifacts = _mapping(config.get("artifacts"), "artifacts")
    for key in ("output_dir", "preflight", "metrics", "result"):
        if type(artifacts.get(key)) is not str or not artifacts[key]:
            raise B0ConfigError(f"artifacts.{key} must be nonempty")
    if artifacts.get("checkpoint_policy") != "none":
        raise B0ConfigError("B0 must not create or load checkpoints")


def _clean_config(config: Mapping[str, Any]) -> JsonDict:
    return {key: value for key, value in config.items() if not key.startswith("_")}


def _model_config(config: Mapping[str, Any]) -> GPTConfig:
    model = config["model"]
    return GPTConfig(
        block_size=int(model["block_size"]),
        vocab_size=int(model["vocab_size"]),
        n_layer=int(model["n_layer"]),
        n_head=int(model["n_head"]),
        n_embd=int(model["n_embd"]),
        dropout=float(model["dropout"]),
        bias=bool(model["bias"]),
    )


def _load_token_data(
    config: Mapping[str, Any],
) -> tuple[np.memmap, np.memmap, JsonDict]:
    data = config["data"]
    verified: JsonDict = {}
    arrays: dict[str, np.memmap] = {}
    for role in ("train", "validation"):
        spec = data[role]
        path = resolve_path(spec["path"])
        if not path.is_file():
            raise FileNotFoundError(f"{role} token file is absent: {path}")
        digest = sha256_file(path)
        if digest != spec["sha256"]:
            raise B0ConfigError(f"{role} token file SHA256 changed")
        if path.stat().st_size % np.dtype(np.uint16).itemsize:
            raise B0ConfigError(f"{role} token file has an odd byte count")
        values = np.memmap(path, dtype=np.uint16, mode="r")
        if values.size != int(spec["token_count"]):
            raise B0ConfigError(f"{role} token count changed")
        arrays[role] = values
        verified[role] = {
            "path": str(path),
            "sha256": digest,
            "bytes": path.stat().st_size,
            "tokens": int(values.size),
            "minimum_token": int(np.min(values)),
            "maximum_token": int(np.max(values)),
        }
    metadata = data["metadata"]
    metadata_path = resolve_path(metadata["path"])
    if sha256_file(metadata_path) != metadata["sha256"]:
        raise B0ConfigError("data metadata SHA256 changed")
    verified["metadata"] = {
        "path": str(metadata_path),
        "sha256": metadata["sha256"],
    }
    vocabulary_size = int(config["model"]["vocab_size"])
    for role, values in arrays.items():
        if int(np.max(values)) >= vocabulary_size:
            raise B0ConfigError(f"{role} token exceeds model vocabulary")
    return arrays["train"], arrays["validation"], verified


def _array_sha256(values: np.ndarray) -> str:
    array = np.ascontiguousarray(values, dtype="<i8")
    return hashlib.sha256(array.tobytes(order="C")).hexdigest()


def _make_offset_plan(
    token_count: int,
    block_size: int,
    updates: int,
    grad_accum_steps: int,
    microbatch_size: int,
    seed: int,
) -> np.ndarray:
    high = token_count - block_size
    if high <= 0:
        raise ValueError("token stream is too short for the configured block size")
    generator = np.random.Generator(np.random.PCG64(int(seed)))
    return generator.integers(
        0,
        high,
        size=(updates, grad_accum_steps, microbatch_size),
        dtype=np.int64,
    )


def _make_probe_offsets(
    token_count: int,
    block_size: int,
    batches: int,
    batch_size: int,
    seed: int,
) -> np.ndarray:
    high = token_count - block_size
    if high <= 0:
        raise ValueError("validation stream is too short for the configured block size")
    generator = np.random.Generator(np.random.PCG64(int(seed)))
    return generator.integers(
        0, high, size=(batches, batch_size), dtype=np.int64
    )


def _batch_from_offsets(
    values: np.ndarray,
    offsets: Sequence[int] | np.ndarray,
    block_size: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    starts = [int(value) for value in offsets]
    x_numpy = np.stack(
        [np.asarray(values[start : start + block_size], dtype=np.int64) for start in starts]
    )
    y_numpy = np.stack(
        [
            np.asarray(values[start + 1 : start + 1 + block_size], dtype=np.int64)
            for start in starts
        ]
    )
    non_blocking = device.type == "cuda"
    return (
        torch.from_numpy(x_numpy).to(device, non_blocking=non_blocking),
        torch.from_numpy(y_numpy).to(device, non_blocking=non_blocking),
    )


def _build_optimizer(model: torch.nn.Module, learning_rate: float) -> torch.optim.SGD:
    if not math.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError("learning rate must be positive and finite")
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if len(parameters) != len({id(parameter) for parameter in parameters}):
        raise RuntimeError("trainable parameter appears more than once")
    return torch.optim.SGD(
        parameters,
        lr=float(learning_rate),
        momentum=0.0,
        dampening=0.0,
        weight_decay=0.0,
        nesterov=False,
        foreach=False,
    )


def _global_l2_norm(tensors: Sequence[torch.Tensor]) -> float:
    present = [tensor for tensor in tensors if tensor is not None]
    if not present:
        return 0.0
    device = present[0].device
    total = torch.zeros((), dtype=torch.float32, device=device)
    for tensor in present:
        detached = tensor.detach().float()
        total = total + torch.sum(detached * detached)
    return math.sqrt(float(total.detach().cpu()))


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps" and hasattr(torch, "mps"):
        torch.mps.synchronize()


def _run_update(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    token_values: np.ndarray,
    update_offsets: np.ndarray,
    block_size: int,
    device: torch.device,
    learning_rate: float,
) -> JsonDict:
    model.train()
    optimizer.zero_grad(set_to_none=True)
    micro_losses: list[float] = []
    accumulation = int(update_offsets.shape[0])
    for offsets in update_offsets:
        x, y = _batch_from_offsets(token_values, offsets, block_size, device)
        _, loss, _ = model(x, y)
        if loss is None or not bool(torch.isfinite(loss).detach().cpu()):
            raise B0Abort("nonfinite_training_loss")
        micro_losses.append(float(loss.detach().cpu()))
        (loss / accumulation).backward()

    gradients = [parameter.grad for parameter in model.parameters() if parameter.grad is not None]
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    gradient_norm = _global_l2_norm(gradients)
    parameter_norm = _global_l2_norm(parameters)
    update_norm = float(learning_rate) * gradient_norm
    relative_update = update_norm / max(parameter_norm, 1e-30)
    if not all(
        math.isfinite(value)
        for value in (gradient_norm, parameter_norm, update_norm, relative_update)
    ):
        raise B0Abort("nonfinite_gradient_or_parameter_norm")

    optimizer.step()
    post_parameter_norm = _global_l2_norm(parameters)
    if not math.isfinite(post_parameter_norm):
        raise B0Abort("nonfinite_parameter_after_update")
    if optimizer.state_dict()["state"]:
        raise RuntimeError("plain SGD unexpectedly created optimizer state")
    return {
        "train_cross_entropy": float(np.mean(micro_losses)),
        "pre_step_gradient_l2_norm": gradient_norm,
        "pre_step_parameter_l2_norm": parameter_norm,
        "update_l2_norm": update_norm,
        "relative_update": relative_update,
        "post_step_parameter_l2_norm": post_parameter_norm,
        "all_finite": True,
    }


@torch.no_grad()
def _evaluate_fixed_probe(
    model: torch.nn.Module,
    token_values: np.ndarray,
    probe_offsets: np.ndarray,
    block_size: int,
    device: torch.device,
) -> tuple[float, float, np.ndarray]:
    was_training = model.training
    model.eval()
    losses: list[float] = []
    for offsets in probe_offsets:
        x, y = _batch_from_offsets(token_values, offsets, block_size, device)
        _, loss, _ = model(x, y)
        if loss is None or not bool(torch.isfinite(loss).detach().cpu()):
            raise B0Abort("nonfinite_validation_loss")
        losses.append(float(loss.detach().cpu()))
    model.train(was_training)
    values = np.asarray(losses, dtype=np.float64)
    standard_error = (
        float(np.std(values, ddof=1) / math.sqrt(len(values))) if len(values) > 1 else 0.0
    )
    return float(np.mean(values)), standard_error, values


def _model_initialization_sha256(model: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(b"\0")
        digest.update(np.asarray(value.shape, dtype="<i8").tobytes())
        digest.update(value.numpy().tobytes(order="C"))
    return digest.hexdigest()


def _paired_improvement(
    initial: np.ndarray, current: np.ndarray
) -> tuple[float, float]:
    differences = np.asarray(initial, dtype=np.float64) - np.asarray(current, dtype=np.float64)
    standard_error = (
        float(np.std(differences, ddof=1) / math.sqrt(len(differences)))
        if len(differences) > 1
        else 0.0
    )
    return float(np.mean(differences)), standard_error


def _finite_or_none(value: float) -> float | None:
    return float(value) if math.isfinite(value) else None


def _trajectory(
    *,
    config: Mapping[str, Any],
    train_values: np.ndarray,
    validation_values: np.ndarray,
    offset_plan: np.ndarray,
    probe_offsets: np.ndarray,
    device: torch.device,
    learning_rate: float,
    initialization_seed: int,
    maximum_updates: int,
    phase: str,
    deadline: float,
    emit: Callable[[Mapping[str, Any]], None],
) -> tuple[JsonDict, list[JsonDict]]:
    _seed_everything_strict(initialization_seed)
    model = GPT(_model_config(config))
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    if parameter_count != int(config["model"]["expected_parameter_count"]):
        raise RuntimeError("model parameter count changed")
    initialization_sha256 = _model_initialization_sha256(model)
    model.to(device)
    optimizer = _build_optimizer(model, learning_rate)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    evaluation_updates = {
        int(value) for value in config["pilot"]["evaluation_updates"] if value <= maximum_updates
    }
    block_size = int(config["model"]["block_size"])
    tokens_per_update = int(config["runtime"]["tokens_per_update"])
    gates = config["gates"]
    update_rows: list[JsonDict] = []
    evaluation_rows: list[JsonDict] = []
    gradient_norms: list[float] = []
    relative_updates: list[float] = []
    step_times: list[float] = []
    abort_reason: str | None = None
    consecutive_loss_increases = 0

    initial_mean, initial_se, initial_batches = _evaluate_fixed_probe(
        model, validation_values, probe_offsets, block_size, device
    )
    initial_row: JsonDict = {
        "event": "evaluation",
        "phase": phase,
        "seed": initialization_seed,
        "learning_rate": learning_rate,
        "update": 0,
        "intrinsic_time": 0.0,
        "tokens_seen": 0,
        "validation_cross_entropy": initial_mean,
        "validation_standard_error": initial_se,
        "validation_improvement": 0.0,
        "validation_improvement_standard_error": 0.0,
    }
    evaluation_rows.append(initial_row)
    emit(initial_row)

    for index in range(maximum_updates):
        if time.monotonic() >= deadline:
            abort_reason = "maximum_wall_seconds_reached"
            break
        _synchronize(device)
        step_start = time.monotonic()
        try:
            metrics = _run_update(
                model,
                optimizer,
                train_values,
                offset_plan[index],
                block_size,
                device,
                learning_rate,
            )
        except B0Abort as error:
            abort_reason = str(error)
            break
        _synchronize(device)
        step_seconds = time.monotonic() - step_start
        completed_update = index + 1
        gradient_norms.append(float(metrics["pre_step_gradient_l2_norm"]))
        relative_updates.append(float(metrics["relative_update"]))
        step_times.append(step_seconds)
        row = {
            "event": "update",
            "phase": phase,
            "seed": initialization_seed,
            "learning_rate": learning_rate,
            "update": completed_update,
            "intrinsic_time": completed_update * learning_rate,
            "tokens_seen": completed_update * tokens_per_update,
            "step_seconds": step_seconds,
            **metrics,
        }
        update_rows.append(row)
        emit(row)

        if float(metrics["relative_update"]) > float(gates["abort_relative_update"]):
            abort_reason = "relative_update_abort_threshold_exceeded"
            break
        if len(gradient_norms) > 8:
            early_median = statistics.median(gradient_norms[:8])
            if gradient_norms[-1] > float(gates["abort_gradient_multiple"]) * early_median:
                abort_reason = "gradient_norm_abort_threshold_exceeded"
                break

        if completed_update in evaluation_updates:
            try:
                current_mean, current_se, current_batches = _evaluate_fixed_probe(
                    model, validation_values, probe_offsets, block_size, device
                )
            except B0Abort as error:
                abort_reason = str(error)
                break
            improvement, improvement_se = _paired_improvement(
                initial_batches, current_batches
            )
            evaluation = {
                "event": "evaluation",
                "phase": phase,
                "seed": initialization_seed,
                "learning_rate": learning_rate,
                "update": completed_update,
                "intrinsic_time": completed_update * learning_rate,
                "tokens_seen": completed_update * tokens_per_update,
                "validation_cross_entropy": current_mean,
                "validation_standard_error": current_se,
                "validation_improvement": improvement,
                "validation_improvement_standard_error": improvement_se,
            }
            evaluation_rows.append(evaluation)
            emit(evaluation)
            if current_mean > initial_mean + float(gates["abort_loss_increase_nats"]):
                consecutive_loss_increases += 1
            else:
                consecutive_loss_increases = 0
            if consecutive_loss_increases >= 2:
                abort_reason = "validation_loss_abort_threshold_exceeded_twice"
                break

    completed_updates = len(update_rows)
    final_evaluation = next(
        (
            row
            for row in reversed(evaluation_rows)
            if int(row["update"]) == completed_updates
        ),
        evaluation_rows[-1],
    )
    maximum_relative_update = max(relative_updates, default=math.inf)
    timed = step_times[8:] if len(step_times) > 8 else step_times
    median_step = statistics.median(timed) if timed else math.inf
    p90_step = float(np.percentile(timed, 90)) if timed else math.inf
    timing_ratio = p90_step / median_step if median_step > 0 else math.inf
    improvement = float(final_evaluation["validation_improvement"])
    improvement_se = float(final_evaluation["validation_improvement_standard_error"])
    required_improvement = max(
        float(gates["minimum_validation_improvement_nats"]),
        float(gates["minimum_validation_improvement_se_multiple"]) * improvement_se,
    )
    stable = bool(
        abort_reason is None
        and completed_updates == maximum_updates
        and maximum_relative_update <= float(gates["stable_relative_update"])
        and timing_ratio <= float(gates["maximum_step_time_p90_over_median"])
    )
    measurable = bool(improvement >= required_improvement)
    selection_update = int(config["pilot"]["selection_updates"])
    selection_evaluation = next(
        (
            row
            for row in evaluation_rows
            if int(row["update"]) == selection_update
        ),
        None,
    )
    continued_improvement_pass: bool | None = None
    if phase == "confirmation":
        continued_improvement_pass = bool(
            selection_evaluation is not None
            and int(final_evaluation["update"]) == maximum_updates
            and float(final_evaluation["validation_cross_entropy"])
            <= float(selection_evaluation["validation_cross_entropy"])
        )

    peak_memory_bytes: int | None = None
    peak_memory_fraction: float | None = None
    memory_pass = True
    if device.type == "cuda":
        peak_memory_bytes = int(torch.cuda.max_memory_allocated(device))
        total_memory = int(torch.cuda.get_device_properties(device).total_memory)
        peak_memory_fraction = peak_memory_bytes / total_memory
        memory_pass = peak_memory_fraction <= float(gates["maximum_peak_memory_fraction"])

    summary: JsonDict = {
        "phase": phase,
        "seed": initialization_seed,
        "learning_rate": learning_rate,
        "initialization_sha256": initialization_sha256,
        "completed_updates": completed_updates,
        "requested_updates": maximum_updates,
        "screen_boundary_reached": completed_updates >= int(config["pilot"]["screen_updates"]),
        "aborted_reason": abort_reason,
        "initial_validation_cross_entropy": initial_mean,
        "selection_validation_cross_entropy": (
            float(selection_evaluation["validation_cross_entropy"])
            if selection_evaluation is not None
            else None
        ),
        "final_validation_cross_entropy": float(final_evaluation["validation_cross_entropy"]),
        "validation_improvement": improvement,
        "validation_improvement_standard_error": improvement_se,
        "required_validation_improvement": required_improvement,
        "maximum_relative_update": _finite_or_none(maximum_relative_update),
        "step_time_median_seconds": _finite_or_none(median_step),
        "step_time_p90_seconds": _finite_or_none(p90_step),
        "step_time_p90_over_median": _finite_or_none(timing_ratio),
        "peak_memory_bytes": peak_memory_bytes,
        "peak_memory_fraction": peak_memory_fraction,
        "optimizer_state_entries": len(optimizer.state_dict()["state"]),
        "stability_pass": stable,
        "measurability_pass": measurable,
        "continued_improvement_pass": continued_improvement_pass,
        "memory_pass": memory_pass,
        "arm_pass": bool(
            stable
            and measurable
            and memory_pass
            and continued_improvement_pass is not False
        ),
    }
    del optimizer, model
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return summary, update_rows


def _select_highest_adjacent_pair(
    learning_rates: Sequence[float], summaries: Sequence[Mapping[str, Any]]
) -> list[float] | None:
    passing = {
        float(summary["learning_rate"])
        for summary in summaries
        if bool(summary.get("arm_pass"))
    }
    candidates = [
        [float(lower), float(upper)]
        for lower, upper in zip(learning_rates, learning_rates[1:])
        if float(lower) in passing and float(upper) in passing
    ]
    return candidates[-1] if candidates else None


def _replay_comparison(
    first: Sequence[Mapping[str, Any]],
    second: Sequence[Mapping[str, Any]],
    updates: int,
    absolute_tolerance: float,
    relative_tolerance: float,
) -> JsonDict:
    keys = (
        "train_cross_entropy",
        "pre_step_gradient_l2_norm",
        "pre_step_parameter_l2_norm",
        "relative_update",
        "post_step_parameter_l2_norm",
    )
    first_prefix = list(first)[:updates]
    second_prefix = list(second)[:updates]
    if len(first_prefix) != updates or len(second_prefix) != updates:
        return {"passed": False, "reason": "replay_prefix_incomplete"}
    maximum_absolute = 0.0
    maximum_relative = 0.0
    all_close = True
    for left, right in zip(first_prefix, second_prefix):
        for key in keys:
            left_value = float(left[key])
            right_value = float(right[key])
            difference = abs(left_value - right_value)
            relative = difference / max(abs(left_value), abs(right_value), 1e-30)
            maximum_absolute = max(maximum_absolute, difference)
            maximum_relative = max(maximum_relative, relative)
            scale = max(abs(left_value), abs(right_value))
            all_close = all_close and difference <= absolute_tolerance + relative_tolerance * scale
    return {
        "passed": all_close,
        "compared_updates": updates,
        "maximum_absolute_difference": maximum_absolute,
        "maximum_relative_difference": maximum_relative,
        "absolute_tolerance": absolute_tolerance,
        "relative_tolerance": relative_tolerance,
    }


def _artifact_paths(config: Mapping[str, Any]) -> dict[str, Path]:
    artifacts = config["artifacts"]
    root = resolve_path(artifacts["output_dir"])
    return {
        "root": root,
        "preflight": root / artifacts["preflight"],
        "metrics": root / artifacts["metrics"],
        "result": root / artifacts["result"],
    }


def _code_identity() -> JsonDict:
    runner_path = Path(__file__).resolve()
    model_path = runner_path.with_name("model.py")
    return {
        "runner_path": str(runner_path),
        "runner_sha256": sha256_file(runner_path),
        "model_path": str(model_path),
        "model_sha256": sha256_file(model_path),
    }


def _write_jsonl_record(path: Path, value: Mapping[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, sort_keys=True, allow_nan=False) + "\n")
        handle.flush()


def _atomic_strict_json(path: Path, value: Mapping[str, Any]) -> None:
    json.dumps(value, sort_keys=True, allow_nan=False)
    atomic_write_json(path, value)


def _tiny_sgd_checks() -> JsonDict:
    tiny_config = GPTConfig(
        block_size=4,
        vocab_size=16,
        n_layer=1,
        n_head=2,
        n_embd=8,
        dropout=0.0,
        bias=False,
    )
    _seed_everything_strict(7401)
    reference = GPT(tiny_config)
    _seed_everything_strict(7401)
    accumulated = GPT(tiny_config)
    tokens = torch.arange(32, dtype=torch.long).reshape(8, 4) % 16
    targets = torch.roll(tokens, shifts=-1, dims=1)
    learning_rate = 0.01
    full_optimizer = _build_optimizer(reference, learning_rate)
    accumulated_optimizer = _build_optimizer(accumulated, learning_rate)

    full_optimizer.zero_grad(set_to_none=True)
    _, full_loss, _ = reference(tokens, targets)
    assert full_loss is not None
    full_loss.backward()
    full_gradients = {
        name: parameter.grad.detach().clone()
        for name, parameter in reference.named_parameters()
    }
    before = {
        name: parameter.detach().clone() for name, parameter in reference.named_parameters()
    }
    full_optimizer.step()
    exact_update_error = max(
        float(
            torch.max(
                torch.abs(
                    parameter.detach()
                    - (before[name] - learning_rate * full_gradients[name])
                )
            )
        )
        for name, parameter in reference.named_parameters()
    )

    accumulated_optimizer.zero_grad(set_to_none=True)
    for micro_tokens, micro_targets in zip(tokens.chunk(4), targets.chunk(4)):
        _, micro_loss, _ = accumulated(micro_tokens, micro_targets)
        assert micro_loss is not None
        (micro_loss / 4).backward()
    maximum_gradient_difference = max(
        float(torch.max(torch.abs(parameter.grad - full_gradients[name])))
        for name, parameter in accumulated.named_parameters()
    )
    accumulated_optimizer.step()
    maximum_parameter_difference = max(
        float(
            torch.max(
                torch.abs(
                    parameter.detach()
                    - dict(reference.named_parameters())[name].detach()
                )
            )
        )
        for name, parameter in accumulated.named_parameters()
    )
    states_empty = not full_optimizer.state_dict()["state"] and not accumulated_optimizer.state_dict()["state"]
    tolerance = 1e-6
    return {
        "passed": bool(
            exact_update_error <= tolerance
            and maximum_gradient_difference <= tolerance
            and maximum_parameter_difference <= tolerance
            and states_empty
        ),
        "tolerance": tolerance,
        "maximum_exact_update_error": exact_update_error,
        "maximum_gradient_accumulation_difference": maximum_gradient_difference,
        "maximum_parameter_difference_after_step": maximum_parameter_difference,
        "optimizer_state_empty": states_empty,
    }


def preflight(config_or_path: str | Path | Mapping[str, Any]) -> JsonDict:
    started = time.monotonic()
    if isinstance(config_or_path, (str, Path)):
        config = load_config(config_or_path)
    else:
        config = dict(config_or_path)
        config["_config_hash"] = canonical_hash(_clean_config(config))
    _validate_config(config)
    train_values, validation_values, data_summary = _load_token_data(config)
    runtime = config["runtime"]
    if runtime["preflight_device"] != "cpu":
        raise B0ConfigError("preflight must remain CPU-only")
    device = torch.device("cpu")
    seed = int(config["pilot"]["initialization_seeds"][0])
    _seed_everything_strict(seed)
    model = GPT(_model_config(config)).to(device)
    initialization_sha256 = _model_initialization_sha256(model)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    all_trainable = all(parameter.requires_grad for parameter in model.parameters())
    tied = model.lm_head.weight.data_ptr() == model.transformer.wte.weight.data_ptr()
    optimizer = _build_optimizer(model, float(config["pilot"]["learning_rates"][0]))
    optimizer_state_before = len(optimizer.state_dict()["state"])

    pilot = config["pilot"]
    two_step_plan = _make_offset_plan(
        len(train_values),
        int(config["model"]["block_size"]),
        2,
        int(runtime["grad_accum_steps"]),
        int(runtime["microbatch_size"]),
        int(pilot["training_offset_seed"]),
    )
    smoke_rows = []
    smoke_start = time.monotonic()
    for index in range(2):
        row = _run_update(
            model,
            optimizer,
            train_values,
            two_step_plan[index],
            int(config["model"]["block_size"]),
            device,
            float(pilot["learning_rates"][0]),
        )
        smoke_rows.append({"update": index + 1, **row})
    smoke_seconds = time.monotonic() - smoke_start
    optimizer_state_after = len(optimizer.state_dict()["state"])

    full_plan = _make_offset_plan(
        len(train_values),
        int(config["model"]["block_size"]),
        int(pilot["confirmation_updates"]),
        int(runtime["grad_accum_steps"]),
        int(runtime["microbatch_size"]),
        int(pilot["training_offset_seed"]),
    )
    repeated_plan = _make_offset_plan(
        len(train_values),
        int(config["model"]["block_size"]),
        int(pilot["confirmation_updates"]),
        int(runtime["grad_accum_steps"]),
        int(runtime["microbatch_size"]),
        int(pilot["training_offset_seed"]),
    )
    probe_offsets = _make_probe_offsets(
        len(validation_values),
        int(config["model"]["block_size"]),
        int(pilot["fixed_validation_probe_batches"]),
        int(runtime["microbatch_size"]),
        int(pilot["validation_probe_seed"]),
    )
    tiny_checks = _tiny_sgd_checks()
    checks = {
        "data_hashes_and_counts": True,
        "token_range_within_vocabulary": all(
            int(data_summary[role]["maximum_token"]) < int(config["model"]["vocab_size"])
            for role in ("train", "validation")
        ),
        "model_parameter_count": parameter_count
        == int(config["model"]["expected_parameter_count"]),
        "all_model_parameters_trainable": all_trainable,
        "embedding_and_lm_head_weight_tied": tied,
        "optimizer_is_plain_sgd": type(optimizer) is torch.optim.SGD,
        "optimizer_state_empty_before_and_after": optimizer_state_before == 0
        and optimizer_state_after == 0,
        "offset_plan_reproducible": bool(np.array_equal(full_plan, repeated_plan)),
        "tiny_exact_update_and_accumulation": bool(tiny_checks["passed"]),
        "full_shape_two_step_smoke": len(smoke_rows) == 2
        and all(bool(row["all_finite"]) for row in smoke_rows),
    }
    result: JsonDict = {
        "schema_version": PREFLIGHT_SCHEMA,
        "run_id": RUN_ID,
        "status": "passed" if all(checks.values()) else "failed",
        "config_sha256": config["_config_hash"],
        "code": _code_identity(),
        "checks": checks,
        "data": data_summary,
        "model": {
            "config": asdict(model.config),
            "parameters": parameter_count,
            "initialization_sha256": initialization_sha256,
        },
        "optimizer": {
            "name": "sgd",
            "state_entries_before": optimizer_state_before,
            "state_entries_after": optimizer_state_after,
            "momentum": 0.0,
            "weight_decay": 0.0,
            "gradient_clip": 0.0,
        },
        "training_shape": {
            "microbatch_size": runtime["microbatch_size"],
            "grad_accum_steps": runtime["grad_accum_steps"],
            "effective_batch_size": runtime["effective_batch_size"],
            "block_size": config["model"]["block_size"],
            "tokens_per_update": runtime["tokens_per_update"],
        },
        "reproducibility": {
            "training_offset_plan_sha256": _array_sha256(full_plan),
            "validation_probe_sha256": _array_sha256(probe_offsets),
        },
        "tiny_checks": tiny_checks,
        "two_step_smoke": {
            "device": "cpu",
            "updates": smoke_rows,
            "elapsed_seconds": smoke_seconds,
        },
        "environment": {
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "numpy": np.__version__,
            "platform": platform.platform(),
        },
        "scientific_training_executed": False,
        "mechanical_smoke_updates": 2,
        "gpu_submission_performed": False,
        "elapsed_seconds": time.monotonic() - started,
    }
    paths = _artifact_paths(config)
    paths["root"].mkdir(parents=True, exist_ok=True)
    _atomic_strict_json(paths["preflight"], result)
    return result


def run_b0(config_or_path: str | Path | Mapping[str, Any]) -> JsonDict:
    started = time.monotonic()
    if isinstance(config_or_path, (str, Path)):
        config = load_config(config_or_path)
    else:
        config = dict(config_or_path)
        config["_config_hash"] = canonical_hash(_clean_config(config))
    _validate_config(config)
    train_values, validation_values, data_summary = _load_token_data(config)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.use_deterministic_algorithms(True)
    device = resolve_device(str(config["runtime"]["training_device"]))
    if device.type != "cuda":
        raise RuntimeError("the checked-in B0 run requires one CUDA GPU")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    paths = _artifact_paths(config)
    paths["root"].mkdir(parents=True, exist_ok=True)
    partial_metrics = Path(str(paths["metrics"]) + ".partial")
    for role in ("metrics", "result"):
        if paths[role].exists():
            raise FileExistsError(f"refusing to overwrite existing {role}: {paths[role]}")
    if partial_metrics.exists():
        raise FileExistsError(f"refusing to overwrite partial metrics: {partial_metrics}")
    partial_metrics.write_text("", encoding="utf-8")
    emit = lambda value: _write_jsonl_record(partial_metrics, value)

    pilot = config["pilot"]
    runtime = config["runtime"]
    maximum_updates = int(pilot["confirmation_updates"])
    offset_plan = _make_offset_plan(
        len(train_values),
        int(config["model"]["block_size"]),
        maximum_updates,
        int(runtime["grad_accum_steps"]),
        int(runtime["microbatch_size"]),
        int(pilot["training_offset_seed"]),
    )
    probe_offsets = _make_probe_offsets(
        len(validation_values),
        int(config["model"]["block_size"]),
        int(pilot["fixed_validation_probe_batches"]),
        int(runtime["microbatch_size"]),
        int(pilot["validation_probe_seed"]),
    )
    deadline = started + int(runtime["maximum_wall_seconds"])
    rates = [float(value) for value in pilot["learning_rates"]]
    first_seed = int(pilot["initialization_seeds"][0])
    screen_summaries: list[JsonDict] = []
    screen_updates: dict[float, list[JsonDict]] = {}
    for learning_rate in rates:
        if time.monotonic() >= deadline:
            break
        summary, update_rows = _trajectory(
            config=config,
            train_values=train_values,
            validation_values=validation_values,
            offset_plan=offset_plan,
            probe_offsets=probe_offsets,
            device=device,
            learning_rate=learning_rate,
            initialization_seed=first_seed,
            maximum_updates=int(pilot["selection_updates"]),
            phase="screen",
            deadline=deadline,
            emit=emit,
        )
        screen_summaries.append(summary)
        screen_updates[learning_rate] = update_rows

    expected_screen_rates = set(rates)
    observed_screen_rates = {
        float(summary["learning_rate"]) for summary in screen_summaries
    }
    full_screen_grid_completed = bool(
        observed_screen_rates == expected_screen_rates
        and not any(
            summary.get("aborted_reason") == "maximum_wall_seconds_reached"
            for summary in screen_summaries
        )
    )
    selected_pair = (
        _select_highest_adjacent_pair(rates, screen_summaries)
        if full_screen_grid_completed
        else None
    )
    confirmation_summaries: list[JsonDict] = []
    confirmation_updates: dict[tuple[float, int], list[JsonDict]] = {}
    replay: JsonDict = {"passed": False, "reason": "no_selected_pair"}
    if selected_pair is not None:
        for initialization_seed in pilot["initialization_seeds"]:
            for learning_rate in selected_pair:
                if time.monotonic() >= deadline:
                    break
                summary, update_rows = _trajectory(
                    config=config,
                    train_values=train_values,
                    validation_values=validation_values,
                    offset_plan=offset_plan,
                    probe_offsets=probe_offsets,
                    device=device,
                    learning_rate=float(learning_rate),
                    initialization_seed=int(initialization_seed),
                    maximum_updates=maximum_updates,
                    phase="confirmation",
                    deadline=deadline,
                    emit=emit,
                )
                confirmation_summaries.append(summary)
                confirmation_updates[(float(learning_rate), int(initialization_seed))] = update_rows
            if time.monotonic() >= deadline:
                break
        replay_rate = float(selected_pair[-1])
        replay_key = (replay_rate, first_seed)
        if replay_key in confirmation_updates:
            replay = _replay_comparison(
                screen_updates[replay_rate],
                confirmation_updates[replay_key],
                int(pilot["replay_updates"]),
                float(config["gates"]["replay_absolute_tolerance"]),
                float(config["gates"]["replay_relative_tolerance"]),
            )
            screen_summary = next(
                summary
                for summary in screen_summaries
                if float(summary["learning_rate"]) == replay_rate
            )
            confirmation_summary = next(
                summary
                for summary in confirmation_summaries
                if float(summary["learning_rate"]) == replay_rate
                and int(summary["seed"]) == first_seed
            )
            initialization_match = (
                screen_summary["initialization_sha256"]
                == confirmation_summary["initialization_sha256"]
            )
            replay["initialization_sha256_match"] = initialization_match
            replay["passed"] = bool(replay["passed"] and initialization_match)
        else:
            replay = {"passed": False, "reason": "confirmation_replay_missing"}

    expected_confirmation_keys = (
        {
            (float(learning_rate), int(initialization_seed))
            for initialization_seed in pilot["initialization_seeds"]
            for learning_rate in selected_pair
        }
        if selected_pair is not None
        else set()
    )
    observed_confirmation_keys = {
        (float(summary["learning_rate"]), int(summary["seed"]))
        for summary in confirmation_summaries
    }
    full_confirmation_grid_completed = bool(
        selected_pair is None
        or (
            observed_confirmation_keys == expected_confirmation_keys
            and not any(
                summary.get("aborted_reason") == "maximum_wall_seconds_reached"
                for summary in confirmation_summaries
            )
        )
    )
    confirmation_pass = bool(
        selected_pair is not None
        and full_confirmation_grid_completed
        and all(bool(summary["arm_pass"]) for summary in confirmation_summaries)
    )
    resource_incomplete = bool(
        not full_screen_grid_completed
        or (selected_pair is not None and not full_confirmation_grid_completed)
    )
    if resource_incomplete:
        status = "incomplete"
        decision = "INCOMPLETE"
    else:
        status = "completed"
        decision = (
            "GO"
            if selected_pair is not None and confirmation_pass and replay["passed"]
            else "NO_GO"
        )
    elapsed = time.monotonic() - started
    result: JsonDict = {
        "schema_version": RESULT_SCHEMA,
        "run_id": RUN_ID,
        "status": status,
        "decision": decision,
        "primary_prediction_id": PRIMARY_PREDICTION_ID,
        "theorem_facing": False,
        "config_sha256": config["_config_hash"],
        "code": _code_identity(),
        "data": data_summary,
        "model": {
            "family": config["model"]["family"],
            "block_size": config["model"]["block_size"],
            "vocab_size": config["model"]["vocab_size"],
            "n_layer": config["model"]["n_layer"],
            "n_head": config["model"]["n_head"],
            "n_embd": config["model"]["n_embd"],
            "dropout": config["model"]["dropout"],
            "bias": config["model"]["bias"],
            "parameters": config["model"]["expected_parameter_count"],
            "initialization": config["model"]["initialization"],
        },
        "optimizer": {
            "name": config["optimizer"]["name"],
            "momentum": config["optimizer"]["momentum"],
            "dampening": config["optimizer"]["dampening"],
            "nesterov": config["optimizer"]["nesterov"],
            "weight_decay": config["optimizer"]["weight_decay"],
            "gradient_clip": config["optimizer"]["gradient_clip"],
            "foreach": False,
            "expected_state_entries": 0,
        },
        "design": {
            "learning_rates": rates,
            "initialization_seeds": pilot["initialization_seeds"],
            "training_offset_plan_sha256": _array_sha256(offset_plan),
            "validation_probe_sha256": _array_sha256(probe_offsets),
            "screen_updates": pilot["screen_updates"],
            "selection_updates": pilot["selection_updates"],
            "confirmation_updates": pilot["confirmation_updates"],
            "selected_pair": selected_pair,
        },
        "screen": screen_summaries,
        "confirmation": confirmation_summaries,
        "replay": replay,
        "gates": {
            "full_screen_grid_completed": full_screen_grid_completed,
            "adjacent_pair_selected": selected_pair is not None,
            "full_confirmation_grid_completed": full_confirmation_grid_completed,
            "all_confirmation_arms_passed": confirmation_pass,
            "replay_passed": bool(replay["passed"]),
        },
        "runtime": {
            "device": str(device),
            "device_name": torch.cuda.get_device_name(device),
            "device_total_memory_bytes": int(
                torch.cuda.get_device_properties(device).total_memory
            ),
            "elapsed_seconds": elapsed,
            "actual_gpu_hours": elapsed / 3600.0,
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "numpy": np.__version__,
        },
        "artifacts": {
            "metrics": str(paths["metrics"]),
            "checkpoint": None,
        },
        "gpu_submission_performed": False,
    }
    json.dumps(result, sort_keys=True, allow_nan=False)
    os.replace(partial_metrics, paths["metrics"])
    _atomic_strict_json(paths["result"], result)
    return result


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--preflight", action="store_true", help="run local CPU checks only")
    modes.add_argument("--run", action="store_true", help="run the configured CUDA B0")
    return parser


def main() -> int:
    args = build_argument_parser().parse_args()
    try:
        result = preflight(args.config) if args.preflight else run_b0(args.config)
    except (B0ConfigError, OSError, RuntimeError, ValueError) as error:
        print(json.dumps({"status": "failed", "error": str(error)}, sort_keys=True))
        return 2
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0 if result.get("status") in {"passed", "completed"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
