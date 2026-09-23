"""Deterministic, resumable local nanoGPT pretraining.

The runner consumes a pipeline JSON file, selects its ``pretrain`` section,
and trains on the standard nanoGPT pair of flat uint16 token files.  It is
deliberately small and dependency-light so that the same checkpoint can feed
the repository's frozen contextual-feature stage.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import math
import os
import platform
import random
import subprocess
import sys
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

try:
    from .model import GPT, GPTConfig
except ImportError:  # pragma: no cover - supports direct script execution
    from model import GPT, GPTConfig


JsonDict = dict[str, Any]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_config(path: str | Path) -> JsonDict:
    """Load a JSON mapping and reject ambiguous non-object roots."""

    config_path = Path(path).expanduser().resolve()
    with config_path.open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"configuration root must be a JSON object: {config_path}")
    return config


def resolve_device(requested: str | None = "auto") -> torch.device:
    """Resolve auto/cuda/mps/cpu and fail clearly for unavailable hardware."""

    value = (requested or "auto").lower()
    if value == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if value.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA was requested but is unavailable: {value}")
    if value == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS was requested but is unavailable")
    if value != "cpu" and value != "mps" and not value.startswith("cuda"):
        raise ValueError(f"unsupported device {requested!r}")
    return torch.device(value)


def seed_everything(seed: int, deterministic: bool = True) -> None:
    """Seed Python, NumPy, and PyTorch without forcing an unavailable backend."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True


def cosine_learning_rate(
    iteration: int,
    learning_rate: float,
    min_lr: float,
    warmup_iters: int,
    lr_decay_iters: int,
) -> float:
    """Linear warmup followed by cosine decay."""

    if warmup_iters > 0 and iteration < warmup_iters:
        return learning_rate * float(iteration + 1) / float(warmup_iters)
    if iteration >= lr_decay_iters:
        return min_lr
    denominator = max(1, lr_decay_iters - warmup_iters)
    decay_ratio = (iteration - warmup_iters) / denominator
    decay_ratio = min(1.0, max(0.0, decay_ratio))
    coefficient = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coefficient * (learning_rate - min_lr)


def _discover_repo_root(config_path: Path | None) -> Path:
    starts = [config_path.parent] if config_path is not None else []
    starts.append(Path.cwd().resolve())
    for start in starts:
        for candidate in (start, *start.parents):
            if (candidate / "README.md").is_file() and (candidate / "experiments").is_dir():
                return candidate
    return Path.cwd().resolve()


def _resolve_path(value: str | Path, repo_root: Path) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = repo_root / path
    return path.resolve()


def _configuration_hash(config: Mapping[str, Any]) -> str:
    serialized = json.dumps(config, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _git_revision(repo_root: Path) -> str | None:
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return completed.stdout.strip() or None


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, default=str)
        handle.write("\n")
    os.replace(temporary, path)


def _append_jsonl(path: Path, payload: Mapping[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True, default=str) + "\n")
        handle.flush()


def _open_uint16_memmap(path: Path) -> np.memmap:
    if not path.is_file():
        raise FileNotFoundError(f"token file does not exist: {path}")
    if path.stat().st_size % np.dtype(np.uint16).itemsize != 0:
        raise ValueError(f"uint16 token file has an odd byte count: {path}")
    data = np.memmap(path, dtype=np.uint16, mode="r")
    if data.size == 0:
        raise ValueError(f"token file is empty: {path}")
    return data


class MemmapBatcher:
    """Draw reproducible autoregressive batches from uint16 memmaps.

    Training and evaluation have independent generators, so changing the
    evaluation cadence does not alter the training token sequence.
    """

    def __init__(self, train_path: Path, val_path: Path, seed: int) -> None:
        self.data = {
            "train": _open_uint16_memmap(train_path),
            "val": _open_uint16_memmap(val_path),
        }
        self.generators = {
            "train_step": torch.Generator(device="cpu").manual_seed(seed + 101),
            "train_eval": torch.Generator(device="cpu").manual_seed(seed + 202),
            "val_eval": torch.Generator(device="cpu").manual_seed(seed + 303),
        }

    def validate_block_size(self, block_size: int) -> None:
        for split, values in self.data.items():
            if values.size <= block_size:
                raise ValueError(
                    f"{split} contains {values.size} tokens but block_size is {block_size}"
                )

    def batch(
        self,
        split: str,
        batch_size: int,
        block_size: int,
        device: torch.device,
        *,
        evaluation: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if split not in self.data:
            raise ValueError(f"unknown split {split!r}")
        stream_name = f"{split}_eval" if evaluation else "train_step"
        if split != "train" and not evaluation:
            raise ValueError("non-training splits can only be sampled for evaluation")
        values = self.data[split]
        high = values.size - block_size
        offsets = torch.randint(
            high,
            (batch_size,),
            generator=self.generators[stream_name],
            device="cpu",
        ).tolist()
        x_numpy = np.stack(
            [np.asarray(values[offset : offset + block_size], dtype=np.int64) for offset in offsets]
        )
        y_numpy = np.stack(
            [
                np.asarray(values[offset + 1 : offset + 1 + block_size], dtype=np.int64)
                for offset in offsets
            ]
        )
        x = torch.from_numpy(x_numpy)
        y = torch.from_numpy(y_numpy)
        non_blocking = device.type == "cuda"
        return (
            x.to(device, non_blocking=non_blocking),
            y.to(device, non_blocking=non_blocking),
        )

    def state_dict(self) -> dict[str, torch.Tensor]:
        return {name: generator.get_state() for name, generator in self.generators.items()}

    def load_state_dict(self, state: Mapping[str, torch.Tensor]) -> None:
        for name, generator_state in state.items():
            if name in self.generators:
                self.generators[name].set_state(generator_state.cpu())


def _dtype_from_name(name: str) -> torch.dtype:
    normalized = name.lower()
    options = {
        "float32": torch.float32,
        "fp32": torch.float32,
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
    }
    if normalized not in options:
        raise ValueError(f"unsupported dtype {name!r}")
    return options[normalized]


def _autocast_context(device: torch.device, dtype: torch.dtype):
    if dtype == torch.float32:
        return contextlib.nullcontext()
    if device.type == "cpu" and dtype == torch.float16:
        raise ValueError("CPU float16 autocast is unsupported; use float32 or bfloat16")
    return torch.autocast(device_type=device.type, dtype=dtype)


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps" and hasattr(torch, "mps"):
        torch.mps.synchronize()


def _capture_rng_state() -> JsonDict:
    state: JsonDict = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["torch_cuda"] = torch.cuda.get_rng_state_all()
    if torch.backends.mps.is_available() and hasattr(torch.mps, "get_rng_state"):
        state["torch_mps"] = torch.mps.get_rng_state()
    return state


def _restore_rng_state(state: Mapping[str, Any]) -> None:
    if "python" in state:
        random.setstate(state["python"])
    if "numpy" in state:
        numpy_state = state["numpy"]
        if isinstance(numpy_state, list):
            numpy_state = tuple(numpy_state)
        np.random.set_state(numpy_state)
    if "torch_cpu" in state:
        torch.set_rng_state(state["torch_cpu"].cpu())
    if "torch_cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["torch_cuda"])
    if (
        "torch_mps" in state
        and torch.backends.mps.is_available()
        and hasattr(torch.mps, "set_rng_state")
    ):
        torch.mps.set_rng_state(state["torch_mps"].cpu())


@torch.no_grad()
def estimate_losses(
    model: torch.nn.Module,
    batcher: MemmapBatcher,
    batch_size: int,
    block_size: int,
    eval_iters: int,
    device: torch.device,
    dtype: torch.dtype,
) -> dict[str, float]:
    """Estimate train and validation loss without advancing training batches."""

    if eval_iters <= 0:
        raise ValueError("eval_iters must be positive")
    was_training = model.training
    model.eval()
    result: dict[str, float] = {}
    for split in ("train", "val"):
        losses = []
        for _ in range(eval_iters):
            x, y = batcher.batch(
                split,
                batch_size,
                block_size,
                device,
                evaluation=True,
            )
            with _autocast_context(device, dtype):
                _, loss, _ = model(x, y)
            if loss is None:
                raise RuntimeError("model did not return a loss")
            losses.append(float(loss.detach().cpu()))
        result[split] = float(np.mean(losses))
    model.train(was_training)
    return result


def _normalise_pipeline_config(
    raw: Mapping[str, Any],
) -> tuple[JsonDict, JsonDict, JsonDict, JsonDict]:
    selected = raw.get("pretrain", raw)
    if not isinstance(selected, Mapping):
        raise ValueError("pretrain section must be a JSON object")
    pretrain = dict(selected)
    data = pretrain.get("data", raw.get("data", {}))
    model = pretrain.get("model", {})
    training = pretrain.get("training", {})
    for name, section in (("data", data), ("model", model), ("training", training)):
        if not isinstance(section, Mapping):
            raise ValueError(f"{name} section must be a JSON object")
    return pretrain, dict(data), dict(model), dict(training)


def _validate_training_config(training: Mapping[str, Any]) -> None:
    positive_integer_keys = (
        "batch_size",
        "grad_accum_steps",
        "max_iters",
        "eval_iters",
        "log_interval",
        "eval_interval",
        "checkpoint_interval",
    )
    for key in positive_integer_keys:
        if key in training and int(training[key]) <= 0:
            raise ValueError(f"training.{key} must be positive")
    for key in ("learning_rate", "min_lr"):
        if key in training and float(training[key]) < 0:
            raise ValueError(f"training.{key} must be nonnegative")


def _checkpoint_payload(
    raw_model: GPT,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    batcher: MemmapBatcher,
    *,
    iteration: int,
    best_val_loss: float,
    cumulative_elapsed_seconds: float,
    config_hash: str,
    raw_config: Mapping[str, Any],
) -> JsonDict:
    return {
        "format_version": 1,
        "saved_at": utc_now(),
        "model_config": asdict(raw_model.config),
        "model_state": raw_model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scaler_state": scaler.state_dict(),
        "iter_num": iteration,
        "best_val_loss": best_val_loss,
        "elapsed_seconds": cumulative_elapsed_seconds,
        "config_sha256": config_hash,
        "pipeline_config": dict(raw_config),
        "rng_state": _capture_rng_state(),
        "batcher_state": batcher.state_dict(),
    }


def _atomic_torch_save(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(dict(payload), temporary)
    os.replace(temporary, path)


def run_pretraining(config_or_path: str | Path | Mapping[str, Any]) -> JsonDict:
    """Run or resume pretraining and return a JSON-compatible runtime summary.

    Relative paths are resolved against the repository root.  A boolean
    ``resume: true`` means "resume if checkpoint_last.pt exists".  A string
    resume value is an explicit checkpoint and must exist.
    """

    config_path: Path | None
    if isinstance(config_or_path, (str, Path)):
        config_path = Path(config_or_path).expanduser().resolve()
        raw_config = load_config(config_path)
    elif isinstance(config_or_path, Mapping):
        config_path = None
        raw_config = dict(config_or_path)
    else:
        raise TypeError("config_or_path must be a path or mapping")

    pretrain, data_config, model_config, training = _normalise_pipeline_config(raw_config)
    if pretrain.get("enabled", True) is False:
        return {"status": "skipped", "reason": "pretrain.enabled is false"}
    _validate_training_config(training)

    repo_root_value = pretrain.get("repo_root", raw_config.get("repo_root"))
    repo_root = (
        _resolve_path(repo_root_value, Path.cwd().resolve())
        if repo_root_value
        else _discover_repo_root(config_path)
    )
    if "train_bin" not in data_config or "val_bin" not in data_config:
        raise ValueError("pretrain.data must define train_bin and val_bin")
    train_path = _resolve_path(data_config["train_bin"], repo_root)
    val_path = _resolve_path(data_config["val_bin"], repo_root)
    output_value = pretrain.get("output_dir", "experiments/results/nanogpt-local-pretrain")
    output_dir = _resolve_path(output_value, repo_root)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_last = output_dir / "checkpoint_last.pt"
    checkpoint_best = output_dir / "checkpoint_best.pt"
    log_path = output_dir / "train_log.jsonl"
    summary_path = output_dir / "runtime_summary.json"

    seed = int(pretrain.get("seed", raw_config.get("seed", 1337)))
    deterministic = bool(training.get("deterministic", True))
    seed_everything(seed, deterministic=deterministic)
    device = resolve_device(str(pretrain.get("device", raw_config.get("device", "auto"))))
    dtype_name = str(pretrain.get("dtype", training.get("dtype", "float32")))
    dtype = _dtype_from_name(dtype_name)
    if device.type == "mps" and dtype == torch.bfloat16:
        raise ValueError("MPS bfloat16 is not used by this runner; choose float32 or float16")

    gpt_config = GPTConfig(**model_config)
    batch_size = int(training.get("batch_size", 4))
    grad_accum_steps = int(training.get("grad_accum_steps", 8))
    max_iters = int(training.get("max_iters", 1_000))
    learning_rate = float(training.get("learning_rate", 3e-4))
    min_lr = float(training.get("min_lr", learning_rate * 0.1))
    warmup_iters = int(training.get("warmup_iters", max(1, max_iters // 20)))
    lr_decay_iters = int(training.get("lr_decay_iters", max_iters))
    weight_decay = float(training.get("weight_decay", 0.1))
    beta1 = float(training.get("beta1", 0.9))
    beta2 = float(training.get("beta2", 0.95))
    grad_clip = float(training.get("grad_clip", 1.0))
    log_interval = int(training.get("log_interval", 10))
    eval_interval = int(training.get("eval_interval", 100))
    eval_iters = int(training.get("eval_iters", 20))
    checkpoint_interval = int(training.get("checkpoint_interval", eval_interval))
    evaluate_at_start = bool(training.get("evaluate_at_start", False))
    wall_time_seconds = training.get("wall_time_seconds")
    if wall_time_seconds is None and training.get("wall_time_hours") is not None:
        wall_time_seconds = float(training["wall_time_hours"]) * 3600.0
    wall_time_seconds = float(wall_time_seconds) if wall_time_seconds is not None else None
    if wall_time_seconds is not None and wall_time_seconds <= 0:
        raise ValueError("wall_time_seconds must be positive")
    if warmup_iters < 0 or lr_decay_iters <= 0 or warmup_iters > lr_decay_iters:
        raise ValueError("require 0 <= warmup_iters <= lr_decay_iters")
    if min_lr > learning_rate:
        raise ValueError("min_lr must not exceed learning_rate")
    if weight_decay < 0 or grad_clip < 0:
        raise ValueError("weight_decay and grad_clip must be nonnegative")
    if not (0.0 <= beta1 < 1.0 and 0.0 <= beta2 < 1.0):
        raise ValueError("AdamW beta1 and beta2 must lie in [0, 1)")

    config_hash = _configuration_hash(raw_config)
    batcher = MemmapBatcher(train_path, val_path, seed)
    batcher.validate_block_size(gpt_config.block_size)
    raw_model = GPT(gpt_config).to(device)
    expected_parameter_count = pretrain.get("expected_parameter_count")
    actual_parameter_count = raw_model.get_num_params(non_embedding=False)
    if (
        expected_parameter_count is not None
        and actual_parameter_count != int(expected_parameter_count)
    ):
        raise ValueError(
            f"model has {actual_parameter_count:,} parameters; expected "
            f"{int(expected_parameter_count):,}. Check weight tying and architecture."
        )
    optimizer = raw_model.configure_optimizers(
        weight_decay=weight_decay,
        learning_rate=learning_rate,
        betas=(beta1, beta2),
        device_type=device.type,
    )
    scaler = torch.amp.GradScaler(
        "cuda", enabled=(device.type == "cuda" and dtype == torch.float16)
    )

    resume_value = training.get("resume", pretrain.get("resume", True))
    resume_path: Path | None = None
    if isinstance(resume_value, str):
        resume_path = _resolve_path(resume_value, repo_root)
        if not resume_path.is_file():
            raise FileNotFoundError(f"explicit resume checkpoint does not exist: {resume_path}")
    elif bool(resume_value) and checkpoint_last.is_file():
        resume_path = checkpoint_last
    elif not bool(resume_value) and checkpoint_last.exists() and not bool(training.get("overwrite", False)):
        raise FileExistsError(
            f"checkpoint already exists at {checkpoint_last}; enable resume or explicit overwrite"
        )

    iteration = 0
    best_val_loss = float("inf")
    prior_elapsed_seconds = 0.0
    resumed = resume_path is not None
    if resume_path is not None:
        checkpoint = torch.load(resume_path, map_location="cpu", weights_only=False)
        if checkpoint.get("model_config") != asdict(gpt_config):
            raise ValueError("resume checkpoint architecture does not match pretrain.model")
        checkpoint_hash = checkpoint.get("config_sha256")
        if (
            checkpoint_hash
            and checkpoint_hash != config_hash
            and not bool(training.get("allow_resume_config_mismatch", False))
        ):
            raise ValueError(
                "resume checkpoint configuration differs from the current JSON; "
                "set training.allow_resume_config_mismatch only for a declared deviation"
            )
        raw_model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        if checkpoint.get("scaler_state"):
            scaler.load_state_dict(checkpoint["scaler_state"])
        iteration = int(checkpoint.get("iter_num", 0))
        best_val_loss = float(checkpoint.get("best_val_loss", float("inf")))
        prior_elapsed_seconds = float(checkpoint.get("elapsed_seconds", 0.0))
        if checkpoint.get("rng_state"):
            _restore_rng_state(checkpoint["rng_state"])
        if checkpoint.get("batcher_state"):
            batcher.load_state_dict(checkpoint["batcher_state"])

    compile_requested = bool(training.get("compile", False))
    compiled = False
    model: torch.nn.Module = raw_model
    if compile_requested and device.type != "mps":
        model = torch.compile(raw_model)
        compiled = True
    # torch.compile is intentionally never invoked on MPS: it is not needed
    # for this local pilot and is a common source of unsupported-operator errors.

    tokens_per_iteration = batch_size * grad_accum_steps * gpt_config.block_size
    planned_token_exposure = max_iters * tokens_per_iteration
    maximum_token_exposure = pretrain.get("maximum_token_exposure")
    if (
        maximum_token_exposure is not None
        and planned_token_exposure > int(maximum_token_exposure)
    ):
        raise ValueError(
            f"planned token exposure {planned_token_exposure:,} exceeds configured "
            f"maximum {int(maximum_token_exposure):,}"
        )
    common_summary: JsonDict = {
        "run_id": pretrain.get("run_id", raw_config.get("run_id")),
        "config_path": str(config_path) if config_path else None,
        "config_sha256": config_hash,
        "git_revision": _git_revision(repo_root),
        "repo_root": str(repo_root),
        "output_dir": str(output_dir),
        "train_bin": str(train_path),
        "val_bin": str(val_path),
        "device": str(device),
        "dtype": dtype_name,
        "torch_version": torch.__version__,
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "seed": seed,
        "deterministic": deterministic,
        "compiled": compiled,
        "compile_requested": compile_requested,
        "model_config": asdict(gpt_config),
        "model_parameters": actual_parameter_count,
        "non_position_parameters": raw_model.get_num_params(non_embedding=True),
        "tokens_per_iteration": tokens_per_iteration,
        "planned_token_exposure": planned_token_exposure,
        "maximum_token_exposure": maximum_token_exposure,
        "max_iters": max_iters,
        "training": {
            "batch_size": batch_size,
            "grad_accum_steps": grad_accum_steps,
            "learning_rate": learning_rate,
            "min_lr": min_lr,
            "warmup_iters": warmup_iters,
            "lr_decay_iters": lr_decay_iters,
            "weight_decay": weight_decay,
            "beta1": beta1,
            "beta2": beta2,
            "grad_clip": grad_clip,
            "eval_interval": eval_interval,
            "eval_iters": eval_iters,
            "checkpoint_interval": checkpoint_interval,
        },
        "wall_time_seconds": wall_time_seconds,
        "resumed": resumed,
        "resume_path": str(resume_path) if resume_path else None,
        "started_at": utc_now(),
    }

    if iteration >= max_iters:
        summary = {
            **common_summary,
            "status": "already_completed",
            "completed_iters": iteration,
            "best_val_loss": best_val_loss,
            "elapsed_seconds": prior_elapsed_seconds,
            "session_elapsed_seconds": 0.0,
            "tokens_seen": iteration * tokens_per_iteration,
            "finished_at": utc_now(),
        }
        _atomic_json(summary_path, summary)
        return summary
    if wall_time_seconds is not None and prior_elapsed_seconds >= wall_time_seconds:
        summary = {
            **common_summary,
            "status": "wall_time_already_exhausted",
            "completed_iters": iteration,
            "best_val_loss": best_val_loss,
            "elapsed_seconds": prior_elapsed_seconds,
            "session_elapsed_seconds": 0.0,
            "tokens_seen": iteration * tokens_per_iteration,
            "finished_at": utc_now(),
        }
        _atomic_json(summary_path, summary)
        return summary

    if not resumed:
        log_path.write_text("", encoding="utf-8")
    _append_jsonl(
        log_path,
        {
            "event": "start" if not resumed else "resume",
            "iteration": iteration,
            "timestamp": utc_now(),
            "device": str(device),
            "model_parameters": common_summary["model_parameters"],
            "config_sha256": config_hash,
        },
    )

    session_start = time.monotonic()
    last_train_loss: float | None = None
    last_eval: dict[str, float] | None = None
    status = "completed"

    def total_elapsed() -> float:
        return prior_elapsed_seconds + (time.monotonic() - session_start)

    def save_last() -> JsonDict:
        _synchronize(device)
        payload = _checkpoint_payload(
            raw_model,
            optimizer,
            scaler,
            batcher,
            iteration=iteration,
            best_val_loss=best_val_loss,
            cumulative_elapsed_seconds=total_elapsed(),
            config_hash=config_hash,
            raw_config=raw_config,
        )
        _atomic_torch_save(checkpoint_last, payload)
        return payload

    try:
        raw_model.train()
        while iteration < max_iters:
            if wall_time_seconds is not None and total_elapsed() >= wall_time_seconds:
                status = "wall_time_reached"
                break

            if evaluate_at_start and iteration == 0:
                last_eval = estimate_losses(
                    model,
                    batcher,
                    batch_size,
                    gpt_config.block_size,
                    eval_iters,
                    device,
                    dtype,
                )
                _append_jsonl(
                    log_path,
                    {"event": "evaluation", "iteration": 0, **last_eval, "timestamp": utc_now()},
                )

            lr = cosine_learning_rate(
                iteration,
                learning_rate,
                min_lr,
                warmup_iters,
                lr_decay_iters,
            )
            for parameter_group in optimizer.param_groups:
                parameter_group["lr"] = lr
            optimizer.zero_grad(set_to_none=True)
            micro_losses: list[float] = []
            for _ in range(grad_accum_steps):
                x, y = batcher.batch(
                    "train", batch_size, gpt_config.block_size, device, evaluation=False
                )
                with _autocast_context(device, dtype):
                    _, loss, _ = model(x, y)
                    if loss is None:
                        raise RuntimeError("model did not return a loss")
                    scaled_loss = loss / grad_accum_steps
                micro_losses.append(float(loss.detach().cpu()))
                scaler.scale(scaled_loss).backward()
            if grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(raw_model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
            iteration += 1
            last_train_loss = float(np.mean(micro_losses))

            if iteration == 1 or iteration % log_interval == 0:
                _synchronize(device)
                _append_jsonl(
                    log_path,
                    {
                        "event": "train",
                        "iteration": iteration,
                        "train_loss": last_train_loss,
                        "learning_rate": lr,
                        "elapsed_seconds": total_elapsed(),
                        "tokens_seen": iteration * tokens_per_iteration,
                        "timestamp": utc_now(),
                    },
                )

            if (
                iteration < max_iters
                and wall_time_seconds is not None
                and total_elapsed() >= wall_time_seconds
            ):
                status = "wall_time_reached"
                break

            should_evaluate = iteration % eval_interval == 0 or iteration == max_iters
            if should_evaluate:
                last_eval = estimate_losses(
                    model,
                    batcher,
                    batch_size,
                    gpt_config.block_size,
                    eval_iters,
                    device,
                    dtype,
                )
                improved = last_eval["val"] < best_val_loss
                if improved:
                    best_val_loss = last_eval["val"]
                _append_jsonl(
                    log_path,
                    {
                        "event": "evaluation",
                        "iteration": iteration,
                        "learning_rate": lr,
                        "best_val_loss": best_val_loss,
                        "improved": improved,
                        "elapsed_seconds": total_elapsed(),
                        "timestamp": utc_now(),
                        **last_eval,
                    },
                )
                payload = save_last()
                if improved:
                    _atomic_torch_save(checkpoint_best, payload)
            elif iteration % checkpoint_interval == 0:
                save_last()
    except KeyboardInterrupt:
        status = "interrupted"
    except Exception as error:
        status = "failed"
        _append_jsonl(
            log_path,
            {
                "event": "failure",
                "iteration": iteration,
                "error_type": type(error).__name__,
                "error": str(error),
                "timestamp": utc_now(),
            },
        )
        try:
            save_last()
        finally:
            summary = {
                **common_summary,
                "status": status,
                "completed_iters": iteration,
                "best_val_loss": best_val_loss,
                "last_train_loss": last_train_loss,
                "last_eval": last_eval,
                "elapsed_seconds": total_elapsed(),
                "session_elapsed_seconds": time.monotonic() - session_start,
                "tokens_seen": iteration * tokens_per_iteration,
                "error_type": type(error).__name__,
                "error": str(error),
                "finished_at": utc_now(),
            }
            _atomic_json(summary_path, summary)
        raise

    save_last()
    summary = {
        **common_summary,
        "status": status,
        "completed_iters": iteration,
        "best_val_loss": best_val_loss,
        "last_train_loss": last_train_loss,
        "last_eval": last_eval,
        "elapsed_seconds": total_elapsed(),
        "session_elapsed_seconds": time.monotonic() - session_start,
        "tokens_seen": iteration * tokens_per_iteration,
        "checkpoint_last": str(checkpoint_last),
        "checkpoint_best": str(checkpoint_best) if checkpoint_best.exists() else None,
        "finished_at": utc_now(),
    }
    _append_jsonl(
        log_path,
        {
            "event": "finish",
            "iteration": iteration,
            "status": status,
            "elapsed_seconds": summary["elapsed_seconds"],
            "timestamp": utc_now(),
        },
    )
    _atomic_json(summary_path, summary)
    return summary


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True, help="Pipeline JSON configuration")
    return parser


def main() -> int:
    args = build_argument_parser().parse_args()
    summary = run_pretraining(args.config)
    print(json.dumps(summary, indent=2, sort_keys=True, default=str))
    return 0 if summary.get("status") != "failed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
