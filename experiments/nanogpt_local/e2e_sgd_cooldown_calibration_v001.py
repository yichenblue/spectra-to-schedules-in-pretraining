"""Long-horizon 30M nanoGPT cooldown-age calibration.

The experiment trains one common constant-LR prefix and takes ephemeral CPU
snapshots at three source ages.  From every source snapshot it runs three
independent future-tape replicates.  Each replicate is a paired fork: one arm
continues at ``eta0`` and the other immediately drops to ``eta0 / sqrt(10)``.
The two arms in a pair reuse the same future tape, while every other tape
segment is disjoint.  No checkpoint is ever written to disk.

This module contains no submission or networking capability.  Its result JSON
embeds the loss trace and all per-context validation losses as compressed,
hashed columnar arrays so that a lean worker response remains self-contained.
"""

from __future__ import annotations

import base64
import copy
import hashlib
import json
import math
import os
import time
import traceback
import zlib
from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F

try:
    from . import e2e_sgd_b0_v001 as b0
except ImportError:  # pragma: no cover - direct script execution
    import e2e_sgd_b0_v001 as b0


RUN_ID = "nanogpt30m-e2e-sgd-cooldown-calibration-v001"
CONFIG_SCHEMA = "nanogpt30m_e2e_sgd_cooldown_calibration_config_v001"
PREFLIGHT_SCHEMA = "nanogpt30m_e2e_sgd_cooldown_calibration_preflight_v001"
RESULT_SCHEMA = "nanogpt30m_e2e_sgd_cooldown_calibration_result_v001"
PRIMARY_PREDICTION_ID = "A2_EXTERNAL"
STAGE = "pilot_long_horizon_cooldown_calibration"

BLOCK_SIZE = 256
BATCH_SIZE = 8
ALIGNED_STRIDE = 256
ETA0 = 0.005941406469904266
ETA_COOLDOWN = ETA0 / math.sqrt(10.0)
SOURCE_UPDATES = (49_152, 98_304, 196_608)
FUTURE_REPLICATES = 3
TAIL_UPDATES = 12_288
EVALUATION_TICKS = (0, 64, 128, 256, 512, 1_024, 2_048, 4_096, 8_192, 12_288)
VALIDATION_CONTEXTS = 1_024
MAXIMUM_WALL_SECONDS = 16_800
PREFIX_UPDATES = SOURCE_UPDATES[-1]
FUTURE_SLICE_COUNT = len(SOURCE_UPDATES) * FUTURE_REPLICATES
TAPE_UPDATES = PREFIX_UPDATES + FUTURE_SLICE_COUNT * TAIL_UPDATES
TAPE_CONTEXTS = TAPE_UPDATES * BATCH_SIZE
ACTUAL_UPDATE_CALLS = PREFIX_UPDATES + 2 * FUTURE_SLICE_COUNT * TAIL_UPDATES
PROCESSED_TOKENS = ACTUAL_UPDATE_CALLS * BATCH_SIZE * BLOCK_SIZE
EXPECTED_EVALUATION_ROWS = (
    len(SOURCE_UPDATES) * FUTURE_REPLICATES * 2 * len(EVALUATION_TICKS)
)

ARM_PREFIX = 0
ARM_CONTINUE = 1
ARM_COOLDOWN = 2
ARM_NAMES = {ARM_PREFIX: "prefix", ARM_CONTINUE: "continue", ARM_COOLDOWN: "cooldown"}
EMBEDDED_ARRAY_ENCODING = "zlib_base64_little_endian_raw_v001"

JsonDict = dict[str, Any]
ProgressCallback = Callable[[Mapping[str, Any]], None]


class CooldownConfigError(ValueError):
    """The frozen cooldown calibration configuration is inconsistent."""


class CooldownResultError(ValueError):
    """A result is incomplete, corrupt, or inconsistent with its config."""


class CooldownAbort(RuntimeError):
    """A numerical or wall-time guard stopped execution."""


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise CooldownConfigError(f"{label} must be a JSON object")
    return value


def _clean_config(config: Mapping[str, Any]) -> JsonDict:
    return {key: value for key, value in config.items() if not key.startswith("_")}


def _load_config(config_or_path: str | Path | Mapping[str, Any]) -> JsonDict:
    if isinstance(config_or_path, (str, Path)):
        return b0.load_config(config_or_path)
    config = copy.deepcopy(dict(config_or_path))
    config["_config_hash"] = b0.canonical_hash(_clean_config(config))
    return config


def _hex64(value: Any, label: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise CooldownConfigError(f"{label} must be lowercase hex64")
    return value


def _is_hex64(value: Any) -> bool:
    return (
        type(value) is str
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _exact_sequence(value: Any, expected: Sequence[int], label: str) -> None:
    if not isinstance(value, list) or value != list(expected):
        raise CooldownConfigError(f"{label} must equal {list(expected)!r}")


def _validate_config(config: Mapping[str, Any]) -> None:
    identities = {
        "schema_version": CONFIG_SCHEMA,
        "run_id": RUN_ID,
        "stage": STAGE,
        "primary_prediction_id": PRIMARY_PREDICTION_ID,
    }
    for key, expected in identities.items():
        if config.get(key) != expected:
            raise CooldownConfigError(f"{key} must equal {expected!r}")

    # Some lean run packages keep these declarations in science.json instead
    # of duplicating them in the core config.  When present, they remain hard
    # assertions; their absence does not add any capability to this module.
    if "launch_control" in config:
        launch = _mapping(config.get("launch_control"), "launch_control")
        if launch.get("cluster_submission_authorized") is not False:
            raise CooldownConfigError("core config must remain submission-unauthorized")
        if launch.get("runner_has_no_submission_capability") is not True:
            raise CooldownConfigError("core must declare no submission capability")
    if "scientific_scope" in config:
        scope = _mapping(config.get("scientific_scope"), "scientific_scope")
        if scope.get("theorem_facing") is not False:
            raise CooldownConfigError("end-to-end cooldown calibration is not theorem-facing")
        if scope.get("evidence_class") not in (None, PRIMARY_PREDICTION_ID):
            raise CooldownConfigError("scientific_scope.evidence_class changed")

    data = _mapping(config.get("data"), "data")
    for role, dtype in (("train", "uint16"), ("validation", "uint16")):
        spec = _mapping(data.get(role), f"data.{role}")
        if type(spec.get("path")) is not str or not spec["path"]:
            raise CooldownConfigError(f"data.{role}.path must be nonempty")
        _hex64(spec.get("sha256"), f"data.{role}.sha256")
        if type(spec.get("token_count")) is not int or spec["token_count"] <= BLOCK_SIZE:
            raise CooldownConfigError(f"data.{role}.token_count is invalid")
        if spec.get("dtype") not in (dtype, f"{dtype}_le"):
            raise CooldownConfigError(f"data.{role}.dtype must be little-endian {dtype}")
    probe = _mapping(data.get("validation_probe"), "data.validation_probe")
    if type(probe.get("path")) is not str or not probe["path"]:
        raise CooldownConfigError("data.validation_probe.path must be nonempty")
    _hex64(probe.get("sha256"), "data.validation_probe.sha256")
    if probe.get("context_count", probe.get("count")) != VALIDATION_CONTEXTS:
        raise CooldownConfigError("validation probe must contain exactly 1024 contexts")
    if probe.get("dtype") not in ("int64", "int64_le"):
        raise CooldownConfigError("validation probe offsets must be int64")

    model = _mapping(config.get("model"), "model")
    expected_model = {
        "family": "nanogpt",
        "block_size": BLOCK_SIZE,
        "vocab_size": 50_304,
        "dropout": 0.0,
        "bias": False,
    }
    for key, expected in expected_model.items():
        if model.get(key) != expected or type(model.get(key)) is not type(expected):
            raise CooldownConfigError(f"model.{key} must equal {expected!r}")
    aliases = {
        "parameters": ("parameters", "expected_parameter_count", 30_036_864),
        "layers": ("layers", "n_layer", 6),
        "heads": ("heads", "n_head", 6),
        "embedding": ("embedding", "n_embd", 384),
    }
    for label, (primary, alternate, expected) in aliases.items():
        value = model.get(primary, model.get(alternate))
        if type(value) is not int or value != expected:
            raise CooldownConfigError(f"model.{label} must equal {expected!r}")

    optimizer = _mapping(config.get("optimizer"), "optimizer")
    expected_optimizer = {
        "name": "sgd",
        "momentum": 0.0,
        "dampening": 0.0,
        "nesterov": False,
        "weight_decay": 0.0,
        "gradient_clip": 0.0,
        "foreach": False,
    }
    for key, expected in expected_optimizer.items():
        if optimizer.get(key) != expected or type(optimizer.get(key)) is not type(expected):
            raise CooldownConfigError(f"optimizer.{key} must equal {expected!r}")
    configured_eta = optimizer.get("learning_rate", optimizer.get("eta0"))
    if type(configured_eta) is not float or configured_eta != ETA0:
        raise CooldownConfigError(f"optimizer learning rate must equal {ETA0!r}")

    runtime = _mapping(config.get("runtime"), "runtime")
    runtime_expected = {
        "training_device": "cuda",
        "preflight_device": "cpu",
        "dtype": "float32",
        "amp": False,
        "tf32": False,
        "compile": False,
        "deterministic": True,
        "maximum_wall_seconds": MAXIMUM_WALL_SECONDS,
    }
    for key, expected in runtime_expected.items():
        if runtime.get(key) != expected or type(runtime.get(key)) is not type(expected):
            raise CooldownConfigError(f"runtime.{key} must equal {expected!r}")
    if runtime.get("microbatch_size", BATCH_SIZE) != BATCH_SIZE:
        raise CooldownConfigError("runtime.microbatch_size must equal 8")
    if runtime.get("grad_accum_steps", 1) != 1:
        raise CooldownConfigError("runtime.grad_accum_steps must equal 1")

    design = _mapping(config.get("design"), "design")
    scalar_expected = {
        "block_size": BLOCK_SIZE,
        "batch_size": BATCH_SIZE,
        "aligned_stride": ALIGNED_STRIDE,
        "future_replicates": FUTURE_REPLICATES,
        "tail_updates": TAIL_UPDATES,
        "validation_batch_size": 32,
    }
    for key, expected in scalar_expected.items():
        if design.get(key) != expected or type(design.get(key)) is not type(expected):
            raise CooldownConfigError(f"design.{key} must equal {expected!r}")
    if design.get("snapshot_policy") not in (
        "ephemeral_cpu_only",
        "exact_model_optimizer_and_rng_restore_no_resume",
    ):
        raise CooldownConfigError("design.snapshot_policy changed")
    _exact_sequence(design.get("source_updates"), SOURCE_UPDATES, "design.source_updates")
    _exact_sequence(
        design.get("evaluation_ticks"), EVALUATION_TICKS, "design.evaluation_ticks"
    )
    for key in ("training_tape_seed", "initialization_seed"):
        if type(design.get(key)) is not int:
            raise CooldownConfigError(f"design.{key} must be an integer")

    eligible = _eligible_aligned_starts(int(data["train"]["token_count"]))
    if eligible < TAPE_CONTEXTS:
        raise CooldownConfigError(
            f"expanded train corpus exposes {eligible} aligned contexts; "
            f"at least {TAPE_CONTEXTS} are required"
        )

    artifacts = _mapping(config.get("artifacts", {}), "artifacts")
    if artifacts.get("checkpoint_policy", "ephemeral_cpu_only") not in (
        "ephemeral_cpu_only",
        "ephemeral_exact_fork_snapshots_only",
    ):
        raise CooldownConfigError("disk checkpoints are outside this experiment")


def _eligible_aligned_starts(token_count: int) -> int:
    """Number of stride-256 starts that expose both x and next-token y."""

    if token_count < BLOCK_SIZE + 1:
        return 0
    return (int(token_count) - (BLOCK_SIZE + 1)) // ALIGNED_STRIDE + 1


def _array_sha256(values: np.ndarray) -> str:
    array = np.ascontiguousarray(values)
    return hashlib.sha256(array.tobytes(order="C")).hexdigest()


def _make_training_tape(token_count: int, seed: int) -> np.ndarray:
    """Make the single frozen without-replacement tape of aligned starts."""

    eligible = _eligible_aligned_starts(int(token_count))
    if eligible < TAPE_CONTEXTS:
        raise CooldownConfigError("train corpus is too small for the disjoint tape")
    generator = np.random.Generator(np.random.PCG64(int(seed)))
    selected = generator.permutation(eligible)[:TAPE_CONTEXTS]
    offsets = np.asarray(selected, dtype=np.int64) * ALIGNED_STRIDE
    tape = np.ascontiguousarray(offsets.reshape(TAPE_UPDATES, BATCH_SIZE))
    if np.unique(tape).size != tape.size:
        raise RuntimeError("without-replacement tape contains a duplicate context")
    return tape


def _future_tape_bounds(source_update: int, replicate: int) -> tuple[int, int]:
    try:
        age_index = SOURCE_UPDATES.index(int(source_update))
    except ValueError as error:
        raise CooldownConfigError(f"unknown source update {source_update}") from error
    if type(replicate) is not int or not 0 <= replicate < FUTURE_REPLICATES:
        raise CooldownConfigError("future replicate is outside [0, 3)")
    slice_index = age_index * FUTURE_REPLICATES + replicate
    start = PREFIX_UPDATES + slice_index * TAIL_UPDATES
    return start, start + TAIL_UPDATES


def _tape_manifest(tape: np.ndarray, token_count: int, seed: int) -> JsonDict:
    if tape.shape != (TAPE_UPDATES, BATCH_SIZE) or tape.dtype != np.int64:
        raise CooldownResultError("training tape has the wrong shape or dtype")
    segments: list[JsonDict] = []
    prefix = tape[:PREFIX_UPDATES]
    segments.append(
        {
            "kind": "prefix",
            "tape_update_start": 0,
            "tape_update_stop": PREFIX_UPDATES,
            "shape": list(prefix.shape),
            "offsets_sha256": _array_sha256(prefix),
        }
    )
    for source_update in SOURCE_UPDATES:
        for replicate in range(FUTURE_REPLICATES):
            start, stop = _future_tape_bounds(source_update, replicate)
            segment = tape[start:stop]
            segments.append(
                {
                    "kind": "future",
                    "source_update": source_update,
                    "replicate": replicate,
                    "tape_update_start": start,
                    "tape_update_stop": stop,
                    "shape": list(segment.shape),
                    "offsets_sha256": _array_sha256(segment),
                    "paired_arm_reuse": ["continue", "cooldown"],
                }
            )
    return {
        "algorithm": "PCG64(seed).permutation(eligible_aligned_starts)[:required]",
        "seed": int(seed),
        "aligned_stride": ALIGNED_STRIDE,
        "eligible_aligned_starts": _eligible_aligned_starts(token_count),
        "shape": list(tape.shape),
        "unique_contexts": int(tape.size),
        "offsets_sha256": _array_sha256(tape),
        "segments": segments,
    }


def _resolve_data_path(spec: Mapping[str, Any]) -> Path:
    return b0.resolve_path(str(spec["path"]))


def _load_uint16_stream(spec: Mapping[str, Any], label: str) -> tuple[np.memmap, JsonDict]:
    path = _resolve_data_path(spec)
    if not path.is_file():
        raise FileNotFoundError(f"{label} token stream is absent: {path}")
    digest = b0.sha256_file(path)
    if digest != spec["sha256"]:
        raise CooldownConfigError(f"{label} token SHA256 changed")
    if path.stat().st_size % np.dtype(np.uint16).itemsize:
        raise CooldownConfigError(f"{label} token stream has an odd byte count")
    values = np.memmap(path, dtype=np.uint16, mode="r")
    if values.size != int(spec["token_count"]):
        raise CooldownConfigError(f"{label} token count changed")
    return values, {
        "path": str(path),
        "sha256": digest,
        "token_count": int(values.size),
        "bytes": int(path.stat().st_size),
    }


def _load_validation_probe(
    spec: Mapping[str, Any], validation_token_count: int
) -> tuple[np.ndarray, JsonDict]:
    path = _resolve_data_path(spec)
    if not path.is_file():
        raise FileNotFoundError(f"validation probe is absent: {path}")
    digest = b0.sha256_file(path)
    if digest != spec["sha256"]:
        raise CooldownConfigError("validation probe SHA256 changed")
    offsets = np.load(path, allow_pickle=False)
    if offsets.dtype != np.dtype(np.int64) or offsets.shape != (VALIDATION_CONTEXTS,):
        raise CooldownConfigError("validation probe must be an int64 vector of length 1024")
    offsets = np.ascontiguousarray(offsets)
    if np.unique(offsets).size != VALIDATION_CONTEXTS:
        raise CooldownConfigError("validation probe contains duplicate contexts")
    if int(np.min(offsets)) < 0 or int(np.max(offsets)) + BLOCK_SIZE >= validation_token_count:
        raise CooldownConfigError("validation probe context exceeds the validation stream")
    return offsets, {
        "path": str(path),
        "sha256": digest,
        "dtype": "int64",
        "context_count": VALIDATION_CONTEXTS,
        "offsets_sha256": _array_sha256(offsets),
        "unique_offsets": int(np.unique(offsets).size),
    }


def _prepare_inputs(config: Mapping[str, Any]) -> tuple[np.memmap, np.memmap, np.ndarray, np.ndarray, JsonDict]:
    train_values, train_info = _load_uint16_stream(config["data"]["train"], "train")
    validation_values, validation_info = _load_uint16_stream(
        config["data"]["validation"], "validation"
    )
    probe_offsets, probe_info = _load_validation_probe(
        config["data"]["validation_probe"], len(validation_values)
    )
    vocabulary_size = int(config["model"]["vocab_size"])
    if int(np.max(train_values)) >= vocabulary_size:
        raise CooldownConfigError("train token exceeds model vocabulary")
    if int(np.max(validation_values)) >= vocabulary_size:
        raise CooldownConfigError("validation token exceeds model vocabulary")
    tape = _make_training_tape(
        len(train_values), int(config["design"]["training_tape_seed"])
    )
    return train_values, validation_values, probe_offsets, tape, {
        "train": train_info,
        "validation": validation_info,
        "validation_probe": probe_info,
    }


def _model_config(config: Mapping[str, Any]) -> b0.GPTConfig:
    model = config["model"]
    return b0.GPTConfig(
        block_size=int(model["block_size"]),
        vocab_size=int(model["vocab_size"]),
        n_layer=int(model.get("n_layer", model.get("layers"))),
        n_head=int(model.get("n_head", model.get("heads"))),
        n_embd=int(model.get("n_embd", model.get("embedding"))),
        dropout=float(model["dropout"]),
        bias=bool(model["bias"]),
    )


def _new_model(config: Mapping[str, Any], device: torch.device) -> torch.nn.Module:
    b0._seed_everything_strict(int(config["design"]["initialization_seed"]))
    model = b0.GPT(_model_config(config))
    count = sum(parameter.numel() for parameter in model.parameters())
    if count != 30_036_864:
        raise RuntimeError(f"30M model parameter count changed: {count}")
    if model.lm_head.weight.data_ptr() != model.transformer.wte.weight.data_ptr():
        raise RuntimeError("embedding and language-model head are no longer tied")
    return model.to(device)


def _set_learning_rate(optimizer: torch.optim.Optimizer, learning_rate: float) -> None:
    if not math.isfinite(learning_rate) or learning_rate <= 0.0:
        raise CooldownAbort("nonpositive_or_nonfinite_learning_rate")
    for group in optimizer.param_groups:
        group["lr"] = float(learning_rate)
    if any(float(group["lr"]) != float(learning_rate) for group in optimizer.param_groups):
        raise RuntimeError("optimizer learning-rate synchronization failed")


def _configure_deterministic_cuda() -> None:
    """Freeze the cuBLAS contract before any CUDA context is created."""

    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8":
        raise CooldownConfigError(
            "CUBLAS_WORKSPACE_CONFIG must equal ':4096:8' before CUDA initialization"
        )
    torch.use_deterministic_algorithms(True)


def _training_step(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    train_values: np.ndarray,
    offsets: np.ndarray,
    device: torch.device,
    learning_rate: float,
) -> float:
    """Execute exactly one B8 plain-SGD update and return raw batch CE."""

    if offsets.shape != (BATCH_SIZE,):
        raise CooldownAbort("training tape row has the wrong batch shape")
    _set_learning_rate(optimizer, learning_rate)
    model.train()
    optimizer.zero_grad(set_to_none=True)
    x, y = b0._batch_from_offsets(train_values, offsets, BLOCK_SIZE, device)
    _, loss, _ = model(x, y)
    if loss is None or not bool(torch.isfinite(loss).detach().cpu()):
        raise CooldownAbort("nonfinite_training_loss")
    loss.backward()
    optimizer.step()
    if optimizer.state_dict()["state"]:
        raise RuntimeError("plain SGD unexpectedly created optimizer state")
    return float(loss.detach().cpu())


@torch.no_grad()
def _evaluate_contexts(
    model: torch.nn.Module,
    validation_values: np.ndarray,
    probe_offsets: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    """Return one mean token CE for each fixed validation context."""

    was_training = model.training
    model.eval()
    losses: list[np.ndarray] = []
    for start in range(0, len(probe_offsets), batch_size):
        offsets = probe_offsets[start : start + batch_size]
        x, y = b0._batch_from_offsets(validation_values, offsets, BLOCK_SIZE, device)
        logits, _, _ = model(x, None)
        token_losses = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            y.long().reshape(-1),
            reduction="none",
        ).reshape(len(offsets), BLOCK_SIZE)
        context_losses = torch.mean(token_losses, dim=1)
        if not bool(torch.all(torch.isfinite(context_losses)).detach().cpu()):
            raise CooldownAbort("nonfinite_validation_loss")
        losses.append(context_losses.detach().float().cpu().numpy())
    model.train(was_training)
    result = np.ascontiguousarray(np.concatenate(losses), dtype=np.float32)
    if result.shape != (VALIDATION_CONTEXTS,):
        raise RuntimeError("validation evaluation returned the wrong number of contexts")
    return result


def _capture_cpu_snapshot(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: tensor.detach().cpu().contiguous().clone()
        for name, tensor in model.state_dict().items()
    }


def _restore_cpu_snapshot(
    model: torch.nn.Module, snapshot: Mapping[str, torch.Tensor]
) -> None:
    model.load_state_dict(snapshot, strict=True)


def _state_dict_sha256(snapshot: Mapping[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name in sorted(snapshot):
        tensor = snapshot[name].detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(b"\0")
        digest.update(np.asarray(tensor.shape, dtype="<i8").tobytes())
        digest.update(tensor.numpy().tobytes(order="C"))
    return digest.hexdigest()


def _capture_rng_state(device: torch.device) -> JsonDict:
    state: JsonDict = {"cpu": torch.get_rng_state().cpu().clone()}
    if device.type == "cuda":
        state["cuda"] = [value.cpu().clone() for value in torch.cuda.get_rng_state_all()]
    return state


def _restore_rng_state(state: Mapping[str, Any], device: torch.device) -> None:
    torch.set_rng_state(state["cpu"])
    if device.type == "cuda" and "cuda" in state:
        torch.cuda.set_rng_state_all(state["cuda"])


def _rng_state_sha256(state: Mapping[str, Any]) -> str:
    digest = hashlib.sha256()
    for name in ("cpu", "cuda"):
        if name not in state:
            continue
        values = state[name] if isinstance(state[name], list) else [state[name]]
        digest.update(name.encode("ascii"))
        digest.update(b"\0")
        for index, value in enumerate(values):
            tensor = value.detach().cpu().contiguous()
            digest.update(np.asarray([index], dtype="<i8").tobytes())
            digest.update(tensor.numpy().tobytes(order="C"))
    return digest.hexdigest()


def _encode_array(values: np.ndarray) -> JsonDict:
    array = np.ascontiguousarray(values)
    if array.dtype.byteorder == ">" or (
        array.dtype.byteorder == "=" and not np.little_endian
    ):
        array = array.byteswap().view(array.dtype.newbyteorder("<"))
    dtype = array.dtype.newbyteorder("<")
    array = np.ascontiguousarray(array.astype(dtype, copy=False))
    raw = array.tobytes(order="C")
    compressed = zlib.compress(raw, level=9)
    return {
        "encoding": EMBEDDED_ARRAY_ENCODING,
        "dtype": array.dtype.str,
        "shape": list(array.shape),
        "raw_bytes": len(raw),
        "compressed_bytes": len(compressed),
        "raw_sha256": hashlib.sha256(raw).hexdigest(),
        "data_base64": base64.b64encode(compressed).decode("ascii"),
    }


def _decode_array(record: Mapping[str, Any], label: str) -> np.ndarray:
    if record.get("encoding") != EMBEDDED_ARRAY_ENCODING:
        raise CooldownResultError(f"{label} encoding changed")
    try:
        dtype = np.dtype(str(record["dtype"]))
        shape = tuple(int(value) for value in record["shape"])
        compressed = base64.b64decode(str(record["data_base64"]), validate=True)
        raw = zlib.decompress(compressed)
    except Exception as error:
        raise CooldownResultError(f"{label} cannot be decoded") from error
    if len(raw) != int(record.get("raw_bytes", -1)):
        raise CooldownResultError(f"{label} raw byte count changed")
    if len(compressed) != int(record.get("compressed_bytes", -1)):
        raise CooldownResultError(f"{label} compressed byte count changed")
    if hashlib.sha256(raw).hexdigest() != record.get("raw_sha256"):
        raise CooldownResultError(f"{label} raw SHA256 changed")
    expected = math.prod(shape) * dtype.itemsize
    if expected != len(raw):
        raise CooldownResultError(f"{label} shape and dtype do not match bytes")
    return np.frombuffer(raw, dtype=dtype).reshape(shape).copy()


def _expected_trace_coordinates() -> dict[str, np.ndarray]:
    columns: dict[str, list[Any]] = {
        "phase": [],
        "source_update": [],
        "replicate": [],
        "arm": [],
        "local_update": [],
        "tape_update_index": [],
        "learning_rate": [],
    }
    source_set = set(SOURCE_UPDATES)
    for prefix_update in range(1, PREFIX_UPDATES + 1):
        columns["phase"].append(0)
        columns["source_update"].append(0)
        columns["replicate"].append(-1)
        columns["arm"].append(ARM_PREFIX)
        columns["local_update"].append(prefix_update)
        columns["tape_update_index"].append(prefix_update - 1)
        columns["learning_rate"].append(ETA0)
        if prefix_update not in source_set:
            continue
        for replicate in range(FUTURE_REPLICATES):
            tape_start, _ = _future_tape_bounds(prefix_update, replicate)
            for arm, learning_rate in (
                (ARM_CONTINUE, ETA0),
                (ARM_COOLDOWN, ETA_COOLDOWN),
            ):
                for local_update in range(1, TAIL_UPDATES + 1):
                    columns["phase"].append(1)
                    columns["source_update"].append(prefix_update)
                    columns["replicate"].append(replicate)
                    columns["arm"].append(arm)
                    columns["local_update"].append(local_update)
                    columns["tape_update_index"].append(tape_start + local_update - 1)
                    columns["learning_rate"].append(learning_rate)
    arrays = {
        "phase": np.asarray(columns["phase"], dtype=np.uint8),
        "source_update": np.asarray(columns["source_update"], dtype=np.int32),
        "replicate": np.asarray(columns["replicate"], dtype=np.int8),
        "arm": np.asarray(columns["arm"], dtype=np.uint8),
        "local_update": np.asarray(columns["local_update"], dtype=np.int32),
        "tape_update_index": np.asarray(columns["tape_update_index"], dtype=np.int32),
        "learning_rate": np.asarray(columns["learning_rate"], dtype=np.float64),
    }
    if any(len(value) != ACTUAL_UPDATE_CALLS for value in arrays.values()):
        raise RuntimeError("expected trace construction has the wrong length")
    return arrays


def _trace_payload(columns: Mapping[str, np.ndarray]) -> JsonDict:
    encoded = {name: _encode_array(value) for name, value in columns.items()}
    unsigned = {
        "schema_version": "nanogpt30m_e2e_sgd_cooldown_training_trace_v001",
        "row_count": int(len(next(iter(columns.values()), []))),
        "column_order": list(columns),
        "columns": encoded,
    }
    return {**unsigned, "sha256": b0.canonical_hash(unsigned)}


def _evaluation_payload(
    metadata: Sequence[Mapping[str, Any]], context_losses: Sequence[np.ndarray]
) -> JsonDict:
    matrix = (
        np.ascontiguousarray(np.stack(context_losses), dtype=np.float32)
        if context_losses
        else np.empty((0, VALIDATION_CONTEXTS), dtype=np.float32)
    )
    unsigned = {
        "schema_version": "nanogpt30m_e2e_sgd_cooldown_evaluations_v001",
        "row_count": len(metadata),
        "context_count": VALIDATION_CONTEXTS,
        "rows": [dict(row) for row in metadata],
        "context_cross_entropies": _encode_array(matrix),
    }
    return {**unsigned, "sha256": b0.canonical_hash(unsigned)}


def _evaluation_row(
    source_update: int,
    replicate: int,
    arm: int,
    tail_update: int,
    losses: np.ndarray,
) -> JsonDict:
    values = np.asarray(losses, dtype=np.float64)
    mean = float(np.mean(values))
    standard_error = float(np.std(values, ddof=1) / math.sqrt(len(values)))
    learning_rate = ETA0 if arm == ARM_CONTINUE else ETA_COOLDOWN
    return {
        "source_update": int(source_update),
        "replicate": int(replicate),
        "arm": ARM_NAMES[arm],
        "tail_update": int(tail_update),
        "effective_optimizer_update": int(source_update + tail_update),
        "branch_tokens_seen": int(tail_update * BATCH_SIZE * BLOCK_SIZE),
        "branch_intrinsic_time": float(tail_update * learning_rate),
        "learning_rate": float(learning_rate),
        "validation_cross_entropy": mean,
        "validation_standard_error": standard_error,
        "context_cross_entropies_sha256": _array_sha256(
            np.ascontiguousarray(losses, dtype=np.float32)
        ),
    }


def _snapshot_ledger_identity_exact(ledger: Any) -> bool:
    if not isinstance(ledger, list) or len(ledger) != len(SOURCE_UPDATES):
        return False
    for source_update, entry in zip(SOURCE_UPDATES, ledger, strict=True):
        if not isinstance(entry, Mapping) or set(entry) != {
            "source_update", "state_sha256", "rng_sha256", "storage",
            "written_to_disk", "branch_starts",
        }:
            return False
        state_hash = entry.get("state_sha256")
        rng_hash = entry.get("rng_sha256")
        if (
            entry.get("source_update") != source_update
            or not _is_hex64(state_hash)
            or not _is_hex64(rng_hash)
            or entry.get("storage") != "ephemeral_cpu_only"
            or entry.get("written_to_disk") is not False
        ):
            return False
        starts = entry.get("branch_starts")
        expected_coordinates = [
            (replicate, ARM_NAMES[arm])
            for replicate in range(FUTURE_REPLICATES)
            for arm in (ARM_CONTINUE, ARM_COOLDOWN)
        ]
        if not isinstance(starts, list) or len(starts) != len(expected_coordinates):
            return False
        for start, (replicate, arm_name) in zip(
            starts, expected_coordinates, strict=True
        ):
            if not isinstance(start, Mapping) or start != {
                "replicate": replicate,
                "arm": arm_name,
                "state_sha256": state_hash,
                "rng_sha256": rng_hash,
                "optimizer_state_empty": True,
            }:
                return False
    return True


def _analysis_from_evaluations(
    rows: Sequence[Mapping[str, Any]],
    context_matrix: np.ndarray,
    snapshot_ledger: Any,
) -> JsonDict:
    lookup: dict[tuple[int, int, str, int], int] = {}
    for index, row in enumerate(rows):
        key = (
            int(row["source_update"]),
            int(row["replicate"]),
            str(row["arm"]),
            int(row["tail_update"]),
        )
        if key in lookup:
            raise CooldownResultError(f"duplicate evaluation row {key!r}")
        lookup[key] = index
    gains: list[JsonDict] = []
    tick0_exact = _snapshot_ledger_identity_exact(snapshot_ledger)
    for source_update in SOURCE_UPDATES:
        for replicate in range(FUTURE_REPLICATES):
            for tick in EVALUATION_TICKS:
                continue_index = lookup.get((source_update, replicate, "continue", tick))
                cooldown_index = lookup.get((source_update, replicate, "cooldown", tick))
                if continue_index is None or cooldown_index is None:
                    continue
                paired = (
                    context_matrix[continue_index].astype(np.float64)
                    - context_matrix[cooldown_index].astype(np.float64)
                )
                gain = float(np.mean(paired))
                standard_error = float(np.std(paired, ddof=1) / math.sqrt(len(paired)))
                if tick == 0:
                    tick0_exact = tick0_exact and bool(np.array_equal(
                        context_matrix[continue_index], context_matrix[cooldown_index]
                    ))
                gains.append(
                    {
                        "source_update": source_update,
                        "replicate": replicate,
                        "tail_update": tick,
                        "cooldown_gain_nats": gain,
                        "paired_context_standard_error": standard_error,
                    }
                )
    terminal = [row for row in gains if row["tail_update"] == TAIL_UPDATES]
    by_age = []
    for source_update in SOURCE_UPDATES:
        values = [
            float(row["cooldown_gain_nats"])
            for row in terminal
            if row["source_update"] == source_update
        ]
        replicate_standard_error = (
            float(np.std(values, ddof=1) / math.sqrt(len(values)))
            if len(values) > 1
            else None
        )
        by_age.append(
            {
                "source_update": source_update,
                "replicate_count": len(values),
                "replicate_values": values,
                "mean_cooldown_gain_nats": (
                    float(np.mean(values)) if values else None
                ),
                "median_cooldown_gain_nats": float(np.median(values)) if values else None,
                "replicate_standard_error": replicate_standard_error,
                "positive_gain_replicates": sum(value > 0.0 for value in values),
            }
        )
    complete = len(rows) == EXPECTED_EVALUATION_ROWS and len(gains) == (
        len(SOURCE_UPDATES) * FUTURE_REPLICATES * len(EVALUATION_TICKS)
    )
    finite = bool(
        np.all(np.isfinite(context_matrix))
        and all(math.isfinite(float(row["cooldown_gain_nats"])) for row in gains)
    )
    return {
        "estimand": "cooldown_gain_nats=CE_continue-CE_cooldown; positive favors cooldown",
        "cooldown_gain_rows": gains,
        "terminal_summary": by_age,
        "gates": {
            "complete_grid": complete,
            "tick0_pair_identity_exact": tick0_exact,
            "all_values_finite": finite,
        },
        "calibration_note": (
            "No significance threshold is promoted to a confirmatory gate in this pilot."
        ),
    }


def _emit(callback: ProgressCallback | None, event: Mapping[str, Any]) -> None:
    if callback is not None:
        callback(dict(event))


def _check_deadline(deadline: float) -> None:
    if time.monotonic() >= deadline:
        raise CooldownAbort("maximum_wall_seconds_reached")


def _failure(error: BaseException) -> JsonDict:
    return {
        "type": type(error).__name__,
        "message": str(error),
        "traceback_tail": traceback.format_exception(error)[-8:],
    }


def _artifact_result_path(config: Mapping[str, Any], output_dir: str | Path | None) -> Path:
    artifacts = _mapping(config.get("artifacts", {}), "artifacts")
    filename = artifacts.get("result", "result.json")
    if type(filename) is not str or not filename or Path(filename).name != filename:
        raise CooldownConfigError("artifacts.result must be a plain filename")
    if output_dir is None:
        configured = artifacts.get("output_dir")
        if type(configured) is not str or not configured:
            raise CooldownConfigError("output_dir argument or artifacts.output_dir is required")
        root = b0.resolve_path(configured)
    else:
        root = Path(output_dir).expanduser().resolve()
    return root / filename


def preflight(
    config_or_path: str | Path | Mapping[str, Any],
    output_dir: str | Path | None = None,
) -> JsonDict:
    started = time.monotonic()
    config = _load_config(config_or_path)
    _validate_config(config)
    train, validation, probe, tape, data_info = _prepare_inputs(config)
    model = _new_model(config, torch.device("cpu"))
    optimizer = b0._build_optimizer(model, ETA0)
    tape_manifest = _tape_manifest(
        tape, len(train), int(config["design"]["training_tape_seed"])
    )
    checks = {
        "data_hashes_and_counts": True,
        "expanded_corpus_has_enough_disjoint_contexts": tape_manifest["eligible_aligned_starts"]
        >= TAPE_CONTEXTS,
        "global_tape_without_replacement": int(np.unique(tape).size) == tape.size,
        "prefix_and_nine_future_slices_declared": len(tape_manifest["segments"]) == 10,
        "validation_probe_unique_and_in_bounds": len(probe) == len(np.unique(probe))
        == VALIDATION_CONTEXTS,
        "model_parameter_count": sum(p.numel() for p in model.parameters()) == 30_036_864,
        "plain_sgd_has_no_state": not optimizer.state_dict()["state"],
        "expected_update_accounting": ACTUAL_UPDATE_CALLS == 417_792,
        "expected_token_accounting": PROCESSED_TOKENS == 855_638_016,
        "no_disk_checkpoint_policy": config.get("artifacts", {}).get(
            "checkpoint_policy", "ephemeral_cpu_only"
        )
        in ("ephemeral_cpu_only", "ephemeral_exact_fork_snapshots_only"),
    }
    result: JsonDict = {
        "schema_version": PREFLIGHT_SCHEMA,
        "run_id": RUN_ID,
        "status": "passed" if all(checks.values()) else "failed",
        "config_sha256": config["_config_hash"],
        "checks": checks,
        "data": data_info,
        "model": {"config": asdict(model.config), "parameters": 30_036_864},
        "design": {
            "source_updates": list(SOURCE_UPDATES),
            "future_replicates": FUTURE_REPLICATES,
            "tail_updates": TAIL_UPDATES,
            "evaluation_ticks": list(EVALUATION_TICKS),
            "tape_updates": TAPE_UPDATES,
            "tape_contexts": TAPE_CONTEXTS,
            "actual_update_calls": ACTUAL_UPDATE_CALLS,
            "processed_tokens": PROCESSED_TOKENS,
        },
        "training_tape_manifest": tape_manifest,
        "elapsed_seconds": time.monotonic() - started,
        "scientific_training_executed": False,
        "gpu_submission_performed": False,
    }
    del optimizer, model, tape, probe, validation, train
    if output_dir is not None:
        path = Path(output_dir).expanduser().resolve() / "preflight.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        b0._atomic_strict_json(path, result)
    return result


def _run_training(
    config: Mapping[str, Any],
    train_values: np.ndarray,
    validation_values: np.ndarray,
    probe_offsets: np.ndarray,
    tape: np.ndarray,
    device: torch.device,
    deadline: float,
    progress_callback: ProgressCallback | None,
) -> tuple[dict[str, np.ndarray], list[JsonDict], list[np.ndarray], list[JsonDict]]:
    expected = _expected_trace_coordinates()
    train_losses = np.empty(ACTUAL_UPDATE_CALLS, dtype=np.float32)
    model = _new_model(config, device)
    optimizer = b0._build_optimizer(model, ETA0)
    validation_batch_size = int(config["design"]["validation_batch_size"])
    trace_cursor = 0
    evaluation_rows: list[JsonDict] = []
    evaluation_losses: list[np.ndarray] = []
    snapshots: list[JsonDict] = []
    source_set = set(SOURCE_UPDATES)

    for prefix_update in range(1, PREFIX_UPDATES + 1):
        _check_deadline(deadline)
        coordinate_index = trace_cursor
        if (
            int(expected["arm"][coordinate_index]) != ARM_PREFIX
            or int(expected["local_update"][coordinate_index]) != prefix_update
        ):
            raise RuntimeError("prefix trace coordinate drift")
        train_losses[trace_cursor] = _training_step(
            model, optimizer, train_values, tape[prefix_update - 1], device, ETA0
        )
        trace_cursor += 1
        if prefix_update % 256 == 0:
            _emit(
                progress_callback,
                {
                    "kind": "progress",
                    "phase": "prefix",
                    "prefix_update": prefix_update,
                    "completed_actual_updates": trace_cursor,
                    "expected_actual_updates": ACTUAL_UPDATE_CALLS,
                },
            )
        if prefix_update not in source_set:
            continue

        _check_deadline(deadline)
        snapshot = _capture_cpu_snapshot(model)
        snapshot_hash = _state_dict_sha256(snapshot)
        rng_state = _capture_rng_state(device)
        rng_hash = _rng_state_sha256(rng_state)
        snapshot_record: JsonDict = {
            "source_update": prefix_update,
            "state_sha256": snapshot_hash,
            "rng_sha256": rng_hash,
            "storage": "ephemeral_cpu_only",
            "written_to_disk": False,
            "branch_starts": [],
        }

        for replicate in range(FUTURE_REPLICATES):
            tape_start, tape_stop = _future_tape_bounds(prefix_update, replicate)
            for arm, learning_rate in (
                (ARM_CONTINUE, ETA0),
                (ARM_COOLDOWN, ETA_COOLDOWN),
            ):
                _restore_cpu_snapshot(model, snapshot)
                _restore_rng_state(rng_state, device)
                optimizer = b0._build_optimizer(model, learning_rate)
                restored_state_hash = _state_dict_sha256(
                    _capture_cpu_snapshot(model)
                )
                restored_rng_hash = _rng_state_sha256(
                    _capture_rng_state(device)
                )
                optimizer_state_empty = not optimizer.state_dict()["state"]
                if (
                    restored_state_hash != snapshot_hash
                    or restored_rng_hash != rng_hash
                    or not optimizer_state_empty
                ):
                    raise RuntimeError("paired branch did not restore exact fork state")
                snapshot_record["branch_starts"].append(
                    {
                        "replicate": replicate,
                        "arm": ARM_NAMES[arm],
                        "state_sha256": restored_state_hash,
                        "rng_sha256": restored_rng_hash,
                        "optimizer_state_empty": optimizer_state_empty,
                    }
                )
                initial_losses = _evaluate_contexts(
                    model,
                    validation_values,
                    probe_offsets,
                    device,
                    validation_batch_size,
                )
                _restore_rng_state(rng_state, device)
                evaluation_rows.append(
                    _evaluation_row(prefix_update, replicate, arm, 0, initial_losses)
                )
                evaluation_losses.append(initial_losses)
                for local_update in range(1, TAIL_UPDATES + 1):
                    _check_deadline(deadline)
                    coordinate_index = trace_cursor
                    expected_identity = (
                        int(expected["source_update"][coordinate_index]),
                        int(expected["replicate"][coordinate_index]),
                        int(expected["arm"][coordinate_index]),
                        int(expected["local_update"][coordinate_index]),
                    )
                    if expected_identity != (prefix_update, replicate, arm, local_update):
                        raise RuntimeError("branch trace coordinate drift")
                    tape_index = tape_start + local_update - 1
                    if tape_index >= tape_stop:
                        raise RuntimeError("future tape slice exhausted")
                    train_losses[trace_cursor] = _training_step(
                        model,
                        optimizer,
                        train_values,
                        tape[tape_index],
                        device,
                        learning_rate,
                    )
                    trace_cursor += 1
                    if local_update in EVALUATION_TICKS[1:]:
                        losses = _evaluate_contexts(
                            model,
                            validation_values,
                            probe_offsets,
                            device,
                            validation_batch_size,
                        )
                        evaluation_rows.append(
                            _evaluation_row(
                                prefix_update, replicate, arm, local_update, losses
                            )
                        )
                        evaluation_losses.append(losses)
                    if local_update % 256 == 0:
                        _emit(
                            progress_callback,
                            {
                                "kind": "progress",
                                "phase": "future",
                                "source_update": prefix_update,
                                "replicate": replicate,
                                "arm": ARM_NAMES[arm],
                                "tail_update": local_update,
                                "completed_actual_updates": trace_cursor,
                                "expected_actual_updates": ACTUAL_UPDATE_CALLS,
                            },
                        )
        snapshots.append(snapshot_record)
        _restore_cpu_snapshot(model, snapshot)
        _restore_rng_state(rng_state, device)
        optimizer = b0._build_optimizer(model, ETA0)
        if _state_dict_sha256(_capture_cpu_snapshot(model)) != snapshot_hash:
            raise RuntimeError("ephemeral source snapshot did not restore exactly")
        del snapshot

    if trace_cursor != ACTUAL_UPDATE_CALLS:
        raise RuntimeError("training completed with the wrong update count")
    if len(evaluation_rows) != EXPECTED_EVALUATION_ROWS:
        raise RuntimeError("training completed with the wrong evaluation count")
    trace_columns = {**expected, "train_cross_entropy": train_losses}
    del optimizer, model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return trace_columns, evaluation_rows, evaluation_losses, snapshots


def _result_base(config: Mapping[str, Any], data_info: Mapping[str, Any], tape: np.ndarray) -> JsonDict:
    return {
        "schema_version": RESULT_SCHEMA,
        "run_id": RUN_ID,
        "status": "running",
        "decision": "INCOMPLETE",
        "config_sha256": config["_config_hash"],
        "data": dict(data_info),
        "model": {
            "family": "nanogpt",
            "parameters": 30_036_864,
            "block_size": BLOCK_SIZE,
            "vocab_size": 50_304,
        },
        "optimizer": {
            "name": "sgd",
            "eta0": ETA0,
            "eta_cooldown": ETA_COOLDOWN,
            "momentum": 0.0,
            "weight_decay": 0.0,
        },
        "design": {
            "source_updates": list(SOURCE_UPDATES),
            "future_replicates": FUTURE_REPLICATES,
            "tail_updates": TAIL_UPDATES,
            "evaluation_ticks": list(EVALUATION_TICKS),
            "validation_contexts": VALIDATION_CONTEXTS,
            "pairing": "same_snapshot_same_future_tape_different_learning_rate",
            "snapshot_policy": "ephemeral_cpu_only",
        },
        "measurements": {
            "training_tape_manifest": _tape_manifest(
                tape,
                int(config["data"]["train"]["token_count"]),
                int(config["design"]["training_tape_seed"]),
            ),
            "training_trace": _trace_payload({}),
            "evaluations": _evaluation_payload([], []),
            "snapshot_ledger": [],
        },
        "analysis": {
            "estimand": "cooldown_gain_nats=CE_continue-CE_cooldown; positive favors cooldown",
            "cooldown_gain_rows": [],
            "terminal_summary": [],
            "gates": {
                "complete_grid": False,
                "tick0_pair_identity_exact": False,
                "all_values_finite": False,
            },
        },
        "counts": {
            "expected_update_calls": ACTUAL_UPDATE_CALLS,
            "actual_update_calls": 0,
            "expected_processed_tokens": PROCESSED_TOKENS,
            "actual_processed_tokens": 0,
            "expected_evaluation_rows": EXPECTED_EVALUATION_ROWS,
            "actual_evaluation_rows": 0,
        },
        "runtime": {},
        "artifacts": {
            "self_contained_result": True,
            "embedded_array_encoding": EMBEDDED_ARRAY_ENCODING,
            "checkpoint_files_written": [],
        },
        "failure": None,
        "gpu_submission_performed": False,
    }


def run_experiment(
    config_or_path: str | Path | Mapping[str, Any],
    output_dir: str | Path | None = None,
    progress_callback: ProgressCallback | None = None,
) -> JsonDict:
    """Execute the frozen calibration locally; never submit external work."""

    started = time.monotonic()
    config = _load_config(config_or_path)
    _validate_config(config)
    train, validation, probe, tape, data_info = _prepare_inputs(config)
    result = _result_base(config, data_info, tape)
    result_path = _artifact_result_path(config, output_dir)
    result_path.parent.mkdir(parents=True, exist_ok=True)
    _configure_deterministic_cuda()
    device = b0.resolve_device(str(config["runtime"]["training_device"]))
    if device.type != "cuda":
        raise CooldownConfigError("scientific execution requires CUDA")
    required_name = str(config["runtime"].get("required_device_name_substring", "A100"))
    device_name = torch.cuda.get_device_name(device)
    if required_name and required_name not in device_name:
        raise CooldownConfigError(
            f"CUDA device {device_name!r} does not contain {required_name!r}"
        )
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    deadline = started + MAXIMUM_WALL_SECONDS
    peak_memory_bytes: int | None = None
    try:
        torch.cuda.reset_peak_memory_stats(device)
        trace_columns, evaluation_rows, evaluation_losses, snapshots = _run_training(
            config,
            train,
            validation,
            probe,
            tape,
            device,
            deadline,
            progress_callback,
        )
        trace = _trace_payload(trace_columns)
        evaluations = _evaluation_payload(evaluation_rows, evaluation_losses)
        context_matrix = np.stack(evaluation_losses).astype(np.float32, copy=False)
        analysis = _analysis_from_evaluations(
            evaluation_rows, context_matrix, snapshots
        )
        peak_memory_bytes = int(torch.cuda.max_memory_allocated(device))
        result["status"] = "completed"
        result["decision"] = "CALIBRATION_COMPLETED"
        result["measurements"]["training_trace"] = trace
        result["measurements"]["evaluations"] = evaluations
        result["measurements"]["snapshot_ledger"] = snapshots
        result["analysis"] = analysis
        result["counts"].update(
            {
                "actual_update_calls": ACTUAL_UPDATE_CALLS,
                "actual_processed_tokens": PROCESSED_TOKENS,
                "actual_evaluation_rows": EXPECTED_EVALUATION_ROWS,
            }
        )
    except Exception as error:
        result["status"] = "incomplete"
        result["decision"] = "INCOMPLETE"
        result["failure"] = _failure(error)
    elapsed_seconds = time.monotonic() - started
    result["runtime"] = {
        "device": str(device),
        "device_name": device_name,
        "dtype": "float32",
        "elapsed_seconds": elapsed_seconds,
        "actual_gpu_hours": elapsed_seconds / 3600.0,
        "resource_budget_passed": bool(
            elapsed_seconds <= MAXIMUM_WALL_SECONDS
            and elapsed_seconds / 3600.0
            <= float(config["runtime"].get("maximum_a100_gpu_hours", 5.0))
        ),
        "maximum_wall_seconds": MAXIMUM_WALL_SECONDS,
        "peak_memory_bytes": peak_memory_bytes,
    }
    if result["status"] == "completed" and not result["runtime"]["resource_budget_passed"]:
        result["status"] = "incomplete"
        result["decision"] = "INCOMPLETE"
        result["failure"] = {
            "type": "CooldownAbort",
            "message": "resource_budget_exceeded",
            "traceback_tail": [],
        }
    b0._atomic_strict_json(result_path, result)
    validate_result(
        config,
        result,
        output_dir=output_dir,
        require_complete=result["status"] == "completed",
    )
    return result


def _load_result(result_or_path: str | Path | Mapping[str, Any]) -> JsonDict:
    if isinstance(result_or_path, (str, Path)):
        with Path(result_or_path).expanduser().resolve().open("r", encoding="utf-8") as handle:
            value = json.load(handle)
        if not isinstance(value, dict):
            raise CooldownResultError("result root must be a JSON object")
        return value
    return copy.deepcopy(dict(result_or_path))


def _validate_encoded_object_hash(record: Mapping[str, Any], label: str) -> None:
    unsigned = dict(record)
    digest = unsigned.pop("sha256", None)
    if digest != b0.canonical_hash(unsigned):
        raise CooldownResultError(f"{label} object SHA256 changed")


def _validate_immutable_result_sections(
    config: Mapping[str, Any], result: Mapping[str, Any]
) -> None:
    if result.get("model") != {
        "family": "nanogpt",
        "parameters": 30_036_864,
        "block_size": BLOCK_SIZE,
        "vocab_size": 50_304,
    }:
        raise CooldownResultError("result model declaration changed")
    if result.get("optimizer") != {
        "name": "sgd",
        "eta0": ETA0,
        "eta_cooldown": ETA_COOLDOWN,
        "momentum": 0.0,
        "weight_decay": 0.0,
    }:
        raise CooldownResultError("result optimizer declaration changed")
    if result.get("design") != {
        "source_updates": list(SOURCE_UPDATES),
        "future_replicates": FUTURE_REPLICATES,
        "tail_updates": TAIL_UPDATES,
        "evaluation_ticks": list(EVALUATION_TICKS),
        "validation_contexts": VALIDATION_CONTEXTS,
        "pairing": "same_snapshot_same_future_tape_different_learning_rate",
        "snapshot_policy": "ephemeral_cpu_only",
    }:
        raise CooldownResultError("result design declaration changed")
    if result.get("artifacts") != {
        "self_contained_result": True,
        "embedded_array_encoding": EMBEDDED_ARRAY_ENCODING,
        "checkpoint_files_written": [],
    }:
        raise CooldownResultError("result artifact declaration changed")

    data = result.get("data")
    if not isinstance(data, Mapping) or set(data) != {
        "train", "validation", "validation_probe"
    }:
        raise CooldownResultError("result data declaration changed")
    for role in ("train", "validation"):
        spec = config["data"][role]
        if data.get(role) != {
            "path": str(_resolve_data_path(spec)),
            "sha256": spec["sha256"],
            "token_count": int(spec["token_count"]),
            "bytes": int(spec["size_bytes"]),
        }:
            raise CooldownResultError(f"result {role} data identity changed")
    probe = data.get("validation_probe")
    probe_spec = config["data"]["validation_probe"]
    if (
        not isinstance(probe, Mapping)
        or set(probe) != {
            "path", "sha256", "dtype", "context_count", "offsets_sha256",
            "unique_offsets",
        }
        or probe.get("path") != str(_resolve_data_path(probe_spec))
        or probe.get("sha256") != probe_spec["sha256"]
        or probe.get("dtype") != "int64"
        or probe.get("context_count") != VALIDATION_CONTEXTS
        or probe.get("unique_offsets") != VALIDATION_CONTEXTS
        or not _is_hex64(probe.get("offsets_sha256"))
    ):
        raise CooldownResultError("result validation-probe identity changed")


def _validate_runtime(result: Mapping[str, Any], config: Mapping[str, Any]) -> None:
    runtime = result.get("runtime")
    if not isinstance(runtime, Mapping) or set(runtime) != {
        "device", "device_name", "dtype", "elapsed_seconds",
        "actual_gpu_hours", "resource_budget_passed", "maximum_wall_seconds",
        "peak_memory_bytes",
    }:
        raise CooldownResultError("result runtime schema changed")
    elapsed = runtime.get("elapsed_seconds")
    hours = runtime.get("actual_gpu_hours")
    peak = runtime.get("peak_memory_bytes")
    if (
        runtime.get("device") != "cuda"
        or str(config["runtime"]["required_device_name_substring"]).lower()
        not in str(runtime.get("device_name", "")).lower()
        or runtime.get("dtype") != "float32"
        or type(elapsed) not in (int, float) or type(elapsed) is bool
        or type(hours) not in (int, float) or type(hours) is bool
        or not math.isfinite(float(elapsed)) or float(elapsed) < 0.0
        or not math.isfinite(float(hours)) or float(hours) < 0.0
        or not math.isclose(
            float(hours), float(elapsed) / 3600.0,
            rel_tol=0.0, abs_tol=1e-9,
        )
        or runtime.get("maximum_wall_seconds") != MAXIMUM_WALL_SECONDS
        or (peak is not None and (type(peak) is not int or peak < 0))
    ):
        raise CooldownResultError("result runtime accounting changed")
    expected_resource = bool(
        float(elapsed) <= MAXIMUM_WALL_SECONDS
        and float(hours) <= float(config["runtime"]["maximum_a100_gpu_hours"])
    )
    if runtime.get("resource_budget_passed") is not expected_resource:
        raise CooldownResultError("result resource-budget flag changed")


def validate_result(
    config_or_path: str | Path | Mapping[str, Any],
    result_or_path: str | Path | Mapping[str, Any],
    *,
    output_dir: str | Path | None = None,
    require_complete: bool = True,
) -> JsonDict:
    """Validate a self-contained worker result using only config plus result."""

    del output_dir  # Artifacts are deliberately not an input to validation.
    config = _load_config(config_or_path)
    _validate_config(config)
    result = _load_result(result_or_path)
    expected_top = {
        "schema_version",
        "run_id",
        "status",
        "decision",
        "config_sha256",
        "data",
        "model",
        "optimizer",
        "design",
        "measurements",
        "analysis",
        "counts",
        "runtime",
        "artifacts",
        "failure",
        "gpu_submission_performed",
    }
    if set(result) != expected_top:
        raise CooldownResultError("result top-level keys changed")
    if result["schema_version"] != RESULT_SCHEMA or result["run_id"] != RUN_ID:
        raise CooldownResultError("result identity changed")
    if result["config_sha256"] != config["_config_hash"]:
        raise CooldownResultError("result is not bound to this config")
    if result["gpu_submission_performed"] is not False:
        raise CooldownResultError("core falsely reports GPU submission")
    status = result.get("status")
    if status not in ("completed", "incomplete"):
        raise CooldownResultError("result status changed")
    if status == "completed":
        if result.get("decision") != "CALIBRATION_COMPLETED" or result.get("failure") is not None:
            raise CooldownResultError("completed result status is inconsistent")
    else:
        failure = result.get("failure")
        if (
            result.get("decision") != "INCOMPLETE"
            or not isinstance(failure, Mapping)
            or set(failure) != {"type", "message", "traceback_tail"}
            or type(failure.get("type")) is not str
            or type(failure.get("message")) is not str
            or not isinstance(failure.get("traceback_tail"), list)
            or any(type(line) is not str for line in failure["traceback_tail"])
        ):
            raise CooldownResultError("incomplete result failure record changed")
    if require_complete and status != "completed":
        raise CooldownResultError("result is incomplete")
    _validate_immutable_result_sections(config, result)
    _validate_runtime(result, config)

    measurements = _mapping(result["measurements"], "result.measurements")
    if set(measurements) != {
        "training_tape_manifest", "training_trace", "evaluations",
        "snapshot_ledger",
    }:
        raise CooldownResultError("measurement inventory changed")
    tape = _make_training_tape(
        int(config["data"]["train"]["token_count"]),
        int(config["design"]["training_tape_seed"]),
    )
    expected_tape = _tape_manifest(
        tape,
        int(config["data"]["train"]["token_count"]),
        int(config["design"]["training_tape_seed"]),
    )
    if measurements.get("training_tape_manifest") != expected_tape:
        raise CooldownResultError("training tape manifest changed")

    trace = _mapping(measurements.get("training_trace"), "training_trace")
    evaluations = _mapping(measurements.get("evaluations"), "evaluations")
    _validate_encoded_object_hash(trace, "training trace")
    _validate_encoded_object_hash(evaluations, "evaluations")
    counts = _mapping(result.get("counts"), "result.counts")
    if set(counts) != {
        "expected_update_calls", "actual_update_calls",
        "expected_processed_tokens", "actual_processed_tokens",
        "expected_evaluation_rows", "actual_evaluation_rows",
    } or (
        counts.get("expected_update_calls") != ACTUAL_UPDATE_CALLS
        or counts.get("expected_processed_tokens") != PROCESSED_TOKENS
        or counts.get("expected_evaluation_rows") != EXPECTED_EVALUATION_ROWS
    ):
        raise CooldownResultError("result accounting declaration changed")

    if status == "incomplete" and counts.get("actual_update_calls") == 0:
        if (
            counts.get("actual_processed_tokens") != 0
            or counts.get("actual_evaluation_rows") != 0
            or trace != _trace_payload({})
            or evaluations != _evaluation_payload([], [])
            or measurements.get("snapshot_ledger") != []
            or result.get("analysis") != {
                "estimand": (
                    "cooldown_gain_nats=CE_continue-CE_cooldown; "
                    "positive favors cooldown"
                ),
                "cooldown_gain_rows": [],
                "terminal_summary": [],
                "gates": {
                    "complete_grid": False,
                    "tick0_pair_identity_exact": False,
                    "all_values_finite": False,
                },
            }
        ):
            raise CooldownResultError("incomplete result retained inconsistent partial data")
        return {
            "schema_version": "nanogpt30m_e2e_sgd_cooldown_validation_v001",
            "run_id": RUN_ID,
            "status": "incomplete_evidence_preserved",
            "config_sha256": config["_config_hash"],
            "training_tape_sha256": expected_tape["offsets_sha256"],
            "training_trace_sha256": trace["sha256"],
            "evaluations_sha256": evaluations["sha256"],
            "recomputed_analysis": result["analysis"],
        }

    expected_trace_order = [
        "phase", "source_update", "replicate", "arm", "local_update",
        "tape_update_index", "learning_rate", "train_cross_entropy",
    ]
    if (
        trace.get("schema_version")
        != "nanogpt30m_e2e_sgd_cooldown_training_trace_v001"
        or trace.get("column_order") != expected_trace_order
        or set(_mapping(trace.get("columns"), "training_trace.columns"))
        != set(expected_trace_order)
    ):
        raise CooldownResultError("training trace schema changed")
    trace_columns = {
        str(name): _decode_array(record, f"training_trace.{name}")
        for name, record in _mapping(trace.get("columns"), "training_trace.columns").items()
    }
    expected_coordinates = _expected_trace_coordinates()
    for name, expected in expected_coordinates.items():
        actual = trace_columns.get(name)
        if actual is None or not np.array_equal(actual, expected):
            raise CooldownResultError(f"training trace coordinate {name} changed")
    train_ce = trace_columns.get("train_cross_entropy")
    if train_ce is None or train_ce.shape != (ACTUAL_UPDATE_CALLS,):
        raise CooldownResultError("training CE trace has the wrong shape")
    if train_ce.dtype != np.dtype(np.float32) or not bool(np.all(np.isfinite(train_ce))):
        raise CooldownResultError("training CE trace is nonfinite or not float32")
    if int(trace.get("row_count", -1)) != ACTUAL_UPDATE_CALLS:
        raise CooldownResultError("training trace row count changed")

    rows = evaluations.get("rows")
    if not isinstance(rows, list):
        raise CooldownResultError("evaluation rows must be a list")
    if (
        evaluations.get("schema_version")
        != "nanogpt30m_e2e_sgd_cooldown_evaluations_v001"
        or evaluations.get("context_count") != VALIDATION_CONTEXTS
    ):
        raise CooldownResultError("evaluation schema changed")
    context_matrix = _decode_array(
        _mapping(evaluations.get("context_cross_entropies"), "context_cross_entropies"),
        "evaluation context losses",
    )
    if context_matrix.dtype != np.dtype(np.float32):
        raise CooldownResultError("evaluation contexts must be float32")
    if context_matrix.shape != (EXPECTED_EVALUATION_ROWS, VALIDATION_CONTEXTS):
        raise CooldownResultError("evaluation context matrix shape changed")
    if int(evaluations.get("row_count", -1)) != len(rows):
        raise CooldownResultError("evaluation row count changed")
    expected_row_coordinates = [
        (source_update, replicate, ARM_NAMES[arm], tick)
        for source_update in SOURCE_UPDATES
        for replicate in range(FUTURE_REPLICATES)
        for arm in (ARM_CONTINUE, ARM_COOLDOWN)
        for tick in EVALUATION_TICKS
    ]
    actual_row_coordinates = [
        (
            row.get("source_update"), row.get("replicate"),
            row.get("arm"), row.get("tail_update"),
        )
        if isinstance(row, Mapping) else None
        for row in rows
    ]
    if actual_row_coordinates != expected_row_coordinates:
        raise CooldownResultError("evaluation row coordinates changed")
    expected_row_keys = {
        "source_update", "replicate", "arm", "tail_update",
        "effective_optimizer_update", "branch_tokens_seen",
        "branch_intrinsic_time", "learning_rate",
        "validation_cross_entropy", "validation_standard_error",
        "context_cross_entropies_sha256",
    }
    for index, row in enumerate(rows):
        if set(row) != expected_row_keys:
            raise CooldownResultError("evaluation row schema changed")
        source_update, _, arm_name, tick = expected_row_coordinates[index]
        learning_rate = ETA0 if arm_name == "continue" else ETA_COOLDOWN
        if (
            row.get("effective_optimizer_update") != source_update + tick
            or row.get("branch_tokens_seen") != tick * BATCH_SIZE * BLOCK_SIZE
            or row.get("branch_intrinsic_time") != float(tick * learning_rate)
            or row.get("learning_rate") != float(learning_rate)
        ):
            raise CooldownResultError("evaluation row derivation changed")
        values = context_matrix[index].astype(np.float64)
        if row.get("context_cross_entropies_sha256") != _array_sha256(
            context_matrix[index]
        ):
            raise CooldownResultError("per-row validation context hash changed")
        if float(row.get("validation_cross_entropy")) != float(np.mean(values)):
            raise CooldownResultError("validation mean is not reproducible")
        expected_se = float(np.std(values, ddof=1) / math.sqrt(len(values)))
        if float(row.get("validation_standard_error")) != expected_se:
            raise CooldownResultError("validation standard error is not reproducible")
    ledger = measurements.get("snapshot_ledger")
    if not _snapshot_ledger_identity_exact(ledger):
        raise CooldownResultError("snapshot ledger does not prove exact paired forks")
    recomputed_analysis = _analysis_from_evaluations(rows, context_matrix, ledger)
    if result.get("analysis") != recomputed_analysis:
        raise CooldownResultError("analysis is not reproducible from embedded evaluations")

    expected_counts = {
        "expected_update_calls": ACTUAL_UPDATE_CALLS,
        "actual_update_calls": ACTUAL_UPDATE_CALLS,
        "expected_processed_tokens": PROCESSED_TOKENS,
        "actual_processed_tokens": PROCESSED_TOKENS,
        "expected_evaluation_rows": EXPECTED_EVALUATION_ROWS,
        "actual_evaluation_rows": EXPECTED_EVALUATION_ROWS,
    }
    if dict(counts) != expected_counts:
        raise CooldownResultError("full result accounting changed")
    if status == "completed" and not all(recomputed_analysis["gates"].values()):
        raise CooldownResultError("completed result failed a mechanical analysis gate")
    return {
        "schema_version": "nanogpt30m_e2e_sgd_cooldown_validation_v001",
        "run_id": RUN_ID,
        "status": "passed",
        "config_sha256": config["_config_hash"],
        "training_tape_sha256": expected_tape["offsets_sha256"],
        "training_trace_sha256": trace["sha256"],
        "evaluations_sha256": evaluations["sha256"],
        "recomputed_analysis": recomputed_analysis,
    }
