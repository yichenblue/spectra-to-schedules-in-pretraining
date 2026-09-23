"""Analyze the 30M end-to-end plain-SGD theta power-law tail campaign.

The five retained tail trajectories are an ``A2_EXTERNAL`` check.  Their
fixed-probe validation cross-entropies are fitted jointly to

    CE_theta(X) = L_inf + A_theta * X**(-q_theta),

once with arm-specific exponents and once under a common-exponent null.  This
module deliberately makes no theorem-facing claim: end-to-end language-model
training, cross-entropy, and representation learning are outside the paper's
controlled frozen-feature assumptions.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

try:  # Kept optional at import time so artifact validation remains usable.
    from scipy.optimize import least_squares
except ImportError:  # pragma: no cover - the repository environment has SciPy.
    least_squares = None  # type: ignore[assignment]


Json = dict[str, Any]

ANALYSIS_SCHEMA = "nanogpt30m_e2e_sgd_theta_powerlaw_analysis_v001"
CAMPAIGN_ID = "nanogpt30m-e2e-sgd-theta-powerlaw-v001"
RESULT_SCHEMA = "nanogpt30m_e2e_sgd_theta_powerlaw_result_v001"
EXPECTED_THETAS = (0.0, 0.5, 0.75, 1.0, 1.5)
RESULT_FILENAME = "result.json"
TRACE_FILENAME = "training_trace.npz"
EVALUATION_FILENAME = "validation_evaluations.npz"
EXPECTED_BATCH_SIZE = 8
EXPECTED_VALIDATION_CONTEXTS = 1024
EXPECTED_VALIDATION_EVALUATIONS = 410
EXPECTED_X_START = 1.0
EXPECTED_X_STOP = 15.0
LOCAL_SLOPE_POINTS = 63
Q_LOWER = 0.0
Q_UPPER = 8.0
FLOOR_BOUND_FRACTION_TOLERANCE = 1e-5

DESCRIPTIVE = "descriptive exponent estimate in the measured window"
INCONCLUSIVE = "inconclusive"


class AnalysisError(RuntimeError):
    """A retained campaign artifact is malformed or violates the contract."""


@dataclass(frozen=True)
class TailCurve:
    """Validated fixed-probe observations and provenance for one theta arm."""

    theta: float
    directory: Path
    result: Mapping[str, Any]
    target_x: np.ndarray
    x: np.ndarray
    tail_intrinsic_time: np.ndarray
    update: np.ndarray
    tokens_seen: np.ndarray
    loss: np.ndarray
    context_losses: np.ndarray
    source_files: Mapping[str, Mapping[str, Any]]


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_bytes(value: Any) -> bytes:
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


def _load_json(path: Path) -> Json:
    if not path.is_file() or path.is_symlink():
        raise AnalysisError(f"JSON input must be a regular file: {path}")
    try:
        with path.open("r", encoding="utf-8") as handle:
            value = json.load(handle)
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise AnalysisError(f"unable to read {path}: {error}") from error
    if type(value) is not dict:
        raise AnalysisError(f"JSON root must be an object: {path}")
    return value


def _require_hex64(value: Any, label: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise AnalysisError(f"{label} must be a lowercase SHA-256 digest")
    return value


def _number(value: Any, label: str, *, positive: bool = False) -> float:
    if type(value) not in (int, float) or type(value) is bool:
        raise AnalysisError(f"{label} must be numeric")
    parsed = float(value)
    if not math.isfinite(parsed) or (positive and parsed <= 0.0):
        raise AnalysisError(f"{label} must be finite" + (" and positive" if positive else ""))
    return parsed


def _scalar_text(value: np.ndarray, label: str) -> str:
    array = np.asarray(value)
    if array.shape != () or array.dtype.kind not in {"U", "S"}:
        raise AnalysisError(f"{label} must be a scalar string")
    return str(array.item())


def _scalar_number(value: np.ndarray, label: str) -> float:
    array = np.asarray(value)
    if array.shape != () or array.dtype.kind not in {"i", "u", "f"}:
        raise AnalysisError(f"{label} must be a scalar number")
    parsed = float(array.item())
    if not math.isfinite(parsed):
        raise AnalysisError(f"{label} is nonfinite")
    return parsed


def _first_member(
    payload: Mapping[str, np.ndarray],
    names: Sequence[str],
    label: str,
) -> np.ndarray:
    present = [name for name in names if name in payload]
    if len(present) != 1:
        raise AnalysisError(
            f"{label} requires exactly one of {tuple(names)!r}; found {present!r}"
        )
    return np.ascontiguousarray(payload[present[0]])


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    if not path.is_file() or path.is_symlink():
        raise AnalysisError(f"NPZ input must be a regular file: {path}")
    try:
        with np.load(path, allow_pickle=False) as payload:
            return {
                # ``ascontiguousarray`` promotes zero-dimensional metadata to
                # shape (1,); an explicit copy preserves scalar NPZ identity.
                str(name): np.array(payload[name], copy=True)
                for name in payload.files
            }
    except (OSError, ValueError, KeyError) as error:
        raise AnalysisError(f"unable to read safe NPZ artifact {path}: {error}") from error


def _artifact_path(
    root: Path,
    result: Mapping[str, Any],
    logical_name: str,
    filename: str,
) -> tuple[Path, Json]:
    artifacts = result.get("artifacts")
    if type(artifacts) is not dict:
        raise AnalysisError("result.artifacts must be an object")
    record = artifacts.get(logical_name)
    if type(record) is not dict or record.get("path") != filename:
        raise AnalysisError(f"artifact record changed for {logical_name}")
    path = root / filename
    if not path.is_file() or path.is_symlink():
        raise AnalysisError(f"artifact is absent or linked: {path}")
    expected_size = record.get("size_bytes")
    if type(expected_size) is not int or expected_size < 0:
        raise AnalysisError(f"artifact size is invalid for {logical_name}")
    if path.stat().st_size != expected_size:
        raise AnalysisError(f"artifact byte count changed: {path}")
    expected_hash = _require_hex64(record.get("sha256"), f"{logical_name}.sha256")
    observed_hash = _sha256_file(path)
    if observed_hash != expected_hash:
        raise AnalysisError(f"artifact SHA-256 changed: {path}")
    return path, {
        "path": str(path),
        "sha256": observed_hash,
        "size_bytes": int(expected_size),
    }


def _one_dimensional(array: np.ndarray, label: str, rows: int | None = None) -> np.ndarray:
    value = np.asarray(array)
    if value.ndim != 1 or (rows is not None and value.shape != (rows,)):
        expected = "a vector" if rows is None else f"shape ({rows},)"
        raise AnalysisError(f"{label} must have {expected}")
    return np.ascontiguousarray(value)


def _finite_float_vector(
    array: np.ndarray,
    label: str,
    rows: int | None = None,
    *,
    positive: bool = False,
    nonnegative: bool = False,
) -> np.ndarray:
    value = _one_dimensional(array, label, rows).astype(np.float64, copy=False)
    if not np.all(np.isfinite(value)):
        raise AnalysisError(f"{label} contains a nonfinite value")
    if positive and np.any(value <= 0.0):
        raise AnalysisError(f"{label} must be positive")
    if nonnegative and np.any(value < 0.0):
        raise AnalysisError(f"{label} must be nonnegative")
    return np.ascontiguousarray(value)


def _integer_vector(
    array: np.ndarray,
    label: str,
    rows: int | None = None,
    *,
    nonnegative: bool = True,
) -> np.ndarray:
    value = _one_dimensional(array, label, rows)
    if value.dtype.kind not in {"i", "u"}:
        raise AnalysisError(f"{label} must have an integer dtype")
    value = value.astype(np.int64, copy=False)
    if nonnegative and np.any(value < 0):
        raise AnalysisError(f"{label} must be nonnegative")
    return np.ascontiguousarray(value)


def _strictly_increasing(values: np.ndarray, label: str) -> None:
    if values.size < 2 or not np.all(np.diff(values) > 0):
        raise AnalysisError(f"{label} must be strictly increasing")


def _validate_optimizer(result: Mapping[str, Any]) -> None:
    optimizer = result.get("optimizer")
    if type(optimizer) is not dict:
        raise AnalysisError("result.optimizer must be an object")
    exact = {
        "name": "sgd",
        "batch_size": EXPECTED_BATCH_SIZE,
        "momentum": 0.0,
        "dampening": 0.0,
        "nesterov": False,
        "weight_decay": 0.0,
        "gradient_clip": 0.0,
        "foreach": False,
        "state_entries": 0,
    }
    for key, expected in exact.items():
        observed = optimizer.get(key)
        if type(observed) is not type(expected) or observed != expected:
            raise AnalysisError(
                f"plain-SGD metadata changed: optimizer.{key}={observed!r}"
            )
    _number(optimizer.get("learning_rate_base"), "optimizer.learning_rate_base", positive=True)


def _validate_scope(result: Mapping[str, Any]) -> None:
    scope = result.get("scope")
    if type(scope) is not dict:
        raise AnalysisError("result.scope must be an object")
    if (
        scope.get("prediction_id") != "A2_EXTERNAL"
        or scope.get("evidence_class") != "A2_EXTERNAL"
        or scope.get("theorem_facing") is not False
    ):
        raise AnalysisError("result scope must remain non-theorem-facing A2_EXTERNAL")


def _validate_npz_identity(
    payload: Mapping[str, np.ndarray], theta: float, phase: str, label: str
) -> None:
    if "schema_version" not in payload:
        raise AnalysisError(f"{label} is missing schema_version")
    schema = _scalar_text(payload["schema_version"], f"{label}.schema_version")
    if not schema.startswith("nanogpt30m_e2e_sgd_theta_powerlaw"):
        raise AnalysisError(f"{label} schema identity changed: {schema!r}")
    if "phase" not in payload or _scalar_text(payload["phase"], f"{label}.phase") != phase:
        raise AnalysisError(f"{label} phase must be {phase!r}")
    if "theta" not in payload or _scalar_number(payload["theta"], f"{label}.theta") != theta:
        raise AnalysisError(f"{label} theta does not match result.json")


def _validate_trace(payload: Mapping[str, np.ndarray], theta: float) -> Json:
    _validate_npz_identity(payload, theta, "tail", "training trace")
    update = _integer_vector(
        _first_member(payload, ("update", "tail_update"), "training update"),
        "training update",
    )
    rows = int(update.size)
    if rows < 1:
        raise AnalysisError("training trace is empty")
    _strictly_increasing(update, "training update")
    if "global_update" in payload:
        global_update = _integer_vector(payload["global_update"], "global update", rows)
        _strictly_increasing(global_update, "global update")
    learning_rate = _finite_float_vector(
        _first_member(payload, ("learning_rate",), "learning rate"),
        "learning rate",
        rows,
        positive=True,
    )
    if "nominal_learning_rate" in payload:
        nominal = _finite_float_vector(
            payload["nominal_learning_rate"],
            "nominal learning rate",
            rows,
            positive=True,
        )
        schedule_roundoff = 1e-8 * float(np.max(nominal))
        if np.any(learning_rate - nominal > schedule_roundoff):
            raise AnalysisError("effective learning rate exceeds nominal learning rate")
    before_t = _finite_float_vector(
        _first_member(
            payload,
            ("tail_intrinsic_time_before", "tau_before"),
            "tail intrinsic time before",
        ),
        "tail intrinsic time before",
        rows,
        nonnegative=True,
    )
    after_t = _finite_float_vector(
        _first_member(
            payload,
            ("tail_intrinsic_time_after", "tau", "intrinsic_time_after"),
            "tail intrinsic time after",
        ),
        "tail intrinsic time after",
        rows,
        positive=True,
    )
    if np.any(after_t <= before_t):
        raise AnalysisError("tail intrinsic-time steps must have positive width")
    _strictly_increasing(after_t, "tail intrinsic time after")
    if rows > 1 and not np.allclose(before_t[1:], after_t[:-1], rtol=0.0, atol=2e-12):
        raise AnalysisError("tail intrinsic-time endpoints are not contiguous")
    before_x = _finite_float_vector(
        _first_member(
            payload,
            ("schedule_coordinate_x_before", "X_before", "x_before"),
            "schedule coordinate before",
        ),
        "schedule coordinate before",
        rows,
        positive=True,
    )
    after_x = _finite_float_vector(
        _first_member(
            payload,
            ("schedule_coordinate_x_after", "X", "x", "schedule_coordinate_x"),
            "schedule coordinate after",
        ),
        "schedule coordinate after",
        rows,
        positive=True,
    )
    if np.any(after_x <= before_x):
        raise AnalysisError("schedule-coordinate steps must have positive width")
    _strictly_increasing(after_x, "schedule coordinate after")
    if rows > 1 and not np.allclose(before_x[1:], after_x[:-1], rtol=0.0, atol=2e-12):
        raise AnalysisError("schedule-coordinate endpoints are not contiguous")
    tokens = _integer_vector(
        _first_member(payload, ("tokens_seen", "tokens"), "training tokens"),
        "training tokens",
        rows,
    )
    _strictly_increasing(tokens, "training tokens")
    if "tail_tokens" in payload:
        tail_tokens = _integer_vector(payload["tail_tokens"], "tail tokens", rows)
        _strictly_increasing(tail_tokens, "tail tokens")
    train_ce = _finite_float_vector(
        _first_member(
            payload,
            ("training_cross_entropy", "train_cross_entropy"),
            "training cross-entropy",
        ),
        "training cross-entropy",
        rows,
        positive=True,
    )
    if "terminal_eta_truncated" in payload:
        truncated = _one_dimensional(payload["terminal_eta_truncated"], "terminal_eta_truncated", rows)
        if truncated.dtype.kind != "b":
            raise AnalysisError("terminal_eta_truncated must have boolean dtype")
        if np.count_nonzero(truncated) > 1 or (
            np.any(truncated) and not bool(truncated[-1])
        ):
            raise AnalysisError("only the terminal update may truncate eta")
    if not math.isclose(float(before_x[0]), EXPECTED_X_START, rel_tol=0.0, abs_tol=1e-12):
        raise AnalysisError("training schedule coordinate does not start at X=1")
    if not math.isclose(float(after_x[-1]), EXPECTED_X_STOP, rel_tol=0.0, abs_tol=1e-12):
        raise AnalysisError("training schedule coordinate does not terminate at X=15")
    return {
        "rows": rows,
        "first_update": int(update[0]),
        "last_update": int(update[-1]),
        "terminal_tail_intrinsic_time": float(after_t[-1]),
        "terminal_schedule_coordinate_x": float(after_x[-1]),
        "terminal_tokens_seen": int(tokens[-1]),
        "training_loss_min": float(np.min(train_ce)),
        "training_loss_max": float(np.max(train_ce)),
    }


def _validate_evaluations(
    payload: Mapping[str, np.ndarray], theta: float
) -> tuple[dict[str, np.ndarray], Json]:
    _validate_npz_identity(payload, theta, "tail", "validation evaluations")
    target_x = _finite_float_vector(
        _first_member(payload, ("target_x", "target_X"), "target X"),
        "target X",
        positive=True,
    )
    rows = int(target_x.size)
    if rows != EXPECTED_VALIDATION_EVALUATIONS:
        raise AnalysisError(
            "validation grid must contain exactly "
            f"{EXPECTED_VALIDATION_EVALUATIONS} evaluations"
        )
    if rows < LOCAL_SLOPE_POINTS:
        raise AnalysisError(
            f"validation grid needs at least {LOCAL_SLOPE_POINTS} rows"
        )
    x = _finite_float_vector(
        _first_member(
            payload,
            ("actual_x", "x", "X", "schedule_coordinate_x"),
            "actual X",
        ),
        "actual X",
        rows,
        positive=True,
    )
    update = _integer_vector(
        _first_member(
            payload,
            ("update", "tail_update", "evaluation_update"),
            "validation update",
        ),
        "validation update",
        rows,
    )
    tail_time = _finite_float_vector(
        _first_member(
            payload,
            ("tail_intrinsic_time", "tau", "intrinsic_time"),
            "validation tail intrinsic time",
        ),
        "validation tail intrinsic time",
        rows,
        nonnegative=True,
    )
    tokens = _integer_vector(
        _first_member(payload, ("tokens_seen", "tokens"), "validation tokens"),
        "validation tokens",
        rows,
    )
    loss = _finite_float_vector(
        _first_member(
            payload,
            ("validation_cross_entropy", "mean_cross_entropy"),
            "validation cross-entropy",
        ),
        "validation cross-entropy",
        rows,
        positive=True,
    )
    contexts = _first_member(
        payload,
        ("context_cross_entropies", "context_cross_entropy"),
        "context cross-entropies",
    ).astype(np.float64, copy=False)
    if contexts.shape != (rows, EXPECTED_VALIDATION_CONTEXTS):
        raise AnalysisError(
            "context cross-entropies must have shape "
            f"({rows}, {EXPECTED_VALIDATION_CONTEXTS})"
        )
    if not np.all(np.isfinite(contexts)) or np.any(contexts <= 0.0):
        raise AnalysisError("context cross-entropies must be finite and positive")
    means = np.mean(contexts, axis=1, dtype=np.float64)
    if not np.allclose(loss, means, rtol=0.0, atol=2e-7):
        raise AnalysisError("validation means do not match retained context losses")
    _strictly_increasing(target_x, "target X")
    _strictly_increasing(x, "actual X")
    _strictly_increasing(update, "validation update")
    _strictly_increasing(tail_time, "validation tail intrinsic time")
    _strictly_increasing(tokens, "validation tokens")
    for name, values in (("target X", target_x), ("actual X", x)):
        if not math.isclose(float(values[0]), EXPECTED_X_START, rel_tol=0.0, abs_tol=1e-12):
            raise AnalysisError(f"{name} does not start at X=1")
        if not math.isclose(float(values[-1]), EXPECTED_X_STOP, rel_tol=0.0, abs_tol=1e-12):
            raise AnalysisError(f"{name} does not end at X=15")
        decades = math.log10(float(values[-1]) / float(values[0]))
        if decades < 1.0 - 1e-12:
            raise AnalysisError(f"{name} spans less than one decade")
    if np.any(x + 2e-12 < target_x):
        raise AnalysisError("actual X precedes its requested target X")
    arrays = {
        "target_x": np.ascontiguousarray(target_x),
        "x": np.ascontiguousarray(x),
        "tail_intrinsic_time": np.ascontiguousarray(tail_time),
        "update": np.ascontiguousarray(update),
        "tokens_seen": np.ascontiguousarray(tokens),
        "loss": np.ascontiguousarray(loss),
        "context_losses": np.ascontiguousarray(contexts),
    }
    report = {
        "rows": rows,
        "contexts_per_evaluation": int(contexts.shape[1]),
        "x_start": float(x[0]),
        "x_stop": float(x[-1]),
        "x_span_decades": float(math.log10(x[-1] / x[0])),
        "terminal_tail_intrinsic_time": float(tail_time[-1]),
        "first_validation_cross_entropy": float(loss[0]),
        "last_validation_cross_entropy": float(loss[-1]),
        "validation_cross_entropy_drop": float(loss[0] - loss[-1]),
    }
    return arrays, report


def validate_tail_directory(path: str | Path) -> tuple[TailCurve, Json]:
    """Load and strictly validate one completed theta tail directory."""

    root = Path(path).expanduser().resolve()
    if not root.is_dir() or root.is_symlink():
        raise AnalysisError(f"tail directory must be a real directory: {root}")
    result_path = root / RESULT_FILENAME
    result = _load_json(result_path)
    if (
        result.get("schema_version") != RESULT_SCHEMA
        or result.get("campaign_id") != CAMPAIGN_ID
        or result.get("run_id") != CAMPAIGN_ID
    ):
        raise AnalysisError(f"campaign identity changed in {result_path}")
    if result.get("status") != "completed" or result.get("phase") != "tail":
        raise AnalysisError(f"tail result is not completed: {result_path}")
    _validate_scope(result)
    _validate_optimizer(result)
    theta = _number(result.get("theta"), "result.theta")
    if theta not in EXPECTED_THETAS:
        raise AnalysisError(f"unexpected theta {theta!r} in {result_path}")
    identities = {
        name: _require_hex64(result.get(name), f"result.{name}")
        for name in (
            "prefix_state_sha256",
            "validation_probe_sha256",
            "future_tape_sha256",
            "schedule_manifest_sha256",
            "data_identity_sha256",
            "prefix_checkpoint_sha256",
            "validation_anchor_sha256",
        )
    }
    trace_path, trace_source = _artifact_path(
        root, result, "training_trace", TRACE_FILENAME
    )
    evaluation_path, evaluation_source = _artifact_path(
        root, result, "validation_evaluations", EVALUATION_FILENAME
    )
    trace = _load_npz(trace_path)
    evaluations = _load_npz(evaluation_path)
    trace_report = _validate_trace(trace, theta)
    arrays, evaluation_report = _validate_evaluations(evaluations, theta)
    terminal_t = _number(
        result.get("terminal_tail_intrinsic_time"),
        "result.terminal_tail_intrinsic_time",
        positive=True,
    )
    terminal_x = _number(
        result.get("terminal_schedule_coordinate_x"),
        "result.terminal_schedule_coordinate_x",
        positive=True,
    )
    if not (
        math.isclose(
            terminal_t,
            trace_report["terminal_tail_intrinsic_time"],
            rel_tol=0.0,
            abs_tol=2e-12,
        )
        and math.isclose(
            terminal_t,
            evaluation_report["terminal_tail_intrinsic_time"],
            rel_tol=0.0,
            abs_tol=2e-12,
        )
    ):
        raise AnalysisError("terminal tail intrinsic time disagrees across artifacts")
    if not (
        math.isclose(terminal_x, EXPECTED_X_STOP, rel_tol=0.0, abs_tol=1e-12)
        and math.isclose(
            terminal_x,
            trace_report["terminal_schedule_coordinate_x"],
            rel_tol=0.0,
            abs_tol=1e-12,
        )
    ):
        raise AnalysisError("terminal schedule coordinate disagrees across artifacts")
    counts = result.get("counts")
    if type(counts) is not dict:
        raise AnalysisError("result.counts must be an object")
    expected_count_pairs = {
        "tail_updates": trace_report["rows"],
        "validation_evaluations": evaluation_report["rows"],
    }
    for key, expected in expected_count_pairs.items():
        if counts.get(key) != expected or type(counts.get(key)) is not int:
            raise AnalysisError(f"result.counts.{key} does not match its artifact")
    report = {
        "theta": theta,
        "directory": str(root),
        "result": {
            "path": str(result_path),
            "sha256": _sha256_file(result_path),
            "size_bytes": result_path.stat().st_size,
        },
        "identities": identities,
        "optimizer": dict(result["optimizer"]),
        "trace": trace_report,
        "evaluations": evaluation_report,
        "source_files": {
            "training_trace": trace_source,
            "validation_evaluations": evaluation_source,
        },
        "gates": {
            "completed_tail": True,
            "A2_EXTERNAL_only": True,
            "plain_SGD_batch_8": True,
            "artifact_hashes_verified": True,
            "coordinates_finite_and_increasing": True,
            "fixed_probe_means_reproduced": True,
            "X_spans_1_to_15": True,
        },
    }
    curve = TailCurve(
        theta=theta,
        directory=root,
        result=result,
        target_x=arrays["target_x"],
        x=arrays["x"],
        tail_intrinsic_time=arrays["tail_intrinsic_time"],
        update=arrays["update"],
        tokens_seen=arrays["tokens_seen"],
        loss=arrays["loss"],
        context_losses=arrays["context_losses"],
        source_files=report["source_files"],
    )
    return curve, report


def _validate_campaign(curves: Sequence[TailCurve], reports: Sequence[Json]) -> Json:
    if len(curves) != len(EXPECTED_THETAS):
        raise AnalysisError("campaign analysis requires exactly five tail directories")
    observed = tuple(sorted(curve.theta for curve in curves))
    if observed != EXPECTED_THETAS or len(set(observed)) != len(observed):
        raise AnalysisError(
            f"theta arms must be exact and distinct: {EXPECTED_THETAS!r}; got {observed!r}"
        )
    identity_names = (
        "prefix_state_sha256",
        "validation_probe_sha256",
        "future_tape_sha256",
        "schedule_manifest_sha256",
        "data_identity_sha256",
        "prefix_checkpoint_sha256",
        "validation_anchor_sha256",
    )
    shared = {}
    for name in identity_names:
        values = {
            str(report["identities"][name])
            for report in reports
        }
        if len(values) != 1:
            raise AnalysisError(f"campaign identity is not shared: {name}")
        shared[name] = next(iter(values))
    terminal_times = np.asarray(
        [curve.tail_intrinsic_time[-1] for curve in curves], dtype=np.float64
    )
    if not np.allclose(terminal_times, terminal_times[0], rtol=0.0, atol=2e-12):
        raise AnalysisError("tail arms do not end at equal intrinsic time")
    target_reference = curves[0].target_x
    for curve in curves[1:]:
        if not np.array_equal(curve.target_x, target_reference):
            raise AnalysisError("tail arms do not share the exact target-X grid")
    return {
        "status": "passed",
        "gates": {
            "five_exact_distinct_theta_arms": True,
            "common_prefix_state": True,
            "common_fixed_validation_probe": True,
            "common_future_tape": True,
            "common_schedule_manifest": True,
            "common_data_identity": True,
            "common_prefix_checkpoint": True,
            "common_validation_anchor": True,
            "equal_terminal_tail_intrinsic_time": True,
            "common_target_X_grid": True,
            "one_decade_X_window": True,
            "plain_SGD_batch_8": True,
        },
        "shared_identities": shared,
        "terminal_tail_intrinsic_time": float(terminal_times[0]),
        "target_x_rows": int(target_reference.size),
        "target_x_start": float(target_reference[0]),
        "target_x_stop": float(target_reference[-1]),
        "target_x_span_decades": float(
            math.log10(target_reference[-1] / target_reference[0])
        ),
    }


def _require_scipy() -> None:
    if least_squares is None:
        raise AnalysisError(
            "scipy.optimize.least_squares is required for the nonlinear fits"
        )


def _curve_arrays(
    curves: Mapping[float, tuple[np.ndarray, np.ndarray]]
) -> tuple[tuple[float, ...], list[np.ndarray], list[np.ndarray]]:
    thetas = tuple(sorted(float(theta) for theta in curves))
    if len(thetas) < 1 or len(set(thetas)) != len(thetas):
        raise AnalysisError("fit curves require distinct theta keys")
    xs: list[np.ndarray] = []
    losses: list[np.ndarray] = []
    for theta in thetas:
        raw_x, raw_loss = curves[theta]
        x = np.asarray(raw_x, dtype=np.float64).reshape(-1)
        loss = np.asarray(raw_loss, dtype=np.float64).reshape(-1)
        if x.size < 3 or x.shape != loss.shape:
            raise AnalysisError(f"theta={theta:g} fit arrays have incompatible shapes")
        if (
            np.any(~np.isfinite(x))
            or np.any(x <= 0.0)
            or np.any(~np.isfinite(loss))
            or np.any(loss <= 0.0)
            or np.any(np.diff(x) <= 0.0)
        ):
            raise AnalysisError(f"theta={theta:g} fit curve is nonfinite or unordered")
        xs.append(np.ascontiguousarray(x))
        losses.append(np.ascontiguousarray(loss))
    return thetas, xs, losses


def _fit_bounds(losses: Sequence[np.ndarray], arm_count: int, common_q: bool) -> tuple[np.ndarray, np.ndarray, float]:
    minimum = min(float(np.min(loss)) for loss in losses)
    maximum = max(float(np.max(loss)) for loss in losses)
    margin = max(1e-12, 1e-12 * maximum)
    floor_upper = minimum - margin
    if floor_upper <= 0.0:
        raise AnalysisError("positive CE data leave no nonnegative floor interval")
    amplitude_lower = max(1e-14, (maximum - minimum) * 1e-10)
    amplitude_upper = max(100.0, 100.0 * maximum)
    if common_q:
        lower = np.asarray(
            [0.0, *([math.log(amplitude_lower)] * arm_count), Q_LOWER],
            dtype=np.float64,
        )
        upper = np.asarray(
            [floor_upper, *([math.log(amplitude_upper)] * arm_count), Q_UPPER],
            dtype=np.float64,
        )
    else:
        lower_values = [0.0]
        upper_values = [floor_upper]
        for _ in range(arm_count):
            lower_values.extend((math.log(amplitude_lower), Q_LOWER))
            upper_values.extend((math.log(amplitude_upper), Q_UPPER))
        lower = np.asarray(lower_values, dtype=np.float64)
        upper = np.asarray(upper_values, dtype=np.float64)
    return lower, upper, floor_upper


def _amplitude_start(x: np.ndarray, loss: np.ndarray, floor: float, q: float) -> float:
    basis = np.power(x, -q)
    centered = loss - floor
    estimate = float(np.dot(basis, centered) / np.dot(basis, basis))
    return max(estimate, 1e-12)


def _decode_fit_parameters(
    vector: np.ndarray, thetas: Sequence[float], common_q: bool
) -> tuple[float, dict[float, Json]]:
    floor = float(vector[0])
    arms: dict[float, Json] = {}
    if common_q:
        q = float(vector[-1])
        for index, theta in enumerate(thetas):
            arms[float(theta)] = {
                "amplitude": float(math.exp(float(vector[1 + index]))),
                "q": q,
            }
    else:
        for index, theta in enumerate(thetas):
            arms[float(theta)] = {
                "amplitude": float(math.exp(float(vector[1 + 2 * index]))),
                "q": float(vector[2 + 2 * index]),
            }
    return floor, arms


def _predictions(
    floor: float,
    arms: Mapping[float, Mapping[str, float]],
    thetas: Sequence[float],
    xs: Sequence[np.ndarray],
) -> list[np.ndarray]:
    return [
        floor
        + float(arms[float(theta)]["amplitude"])
        * np.power(x, -float(arms[float(theta)]["q"]))
        for theta, x in zip(thetas, xs, strict=True)
    ]


def _bic(rss: float, observations: int, parameters: int) -> float:
    if observations <= parameters or parameters <= 0:
        raise AnalysisError("BIC requires more observations than fitted parameters")
    mean_square = max(
        rss / observations,
        np.finfo(np.float64).tiny,
    )
    return float(observations * math.log(mean_square) + parameters * math.log(observations))


def _fit_joint_model(
    curves: Mapping[float, tuple[np.ndarray, np.ndarray]], *, common_q: bool
) -> Json:
    _require_scipy()
    thetas, xs, losses = _curve_arrays(curves)
    arm_count = len(thetas)
    lower, upper, floor_upper = _fit_bounds(losses, arm_count, common_q)
    scale = max(
        max(float(np.ptp(loss)) for loss in losses),
        1e-7 * max(float(np.max(loss)) for loss in losses),
    )

    def residual(vector: np.ndarray) -> np.ndarray:
        floor, arms = _decode_fit_parameters(vector, thetas, common_q)
        predicted = _predictions(floor, arms, thetas, xs)
        return np.concatenate(
            [(prediction - loss) / scale for prediction, loss in zip(predicted, losses, strict=True)]
        )

    starts: list[np.ndarray] = []
    q_starts = (0.05, 0.15, 0.35, 0.65, 1.0, 1.75, 3.0)
    floor_fractions = (0.15, 0.45, 0.72, 0.90, 0.975)
    for start_index in range(max(len(q_starts), len(floor_fractions))):
        q_start = q_starts[start_index % len(q_starts)]
        floor = floor_upper * floor_fractions[start_index % len(floor_fractions)]
        values = [floor]
        if common_q:
            for x, loss in zip(xs, losses, strict=True):
                values.append(math.log(_amplitude_start(x, loss, floor, q_start)))
            values.append(q_start)
        else:
            for arm_index, (x, loss) in enumerate(zip(xs, losses, strict=True)):
                q_arm = q_starts[(start_index + arm_index) % len(q_starts)]
                values.extend(
                    (math.log(_amplitude_start(x, loss, floor, q_arm)), q_arm)
                )
        starts.append(np.clip(np.asarray(values), lower + 1e-11, upper - 1e-11))

    solutions = []
    assert least_squares is not None
    for start in starts:
        fit = least_squares(
            residual,
            start,
            bounds=(lower, upper),
            method="trf",
            x_scale="jac",
            max_nfev=5000,
            ftol=1e-12,
            xtol=1e-12,
            gtol=1e-12,
        )
        floor, arms = _decode_fit_parameters(fit.x, thetas, common_q)
        predicted = _predictions(floor, arms, thetas, xs)
        raw_residual = np.concatenate(
            [prediction - loss for prediction, loss in zip(predicted, losses, strict=True)]
        )
        rss = float(np.dot(raw_residual, raw_residual))
        solutions.append((rss, fit, floor, arms, predicted))
    rss, fit, floor, arms, predicted = min(solutions, key=lambda item: item[0])
    observations = int(sum(loss.size for loss in losses))
    parameter_count = 2 + arm_count if common_q else 1 + 2 * arm_count
    floor_range = float(floor_upper)
    floor_tolerance = max(1e-10, FLOOR_BOUND_FRACTION_TOLERANCE * floor_range)
    floor_at_lower = bool(floor <= lower[0] + floor_tolerance)
    floor_at_upper = bool(floor >= upper[0] - floor_tolerance)
    singular_values = np.linalg.svd(np.asarray(fit.jac), compute_uv=False)
    positive_singular = singular_values[singular_values > 0.0]
    rank = int(np.linalg.matrix_rank(np.asarray(fit.jac)))
    condition = (
        float(positive_singular[0] / positive_singular[-1])
        if positive_singular.size
        else math.inf
    )
    near_tolerance = max(
        rss * 1e-4,
        np.finfo(np.float64).eps
        * 100.0
        * sum(float(np.dot(loss, loss)) for loss in losses),
    )
    near = [item for item in solutions if item[0] <= rss + near_tolerance]
    near_floors = [float(item[2]) for item in near]
    near_qs = [
        tuple(float(item[3][theta]["q"]) for theta in thetas)
        for item in near
    ]
    maximum_near_q_spread = max(
        (
            max(values[index] for values in near_qs)
            - min(values[index] for values in near_qs)
            for index in range(arm_count)
        ),
        default=0.0,
    )
    multistart_unstable = bool(
        len(near) > 1
        and (
            max(near_floors) - min(near_floors) > max(1e-8, 0.01 * floor_range)
            or maximum_near_q_spread > 0.10
        )
    )
    optimizer_stable = bool(
        fit.success
        and rank == parameter_count
        and math.isfinite(condition)
        and condition <= 1e12
        and not multistart_unstable
    )
    arm_records = {
        f"{theta:g}": {
            "theta": float(theta),
            "amplitude": float(arms[theta]["amplitude"]),
            "q": float(arms[theta]["q"]),
            "rss": float(
                np.sum(np.square(prediction - loss), dtype=np.float64)
            ),
            "rmse": float(np.sqrt(np.mean(np.square(prediction - loss)))),
        }
        for theta, loss, prediction in zip(thetas, losses, predicted, strict=True)
    }
    return {
        "model": "common_q_null" if common_q else "free_arm_exponents",
        "objective": "ordinary_unweighted_least_squares_on_validation_cross_entropy",
        "shared_L_inf": floor,
        "arms": arm_records,
        "rss": rss,
        "rmse": float(math.sqrt(rss / observations)),
        "bic": _bic(rss, observations, parameter_count),
        "observations": observations,
        "parameter_count": parameter_count,
        "optimizer": {
            "library": "scipy.optimize.least_squares",
            "success": bool(fit.success),
            "status": int(fit.status),
            "message": str(fit.message),
            "function_evaluations": int(fit.nfev),
            "multistarts": len(starts),
        },
        "identifiability": {
            "floor_lower_bound": float(lower[0]),
            "floor_upper_bound": float(upper[0]),
            "floor_at_lower_bound": floor_at_lower,
            "floor_at_upper_bound": floor_at_upper,
            "jacobian_rank": rank,
            "jacobian_columns": parameter_count,
            "jacobian_condition_number": None if not math.isfinite(condition) else condition,
            "near_optimal_multistarts": len(near),
            "near_optimal_floor_min": min(near_floors),
            "near_optimal_floor_max": max(near_floors),
            "near_optimal_maximum_q_spread": maximum_near_q_spread,
            "multistart_unstable": multistart_unstable,
            "optimizer_profile_stable": optimizer_stable,
        },
        "_prediction_arrays": predicted,
    }


def _fit_one_fixed_floor(x: np.ndarray, loss: np.ndarray, floor: float) -> Json:
    _require_scipy()
    if not math.isfinite(floor) or floor < 0.0 or floor >= float(np.min(loss)):
        raise AnalysisError("fixed floor must be finite, nonnegative, and below every loss")
    maximum_amplitude = max(100.0, 100.0 * float(np.max(loss)))
    lower = np.asarray([math.log(1e-14), Q_LOWER], dtype=np.float64)
    upper = np.asarray([math.log(maximum_amplitude), Q_UPPER], dtype=np.float64)
    scale = max(float(np.ptp(loss)), 1e-7 * float(np.max(loss)))

    def residual(vector: np.ndarray) -> np.ndarray:
        prediction = floor + math.exp(float(vector[0])) * np.power(x, -float(vector[1]))
        return (prediction - loss) / scale

    log_x = np.log(x)
    log_centered = np.log(loss - floor)
    q_log_start = max(Q_LOWER + 1e-6, min(Q_UPPER - 1e-6, -float(np.polyfit(log_x, log_centered, 1)[0])))
    candidates = (q_log_start, 0.05, 0.2, 0.5, 1.0, 2.0, 4.0)
    fits = []
    assert least_squares is not None
    for q_start in candidates:
        start = np.asarray(
            [math.log(_amplitude_start(x, loss, floor, q_start)), q_start],
            dtype=np.float64,
        )
        fit = least_squares(
            residual,
            np.clip(start, lower + 1e-11, upper - 1e-11),
            bounds=(lower, upper),
            method="trf",
            x_scale="jac",
            max_nfev=2000,
            ftol=1e-12,
            xtol=1e-12,
            gtol=1e-12,
        )
        amplitude = math.exp(float(fit.x[0]))
        q = float(fit.x[1])
        prediction = floor + amplitude * np.power(x, -q)
        rss = float(np.sum(np.square(prediction - loss), dtype=np.float64))
        fits.append((rss, fit, amplitude, q))
    rss, fit, amplitude, q = min(fits, key=lambda item: item[0])
    return {
        "amplitude": float(amplitude),
        "q": q,
        "rss": rss,
        "rmse": float(math.sqrt(rss / loss.size)),
        "optimizer_success": bool(fit.success),
        "q_at_bound": bool(q <= Q_LOWER + 1e-5 or q >= Q_UPPER - 1e-5),
    }


def _fixed_floor_grid(losses: Sequence[np.ndarray], shared_floor: float) -> list[tuple[float, str]]:
    minimum = min(float(np.min(loss)) for loss in losses)
    maximum = max(float(np.max(loss)) for loss in losses)
    empirical_scale = max(
        max(float(np.ptp(loss)) for loss in losses),
        max(abs(float(loss[0] - loss[-1])) for loss in losses),
        1e-4 * maximum,
    )
    multipliers = (0.02, 0.05, 0.10, 0.20, 0.40, 0.75, 1.25, 2.0, 4.0, 8.0, 16.0, 32.0)
    candidates: list[tuple[float, str]] = [
        (max(0.0, minimum - multiplier * empirical_scale), f"min_loss_minus_{multiplier:g}_scale")
        for multiplier in multipliers
    ]
    candidates.extend(((0.0, "nonnegative_lower_bound"), (shared_floor, "shared_fit")))
    margin = max(1e-12, 1e-12 * maximum)
    normalized: list[tuple[float, str]] = []
    for floor, source in sorted(candidates):
        floor = min(float(floor), minimum - margin)
        if floor < 0.0:
            continue
        if normalized and math.isclose(floor, normalized[-1][0], rel_tol=0.0, abs_tol=1e-12):
            if source == "shared_fit":
                normalized[-1] = (normalized[-1][0], normalized[-1][1] + "+shared_fit")
            continue
        normalized.append((floor, source))
    return normalized


def _fixed_floor_sensitivity(
    curves: Mapping[float, tuple[np.ndarray, np.ndarray]], shared_floor: float
) -> tuple[Json, list[Json]]:
    thetas, xs, losses = _curve_arrays(curves)
    grid = _fixed_floor_grid(losses, shared_floor)
    rows: list[Json] = []
    totals = []
    for floor_index, (floor, source) in enumerate(grid):
        total_rss = 0.0
        arm_fits = []
        for theta, x, loss in zip(thetas, xs, losses, strict=True):
            fit = _fit_one_fixed_floor(x, loss, floor)
            total_rss += float(fit["rss"])
            arm_fits.append((theta, fit))
        totals.append(total_rss)
        for theta, fit in arm_fits:
            rows.append(
                {
                    "floor_index": floor_index,
                    "floor_source": source,
                    "fixed_L_inf": floor,
                    "theta": float(theta),
                    "amplitude": fit["amplitude"],
                    "q": fit["q"],
                    "arm_rss": fit["rss"],
                    "arm_rmse": fit["rmse"],
                    "total_rss": total_rss,
                    "optimizer_success": fit["optimizer_success"],
                    "q_at_bound": fit["q_at_bound"],
                }
            )
    total_array = np.asarray(totals, dtype=np.float64)
    best = float(np.min(total_array))
    scale = sum(float(np.dot(loss, loss)) for loss in losses)
    near_tolerance = max(0.01 * best, 100.0 * np.finfo(np.float64).eps * scale)
    near_indices = np.flatnonzero(total_array <= best + near_tolerance)
    floor_values = np.asarray([value for value, _ in grid], dtype=np.float64)
    search_span = float(np.ptp(floor_values))
    near_span = (
        float(np.ptp(floor_values[near_indices])) if near_indices.size > 1 else 0.0
    )
    relative_profile_range = float(
        np.ptp(total_array) / max(float(np.median(total_array)), np.finfo(np.float64).tiny)
    )
    flat = bool(
        relative_profile_range <= 0.01
        or (
            near_indices.size >= 3
            and search_span > 0.0
            and near_span / search_span >= 0.25
        )
    )
    stable = bool(
        np.all(np.isfinite(total_array))
        and not flat
        and all(bool(row["optimizer_success"]) for row in rows)
    )
    summary = {
        "floor_grid_rule": (
            "nonnegative floors below the global minimum CE at declared multiples "
            "of the maximum observed arm range/drop, plus zero and the shared-fit floor"
        ),
        "grid_rows": len(grid),
        "floor_min": float(floor_values[0]),
        "floor_max": float(floor_values[-1]),
        "best_grid_floor": float(floor_values[int(np.argmin(total_array))]),
        "best_grid_rss": best,
        "near_optimal_grid_points": int(near_indices.size),
        "near_optimal_floor_span": near_span,
        "search_floor_span": search_span,
        "relative_profile_rss_range": relative_profile_range,
        "profile_flat": flat,
        "profile_stable": stable,
    }
    return summary, rows


def rolling_local_slopes(
    x: np.ndarray, loss: np.ndarray, floor: float, width: int = LOCAL_SLOPE_POINTS
) -> list[Json]:
    """Return rolling log-log OLS exponents for CE minus a fixed floor."""

    x = np.asarray(x, dtype=np.float64).reshape(-1)
    loss = np.asarray(loss, dtype=np.float64).reshape(-1)
    if type(width) is not int or width < 3 or width % 2 == 0:
        raise AnalysisError("local-slope width must be an odd integer at least three")
    if x.shape != loss.shape or x.size < width:
        return []
    centered = loss - float(floor)
    if np.any(~np.isfinite(centered)) or np.any(centered <= 0.0):
        return []
    rows: list[Json] = []
    for start in range(0, x.size - width + 1):
        stop = start + width
        log_x = np.log(x[start:stop])
        log_y = np.log(centered[start:stop])
        slope, intercept = np.polyfit(log_x, log_y, 1)
        prediction = intercept + slope * log_x
        residual = log_y - prediction
        total = log_y - np.mean(log_y)
        denominator = float(np.dot(total, total))
        r_squared = 1.0 - float(np.dot(residual, residual)) / denominator if denominator > 0 else None
        rows.append(
            {
                "window_start_index": start,
                "window_stop_index_exclusive": stop,
                "window_points": width,
                "x_start": float(x[start]),
                "x_stop": float(x[stop - 1]),
                "x_center_geometric": float(math.exp(float(np.mean(log_x)))),
                "q_local": float(-slope),
                "log_centered_r_squared": r_squared,
            }
        )
    return rows


def analyze_powerlaw_curves(
    curves: Mapping[float, tuple[np.ndarray, np.ndarray]]
) -> Json:
    """Fit in-memory curves and return JSON-safe models and diagnostics.

    This pure numerical entry point is also used by the synthetic self-test.
    Artifact/campaign validation is intentionally handled separately.
    """

    thetas, xs, losses = _curve_arrays(curves)
    free = _fit_joint_model(curves, common_q=False)
    common = _fit_joint_model(curves, common_q=True)
    free_predictions = free.pop("_prediction_arrays")
    common_predictions = common.pop("_prediction_arrays")
    sensitivity, sensitivity_rows = _fixed_floor_sensitivity(
        curves, float(free["shared_L_inf"])
    )
    local_rows: list[Json] = []
    local_by_theta: dict[str, Json] = {}
    for theta, x, loss in zip(thetas, xs, losses, strict=True):
        rows = rolling_local_slopes(x, loss, float(free["shared_L_inf"]))
        for row in rows:
            local_rows.append({"theta": float(theta), **row})
        local_values = [float(row["q_local"]) for row in rows]
        local_by_theta[f"{theta:g}"] = {
            "theta": float(theta),
            "rolling_points": LOCAL_SLOPE_POINTS,
            "windows": len(rows),
            "q_local_min": min(local_values) if local_values else None,
            "q_local_max": max(local_values) if local_values else None,
            "q_local_range": (
                max(local_values) - min(local_values) if local_values else None
            ),
        }
    span_decades = {
        f"{theta:g}": float(math.log10(x[-1] / x[0]))
        for theta, x in zip(thetas, xs, strict=True)
    }
    global_reasons: list[str] = []
    identifiability = free["identifiability"]
    if identifiability["floor_at_lower_bound"]:
        global_reasons.append("shared_floor_hit_lower_bound")
    if identifiability["floor_at_upper_bound"]:
        global_reasons.append("shared_floor_hit_upper_bound")
    if not identifiability["optimizer_profile_stable"]:
        global_reasons.append("shared_fit_profile_unstable")
    if not sensitivity["profile_stable"]:
        global_reasons.append("fixed_floor_sensitivity_profile_flat_or_unstable")
    if any(value < 1.0 - 1e-12 for value in span_decades.values()):
        global_reasons.append("fit_window_shorter_than_one_decade")
    inference: dict[str, Json] = {}
    for theta, x, loss in zip(thetas, xs, losses, strict=True):
        reasons = list(global_reasons)
        if float(loss[0] - loss[-1]) <= 0.0:
            reasons.append("validation_loss_drop_nonpositive")
        arm = free["arms"][f"{theta:g}"]
        if float(arm["q"]) <= Q_LOWER + 1e-5 or float(arm["q"]) >= Q_UPPER - 1e-5:
            reasons.append("fitted_exponent_hit_bound")
        if not local_rows or local_by_theta[f"{theta:g}"]["windows"] == 0:
            reasons.append("rolling_63_local_slopes_unavailable")
        inference[f"{theta:g}"] = {
            "theta": float(theta),
            "label": INCONCLUSIVE if reasons else DESCRIPTIVE,
            "reason_codes": sorted(set(reasons)),
            "window_x_start": float(x[0]),
            "window_x_stop": float(x[-1]),
            "window_decades": span_decades[f"{theta:g}"],
            "loss_drop": float(loss[0] - loss[-1]),
            "q": float(arm["q"]),
        }
    overall = (
        INCONCLUSIVE
        if any(item["label"] == INCONCLUSIVE for item in inference.values())
        else DESCRIPTIVE
    )
    predictions: list[Json] = []
    for theta, x, loss, free_prediction, common_prediction in zip(
        thetas, xs, losses, free_predictions, common_predictions, strict=True
    ):
        for index in range(x.size):
            predictions.append(
                {
                    "theta": float(theta),
                    "evaluation_index": index,
                    "x": float(x[index]),
                    "observed_validation_cross_entropy": float(loss[index]),
                    "free_q_prediction": float(free_prediction[index]),
                    "free_q_residual": float(free_prediction[index] - loss[index]),
                    "common_q_prediction": float(common_prediction[index]),
                    "common_q_residual": float(common_prediction[index] - loss[index]),
                }
            )
    return {
        "fit_protocol": {
            "target": "ordinary_fixed_probe_validation_cross_entropy",
            "model": "L_inf + A_theta * X**(-q_theta)",
            "primary_objective": "ordinary_unweighted_least_squares_in_cross_entropy_units",
            "clock": "actual_schedule_coordinate_X",
            "floor_constraint": "one shared nonnegative L_inf below every observed CE",
            "local_slope_estimator": "rolling_63_point_OLS_on_log(CE-L_inf)_versus_log(X)",
            "bic_definition": "n*log(RSS/n)+k*log(n)",
        },
        "free_arm_exponent_fit": free,
        "common_exponent_null_fit": common,
        "model_comparison": {
            "free_rss": free["rss"],
            "common_rss": common["rss"],
            "free_bic": free["bic"],
            "common_bic": common["bic"],
            "delta_bic_free_minus_common": float(free["bic"] - common["bic"]),
            "negative_delta_favors_free_arm_exponents": True,
        },
        "fixed_floor_sensitivity": sensitivity,
        "local_slopes_shared_floor": local_by_theta,
        "exponent_inference": {
            "overall_label": overall,
            "arms": inference,
            "global_reason_codes": sorted(set(global_reasons)),
        },
        "_fixed_floor_rows": sensitivity_rows,
        "_local_slope_rows": local_rows,
        "_prediction_rows": predictions,
    }


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fields: Sequence[str]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), extrasaction="raise")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fields})
    temporary.replace(path)


def _parameter_rows(numerical: Mapping[str, Any]) -> list[Json]:
    rows: list[Json] = []
    for key in ("free_arm_exponent_fit", "common_exponent_null_fit"):
        fit = numerical[key]
        for arm in fit["arms"].values():
            rows.append(
                {
                    "model": fit["model"],
                    "theta": arm["theta"],
                    "shared_L_inf": fit["shared_L_inf"],
                    "amplitude": arm["amplitude"],
                    "q": arm["q"],
                    "arm_rss": arm["rss"],
                    "total_rss": fit["rss"],
                    "bic": fit["bic"],
                    "observations": fit["observations"],
                    "parameter_count": fit["parameter_count"],
                    "optimizer_success": fit["optimizer"]["success"],
                }
            )
    return rows


def analyze_tail_directories(
    tail_directories: Sequence[str | Path], output_dir: str | Path
) -> Json:
    """Validate the five tail directories, fit them, and write analysis files."""

    loaded = [validate_tail_directory(path) for path in tail_directories]
    curves = sorted((item[0] for item in loaded), key=lambda curve: curve.theta)
    reports_by_theta = {item[0].theta: item[1] for item in loaded}
    reports = [reports_by_theta[curve.theta] for curve in curves]
    campaign_validation = _validate_campaign(curves, reports)
    fit_input = {curve.theta: (curve.x, curve.loss) for curve in curves}
    numerical = analyze_powerlaw_curves(fit_input)
    sensitivity_rows = numerical.pop("_fixed_floor_rows")
    local_rows = numerical.pop("_local_slope_rows")
    prediction_rows = numerical.pop("_prediction_rows")
    for row in prediction_rows:
        curve = next(value for value in curves if value.theta == row["theta"])
        index = int(row["evaluation_index"])
        row.update(
            {
                "target_x": float(curve.target_x[index]),
                "tail_intrinsic_time": float(curve.tail_intrinsic_time[index]),
                "update": int(curve.update[index]),
                "tokens_seen": int(curve.tokens_seen[index]),
            }
        )
    destination = Path(output_dir).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    parameter_path = destination / "fit_parameters.csv"
    sensitivity_path = destination / "fixed_floor_sensitivity.csv"
    local_path = destination / "local_slopes_rolling63.csv"
    prediction_path = destination / "fit_predictions.csv"
    _write_csv(
        parameter_path,
        _parameter_rows(numerical),
        (
            "model", "theta", "shared_L_inf", "amplitude", "q", "arm_rss",
            "total_rss", "bic", "observations", "parameter_count", "optimizer_success",
        ),
    )
    _write_csv(
        sensitivity_path,
        sensitivity_rows,
        (
            "floor_index", "floor_source", "fixed_L_inf", "theta", "amplitude",
            "q", "arm_rss", "arm_rmse", "total_rss", "optimizer_success", "q_at_bound",
        ),
    )
    _write_csv(
        local_path,
        local_rows,
        (
            "theta", "window_start_index", "window_stop_index_exclusive",
            "window_points", "x_start", "x_stop", "x_center_geometric", "q_local",
            "log_centered_r_squared",
        ),
    )
    _write_csv(
        prediction_path,
        prediction_rows,
        (
            "theta", "evaluation_index", "target_x", "x", "tail_intrinsic_time",
            "update", "tokens_seen", "observed_validation_cross_entropy",
            "free_q_prediction", "free_q_residual", "common_q_prediction",
            "common_q_residual",
        ),
    )
    artifact_paths = (
        parameter_path,
        sensitivity_path,
        local_path,
        prediction_path,
    )
    summary: Json = {
        "schema_version": ANALYSIS_SCHEMA,
        "campaign_id": CAMPAIGN_ID,
        "status": "completed",
        "classification": "A2_EXTERNAL",
        "prediction_id": "A2_EXTERNAL",
        "theorem_facing": False,
        "claim_boundary": {
            "allowed": [
                "descriptive end-to-end fixed-probe validation-CE exponent comparison",
                "shared-floor versus common-exponent empirical model comparison",
            ],
            "prohibited": [
                "theorem proof or confirmation",
                "controlled frozen-feature P2 phase-response evidence",
                "stationary memory-kernel identification",
            ],
        },
        "campaign_validation": campaign_validation,
        "sources": reports,
        **numerical,
        "artifacts": {
            path.name: {
                "path": str(path),
                "sha256": _sha256_file(path),
                "size_bytes": path.stat().st_size,
            }
            for path in artifact_paths
        },
    }
    analysis_path = destination / "analysis.json"
    temporary = analysis_path.with_suffix(".json.tmp")
    temporary.write_bytes(_canonical_bytes(summary))
    temporary.replace(analysis_path)
    return summary


def discover_tail_directories(campaign_root: str | Path) -> list[Path]:
    """Find exactly five completed theta-tail directories beneath a root."""

    root = Path(campaign_root).expanduser().resolve()
    if not root.is_dir() or root.is_symlink():
        raise AnalysisError(f"campaign root must be a real directory: {root}")
    result_paths = [root / RESULT_FILENAME] if (root / RESULT_FILENAME).is_file() else []
    result_paths.extend(sorted(root.rglob(RESULT_FILENAME)))
    seen: set[Path] = set()
    tails: list[tuple[float, Path]] = []
    for result_path in result_paths:
        directory = result_path.parent.resolve()
        if directory in seen:
            continue
        seen.add(directory)
        try:
            result = _load_json(result_path)
        except AnalysisError:
            continue
        if (
            result.get("campaign_id") == CAMPAIGN_ID
            and result.get("phase") == "tail"
            and result.get("status") == "completed"
        ):
            theta = _number(result.get("theta"), f"{result_path}.theta")
            tails.append((theta, directory))
    if len(tails) != len(EXPECTED_THETAS):
        raise AnalysisError(
            f"campaign root must contain exactly five completed tail results; found {len(tails)}"
        )
    return [directory for _, directory in sorted(tails)]


def run_self_test() -> Json:
    """Recover known synthetic exponents and make a sub-decade veto explicit."""

    x = np.geomspace(1.0, EXPECTED_X_STOP, 410, dtype=np.float64)
    floor = 2.15
    known_q = {
        0.0: 0.18,
        0.5: 0.36,
        0.75: 0.57,
        1.0: 0.83,
        1.5: 1.18,
    }
    amplitudes = {theta: 0.38 + 0.07 * index for index, theta in enumerate(EXPECTED_THETAS)}
    curves = {}
    for index, theta in enumerate(EXPECTED_THETAS):
        clean = floor + amplitudes[theta] * np.power(x, -known_q[theta])
        perturbation = 2e-7 * np.sin((index + 1) * np.arange(x.size, dtype=np.float64))
        curves[theta] = (x, clean + perturbation)
    recovered = analyze_powerlaw_curves(curves)
    estimates = {
        theta: float(recovered["free_arm_exponent_fit"]["arms"][f"{theta:g}"]["q"])
        for theta in EXPECTED_THETAS
    }
    maximum_q_error = max(abs(estimates[theta] - known_q[theta]) for theta in EXPECTED_THETAS)
    floor_error = abs(float(recovered["free_arm_exponent_fit"]["shared_L_inf"]) - floor)
    recovery_passed = bool(maximum_q_error <= 0.02 and floor_error <= 0.02)

    short_x = np.geomspace(1.0, 9.0, 410, dtype=np.float64)
    short_curves = {
        theta: (
            short_x,
            floor + amplitudes[theta] * np.power(short_x, -known_q[theta]),
        )
        for theta in EXPECTED_THETAS
    }
    short = analyze_powerlaw_curves(short_curves)
    short_rejected = bool(
        short["exponent_inference"]["overall_label"] == INCONCLUSIVE
        and "fit_window_shorter_than_one_decade"
        in short["exponent_inference"]["global_reason_codes"]
    )
    passed = bool(recovery_passed and short_rejected)
    report = {
        "schema_version": ANALYSIS_SCHEMA + "_self_test",
        "passed": passed,
        "synthetic_recovery": {
            "known_floor": floor,
            "fitted_floor": recovered["free_arm_exponent_fit"]["shared_L_inf"],
            "floor_absolute_error": floor_error,
            "known_q": {f"{key:g}": value for key, value in known_q.items()},
            "fitted_q": {f"{key:g}": value for key, value in estimates.items()},
            "maximum_q_absolute_error": maximum_q_error,
            "tolerance": 0.02,
            "passed": recovery_passed,
        },
        "sub_decade_rejection": {
            "x_start": float(short_x[0]),
            "x_stop": float(short_x[-1]),
            "span_decades": float(math.log10(short_x[-1] / short_x[0])),
            "label": short["exponent_inference"]["overall_label"],
            "reason_codes": short["exponent_inference"]["global_reason_codes"],
            "passed": short_rejected,
        },
    }
    if not passed:
        raise AssertionError(json.dumps(report, sort_keys=True, allow_nan=False))
    return report


synthetic_self_test = run_self_test
self_test = run_self_test


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    inputs = parser.add_mutually_exclusive_group()
    inputs.add_argument(
        "--tail-dir",
        type=Path,
        action="append",
        help="one completed tail directory; repeat exactly five times",
    )
    inputs.add_argument(
        "--campaign-root",
        type=Path,
        help="root beneath which the five completed tail directories are discovered",
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args(argv)
    if args.self_test:
        if args.tail_dir or args.campaign_root or args.output_dir:
            parser.error("--self-test cannot be combined with campaign inputs or --output-dir")
        print(json.dumps(run_self_test(), sort_keys=True, allow_nan=False))
        return 0
    if args.output_dir is None:
        parser.error("--output-dir is required for campaign analysis")
    if args.tail_dir is not None:
        if len(args.tail_dir) != len(EXPECTED_THETAS):
            parser.error("--tail-dir must be repeated exactly five times")
        tail_directories = list(args.tail_dir)
    elif args.campaign_root is not None:
        tail_directories = discover_tail_directories(args.campaign_root)
    else:
        parser.error("provide five --tail-dir arguments or --campaign-root")
    summary = analyze_tail_directories(tail_directories, args.output_dir)
    print(
        json.dumps(
            {
                "analysis": str(Path(args.output_dir).expanduser().resolve() / "analysis.json"),
                "exponent_inference": summary["exponent_inference"]["overall_label"],
                "delta_bic_free_minus_common": summary["model_comparison"]["delta_bic_free_minus_common"],
                "classification": summary["classification"],
            },
            sort_keys=True,
            allow_nan=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
