"""Shared, deterministic utilities for theorem-validation experiments."""

from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable

import numpy as np


Array = np.ndarray


def log_binned_indices(
    effective_width: int,
    relative_bin_width: float,
) -> tuple[Array, Array, Array, Array]:
    """Head-preserving logarithmic bins for the integer indices 1,...,m."""

    if effective_width <= 0:
        raise ValueError("effective_width must be positive")
    if not (0.0 < relative_bin_width < 1.0):
        raise ValueError("relative_bin_width must lie in (0,1)")
    starts: list[int] = []
    cursor = 1
    while cursor <= effective_width:
        starts.append(cursor)
        cursor += max(1, int(math.floor(relative_bin_width * cursor)))
    lower = np.asarray(starts, dtype=float)
    upper = np.minimum(
        float(effective_width),
        np.append(lower[1:] - 1.0, float(effective_width)),
    )
    multiplicity = upper - lower + 1.0
    nodes = np.exp(0.5 * (np.log(lower) + np.log(upper)))
    return lower, upper, nodes, multiplicity


def log_ols_exponent(times: Array, values: Array) -> dict[str, float]:
    """Fit values approximately proportional to T**(-q) on positive data."""

    time = np.asarray(times, dtype=float)
    value = np.asarray(values, dtype=float)
    if time.ndim != 1 or value.shape != time.shape:
        raise ValueError("times and values must be matching one-dimensional arrays")
    if time.size < 3 or np.any(time <= 0.0) or np.any(value <= 0.0):
        raise ValueError("positive data with at least three points are required")
    x = np.log(time)
    y = np.log(value)
    design = np.column_stack([np.ones_like(x), x])
    coefficients, _, _, _ = np.linalg.lstsq(design, y, rcond=None)
    fitted = design @ coefficients
    residual = y - fitted
    centered = y - np.mean(y)
    denominator = max(float(centered @ centered), 1.0e-300)
    return {
        "exponent": float(-coefficients[1]),
        "intercept": float(coefficients[0]),
        "r2": float(1.0 - (residual @ residual) / denominator),
    }


def local_exponent_statistics(
    times: Array,
    values: Array,
    half_window_decades: float = 0.20,
    minimum_points: int = 7,
) -> dict[str, float]:
    """Return robust local-slope summaries on a log-time grid."""

    time = np.asarray(times, dtype=float)
    value = np.asarray(values, dtype=float)
    log_time = np.log10(time)
    local: list[float] = []
    for center in log_time:
        mask = np.abs(log_time - center) <= half_window_decades
        if int(np.sum(mask)) < minimum_points:
            continue
        local.append(log_ols_exponent(time[mask], value[mask])["exponent"])
    if not local:
        return {
            "median": math.nan,
            "p10": math.nan,
            "p90": math.nan,
            "variation": math.inf,
            "count": 0,
        }
    values_local = np.asarray(local, dtype=float)
    p10, p90 = np.quantile(values_local, [0.10, 0.90])
    return {
        "median": float(np.median(values_local)),
        "p10": float(p10),
        "p90": float(p90),
        "variation": float(p90 - p10),
        "count": int(values_local.size),
    }


def positive_log_interpolate(
    source_times: Array,
    source_values: Array,
    target_times: Array,
) -> Array:
    """Interpolate positive values linearly in log(time)-log(value)."""

    source_t = np.asarray(source_times, dtype=float)
    source_v = np.asarray(source_values, dtype=float)
    target_t = np.asarray(target_times, dtype=float)
    if np.any(source_t <= 0.0) or np.any(source_v <= 0.0) or np.any(target_t <= 0.0):
        raise ValueError("positive log interpolation requires positive inputs")
    if target_t[0] < source_t[0] or target_t[-1] > source_t[-1]:
        raise ValueError("target times must remain inside source support")
    return np.exp(
        np.interp(np.log(target_t), np.log(source_t), np.log(source_v))
    )


def relative_l2(reference: Array, candidate: Array) -> float:
    reference_array = np.asarray(reference, dtype=float)
    candidate_array = np.asarray(candidate, dtype=float)
    return float(
        np.linalg.norm(candidate_array - reference_array)
        / max(np.linalg.norm(reference_array), 1.0e-300)
    )


def write_csv(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    records = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not records:
        raise ValueError("cannot write an empty CSV")
    fieldnames: list[str] = []
    for row in records:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)


def finite_json(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): finite_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [finite_json(item) for item in value]
    if isinstance(value, np.ndarray):
        return finite_json(value.tolist())
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    return value


def sha256_file(path: Path) -> str:
    """Return a reproducible source fingerprint for artifact provenance."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(finite_json(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
