"""Self-contained support-adaptive resolvent deterministic equivalent.

This module contains only the numerical ingredients used by Experiment 0b:
the power-law population spectrum, the zero-regularization real-axis
Stieltjes solver, support-adaptive refinement, and positive modal
compression.  It replaces the former runtime import from the historical
``ws_443`` workstream, while preserving that solver's numerical contract.

The checked-in density files are deterministic caches, not required inputs.
On a cache miss the bundled C backend recomputes the density locally.
"""

from __future__ import annotations

import ctypes
import math
import os
import platform
import subprocess
import time
from pathlib import Path
from typing import Any

import numpy as np
from numpy.ctypeslib import ndpointer


Array = np.ndarray
ROOT = Path(__file__).resolve().parent
C_SOURCE = ROOT / "stieltjes_exact_density.c"

AMBIENT_TO_WIDTH_RATIO = 2
DE_XMAX = 8.0
INITIAL_INTERVALS = 40_000
SEED_ETA_RELATIVE = 3.0e-4
HOMOTOPY_LEVELS = 8
CACHE_SCHEMA_VERSION = 1
MIN_MODE_MASS_FRACTION = 1.0e-18
MAX_REAL_AXIS_M_RESIDUAL = 5.0e-11

_LIBRARY_HANDLES: dict[Path, ctypes.CDLL] = {}


def power_law_objects(alpha: float, beta: float, width: int) -> dict[str, Array]:
    """Return the exact ``2m`` covariance and target sequences."""

    if alpha <= 0.0 or beta <= 0.0 or width <= 0:
        raise ValueError("alpha, beta, and width must be positive")
    ambient_dimension = AMBIENT_TO_WIDTH_RATIO * int(width)
    indices = np.arange(1, ambient_dimension + 1, dtype=float)
    covariance = indices ** (-2.0 * alpha)
    target_squared = indices ** (-2.0 * (alpha + beta))
    return {
        "covariance": covariance,
        "target_squared": target_squared,
    }


def null_space_parameter(covariance: Array, width: int) -> float:
    """Solve the zero-regularization fixed point for the null-space atom."""

    covariance = np.asarray(covariance, dtype=float)
    if width <= 0 or covariance.size < width or np.any(covariance <= 0.0):
        raise ValueError("invalid covariance or width")

    def equation(value: float) -> float:
        return (
            value
            * float(np.sum(covariance / (1.0 + value * covariance)))
            / width
        )

    low = 0.0
    high = 1.0
    while equation(high) < 1.0:
        high *= 2.0
    for _ in range(100):
        middle = 0.5 * (low + high)
        if equation(middle) < 1.0:
            low = middle
        else:
            high = middle
    return 0.5 * (low + high)


def _library_path(cache_dir: Path) -> Path:
    suffix = ".dylib" if platform.system() == "Darwin" else ".so"
    return cache_dir / "compiled_backend" / f"stieltjes_exact_density{suffix}"


def compile_c_library(cache_dir: Path) -> Path:
    """Compile the bundled Stieltjes backend when its cached library is stale."""

    if not C_SOURCE.is_file():
        raise FileNotFoundError(C_SOURCE)
    library = _library_path(cache_dir)
    library.parent.mkdir(parents=True, exist_ok=True)
    if library.exists() and library.stat().st_mtime >= C_SOURCE.stat().st_mtime:
        return library

    temporary = library.with_name(
        f"{library.stem}.{os.getpid()}.tmp{library.suffix}"
    )
    command = [
        os.environ.get("CC", "cc"),
        "-O3",
        "-ffast-math",
        "-march=native",
        "-std=c11",
        "-fPIC",
    ]
    command.append("-dynamiclib" if platform.system() == "Darwin" else "-shared")
    command.extend([str(C_SOURCE), "-lm", "-o", str(temporary)])
    subprocess.run(command, check=True)
    temporary.replace(library)
    return library


def _c_library(cache_dir: Path) -> ctypes.CDLL:
    library_path = compile_c_library(cache_dir).resolve()
    cached = _LIBRARY_HANDLES.get(library_path)
    if cached is not None:
        return cached

    library = ctypes.CDLL(str(library_path))
    function = library.plrf_stieltjes_real_axis_density
    double_vector = ndpointer(
        dtype=np.float64,
        ndim=1,
        flags=("C_CONTIGUOUS", "ALIGNED"),
    )
    int_vector = ndpointer(
        dtype=np.int32,
        ndim=1,
        flags=("C_CONTIGUOUS", "ALIGNED"),
    )
    function.argtypes = [
        double_vector,
        double_vector,
        ctypes.c_size_t,
        ctypes.c_size_t,
        double_vector,
        ctypes.c_size_t,
        ctypes.c_double,
        ctypes.c_double,
        ctypes.c_int,
        double_vector,
        double_vector,
        double_vector,
        int_vector,
    ]
    function.restype = ctypes.c_int
    _LIBRARY_HANDLES[library_path] = library
    return library


def exact_real_axis_density(
    covariance: Array,
    target_squared: Array,
    width: int,
    x: Array,
    cache_dir: Path,
    seed_eta_relative: float = SEED_ETA_RELATIVE,
    homotopy_levels: int = HOMOTOPY_LEVELS,
) -> dict[str, Any]:
    """Evaluate the zero-regularization trace and target densities."""

    covariance = np.ascontiguousarray(covariance, dtype=np.float64)
    target_squared = np.ascontiguousarray(target_squared, dtype=np.float64)
    x = np.ascontiguousarray(x, dtype=np.float64)
    if (
        covariance.ndim != 1
        or target_squared.shape != covariance.shape
        or width <= 0
        or covariance.size < width
        or x.ndim != 1
        or x.size < 2
        or np.any(np.diff(x) <= 0.0)
        or seed_eta_relative <= 0.0
        or homotopy_levels < 1
    ):
        raise ValueError("invalid real-axis density inputs")

    trace_density = np.empty_like(x)
    target_density = np.empty_like(x)
    residual = np.empty_like(x)
    iterations = np.empty(x.size, dtype=np.int32)
    seed_eta_floor = 0.1 * float(x[0])
    failure_index = _c_library(Path(cache_dir)).plrf_stieltjes_real_axis_density(
        covariance,
        target_squared,
        covariance.size,
        width,
        x,
        x.size,
        seed_eta_relative,
        seed_eta_floor,
        homotopy_levels,
        trace_density,
        target_density,
        residual,
        iterations,
    )
    if failure_index:
        raise RuntimeError(
            "real-axis Stieltjes solver failed at spectral index "
            f"{failure_index - 1}"
        )
    return {
        "trace_density": trace_density,
        "target_density": target_density,
        "max_m_residual": float(np.max(residual)),
        "max_m_iterations": int(np.max(iterations)),
    }


def density_cache_path(
    cache_dir: Path,
    phase: str,
    width: int,
    level: int,
    homotopy_levels: int = HOMOTOPY_LEVELS,
) -> Path:
    return Path(cache_dir) / (
        f"{phase}_d{width}_v{AMBIENT_TO_WIDTH_RATIO * width}"
        f"_support_adaptive_level{level}_homotopy{homotopy_levels}_density.npz"
    )


def _save_npz_atomic(path: Path, **arrays: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp.npz")
    np.savez_compressed(temporary, **arrays)
    temporary.replace(path)


def _load_density(
    cache_dir: Path,
    phase: str,
    alpha: float,
    beta: float,
    width: int,
    level: int,
    initial_intervals: int,
    seed_eta_relative: float,
    homotopy_levels: int,
) -> dict[str, Any] | None:
    path = density_cache_path(
        cache_dir, phase, width, level, homotopy_levels=homotopy_levels
    )
    if not path.exists():
        return None
    try:
        with np.load(path) as payload:
            valid = (
                int(payload["schema_version"]) == CACHE_SCHEMA_VERSION
                and str(payload["phase"]) == phase
                and math.isclose(
                    float(payload["alpha"]), alpha, rel_tol=0.0, abs_tol=1.0e-15
                )
                and math.isclose(
                    float(payload["beta"]), beta, rel_tol=0.0, abs_tol=1.0e-15
                )
                and int(payload["d"]) == width
                and int(payload["v"]) == AMBIENT_TO_WIDTH_RATIO * width
                and int(payload["adaptive_level"]) == level
                and int(payload["initial_intervals"]) == initial_intervals
                and math.isclose(
                    float(payload["seed_eta_relative"]),
                    seed_eta_relative,
                    rel_tol=0.0,
                    abs_tol=1.0e-15,
                )
                and int(payload["homotopy_levels"]) == homotopy_levels
            )
            if not valid:
                return None
            return {
                "x": payload["x"],
                "trace_density": payload["trace_density"],
                "target_density": payload["target_density"],
                "max_m_residual": float(payload["max_m_residual"]),
                "max_m_iterations": int(payload["max_m_iterations"]),
                "new_point_count": int(payload["new_point_count"]),
                "elapsed_seconds": float(payload["elapsed_seconds"]),
                "cache_source": "density_cache",
            }
    except (OSError, KeyError, ValueError):
        return None


def active_refinement_mask(
    trace_density: Array,
    target_density: Array,
    level: int,
) -> Array:
    """Select support intervals for the next midpoint refinement."""

    trace_density = np.asarray(trace_density, dtype=float)
    target_density = np.asarray(target_density, dtype=float)
    if trace_density.shape != target_density.shape or trace_density.size < 2:
        raise ValueError("density arrays must have one common nontrivial grid")
    interval_count = trace_density.size - 1
    if level == 1:
        return np.ones(interval_count, dtype=bool)
    positive = (
        (trace_density[:-1] > 0.0)
        | (trace_density[1:] > 0.0)
        | (target_density[:-1] > 0.0)
        | (target_density[1:] > 0.0)
    )
    expanded = positive.copy()
    expanded[1:] |= positive[:-1]
    expanded[:-1] |= positive[1:]
    return expanded


def compute_density_level(
    phase: str,
    alpha: float,
    beta: float,
    width: int,
    level: int,
    cache_dir: Path,
    *,
    initial_intervals: int = INITIAL_INTERVALS,
    seed_eta_relative: float = SEED_ETA_RELATIVE,
    homotopy_levels: int = HOMOTOPY_LEVELS,
) -> dict[str, Any]:
    """Load or compute one support-adaptive density level."""

    if level < 0 or initial_intervals < 8:
        raise ValueError("level must be nonnegative and the grid nontrivial")
    cached = _load_density(
        cache_dir,
        phase,
        alpha,
        beta,
        width,
        level,
        initial_intervals,
        seed_eta_relative,
        homotopy_levels,
    )
    if cached is not None:
        return cached

    objects = power_law_objects(alpha, beta, width)
    covariance = objects["covariance"]
    target_squared = objects["target_squared"]
    started = time.perf_counter()
    if level == 0:
        x_min = max(1.0e-12, float(covariance[-1]) * 2.0e-3)
        x = np.geomspace(x_min, DE_XMAX, initial_intervals + 1)
        density = exact_real_axis_density(
            covariance,
            target_squared,
            width,
            x,
            cache_dir,
            seed_eta_relative,
            homotopy_levels,
        )
        trace_density = density["trace_density"]
        target_density = density["target_density"]
        new_point_count = x.size
        max_m_residual = float(density["max_m_residual"])
        max_m_iterations = int(density["max_m_iterations"])
    else:
        previous = compute_density_level(
            phase,
            alpha,
            beta,
            width,
            level - 1,
            cache_dir,
            initial_intervals=initial_intervals,
            seed_eta_relative=seed_eta_relative,
            homotopy_levels=homotopy_levels,
        )
        previous_x = np.asarray(previous["x"], dtype=float)
        refine = active_refinement_mask(
            previous["trace_density"], previous["target_density"], level
        )
        new_x = np.sqrt(previous_x[:-1] * previous_x[1:])[refine]
        density = exact_real_axis_density(
            covariance,
            target_squared,
            width,
            new_x,
            cache_dir,
            seed_eta_relative,
            homotopy_levels,
        )
        all_x = np.concatenate((previous_x, new_x))
        all_trace = np.concatenate(
            (previous["trace_density"], density["trace_density"])
        )
        all_target = np.concatenate(
            (previous["target_density"], density["target_density"])
        )
        order = np.argsort(all_x)
        x = all_x[order]
        trace_density = all_trace[order]
        target_density = all_target[order]
        new_point_count = new_x.size
        max_m_residual = max(
            float(previous["max_m_residual"]),
            float(density["max_m_residual"]),
        )
        max_m_iterations = max(
            int(previous["max_m_iterations"]),
            int(density["max_m_iterations"]),
        )

    elapsed_seconds = time.perf_counter() - started
    path = density_cache_path(
        cache_dir, phase, width, level, homotopy_levels=homotopy_levels
    )
    _save_npz_atomic(
        path,
        schema_version=CACHE_SCHEMA_VERSION,
        phase=phase,
        alpha=alpha,
        beta=beta,
        d=width,
        v=AMBIENT_TO_WIDTH_RATIO * width,
        adaptive_level=level,
        initial_intervals=initial_intervals,
        global_midpoint_probe=(level == 1),
        support_neighbor_expansion=(level >= 2),
        seed_eta_relative=seed_eta_relative,
        homotopy_levels=homotopy_levels,
        x=x,
        trace_density=trace_density,
        target_density=target_density,
        max_m_residual=max_m_residual,
        max_m_iterations=max_m_iterations,
        new_point_count=new_point_count,
        elapsed_seconds=elapsed_seconds,
    )
    return {
        "x": x,
        "trace_density": trace_density,
        "target_density": target_density,
        "max_m_residual": max_m_residual,
        "max_m_iterations": max_m_iterations,
        "new_point_count": new_point_count,
        "elapsed_seconds": elapsed_seconds,
        "cache_source": "computed",
    }


def compress_density_local_gauss(
    x: Array,
    density: Array,
    total_mass: float,
    first_moment: float,
    mode_count: int,
) -> tuple[Array, Array]:
    """Compress a positive density with local two-node Gaussian rules."""

    x = np.asarray(x, dtype=float)
    density = np.asarray(density, dtype=float)
    if (
        x.ndim != 1
        or density.shape != x.shape
        or x.size < 2
        or np.any(np.diff(x) <= 0.0)
        or total_mass <= 0.0
        or first_moment <= 0.0
        or mode_count <= 0
    ):
        raise ValueError("invalid density-compression inputs")
    dx = np.diff(x)
    interval_moments = [
        np.maximum(
            0.0,
            0.5
            * (density[:-1] * x[:-1] ** order + density[1:] * x[1:] ** order)
            * dx,
        )
        for order in range(4)
    ]
    positive = interval_moments[0] > 0.0
    moments = [moment[positive] for moment in interval_moments]
    if not moments[0].size:
        raise RuntimeError("real-axis density has zero positive mass")
    bin_count = max(1, mode_count // 2)
    positive_interval_centers = np.sqrt(x[:-1] * x[1:])[positive]
    log_min = float(np.log(x[0]))
    log_span = float(np.log(x[-1]) - log_min)
    bins = np.minimum(
        bin_count - 1,
        np.floor(
            (np.log(positive_interval_centers) - log_min)
            * bin_count
            / log_span
        ).astype(int),
    )
    aggregated = [
        np.bincount(bins, weights=moment, minlength=bin_count)
        for moment in moments
    ]
    nodes_list: list[float] = []
    weights_list: list[float] = []
    for mass, first, second, third in zip(*aggregated, strict=True):
        if mass <= 0.0:
            continue
        mean = first / mass
        variance_mass = max(
            0.0,
            second - 2.0 * mean * first + mean * mean * mass,
        )
        if variance_mass <= 1.0e-28 * max(second, 1.0):
            nodes_list.append(mean)
            weights_list.append(mass)
            continue
        variance = variance_mass / mass
        alpha_one = (third - 2.0 * mean * second + mean * mean * first) / (
            variance_mass
        )
        jacobi = np.asarray(
            [[mean, np.sqrt(variance)], [np.sqrt(variance), alpha_one]],
            dtype=float,
        )
        local_nodes, vectors = np.linalg.eigh(jacobi)
        local_weights = mass * vectors[0] ** 2
        nodes_list.extend(local_nodes.tolist())
        weights_list.extend(local_weights.tolist())

    nodes = np.asarray(nodes_list, dtype=float)
    weights = np.asarray(weights_list, dtype=float)
    positive_modes = (
        np.isfinite(nodes)
        & np.isfinite(weights)
        & (nodes > 0.0)
        & (weights > 0.0)
    )
    nodes = nodes[positive_modes]
    weights = weights[positive_modes]
    order = np.argsort(nodes)
    nodes = nodes[order]
    weights = weights[order]
    unique_nodes, inverse = np.unique(nodes, return_inverse=True)
    if unique_nodes.size < nodes.size:
        weights = np.bincount(inverse, weights=weights, minlength=unique_nodes.size)
        nodes = unique_nodes
    keep_mass = weights >= float(np.sum(weights)) * MIN_MODE_MASS_FRACTION
    nodes = nodes[keep_mass]
    weights = weights[keep_mass]
    weights *= total_mass / float(np.sum(weights))
    modal_first_moment = float(nodes @ weights)
    nodes *= first_moment / modal_first_moment
    if nodes.size > mode_count:
        raise RuntimeError("local Gaussian compression exceeded mode_count")
    if np.any(np.diff(nodes) <= 0.0):
        raise RuntimeError("local Gaussian compression produced repeated modes")
    return nodes, weights


def density_to_modes(
    alpha: float,
    beta: float,
    width: int,
    density: dict[str, Any],
) -> dict[str, Any]:
    """Convert the resolved density to positive kernel and forcing modes."""

    objects = power_law_objects(alpha, beta, width)
    covariance = objects["covariance"]
    target_squared = objects["target_squared"]
    null_parameter = null_space_parameter(covariance, width)
    null_weight = float(
        np.sum(target_squared / (1.0 + null_parameter * covariance))
    )
    positive_target_mass = float(np.sum(target_squared) - null_weight)
    trace_first_moment = float(np.sum(covariance))
    target_first_moment = float(np.sum(target_squared * covariance))
    x = np.asarray(density["x"], dtype=float)
    trace_density = np.asarray(density["trace_density"], dtype=float)
    target_density = np.asarray(density["target_density"], dtype=float)
    raw_trace_mass = float(np.trapezoid(trace_density, x))
    raw_target_mass = float(np.trapezoid(target_density, x))
    raw_trace_first_moment = float(np.trapezoid(trace_density * x, x))
    raw_target_first_moment = float(np.trapezoid(target_density * x, x))
    kernel_nodes, kernel_trace_weights = compress_density_local_gauss(
        x,
        trace_density,
        total_mass=float(width),
        first_moment=trace_first_moment,
        mode_count=width,
    )
    forcing_nodes, forcing_weights = compress_density_local_gauss(
        x,
        target_density,
        total_mass=positive_target_mass,
        first_moment=target_first_moment,
        mode_count=width,
    )
    return {
        "kernel_nodes": kernel_nodes,
        "kernel_trace_weights": kernel_trace_weights,
        "forcing_nodes": forcing_nodes,
        "forcing_weights": forcing_weights,
        "null_weight": null_weight,
        "target_energy": float(np.sum(target_squared)),
        "raw_trace_mass_relative_error": abs(raw_trace_mass - width) / width,
        "raw_target_mass_relative_error": abs(
            raw_target_mass - positive_target_mass
        )
        / positive_target_mass,
        "raw_trace_first_moment_relative_error": abs(
            raw_trace_first_moment - trace_first_moment
        )
        / trace_first_moment,
        "raw_target_first_moment_relative_error": abs(
            raw_target_first_moment - target_first_moment
        )
        / target_first_moment,
    }


__all__ = [
    "HOMOTOPY_LEVELS",
    "INITIAL_INTERVALS",
    "MAX_REAL_AXIS_M_RESIDUAL",
    "SEED_ETA_RELATIVE",
    "active_refinement_mask",
    "compile_c_library",
    "compute_density_level",
    "density_cache_path",
    "density_to_modes",
    "exact_real_axis_density",
    "null_space_parameter",
    "power_law_objects",
]
