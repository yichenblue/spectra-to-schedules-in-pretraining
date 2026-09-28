"""Rerun experiment one at m=100,000 with an exact-power continuum path.

The run is intentionally isolated from historical integer-batch, discrete
exact, and true-SGD artifacts.  It uses

    r_theta(T) = 128 * max(1,T)**theta

and exact inverse-ratio mass on every continuum time cell.  The original eight
schedule exponents remain the frozen primary grid; theta=0 is a separately
reported destroy control, while theta=1.5 and theta=2.5 are separately reported
strong-preserve sentinels.  Theta=0.9 is retained as a near-boundary stress test
rather than promoted as an absolute-gap exponent gate.
"""

from __future__ import annotations

import csv
import json
import math
import os
from pathlib import Path
import time
from typing import Any

import numpy as np

from .de_quadrature import (
    DEQuadrature,
    DETrajectory,
    head_preserving_quadrature,
    triangular_time_grid,
)
from .power_ratio_continuum import (
    exact_power_inverse_ratio_cell_integrals,
    run_exact_power_ratio_continuum_modal_dynamics,
)
from .slopes import local_log_slopes


ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "artifacts" / "exact_power_continuum_m100000"
OLD_OUTPUT_DIR = ROOT / "artifacts" / "continuum_only_m100000"

WIDTH = 100_000
ALPHA = 0.4
BETA = 0.3
SIGMA2 = 10.0
INVERSE_RATIO_AMPLITUDE = 1.0 / 128.0
EQUIVALENT_BATCH_SCALE = 16.0
PRIMARY_THETAS = (0.25, 0.35, 0.50, 0.65, 0.75, 0.90, 1.00, 1.10)
CONTROL_THETAS = (0.00,)
SENTINEL_THETAS = (1.50, 2.50)
THETAS = CONTROL_THETAS + PRIMARY_THETAS + SENTINEL_THETAS
REPRESENTATIVES = (0.00, 0.25, 0.50, 1.50, 2.50)
PAPER_THETAS = (0.00, 0.25, 0.50, 2.00)
PAPER_FIGURE_ASPECT_RATIO = 12.40 / 4.15
PAPER_FIGURE_SIZE_INCHES = (12.40, 4.15)
PAPER_FIGURE_LAYOUT = "wide single-row: panels a/b/c side by side; legend below"
STRESS_THETA = 0.90
REFERENCE_THETA = 0.25

TIME_MIN = 1.0e-3
HORIZON_EXPONENT = 0.56
FIT_LOWER_EXPONENT = 0.25
FIT_UPPER_EXPONENT = 0.50
FIT_POINTS = 41
PANEL_A_DISPLAY_LOWER_TIME = 0.1
PANEL_A_DISPLAY_POINTS = 181
FULL_LOSS_GUIDE_WINDOWS = {
    0.25: (80.0, 160.0),
    0.50: (80.0, 160.0),
    1.50: (80.0, 160.0),
}
FULL_LOSS_CONTROL_GUIDE_WINDOW = (80.0, 160.0)
FULL_LOSS_GUIDE_OFFSETS = {
    0.25: 1.15,
    0.50: 1.15,
    1.50: 1.15,
}
FULL_LOSS_CONTROL_GUIDE_OFFSET = 0.82
LOCAL_HALF_WINDOW_DECADES = 0.15
BASE_TIME_POINTS_PER_DECADE = 36
REFINED_TIME_POINTS_PER_DECADE = 72
BASE_SPECTRAL_BIN_WIDTH = 2.0e-3
REFINED_SPECTRAL_BIN_WIDTH = 1.0e-3

MAXIMUM_EXPONENT_ERROR = 0.05
MAXIMUM_LOCAL_VARIATION = 0.15
MAXIMUM_RESOLUTION_CURVE_ERROR = 3.0e-3
MAXIMUM_RESOLUTION_SLOPE_ERROR = 2.0e-3
MAXIMUM_CELL_MASS_CONSERVATION_ERROR = 1.0e-12
ROW_MASS_CAP = 0.20


def _q_clean_theory() -> float:
    return (2.0 * ALPHA + 2.0 * BETA - 1.0) / (2.0 * ALPHA)


def _q_kernel_theory() -> float:
    return 2.0 - 1.0 / (2.0 * ALPHA)


def _destroy_boundary_theory() -> float:
    return 1.0 - _q_kernel_theory()


def _q_gap_theory(theta: float) -> float:
    destroy = _destroy_boundary_theory()
    if theta <= destroy:
        return 0.0
    if theta < 1.0:
        return theta + _q_kernel_theory() - 1.0
    return _q_kernel_theory()


def _q_phase_theory(theta: float) -> float:
    return min(_q_clean_theory(), _q_gap_theory(theta))


def _admissible_decay_theory(theta: float) -> tuple[float, float] | None:
    if theta < _destroy_boundary_theory() - 1.0e-12:
        return None
    return _q_gap_theory(theta), _q_phase_theory(theta)


def _component_guide_exponent(theta: float) -> float:
    if theta < _destroy_boundary_theory() - 1.0e-12:
        return theta + _q_kernel_theory() - 1.0
    return _q_gap_theory(theta)


def _phase(theta: float) -> str:
    if theta < _destroy_boundary_theory() - 1.0e-12:
        return "below-boundary instability"
    if theta <= _destroy_boundary_theory() + 1.0e-12:
        return "destroy boundary"
    gap = _q_gap_theory(theta)
    if gap < _q_clean_theory() - 1.0e-12:
        return "change"
    if abs(gap - _q_clean_theory()) <= 1.0e-12:
        return "critical"
    return "preserve"


def _positive_log_interpolate(
    target_times: np.ndarray,
    source_times: np.ndarray,
    source_values: np.ndarray,
) -> np.ndarray:
    valid = (
        (source_times > 0.0)
        & (source_values > 0.0)
        & np.isfinite(source_values)
    )
    if int(np.count_nonzero(valid)) < 2:
        raise ValueError("positive log interpolation needs at least two points")
    return np.exp(
        np.interp(
            np.log(target_times),
            np.log(source_times[valid]),
            np.log(source_values[valid]),
        )
    )


def _exponent(times: np.ndarray, values: np.ndarray) -> float:
    if np.any(times <= 0.0) or np.any(values <= 0.0):
        raise ValueError("power-law fit requires positive inputs")
    return -float(np.polyfit(np.log(times), np.log(values), 1)[0])


def _increasing_exponent(times: np.ndarray, values: np.ndarray) -> float:
    return -_exponent(times, values)


def _local_statistics(
    times: np.ndarray,
    values: np.ndarray,
) -> tuple[np.ndarray, float, float]:
    slopes = local_log_slopes(
        times,
        values,
        half_window_decades=LOCAL_HALF_WINDOW_DECADES,
        minimum_points=7,
    )
    finite = slopes[np.isfinite(slopes)]
    if finite.size < 3:
        return slopes, float("nan"), float("inf")
    median = float(np.median(finite))
    variation = float(
        (np.quantile(finite, 0.9) - np.quantile(finite, 0.1))
        / max(abs(median), 0.05)
    )
    return slopes, median, variation


def _run_route(
    spectral_bin_width: float,
    time_points_per_decade: int,
    sigma2: float = SIGMA2,
    theta_values: tuple[float, ...] = THETAS,
) -> tuple[DEQuadrature, dict[float, DETrajectory], np.ndarray]:
    quadrature = head_preserving_quadrature(
        alpha=ALPHA,
        beta=BETA,
        effective_width=WIDTH,
        relative_bin_width=spectral_bin_width,
    )
    times = triangular_time_grid(
        effective_width=WIDTH,
        horizon_exponent=HORIZON_EXPONENT,
        time_min=TIME_MIN,
        points_per_decade=time_points_per_decade,
    )
    trajectories = run_exact_power_ratio_continuum_modal_dynamics(
        quadrature=quadrature,
        times=times,
        theta_values=theta_values,
        sigma2=sigma2,
        inverse_ratio_amplitude=INVERSE_RATIO_AMPLITUDE,
        row_mass_cap=ROW_MASS_CAP,
    )
    return quadrature, trajectories, times


def _process(
    quadrature: DEQuadrature,
    trajectories: dict[float, DETrajectory],
    target_times: np.ndarray,
    theta_values: tuple[float, ...] = THETAS,
    display_only_thetas: tuple[float, ...] = (),
) -> dict[str, Any]:
    clean: dict[float, np.ndarray] = {}
    centered_clean: dict[float, np.ndarray] = {}
    gap: dict[float, np.ndarray] = {}
    total: dict[float, np.ndarray] = {}
    for theta in theta_values:
        trajectory = trajectories[theta]
        centered_clean[theta] = _positive_log_interpolate(
            target_times, trajectory.times, trajectory.clean_centered
        )
        clean[theta] = centered_clean[theta] + quadrature.approximation_floor
        gap[theta] = _positive_log_interpolate(
            target_times, trajectory.times, trajectory.noise_gap
        )
        total[theta] = clean[theta] + gap[theta]

    primary_clean_stack = np.stack([clean[theta] for theta in PRIMARY_THETAS])
    pooled_clean = np.exp(np.mean(np.log(primary_clean_stack), axis=0))
    primary_centered_stack = np.stack(
        [centered_clean[theta] for theta in PRIMARY_THETAS]
    )
    pooled_centered_clean = np.exp(
        np.mean(np.log(primary_centered_stack), axis=0)
    )
    normalized_clean = pooled_clean / pooled_clean[0]
    q_clean = _exponent(target_times, pooled_clean)
    q_centered_clean = _exponent(target_times, pooled_centered_clean)
    clean_slopes, clean_local_median, clean_local_variation = _local_statistics(
        target_times, pooled_clean
    )

    reference = gap[REFERENCE_THETA]
    contrasts: dict[float, np.ndarray] = {}
    contrast_slopes: dict[float, np.ndarray] = {}
    envelopes: dict[float, np.ndarray] = {}
    envelope_slopes: dict[float, np.ndarray] = {}
    metrics: dict[float, dict[str, Any]] = {}
    for theta in theta_values:
        contrast = gap[theta] / reference
        contrast = contrast / contrast[0]
        contrasts[theta] = contrast
        q_contrast = _exponent(target_times, contrast)
        contrast_local, contrast_median, contrast_variation = _local_statistics(
            target_times, contrast
        )
        contrast_slopes[theta] = contrast_local

        envelope = np.maximum(normalized_clean, contrast)
        envelopes[theta] = envelope
        q_envelope = _exponent(target_times, envelope)
        envelope_local, envelope_median, envelope_variation = _local_statistics(
            target_times, envelope
        )
        envelope_slopes[theta] = envelope_local
        admissible_theory = _admissible_decay_theory(theta)
        if admissible_theory is None:
            gap_theory = None
            phase_theory = None
            relative_gap_error = None
            phase_envelope_error = None
        else:
            gap_theory, phase_theory = admissible_theory
            relative_gap_error = abs(q_contrast - gap_theory)
            phase_envelope_error = abs(q_envelope - phase_theory)
        metrics[theta] = {
            "theta": theta,
            "group": (
                "primary"
                if theta in PRIMARY_THETAS
                else "control"
                if theta in CONTROL_THETAS
                else "sentinel"
                if theta in SENTINEL_THETAS
                else "display-only"
                if theta in display_only_thetas
                else "unclassified"
            ),
            "phase": _phase(theta),
            "theory_gap_exponent": gap_theory,
            "theory_phase_exponent": phase_theory,
            "measured_absolute_gap_exponent": _exponent(target_times, gap[theta]),
            "measured_relative_gap_exponent": q_contrast,
            "relative_gap_error": relative_gap_error,
            "relative_gap_local_median": contrast_median,
            "relative_gap_local_variation": contrast_variation,
            "measured_phase_envelope_exponent": q_envelope,
            "phase_envelope_error": phase_envelope_error,
            "phase_envelope_local_median": envelope_median,
            "phase_envelope_local_variation": envelope_variation,
            "raw_total_exponent": _exponent(target_times, total[theta]),
            "phase_exponent_gate": bool(
                phase_theory is not None
                and phase_envelope_error is not None
                and phase_envelope_error <= MAXIMUM_EXPONENT_ERROR
            ),
            "phase_local_gate": envelope_variation <= MAXIMUM_LOCAL_VARIATION,
            "relative_gap_exponent_gate": bool(
                gap_theory is not None
                and relative_gap_error is not None
                and relative_gap_error <= MAXIMUM_EXPONENT_ERROR
            ),
            "included_in_primary_phase_gate": theta in PRIMARY_THETAS,
            "included_in_sentinel_phase_gate": theta in SENTINEL_THETAS,
            "included_in_overall_observable_gate": (
                theta in PRIMARY_THETAS or theta in SENTINEL_THETAS
            ),
            "included_in_frozen_numerical_contract": theta in THETAS,
        }

    primary_phase_errors = np.asarray(
        [metrics[theta]["phase_envelope_error"] for theta in PRIMARY_THETAS]
    )
    return {
        "target_times": target_times,
        "clean": clean,
        "centered_clean": centered_clean,
        "gap": gap,
        "total": total,
        "pooled_clean": pooled_clean,
        "pooled_centered_clean": pooled_centered_clean,
        "normalized_clean": normalized_clean,
        "q_clean": q_clean,
        "q_centered_clean": q_centered_clean,
        "clean_slopes": clean_slopes,
        "clean_local_median": clean_local_median,
        "clean_local_variation": clean_local_variation,
        "contrasts": contrasts,
        "contrast_slopes": contrast_slopes,
        "envelopes": envelopes,
        "envelope_slopes": envelope_slopes,
        "metrics": metrics,
        "primary_phase_mae": float(np.mean(primary_phase_errors)),
        "primary_phase_max_error": float(np.max(primary_phase_errors)),
        "primary_phase_exponent_pass_count": sum(
            bool(metrics[theta]["phase_exponent_gate"])
            for theta in PRIMARY_THETAS
        ),
        "primary_phase_local_pass_count": sum(
            bool(metrics[theta]["phase_local_gate"])
            for theta in PRIMARY_THETAS
        ),
        "primary_gap_exponent_pass_count": sum(
            bool(metrics[theta]["relative_gap_exponent_gate"])
            for theta in PRIMARY_THETAS
        ),
    }


def _compare_resolution(
    comparison: str,
    candidate: dict[str, Any],
    refined: dict[str, Any],
    theta_values: tuple[float, ...] = THETAS,
    included_in_numerical_gate: bool = True,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    def add(observable: str, theta: float | None, left: np.ndarray, right: np.ndarray) -> None:
        relative = np.abs(left - right) / np.maximum(np.abs(right), 1.0e-300)
        curve_error = float(np.max(relative))
        slope_error = abs(
            _exponent(refined["target_times"], left)
            - _exponent(refined["target_times"], right)
        )
        rows.append(
            {
                "comparison": comparison,
                "observable": observable,
                "theta": "" if theta is None else theta,
                "maximum_relative_curve_difference": curve_error,
                "absolute_slope_difference": slope_error,
                "included_in_numerical_gate": included_in_numerical_gate,
                "status": "PASS"
                if curve_error <= MAXIMUM_RESOLUTION_CURVE_ERROR
                and slope_error <= MAXIMUM_RESOLUTION_SLOPE_ERROR
                else "FAIL",
            }
        )

    if included_in_numerical_gate:
        add("pooled_tail_completed_clean", None, candidate["pooled_clean"], refined["pooled_clean"])
        add("pooled_centered_clean", None, candidate["pooled_centered_clean"], refined["pooled_centered_clean"])
    for theta in theta_values:
        add("absolute_gap", theta, candidate["gap"][theta], refined["gap"][theta])
        add("relative_gap_contrast", theta, candidate["contrasts"][theta], refined["contrasts"][theta])
        add("phase_envelope", theta, candidate["envelopes"][theta], refined["envelopes"][theta])
        add("raw_total", theta, candidate["total"][theta], refined["total"][theta])
    return rows


def _schedule_diagnostics(times: np.ndarray) -> dict[str, Any]:
    pure = EQUIVALENT_BATCH_SCALE * np.maximum(1.0, times) ** STRESS_THETA
    shifted = EQUIVALENT_BATCH_SCALE * (1.0 + times) ** STRESS_THETA
    shifted_ceiling = np.ceil(shifted)
    annotated_times = np.geomspace(10.0, 100.0, 41)
    annotated_shifted = EQUIVALENT_BATCH_SCALE * (
        1.0 + annotated_times
    ) ** STRESS_THETA
    return {
        "main_window": {
            "pure_real_exponent": _increasing_exponent(times, pure),
            "shifted_real_exponent": _increasing_exponent(times, shifted),
            "shifted_ceiling_exponent": _increasing_exponent(
                times, shifted_ceiling
            ),
        },
        "annotated_window_10_100": {
            "pure_real_exponent": STRESS_THETA,
            "shifted_real_exponent": _increasing_exponent(
                annotated_times, annotated_shifted
            ),
            "shifted_ceiling_exponent": _increasing_exponent(
                annotated_times, np.ceil(annotated_shifted)
            ),
        },
    }


def _load_old_stress_curve(target_times: np.ndarray) -> dict[str, Any] | None:
    curve_path = OLD_OUTPUT_DIR / "curves.csv"
    summary_path = OLD_OUTPUT_DIR / "summary.json"
    if not curve_path.exists() or not summary_path.exists():
        return None
    times: list[float] = []
    values: list[float] = []
    with curve_path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["route"] == "refined" and abs(float(row["theta"]) - STRESS_THETA) < 1.0e-12:
                times.append(float(row["intrinsic_time"]))
                values.append(float(row["relative_gap_contrast"]))
    if len(times) < 2:
        return None
    order = np.argsort(np.asarray(times))
    old_times = np.asarray(times)[order]
    old_values = np.asarray(values)[order]
    interpolated = _positive_log_interpolate(target_times, old_times, old_values)
    local_slopes, _, _ = _local_statistics(target_times, interpolated)
    return {
        "values": interpolated,
        "exponent": _exponent(target_times, interpolated),
        "local_slopes": local_slopes,
        "source": str(curve_path),
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _finite(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _finite(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_finite(item) for item in value]
    if isinstance(value, np.ndarray):
        return _finite(value.tolist())
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if math.isfinite(number) else None
    return value


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(_finite(payload), indent=2, sort_keys=True, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _representative_marker_style(theta: float) -> dict[str, Any]:
    marker = "s" if theta == 0.00 else "o" if theta == 2.50 else None
    if marker is None:
        return {}
    return {
        "marker": marker,
        "markevery": 18,
        "markersize": 3.4,
        "markerfacecolor": "white",
        "markeredgewidth": 0.9,
    }


def _make_figure(
    refined: dict[str, Any],
    coarse: dict[str, Any],
    display_refined: dict[str, Any],
    display_coarse: dict[str, Any],
    old_stress: dict[str, Any] | None,
    sigma2: float,
    output_path: Path,
) -> tuple[Path, Path]:
    os.environ.setdefault("MPLCONFIGDIR", str(output_path.parent / ".matplotlib"))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.ticker import LogLocator, NullFormatter, ScalarFormatter

    colors = {
        "clean": "#b43f8f",
        0.00: "#c44e52",
        0.25: "#2878b5",
        0.50: "#d18700",
        1.50: "#3a9856",
        2.50: "#8e5aad",
        0.90: "#6857a6",
    }
    fit_times = refined["target_times"]
    display_times = display_refined["target_times"]
    figure, axes = plt.subplots(1, 3, figsize=(14.5, 4.05))

    axis = axes[0]
    axis.loglog(display_times, display_refined["normalized_clean"], color=colors["clean"], linewidth=2.2, label="tail-completed clean")
    axis.loglog(display_times, display_coarse["normalized_clean"], color=colors["clean"], linewidth=1.0, linestyle="--")
    guide_mask = (display_times >= fit_times[0]) & (
        display_times <= fit_times[-1]
    )
    guide_times = display_times[guide_mask]
    clean_anchor = display_refined["normalized_clean"][guide_mask][0]
    axis.loglog(
        guide_times,
        clean_anchor * (guide_times / guide_times[0]) ** (-_q_clean_theory()),
        color=colors["clean"],
        linewidth=1.0,
        linestyle=":",
    )
    for theta in REPRESENTATIVES:
        axis.loglog(
            display_times,
            display_refined["contrasts"][theta],
            color=colors[theta],
            linewidth=2.2,
            label=rf"$\vartheta={theta:g}$",
            **_representative_marker_style(theta),
        )
        axis.loglog(display_times, display_coarse["contrasts"][theta], color=colors[theta], linewidth=1.0, linestyle="--")
        contrast_anchor = display_refined["contrasts"][theta][guide_mask][0]
        axis.loglog(
            guide_times,
            contrast_anchor
            * (guide_times / guide_times[0])
            ** (-_component_guide_exponent(theta)),
            color=colors[theta],
            linewidth=1.0,
            linestyle=":",
        )
    axis.axvspan(
        PANEL_A_DISPLAY_LOWER_TIME,
        1.0,
        color="#7f7f7f",
        alpha=0.07,
        linewidth=0.0,
    )
    axis.axvspan(
        fit_times[-1],
        display_times[-1],
        color="#7f7f7f",
        alpha=0.055,
        linewidth=0.0,
    )
    axis.axvline(1.0, color="#888888", linewidth=0.8, linestyle=":")
    for fit_boundary in (fit_times[0], fit_times[-1]):
        axis.axvline(
            fit_boundary,
            color="#888888",
            linewidth=0.8,
            linestyle=":",
        )
    axis.set_xlim(PANEL_A_DISPLAY_LOWER_TIME, display_times[-1])
    axis.set_xlabel(r"intrinsic time $T$")
    axis.set_ylabel(r"normalized component at $T=0.1$")
    axis.set_title(r"(a) Component race from $T=0.1$")
    axis.grid(True, which="both", alpha=0.17)
    primary_legend = axis.legend(frameon=False, fontsize=8, loc="lower left")
    axis.add_artist(primary_legend)
    axis.legend(
        handles=[
            Line2D([0], [0], color="black", linewidth=2.2, label="refined continuum"),
            Line2D([0], [0], color="black", linewidth=1.0, linestyle="--", label="coarse control"),
            Line2D([0], [0], color="black", linewidth=1.0, linestyle=":", label="fit-window theory slope"),
        ],
        frameon=False,
        fontsize=7,
        loc="upper left",
    )
    axis.xaxis.set_major_locator(LogLocator(base=10.0, subs=(1.0,)))
    axis.xaxis.set_major_formatter(ScalarFormatter())
    axis.xaxis.set_minor_formatter(NullFormatter())

    axis = axes[1]
    theta_grid = np.linspace(
        _destroy_boundary_theory(), max(SENTINEL_THETAS), 500
    )
    axis.plot(theta_grid, [_q_phase_theory(value) for value in theta_grid], color="black", linewidth=2.0, label="theory phase envelope")
    axis.scatter(
        PRIMARY_THETAS,
        [refined["metrics"][theta]["measured_phase_envelope_exponent"] for theta in PRIMARY_THETAS],
        color="#2f4858",
        s=43,
        label="primary grid",
        zorder=4,
    )
    axis.scatter(
        CONTROL_THETAS,
        [
            refined["metrics"][theta]["measured_phase_envelope_exponent"]
            for theta in CONTROL_THETAS
        ],
        marker="s",
        facecolors="white",
        edgecolors=colors[0.00],
        linewidths=1.5,
        s=52,
        label=r"destroy control $\vartheta=0$",
        zorder=5,
    )
    axis.scatter(
        SENTINEL_THETAS,
        [refined["metrics"][theta]["measured_phase_envelope_exponent"] for theta in SENTINEL_THETAS],
        marker="*",
        color="#3a9856",
        s=105,
        label=r"strong-preserve sentinels $\vartheta=1.5,2.5$",
        zorder=5,
    )
    axis.scatter(
        THETAS,
        [refined["metrics"][theta]["measured_relative_gap_exponent"] for theta in THETAS],
        marker="o",
        facecolors="white",
        edgecolors="#d18700",
        s=30,
        label="relative gap",
        zorder=3,
    )
    axis.scatter(
        [STRESS_THETA],
        [refined["metrics"][STRESS_THETA]["measured_relative_gap_exponent"]],
        marker="D",
        facecolors="white",
        edgecolors=colors[STRESS_THETA],
        s=52,
        linewidths=1.4,
        label=r"near-$1$ stress point",
        zorder=6,
    )
    axis.axvspan(
        -0.05,
        _destroy_boundary_theory(),
        color="#c44e52",
        alpha=0.06,
        linewidth=0.0,
        zorder=0,
    )
    axis.text(
        0.10,
        0.12,
        "unstable",
        color="#a33d42",
        fontsize=7,
        ha="center",
        va="center",
        rotation=90,
    )
    axis.axhline(refined["q_clean"], color=colors["clean"], linestyle="--", linewidth=1.1, label="clean reference")
    for boundary in (0.25, 0.75, 1.0):
        axis.axvline(boundary, color="#888888", linestyle=":", linewidth=0.9)
    axis.set_xlim(-0.05, 2.60)
    axis.set_ylim(-0.35, 1.05)
    axis.set_xlabel(r"schedule exponent $\vartheta$")
    axis.set_ylabel("effective exponent")
    axis.set_title("(b) Phase envelope, control, and sentinels")
    axis.grid(True, alpha=0.17)
    axis.legend(frameon=False, fontsize=7, loc="upper left")

    axis = axes[2]
    times = fit_times
    new_q = refined["metrics"][STRESS_THETA]["measured_relative_gap_exponent"]
    axis.semilogx(
        times,
        refined["contrast_slopes"][STRESS_THETA],
        color=colors[STRESS_THETA],
        linewidth=2.3,
        label=rf"pure-power gap, $\hat q={new_q:.3f}$",
    )
    if old_stress is not None:
        axis.semilogx(
            times,
            old_stress["local_slopes"],
            color="#777777",
            linewidth=1.7,
            linestyle="--",
            label=rf"old shifted+ceil gap, $\hat q={old_stress['exponent']:.3f}$",
        )
    axis.semilogx(
        times,
        refined["clean_slopes"],
        color=colors["clean"],
        linewidth=1.6,
        linestyle="--",
        label=rf"clean, $\hat q={refined['q_clean']:.3f}$",
    )
    axis.semilogx(
        times,
        refined["envelope_slopes"][STRESS_THETA],
        color="#2f4858",
        linewidth=1.2,
        linestyle="-.",
        label="phase envelope",
    )
    axis.axhline(
        _q_gap_theory(STRESS_THETA),
        color=colors[STRESS_THETA],
        linewidth=1.0,
        linestyle=":",
    )
    axis.axhline(
        _q_clean_theory(),
        color=colors["clean"],
        linewidth=1.0,
        linestyle=":",
    )
    axis.set_xlabel(r"intrinsic time $T$")
    axis.set_ylabel("local effective exponent")
    axis.set_title(r"(c) Frozen-window stress test, $\vartheta=0.90$")
    axis.grid(True, which="both", alpha=0.17)
    axis.set_ylim(0.45, 0.68)
    axis.legend(
        frameon=False,
        fontsize=7,
        loc="center left",
        bbox_to_anchor=(1.01, 0.5),
    )
    axis.xaxis.set_major_locator(LogLocator(base=10.0, subs=(1.0, 2.0, 5.0)))
    axis.xaxis.set_major_formatter(ScalarFormatter())
    axis.xaxis.set_minor_formatter(NullFormatter())

    figure.suptitle(
        rf"Exact intrinsic-time power schedule: $m=10^5$, $\sigma^2={sigma2:g}$, "
        r"$r_\vartheta(T)=128\max(1,T)^\vartheta$; "
        rf"display $T\leq {display_times[-1]:.0f}$ (shaded tail not fitted)",
        fontsize=11,
        y=0.995,
    )
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.94))
    png = output_path.with_suffix(".png")
    pdf = output_path.with_suffix(".pdf")
    figure.savefig(png, dpi=220, bbox_inches="tight")
    figure.savefig(pdf, bbox_inches="tight")
    plt.close(figure)
    return png, pdf


def _full_loss_fit_metrics(refined: dict[str, Any]) -> dict[float, dict[str, Any]]:
    times = refined["target_times"]
    metrics: dict[float, dict[str, Any]] = {}
    for theta in REPRESENTATIVES:
        centered_total = refined["centered_clean"][theta] + refined["gap"][theta]
        measured_centered = _exponent(times, centered_total)
        measured_raw = _exponent(times, refined["total"][theta])
        admissible_theory = _admissible_decay_theory(theta)
        theory = (
            None if admissible_theory is None else admissible_theory[1]
        )
        metrics[theta] = {
            "theta": theta,
            "theory_phase_exponent": theory,
            "measured_floor_centered_total_exponent": measured_centered,
            "floor_centered_total_exponent_error": (
                None if theory is None else abs(measured_centered - theory)
            ),
            "measured_raw_total_exponent": measured_raw,
            "raw_total_exponent_error": (
                None if theory is None else abs(measured_raw - theory)
            ),
        }
    return metrics


def _make_full_loss_figure(
    refined: dict[str, Any],
    display_refined: dict[str, Any],
    approximation_floor: float,
    sigma2: float,
    output_path: Path,
) -> tuple[Path, Path]:
    """Plot actual total losses and the correctly centered theory target.

    The raw panel is an authenticity view and deliberately has no asymptotic
    slope guides: the paper's phase exponent is stated for loss after removing
    the fixed rank-m approximation floor.  The right panel subtracts exactly
    that precomputed floor, so the short dashed segments there are the relevant
    theoretical comparisons rather than fitted lines.  The separately styled
    theta=0 dash-dot segment is only the formal below-boundary direct-response
    continuation; it is not a stable full-loss exponent.
    """

    os.environ.setdefault("MPLCONFIGDIR", str(output_path.parent / ".matplotlib"))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import LogLocator, NullFormatter, ScalarFormatter

    colors = {
        0.00: "#c44e52",
        0.25: "#2878b5",
        0.50: "#d18700",
        1.50: "#3a9856",
        2.50: "#8e5aad",
    }
    times = display_refined["target_times"]
    fit_times = refined["target_times"]
    fit_metrics = _full_loss_fit_metrics(refined)
    figure, axes = plt.subplots(
        1, 2, figsize=(12.4, 4.55), sharex=True, sharey=True
    )

    raw_axis, centered_axis = axes
    for theta in REPRESENTATIVES:
        raw_axis.loglog(
            times,
            display_refined["total"][theta],
            color=colors[theta],
            linewidth=2.35,
            label=(
                rf"$\vartheta={theta:g}$"
                rf"  ($\hat q_{{\rm raw}}={fit_metrics[theta]['measured_raw_total_exponent']:.3f}$)"
            ),
            **_representative_marker_style(theta),
        )
    raw_axis.axhline(
        approximation_floor,
        color="#666666",
        linewidth=1.25,
        linestyle=(0, (4, 2)),
        label=rf"fixed $R_{{\mathrm{{app}}}}={approximation_floor:.3f}$",
    )
    for theta in REPRESENTATIVES:
        centered_total = (
            display_refined["centered_clean"][theta]
            + display_refined["gap"][theta]
        )
        centered_axis.loglog(
            times,
            centered_total,
            color=colors[theta],
            linewidth=2.35,
            label=(
                rf"$\vartheta={theta:g}$"
                rf"  ($\hat q={fit_metrics[theta]['measured_floor_centered_total_exponent']:.3f}$)"
            ),
            **_representative_marker_style(theta),
        )

        # Theta=0 is below the stability boundary and has no admissible total
        # decay exponent.  Draw only its separately styled formal direct-gap
        # growth continuation, with an explicit warning that it is not a
        # full-loss theorem.
        if theta == 0.00:
            guide_left, guide_right = FULL_LOSS_CONTROL_GUIDE_WINDOW
            guide_times = np.geomspace(guide_left, guide_right, 31)
            guide_anchor_time = math.sqrt(guide_left * guide_right)
            guide_anchor_value = float(
                np.exp(
                    np.interp(
                        math.log(guide_anchor_time),
                        np.log(times),
                        np.log(centered_total),
                    )
                )
            )
            formal_exponent = _component_guide_exponent(theta)
            guide_values = (
                FULL_LOSS_CONTROL_GUIDE_OFFSET
                * guide_anchor_value
                * (guide_times / guide_anchor_time) ** (-formal_exponent)
            )
            centered_axis.loglog(
                guide_times,
                guide_values,
                color=colors[theta],
                linewidth=1.8,
                linestyle=(0, (5, 1.5, 1.2, 1.5)),
                solid_capstyle="round",
                zorder=5,
            )
            centered_axis.annotate(
                (
                    r"formal $q_{\rm dir}=-0.25$"
                    "\n"
                    r"(no stable full-loss $q$)"
                ),
                xy=(guide_times[-1], guide_values[-1]),
                xytext=(5, 2),
                textcoords="offset points",
                color=colors[theta],
                fontsize=8,
                ha="left",
                va="center",
            )
            continue

        # Theta=1.5 and theta=2.5 share q_phase=0.5, so one guide avoids
        # coincident dashed lines while both trajectories remain visible.
        if theta == 2.50:
            continue

        guide_left, guide_right = FULL_LOSS_GUIDE_WINDOWS[theta]
        guide_times = np.geomspace(guide_left, guide_right, 31)
        guide_anchor_time = math.sqrt(guide_left * guide_right)
        guide_anchor_value = float(
            np.exp(
                np.interp(
                    math.log(guide_anchor_time),
                    np.log(times),
                    np.log(centered_total),
                )
            )
        )
        theory_exponent = _q_phase_theory(theta)
        guide_values = (
            FULL_LOSS_GUIDE_OFFSETS[theta]
            * guide_anchor_value
            * (guide_times / guide_anchor_time) ** (-theory_exponent)
        )
        centered_axis.loglog(
            guide_times,
            guide_values,
            color=colors[theta],
            linewidth=2.0,
            linestyle=(0, (4, 2)),
            solid_capstyle="round",
            zorder=5,
        )
        guide_label = rf"theory $q={theory_exponent:g}$"
        if theta == 1.50:
            guide_label += r"  ($\vartheta=1.5,2.5$)"
        centered_axis.annotate(
            guide_label,
            xy=(guide_times[-1], guide_values[-1]),
            xytext=(5, 2),
            textcoords="offset points",
            color=colors[theta],
            fontsize=8,
            va="bottom",
        )

    for axis in axes:
        axis.axvspan(
            PANEL_A_DISPLAY_LOWER_TIME,
            1.0,
            color="#7f7f7f",
            alpha=0.07,
            linewidth=0.0,
        )
        axis.axvspan(
            fit_times[-1],
            times[-1],
            color="#7f7f7f",
            alpha=0.055,
            linewidth=0.0,
        )
        for fit_boundary in (fit_times[0], fit_times[-1]):
            axis.axvline(
                fit_boundary,
                color="#888888",
                linewidth=0.9,
                linestyle=":",
            )
        axis.set_xlim(PANEL_A_DISPLAY_LOWER_TIME, times[-1])
        axis.set_xlabel(r"intrinsic time $T$")
        axis.grid(True, which="both", alpha=0.17)
        axis.xaxis.set_major_locator(LogLocator(base=10.0, subs=(1.0,)))
        axis.xaxis.set_major_formatter(ScalarFormatter())
        axis.xaxis.set_minor_formatter(NullFormatter())

    raw_axis.set_ylabel("population loss")
    raw_axis.set_title(r"(a) Raw total loss $R_\sigma(T)$")
    raw_axis.legend(frameon=False, fontsize=8, loc="lower left")

    centered_axis.set_ylabel(r"floor-centered population loss")
    centered_axis.set_title(
        r"(b) Floor-centered total $R_\sigma(T)-R_{\mathrm{app}}$"
    )
    curve_legend = centered_axis.legend(
        frameon=False, fontsize=8, loc="lower left"
    )
    centered_axis.add_artist(curve_legend)

    figure.suptitle(
        rf"Finite-width total-risk trajectories: $m=10^5$, $\sigma^2={sigma2:g}$, "
        rf"display $T\leq {times[-1]:.0f}$; dotted lines bracket the frozen fit window",
        fontsize=11,
        y=1.01,
    )
    figure.tight_layout()
    png = output_path.with_suffix(".png")
    pdf = output_path.with_suffix(".pdf")
    figure.savefig(png, dpi=220, bbox_inches="tight")
    figure.savefig(pdf, bbox_inches="tight")
    plt.close(figure)
    return png, pdf


def _paper_figure_fit_metrics(
    refined: dict[str, Any],
    sigma2: float,
) -> dict[float, dict[str, Any]]:
    """Return descriptive finite-window metrics for the compact paper figure."""

    times = refined["target_times"]
    rows: dict[float, dict[str, Any]] = {}
    for theta in PAPER_THETAS:
        gap = refined["gap"][theta]
        centered_total = refined["centered_clean"][theta] + gap
        _, gap_local_median, gap_local_variation = _local_statistics(times, gap)
        _, total_local_median, total_local_variation = _local_statistics(
            times, centered_total
        )
        admissible = _admissible_decay_theory(theta)
        rows[theta] = {
            "route": "refined",
            "effective_width": WIDTH,
            "sigma2": sigma2,
            "theta": theta,
            "group": refined["metrics"][theta]["group"],
            "role": (
                "finite-horizon instability control"
                if theta == 0.0
                else _phase(theta)
            ),
            "theory_noise_gap_exponent": (
                None if theta == 0.0 else _q_gap_theory(theta)
            ),
            "formal_direct_response_exponent": (
                _component_guide_exponent(theta)
                if theta == 0.0
                else None
            ),
            "theory_floor_centered_total_exponent": (
                None if admissible is None else admissible[1]
            ),
            "measured_noise_gap_exponent": _exponent(times, gap),
            "noise_gap_local_median": gap_local_median,
            "noise_gap_local_variation": gap_local_variation,
            "measured_floor_centered_total_exponent": _exponent(
                times, centered_total
            ),
            "floor_centered_total_local_median": total_local_median,
            "floor_centered_total_local_variation": total_local_variation,
            "measured_component_envelope_exponent": refined["metrics"][theta][
                "measured_phase_envelope_exponent"
            ],
            "theory_component_envelope_exponent": (
                None if admissible is None else admissible[1]
            ),
            "included_in_frozen_primary_or_sentinel_gate": (
                theta in PRIMARY_THETAS or theta in SENTINEL_THETAS
            ),
        }
    return rows


def _draw_paper_power_guide(
    axis: Any,
    source_times: np.ndarray,
    source_values: np.ndarray,
    exponent: float,
    color: str,
    label: str,
    *,
    linestyle: Any = (0, (4, 2)),
    visual_offset: float = 1.10,
    label_offset_points: tuple[float, float] = (0.0, 4.0),
    label_vertical_alignment: str = "bottom",
    label_at_right: bool = False,
    label_axes_x: float = 0.97,
) -> None:
    """Draw one fixed-window asymptotic guide with a common visual offset."""

    guide_left, guide_right = 80.0, 160.0
    guide_times = np.geomspace(guide_left, guide_right, 31)
    anchor_time = math.sqrt(guide_left * guide_right)
    anchor_value = float(
        np.exp(
            np.interp(
                math.log(anchor_time),
                np.log(source_times),
                np.log(source_values),
            )
        )
    )
    guide_values = (
        visual_offset
        * anchor_value
        * (guide_times / anchor_time) ** (-exponent)
    )
    axis.loglog(
        guide_times,
        guide_values,
        color=color,
        linewidth=1.45,
        linestyle=linestyle,
        zorder=5,
    )
    if label:
        label_value = (
            guide_values[-1] if label_at_right else visual_offset * anchor_value
        )
        label_location = (
            (label_axes_x, label_value)
            if label_at_right
            else (anchor_time, label_value)
        )
        axis.annotate(
            label,
            xy=label_location,
            xycoords=axis.get_yaxis_transform() if label_at_right else "data",
            xytext=label_offset_points,
            textcoords="offset points",
            color=color,
            fontsize=12.0,
            ha="right" if label_at_right else "center",
            va=label_vertical_alignment,
            zorder=8,
        )


def _make_compact_paper_triptych(
    refined: dict[str, Any],
    display_refined: dict[str, Any],
    sigma2: float,
    output_path: Path,
) -> tuple[Path, Path]:
    """Create the claim-scoped three-panel figure requested for the paper."""

    os.environ.setdefault("MPLCONFIGDIR", str(output_path.parent / ".matplotlib"))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.ticker import LogLocator, NullFormatter, ScalarFormatter

    colors = {
        0.00: "#9f9f9f",
        0.25: "#b23a48",
        0.50: "#d9822b",
        2.00: "#3b6fb6",
    }
    empirical_linewidth = 2.1
    times = display_refined["target_times"]
    fit_times = refined["target_times"]
    figure = plt.figure(figsize=PAPER_FIGURE_SIZE_INCHES)
    grid = figure.add_gridspec(
        1,
        3,
        left=0.065,
        right=0.985,
        bottom=0.245,
        top=0.895,
        wspace=0.48,
    )
    total_axis = figure.add_subplot(grid[0, 0])
    gap_axis = figure.add_subplot(grid[0, 1])
    phase_axis = figure.add_subplot(grid[0, 2])

    for theta in PAPER_THETAS:
        centered_total = (
            display_refined["centered_clean"][theta]
            + display_refined["gap"][theta]
        )
        total_axis.loglog(
            times,
            centered_total,
            color=colors[theta],
            linewidth=empirical_linewidth,
            linestyle="-",
            alpha=0.78 if theta == 0.0 else 1.0,
            zorder=3,
        )
        gap_axis.loglog(
            times,
            display_refined["gap"][theta],
            color=colors[theta],
            linewidth=empirical_linewidth,
            linestyle="-",
            alpha=0.78 if theta == 0.0 else 1.0,
            zorder=3,
        )

    for theta, label in (
        (0.25, r"$T^0$"),
        (0.50, r"$T^{-1/4}$"),
        (2.00, r"$T^{-1/2}$"),
    ):
        centered_total = (
            display_refined["centered_clean"][theta]
            + display_refined["gap"][theta]
        )
        _draw_paper_power_guide(
            total_axis,
            times,
            centered_total,
            _q_phase_theory(theta),
            colors[theta],
            label,
            label_offset_points=(-2.0, 14.0) if theta == 2.0 else (-2.0, 4.0),
            label_vertical_alignment="bottom",
            label_at_right=True,
            label_axes_x=0.91 if theta == 2.0 else 0.88,
        )

    gap_labels = {
        0.00: "",
        0.25: r"$T^0$",
        0.50: r"$T^{-1/4}$",
        2.00: r"$T^{-3/4}$",
    }
    for theta in PAPER_THETAS:
        _draw_paper_power_guide(
            gap_axis,
            times,
            display_refined["gap"][theta],
            _component_guide_exponent(theta),
            colors[theta],
            gap_labels[theta],
            linestyle="-." if theta == 0.0 else (0, (4, 2)),
            visual_offset=0.82 if theta == 0.0 else 1.10,
            label_offset_points=(
                (-2.0, -4.0)
                if theta == 0.0
                else (-2.0, 14.0)
                if theta == 2.0
                else (-2.0, 4.0)
            ),
            label_vertical_alignment="top" if theta == 0.0 else "bottom",
            label_at_right=True,
            label_axes_x=0.91 if theta == 2.0 else 0.88,
        )
    gap_axis.text(
        0.26,
        0.99,
        "$T^{1/4}$\n(unstable)",
        transform=gap_axis.transAxes,
        color=colors[0.0],
        fontsize=12.0,
        ha="left",
        va="top",
        zorder=8,
    )

    for axis in (total_axis, gap_axis):
        axis.axvspan(
            fit_times[-1],
            times[-1],
            color="#7f7f7f",
            alpha=0.055,
            linewidth=0.0,
            zorder=0,
        )
        for fit_boundary in (fit_times[0], fit_times[-1]):
            axis.axvline(
                fit_boundary,
                color="#888888",
                linewidth=0.75,
                linestyle=":",
                zorder=1,
            )
        axis.set_xlim(PANEL_A_DISPLAY_LOWER_TIME, times[-1])
        axis.set_xlabel(r"intrinsic time $T$", fontsize=15.0)
        axis.grid(True, which="both", alpha=0.13)
        axis.tick_params(axis="both", which="major", labelsize=9.3)
        axis.xaxis.set_major_locator(LogLocator(base=10.0, subs=(1.0,)))
        axis.xaxis.set_major_formatter(ScalarFormatter())
        axis.xaxis.set_minor_formatter(NullFormatter())

    total_axis.set_ylabel(r"$R_\sigma(T)-R_{\mathrm{app}}$", fontsize=15.0)
    total_axis.set_title("(a) Floor-centered total risk", fontsize=15.0)
    gap_axis.set_ylabel(r"$R_\sigma(T)-R_0(T)$", fontsize=15.0)
    gap_axis.set_title("(b) Noisy–clean gap", fontsize=15.0)

    destroy = _destroy_boundary_theory()
    preserve = destroy + _q_clean_theory()
    phase_axis.axvspan(
        -0.04, destroy, color="#777777", alpha=0.075, linewidth=0.0
    )
    phase_axis.axvspan(
        destroy, preserve, color="#d9822b", alpha=0.075, linewidth=0.0
    )
    phase_axis.axvspan(
        preserve, 2.08, color="#3b6fb6", alpha=0.065, linewidth=0.0
    )
    phase_theta = np.linspace(destroy, 2.05, 500)
    phase_axis.plot(
        phase_theta,
        [_q_phase_theory(theta) for theta in phase_theta],
        color="#202020",
        linewidth=2.0,
        zorder=3,
    )
    for theta in PAPER_THETAS:
        if theta == 0.0:
            continue
        phase_axis.scatter(
            [theta],
            [
                refined["metrics"][theta][
                    "measured_phase_envelope_exponent"
                ]
            ],
            color=colors[theta],
            edgecolors="white",
            linewidths=0.7,
            s=46,
            alpha=1.0,
            zorder=5,
        )
    phase_axis.axvline(
        destroy, color="#b23a48", linewidth=0.9, linestyle=":"
    )
    phase_axis.axvline(
        preserve, color="#777777", linewidth=0.9, linestyle=":"
    )
    phase_axis.axhline(0.0, color="#999999", linewidth=0.6, alpha=0.5)
    phase_axis.text(
        0.105,
        0.30,
        "unstable",
        color="#777777",
        fontsize=12.0,
        ha="center",
        rotation=90,
    )
    phase_axis.text(
        0.50,
        0.53,
        "changed",
        color="#a65f1c",
        fontsize=12.0,
        ha="center",
        va="center",
        rotation=90,
    )
    phase_axis.text(
        1.40, 0.63, "preserved", color="#315f9b", fontsize=12.0, ha="center"
    )
    phase_axis.text(
        1.40,
        0.43,
        r"theory: $\min\{q_0,q_{\mathcal{N}}(\vartheta)\}$",
        color="#202020",
        fontsize=9.2,
        ha="center",
        va="top",
    )
    phase_axis.set_xlim(-0.04, 2.08)
    phase_axis.set_ylim(-0.20, 0.70)
    phase_axis.set_xlabel(r"noise-control exponent $\vartheta$", fontsize=15.0)
    phase_axis.set_ylabel("component-envelope exponent", fontsize=15.0)
    phase_axis.set_title("(c) Preserve–change–destroy", fontsize=15.0)
    phase_axis.grid(True, alpha=0.11)
    phase_axis.tick_params(axis="both", which="major", labelsize=9.3)

    legend_handles = [
        Line2D(
            [0],
            [0],
            color=colors[theta],
            linewidth=empirical_linewidth,
            linestyle="-",
            alpha=0.78 if theta == 0.0 else 1.0,
            label=(
                r"$\vartheta=0$"
                if theta == 0.0
                else r"$\vartheta=2$"
                if theta == 2.0
                else r"$\vartheta=1/4$"
                if theta == 0.25
                else r"$\vartheta=1/2$"
            ),
        )
        for theta in PAPER_THETAS
    ]
    figure.legend(
        handles=legend_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.025),
        ncol=4,
        frameon=False,
        fontsize=15.0,
        handlelength=2.0,
        columnspacing=1.5,
    )
    png = output_path.with_suffix(".png")
    pdf = output_path.with_suffix(".pdf")
    figure.savefig(png, dpi=320)
    figure.savefig(pdf)
    plt.close(figure)
    return png, pdf


def run_experiment(
    output_dir: Path = OUTPUT_DIR,
    sigma2: float = SIGMA2,
    display_only_thetas: tuple[float, ...] = (),
    make_compact_triptych: bool = False,
) -> dict[str, Any]:
    started = time.perf_counter()
    output_dir.mkdir(parents=True, exist_ok=True)
    display_only_thetas = tuple(float(theta) for theta in display_only_thetas)
    if len(set(display_only_thetas)) != len(display_only_thetas):
        raise ValueError("display-only theta values must be unique")
    if set(display_only_thetas).intersection(THETAS):
        raise ValueError("display-only theta values must not alter frozen grids")
    simulation_thetas = THETAS + display_only_thetas
    if make_compact_triptych and not set(PAPER_THETAS).issubset(
        simulation_thetas
    ):
        raise ValueError(
            "compact paper figure requires theta values 0, 0.25, 0.5, and 2"
        )
    lower = float(WIDTH) ** FIT_LOWER_EXPONENT
    upper = float(WIDTH) ** FIT_UPPER_EXPONENT
    display_upper = float(WIDTH) ** HORIZON_EXPONENT
    target_times = np.geomspace(lower, upper, FIT_POINTS)
    display_times = np.unique(
        np.concatenate(
            [
                np.geomspace(
                    PANEL_A_DISPLAY_LOWER_TIME,
                    display_upper,
                    PANEL_A_DISPLAY_POINTS,
                ),
                target_times,
            ]
        )
    )

    route_specs = {
        "coarse": (BASE_SPECTRAL_BIN_WIDTH, BASE_TIME_POINTS_PER_DECADE),
        "time_refined": (BASE_SPECTRAL_BIN_WIDTH, REFINED_TIME_POINTS_PER_DECADE),
        "spectrum_refined": (REFINED_SPECTRAL_BIN_WIDTH, BASE_TIME_POINTS_PER_DECADE),
        "refined": (REFINED_SPECTRAL_BIN_WIDTH, REFINED_TIME_POINTS_PER_DECADE),
    }
    quadratures: dict[str, DEQuadrature] = {}
    trajectories: dict[str, dict[float, DETrajectory]] = {}
    grids: dict[str, np.ndarray] = {}
    processed: dict[str, dict[str, Any]] = {}
    display_processed: dict[str, dict[str, Any]] = {}
    for route, (spectral_width, time_density) in route_specs.items():
        quadrature, route_trajectories, grid = _run_route(
            spectral_width,
            time_density,
            sigma2,
            theta_values=simulation_thetas,
        )
        quadratures[route] = quadrature
        trajectories[route] = route_trajectories
        grids[route] = grid
        endpoint_tolerance = 64.0 * np.finfo(float).eps * max(
            1.0, float(grid[-1]), float(display_times[-1])
        )
        if float(display_times[-1]) > float(grid[-1]) + endpoint_tolerance:
            raise RuntimeError(
                "display horizon exceeds the computed continuum trajectory"
            )
        processed[route] = _process(
            quadrature,
            route_trajectories,
            target_times,
            theta_values=simulation_thetas,
            display_only_thetas=display_only_thetas,
        )
        display_processed[route] = _process(
            quadrature,
            route_trajectories,
            display_times,
            theta_values=simulation_thetas,
            display_only_thetas=display_only_thetas,
        )

    refined = processed["refined"]
    resolution_rows: list[dict[str, Any]] = []
    resolution_rows.extend(_compare_resolution("joint", processed["coarse"], refined))
    resolution_rows.extend(_compare_resolution("time_only", processed["spectrum_refined"], refined))
    resolution_rows.extend(_compare_resolution("spectrum_only", processed["time_refined"], refined))
    if display_only_thetas:
        resolution_rows.extend(
            _compare_resolution(
                "display_only_joint",
                processed["coarse"],
                refined,
                theta_values=display_only_thetas,
                included_in_numerical_gate=False,
            )
        )
        resolution_rows.extend(
            _compare_resolution(
                "display_only_time_only",
                processed["spectrum_refined"],
                refined,
                theta_values=display_only_thetas,
                included_in_numerical_gate=False,
            )
        )
        resolution_rows.extend(
            _compare_resolution(
                "display_only_spectrum_only",
                processed["time_refined"],
                refined,
                theta_values=display_only_thetas,
                included_in_numerical_gate=False,
            )
        )

    # Cell-integral conservation catches omitted, duplicated, or shifted cells.
    cell_mass_errors: dict[str, float] = {}
    display_only_cell_mass_errors: dict[str, float] = {}
    for route, grid in grids.items():
        cell_masses = exact_power_inverse_ratio_cell_integrals(
            grid, THETAS, INVERSE_RATIO_AMPLITUDE
        ).sum(axis=1)
        closed = exact_power_inverse_ratio_cell_integrals(
            np.asarray([0.0, grid[-1]]), THETAS, INVERSE_RATIO_AMPLITUDE
        )[:, 0]
        cell_mass_errors[route] = float(
            np.max(np.abs(cell_masses - closed) / np.maximum(closed, 1.0e-300))
        )
        if display_only_thetas:
            display_cell_masses = exact_power_inverse_ratio_cell_integrals(
                grid, display_only_thetas, INVERSE_RATIO_AMPLITUDE
            ).sum(axis=1)
            display_closed = exact_power_inverse_ratio_cell_integrals(
                np.asarray([0.0, grid[-1]]),
                display_only_thetas,
                INVERSE_RATIO_AMPLITUDE,
            )[:, 0]
            display_only_cell_mass_errors[route] = float(
                np.max(
                    np.abs(display_cell_masses - display_closed)
                    / np.maximum(display_closed, 1.0e-300)
                )
            )

    old_stress = _load_old_stress_curve(target_times)
    figure_png, figure_pdf = _make_figure(
        refined,
        processed["coarse"],
        display_processed["refined"],
        display_processed["coarse"],
        old_stress,
        sigma2,
        output_dir / "exact_power_continuum_m100000",
    )
    full_loss_metrics = _full_loss_fit_metrics(refined)
    full_loss_figure_png, full_loss_figure_pdf = _make_full_loss_figure(
        refined,
        display_processed["refined"],
        quadratures["refined"].approximation_floor,
        sigma2,
        output_dir / "full_loss_curves_m100000",
    )
    paper_figure_metrics: dict[float, dict[str, Any]] = {}
    compact_figure_png: Path | None = None
    compact_figure_pdf: Path | None = None
    if make_compact_triptych:
        paper_figure_metrics = _paper_figure_fit_metrics(refined, sigma2)
        compact_figure_png, compact_figure_pdf = _make_compact_paper_triptych(
            refined,
            display_processed["refined"],
            sigma2,
            output_dir / "preserve_change_destroy_continuum_diagnostic",
        )

    curve_rows: list[dict[str, Any]] = []
    phase_rows: list[dict[str, Any]] = []
    for route, result in processed.items():
        for theta in simulation_thetas:
            phase_rows.append({"route": route, **result["metrics"][theta]})
            for index, intrinsic_time in enumerate(target_times):
                curve_rows.append(
                    {
                        "route": route,
                        "effective_width": WIDTH,
                        "theta": theta,
                        "intrinsic_time": intrinsic_time,
                        "tail_completed_clean": result["clean"][theta][index],
                        "centered_clean": result["centered_clean"][theta][index],
                        "pooled_tail_completed_clean": result["pooled_clean"][index],
                        "absolute_gap": result["gap"][theta][index],
                        "relative_gap_contrast": result["contrasts"][theta][index],
                        "phase_envelope": result["envelopes"][theta][index],
                        "raw_total": result["total"][theta][index],
                        "relative_gap_local_exponent": result["contrast_slopes"][theta][index],
                        "phase_envelope_local_exponent": result["envelope_slopes"][theta][index],
                    }
                )

    _write_csv(output_dir / "curves.csv", curve_rows)
    display_curve_rows: list[dict[str, Any]] = []
    for route in ("coarse", "refined"):
        result = display_processed[route]
        for theta in simulation_thetas:
            for index, intrinsic_time in enumerate(display_times):
                display_curve_rows.append(
                    {
                        "route": route,
                        "effective_width": WIDTH,
                        "theta": theta,
                        "intrinsic_time": intrinsic_time,
                        "normalized_tail_completed_clean": result[
                            "normalized_clean"
                        ][index],
                        "relative_gap_contrast": result["contrasts"][theta][
                            index
                        ],
                    }
                )
    _write_csv(output_dir / "panel_a_display_curves.csv", display_curve_rows)
    full_loss_display_rows: list[dict[str, Any]] = []
    for route in ("coarse", "refined"):
        result = display_processed[route]
        floor = quadratures[route].approximation_floor
        for theta in REPRESENTATIVES:
            for index, intrinsic_time in enumerate(display_times):
                floor_centered_total = (
                    result["centered_clean"][theta][index]
                    + result["gap"][theta][index]
                )
                full_loss_display_rows.append(
                    {
                        "route": route,
                        "effective_width": WIDTH,
                        "theta": theta,
                        "intrinsic_time": intrinsic_time,
                        "centered_clean": result["centered_clean"][theta][index],
                        "absolute_gap": result["gap"][theta][index],
                        "approximation_floor": floor,
                        "floor_centered_total": floor_centered_total,
                        "raw_total": result["total"][theta][index],
                    }
                )
    _write_csv(
        output_dir / "full_loss_display_curves.csv", full_loss_display_rows
    )
    _write_csv(
        output_dir / "full_loss_fit_metrics.csv",
        [full_loss_metrics[theta] for theta in REPRESENTATIVES],
    )
    if make_compact_triptych:
        compact_curve_rows: list[dict[str, Any]] = []
        for route in ("coarse", "refined"):
            result = display_processed[route]
            for theta in PAPER_THETAS:
                for index, intrinsic_time in enumerate(display_times):
                    compact_curve_rows.append(
                        {
                            "route": route,
                            "effective_width": WIDTH,
                            "sigma2": sigma2,
                            "theta": theta,
                            "intrinsic_time": intrinsic_time,
                            "floor_centered_total": (
                                result["centered_clean"][theta][index]
                                + result["gap"][theta][index]
                            ),
                            "noise_clean_gap": result["gap"][theta][index],
                            "normalized_component_envelope": result[
                                "envelopes"
                            ][theta][index],
                        }
                    )
        _write_csv(
            output_dir / "joint_schedule_triptych_curves.csv",
            compact_curve_rows,
        )
        _write_csv(
            output_dir / "joint_schedule_triptych_metrics.csv",
            [paper_figure_metrics[theta] for theta in PAPER_THETAS],
        )
    _write_csv(output_dir / "phase_metrics.csv", phase_rows)
    _write_csv(output_dir / "resolution_checks.csv", resolution_rows)

    frozen_resolution_rows = [
        row for row in resolution_rows if row["included_in_numerical_gate"]
    ]
    display_resolution_rows = [
        row for row in resolution_rows if not row["included_in_numerical_gate"]
    ]
    maximum_curve_error = max(
        row["maximum_relative_curve_difference"]
        for row in frozen_resolution_rows
    )
    maximum_slope_error = max(
        row["absolute_slope_difference"] for row in frozen_resolution_rows
    )
    resolution_pass = all(
        row["status"] == "PASS" for row in frozen_resolution_rows
    )
    display_resolution_pass: bool | None = (
        all(row["status"] == "PASS" for row in display_resolution_rows)
        if display_only_thetas
        else None
    )
    stability_pass = all(
        trajectories[route][theta].stable
        for route in trajectories
        for theta in THETAS
    )
    display_stability_pass: bool | None = (
        all(
            trajectories[route][theta].stable
            for route in trajectories
            for theta in display_only_thetas
        )
        if display_only_thetas
        else None
    )
    primary_phase_pass = bool(
        refined["primary_phase_exponent_pass_count"] == len(PRIMARY_THETAS)
        and refined["primary_phase_local_pass_count"] == len(PRIMARY_THETAS)
    )
    sentinel_phase_pass = all(
        bool(refined["metrics"][theta]["phase_exponent_gate"])
        and bool(refined["metrics"][theta]["phase_local_gate"])
        for theta in SENTINEL_THETAS
    )
    schedule_diagnostics = _schedule_diagnostics(target_times)

    stress_payload: dict[str, Any] = {
        "theta": STRESS_THETA,
        "theory_gap_exponent": _q_gap_theory(STRESS_THETA),
        "new_relative_gap_exponent": refined["metrics"][STRESS_THETA]["measured_relative_gap_exponent"],
        "new_relative_gap_error": refined["metrics"][STRESS_THETA]["relative_gap_error"],
        "new_gap_gate": refined["metrics"][STRESS_THETA]["relative_gap_exponent_gate"],
        "new_phase_envelope_exponent": refined["metrics"][STRESS_THETA]["measured_phase_envelope_exponent"],
        "new_phase_gate": refined["metrics"][STRESS_THETA]["phase_exponent_gate"],
    }
    stress_gap_local = refined["contrast_slopes"][STRESS_THETA]
    stress_clean_local = refined["clean_slopes"]
    finite_gap_local = stress_gap_local[np.isfinite(stress_gap_local)]
    finite_clean_local = stress_clean_local[np.isfinite(stress_clean_local)]
    stress_payload.update(
        {
            "new_gap_local_minimum": float(np.min(finite_gap_local)),
            "new_gap_local_maximum": float(np.max(finite_gap_local)),
            "clean_local_minimum": float(np.min(finite_clean_local)),
            "clean_local_maximum": float(np.max(finite_clean_local)),
            "preserve_order_margin": float(
                np.min(finite_gap_local) - np.max(finite_clean_local)
            ),
            "preserve_order_gate": bool(
                np.min(finite_gap_local) > np.max(finite_clean_local)
            ),
        }
    )
    if old_stress is not None:
        stress_payload.update(
            {
                "old_relative_gap_exponent": old_stress["exponent"],
                "gap_exponent_improvement": refined["metrics"][STRESS_THETA]["measured_relative_gap_exponent"] - old_stress["exponent"],
                "old_source": old_stress["source"],
            }
        )

    not_claimed = [
        "raw-total phase-law validation",
        "absolute-gap convergence at theta=0.90",
        "memory-ceiling gap exponent from the theta=1.50/2.50 sentinels",
        "finite-random-feature deterministic equivalence",
        "independent discovery of the destroy boundary",
        "paper promotion while the external computation gate is blocked",
    ]
    if 2.0 in display_only_thetas:
        not_claimed.append(
            "memory-ceiling absolute-gap convergence from the theta=2 "
            "display-only trajectory"
        )

    summary = {
        "experiment": (
            "exact_power_continuum_experiment_one_m100000_"
            f"sigma2_{sigma2:g}"
        ),
        "status": "PILOT_UNPROMOTED",
        "object_scope": "analytic-spectrum continuum proxy at effective width m=100000",
        "not_run": ["integer batch schedule", "discrete exact recursion", "true Gaussian SGD"],
        "not_claimed": not_claimed,
        "protocol": {
            "effective_width": WIDTH,
            "alpha": ALPHA,
            "beta": BETA,
            "sigma2": sigma2,
            "ratio_path": "r_theta(T)=128*max(1,T)^theta",
            "inverse_ratio_cell_rule": "exact integral per cell; cell-mean times exact constant modal survival",
            "primary_thetas": PRIMARY_THETAS,
            "control_thetas": CONTROL_THETAS,
            "sentinel_thetas": SENTINEL_THETAS,
            "representative_thetas": REPRESENTATIVES,
            "display_only_thetas": display_only_thetas,
            "compact_paper_thetas": PAPER_THETAS
            if make_compact_triptych
            else (),
            "stress_theta": STRESS_THETA,
            "reference_theta": REFERENCE_THETA,
            "fit_window": [lower, upper],
            "fit_window_width_exponents": [FIT_LOWER_EXPONENT, FIT_UPPER_EXPONENT],
            "fit_points": FIT_POINTS,
            "fit_weighting": "41 equal-weight log-uniform points",
            "panel_a_display_window": [
                PANEL_A_DISPLAY_LOWER_TIME,
                display_upper,
            ],
            "panel_a_display_only_tail": [upper, display_upper],
            "display_horizon_exponent": HORIZON_EXPONENT,
            "panel_a_normalization_time": PANEL_A_DISPLAY_LOWER_TIME,
            "panel_a_theory_guides_start": lower,
            "panel_a_theory_guide_window": [lower, upper],
            "local_slope_half_window_decades": LOCAL_HALF_WINDOW_DECADES,
            "route_specs": route_specs,
        },
        "theory": {
            "q_clean": _q_clean_theory(),
            "q_kernel": _q_kernel_theory(),
            "destroy_boundary": _destroy_boundary_theory(),
            "preservation_boundary": _destroy_boundary_theory()
            + _q_clean_theory(),
            "gap_exponents": {
                theta: (
                    None
                    if _admissible_decay_theory(theta) is None
                    else _admissible_decay_theory(theta)[0]
                )
                for theta in THETAS
            },
            "phase_exponents": {
                theta: (
                    None
                    if _admissible_decay_theory(theta) is None
                    else _admissible_decay_theory(theta)[1]
                )
                for theta in THETAS
            },
            "display_only_gap_exponents": {
                theta: _q_gap_theory(theta) for theta in display_only_thetas
            },
            "display_only_centered_total_exponents": {
                theta: _q_phase_theory(theta)
                for theta in display_only_thetas
            },
        },
        "refined_results": {
            "tail_completed_clean_exponent": refined["q_clean"],
            "centered_clean_exponent": refined["q_centered_clean"],
            "tail_completed_clean_local_variation": refined["clean_local_variation"],
            "primary_phase_mae": refined["primary_phase_mae"],
            "primary_phase_max_error": refined["primary_phase_max_error"],
            "primary_phase_exponent_pass_count": refined["primary_phase_exponent_pass_count"],
            "primary_phase_local_pass_count": refined["primary_phase_local_pass_count"],
            "primary_gap_exponent_pass_count": refined["primary_gap_exponent_pass_count"],
            "clean_proxy_gate_pass": bool(
                abs(refined["q_clean"] - _q_clean_theory())
                <= MAXIMUM_EXPONENT_ERROR
                and refined["clean_local_variation"]
                <= MAXIMUM_LOCAL_VARIATION
            ),
            "primary_phase_mae_gate_pass": bool(
                refined["primary_phase_mae"] <= 0.025
            ),
            "metrics": refined["metrics"],
        },
        "schedule_diagnostics": schedule_diagnostics,
        "stress_test_theta_0.90": stress_payload,
        "strong_preserve_sentinels": {
            theta: refined["metrics"][theta] for theta in SENTINEL_THETAS
        },
        "destroy_control": {
            "status": "DESCRIPTIVE_CONTROL",
            "included_in_overall_gate": False,
            "interpretation": (
                "below-boundary instability check; the direct response grows "
                "and no admissible infinite-horizon decay exponent is claimed"
            ),
            "metrics": refined["metrics"][0.00],
        },
        "destroy_control_theta_0.00": refined["metrics"][0.00],
        "strong_preserve_sentinel_theta_1.50": refined["metrics"][1.50],
        "strong_preserve_sentinel_theta_2.50": refined["metrics"][2.50],
        "compact_paper_figure_diagnostic": {
            "status": "DESCRIPTIVE_ONLY",
            "generated": make_compact_triptych,
            "layout": PAPER_FIGURE_LAYOUT,
            "figure_size_inches": PAPER_FIGURE_SIZE_INCHES,
            "theta_values": PAPER_THETAS if make_compact_triptych else (),
            "fit_window": [lower, upper],
            "display_window": [PANEL_A_DISPLAY_LOWER_TIME, display_upper],
            "guide_window": [80.0, 160.0],
            "guide_semantics": (
                "asymptotic predictions with one common visual offset; not fits"
            ),
            "component_envelope_semantics": (
                "normalized finite-width component diagnostic; not the direct "
                "floor-centered total-loss slope"
            ),
            "theta_zero_semantics": (
                "no stable full-loss exponent; q=-0.25 is drawn only as the "
                "formal direct-response growth prediction in the noisy-clean gap"
            ),
            "metrics": paper_figure_metrics,
        },
        "full_loss_diagnostic": {
            "status": "DESCRIPTIVE_ONLY",
            "display_window": [PANEL_A_DISPLAY_LOWER_TIME, display_upper],
            "display_only_tail": [upper, display_upper],
            "fit_window": [lower, upper],
            "raw_panel_role": (
                "authentic full population loss; asymptotic phase guides are "
                "intentionally not applied before subtracting the fixed floor"
            ),
            "theory_target": "R_sigma(T)-R_app",
            "theory_guide_semantics": (
                "short admissible asymptotic reference segments are not "
                "finite-window fits; the separately styled theta=0 segment "
                "is only a formal direct-response growth continuation"
            ),
            "guide_windows": FULL_LOSS_GUIDE_WINDOWS,
            "below_boundary_direct_response_guide": {
                "theta": 0.0,
                "formal_direct_gap_exponent": _component_guide_exponent(0.0),
                "growth_power": -_component_guide_exponent(0.0),
                "window": FULL_LOSS_CONTROL_GUIDE_WINDOW,
                "included_in_phase_gate": False,
                "reason": (
                    "formal pre-feedback growth reference only; no stable "
                    "full-loss exponent is claimed below the destroy boundary"
                ),
            },
            "destroy_boundary_guide": {
                "theta": 0.25,
                "theory_phase_exponent": _q_phase_theory(0.25),
                "reason": "q=0 applies at the stability boundary, not below it",
            },
            "shared_preserve_guide": {
                "theta_values": [1.50, 2.50],
                "theory_phase_exponent": _q_phase_theory(1.50),
                "reason": "both sentinels have the same joint-limit phase exponent",
            },
            "metrics": full_loss_metrics,
            "claim_limit": (
                "direct full-loss slopes do not replace the normalized "
                "component/phase-envelope observable gate"
            ),
        },
        "quadrature": {
            route: {
                "mode_count": quadratures[route].mode_count,
                "spectral_relative_bin_width": route_specs[route][0],
                "time_points_per_decade": route_specs[route][1],
                "time_grid_size": int(grids[route].size),
                "approximation_floor": quadratures[route].approximation_floor,
                "teacher_energy_relative_error": abs(
                    float(np.sum(quadratures[route].initial_clean_mass))
                    + quadratures[route].approximation_floor
                    - quadratures[route].total_teacher_energy
                )
                / quadratures[route].total_teacher_energy,
            }
            for route in route_specs
        },
        "stability": {
            "all_trajectories_pass": stability_pass,
            "maximum_row_mass": max(
                trajectories[route][theta].maximum_row_mass
                for route in trajectories
                for theta in THETAS
            ),
            "maximum_cell_coupling": max(
                trajectories[route][theta].maximum_cell_coupling
                for route in trajectories
                for theta in THETAS
            ),
            "row_mass_cap": ROW_MASS_CAP,
            "maximum_cell_mass_conservation_error": max(cell_mass_errors.values()),
            "cell_mass_conservation_tolerance": (
                MAXIMUM_CELL_MASS_CONSERVATION_ERROR
            ),
            "cell_mass_errors": cell_mass_errors,
            "display_only": {
                "evaluated": bool(display_only_thetas),
                "included_in_numerical_contract": False,
                "all_trajectories_pass": display_stability_pass,
                "maximum_row_mass": (
                    max(
                        trajectories[route][theta].maximum_row_mass
                        for route in trajectories
                        for theta in display_only_thetas
                    )
                    if display_only_thetas
                    else None
                ),
                "maximum_cell_coupling": (
                    max(
                        trajectories[route][theta].maximum_cell_coupling
                        for route in trajectories
                        for theta in display_only_thetas
                    )
                    if display_only_thetas
                    else None
                ),
                "cell_mass_errors": display_only_cell_mass_errors,
            },
        },
        "resolution": {
            "all_checks_pass": resolution_pass,
            "maximum_curve_error": maximum_curve_error,
            "maximum_slope_error": maximum_slope_error,
            "curve_tolerance": MAXIMUM_RESOLUTION_CURVE_ERROR,
            "slope_tolerance": MAXIMUM_RESOLUTION_SLOPE_ERROR,
            "display_only": {
                "evaluated": bool(display_only_thetas),
                "included_in_numerical_contract": False,
                "all_checks_pass": display_resolution_pass,
                "maximum_curve_error": (
                    max(
                        row["maximum_relative_curve_difference"]
                        for row in display_resolution_rows
                    )
                    if display_resolution_rows
                    else None
                ),
                "maximum_slope_error": (
                    max(
                        row["absolute_slope_difference"]
                        for row in display_resolution_rows
                    )
                    if display_resolution_rows
                    else None
                ),
            },
        },
        "display_only_numerical_pass": (
            bool(
                display_resolution_pass
                and display_stability_pass
                and max(display_only_cell_mass_errors.values())
                <= MAXIMUM_CELL_MASS_CONSERVATION_ERROR
            )
            if display_only_thetas
            else None
        ),
        "numerical_contract_pass": bool(
            stability_pass
            and resolution_pass
            and max(cell_mass_errors.values())
            <= MAXIMUM_CELL_MASS_CONSERVATION_ERROR
        ),
        "primary_phase_gate_pass": primary_phase_pass,
        "sentinel_phase_gate_pass": sentinel_phase_pass,
        "overall_observable_gate_pass": bool(
            primary_phase_pass
            and sentinel_phase_pass
            and abs(refined["q_clean"] - _q_clean_theory())
            <= MAXIMUM_EXPONENT_ERROR
            and refined["clean_local_variation"]
            <= MAXIMUM_LOCAL_VARIATION
            and refined["primary_phase_mae"] <= 0.025
        ),
        "elapsed_seconds": time.perf_counter() - started,
        "artifacts": {
            "curves": str(output_dir / "curves.csv"),
            "panel_a_display_curves": str(
                output_dir / "panel_a_display_curves.csv"
            ),
            "phase_metrics": str(output_dir / "phase_metrics.csv"),
            "resolution_checks": str(output_dir / "resolution_checks.csv"),
            "figure_png": str(figure_png),
            "figure_pdf": str(figure_pdf),
            "full_loss_display_curves": str(
                output_dir / "full_loss_display_curves.csv"
            ),
            "full_loss_fit_metrics": str(
                output_dir / "full_loss_fit_metrics.csv"
            ),
            "full_loss_figure_png": str(full_loss_figure_png),
            "full_loss_figure_pdf": str(full_loss_figure_pdf),
        },
    }
    if make_compact_triptych:
        if compact_figure_png is None or compact_figure_pdf is None:
            raise RuntimeError("compact paper figure paths were not created")
        summary["artifacts"].update(
            {
                "compact_paper_figure_png": str(compact_figure_png),
                "compact_paper_figure_pdf": str(compact_figure_pdf),
                "compact_paper_figure_curves": str(
                    output_dir / "joint_schedule_triptych_curves.csv"
                ),
                "compact_paper_figure_metrics": str(
                    output_dir / "joint_schedule_triptych_metrics.csv"
                ),
            }
        )
    _write_json(output_dir / "summary.json", summary)
    return summary


def main() -> None:
    summary = run_experiment()
    print(
        json.dumps(
            {
                "status": summary["status"],
                "numerical_contract_pass": summary["numerical_contract_pass"],
                "primary_phase_gate_pass": summary["primary_phase_gate_pass"],
                "sentinel_phase_gate_pass": summary["sentinel_phase_gate_pass"],
                "overall_observable_gate_pass": summary[
                    "overall_observable_gate_pass"
                ],
                "stress_test_theta_0.90": summary["stress_test_theta_0.90"],
                "elapsed_seconds": summary["elapsed_seconds"],
                "artifacts": summary["artifacts"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
