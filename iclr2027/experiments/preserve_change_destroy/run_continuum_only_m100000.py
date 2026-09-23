"""Run experiment one at effective width 100,000 using continuum only.

This deterministic run keeps the previously fixed analytic spectrum, integer
batch schedules, intrinsic-time grid, noise scale, and component-normalized
observable.  It intentionally runs neither the discrete exact recursion nor
true SGD.  A doubled spectral resolution is used as the numerical control.
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
    run_scheduled_continuum_modal_dynamics,
    spectral_profiles,
)
from .dynamics import batch_schedule
from .slopes import local_log_slopes


ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "artifacts" / "continuum_only_m100000"

WIDTH = 100_000
ALPHA = 0.4
BETA = 0.3
ETA = 1.0 / 8.0
INITIAL_BATCH = 16
SIGMA2 = 10.0
THETAS = (0.25, 0.35, 0.50, 0.65, 0.75, 0.90, 1.00, 1.10)
REPRESENTATIVES = (0.25, 0.50, 0.90)
HORIZON_EXPONENT = 0.50
FIT_LOWER_EXPONENT = 0.25
FIT_UPPER_EXPONENT = 0.50
FIT_POINTS = 41
LOCAL_HALF_WINDOW_DECADES = 0.15
MAXIMUM_EXPONENT_ERROR = 0.05
MAXIMUM_LOCAL_VARIATION = 0.15
ROW_MASS_CAP = 0.20
BASE_RELATIVE_BIN_WIDTH = 2.0e-3
REFINED_RELATIVE_BIN_WIDTH = 1.0e-3
MAXIMUM_RESOLUTION_CURVE_ERROR = 3.0e-3
MAXIMUM_RESOLUTION_SLOPE_ERROR = 2.0e-3


def _q_clean() -> float:
    return (2.0 * ALPHA + 2.0 * BETA - 1.0) / (2.0 * ALPHA)


def _q_kernel() -> float:
    return 2.0 - 1.0 / (2.0 * ALPHA)


def _q_noise(theta: float) -> float:
    boundary = 1.0 - _q_kernel()
    if theta <= boundary:
        return 0.0
    if theta < 1.0:
        return theta + _q_kernel() - 1.0
    return _q_kernel()


def _q_phase(theta: float) -> float:
    return min(_q_clean(), _q_noise(theta))


def _phase(theta: float) -> str:
    if theta <= 1.0 - _q_kernel():
        return "destroy"
    if _q_noise(theta) < _q_clean() - 1.0e-12:
        return "change"
    if abs(_q_noise(theta) - _q_clean()) <= 1.0e-12:
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


def _run_resolution(
    relative_bin_width: float,
) -> tuple[DEQuadrature, dict[float, DETrajectory], dict[float, np.ndarray]]:
    quadrature = head_preserving_quadrature(
        alpha=ALPHA,
        beta=BETA,
        effective_width=WIDTH,
        relative_bin_width=relative_bin_width,
    )
    maximum_time = float(WIDTH) ** HORIZON_EXPONENT
    horizon = max(1, int(math.ceil(maximum_time / ETA)))
    trajectories: dict[float, DETrajectory] = {}
    schedules: dict[float, np.ndarray] = {}
    for theta in THETAS:
        times, batches = batch_schedule(ETA, theta, INITIAL_BATCH, horizon)
        trajectories[theta] = run_scheduled_continuum_modal_dynamics(
            quadrature=quadrature,
            times=times,
            batches=batches,
            eta=ETA,
            theta=theta,
            sigma2=SIGMA2,
            row_mass_cap=ROW_MASS_CAP,
        )
        schedules[theta] = batches
    return quadrature, trajectories, schedules


def _process(
    quadrature: DEQuadrature,
    trajectories: dict[float, DETrajectory],
    target_times: np.ndarray,
) -> dict[str, Any]:
    clean_by_theta: dict[float, np.ndarray] = {}
    gap_by_theta: dict[float, np.ndarray] = {}
    total_by_theta: dict[float, np.ndarray] = {}
    for theta in THETAS:
        trajectory = trajectories[theta]
        tail_completed_clean = (
            quadrature.approximation_floor + trajectory.clean_centered
        )
        clean_by_theta[theta] = _positive_log_interpolate(
            target_times, trajectory.times, tail_completed_clean
        )
        gap_by_theta[theta] = _positive_log_interpolate(
            target_times, trajectory.times, trajectory.noise_gap
        )
        total_by_theta[theta] = _positive_log_interpolate(
            target_times,
            trajectory.times,
            tail_completed_clean + trajectory.noise_gap,
        )

    clean_stack = np.stack([clean_by_theta[theta] for theta in THETAS])
    pooled_clean = np.exp(np.mean(np.log(clean_stack), axis=0))
    normalized_clean = pooled_clean / pooled_clean[0]
    q_clean_measured = _exponent(target_times, pooled_clean)
    clean_slopes, clean_local_median, clean_local_variation = _local_statistics(
        target_times, pooled_clean
    )

    reference = gap_by_theta[THETAS[0]]
    contrasts: dict[float, np.ndarray] = {}
    contrast_slopes: dict[float, np.ndarray] = {}
    envelopes: dict[float, np.ndarray] = {}
    envelope_slopes: dict[float, np.ndarray] = {}
    metrics: dict[float, dict[str, Any]] = {}
    for theta in THETAS:
        contrast = gap_by_theta[theta] / reference
        contrast = contrast / contrast[0]
        contrasts[theta] = contrast
        q_contrast = _exponent(target_times, contrast)
        local, local_median, local_variation = _local_statistics(
            target_times, contrast
        )
        contrast_slopes[theta] = local
        # The normalized phase-envelope diagnostic is the slower of the
        # tail-completed clean response and the boundary-normalized gap
        # response.  Its exponent is therefore the finite-window realization
        # of min(q_clean, q_gap-relative); fitting the actual curve also makes
        # the local-slope gate refer to the observable shown in the phase plot.
        envelope = np.maximum(normalized_clean, contrast)
        envelopes[theta] = envelope
        q_envelope = _exponent(target_times, envelope)
        envelope_local, envelope_local_median, envelope_local_variation = (
            _local_statistics(target_times, envelope)
        )
        envelope_slopes[theta] = envelope_local
        predicted = _q_phase(theta)
        metrics[theta] = {
            "theta": theta,
            "phase": _phase(theta),
            "predicted_noise_exponent": _q_noise(theta),
            "predicted_phase_exponent": predicted,
            "measured_tail_completed_clean_exponent": q_clean_measured,
            "measured_absolute_gap_exponent": _exponent(
                target_times, gap_by_theta[theta]
            ),
            "measured_relative_gap_exponent": q_contrast,
            "measured_phase_envelope": q_envelope,
            "absolute_phase_error": abs(q_envelope - predicted),
            "relative_gap_local_median": local_median,
            "relative_gap_local_variation": local_variation,
            "phase_envelope_local_median": envelope_local_median,
            "phase_envelope_local_variation": envelope_local_variation,
            "raw_total_exponent": _exponent(
                target_times, total_by_theta[theta]
            ),
            "exponent_gate": abs(q_envelope - predicted)
            <= MAXIMUM_EXPONENT_ERROR,
            "local_gate": envelope_local_variation
            <= MAXIMUM_LOCAL_VARIATION,
        }

    phase_errors = np.asarray(
        [metrics[theta]["absolute_phase_error"] for theta in THETAS],
        dtype=float,
    )
    return {
        "target_times": target_times,
        "clean_by_theta": clean_by_theta,
        "gap_by_theta": gap_by_theta,
        "total_by_theta": total_by_theta,
        "pooled_clean": pooled_clean,
        "normalized_clean": normalized_clean,
        "clean_slopes": clean_slopes,
        "clean_local_median": clean_local_median,
        "clean_local_variation": clean_local_variation,
        "contrasts": contrasts,
        "contrast_slopes": contrast_slopes,
        "envelopes": envelopes,
        "envelope_slopes": envelope_slopes,
        "metrics": metrics,
        "q_clean": q_clean_measured,
        "phase_mae": float(np.mean(phase_errors)),
        "phase_max_error": float(np.max(phase_errors)),
        "exponent_pass_count": int(
            sum(bool(metrics[theta]["exponent_gate"]) for theta in THETAS)
        ),
        "local_pass_count": int(
            sum(bool(metrics[theta]["local_gate"]) for theta in THETAS)
        ),
    }


def _resolution_rows(
    base: dict[str, Any],
    refined: dict[str, Any],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    def compare(
        observable: str,
        theta: float | None,
        base_values: np.ndarray,
        refined_values: np.ndarray,
    ) -> None:
        relative = np.abs(base_values - refined_values) / np.maximum(
            np.abs(refined_values), 1.0e-300
        )
        base_slope = _exponent(base["target_times"], base_values)
        refined_slope = _exponent(refined["target_times"], refined_values)
        curve_error = float(np.max(relative))
        slope_error = abs(base_slope - refined_slope)
        rows.append(
            {
                "observable": observable,
                "theta": "" if theta is None else theta,
                "maximum_relative_curve_difference": curve_error,
                "base_exponent": base_slope,
                "refined_exponent": refined_slope,
                "absolute_slope_difference": slope_error,
                "status": (
                    "PASS"
                    if curve_error <= MAXIMUM_RESOLUTION_CURVE_ERROR
                    and slope_error <= MAXIMUM_RESOLUTION_SLOPE_ERROR
                    else "FAIL"
                ),
            }
        )

    compare(
        "pooled_tail_completed_clean",
        None,
        base["pooled_clean"],
        refined["pooled_clean"],
    )
    for theta in THETAS:
        compare(
            "tail_completed_clean",
            theta,
            base["clean_by_theta"][theta],
            refined["clean_by_theta"][theta],
        )
        compare(
            "absolute_gap",
            theta,
            base["gap_by_theta"][theta],
            refined["gap_by_theta"][theta],
        )
        compare(
            "raw_total",
            theta,
            base["total_by_theta"][theta],
            refined["total_by_theta"][theta],
        )
        compare(
            "relative_gap_contrast",
            theta,
            base["contrasts"][theta],
            refined["contrasts"][theta],
        )
        compare(
            "phase_envelope",
            theta,
            base["envelopes"][theta],
            refined["envelopes"][theta],
        )
    return rows


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
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


def _make_figure(
    base: dict[str, Any],
    refined: dict[str, Any],
    output_path: Path,
) -> tuple[Path, Path]:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(output_path.parent / ".matplotlib"))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    colors = {
        "clean": "#b43f8f",
        0.25: "#2878b5",
        0.50: "#d18700",
        0.90: "#3a9856",
    }
    times = base["target_times"]
    figure, axes = plt.subplots(1, 3, figsize=(14.2, 4.0))

    axis = axes[0]
    clean_base = base["normalized_clean"]
    clean_refined = refined["normalized_clean"]
    axis.loglog(times, clean_refined, color=colors["clean"], linewidth=2.2,
                label="tail-completed clean")
    axis.loglog(times, clean_base, color=colors["clean"], linewidth=1.2,
                linestyle="--")
    axis.loglog(times, (times / times[0]) ** (-_q_clean()),
                color=colors["clean"], linewidth=1.0, linestyle=":")
    for theta in REPRESENTATIVES:
        axis.loglog(times, refined["contrasts"][theta], color=colors[theta],
                    linewidth=2.2, label=rf"$\vartheta={theta:g}$")
        axis.loglog(times, base["contrasts"][theta], color=colors[theta],
                    linewidth=1.2, linestyle="--")
        axis.loglog(times, (times / times[0]) ** (-_q_noise(theta)),
                    color=colors[theta], linewidth=1.0, linestyle=":")
    axis.set_xlabel(r"intrinsic time $T$")
    axis.set_ylabel("normalized response")
    axis.set_title("(a) Component-normalized continuum")
    axis.grid(True, which="both", alpha=0.17)
    theta_legend = axis.legend(frameon=False, fontsize=8, loc="lower left")
    axis.add_artist(theta_legend)
    axis.legend(
        handles=[
            Line2D([0], [0], color="black", linewidth=2.2, label="refined quadrature"),
            Line2D([0], [0], color="black", linewidth=1.2,
                   linestyle="--", label="base quadrature"),
            Line2D([0], [0], color="black", linewidth=1.0,
                   linestyle=":", label="theory slope"),
        ],
        frameon=False,
        fontsize=7,
        loc="upper right",
    )

    axis = axes[1]
    theta_grid = np.linspace(min(THETAS), max(THETAS), 400)
    axis.plot(theta_grid, [_q_phase(value) for value in theta_grid],
              color="black", linewidth=2.0, label="theory envelope")
    axis.scatter(
        THETAS,
        [refined["metrics"][theta]["measured_phase_envelope"] for theta in THETAS],
        marker="o", s=42, color="#2f4858", label="refined continuum", zorder=4,
    )
    axis.scatter(
        THETAS,
        [base["metrics"][theta]["measured_phase_envelope"] for theta in THETAS],
        marker="^", s=44, facecolors="white", edgecolors="#7b5cd6",
        linewidths=1.2, label="base continuum", zorder=4,
    )
    axis.scatter(
        THETAS,
        [refined["metrics"][theta]["measured_relative_gap_exponent"] for theta in THETAS],
        marker="o", s=28, facecolors="white", edgecolors="#d18700",
        linewidths=1.0, label="relative gap", zorder=3,
    )
    axis.axhline(refined["q_clean"], color=colors["clean"], linestyle="--",
                 linewidth=1.2, label="clean reference")
    axis.axvline(0.25, color="#777777", linestyle=":", linewidth=1.0)
    axis.axvline(0.75, color="#777777", linestyle=":", linewidth=1.0)
    axis.text(0.25, 0.02, r"theory $\vartheta_d$", rotation=90,
              va="bottom", ha="right", fontsize=7, color="#666666")
    axis.text(0.75, 0.02, r"theory $\vartheta_p$", rotation=90,
              va="bottom", ha="right", fontsize=7, color="#666666")
    axis.set_xlabel(r"schedule exponent $\vartheta$")
    axis.set_ylabel("effective exponent")
    axis.set_title("(b) Reconstructed phase envelope")
    axis.set_ylim(-0.03, 0.82)
    axis.grid(True, alpha=0.17)
    axis.legend(frameon=False, fontsize=7, loc="upper left")

    axis = axes[2]
    axis.semilogx(times, refined["clean_slopes"], color=colors["clean"],
                  linewidth=1.2, linestyle="--", label="clean component")
    for theta in REPRESENTATIVES:
        axis.semilogx(times, refined["envelope_slopes"][theta], color=colors[theta],
                      linewidth=2.0, label=rf"$\vartheta={theta:g}$")
        axis.axhline(_q_phase(theta), color=colors[theta], linestyle=":",
                     linewidth=1.0)
    axis.set_xlabel(r"intrinsic time $T$")
    axis.set_ylabel("local effective exponent")
    axis.set_title("(c) Phase-envelope local slopes")
    axis.grid(True, which="both", alpha=0.17)
    axis.legend(frameon=False, fontsize=8)

    figure.suptitle(
        r"Continuum-only experiment one: $m=10^5$, $\sigma^2=10$, "
        r"$T\in[m^{1/4},m^{1/2}]$",
        fontsize=11,
        y=1.02,
    )
    figure.tight_layout()
    png = output_path.with_suffix(".png")
    pdf = output_path.with_suffix(".pdf")
    figure.savefig(png, dpi=220, bbox_inches="tight")
    figure.savefig(pdf, bbox_inches="tight")
    plt.close(figure)
    return png, pdf


def run_experiment(output_dir: Path = OUTPUT_DIR) -> dict[str, Any]:
    started = time.perf_counter()
    output_dir.mkdir(parents=True, exist_ok=True)
    lower = float(WIDTH) ** FIT_LOWER_EXPONENT
    upper = float(WIDTH) ** FIT_UPPER_EXPONENT
    target_times = np.geomspace(lower, upper, FIT_POINTS)

    base_quadrature, base_trajectories, base_schedules = _run_resolution(
        BASE_RELATIVE_BIN_WIDTH
    )
    refined_quadrature, refined_trajectories, _ = _run_resolution(
        REFINED_RELATIVE_BIN_WIDTH
    )
    base = _process(base_quadrature, base_trajectories, target_times)
    refined = _process(refined_quadrature, refined_trajectories, target_times)
    resolution_rows = _resolution_rows(base, refined)

    base_kernel, base_forcing = spectral_profiles(base_quadrature, target_times)
    refined_kernel, refined_forcing = spectral_profiles(
        refined_quadrature, target_times
    )
    kernel_resolution = float(
        np.max(
            np.abs(base_kernel - refined_kernel)
            / np.maximum(np.abs(refined_kernel), 1.0e-300)
        )
    )
    forcing_resolution = float(
        np.max(
            np.abs(base_forcing - refined_forcing)
            / np.maximum(np.abs(refined_forcing), 1.0e-300)
        )
    )

    curve_rows: list[dict[str, Any]] = []
    phase_rows: list[dict[str, Any]] = []
    for route, processed in (("base", base), ("refined", refined)):
        for theta in THETAS:
            metric = processed["metrics"][theta]
            phase_rows.append({"route": route, **metric})
            for index, intrinsic_time in enumerate(target_times):
                curve_rows.append(
                    {
                        "route": route,
                        "effective_width": WIDTH,
                        "theta": theta,
                        "intrinsic_time": intrinsic_time,
                        "tail_completed_clean": processed["clean_by_theta"][theta][index],
                        "pooled_tail_completed_clean": processed["pooled_clean"][index],
                        "absolute_gap": processed["gap_by_theta"][theta][index],
                        "relative_gap_contrast": processed["contrasts"][theta][index],
                        "phase_envelope": processed["envelopes"][theta][index],
                        "raw_total": processed["total_by_theta"][theta][index],
                        "relative_gap_local_exponent": processed["contrast_slopes"][theta][index],
                        "phase_envelope_local_exponent": processed["envelope_slopes"][theta][index],
                        "pooled_clean_local_exponent": processed["clean_slopes"][index],
                    }
                )

    _write_csv(output_dir / "curves.csv", curve_rows)
    _write_csv(output_dir / "phase_metrics.csv", phase_rows)
    _write_csv(output_dir / "resolution_checks.csv", resolution_rows)
    figure_png, figure_pdf = _make_figure(
        base, refined, output_dir / "continuum_only_m100000"
    )

    maximum_resolution_curve_error = max(
        row["maximum_relative_curve_difference"] for row in resolution_rows
    )
    maximum_resolution_slope_error = max(
        row["absolute_slope_difference"] for row in resolution_rows
    )
    resolution_pass = all(row["status"] == "PASS" for row in resolution_rows)
    all_stable = all(
        trajectory.stable
        for trajectory in [
            *base_trajectories.values(),
            *refined_trajectories.values(),
        ]
    )
    summary = {
        "experiment": "continuum_only_experiment_one_m100000",
        "status": "PILOT_UNPROMOTED",
        "object_scope": (
            "head-preserving analytic-spectrum scheduled-continuum proxy at "
            "effective width m=100000"
        ),
        "not_run": ["discrete exact recursion", "true Gaussian SGD"],
        "not_claimed": [
            "fixed-width asymptotic power law",
            "finite-random-feature deterministic equivalence",
            "population-to-empirical random-feature spectral bridge",
            "raw-total phase-law validation",
            "paper-ready evidence before computation-gate review",
        ],
        "protocol": {
            "effective_width": WIDTH,
            "alpha": ALPHA,
            "beta": BETA,
            "eta": ETA,
            "initial_batch": INITIAL_BATCH,
            "batch_schedule": "B_t=ceil(16*(1+T_t)^theta)",
            "sigma2": SIGMA2,
            "theta_values": THETAS,
            "horizon_exponent": HORIZON_EXPONENT,
            "maximum_intrinsic_time": float(
                next(iter(base_trajectories.values())).times[-1]
            ),
            "fit_window": [lower, upper],
            "fit_window_width_exponents": [
                FIT_LOWER_EXPONENT,
                FIT_UPPER_EXPONENT,
            ],
            "fit_points": FIT_POINTS,
            "fit_weighting": "41 equal-weight log-uniform points",
            "local_slope_half_window_decades": LOCAL_HALF_WINDOW_DECADES,
            "maximum_exponent_error": MAXIMUM_EXPONENT_ERROR,
            "maximum_local_variation": MAXIMUM_LOCAL_VARIATION,
            "destroy_reference_theta": THETAS[0],
        },
        "theory": {
            "q_clean": _q_clean(),
            "q_kernel": _q_kernel(),
            "destroy_boundary": 1.0 - _q_kernel(),
            "preservation_boundary": 1.0 - _q_kernel() + _q_clean(),
            "phase_exponents": {theta: _q_phase(theta) for theta in THETAS},
        },
        "quadrature": {
            "base_relative_bin_width": BASE_RELATIVE_BIN_WIDTH,
            "base_mode_count": base_quadrature.mode_count,
            "refined_relative_bin_width": REFINED_RELATIVE_BIN_WIDTH,
            "refined_mode_count": refined_quadrature.mode_count,
            "approximation_floor": base_quadrature.approximation_floor,
            "base_teacher_energy_relative_error": abs(
                float(np.sum(base_quadrature.initial_clean_mass))
                + base_quadrature.approximation_floor
                - base_quadrature.total_teacher_energy
            )
            / base_quadrature.total_teacher_energy,
            "refined_teacher_energy_relative_error": abs(
                float(np.sum(refined_quadrature.initial_clean_mass))
                + refined_quadrature.approximation_floor
                - refined_quadrature.total_teacher_energy
            )
            / refined_quadrature.total_teacher_energy,
        },
        "base_results": {
            "tail_completed_clean_exponent": base["q_clean"],
            "tail_completed_clean_local_variation": base[
                "clean_local_variation"
            ],
            "phase_mae": base["phase_mae"],
            "phase_max_error": base["phase_max_error"],
            "exponent_pass_count": base["exponent_pass_count"],
            "local_pass_count": base["local_pass_count"],
            "metrics": base["metrics"],
        },
        "refined_results": {
            "tail_completed_clean_exponent": refined["q_clean"],
            "tail_completed_clean_local_variation": refined[
                "clean_local_variation"
            ],
            "phase_mae": refined["phase_mae"],
            "phase_max_error": refined["phase_max_error"],
            "exponent_pass_count": refined["exponent_pass_count"],
            "local_pass_count": refined["local_pass_count"],
            "metrics": refined["metrics"],
        },
        "spectral_window_diagnostics": {
            "base_bare_kernel_exponent": _exponent(target_times, base_kernel),
            "refined_bare_kernel_exponent": _exponent(
                target_times, refined_kernel
            ),
            "predicted_bare_kernel_exponent": _q_kernel(),
            "base_clean_forcing_exponent": _exponent(
                target_times, base_forcing
            ),
            "refined_clean_forcing_exponent": _exponent(
                target_times, refined_forcing
            ),
            "predicted_clean_forcing_exponent": _q_clean(),
            "maximum_kernel_resolution_error": kernel_resolution,
            "maximum_forcing_resolution_error": forcing_resolution,
        },
        "stability": {
            "all_trajectories_pass": all_stable,
            "maximum_base_row_mass": max(
                trajectory.maximum_row_mass
                for trajectory in base_trajectories.values()
            ),
            "maximum_refined_row_mass": max(
                trajectory.maximum_row_mass
                for trajectory in refined_trajectories.values()
            ),
            "row_mass_cap": ROW_MASS_CAP,
            "maximum_base_cell_coupling": max(
                trajectory.maximum_cell_coupling
                for trajectory in base_trajectories.values()
            ),
        },
        "resolution": {
            "all_checks_pass": resolution_pass,
            "maximum_curve_error": maximum_resolution_curve_error,
            "maximum_slope_error": maximum_resolution_slope_error,
            "curve_tolerance": MAXIMUM_RESOLUTION_CURVE_ERROR,
            "slope_tolerance": MAXIMUM_RESOLUTION_SLOPE_ERROR,
        },
        "numerical_contract_pass": bool(all_stable and resolution_pass),
        "phase_envelope_gate_pass": bool(
            base["exponent_pass_count"] == len(THETAS)
            and base["local_pass_count"] == len(THETAS)
            and refined["exponent_pass_count"] == len(THETAS)
            and refined["local_pass_count"] == len(THETAS)
        ),
        "final_batches": {
            theta: int(base_schedules[theta][-1]) for theta in THETAS
        },
        "elapsed_seconds": time.perf_counter() - started,
        "artifacts": {
            "curves": str(output_dir / "curves.csv"),
            "phase_metrics": str(output_dir / "phase_metrics.csv"),
            "resolution_checks": str(output_dir / "resolution_checks.csv"),
            "figure_png": str(figure_png),
            "figure_pdf": str(figure_pdf),
        },
    }
    _write_json(output_dir / "summary.json", summary)
    return summary


def main() -> None:
    summary = run_experiment()
    print(
        json.dumps(
            {
                "status": summary["status"],
                "numerical_contract_pass": summary["numerical_contract_pass"],
                "phase_envelope_gate_pass": summary[
                    "phase_envelope_gate_pass"
                ],
                "phase_max_error": summary["base_results"][
                    "phase_max_error"
                ],
                "elapsed_seconds": summary["elapsed_seconds"],
                "artifacts": summary["artifacts"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
