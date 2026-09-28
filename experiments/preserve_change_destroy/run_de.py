"""Run the scalable deterministic-equivalent/spectral-quadrature backend."""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
import json
import math
from pathlib import Path
import time
from typing import Any

import numpy as np

from .de_config import DEProfile, DE_PROFILES
from .de_plotting import make_de_figure, math_power
from .de_quadrature import (
    DEQuadrature,
    DETrajectory,
    head_preserving_quadrature,
    run_continuum_modal_dynamics,
    spectral_profiles,
    triangular_time_grid,
)
from .slopes import audit_fixed_window, local_log_slopes


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
        json.dumps(_finite(payload), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


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


def _run_width(
    profile: DEProfile,
    width: int,
    *,
    relative_bin_width: float | None = None,
    points_per_decade: int | None = None,
) -> tuple[DEQuadrature, dict[float, DETrajectory], np.ndarray, np.ndarray]:
    quadrature = head_preserving_quadrature(
        profile.alpha,
        profile.beta,
        width,
        profile.spectral_relative_bin_width
        if relative_bin_width is None
        else relative_bin_width,
    )
    times = triangular_time_grid(
        width,
        profile.horizon_exponent,
        profile.time_min,
        profile.time_points_per_decade
        if points_per_decade is None
        else points_per_decade,
    )
    trajectories = run_continuum_modal_dynamics(
        quadrature,
        times,
        profile.theta_values,
        profile.sigma2,
        profile.inverse_ratio_amplitude,
        profile.row_mass_cap,
    )
    kernel, forcing = spectral_profiles(quadrature, times)
    return quadrature, trajectories, kernel, forcing


def _fit_rows(
    profile: DEProfile,
    trajectories: dict[tuple[int, float], DETrajectory],
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    observables = {
        "clean_centered": ("clean_centered", lambda theta: profile.q_clean),
        "noise_gap": ("noise_gap", profile.noise_exponent),
        "noisy_centered": ("noisy_centered", profile.total_exponent),
    }
    for width in profile.effective_widths:
        for theta in profile.theta_values:
            trajectory = trajectories[(width, theta)]
            for window_name, lower_power, upper_power in profile.fit_windows:
                lower = math_power(width, lower_power)
                upper = math_power(width, upper_power)
                for observable, (attribute, theory_function) in observables.items():
                    audit = audit_fixed_window(
                        trajectory.times,
                        getattr(trajectory, attribute),
                        observable=observable,
                        lower_time=lower,
                        upper_time=upper,
                        minimum_decades=profile.minimum_fit_decades,
                        minimum_points=profile.minimum_fit_points,
                        half_window_decades=profile.local_slope_half_window_decades,
                        maximum_local_variation=profile.maximum_local_slope_variation,
                    )
                    predicted = float(theory_function(theta))
                    row = audit.to_dict()
                    error = (
                        abs(audit.exponent - predicted)
                        if math.isfinite(audit.exponent)
                        else math.inf
                    )
                    row.update(
                        {
                            "effective_width": width,
                            "theta": theta,
                            "phase": profile.phase_label(theta),
                            "window": window_name,
                            "lower_width_exponent": lower_power,
                            "upper_width_exponent": upper_power,
                            "nominal_window": window_name == profile.nominal_window_name,
                            "predicted_exponent": predicted,
                            "absolute_exponent_error": error,
                            "audit_status": audit.status,
                            "theory_status": (
                                "PASS"
                                if audit.status == "PASS"
                                and error <= profile.maximum_exponent_error
                                else "INCONCLUSIVE"
                            ),
                        }
                    )
                    output.append(row)
    return output


def _positive_log_interpolate(
    target_times: np.ndarray,
    source_times: np.ndarray,
    source_values: np.ndarray,
) -> np.ndarray:
    if np.any(target_times <= 0.0) or np.any(source_times <= 0.0):
        raise ValueError("log interpolation requires positive times")
    if np.any(source_values <= 0.0):
        raise ValueError("log interpolation requires positive values")
    return np.exp(
        np.interp(
            np.log(target_times),
            np.log(source_times),
            np.log(source_values),
        )
    )


def _resolution_checks(
    profile: DEProfile,
    base_trajectories: dict[tuple[int, float], DETrajectory],
    base_kernel: dict[int, np.ndarray],
    base_forcing: dict[int, np.ndarray],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    width = max(profile.effective_widths)
    refined_quadrature, refined, refined_kernel, refined_forcing = _run_width(
        profile,
        width,
        relative_bin_width=0.5 * profile.spectral_relative_bin_width,
        points_per_decade=2 * profile.time_points_per_decade,
    )
    base_time = base_trajectories[(width, profile.theta_values[0])].times
    refined_time = refined[profile.theta_values[0]].times
    lower_power, upper_power = profile.fit_window(profile.nominal_window_name)
    lower = math_power(width, lower_power)
    upper = math_power(width, upper_power)
    base_mask = (base_time >= lower) & (base_time <= upper)
    rows: list[dict[str, Any]] = []

    def compare(
        label: str,
        theta: float | None,
        base_values: np.ndarray,
        refined_values: np.ndarray,
        predicted: float,
    ) -> None:
        target = base_time[base_mask]
        interpolated = _positive_log_interpolate(
            target,
            refined_time[1:],
            refined_values[1:],
        )
        baseline = base_values[base_mask]
        relative = np.abs(baseline - interpolated) / np.maximum(
            np.abs(interpolated), 1.0e-300
        )
        base_audit = audit_fixed_window(
            base_time,
            base_values,
            label,
            lower,
            upper,
            profile.minimum_fit_decades,
            profile.minimum_fit_points,
            profile.local_slope_half_window_decades,
            profile.maximum_local_slope_variation,
        )
        refined_audit = audit_fixed_window(
            refined_time,
            refined_values,
            label,
            lower,
            upper,
            profile.minimum_fit_decades,
            profile.minimum_fit_points,
            profile.local_slope_half_window_decades,
            profile.maximum_local_slope_variation,
        )
        slope_difference = abs(base_audit.exponent - refined_audit.exponent)
        rows.append(
            {
                "observable": label,
                "theta": theta,
                "effective_width": width,
                "predicted_exponent": predicted,
                "base_exponent": base_audit.exponent,
                "refined_exponent": refined_audit.exponent,
                "maximum_relative_curve_difference": float(np.max(relative)),
                "median_relative_curve_difference": float(np.median(relative)),
                "absolute_slope_difference": slope_difference,
                "status": (
                    "PASS"
                    if float(np.max(relative))
                    <= profile.maximum_resolution_curve_error
                    and slope_difference <= profile.maximum_resolution_slope_error
                    else "FAIL"
                ),
            }
        )

    for theta in profile.theta_values:
        base = base_trajectories[(width, theta)]
        candidate = refined[theta]
        compare(
            "clean_centered",
            theta,
            base.clean_centered,
            candidate.clean_centered,
            profile.q_clean,
        )
        compare(
            "noise_gap",
            theta,
            base.noise_gap,
            candidate.noise_gap,
            profile.noise_exponent(theta),
        )
        compare(
            "noisy_centered",
            theta,
            base.noisy_centered,
            candidate.noisy_centered,
            profile.total_exponent(theta),
        )
    compare(
        "bare_kernel",
        None,
        base_kernel[width],
        refined_kernel,
        profile.q_kernel,
    )
    compare(
        "clean_forcing",
        None,
        base_forcing[width],
        refined_forcing,
        profile.q_clean,
    )
    return rows, {
        "base_relative_bin_width": profile.spectral_relative_bin_width,
        "refined_relative_bin_width": 0.5 * profile.spectral_relative_bin_width,
        "base_points_per_decade": profile.time_points_per_decade,
        "refined_points_per_decade": 2 * profile.time_points_per_decade,
        "refined_mode_count": refined_quadrature.mode_count,
    }


def run_de_experiment(profile: DEProfile, output_dir: Path) -> dict[str, Any]:
    started = time.perf_counter()
    output_dir.mkdir(parents=True, exist_ok=True)
    quadratures: dict[int, DEQuadrature] = {}
    trajectories: dict[tuple[int, float], DETrajectory] = {}
    kernels: dict[int, np.ndarray] = {}
    forcings: dict[int, np.ndarray] = {}
    for width in profile.effective_widths:
        quadrature, width_trajectories, kernel, forcing = _run_width(profile, width)
        quadratures[width] = quadrature
        kernels[width] = kernel
        forcings[width] = forcing
        for theta, trajectory in width_trajectories.items():
            trajectories[(width, theta)] = trajectory

    fit_rows = _fit_rows(profile, trajectories)
    resolution_rows, resolution_design = _resolution_checks(
        profile, trajectories, kernels, forcings
    )
    resolution_pass = all(row["status"] == "PASS" for row in resolution_rows)
    all_stable = all(item.stable for item in trajectories.values())
    largest = max(profile.effective_widths)
    nominal = {
        theta: next(
            row
            for row in fit_rows
            if int(row["effective_width"]) == largest
            and float(row["theta"]) == theta
            and row["observable"] == "noisy_centered"
            and bool(row["nominal_window"])
        )
        for theta in profile.theta_values
    }
    representative_pass = all(
        nominal[theta]["theory_status"] == "PASS"
        for theta in profile.representative_thetas
    )

    curve_rows: list[dict[str, Any]] = []
    spectral_rows: list[dict[str, Any]] = []
    for width in profile.effective_widths:
        time_grid = trajectories[(width, profile.theta_values[0])].times
        kernel_slopes = local_log_slopes(
            time_grid, kernels[width], half_window_decades=0.25, minimum_points=7
        )
        forcing_slopes = local_log_slopes(
            time_grid, forcings[width], half_window_decades=0.25, minimum_points=7
        )
        for index, intrinsic_time in enumerate(time_grid):
            spectral_rows.append(
                {
                    "effective_width": width,
                    "intrinsic_time": intrinsic_time,
                    "bare_kernel": kernels[width][index],
                    "bare_kernel_local_exponent": kernel_slopes[index],
                    "clean_forcing": forcings[width][index],
                    "clean_forcing_local_exponent": forcing_slopes[index],
                }
            )
        for theta in profile.theta_values:
            trajectory = trajectories[(width, theta)]
            for index, intrinsic_time in enumerate(trajectory.times):
                curve_rows.append(
                    {
                        "profile": profile.name,
                        "effective_width": width,
                        "theta": theta,
                        "phase": profile.phase_label(theta),
                        "intrinsic_time": intrinsic_time,
                        "clean_centered": trajectory.clean_centered[index],
                        "noise_gap": trajectory.noise_gap[index],
                        "noisy_centered": trajectory.noisy_centered[index],
                        "row_mass": trajectory.row_mass[index],
                    }
                )

    _write_csv(output_dir / "curves.csv", curve_rows)
    _write_csv(output_dir / "spectral_profiles.csv", spectral_rows)
    _write_csv(output_dir / "slope_audits.csv", fit_rows)
    _write_csv(output_dir / "resolution_checks.csv", resolution_rows)
    figure_png, figure_pdf = make_de_figure(
        profile,
        {theta: trajectories[(largest, theta)] for theta in profile.theta_values},
        nominal,
        output_dir / "preserve_change_destroy_de",
    )

    quadrature_checks = []
    for width, quadrature in quadratures.items():
        reconstructed = (
            float(np.sum(quadrature.initial_clean_mass))
            + quadrature.approximation_floor
        )
        quadrature_checks.append(
            {
                "effective_width": width,
                "mode_count": quadrature.mode_count,
                "relative_bin_width": quadrature.relative_bin_width,
                "approximation_floor": quadrature.approximation_floor,
                "teacher_energy_relative_error": abs(
                    reconstructed - quadrature.total_teacher_energy
                )
                / quadrature.total_teacher_energy,
                "minimum_eigenvalue": float(np.min(quadrature.eigenvalues)),
            }
        )
    summary = {
        "experiment": "preserve_change_destroy_de",
        "profile": profile.name,
        "object_scope": "head-preserving PLRF continuum spectral proxy",
        "not_claimed": [
            "finite-random-feature deterministic-equivalent replacement",
            "contour-DE validation",
            "paper-ready evidence before the computation gate is reviewed",
        ],
        "profile_config": asdict(profile),
        "theory": {
            "q_clean": profile.q_clean,
            "q_kernel": profile.q_kernel,
            "destroy_boundary": profile.destroy_boundary,
            "preservation_boundary": profile.preservation_boundary,
        },
        "triangular_window": {
            "nominal": profile.fit_window(profile.nominal_window_name),
            "meaning": "m^lower <= T <= m^upper, so T grows and T/m^(2 alpha) vanishes",
        },
        "quadrature_checks": quadrature_checks,
        "stability": {
            "all_trajectories_pass": all_stable,
            "maximum_row_mass": max(item.maximum_row_mass for item in trajectories.values()),
            "maximum_cell_coupling": max(
                item.maximum_cell_coupling for item in trajectories.values()
            ),
        },
        "resolution": {
            **resolution_design,
            "all_pass": resolution_pass,
            "maximum_curve_error": max(
                row["maximum_relative_curve_difference"] for row in resolution_rows
            ),
            "maximum_slope_error": max(
                row["absolute_slope_difference"] for row in resolution_rows
            ),
        },
        "largest_width_nominal_fits": {
            str(theta): nominal[theta] for theta in profile.theta_values
        },
        "numerical_contract_pass": bool(all_stable and resolution_pass),
        "slope_claim_ready": bool(
            profile.name == "de_main"
            and all_stable
            and resolution_pass
            and representative_pass
        ),
        "reuse_contract": {
            "reused": [
                "head-preserving leading modes",
                "logarithmic spectral-tail bins",
                "chunk-safe spectral profile evaluation",
                "resolution-doubling curve and slope gates",
                "fail-closed local-slope audit",
            ],
            "rejected": [
                "equal-mass spectral compression",
                "fixed-width late-time fitting",
                "profile-dependent sigma calibration",
                "theory-guided best-window search",
            ],
        },
        "elapsed_seconds": time.perf_counter() - started,
        "artifacts": {
            "curves": str(output_dir / "curves.csv"),
            "spectral_profiles": str(output_dir / "spectral_profiles.csv"),
            "slope_audits": str(output_dir / "slope_audits.csv"),
            "resolution_checks": str(output_dir / "resolution_checks.csv"),
            "figure_png": str(figure_png),
            "figure_pdf": str(figure_pdf),
        },
    }
    _write_json(output_dir / "summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=sorted(DE_PROFILES), default="smoke")
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()
    profile = DE_PROFILES[arguments.profile]
    output = arguments.output or (
        Path(__file__).resolve().parent / "artifacts" / profile.name
    )
    summary = run_de_experiment(profile, output)
    print(
        json.dumps(
            {
                "profile": profile.name,
                "numerical_contract_pass": summary["numerical_contract_pass"],
                "slope_claim_ready": summary["slope_claim_ready"],
                "elapsed_seconds": summary["elapsed_seconds"],
                "artifacts": summary["artifacts"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
