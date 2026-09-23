"""Run the preserve--change--destroy experiment.

Examples from the repository root:

    ./.venv/bin/python -m iclr2027.experiments.preserve_change_destroy.run \
        --profile smoke

    ./.venv/bin/python -m iclr2027.experiments.preserve_change_destroy.run \
        --profile main --output /path/to/results
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from .config import ExperimentProfile, PROFILES
from .dynamics import horizon_for_width, run_exact_dynamics
from .monte_carlo import compare_to_exact, paired_antithetic_gap
from .slopes import audit_fixed_window
from .spectrum import (
    EmpiricalSpectrum,
    build_empirical_spectrum,
    select_global_learning_rate,
)


def calibrate_shared_sigma2(
    profile: ExperimentProfile,
    spectra: list[EmpiricalSpectrum],
    eta: float,
) -> dict[str, Any]:
    """Choose one shared sigma^2 at the clean/noise transition schedule.

    Because the exact gap is linear in sigma^2, a unit-noise run suffices.
    The declared anchor is the geometric midpoint of a finite pre-saturation
    window.  A median across feature seeds avoids tuning to one realization.
    """

    largest_width = max(profile.widths)
    references = [item for item in spectra if item.width == largest_width]
    upper = profile.nominal_fit_upper_factor * largest_width ** (
        2.0 * profile.alpha
    )
    lower = min(profile.fit_time_min, upper / 4.0)
    lower = max(lower, eta)
    anchor_time = math.sqrt(lower * upper)
    ratios = []
    anchor_rows = []
    horizon = horizon_for_width(
        largest_width, profile.alpha, eta, profile.horizon_factor
    )
    for spectrum in references:
        unit = run_exact_dynamics(
            spectrum=spectrum,
            eta=eta,
            theta=profile.calibration_theta,
            initial_batch=profile.initial_batch,
            sigma2=1.0,
            horizon=horizon,
            row_mass_cap=profile.empirical_row_mass_cap,
        )
        index = int(np.argmin(np.abs(unit.times - anchor_time)))
        if unit.noise_gap[index] <= 0.0:
            raise RuntimeError("unit-noise calibration gap is not positive")
        ratio = float(unit.clean_centered[index] / unit.noise_gap[index])
        ratios.append(ratio)
        anchor_rows.append(
            {
                "seed": spectrum.seed,
                "iteration": index,
                "intrinsic_time": float(unit.times[index]),
                "clean_centered": float(unit.clean_centered[index]),
                "unit_noise_gap": float(unit.noise_gap[index]),
                "implied_sigma2": ratio,
            }
        )
    sigma2 = float(np.median(ratios))
    if not np.isfinite(sigma2) or sigma2 <= 0.0:
        raise RuntimeError("shared noise calibration failed")
    return {
        "sigma2": sigma2,
        "theta": profile.calibration_theta,
        "target_relation": "median clean_centered = sigma2 * unit_noise_gap",
        "declared_anchor_time": anchor_time,
        "per_seed": anchor_rows,
    }


def aggregate_curves(
    profile: ExperimentProfile,
    trajectories: dict[tuple[int, int, float], Any],
) -> dict[tuple[int, float], dict[str, np.ndarray | float]]:
    aggregated: dict[tuple[int, float], dict[str, np.ndarray | float]] = {}
    for width in profile.widths:
        for theta in profile.theta_values:
            members = [
                trajectories[(width, seed, theta)] for seed in profile.seeds
            ]
            times = members[0].times
            if any(not np.array_equal(item.times, times) for item in members[1:]):
                raise RuntimeError("global eta contract violated: time grids differ")
            result: dict[str, np.ndarray | float] = {
                "width": float(width),
                "theta": float(theta),
                "times": times,
                "batches": members[0].batches,
            }
            for name in (
                "clean_centered",
                "noise_gap",
                "noisy_centered",
                "clean_total",
                "noisy_total",
                "row_mass",
            ):
                stacked = np.stack([getattr(item, name) for item in members])
                result[f"{name}_mean"] = np.mean(stacked, axis=0)
                result[f"{name}_sem"] = (
                    np.std(stacked, axis=0, ddof=1) / math.sqrt(len(members))
                    if len(members) > 1
                    else np.zeros(stacked.shape[1], dtype=float)
                )
            result["maximum_pointwise_product"] = max(
                item.maximum_pointwise_product for item in members
            )
            result["maximum_row_mass"] = max(
                item.maximum_row_mass for item in members
            )
            result["all_pointwise_stable"] = float(
                all(item.pointwise_stable for item in members)
            )
            result["all_row_stable"] = float(
                all(item.row_stable for item in members)
            )
            aggregated[(width, theta)] = result
    return aggregated


def build_fit_rows(
    profile: ExperimentProfile,
    aggregated: dict[tuple[int, float], dict[str, np.ndarray | float]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    observables = {
        "clean_centered": "clean_centered_mean",
        "noise_gap": "noise_gap_mean",
        "noisy_centered": "noisy_centered_mean",
    }
    for width in profile.widths:
        for theta in profile.theta_values:
            curve = aggregated[(width, theta)]
            times = np.asarray(curve["times"], dtype=float)
            for factor in profile.fit_upper_sensitivity:
                upper = factor * width ** (2.0 * profile.alpha)
                for observable, key in observables.items():
                    audit = audit_fixed_window(
                        times=times,
                        values=np.asarray(curve[key], dtype=float),
                        observable=observable,
                        lower_time=profile.fit_time_min,
                        upper_time=upper,
                        minimum_decades=profile.minimum_fit_decades,
                        minimum_points=profile.minimum_fit_points,
                        half_window_decades=profile.local_slope_half_window_decades,
                        maximum_local_variation=profile.maximum_local_slope_variation,
                    )
                    row = audit.to_dict()
                    row.update(
                        {
                            "width": width,
                            "theta": theta,
                            "phase": profile.phase_label(theta),
                            "upper_factor": factor,
                            "nominal_window": abs(
                                factor - profile.nominal_fit_upper_factor
                            )
                            < 1.0e-12,
                            "predicted_clean_exponent": profile.q_clean,
                            "predicted_noise_exponent": profile.noise_exponent(theta),
                            "predicted_total_exponent": profile.total_exponent(theta),
                        }
                    )
                    rows.append(row)
    return rows


def run_monte_carlo_checks(
    profile: ExperimentProfile,
    eta: float,
    sigma2: float,
) -> list[dict[str, Any]]:
    spectrum = build_empirical_spectrum(
        alpha=profile.alpha,
        beta=profile.beta,
        width=profile.monte_carlo_width,
        ambient_to_width_ratio=profile.ambient_to_width_ratio,
        seed=10_009,
    )
    validation_selection = select_global_learning_rate(
        spectra=[spectrum],
        initial_batch=profile.initial_batch,
        pointwise_product_cap=profile.pointwise_product_cap,
        row_certificate_cap=profile.row_certificate_cap,
    )
    validation_eta = min(eta, validation_selection.eta)
    rows = []
    for index, theta in enumerate(profile.representative_thetas):
        exact = run_exact_dynamics(
            spectrum=spectrum,
            eta=validation_eta,
            theta=theta,
            initial_batch=profile.initial_batch,
            sigma2=sigma2,
            horizon=profile.monte_carlo_steps,
            row_mass_cap=profile.empirical_row_mass_cap,
        )
        sampled = paired_antithetic_gap(
            eigenvalues=spectrum.eigenvalues,
            eta=validation_eta,
            batches=exact.batches,
            sigma2=sigma2,
            trajectories=profile.monte_carlo_trajectories,
            seed=90_001 + index,
        )
        comparison = compare_to_exact(sampled, exact.noise_gap)
        rows.append(
            {
                "theta": theta,
                "phase": profile.phase_label(theta),
                "width": spectrum.width,
                "feature_seed": spectrum.seed,
                "eta": validation_eta,
                "trajectory_count": profile.monte_carlo_trajectories,
                "steps": profile.monte_carlo_steps,
                "maximum_standardized_error": comparison.maximum_standardized_error,
                "fraction_within_three_standard_errors": (
                    comparison.fraction_within_three_standard_errors
                ),
                "status": "PASS" if comparison.passed else "FAIL",
            }
        )
    return rows


def run_experiment(
    profile: ExperimentProfile,
    output_dir: Path,
    sigma2_override: float | None = None,
    run_monte_carlo: bool = True,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    spectra = [
        build_empirical_spectrum(
            alpha=profile.alpha,
            beta=profile.beta,
            width=width,
            ambient_to_width_ratio=profile.ambient_to_width_ratio,
            seed=seed,
        )
        for width in profile.widths
        for seed in profile.seeds
    ]
    selection = select_global_learning_rate(
        spectra=spectra,
        initial_batch=profile.initial_batch,
        pointwise_product_cap=profile.pointwise_product_cap,
        row_certificate_cap=profile.row_certificate_cap,
    )
    calibration = calibrate_shared_sigma2(profile, spectra, selection.eta)
    if sigma2_override is not None:
        if sigma2_override <= 0.0:
            raise ValueError("sigma2 override must be positive")
        calibration["calibrated_sigma2_before_override"] = calibration["sigma2"]
        calibration["sigma2"] = float(sigma2_override)
        calibration["override_used"] = True
    else:
        calibration["override_used"] = False
    sigma2 = float(calibration["sigma2"])

    trajectories = {}
    for spectrum in spectra:
        horizon = horizon_for_width(
            spectrum.width,
            profile.alpha,
            selection.eta,
            profile.horizon_factor,
        )
        for theta in profile.theta_values:
            trajectories[(spectrum.width, spectrum.seed, theta)] = (
                run_exact_dynamics(
                    spectrum=spectrum,
                    eta=selection.eta,
                    theta=theta,
                    initial_batch=profile.initial_batch,
                    sigma2=sigma2,
                    horizon=horizon,
                    row_mass_cap=profile.empirical_row_mass_cap,
                )
            )
    aggregated = aggregate_curves(profile, trajectories)
    fit_rows = build_fit_rows(profile, aggregated)
    monte_carlo_rows = (
        run_monte_carlo_checks(profile, selection.eta, sigma2)
        if run_monte_carlo
        else []
    )

    _write_curve_csv(output_dir / "curves.csv", profile, aggregated)
    _write_rows_csv(output_dir / "slope_audits.csv", fit_rows)
    _write_rows_csv(output_dir / "paired_sgd_validation.csv", monte_carlo_rows)

    largest_width = max(profile.widths)
    largest_width_curves = {
        theta: aggregated[(largest_width, theta)]
        for theta in profile.theta_values
    }
    nominal_fits = {}
    for theta in profile.theta_values:
        matches = [
            row
            for row in fit_rows
            if row["width"] == largest_width
            and row["theta"] == theta
            and row["observable"] == "noisy_centered"
            and row["nominal_window"]
        ]
        if len(matches) != 1:
            raise RuntimeError("nominal fit lookup is not unique")
        nominal_fits[theta] = matches[0]

    from .plotting import make_three_panel_figure

    figure_png, figure_pdf = make_three_panel_figure(
        profile=profile,
        largest_width_curves=largest_width_curves,
        nominal_fits=nominal_fits,
        output_path=output_dir / "preserve_change_destroy",
    )

    all_exact_stable = all(
        item.pointwise_stable and item.row_stable
        for item in trajectories.values()
    )
    monte_carlo_passed = all(
        row["status"] == "PASS" for row in monte_carlo_rows
    ) if monte_carlo_rows else None
    representative_slope_passed = all(
        nominal_fits[theta]["status"] == "PASS"
        for theta in profile.representative_thetas
    )
    summary = {
        "experiment": "preserve_change_destroy",
        "profile": profile.name,
        "object_scope": "finite frozen-feature exact empirical dynamics",
        "not_claimed": [
            "deterministic-equivalent replacement",
            "infinite-width theorem verification",
            "paper-ready exponent evidence from the smoke profile",
        ],
        "profile_config": asdict(profile),
        "theory": {
            "q_clean": profile.q_clean,
            "q_kernel": profile.q_kernel,
            "destroy_boundary": profile.destroy_boundary,
            "preservation_boundary": profile.preservation_boundary,
        },
        "global_learning_rate": asdict(selection),
        "noise_calibration": calibration,
        "spectrum_checks": {
            "maximum_parseval_residual": max(
                item.parseval_residual for item in spectra
            ),
            "minimum_floor": min(item.approximation_floor for item in spectra),
            "maximum_floor": max(item.approximation_floor for item in spectra),
        },
        "stability": {
            "all_exact_trajectories_pass": all_exact_stable,
            "maximum_empirical_row_mass": max(
                item.maximum_row_mass for item in trajectories.values()
            ),
            "maximum_pointwise_product": max(
                item.maximum_pointwise_product for item in trajectories.values()
            ),
        },
        "paired_sgd": {
            "run": run_monte_carlo,
            "all_pass": monte_carlo_passed,
            "rows": monte_carlo_rows,
        },
        "largest_width_nominal_fits": {
            str(theta): nominal_fits[theta] for theta in profile.theta_values
        },
        "numerical_contract_pass": bool(
            all_exact_stable
            and (monte_carlo_passed is not False)
        ),
        "slope_claim_ready": bool(
            profile.name == "main" and representative_slope_passed
        ),
        "reuse_contract": {
            "reused": [
                "small-Gram empirical spectrum and overlap construction",
                "independent modal/covariance/Volterra validation pattern",
                "paired antithetic noisy-clean gap estimator",
                "local slope stability diagnostics",
            ],
            "rewritten": [
                "averaged-gradient varying-batch one-step polynomial",
                "direct gap and row-mass recursions",
                "global learning-rate and aligned intrinsic-time grid",
            ],
            "excluded": [
                "summed-gradient stationary q_gamma as a main solver",
                "per-realization learning rates",
                "forcing-as-full-batch-GD identification",
                "deterministic-equivalent contour centering",
                "theory-guided best-window selection",
            ],
        },
        "artifacts": {
            "curves": str(output_dir / "curves.csv"),
            "slope_audits": str(output_dir / "slope_audits.csv"),
            "paired_sgd": str(output_dir / "paired_sgd_validation.csv"),
            "figure_png": str(figure_png),
            "figure_pdf": str(figure_pdf),
        },
    }
    _write_json(output_dir / "summary.json", summary)
    return summary


def _write_curve_csv(
    path: Path,
    profile: ExperimentProfile,
    aggregated: dict[tuple[int, float], dict[str, np.ndarray | float]],
) -> None:
    fields = [
        "profile",
        "width",
        "theta",
        "phase",
        "iteration",
        "intrinsic_time",
        "batch",
        "clean_centered_mean",
        "clean_centered_sem",
        "noise_gap_mean",
        "noise_gap_sem",
        "noisy_centered_mean",
        "noisy_centered_sem",
        "row_mass_mean",
        "maximum_pointwise_product",
        "maximum_row_mass",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for width in profile.widths:
            for theta in profile.theta_values:
                curve = aggregated[(width, theta)]
                times = np.asarray(curve["times"])
                batches = np.asarray(curve["batches"])
                for iteration, time in enumerate(times):
                    writer.writerow(
                        {
                            "profile": profile.name,
                            "width": width,
                            "theta": theta,
                            "phase": profile.phase_label(theta),
                            "iteration": iteration,
                            "intrinsic_time": float(time),
                            "batch": int(batches[min(iteration, len(batches) - 1)]),
                            "clean_centered_mean": float(curve["clean_centered_mean"][iteration]),
                            "clean_centered_sem": float(curve["clean_centered_sem"][iteration]),
                            "noise_gap_mean": float(curve["noise_gap_mean"][iteration]),
                            "noise_gap_sem": float(curve["noise_gap_sem"][iteration]),
                            "noisy_centered_mean": float(curve["noisy_centered_mean"][iteration]),
                            "noisy_centered_sem": float(curve["noisy_centered_sem"][iteration]),
                            "row_mass_mean": float(curve["row_mass_mean"][iteration]),
                            "maximum_pointwise_product": float(curve["maximum_pointwise_product"]),
                            "maximum_row_mass": float(curve["maximum_row_mass"]),
                        }
                    )


def _write_rows_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("status\nNOT_RUN\n", encoding="utf-8")
        return
    fields = list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        numeric = float(value)
        return numeric if math.isfinite(numeric) else None
    return value


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(_json_safe(payload), handle, indent=2, sort_keys=True)
        handle.write("\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=sorted(PROFILES), default="smoke")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--sigma2", type=float)
    parser.add_argument("--skip-monte-carlo", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    profile = PROFILES[args.profile]
    output = args.output or (
        Path(__file__).resolve().parent / "artifacts" / profile.name
    )
    summary = run_experiment(
        profile=profile,
        output_dir=output,
        sigma2_override=args.sigma2,
        run_monte_carlo=not args.skip_monte_carlo,
    )
    print(json.dumps(_json_safe({
        "profile": summary["profile"],
        "numerical_contract_pass": summary["numerical_contract_pass"],
        "slope_claim_ready": summary["slope_claim_ready"],
        "artifacts": summary["artifacts"],
    }), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
