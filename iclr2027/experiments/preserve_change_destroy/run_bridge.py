"""Run the true-SGD/discrete/continuum authenticity bridge."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import csv
import json
import math
from pathlib import Path
import time
from typing import Any

import numpy as np

from .bridge_config import BRIDGE_PROFILES, BridgeProfile
from .bridge_plotting import make_bridge_figures
from .bridge_validation import (
    BridgeCase,
    curve_comparison,
    fit_effective_exponent,
    run_bridge_case,
)


ROOT = Path(__file__).resolve().parent


def _observable_arrays(
    case: BridgeCase,
    observable: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if observable == "clean":
        return (
            case.exact.clean_total,
            case.approximation_floor + case.continuum.clean_centered,
            case.sampled.clean,
            case.sampled.clean_standard_error,
        )
    if observable == "gap":
        return (
            case.exact.noise_gap,
            case.continuum.noise_gap,
            case.sampled.gap,
            case.sampled.gap_standard_error,
        )
    if observable == "total":
        return (
            case.exact.noisy_total,
            case.approximation_floor + case.continuum.noisy_centered,
            case.sampled.total,
            case.sampled.total_standard_error,
        )
    raise KeyError(observable)


def _case_seed(profile: BridgeProfile, width: int, theta: float) -> int:
    sequence = np.random.SeedSequence(
        [profile.seed, width, int(round(1000.0 * theta))]
    )
    return int(sequence.generate_state(1, dtype=np.uint32)[0])


def _comparison_rows(
    profile: BridgeProfile,
    cases: dict[tuple[int, float], BridgeCase],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for width in profile.widths:
        lower = float(width) ** profile.fit_lower_exponent
        upper = float(width) ** profile.fit_upper_exponent
        for theta in profile.theta_values:
            case = cases[(width, theta)]
            times = case.exact.times
            full_mask = times > 0.0
            fit_mask = full_mask & (times >= lower) & (times <= upper)
            for observable in ("clean", "gap", "total"):
                exact, continuum, sampled, standard_error = _observable_arrays(
                    case, observable
                )
                exact_slope = fit_effective_exponent(
                    times, exact, lower, upper
                )
                for route, candidate, errors in (
                    ("true_sgd", sampled, standard_error),
                    ("continuum", continuum, None),
                ):
                    full = curve_comparison(
                        candidate[full_mask],
                        exact[full_mask],
                        None if errors is None else errors[full_mask],
                    )
                    fit = curve_comparison(
                        candidate[fit_mask],
                        exact[fit_mask],
                        None if errors is None else errors[fit_mask],
                    )
                    candidate_slope = fit_effective_exponent(
                        times, candidate, lower, upper
                    )
                    slope_error = abs(candidate_slope - exact_slope)
                    if route == "true_sgd":
                        curve_tolerance = (
                            profile.maximum_monte_carlo_relative_l2_error
                        )
                        slope_tolerance = profile.maximum_monte_carlo_slope_error
                    else:
                        curve_tolerance = profile.maximum_continuum_relative_l2_error
                        slope_tolerance = profile.maximum_continuum_slope_error
                    passed = bool(
                        np.isfinite(fit["relative_l2_error"])
                        and np.isfinite(slope_error)
                        and fit["relative_l2_error"] <= curve_tolerance
                        and slope_error <= slope_tolerance
                    )
                    rows.append(
                        {
                            "profile": profile.name,
                            "width": width,
                            "theta": theta,
                            "observable": observable,
                            "route": route,
                            "trajectories": profile.trajectories,
                            "fit_lower_time": lower,
                            "fit_upper_time": upper,
                            "fit_point_count": int(np.count_nonzero(fit_mask)),
                            "exact_effective_exponent": exact_slope,
                            "candidate_effective_exponent": candidate_slope,
                            "absolute_slope_error": slope_error,
                            "full_relative_l2_error": full["relative_l2_error"],
                            "fit_relative_l2_error": fit["relative_l2_error"],
                            "fit_log_rmse": fit["log_rmse"],
                            "maximum_standardized_error": full[
                                "maximum_standardized_error"
                            ],
                            "fraction_within_three_standard_errors": full[
                                "fraction_within_three_standard_errors"
                            ],
                            "curve_tolerance": curve_tolerance,
                            "slope_tolerance": slope_tolerance,
                            "status": "PASS" if passed else "FAIL",
                        }
                    )
    return rows


def _curve_rows(
    profile: BridgeProfile,
    cases: dict[tuple[int, float], BridgeCase],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for width in profile.widths:
        for theta in profile.theta_values:
            case = cases[(width, theta)]
            for index, intrinsic_time in enumerate(case.exact.times):
                rows.append(
                    {
                        "profile": profile.name,
                        "width": width,
                        "theta": theta,
                        "step": index,
                        "intrinsic_time": intrinsic_time,
                        "batch": (
                            int(case.exact.batches[index])
                            if index < case.exact.batches.size
                            else ""
                        ),
                        "exact_clean": case.exact.clean_total[index],
                        "continuum_clean": (
                            case.approximation_floor
                            + case.continuum.clean_centered[index]
                        ),
                        "true_sgd_clean": case.sampled.clean[index],
                        "true_sgd_clean_se": (
                            case.sampled.clean_standard_error[index]
                        ),
                        "exact_gap": case.exact.noise_gap[index],
                        "continuum_gap": case.continuum.noise_gap[index],
                        "true_sgd_gap": case.sampled.gap[index],
                        "true_sgd_gap_se": case.sampled.gap_standard_error[index],
                        "exact_total": case.exact.noisy_total[index],
                        "continuum_total": (
                            case.approximation_floor
                            + case.continuum.noisy_centered[index]
                        ),
                        "true_sgd_total": case.sampled.total[index],
                        "true_sgd_total_se": (
                            case.sampled.total_standard_error[index]
                        ),
                    }
                )
    return rows


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError("cannot write an empty CSV")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run_profile(profile: BridgeProfile, output_dir: Path) -> dict[str, Any]:
    started = time.perf_counter()
    cases: dict[tuple[int, float], BridgeCase] = {}
    for width in profile.widths:
        for theta in profile.theta_values:
            case = run_bridge_case(
                alpha=profile.alpha,
                beta=profile.beta,
                width=width,
                theta=theta,
                eta=profile.eta,
                initial_batch=profile.initial_batch,
                sigma2=profile.sigma2,
                horizon_exponent=profile.horizon_exponent,
                trajectories=profile.trajectories,
                seed=_case_seed(profile, width, theta),
                row_mass_cap=profile.row_mass_cap,
            )
            cases[(width, theta)] = case
            print(
                json.dumps(
                    {
                        "completed_width": width,
                        "theta": theta,
                        "steps": int(case.exact.times.size - 1),
                    }
                ),
                flush=True,
            )

    comparison_rows = _comparison_rows(profile, cases)
    curve_rows = _curve_rows(profile, cases)
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(output_dir / "bridge_metrics.csv", comparison_rows)
    _write_csv(output_dir / "bridge_curves.csv", curve_rows)
    figure_paths = make_bridge_figures(profile, cases, output_dir)
    all_pass = all(row["status"] == "PASS" for row in comparison_rows)
    maximum_errors: dict[str, dict[str, float]] = {}
    for route in ("true_sgd", "continuum"):
        selected = [row for row in comparison_rows if row["route"] == route]
        maximum_errors[route] = {
            "maximum_fit_relative_l2_error": max(
                float(row["fit_relative_l2_error"]) for row in selected
            ),
            "maximum_absolute_slope_error": max(
                float(row["absolute_slope_error"]) for row in selected
            ),
        }
    summary = {
        "experiment": "preserve_change_destroy_authenticity_bridge",
        "profile": profile.name,
        "elapsed_seconds": time.perf_counter() - started,
        "object_scope": (
            "analytic power-law Gaussian online-SGD population risk"
        ),
        "profile_config": asdict(profile),
        "same_object_contract": {
            "spectrum": "lambda_j=j^(-2 alpha), teacher_mass_j=j^(-2(alpha+beta))",
            "time": "T_t=eta*t",
            "batch": "B_t=ceil(B0*(1+T_t)^theta)",
            "noise_variance": profile.sigma2,
            "continuum_cell_inverse_ratio": "eta/B_t",
        },
        "true_sgd_sampler": {
            "method": "distributionally exact collapsed Gaussian mini-batch",
            "uses_second_moment_recursion": False,
            "cost_per_trajectory_step": "O(m), independent of B_t",
        },
        "all_bridge_gates_pass": all_pass,
        "maximum_errors": maximum_errors,
        "maximum_exact_row_mass": max(
            case.exact.maximum_row_mass for case in cases.values()
        ),
        "maximum_continuum_row_mass": max(
            case.continuum.maximum_row_mass for case in cases.values()
        ),
        "not_claimed": [
            "finite-dataset sample-reuse SGD",
            "nonlinear neural-network training",
            "random-feature empirical-spectrum equivalence",
            "asymptotic exponent evidence from these moderate widths",
        ],
        "artifacts": {
            "curves": str(output_dir / "bridge_curves.csv"),
            "metrics": str(output_dir / "bridge_metrics.csv"),
            "figures": [str(path) for path in figure_paths],
        },
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=tuple(BRIDGE_PROFILES), default="smoke")
    parser.add_argument("--output-dir", type=Path)
    arguments = parser.parse_args()
    profile = BRIDGE_PROFILES[arguments.profile]
    output_dir = arguments.output_dir or (
        ROOT / "artifacts" / profile.name
    )
    summary = run_profile(profile, output_dir)
    print(
        json.dumps(
            {
                "profile": summary["profile"],
                "elapsed_seconds": summary["elapsed_seconds"],
                "all_bridge_gates_pass": summary["all_bridge_gates_pass"],
                "maximum_errors": summary["maximum_errors"],
                "artifacts": summary["artifacts"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
