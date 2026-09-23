"""Build the paper-candidate finite-random-feature phase diagnostic.

The figure keeps three logically separate checks visible:

1. conditional finite-W Volterra dynamics versus true online Gaussian SGD;
2. the noisy--clean gap in the destroy/change/preserve representatives; and
3. descriptive finite-width exponents versus the unchanged fixed-equation
   continuation and the joint-limit phase law.

The script deliberately does not edit the manuscript.  Its outputs live under
``artifacts/paper_candidate_finite_random_feature`` for inspection first.
"""

from __future__ import annotations

import csv
import json
import math
import os
from pathlib import Path
from typing import Any

import numpy as np

from .bridge_validation import curve_comparison, simulate_true_gaussian_sgd
from .config import MAIN_PROFILE
from .dynamics import (
    horizon_for_width,
    run_exact_dynamics,
    solve_two_time_volterra,
)
from .spectrum import build_empirical_spectrum


ROOT = Path(__file__).resolve().parent
ARTIFACT_ROOT = ROOT / "artifacts"
MAIN_DIR = ARTIFACT_ROOT / "main"
FINITE_SIZE_DIR = ARTIFACT_ROOT / "finite_size_experiment_one"
OUTPUT_DIR = ARTIFACT_ROOT / "paper_candidate_finite_random_feature"

REPRESENTATIVES = (0.25, 0.50, 0.90)
AUTHENTICITY_WIDTH = 512
AUTHENTICITY_FEATURE_SEED = 11
AUTHENTICITY_SGD_TRAJECTORIES = 256
AUTHENTICITY_SGD_SEED = 20260820
VOLERRA_CROSSCHECK_STEPS = 64
DESCRIPTIVE_WINDOW = (0.04, 0.20)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("cannot write an empty CSV")
    fields: list[str] = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    path.parent.mkdir(parents=True, exist_ok=True)
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
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if math.isfinite(number) else None
    return value


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(_finite(payload), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _phase(theta: float) -> str:
    return MAIN_PROFILE.phase_label(theta)


def _fit_exponent(times: np.ndarray, values: np.ndarray, lower: float, upper: float) -> float:
    mask = (
        (times >= lower)
        & (times <= upper)
        & (times > 0.0)
        & (values > 0.0)
        & np.isfinite(times)
        & np.isfinite(values)
    )
    if int(np.count_nonzero(mask)) < 3:
        return float("nan")
    slope = np.polyfit(np.log(times[mask]), np.log(values[mask]), 1)[0]
    return -float(slope)


def _group_main_curves() -> dict[tuple[int, float], dict[str, np.ndarray]]:
    grouped: dict[tuple[int, float], list[dict[str, str]]] = {}
    for row in _read_csv(MAIN_DIR / "curves.csv"):
        key = (int(row["width"]), float(row["theta"]))
        grouped.setdefault(key, []).append(row)
    output: dict[tuple[int, float], dict[str, np.ndarray]] = {}
    for key, rows in grouped.items():
        rows.sort(key=lambda row: int(row["iteration"]))
        output[key] = {
            field: np.asarray([float(row[field]) for row in rows], dtype=float)
            for field in (
                "intrinsic_time",
                "clean_centered_mean",
                "clean_centered_sem",
                "noise_gap_mean",
                "noise_gap_sem",
                "noisy_centered_mean",
                "noisy_centered_sem",
            )
        }
    return output


def _load_fixed_equation_slopes() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for raw in _read_csv(FINITE_SIZE_DIR / "finite_size_component_slopes.csv"):
        if raw["observable"] != "noisy_centered":
            continue
        rows.append(
            {
                "source": "fixed-equation continuation",
                "effective_width": int(raw["effective_width"]),
                "theta": float(raw["theta"]),
                "phase": raw["phase"],
                "measured_exponent": float(raw["measured_exponent"]),
                "predicted_exponent": float(raw["predicted_exponent"]),
                "descriptive_only": raw["descriptive_only"] == "True",
            }
        )
    return rows


def _run_authenticity(eta: float, sigma2: float) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    spectrum = build_empirical_spectrum(
        alpha=MAIN_PROFILE.alpha,
        beta=MAIN_PROFILE.beta,
        width=AUTHENTICITY_WIDTH,
        ambient_to_width_ratio=MAIN_PROFILE.ambient_to_width_ratio,
        seed=AUTHENTICITY_FEATURE_SEED,
    )
    horizon = horizon_for_width(
        width=AUTHENTICITY_WIDTH,
        alpha=MAIN_PROFILE.alpha,
        eta=eta,
        horizon_factor=MAIN_PROFILE.horizon_factor,
    )
    curve_rows: list[dict[str, Any]] = []
    metric_rows: list[dict[str, Any]] = []
    for phase_index, theta in enumerate(REPRESENTATIVES):
        exact = run_exact_dynamics(
            spectrum=spectrum,
            eta=eta,
            theta=theta,
            initial_batch=MAIN_PROFILE.initial_batch,
            sigma2=sigma2,
            horizon=horizon,
            row_mass_cap=MAIN_PROFILE.empirical_row_mass_cap,
        )
        sampled = simulate_true_gaussian_sgd(
            spectrum=spectrum,
            eta=eta,
            batches=exact.batches,
            sigma2=sigma2,
            trajectories=AUTHENTICITY_SGD_TRAJECTORIES,
            seed=AUTHENTICITY_SGD_SEED + phase_index,
            trajectory_chunk_size=64,
        )
        sampled_centered = sampled.total - spectrum.approximation_floor
        comparison = curve_comparison(
            candidate=sampled.total,
            reference=exact.noisy_total,
            standard_error=sampled.total_standard_error,
        )
        standardized_mask = (
            (exact.times > 0.0)
            & (sampled.total_standard_error > 0.0)
            & np.isfinite(sampled.total_standard_error)
        )
        standardized_error = (
            np.abs(sampled.total[standardized_mask] - exact.noisy_total[standardized_mask])
            / sampled.total_standard_error[standardized_mask]
        )

        check_horizon = min(VOLERRA_CROSSCHECK_STEPS, horizon)
        explicit = solve_two_time_volterra(
            spectrum=spectrum,
            eta=eta,
            theta=theta,
            initial_batch=MAIN_PROFILE.initial_batch,
            sigma2=sigma2,
            horizon=check_horizon,
        )
        modal_prefix = run_exact_dynamics(
            spectrum=spectrum,
            eta=eta,
            theta=theta,
            initial_batch=MAIN_PROFILE.initial_batch,
            sigma2=sigma2,
            horizon=check_horizon,
            row_mass_cap=MAIN_PROFILE.empirical_row_mass_cap,
        )
        denominator = max(float(np.max(explicit.noisy_total)), np.finfo(float).tiny)
        volterra_modal_error = float(
            np.max(np.abs(explicit.noisy_total - modal_prefix.noisy_total)) / denominator
        )
        metric_rows.append(
            {
                "width": AUTHENTICITY_WIDTH,
                "ambient_dimension": spectrum.ambient_dimension,
                "feature_seed": AUTHENTICITY_FEATURE_SEED,
                "theta": theta,
                "phase": _phase(theta),
                "eta": eta,
                "sigma2": sigma2,
                "horizon": horizon,
                "trajectories": AUTHENTICITY_SGD_TRAJECTORIES,
                "relative_l2_true_sgd_vs_volterra": comparison["relative_l2_error"],
                "log_rmse_true_sgd_vs_volterra": comparison["log_rmse"],
                "maximum_standardized_error": float(np.max(standardized_error)),
                "fraction_within_three_standard_errors": float(
                    np.mean(standardized_error <= 3.0)
                ),
                "explicit_volterra_vs_modal_relative_linf": volterra_modal_error,
                "maximum_row_mass": exact.maximum_row_mass,
                "pointwise_stable": exact.pointwise_stable,
                "row_stable": exact.row_stable,
            }
        )
        for index, time in enumerate(exact.times):
            curve_rows.append(
                {
                    "width": AUTHENTICITY_WIDTH,
                    "ambient_dimension": spectrum.ambient_dimension,
                    "feature_seed": AUTHENTICITY_FEATURE_SEED,
                    "theta": theta,
                    "phase": _phase(theta),
                    "iteration": index,
                    "intrinsic_time": float(time),
                    "batch": int(exact.batches[index]) if index < horizon else "",
                    "volterra_centered_total": float(exact.noisy_centered[index]),
                    "true_sgd_centered_total": float(sampled_centered[index]),
                    "true_sgd_standard_error": float(sampled.total_standard_error[index]),
                    "volterra_noise_gap": float(exact.noise_gap[index]),
                    "true_sgd_noise_gap": float(sampled.gap[index]),
                    "true_sgd_noise_gap_standard_error": float(sampled.gap_standard_error[index]),
                }
            )
    return curve_rows, metric_rows


def _panel_b_rows(main_curves: dict[tuple[int, float], dict[str, np.ndarray]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    width = max(MAIN_PROFILE.widths)
    for theta in REPRESENTATIVES:
        data = main_curves[(width, theta)]
        for index, time in enumerate(data["intrinsic_time"]):
            rows.append(
                {
                    "width": width,
                    "feature_realizations": len(MAIN_PROFILE.seeds),
                    "theta": theta,
                    "phase": _phase(theta),
                    "intrinsic_time": float(time),
                    "noise_gap_mean": float(data["noise_gap_mean"][index]),
                    "noise_gap_sem": float(data["noise_gap_sem"][index]),
                    "predicted_noise_exponent": MAIN_PROFILE.noise_exponent(theta),
                }
            )
    return rows


def _panel_c_rows(
    main_curves: dict[tuple[int, float], dict[str, np.ndarray]],
    fixed_equation_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    width = max(MAIN_PROFILE.widths)
    lower = DESCRIPTIVE_WINDOW[0] * width ** (2.0 * MAIN_PROFILE.alpha)
    upper = DESCRIPTIVE_WINDOW[1] * width ** (2.0 * MAIN_PROFILE.alpha)
    for theta in MAIN_PROFILE.theta_values:
        data = main_curves[(width, theta)]
        rows.append(
            {
                "source": "finite random features",
                "effective_width": width,
                "theta": theta,
                "phase": _phase(theta),
                "measured_exponent": _fit_exponent(
                    data["intrinsic_time"],
                    data["noisy_centered_mean"],
                    lower,
                    upper,
                ),
                "predicted_exponent": MAIN_PROFILE.total_exponent(theta),
                "fit_lower_time": lower,
                "fit_upper_time": upper,
                "descriptive_only": True,
            }
        )
    rows.extend(fixed_equation_rows)
    return rows


def _log_marker_indices(times: np.ndarray, count: int = 13) -> np.ndarray:
    positive = np.flatnonzero(times > 0.0)
    targets = np.geomspace(times[positive[0]], times[positive[-1]], count)
    indices = np.unique([positive[np.argmin(np.abs(times[positive] - value))] for value in targets])
    return np.asarray(indices, dtype=int)


def _plot(
    authenticity_rows: list[dict[str, Any]],
    authenticity_metrics: list[dict[str, Any]],
    panel_b_rows: list[dict[str, Any]],
    panel_c_rows: list[dict[str, Any]],
    sigma2: float,
) -> tuple[Path, Path]:
    os.environ.setdefault("MPLCONFIGDIR", str(OUTPUT_DIR / ".matplotlib"))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.5,
            "axes.titlesize": 9.2,
            "axes.labelsize": 8.7,
            "legend.fontsize": 7.0,
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 7.5,
            "axes.linewidth": 0.75,
            "lines.solid_capstyle": "round",
        }
    )
    colors = {0.25: "#C43C39", 0.50: "#D18700", 0.90: "#2878B5"}
    labels = {0.25: "destroy", 0.50: "change", 0.90: "preserve"}
    figure, axes = plt.subplots(1, 3, figsize=(15.2, 4.25))

    grouped_a: dict[float, list[dict[str, Any]]] = {}
    for row in authenticity_rows:
        grouped_a.setdefault(float(row["theta"]), []).append(row)
    axis = axes[0]
    for theta in REPRESENTATIVES:
        rows = sorted(grouped_a[theta], key=lambda row: int(row["iteration"]))
        times = np.asarray([float(row["intrinsic_time"]) for row in rows])
        exact = np.asarray([float(row["volterra_centered_total"]) for row in rows])
        sampled = np.asarray([float(row["true_sgd_centered_total"]) for row in rows])
        error = np.asarray([float(row["true_sgd_standard_error"]) for row in rows])
        positive = (times > 0.0) & (exact > 0.0) & (sampled > 0.0)
        axis.loglog(times[positive], exact[positive], color=colors[theta], linewidth=2.0)
        marker_indices = _log_marker_indices(times)
        marker_indices = marker_indices[sampled[marker_indices] > 0.0]
        lower_error = np.minimum(error[marker_indices], 0.94 * sampled[marker_indices])
        axis.errorbar(
            times[marker_indices],
            sampled[marker_indices],
            yerr=np.vstack([lower_error, error[marker_indices]]),
            color=colors[theta],
            marker="o",
            markerfacecolor="white",
            markeredgewidth=0.85,
            markersize=3.3,
            linestyle="none",
            elinewidth=0.7,
            capsize=1.3,
            zorder=4,
        )
    axis.set_xlabel(r"intrinsic time $T$")
    axis.set_ylabel(r"centered total risk")
    axis.set_title(r"(a) Same finite $W$: Volterra vs true SGD")
    axis.grid(True, which="both", alpha=0.16, linewidth=0.55)
    phase_handles = [
        Line2D([0], [0], color=colors[theta], lw=2.0,
               label=rf"{labels[theta]}  $(\vartheta={theta:g})$")
        for theta in REPRESENTATIVES
    ]
    route_handles = [
        Line2D([0], [0], color="#333333", lw=2.0, label="finite-$W$ Volterra"),
        Line2D([0], [0], color="#333333", marker="o", markerfacecolor="white",
               linestyle="none", markersize=3.8, label=r"true-SGD mean $\pm$ 1 s.e."),
    ]
    first_legend = axis.legend(handles=phase_handles, frameon=False, loc="upper right")
    axis.add_artist(first_legend)
    axis.legend(handles=route_handles, frameon=False, loc="lower left")
    metric_text = "median rel. $L_2$ = " + format(
        float(np.median([row["relative_l2_true_sgd_vs_volterra"] for row in authenticity_metrics])),
        ".2%",
    )
    axis.text(
        0.97,
        0.04,
        metric_text,
        transform=axis.transAxes,
        fontsize=7.0,
        ha="right",
    )

    grouped_b: dict[float, list[dict[str, Any]]] = {}
    for row in panel_b_rows:
        grouped_b.setdefault(float(row["theta"]), []).append(row)
    axis = axes[1]
    for theta in REPRESENTATIVES:
        rows = sorted(grouped_b[theta], key=lambda row: float(row["intrinsic_time"]))
        times = np.asarray([float(row["intrinsic_time"]) for row in rows])
        mean = np.asarray([float(row["noise_gap_mean"]) for row in rows])
        sem = np.asarray([float(row["noise_gap_sem"]) for row in rows])
        positive = (times > 0.0) & (mean > 0.0)
        axis.loglog(times[positive], mean[positive], color=colors[theta], linewidth=2.05)
        lower = np.maximum(mean[positive] - sem[positive], np.finfo(float).tiny)
        upper = mean[positive] + sem[positive]
        axis.fill_between(times[positive], lower, upper, color=colors[theta], alpha=0.14, linewidth=0)

        guide_exponent = MAIN_PROFILE.noise_exponent(theta)
        guide_times = np.geomspace(28.0, 120.0, 40)
        reference_time = 55.0
        reference_value = float(
            np.exp(np.interp(np.log(reference_time), np.log(times[positive]), np.log(mean[positive])))
        )
        guide = reference_value * (guide_times / reference_time) ** (-guide_exponent)
        axis.loglog(guide_times, guide, color=colors[theta], linestyle="--", linewidth=1.05)
        axis.text(
            guide_times[-1] * 1.02,
            guide[-1],
            rf"$T^{{-{guide_exponent:g}}}$",
            color=colors[theta],
            fontsize=7.0,
            va="center",
        )
    axis.set_xlabel(r"intrinsic time $T$")
    axis.set_ylabel(r"noise gap $\mathcal{R}_\sigma-\mathcal{R}_0$")
    axis.set_title(r"(b) Noise gap separates the three regimes")
    axis.grid(True, which="both", alpha=0.16, linewidth=0.55)
    axis.legend(handles=phase_handles, frameon=False, loc="lower left")
    axis.set_ylim(top=1.0)
    axis.text(
        0.03,
        0.97,
        r"$m=4096$; mean $\pm$ s.e. over five $W$'s",
        transform=axis.transAxes,
        fontsize=7.0,
        ha="left",
        va="top",
    )

    axis = axes[2]
    theta_grid = np.linspace(min(MAIN_PROFILE.theta_values), max(MAIN_PROFILE.theta_values), 401)
    theory = np.asarray([MAIN_PROFILE.total_exponent(theta) for theta in theta_grid])
    axis.axvspan(
        min(theta_grid), MAIN_PROFILE.destroy_boundary, color=colors[0.25], alpha=0.055, linewidth=0
    )
    axis.axvspan(
        MAIN_PROFILE.destroy_boundary, MAIN_PROFILE.preservation_boundary,
        color=colors[0.50], alpha=0.055, linewidth=0,
    )
    axis.axvspan(
        MAIN_PROFILE.preservation_boundary, max(theta_grid),
        color=colors[0.90], alpha=0.055, linewidth=0,
    )
    axis.plot(theta_grid, theory, color="#111111", linewidth=2.25, label="joint-limit theory")

    finite_random = [row for row in panel_c_rows if row["source"] == "finite random features"]
    finite_random.sort(key=lambda row: float(row["theta"]))
    axis.plot(
        [float(row["theta"]) for row in finite_random],
        [float(row["measured_exponent"]) for row in finite_random],
        color="#707070",
        marker="o",
        markerfacecolor="white",
        markeredgewidth=1.0,
        markersize=4.0,
        linewidth=1.0,
        linestyle=":",
        label=r"finite RF $m=4096$",
    )

    continuation_styles = {
        10**4: ("#9A9A9A", "o", ":", r"fixed eq. $10^4$"),
        10**12: ("#7656A8", "s", "--", r"fixed eq. $10^{12}$"),
        10**48: ("#2F855A", "^", "-", r"fixed eq. $10^{48}$"),
    }
    for width, (color, marker, style, label) in continuation_styles.items():
        selected = [
            row for row in panel_c_rows
            if row["source"] == "fixed-equation continuation"
            and int(row["effective_width"]) == width
        ]
        selected.sort(key=lambda row: float(row["theta"]))
        axis.plot(
            [float(row["theta"]) for row in selected],
            [float(row["measured_exponent"]) for row in selected],
            color=color,
            marker=marker,
            markersize=3.6,
            linewidth=1.15,
            linestyle=style,
            label=label,
        )
    axis.axvline(MAIN_PROFILE.destroy_boundary, color=colors[0.25], linestyle="--", linewidth=0.8)
    axis.axvline(MAIN_PROFILE.preservation_boundary, color=colors[0.90], linestyle="--", linewidth=0.8)
    axis.text(0.225, 0.96, "destroy", color=colors[0.25], transform=axis.transAxes,
              ha="center", va="top", fontsize=7.1)
    axis.text(0.50, 0.96, "change", color=colors[0.50], transform=axis.transAxes,
              ha="center", va="top", fontsize=7.1)
    axis.text(0.80, 0.96, "preserve", color=colors[0.90], transform=axis.transAxes,
              ha="center", va="top", fontsize=7.1)
    axis.set_xlabel(r"batch-growth exponent $\vartheta$")
    axis.set_ylabel(r"centered-total exponent")
    axis.set_title(r"(c) Exponent flow toward the joint limit")
    axis.set_xlim(min(theta_grid), max(theta_grid))
    axis.set_ylim(-0.045, 0.82)
    axis.grid(True, alpha=0.16, linewidth=0.55)
    axis.legend(frameon=False, loc="lower right", fontsize=6.7)

    figure.suptitle(
        rf"Preserve--change--destroy diagnostic ($\alpha=0.4$, $\beta=0.3$, $\sigma^2={sigma2:.3g}$)",
        fontsize=11.2,
        y=0.995,
    )
    figure.text(
        0.5,
        0.005,
        "Finite random-feature points are descriptive/pre-asymptotic; fixed-equation curves show the unchanged-theory continuation.",
        ha="center",
        va="bottom",
        fontsize=7.2,
        color="#4A4A4A",
    )
    figure.tight_layout(rect=(0.0, 0.035, 1.0, 0.95), w_pad=2.0)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    png = OUTPUT_DIR / "finite_random_feature_preserve_change_destroy.png"
    pdf = OUTPUT_DIR / "finite_random_feature_preserve_change_destroy.pdf"
    figure.savefig(png, dpi=260, bbox_inches="tight")
    figure.savefig(pdf, bbox_inches="tight")
    plt.close(figure)
    return png, pdf


def main() -> None:
    summary = json.loads((MAIN_DIR / "summary.json").read_text(encoding="utf-8"))
    eta = float(summary["global_learning_rate"]["eta"])
    sigma2 = float(summary["noise_calibration"]["sigma2"])
    main_curves = _group_main_curves()
    authenticity_rows, authenticity_metrics = _run_authenticity(eta, sigma2)
    middle_rows = _panel_b_rows(main_curves)
    exponent_rows = _panel_c_rows(main_curves, _load_fixed_equation_slopes())

    _write_csv(OUTPUT_DIR / "finite_W_true_sgd_vs_volterra.csv", authenticity_rows)
    _write_csv(OUTPUT_DIR / "finite_W_authenticity_metrics.csv", authenticity_metrics)
    _write_csv(OUTPUT_DIR / "finite_W_noise_gap.csv", middle_rows)
    _write_csv(OUTPUT_DIR / "exponent_flow.csv", exponent_rows)
    png, pdf = _plot(
        authenticity_rows,
        authenticity_metrics,
        middle_rows,
        exponent_rows,
        sigma2,
    )

    maximum_volterra_error = max(
        float(row["explicit_volterra_vs_modal_relative_linf"])
        for row in authenticity_metrics
    )
    all_stable = all(
        bool(row["pointwise_stable"]) and bool(row["row_stable"])
        for row in authenticity_metrics
    )
    payload = {
        "experiment": "finite_random_feature_preserve_change_destroy_diagnostic",
        "paper_modified": False,
        "parameters": {
            "alpha": MAIN_PROFILE.alpha,
            "beta": MAIN_PROFILE.beta,
            "eta": eta,
            "sigma2": sigma2,
            "initial_batch": MAIN_PROFILE.initial_batch,
            "representative_thetas": list(REPRESENTATIVES),
            "authenticity_width": AUTHENTICITY_WIDTH,
            "authenticity_ambient_dimension": (
                AUTHENTICITY_WIDTH * MAIN_PROFILE.ambient_to_width_ratio
            ),
            "authenticity_feature_seed": AUTHENTICITY_FEATURE_SEED,
            "authenticity_true_sgd_trajectories": AUTHENTICITY_SGD_TRAJECTORIES,
            "random_feature_sweep_width": max(MAIN_PROFILE.widths),
            "random_feature_realizations": len(MAIN_PROFILE.seeds),
            "descriptive_fit_window_as_fraction_of_width_to_2alpha": list(
                DESCRIPTIVE_WINDOW
            ),
        },
        "checks": {
            "same_empirical_spectrum_for_volterra_and_true_sgd": True,
            "explicit_volterra_vs_modal_maximum_relative_linf": maximum_volterra_error,
            "all_conditional_stability_checks_pass": all_stable,
            "median_true_sgd_vs_volterra_relative_l2": float(
                np.median(
                    [
                        row["relative_l2_true_sgd_vs_volterra"]
                        for row in authenticity_metrics
                    ]
                )
            ),
            "minimum_fraction_within_three_standard_errors": min(
                float(row["fraction_within_three_standard_errors"])
                for row in authenticity_metrics
            ),
        },
        "claim_scope": (
            "finite-W stochastic authenticity plus pre-asymptotic random-feature "
            "behavior and convergence of the unchanged fixed equations toward the "
            "joint-limit phase law"
        ),
        "not_claimed": [
            "finite-width exact recovery of every asymptotic exponent",
            "nonlinear neural-network training",
            "finite-dataset sample-reuse SGD",
        ],
        "artifacts": {
            "figure_png": str(png),
            "figure_pdf": str(pdf),
            "authenticity_curves": str(OUTPUT_DIR / "finite_W_true_sgd_vs_volterra.csv"),
            "authenticity_metrics": str(OUTPUT_DIR / "finite_W_authenticity_metrics.csv"),
            "noise_gap_curves": str(OUTPUT_DIR / "finite_W_noise_gap.csv"),
            "exponent_flow": str(OUTPUT_DIR / "exponent_flow.csv"),
        },
    }
    _write_json(OUTPUT_DIR / "summary.json", payload)
    print(json.dumps(_finite(payload), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
