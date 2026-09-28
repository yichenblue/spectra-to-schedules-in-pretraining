"""Experiment 0: same-W finite-random-feature Volterra versus true SGD.

This is an authenticity bridge, not an exponent-verification experiment.  For
representative long-memory and integrable-memory schedules, it constructs an
actual Gaussian random-feature matrix W, evaluates the exact conditional
finite-W Volterra recurrence, and compares it with Monte Carlo population
online Gaussian SGD on the same empirical spectrum.

No manuscript source is edited.  Outputs are written to
``artifacts/experiment_0_finite_rf_sgd_bridge``.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .bridge_validation import fit_effective_exponent, simulate_true_gaussian_sgd
from .dynamics import batch_schedule, run_exact_dynamics, solve_two_time_volterra
from .spectrum import (
    EmpiricalSpectrum,
    build_empirical_spectrum,
    select_global_learning_rate,
)


ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "artifacts" / "experiment_0_finite_rf_sgd_bridge"

AMBIENT_TO_WIDTH_RATIO = 2
WIDTHS = (256, 512)
FEATURE_SEEDS = (11, 23)
DISPLAY_WIDTH = 512
DISPLAY_FEATURE_SEED = 11
INITIAL_BATCH = 16
SIGMA2 = 25.0
TRAJECTORIES = 256
TRUE_SGD_SEED = 20260821
MAXIMUM_INTRINSIC_TIME = 60.0
SOURCE_WINDOW_FRACTION = 0.35
VOL_TERRA_CROSSCHECK_STEPS = 64

MAXIMUM_CASE_RELATIVE_L2 = 0.04
MINIMUM_CASE_FRACTION_WITHIN_THREE_SE = 0.95
MAXIMUM_VOL_TERRA_MODAL_ERROR = 1.0e-12


@dataclass(frozen=True)
class Regime:
    name: str
    alpha: float
    beta: float
    schedules: tuple[tuple[float, str], ...]

    @property
    def q_kernel(self) -> float:
        return 2.0 - 1.0 / (2.0 * self.alpha)

    @property
    def q_clean(self) -> float:
        return (2.0 * self.alpha + 2.0 * self.beta - 1.0) / (
            2.0 * self.alpha
        )


REGIMES = (
    Regime(
        name="LM",
        alpha=0.4,
        beta=0.3,
        schedules=((0.25, "boundary"), (0.50, "changed"), (0.90, "preserved")),
    ),
    Regime(
        name="IM",
        alpha=0.6,
        beta=0.2,
        schedules=((0.00, "boundary"), (0.25, "changed"), (0.75, "preserved")),
    ),
)


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


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(_finite(payload), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _maximum_time(regime: Regime, width: int) -> float:
    return min(
        MAXIMUM_INTRINSIC_TIME,
        SOURCE_WINDOW_FRACTION * float(width) ** (2.0 * regime.alpha),
    )


def _relative_l2(candidate: np.ndarray, reference: np.ndarray, mask: np.ndarray) -> float:
    denominator = float(np.linalg.norm(reference[mask]))
    if denominator <= 0.0:
        return float("nan")
    return float(np.linalg.norm(candidate[mask] - reference[mask]) / denominator)


def _standardized_metrics(
    candidate: np.ndarray,
    reference: np.ndarray,
    standard_error: np.ndarray,
    times: np.ndarray,
) -> tuple[float, float]:
    mask = (
        (times > 0.0)
        & (standard_error > 0.0)
        & np.isfinite(candidate)
        & np.isfinite(reference)
        & np.isfinite(standard_error)
    )
    standardized = np.abs(candidate[mask] - reference[mask]) / standard_error[mask]
    return float(np.max(standardized)), float(np.mean(standardized <= 3.0))


def _fit_window(times: np.ndarray) -> tuple[float, float]:
    maximum = float(np.max(times))
    return 0.25 * maximum, 0.80 * maximum


def _build_spectra() -> dict[tuple[str, int, int], EmpiricalSpectrum]:
    spectra: dict[tuple[str, int, int], EmpiricalSpectrum] = {}
    for regime in REGIMES:
        for width in WIDTHS:
            for seed in FEATURE_SEEDS:
                spectra[(regime.name, width, seed)] = build_empirical_spectrum(
                    alpha=regime.alpha,
                    beta=regime.beta,
                    width=width,
                    ambient_to_width_ratio=AMBIENT_TO_WIDTH_RATIO,
                    seed=seed,
                )
    return spectra


def _run_case(
    regime: Regime,
    width: int,
    feature_seed: int,
    theta: float,
    response: str,
    spectrum: EmpiricalSpectrum,
    eta: float,
    trajectories: int,
    case_index: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    maximum_time = _maximum_time(regime, width)
    horizon = max(1, int(math.ceil(maximum_time / eta)))
    exact = run_exact_dynamics(
        spectrum=spectrum,
        eta=eta,
        theta=theta,
        initial_batch=INITIAL_BATCH,
        sigma2=SIGMA2,
        horizon=horizon,
        row_mass_cap=0.8,
    )
    sampled = simulate_true_gaussian_sgd(
        spectrum=spectrum,
        eta=eta,
        batches=exact.batches,
        sigma2=SIGMA2,
        trajectories=trajectories,
        seed=TRUE_SGD_SEED + case_index,
        trajectory_chunk_size=64,
    )
    sampled_centered = sampled.total - spectrum.approximation_floor
    positive_gap = (exact.times > 0.0) & (exact.noise_gap > 0.0)
    all_times = np.isfinite(exact.noisy_total)
    total_relative_l2 = _relative_l2(sampled.total, exact.noisy_total, all_times)
    gap_relative_l2 = _relative_l2(sampled.gap, exact.noise_gap, positive_gap)
    total_max_z, total_fraction_three = _standardized_metrics(
        sampled.total,
        exact.noisy_total,
        sampled.total_standard_error,
        exact.times,
    )
    gap_max_z, gap_fraction_three = _standardized_metrics(
        sampled.gap,
        exact.noise_gap,
        sampled.gap_standard_error,
        exact.times,
    )
    lower_fit, upper_fit = _fit_window(exact.times)
    exact_total_slope = fit_effective_exponent(
        exact.times, exact.noisy_centered, lower_fit, upper_fit
    )
    sampled_total_slope = fit_effective_exponent(
        exact.times, sampled_centered, lower_fit, upper_fit
    )
    exact_gap_slope = fit_effective_exponent(
        exact.times, np.maximum(exact.noise_gap, np.finfo(float).tiny),
        lower_fit, upper_fit,
    )
    sampled_gap_slope = fit_effective_exponent(
        exact.times, np.maximum(sampled.gap, np.finfo(float).tiny),
        lower_fit, upper_fit,
    )

    crosscheck_horizon = min(VOL_TERRA_CROSSCHECK_STEPS, horizon)
    explicit = solve_two_time_volterra(
        spectrum=spectrum,
        eta=eta,
        theta=theta,
        initial_batch=INITIAL_BATCH,
        sigma2=SIGMA2,
        horizon=crosscheck_horizon,
    )
    modal_prefix = run_exact_dynamics(
        spectrum=spectrum,
        eta=eta,
        theta=theta,
        initial_batch=INITIAL_BATCH,
        sigma2=SIGMA2,
        horizon=crosscheck_horizon,
        row_mass_cap=0.8,
    )
    scale = max(float(np.max(explicit.noisy_total)), np.finfo(float).tiny)
    explicit_modal_error = float(
        np.max(np.abs(explicit.noisy_total - modal_prefix.noisy_total)) / scale
    )

    rows: list[dict[str, Any]] = []
    for index, time in enumerate(exact.times):
        rows.append(
            {
                "regime": regime.name,
                "alpha": regime.alpha,
                "beta": regime.beta,
                "q_clean": regime.q_clean,
                "q_kernel": regime.q_kernel,
                "width": width,
                "ambient_dimension": spectrum.ambient_dimension,
                "feature_seed": feature_seed,
                "theta": theta,
                "response": response,
                "iteration": index,
                "intrinsic_time": float(time),
                "batch": int(exact.batches[index]) if index < horizon else "",
                "volterra_clean_centered": float(exact.clean_centered[index]),
                "volterra_noise_gap": float(exact.noise_gap[index]),
                "volterra_noisy_centered": float(exact.noisy_centered[index]),
                "true_sgd_clean_centered": float(
                    sampled.clean[index] - spectrum.approximation_floor
                ),
                "true_sgd_clean_standard_error": float(
                    sampled.clean_standard_error[index]
                ),
                "true_sgd_noise_gap": float(sampled.gap[index]),
                "true_sgd_noise_gap_standard_error": float(
                    sampled.gap_standard_error[index]
                ),
                "true_sgd_noisy_centered": float(sampled_centered[index]),
                "true_sgd_noisy_standard_error": float(
                    sampled.total_standard_error[index]
                ),
            }
        )

    passed = bool(
        total_relative_l2 <= MAXIMUM_CASE_RELATIVE_L2
        and gap_relative_l2 <= MAXIMUM_CASE_RELATIVE_L2
        and total_fraction_three >= MINIMUM_CASE_FRACTION_WITHIN_THREE_SE
        and gap_fraction_three >= MINIMUM_CASE_FRACTION_WITHIN_THREE_SE
        and explicit_modal_error <= MAXIMUM_VOL_TERRA_MODAL_ERROR
        and exact.pointwise_stable
        and exact.row_stable
    )
    metric = {
        "regime": regime.name,
        "alpha": regime.alpha,
        "beta": regime.beta,
        "q_clean": regime.q_clean,
        "q_kernel": regime.q_kernel,
        "width": width,
        "ambient_dimension": spectrum.ambient_dimension,
        "feature_seed": feature_seed,
        "theta": theta,
        "response": response,
        "eta": eta,
        "sigma2": SIGMA2,
        "initial_batch": INITIAL_BATCH,
        "maximum_intrinsic_time": float(exact.times[-1]),
        "horizon": horizon,
        "trajectories": trajectories,
        "approximation_floor": spectrum.approximation_floor,
        "lambda_max": spectrum.lambda_max,
        "trace": spectrum.trace,
        "total_relative_l2": total_relative_l2,
        "gap_relative_l2": gap_relative_l2,
        "total_maximum_standardized_error": total_max_z,
        "gap_maximum_standardized_error": gap_max_z,
        "total_fraction_within_three_se": total_fraction_three,
        "gap_fraction_within_three_se": gap_fraction_three,
        "fit_lower_time": lower_fit,
        "fit_upper_time": upper_fit,
        "volterra_total_descriptive_slope": exact_total_slope,
        "true_sgd_total_descriptive_slope": sampled_total_slope,
        "absolute_total_slope_difference": abs(
            sampled_total_slope - exact_total_slope
        ),
        "volterra_gap_descriptive_slope": exact_gap_slope,
        "true_sgd_gap_descriptive_slope": sampled_gap_slope,
        "absolute_gap_slope_difference": abs(sampled_gap_slope - exact_gap_slope),
        "explicit_volterra_vs_modal_relative_linf": explicit_modal_error,
        "maximum_row_mass": exact.maximum_row_mass,
        "maximum_pointwise_product": exact.maximum_pointwise_product,
        "pointwise_stable": exact.pointwise_stable,
        "row_stable": exact.row_stable,
        "status": "PASS" if passed else "FAIL",
    }
    return rows, metric


def _log_marker_indices(times: np.ndarray, count: int = 12) -> np.ndarray:
    positive = np.flatnonzero(times > 0.0)
    targets = np.geomspace(times[positive[0]], times[positive[-1]], count)
    indices = [positive[np.argmin(np.abs(times[positive] - target))] for target in targets]
    return np.unique(np.asarray(indices, dtype=int))


def _group_display_rows(
    curve_rows: list[dict[str, Any]],
) -> dict[tuple[str, float], list[dict[str, Any]]]:
    grouped: dict[tuple[str, float], list[dict[str, Any]]] = {}
    for row in curve_rows:
        if int(row["width"]) != DISPLAY_WIDTH:
            continue
        if int(row["feature_seed"]) != DISPLAY_FEATURE_SEED:
            continue
        key = (str(row["regime"]), float(row["theta"]))
        grouped.setdefault(key, []).append(row)
    for rows in grouped.values():
        rows.sort(key=lambda row: int(row["iteration"]))
    return grouped


def _plot_overlay(
    curve_rows: list[dict[str, Any]],
    observable: str,
    output_name: str,
    ylabel: str,
) -> tuple[Path, Path]:
    os.environ.setdefault("MPLCONFIGDIR", str(OUTPUT_DIR / ".matplotlib"))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.3,
            "axes.titlesize": 8.6,
            "axes.labelsize": 8.5,
            "legend.fontsize": 7.0,
            "xtick.labelsize": 7.3,
            "ytick.labelsize": 7.3,
            "axes.linewidth": 0.75,
        }
    )
    grouped = _group_display_rows(curve_rows)
    figure, axes = plt.subplots(2, 3, figsize=(11.7, 6.1), sharex="row", sharey="row")
    colors = {"boundary": "#C43C39", "changed": "#D18700", "preserved": "#2878B5"}

    for row_index, regime in enumerate(REGIMES):
        for column_index, (theta, response) in enumerate(regime.schedules):
            axis = axes[row_index, column_index]
            rows = grouped[(regime.name, theta)]
            times = np.asarray([float(row["intrinsic_time"]) for row in rows])
            if observable == "total":
                exact = np.asarray([float(row["volterra_noisy_centered"]) for row in rows])
                sampled = np.asarray([float(row["true_sgd_noisy_centered"]) for row in rows])
                standard_error = np.asarray(
                    [float(row["true_sgd_noisy_standard_error"]) for row in rows]
                )
            else:
                exact = np.asarray([float(row["volterra_noise_gap"]) for row in rows])
                sampled = np.asarray([float(row["true_sgd_noise_gap"]) for row in rows])
                standard_error = np.asarray(
                    [float(row["true_sgd_noise_gap_standard_error"]) for row in rows]
                )
            positive = (times > 0.0) & (exact > 0.0) & (sampled > 0.0)
            axis.loglog(
                times[positive], exact[positive], color=colors[response], linewidth=2.0
            )
            marker_indices = _log_marker_indices(times)
            marker_indices = marker_indices[
                (sampled[marker_indices] > 0.0) & (times[marker_indices] > 0.0)
            ]
            lower_error = np.minimum(
                standard_error[marker_indices], 0.94 * sampled[marker_indices]
            )
            axis.errorbar(
                times[marker_indices],
                sampled[marker_indices],
                yerr=np.vstack([lower_error, standard_error[marker_indices]]),
                color=colors[response],
                marker="o",
                markerfacecolor="white",
                markeredgewidth=0.8,
                markersize=3.2,
                linestyle="none",
                elinewidth=0.7,
                capsize=1.2,
                zorder=4,
            )
            axis.grid(True, which="both", alpha=0.16, linewidth=0.55)
            axis.set_title(
                rf"{regime.name} {response}: $\vartheta={theta:g}$",
                color=colors[response],
            )
            if column_index == 0:
                axis.set_ylabel(ylabel)
            if row_index == 1:
                axis.set_xlabel(r"intrinsic time $T$")
            if column_index == 2:
                axis.text(
                    0.96,
                    0.07,
                    rf"$q_{{\mathcal{{K}}}}={regime.q_kernel:.3g}$",
                    transform=axis.transAxes,
                    ha="right",
                    fontsize=7.2,
                )

    route_handles = [
        Line2D([0], [0], color="#333333", linewidth=2.0, label="finite-$W$ Volterra"),
        Line2D(
            [0], [0], color="#333333", marker="o", markerfacecolor="white",
            linestyle="none", markersize=3.8, label=r"true-SGD mean $\pm$ 1 s.e.",
        ),
    ]
    figure.legend(handles=route_handles, frameon=False, ncol=2, loc="upper center",
                  bbox_to_anchor=(0.5, 0.956))
    figure.suptitle(
        rf"Experiment 0: same-$W$ conditional authenticity "
        rf"($m={DISPLAY_WIDTH}$, $d/m=2$, $\sigma^2={SIGMA2:g}$)",
        fontsize=11.0,
        y=0.995,
    )
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.925), w_pad=1.25, h_pad=1.35)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    png = OUTPUT_DIR / f"{output_name}.png"
    pdf = OUTPUT_DIR / f"{output_name}.pdf"
    figure.savefig(png, dpi=250, bbox_inches="tight")
    figure.savefig(pdf, bbox_inches="tight")
    plt.close(figure)
    return png, pdf


def _plot_four_panel_schedule_responses(
    curve_rows: list[dict[str, Any]],
) -> tuple[Path, Path]:
    """Combine the three schedule responses in four regime--observable panels."""

    os.environ.setdefault("MPLCONFIGDIR", str(OUTPUT_DIR / ".matplotlib"))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.2,
            "axes.titlesize": 8.9,
            "axes.labelsize": 8.7,
            "legend.fontsize": 7.2,
            "xtick.labelsize": 7.3,
            "ytick.labelsize": 7.3,
            "axes.linewidth": 0.75,
        }
    )
    grouped = _group_display_rows(curve_rows)
    colors = {"boundary": "#C43C39", "changed": "#D18700", "preserved": "#2878B5"}
    panels = (
        (REGIMES[0], "total", r"(a) LM: centered total risk"),
        (REGIMES[1], "total", r"(b) IM: centered total risk"),
        (REGIMES[0], "gap", r"(c) LM: noisy--clean gap"),
        (REGIMES[1], "gap", r"(d) IM: noisy--clean gap"),
    )
    figure, axes = plt.subplots(1, 4, figsize=(15.8, 3.65), sharex=True)

    for axis, (regime, observable, title) in zip(axes, panels, strict=True):
        for theta, response in regime.schedules:
            rows = grouped[(regime.name, theta)]
            times = np.asarray([float(row["intrinsic_time"]) for row in rows])
            if observable == "total":
                exact = np.asarray(
                    [float(row["volterra_noisy_centered"]) for row in rows]
                )
                sampled = np.asarray(
                    [float(row["true_sgd_noisy_centered"]) for row in rows]
                )
                standard_error = np.asarray(
                    [float(row["true_sgd_noisy_standard_error"]) for row in rows]
                )
            else:
                exact = np.asarray(
                    [float(row["volterra_noise_gap"]) for row in rows]
                )
                sampled = np.asarray(
                    [float(row["true_sgd_noise_gap"]) for row in rows]
                )
                standard_error = np.asarray(
                    [float(row["true_sgd_noise_gap_standard_error"]) for row in rows]
                )

            positive = (times > 0.0) & (exact > 0.0) & (sampled > 0.0)
            axis.loglog(
                times[positive],
                exact[positive],
                color=colors[response],
                linewidth=2.15,
            )
            marker_indices = _log_marker_indices(times, count=9)
            marker_indices = marker_indices[
                (sampled[marker_indices] > 0.0) & (times[marker_indices] > 0.0)
            ]
            lower_error = np.minimum(
                standard_error[marker_indices], 0.94 * sampled[marker_indices]
            )
            axis.errorbar(
                times[marker_indices],
                sampled[marker_indices],
                yerr=np.vstack([lower_error, standard_error[marker_indices]]),
                color=colors[response],
                marker="o",
                markerfacecolor="white",
                markeredgewidth=0.8,
                markersize=3.1,
                linestyle="none",
                elinewidth=0.65,
                capsize=1.1,
                zorder=4,
            )

        axis.set_title(title)
        axis.set_xlabel(r"intrinsic time $T$")
        axis.grid(True, which="both", alpha=0.16, linewidth=0.55)
        axis.text(
            0.96,
            0.07,
            rf"$q_{{\mathcal{{K}}}}={regime.q_kernel:.3g}$",
            transform=axis.transAxes,
            ha="right",
            fontsize=7.2,
        )

    axes[0].set_ylabel(r"$R_\sigma(T)-R_{\mathrm{app}}$")
    axes[2].set_ylabel(r"$R_\sigma(T)-R_0(T)$")
    response_handles = [
        Line2D(
            [0],
            [0],
            color=colors[response],
            linewidth=2.15,
            label=label,
        )
        for response, label in (
            ("boundary", r"boundary"),
            ("changed", r"changed"),
            ("preserved", r"preserved"),
        )
    ]
    route_handles = [
        Line2D(
            [0], [0], color="#333333", linewidth=2.15, label="finite-$W$ Volterra"
        ),
        Line2D(
            [0],
            [0],
            color="#333333",
            marker="o",
            markerfacecolor="white",
            linestyle="none",
            markersize=3.8,
            label=r"true-SGD mean $\pm$ 1 s.e.",
        ),
    ]
    figure.legend(
        handles=response_handles + route_handles,
        frameon=False,
        ncol=5,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.965),
        columnspacing=1.7,
        handlelength=2.4,
    )
    figure.suptitle(
        rf"Experiment 0: same-$W$ schedule responses "
        rf"($m={DISPLAY_WIDTH}$, $d/m=2$, $\sigma^2={SIGMA2:g}$)",
        fontsize=11.0,
        y=1.035,
    )
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.88), w_pad=1.3)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    png = OUTPUT_DIR / "experiment0_four_panel_schedule_responses.png"
    pdf = OUTPUT_DIR / "experiment0_four_panel_schedule_responses.pdf"
    figure.savefig(png, dpi=250, bbox_inches="tight")
    figure.savefig(pdf, bbox_inches="tight")
    plt.close(figure)
    return png, pdf


def _plot_error_audit(metric_rows: list[dict[str, Any]]) -> tuple[Path, Path]:
    os.environ.setdefault("MPLCONFIGDIR", str(OUTPUT_DIR / ".matplotlib"))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8.5})
    figure, axes = plt.subplots(1, 3, figsize=(11.7, 3.6))
    response_order = ("boundary", "changed", "preserved")
    x_lookup: dict[tuple[str, str], float] = {}
    labels: list[str] = []
    short_response = {"boundary": "bdry.", "changed": "change", "preserved": "preserve"}
    for regime_index, regime in enumerate(REGIMES):
        for response_index, response in enumerate(response_order):
            position = regime_index * 4.0 + response_index
            x_lookup[(regime.name, response)] = position
            labels.append(f"{regime.name}\n{short_response[response]}")
    tick_positions = [x_lookup[(regime.name, response)] for regime in REGIMES for response in response_order]
    colors = {256: "#7A6FAC", 512: "#2F855A"}

    for metric_index, (field, ylabel, gate) in enumerate(
        (
            ("total_relative_l2", "relative $L_2$: total", MAXIMUM_CASE_RELATIVE_L2),
            ("gap_relative_l2", "relative $L_2$: gap", MAXIMUM_CASE_RELATIVE_L2),
            ("maximum_row_mass", "maximum Volterra row mass", 0.8),
        )
    ):
        axis = axes[metric_index]
        for row in metric_rows:
            base = x_lookup[(str(row["regime"]), str(row["response"]))]
            jitter = -0.10 if int(row["feature_seed"]) == FEATURE_SEEDS[0] else 0.10
            axis.scatter(
                base + jitter,
                float(row[field]),
                color=colors[int(row["width"])],
                marker="o" if int(row["feature_seed"]) == FEATURE_SEEDS[0] else "s",
                s=22,
                alpha=0.82,
                zorder=3,
            )
        axis.axhline(gate, color="#C43C39", linestyle="--", linewidth=1.0)
        axis.set_xticks(tick_positions, labels)
        axis.tick_params(axis="x", labelsize=7.3)
        axis.set_ylabel(ylabel)
        axis.grid(True, axis="y", alpha=0.18)
        axis.set_xlim(-0.55, max(tick_positions) + 0.55)
        if metric_index < 2:
            axis.set_ylim(bottom=0.0)
        else:
            axis.set_yscale("log")
            axis.set_ylim(4.0e-3, 1.0)
    handles = [
        Line2D([0], [0], color=colors[width], marker="o", linestyle="none",
               label=rf"$m={width}$")
        for width in WIDTHS
    ]
    axes[0].legend(handles=handles, frameon=False, loc="upper left")
    figure.suptitle("Experiment 0 quantitative audit across widths and frozen features",
                    fontsize=10.8)
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.93), w_pad=2.0)
    png = OUTPUT_DIR / "experiment0_error_audit.png"
    pdf = OUTPUT_DIR / "experiment0_error_audit.pdf"
    figure.savefig(png, dpi=250, bbox_inches="tight")
    figure.savefig(pdf, bbox_inches="tight")
    plt.close(figure)
    return png, pdf


def _load_reuse() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    curve_rows: list[dict[str, Any]] = [dict(row) for row in _read_csv(OUTPUT_DIR / "curves.csv")]
    metric_rows: list[dict[str, Any]] = [dict(row) for row in _read_csv(OUTPUT_DIR / "metrics.csv")]
    return curve_rows, metric_rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--plot-only", action="store_true",
        help="reuse curves.csv and metrics.csv and only rebuild figures/summary",
    )
    parser.add_argument(
        "--trajectories", type=int, default=TRAJECTORIES,
        help="true-SGD Monte Carlo trajectories per case",
    )
    arguments = parser.parse_args()
    if arguments.trajectories < 16:
        raise ValueError("at least 16 trajectories are required")

    if arguments.plot_only:
        curve_rows, metric_rows = _load_reuse()
        spectra = None
        learning_rate = None
    else:
        spectra = _build_spectra()
        learning_rate = select_global_learning_rate(
            list(spectra.values()),
            initial_batch=INITIAL_BATCH,
            pointwise_product_cap=0.6,
            row_certificate_cap=0.15,
        )
        curve_rows = []
        metric_rows = []
        case_index = 0
        for regime in REGIMES:
            for width in WIDTHS:
                for feature_seed in FEATURE_SEEDS:
                    spectrum = spectra[(regime.name, width, feature_seed)]
                    for theta, response in regime.schedules:
                        rows, metric = _run_case(
                            regime=regime,
                            width=width,
                            feature_seed=feature_seed,
                            theta=theta,
                            response=response,
                            spectrum=spectrum,
                            eta=learning_rate.eta,
                            trajectories=arguments.trajectories,
                            case_index=case_index,
                        )
                        curve_rows.extend(rows)
                        metric_rows.append(metric)
                        case_index += 1
        _write_csv(OUTPUT_DIR / "curves.csv", curve_rows)
        _write_csv(OUTPUT_DIR / "metrics.csv", metric_rows)

    total_png, total_pdf = _plot_overlay(
        curve_rows,
        observable="total",
        output_name="experiment0_centered_total_risk",
        ylabel=r"$R_\sigma(T)-R_{\mathrm{app}}$",
    )
    gap_png, gap_pdf = _plot_overlay(
        curve_rows,
        observable="gap",
        output_name="experiment0_noisy_clean_gap",
        ylabel=r"$R_\sigma(T)-R_0(T)$",
    )
    combined_png, combined_pdf = _plot_four_panel_schedule_responses(curve_rows)
    audit_png, audit_pdf = _plot_error_audit(metric_rows)

    numeric = lambda field: [float(row[field]) for row in metric_rows]
    all_cases_pass = all(str(row["status"]) == "PASS" for row in metric_rows)
    summary = {
        "experiment": "experiment_0_finite_rf_sgd_bridge",
        "purpose": (
            "same-W validation of the exact conditional finite-random-feature "
            "Volterra recurrence against population online Gaussian true SGD"
        ),
        "paper_modified": False,
        "workflow_promoted": False,
        "parameters": {
            "regimes": [
                {
                    "name": regime.name,
                    "alpha": regime.alpha,
                    "beta": regime.beta,
                    "q_clean": regime.q_clean,
                    "q_kernel": regime.q_kernel,
                    "schedules": [
                        {"theta": theta, "response": response}
                        for theta, response in regime.schedules
                    ],
                }
                for regime in REGIMES
            ],
            "widths": list(WIDTHS),
            "ambient_to_width_ratio": AMBIENT_TO_WIDTH_RATIO,
            "feature_seeds": list(FEATURE_SEEDS),
            "sigma2": SIGMA2,
            "initial_batch": INITIAL_BATCH,
            "trajectories_per_case": int(metric_rows[0]["trajectories"]),
            "eta": float(metric_rows[0]["eta"]),
            "maximum_intrinsic_time": MAXIMUM_INTRINSIC_TIME,
            "source_window_fraction": SOURCE_WINDOW_FRACTION,
        },
        "gates": {
            "maximum_case_relative_l2": MAXIMUM_CASE_RELATIVE_L2,
            "minimum_case_fraction_within_three_se": (
                MINIMUM_CASE_FRACTION_WITHIN_THREE_SE
            ),
            "maximum_volterra_modal_error": MAXIMUM_VOL_TERRA_MODAL_ERROR,
            "all_cases_pass": all_cases_pass,
        },
        "aggregate_results": {
            "case_count": len(metric_rows),
            "median_total_relative_l2": float(np.median(numeric("total_relative_l2"))),
            "maximum_total_relative_l2": max(numeric("total_relative_l2")),
            "median_gap_relative_l2": float(np.median(numeric("gap_relative_l2"))),
            "maximum_gap_relative_l2": max(numeric("gap_relative_l2")),
            "minimum_total_fraction_within_three_se": min(
                numeric("total_fraction_within_three_se")
            ),
            "minimum_gap_fraction_within_three_se": min(
                numeric("gap_fraction_within_three_se")
            ),
            "median_absolute_total_slope_difference": float(
                np.median(numeric("absolute_total_slope_difference"))
            ),
            "median_absolute_gap_slope_difference": float(
                np.median(numeric("absolute_gap_slope_difference"))
            ),
            "maximum_explicit_volterra_modal_relative_linf": max(
                numeric("explicit_volterra_vs_modal_relative_linf")
            ),
            "maximum_row_mass": max(numeric("maximum_row_mass")),
            "all_pointwise_stable": all(
                str(row["pointwise_stable"]) == "True" or row["pointwise_stable"] is True
                for row in metric_rows
            ),
            "all_row_stable": all(
                str(row["row_stable"]) == "True" or row["row_stable"] is True
                for row in metric_rows
            ),
        },
        "claim_scope": (
            "conditional finite-W expected-risk authenticity on the audited "
            "finite intrinsic-time windows; not an asymptotic exponent test"
        ),
        "not_claimed": [
            "fixed-width asymptotic power-law verification",
            "population-to-empirical random-matrix equivalence",
            "finite-dataset sample-reuse SGD",
            "nonlinear feature learning",
        ],
        "artifacts": {
            "curves": str(OUTPUT_DIR / "curves.csv"),
            "metrics": str(OUTPUT_DIR / "metrics.csv"),
            "centered_total_png": str(total_png),
            "centered_total_pdf": str(total_pdf),
            "gap_png": str(gap_png),
            "gap_pdf": str(gap_pdf),
            "four_panel_schedule_responses_png": str(combined_png),
            "four_panel_schedule_responses_pdf": str(combined_pdf),
            "audit_png": str(audit_png),
            "audit_pdf": str(audit_pdf),
        },
    }
    _write_json(OUTPUT_DIR / "summary.json", summary)
    print(json.dumps(_finite(summary), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
