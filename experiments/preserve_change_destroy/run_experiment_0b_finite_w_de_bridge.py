"""Experiment 0b: finite-W conditional Volterra versus the RF resolvent DE.

The experiment keeps the exact schedules and normalization of Experiment 0.
For each regime, width, and schedule response, it compares conditional
finite-random-feature Volterra curves across frozen feature realizations with
the support-adaptive, zero-regularization deterministic-equivalent spectral
measure developed and validated in ws_443.  The DE spectral measure is then
propagated by the same discrete varying-batch recursion as the finite-W curve;
no continuum-time approximation is introduced.  The audited numerical core is
bundled locally in :mod:`resolvent_de`; this experiment has no runtime
dependency on the historical workstream.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
from typing import Any

import numpy as np

from .dynamics import batch_schedule, one_step_polynomial, run_exact_dynamics
from .resolvent_de import compute_density_level, density_to_modes
from .run_experiment_0_finite_rf_sgd_bridge import (
    AMBIENT_TO_WIDTH_RATIO,
    INITIAL_BATCH,
    REGIMES,
    Regime,
    SIGMA2,
)
from .spectrum import build_empirical_spectrum


Array = np.ndarray
ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "artifacts" / "experiment_0b_finite_w_de_bridge"
DE_DENSITY_CACHE = OUTPUT_DIR / "cache" / "support_adaptive_density"

WIDTHS = (128, 256, 512)
FEATURE_SEEDS = (11, 23, 37, 53, 71, 89, 107, 131)
DISPLAY_WIDTH = 512
ETA = 0.1542358173412625
MAXIMUM_INTRINSIC_TIME = 60.0
SOURCE_WINDOW_FRACTION = 0.35
ROW_MASS_CAP = 0.8
DE_COARSE_LEVEL = 0
DE_FINE_LEVEL = 1

MAXIMUM_DE_RESOLUTION_RELATIVE_L2 = 0.01
MAXIMUM_DISPLAY_MEDIAN_PER_W_RELATIVE_L2 = 0.08
MAXIMUM_DISPLAY_AGGREGATE_RELATIVE_L2 = 0.04


@dataclass(frozen=True)
class DETrajectory:
    times: Array
    batches: Array
    clean_centered: Array
    noise_gap: Array
    noisy_centered: Array
    row_mass: Array
    maximum_pointwise_product: float
    maximum_row_mass: float
    pointwise_stable: bool
    row_stable: bool


def _maximum_time(regime: Regime, width: int) -> float:
    return min(
        MAXIMUM_INTRINSIC_TIME,
        SOURCE_WINDOW_FRACTION * float(width) ** (2.0 * regime.alpha),
    )


def _relative_l2(candidate: Array, reference: Array) -> float:
    candidate = np.asarray(candidate, dtype=float)
    reference = np.asarray(reference, dtype=float)
    denominator = float(np.linalg.norm(reference))
    if denominator <= 0.0:
        return float("nan")
    return float(np.linalg.norm(candidate - reference) / denominator)


def run_de_schedule(
    modes: dict[str, Any],
    eta: float,
    theta: float,
    initial_batch: int,
    sigma2: float,
    horizon: int,
    row_mass_cap: float = ROW_MASS_CAP,
) -> DETrajectory:
    """Propagate a DE spectral measure through the exact discrete schedule."""

    kernel_nodes = np.asarray(modes["kernel_nodes"], dtype=float)
    kernel_weights = np.asarray(modes["kernel_trace_weights"], dtype=float)
    forcing_nodes = np.asarray(modes["forcing_nodes"], dtype=float)
    direct_clean_modes = np.asarray(modes["forcing_weights"], dtype=float).copy()
    null_weight = float(modes["null_weight"])
    if (
        eta <= 0.0
        or sigma2 < 0.0
        or horizon < 0
        or np.any(kernel_nodes <= 0.0)
        or np.any(kernel_weights < 0.0)
        or np.any(forcing_nodes <= 0.0)
        or np.any(direct_clean_modes < 0.0)
        or null_weight < 0.0
    ):
        raise ValueError("invalid deterministic-equivalent modal inputs")

    clean_feedback_modes = np.zeros_like(kernel_nodes)
    gap_modes = np.zeros_like(kernel_nodes)
    row_modes = np.zeros_like(kernel_nodes)
    times, batches = batch_schedule(eta, theta, initial_batch, horizon)
    clean = np.empty(horizon + 1, dtype=float)
    gap = np.empty_like(clean)
    row = np.empty_like(clean)
    pointwise_products = np.empty(horizon, dtype=float)
    clean[0] = float(np.sum(direct_clean_modes))
    gap[0] = 0.0
    row[0] = 0.0
    maximum_node = max(
        float(np.max(kernel_nodes, initial=0.0)),
        float(np.max(forcing_nodes, initial=0.0)),
    )

    for step, batch in enumerate(batches):
        batch_int = int(batch)
        q_kernel = one_step_polynomial(kernel_nodes, eta, batch_int)
        q_forcing = one_step_polynomial(forcing_nodes, eta, batch_int)
        injection = (
            (eta * eta / batch_int)
            * kernel_nodes
            * kernel_nodes
            * kernel_weights
        )
        pointwise_products[step] = (
            eta * (1.0 + 1.0 / batch_int) * maximum_node
        )

        current_clean_total = null_weight + float(
            np.sum(direct_clean_modes) + np.sum(clean_feedback_modes)
        )
        current_gap = float(np.sum(gap_modes))
        direct_clean_modes *= q_forcing
        clean_feedback_modes = (
            q_kernel * clean_feedback_modes + injection * current_clean_total
        )
        gap_modes = q_kernel * gap_modes + injection * (current_gap + sigma2)
        row_modes = q_kernel * row_modes + injection

        scale = max(1.0, float(modes["target_energy"]), sigma2)
        tolerance = 2.0e-12 * scale
        minimum = min(
            float(np.min(direct_clean_modes, initial=0.0)),
            float(np.min(clean_feedback_modes, initial=0.0)),
            float(np.min(gap_modes, initial=0.0)),
            float(np.min(row_modes, initial=0.0)),
        )
        if minimum < -tolerance:
            raise FloatingPointError("a DE modal state became materially negative")
        direct_clean_modes = np.maximum(direct_clean_modes, 0.0)
        clean_feedback_modes = np.maximum(clean_feedback_modes, 0.0)
        gap_modes = np.maximum(gap_modes, 0.0)
        row_modes = np.maximum(row_modes, 0.0)
        clean[step + 1] = float(
            np.sum(direct_clean_modes) + np.sum(clean_feedback_modes)
        )
        gap[step + 1] = float(np.sum(gap_modes))
        row[step + 1] = float(np.sum(row_modes))

    maximum_product = float(np.max(pointwise_products, initial=0.0))
    maximum_row = float(np.max(row, initial=0.0))
    return DETrajectory(
        times=times,
        batches=batches,
        clean_centered=clean,
        noise_gap=gap,
        noisy_centered=clean + gap,
        row_mass=row,
        maximum_pointwise_product=maximum_product,
        maximum_row_mass=maximum_row,
        pointwise_stable=maximum_product < 2.0,
        row_stable=maximum_row < row_mass_cap,
    )


def _de_modes(
    regime: Regime,
    width: int,
    level: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    phase = f"schedule_{regime.name}"
    density = compute_density_level(
        phase=phase,
        alpha=regime.alpha,
        beta=regime.beta,
        width=width,
        level=level,
        cache_dir=DE_DENSITY_CACHE,
    )
    modes = density_to_modes(regime.alpha, regime.beta, width, density)
    diagnostics = {
        "density_points": int(np.asarray(density["x"]).size),
        "max_m_residual": float(density["max_m_residual"]),
        "max_m_iterations": int(density["max_m_iterations"]),
        "kernel_mode_count": int(len(modes["kernel_nodes"])),
        "forcing_mode_count": int(len(modes["forcing_nodes"])),
        "raw_trace_mass_relative_error": float(
            modes["raw_trace_mass_relative_error"]
        ),
        "raw_target_mass_relative_error": float(
            modes["raw_target_mass_relative_error"]
        ),
        "raw_trace_first_moment_relative_error": float(
            modes["raw_trace_first_moment_relative_error"]
        ),
        "raw_target_first_moment_relative_error": float(
            modes["raw_target_first_moment_relative_error"]
        ),
        "trace_mass_identity_error": abs(
            float(np.sum(modes["kernel_trace_weights"])) - width
        ),
        "target_mass_identity_error": abs(
            float(np.sum(modes["forcing_weights"]))
            + float(modes["null_weight"])
            - float(modes["target_energy"])
        ),
    }
    return modes, diagnostics


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


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _plot_curves(curve_rows: list[dict[str, Any]]) -> tuple[Path, Path]:
    os.environ.setdefault("MPLCONFIGDIR", str(OUTPUT_DIR / ".matplotlib"))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    colors = {"boundary": "#C43C39", "changed": "#D18700", "preserved": "#2878B5"}
    panels = (
        ("LM", "total", r"(a) LM: centered total risk"),
        ("IM", "total", r"(b) IM: centered total risk"),
        ("LM", "gap", r"(c) LM: noisy--clean gap"),
        ("IM", "gap", r"(d) IM: noisy--clean gap"),
    )
    figure, axes = plt.subplots(1, 4, figsize=(15.8, 3.65), sharex=True)
    for axis, (regime_name, observable, title) in zip(axes, panels, strict=True):
        regime = next(item for item in REGIMES if item.name == regime_name)
        for theta, response in regime.schedules:
            selected = [
                row
                for row in curve_rows
                if row["regime"] == regime_name
                and int(row["width"]) == DISPLAY_WIDTH
                and float(row["theta"]) == theta
            ]
            selected.sort(key=lambda row: int(row["iteration"]))
            times = np.asarray([row["intrinsic_time"] for row in selected], dtype=float)
            if observable == "total":
                de = np.asarray([row["de_noisy_centered"] for row in selected])
                median = np.asarray([row["finite_median_noisy_centered"] for row in selected])
                low = np.asarray([row["finite_q10_noisy_centered"] for row in selected])
                high = np.asarray([row["finite_q90_noisy_centered"] for row in selected])
            else:
                de = np.asarray([row["de_noise_gap"] for row in selected])
                median = np.asarray([row["finite_median_noise_gap"] for row in selected])
                low = np.asarray([row["finite_q10_noise_gap"] for row in selected])
                high = np.asarray([row["finite_q90_noise_gap"] for row in selected])
            positive = (times > 0.0) & (de > 0.0) & (median > 0.0)
            axis.fill_between(
                times[positive],
                np.maximum(low[positive], np.finfo(float).tiny),
                high[positive],
                color=colors[response],
                alpha=0.13,
                linewidth=0.0,
            )
            axis.loglog(
                times[positive], median[positive], color=colors[response], linewidth=1.9
            )
            axis.loglog(
                times[positive],
                de[positive],
                color=colors[response],
                linestyle=(0, (4, 2)),
                linewidth=1.9,
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
        Line2D([0], [0], color=colors[key], linewidth=2.0, label=label)
        for key, label in (
            ("boundary", "boundary"),
            ("changed", "changed"),
            ("preserved", "preserved"),
        )
    ]
    route_handles = [
        Line2D([0], [0], color="#333333", linewidth=1.9, label=r"finite-$W$ median"),
        Patch(facecolor="#888888", alpha=0.18, label=r"finite-$W$ 10--90%"),
        Line2D(
            [0],
            [0],
            color="#333333",
            linestyle=(0, (4, 2)),
            linewidth=1.9,
            label="resolvent DE Volterra",
        ),
    ]
    figure.legend(
        handles=response_handles + route_handles,
        frameon=False,
        ncol=6,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.965),
        columnspacing=1.4,
    )
    figure.suptitle(
        rf"Experiment 0b: finite-$W$ versus deterministic-equivalent Volterra "
        rf"($m={DISPLAY_WIDTH}$, $d/m=2$, $\sigma^2={SIGMA2:g}$)",
        fontsize=11.0,
        y=1.035,
    )
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.88), w_pad=1.3)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    png = OUTPUT_DIR / "experiment0b_four_panel_finite_w_vs_de.png"
    pdf = OUTPUT_DIR / "experiment0b_four_panel_finite_w_vs_de.pdf"
    figure.savefig(png, dpi=250, bbox_inches="tight")
    figure.savefig(pdf, bbox_inches="tight")
    plt.close(figure)
    return png, pdf


def _plot_errors(metric_rows: list[dict[str, Any]]) -> tuple[Path, Path]:
    os.environ.setdefault("MPLCONFIGDIR", str(OUTPUT_DIR / ".matplotlib"))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {"boundary": "#C43C39", "changed": "#D18700", "preserved": "#2878B5"}
    panels = (
        ("LM", "total", r"(a) LM: centered total"),
        ("IM", "total", r"(b) IM: centered total"),
        ("LM", "gap", r"(c) LM: noisy--clean gap"),
        ("IM", "gap", r"(d) IM: noisy--clean gap"),
    )
    figure, axes = plt.subplots(1, 4, figsize=(15.8, 3.45), sharex=True, sharey=True)
    for axis, (regime_name, observable, title) in zip(axes, panels, strict=True):
        regime = next(item for item in REGIMES if item.name == regime_name)
        for theta, response in regime.schedules:
            selected = [
                row
                for row in metric_rows
                if row["regime"] == regime_name
                and float(row["theta"]) == theta
            ]
            selected.sort(key=lambda row: int(row["width"]))
            widths = np.asarray([row["width"] for row in selected], dtype=float)
            prefix = "total" if observable == "total" else "gap"
            median = np.asarray(
                [row[f"median_per_w_{prefix}_relative_l2"] for row in selected]
            )
            q10 = np.asarray(
                [row[f"q10_per_w_{prefix}_relative_l2"] for row in selected]
            )
            q90 = np.asarray(
                [row[f"q90_per_w_{prefix}_relative_l2"] for row in selected]
            )
            axis.errorbar(
                widths,
                median,
                yerr=np.vstack([median - q10, q90 - median]),
                color=colors[response],
                marker="o",
                markersize=4.0,
                linewidth=1.7,
                capsize=2.0,
                label=response,
            )
        axis.axhline(
            MAXIMUM_DISPLAY_MEDIAN_PER_W_RELATIVE_L2,
            color="#777777",
            linestyle=":",
            linewidth=1.0,
        )
        axis.set_xscale("log", base=2)
        axis.set_yscale("log")
        axis.set_xticks(WIDTHS, [str(width) for width in WIDTHS])
        axis.set_title(title)
        axis.set_xlabel(r"feature width $m$")
        axis.grid(True, which="both", alpha=0.18)
    axes[0].set_ylabel(r"finite-$W$ versus DE relative $L_2$")
    axes[0].legend(frameon=False, fontsize=7.3)
    figure.suptitle(
        "Experiment 0b: feature-realization error across widths",
        fontsize=11.0,
        y=1.01,
    )
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.93), w_pad=1.25)
    png = OUTPUT_DIR / "experiment0b_relative_error_by_width.png"
    pdf = OUTPUT_DIR / "experiment0b_relative_error_by_width.pdf"
    figure.savefig(png, dpi=250, bbox_inches="tight")
    figure.savefig(pdf, bbox_inches="tight")
    plt.close(figure)
    return png, pdf


def run() -> dict[str, Any]:
    curve_rows: list[dict[str, Any]] = []
    metric_rows: list[dict[str, Any]] = []
    per_w_rows: list[dict[str, Any]] = []
    spectral_rows: list[dict[str, Any]] = []

    for regime in REGIMES:
        for width in WIDTHS:
            horizon = max(1, int(math.ceil(_maximum_time(regime, width) / ETA)))
            coarse_modes, coarse_diagnostics = _de_modes(
                regime, width, DE_COARSE_LEVEL
            )
            fine_modes, fine_diagnostics = _de_modes(
                regime, width, DE_FINE_LEVEL
            )
            for level, diagnostics in (
                (DE_COARSE_LEVEL, coarse_diagnostics),
                (DE_FINE_LEVEL, fine_diagnostics),
            ):
                spectral_rows.append(
                    {
                        "regime": regime.name,
                        "alpha": regime.alpha,
                        "beta": regime.beta,
                        "width": width,
                        "ambient_dimension": AMBIENT_TO_WIDTH_RATIO * width,
                        "adaptive_level": level,
                        **diagnostics,
                    }
                )

            spectra = [
                build_empirical_spectrum(
                    alpha=regime.alpha,
                    beta=regime.beta,
                    width=width,
                    ambient_to_width_ratio=AMBIENT_TO_WIDTH_RATIO,
                    seed=seed,
                )
                for seed in FEATURE_SEEDS
            ]
            for theta, response in regime.schedules:
                de = run_de_schedule(
                    fine_modes, ETA, theta, INITIAL_BATCH, SIGMA2, horizon
                )
                de_coarse = run_de_schedule(
                    coarse_modes, ETA, theta, INITIAL_BATCH, SIGMA2, horizon
                )
                finite = [
                    run_exact_dynamics(
                        spectrum,
                        ETA,
                        theta,
                        INITIAL_BATCH,
                        SIGMA2,
                        horizon,
                        row_mass_cap=ROW_MASS_CAP,
                    )
                    for spectrum in spectra
                ]
                finite_total = np.stack([item.noisy_centered for item in finite])
                finite_gap = np.stack([item.noise_gap for item in finite])
                per_w_total = np.asarray(
                    [_relative_l2(values, de.noisy_centered) for values in finite_total]
                )
                per_w_gap = np.asarray(
                    [_relative_l2(values, de.noise_gap) for values in finite_gap]
                )
                for seed, total_error, gap_error, trajectory in zip(
                    FEATURE_SEEDS, per_w_total, per_w_gap, finite, strict=True
                ):
                    per_w_rows.append(
                        {
                            "regime": regime.name,
                            "alpha": regime.alpha,
                            "beta": regime.beta,
                            "width": width,
                            "feature_seed": seed,
                            "theta": theta,
                            "response": response,
                            "total_relative_l2": float(total_error),
                            "gap_relative_l2": float(gap_error),
                            "maximum_row_mass": trajectory.maximum_row_mass,
                            "maximum_pointwise_product": (
                                trajectory.maximum_pointwise_product
                            ),
                            "pointwise_stable": trajectory.pointwise_stable,
                            "row_stable": trajectory.row_stable,
                        }
                    )

                median_total = np.median(finite_total, axis=0)
                median_gap = np.median(finite_gap, axis=0)
                q10_total, q90_total = np.quantile(finite_total, [0.1, 0.9], axis=0)
                q10_gap, q90_gap = np.quantile(finite_gap, [0.1, 0.9], axis=0)
                for index, intrinsic_time in enumerate(de.times):
                    curve_rows.append(
                        {
                            "regime": regime.name,
                            "alpha": regime.alpha,
                            "beta": regime.beta,
                            "q_clean": regime.q_clean,
                            "q_kernel": regime.q_kernel,
                            "width": width,
                            "ambient_dimension": AMBIENT_TO_WIDTH_RATIO * width,
                            "theta": theta,
                            "response": response,
                            "iteration": index,
                            "intrinsic_time": float(intrinsic_time),
                            "batch": (
                                int(de.batches[index]) if index < horizon else ""
                            ),
                            "de_clean_centered": float(de.clean_centered[index]),
                            "de_noise_gap": float(de.noise_gap[index]),
                            "de_noisy_centered": float(de.noisy_centered[index]),
                            "finite_median_noisy_centered": float(median_total[index]),
                            "finite_q10_noisy_centered": float(q10_total[index]),
                            "finite_q90_noisy_centered": float(q90_total[index]),
                            "finite_median_noise_gap": float(median_gap[index]),
                            "finite_q10_noise_gap": float(q10_gap[index]),
                            "finite_q90_noise_gap": float(q90_gap[index]),
                        }
                    )

                total_quantiles = np.quantile(per_w_total, [0.1, 0.5, 0.9])
                gap_quantiles = np.quantile(per_w_gap, [0.1, 0.5, 0.9])
                metric_rows.append(
                    {
                        "regime": regime.name,
                        "alpha": regime.alpha,
                        "beta": regime.beta,
                        "width": width,
                        "ambient_dimension": AMBIENT_TO_WIDTH_RATIO * width,
                        "feature_realizations": len(FEATURE_SEEDS),
                        "theta": theta,
                        "response": response,
                        "eta": ETA,
                        "sigma2": SIGMA2,
                        "initial_batch": INITIAL_BATCH,
                        "horizon": horizon,
                        "maximum_intrinsic_time": float(de.times[-1]),
                        "aggregate_total_relative_l2": _relative_l2(
                            median_total, de.noisy_centered
                        ),
                        "aggregate_gap_relative_l2": _relative_l2(
                            median_gap, de.noise_gap
                        ),
                        "q10_per_w_total_relative_l2": float(total_quantiles[0]),
                        "median_per_w_total_relative_l2": float(total_quantiles[1]),
                        "q90_per_w_total_relative_l2": float(total_quantiles[2]),
                        "q10_per_w_gap_relative_l2": float(gap_quantiles[0]),
                        "median_per_w_gap_relative_l2": float(gap_quantiles[1]),
                        "q90_per_w_gap_relative_l2": float(gap_quantiles[2]),
                        "de_resolution_total_relative_l2": _relative_l2(
                            de_coarse.noisy_centered, de.noisy_centered
                        ),
                        "de_resolution_gap_relative_l2": _relative_l2(
                            de_coarse.noise_gap, de.noise_gap
                        ),
                        "de_maximum_row_mass": de.maximum_row_mass,
                        "de_maximum_pointwise_product": (
                            de.maximum_pointwise_product
                        ),
                        "de_pointwise_stable": de.pointwise_stable,
                        "de_row_stable": de.row_stable,
                        "all_finite_w_pointwise_stable": all(
                            item.pointwise_stable for item in finite
                        ),
                        "all_finite_w_row_stable": all(
                            item.row_stable for item in finite
                        ),
                    }
                )
                print(
                    f"{regime.name} m={width} {response}: "
                    f"median total={total_quantiles[1]:.3%}, "
                    f"gap={gap_quantiles[1]:.3%}, "
                    f"DE resolution={metric_rows[-1]['de_resolution_total_relative_l2']:.3%}",
                    flush=True,
                )

    _write_csv(OUTPUT_DIR / "curves.csv", curve_rows)
    _write_csv(OUTPUT_DIR / "metrics.csv", metric_rows)
    _write_csv(OUTPUT_DIR / "per_w_metrics.csv", per_w_rows)
    _write_csv(OUTPUT_DIR / "spectral_diagnostics.csv", spectral_rows)
    curves_png, curves_pdf = _plot_curves(curve_rows)
    errors_png, errors_pdf = _plot_errors(metric_rows)

    display_rows = [row for row in metric_rows if row["width"] == DISPLAY_WIDTH]
    maximum_resolution = max(
        max(
            float(row["de_resolution_total_relative_l2"]),
            float(row["de_resolution_gap_relative_l2"]),
        )
        for row in metric_rows
    )
    maximum_display_median = max(
        max(
            float(row["median_per_w_total_relative_l2"]),
            float(row["median_per_w_gap_relative_l2"]),
        )
        for row in display_rows
    )
    maximum_display_aggregate = max(
        max(
            float(row["aggregate_total_relative_l2"]),
            float(row["aggregate_gap_relative_l2"]),
        )
        for row in display_rows
    )
    numerical_contract_pass = bool(
        maximum_resolution <= MAXIMUM_DE_RESOLUTION_RELATIVE_L2
        and max(float(row["max_m_residual"]) for row in spectral_rows) <= 5.0e-11
        and max(float(row["trace_mass_identity_error"]) for row in spectral_rows)
        <= 1.0e-10
        and max(float(row["target_mass_identity_error"]) for row in spectral_rows)
        <= 1.0e-10
        and all(bool(row["de_pointwise_stable"]) for row in metric_rows)
        and all(bool(row["de_row_stable"]) for row in metric_rows)
        and all(bool(row["all_finite_w_pointwise_stable"]) for row in metric_rows)
        and all(bool(row["all_finite_w_row_stable"]) for row in metric_rows)
    )
    bridge_gate_pass = bool(
        numerical_contract_pass
        and maximum_display_median
        <= MAXIMUM_DISPLAY_MEDIAN_PER_W_RELATIVE_L2
        and maximum_display_aggregate
        <= MAXIMUM_DISPLAY_AGGREGATE_RELATIVE_L2
    )
    summary = {
        "experiment": "experiment_0b_finite_w_de_bridge",
        "paper_modified": False,
        "workflow_promoted": False,
        "purpose": (
            "validate the random-feature deterministic-equivalent Volterra "
            "against exact conditional finite-W Volterra under the unchanged "
            "Experiment 0 schedules"
        ),
        "same_object_contract": {
            "ambient_to_width_ratio": AMBIENT_TO_WIDTH_RATIO,
            "widths": list(WIDTHS),
            "feature_seeds": list(FEATURE_SEEDS),
            "eta": ETA,
            "initial_batch": INITIAL_BATCH,
            "sigma2": SIGMA2,
            "schedule": "B_t=ceil(B0*(1+eta*t)^theta)",
            "time": "T_t=eta*t",
            "de_time_dynamics": (
                "same exact discrete varying-batch q_t and injection as finite-W"
            ),
            "de_spectrum": (
                "bundled support-adaptive zero-regularization real-axis "
                "resolvent deterministic equivalent, numerically inherited "
                "from the audited ws_443 backend"
            ),
        },
        "gates": {
            "maximum_de_resolution_relative_l2": (
                MAXIMUM_DE_RESOLUTION_RELATIVE_L2
            ),
            "maximum_display_median_per_w_relative_l2": (
                MAXIMUM_DISPLAY_MEDIAN_PER_W_RELATIVE_L2
            ),
            "maximum_display_aggregate_relative_l2": (
                MAXIMUM_DISPLAY_AGGREGATE_RELATIVE_L2
            ),
            "numerical_contract_pass": numerical_contract_pass,
            "bridge_gate_pass": bridge_gate_pass,
        },
        "aggregate_results": {
            "maximum_de_resolution_relative_l2": maximum_resolution,
            "maximum_display_median_per_w_relative_l2": maximum_display_median,
            "maximum_display_aggregate_relative_l2": maximum_display_aggregate,
            "maximum_m_residual": max(
                float(row["max_m_residual"]) for row in spectral_rows
            ),
            "maximum_de_row_mass": max(
                float(row["de_maximum_row_mass"]) for row in metric_rows
            ),
            "maximum_finite_w_row_mass": max(
                float(row["maximum_row_mass"]) for row in per_w_rows
            ),
        },
        "claim_scope": (
            "finite-width numerical validation of the conditional-to-DE bridge "
            "on the audited schedules and time windows; not a uniform-in-time "
            "random-matrix theorem and not an asymptotic exponent test"
        ),
        "artifacts": {
            "curves": str(OUTPUT_DIR / "curves.csv"),
            "metrics": str(OUTPUT_DIR / "metrics.csv"),
            "per_w_metrics": str(OUTPUT_DIR / "per_w_metrics.csv"),
            "spectral_diagnostics": str(OUTPUT_DIR / "spectral_diagnostics.csv"),
            "curves_png": str(curves_png),
            "curves_pdf": str(curves_pdf),
            "errors_png": str(errors_png),
            "errors_pdf": str(errors_pdf),
        },
    }
    _write_json(OUTPUT_DIR / "summary.json", summary)
    return summary


if __name__ == "__main__":
    print(json.dumps(run(), indent=2, sort_keys=True))
