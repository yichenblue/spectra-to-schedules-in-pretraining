#!/usr/bin/env python3
"""Render the paper-ready finite-width early-stopping trajectory triptych.

Panels (a)--(c) show the sigma^2=1 centered-risk trajectories across five
widths in each integrable-memory subregime.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.cm import ScalarMappable
from matplotlib.colors import LogNorm
from matplotlib.ticker import (
    FixedFormatter,
    FixedLocator,
    LogFormatterMathtext,
    NullFormatter,
    NullLocator,
)
from scipy.interpolate import CubicSpline
from scipy.optimize import minimize_scalar

import plot_im_triptych as trajectory_figure


ROOT = Path(__file__).resolve().parents[3]
FIGURES = ROOT / "reproduced_figures"
OUTPUT_STEM = FIGURES / "fixed_noise_im_early_stopping_composite"
OUTPUT_MANIFEST = FIGURES / "fixed_noise_im_early_stopping_composite_manifest.json"

FINITE_WIDTH = 51_200
DISPLAYED_WIDTHS = trajectory_figure.DISPLAYED_WIDTHS
SIGMA2_TRAJECTORIES = 1.0
T_MIN = trajectory_figure.T_MIN
T_MAX = trajectory_figure.T_MAX
T_POINTS = trajectory_figure.T_POINTS

PHASES = (
    {
        "legacy_key": "Ia",
        "paper_name": r"\mathrm{IM}_1",
        "alpha": 0.7,
        "beta": 0.3,
    },
    {
        "legacy_key": "II",
        "paper_name": r"\mathrm{IM}_2",
        "alpha": 0.7,
        "beta": 0.6,
    },
    {
        "legacy_key": "III",
        "paper_name": r"\mathrm{IM}_3",
        "alpha": 0.6,
        "beta": 0.7,
    },
)


def _configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 6.8,
            "axes.titlesize": 7.8,
            "axes.labelsize": 7.6,
            "xtick.labelsize": 6.5,
            "ytick.labelsize": 6.5,
            "legend.fontsize": 5.9,
            "axes.linewidth": 0.72,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def _refine_minimum(
    training_time: np.ndarray,
    centered_risk: np.ndarray,
    half_window: int,
) -> tuple[float, float]:
    index = int(np.argmin(centered_risk))
    if index < half_window or index >= centered_risk.size - half_window:
        raise RuntimeError("minimum is not interior to the source grid")
    local = slice(index - half_window, index + half_window + 1)
    spline = CubicSpline(np.log(training_time[local]), centered_risk[local])
    result = minimize_scalar(
        lambda value: float(spline(float(value))),
        bounds=(
            float(np.log(training_time[index - 1])),
            float(np.log(training_time[index + 1])),
        ),
        method="bounded",
        options={"xatol": 1.0e-12, "maxiter": 300},
    )
    if not result.success:
        raise RuntimeError(f"local minimum refinement failed: {result.message}")
    return float(np.exp(result.x)), float(result.fun)


def _refined_minimum_with_gate(
    training_time: np.ndarray,
    centered_risk: np.ndarray,
) -> tuple[float, float, float]:
    estimates = [
        _refine_minimum(training_time, centered_risk, half_window)
        for half_window in (2, 3, 4)
    ]
    primary_time, primary_risk = estimates[1]
    times = np.asarray([estimate[0] for estimate in estimates], dtype=float)
    relative_spread = float((times.max() - times.min()) / primary_time)
    if relative_spread > 1.0e-4:
        raise RuntimeError(
            f"minimum refinement is unstable: relative spread={relative_spread}"
        )
    return primary_time, primary_risk, relative_spread


def _style_log_axis(axis: plt.Axes) -> None:
    axis.grid(True, which="major", linewidth=0.38, alpha=0.18)
    axis.grid(False, which="minor")
    axis.tick_params(axis="both", which="major", length=2.6, width=0.65, pad=1.8)
    axis.tick_params(axis="both", which="minor", length=1.5, width=0.45)
    axis.xaxis.set_minor_formatter(NullFormatter())
    axis.yaxis.set_minor_formatter(NullFormatter())


def render() -> dict[str, object]:
    curves = trajectory_figure._read_curves()
    display_time = np.geomspace(T_MIN, T_MAX, T_POINTS)

    _configure_style()
    width_norm = LogNorm(vmin=min(DISPLAYED_WIDTHS), vmax=max(DISPLAYED_WIDTHS))
    width_cmap = trajectory_figure._truncated_colormap("viridis", 0.08, 0.88)

    # The bundled ICLR template fixes \textwidth=5.5in.  Saving without a
    # tight bounding box keeps the vector artifact at exactly that width, so
    # the type sizes below are the final in-paper sizes rather than downscaled
    # preview sizes.
    figure = plt.figure(figsize=(5.50, 2.25), constrained_layout=False)
    grid = figure.add_gridspec(
        1,
        3,
        left=0.085,
        right=0.900,
        bottom=0.255,
        top=0.815,
        wspace=0.12,
    )
    trajectory_axes = [figure.add_subplot(grid[0, index]) for index in range(3)]

    refined_manifest: dict[str, object] = {}
    maximum_refinement_spread = 0.0
    panel_letters = ("a", "b", "c")
    for axis_index, (axis, phase, panel_letter) in enumerate(
        zip(trajectory_axes, PHASES, panel_letters, strict=True)
    ):
        legacy_key = str(phase["legacy_key"])
        phase_minima: dict[str, object] = {}
        for width in DISPLAYED_WIDTHS:
            curve = curves[(legacy_key, width)]
            displayed_risk = trajectory_figure._interpolate_curve(
                curve["training_time"],
                curve["centered_risk"],
                display_time,
            )
            minimum_time, minimum_risk, relative_spread = (
                _refined_minimum_with_gate(
                    curve["training_time"], curve["centered_risk"]
                )
            )
            maximum_refinement_spread = max(
                maximum_refinement_spread, relative_spread
            )
            color = width_cmap(width_norm(width))
            is_anchor_width = width == FINITE_WIDTH
            axis.plot(
                display_time,
                displayed_risk,
                color=color,
                linewidth=1.38 if is_anchor_width else 1.12,
                alpha=0.98 if is_anchor_width else 0.90,
                solid_capstyle="round",
                zorder=3 if is_anchor_width else 2,
            )
            axis.plot(
                minimum_time,
                minimum_risk,
                marker="o",
                markersize=3.9 if is_anchor_width else 3.35,
                markerfacecolor=color,
                markeredgecolor="white",
                markeredgewidth=0.55,
                linestyle="none",
                zorder=6,
            )
            phase_minima[str(width)] = {
                "refined_iteration": minimum_time,
                "refined_centered_risk": minimum_risk,
                "window_relative_spread": relative_spread,
            }

        highlighted_early_stopping_time = float(
            phase_minima[str(FINITE_WIDTH)]["refined_iteration"]
        )
        axis.axvline(
            highlighted_early_stopping_time,
            color="#555555",
            linewidth=0.90,
            linestyle=(0, (3.0, 2.0)),
            alpha=0.88,
            zorder=1,
        )

        axis.set_xscale("log")
        axis.set_yscale("log")
        axis.set_xlim(T_MIN, T_MAX)
        axis.set_ylim(4.0e-5, 1.05)
        trajectory_x_ticks = [1.0e1, 1.0e3, 1.0e5, 1.0e7]
        trajectory_x_labels = (
            [r"$10^1$", r"$10^3$", r"$10^5$", ""],
            ["", r"$10^3$", r"$10^5$", ""],
            ["", r"$10^3$", r"$10^5$", r"$10^7$"],
        )[axis_index]
        axis.xaxis.set_major_locator(FixedLocator(trajectory_x_ticks))
        axis.xaxis.set_major_formatter(FixedFormatter(trajectory_x_labels))
        axis.yaxis.set_major_locator(FixedLocator([1.0e-4, 1.0e-2, 1.0e0]))
        axis.yaxis.set_major_formatter(LogFormatterMathtext(base=10))
        axis.set_title(
            rf"({panel_letter}) ${phase['paper_name']}$",
            fontweight="bold",
            pad=3.0,
        )
        if axis_index > 0:
            axis.tick_params(axis="y", which="both", labelleft=False)
        _style_log_axis(axis)

        refined_manifest[str(phase["paper_name"])] = {
            "sigma2": SIGMA2_TRAJECTORIES,
            "fixed_width_refined_minima": phase_minima,
            "m51200_early_stopping_line_iteration": highlighted_early_stopping_time,
        }

    trajectory_axes[0].set_ylabel("centered risk")
    time_group_center = 0.5 * (
        trajectory_axes[0].get_position().x0
        + trajectory_axes[-1].get_position().x1
    )
    figure.text(
        time_group_center,
        0.075,
        r"training iterations $t$",
        ha="center",
        va="center",
        fontsize=7.7,
    )

    colorbar_axis = figure.add_axes([0.925, 0.310, 0.020, 0.430])
    colorbar = figure.colorbar(
        ScalarMappable(norm=width_norm, cmap=width_cmap),
        cax=colorbar_axis,
        orientation="vertical",
        ticks=[DISPLAYED_WIDTHS[0], DISPLAYED_WIDTHS[2], DISPLAYED_WIDTHS[-1]],
    )
    colorbar.ax.yaxis.set_major_locator(
        FixedLocator([DISPLAYED_WIDTHS[0], DISPLAYED_WIDTHS[2], DISPLAYED_WIDTHS[-1]])
    )
    colorbar.ax.yaxis.set_major_formatter(
        FixedFormatter(["3.2k", "12.8k", "51.2k"])
    )
    colorbar.ax.yaxis.set_minor_locator(NullLocator())
    colorbar.ax.yaxis.set_minor_formatter(NullFormatter())
    colorbar.ax.tick_params(labelsize=5.5, length=2.0, width=0.60, pad=1.2)
    colorbar.outline.set_linewidth(0.75)
    figure.text(
        0.935,
        0.775,
        r"width $m$",
        ha="center",
        va="bottom",
        fontsize=5.8,
    )

    FIGURES.mkdir(parents=True, exist_ok=True)
    png_output = OUTPUT_STEM.with_suffix(".png")
    pdf_output = OUTPUT_STEM.with_suffix(".pdf")
    figure.savefig(png_output, dpi=450)
    figure.savefig(pdf_output)
    plt.close(figure)

    manifest: dict[str, object] = {
        "status": "COMPLETE",
        "figure": "finite-width early-stopping centered-risk trajectories",
        "layout": "one row with three panels and a right-side width colorbar",
        "solver": "support-adaptive zero-regularization deterministic-equivalent Volterra",
        "primary_object": "finite-width centered-risk trajectories and their refined minima",
        "trajectory_sigma2": SIGMA2_TRAJECTORIES,
        "batch_size": trajectory_figure.BATCH_SIZE,
        "learning_rate": trajectory_figure.LEARNING_RATE,
        "ambient_to_width_ratio": trajectory_figure.AMBIENT_TO_WIDTH_RATIO,
        "displayed_widths": list(DISPLAYED_WIDTHS),
        "trajectory_panels": refined_manifest,
        "trajectory_vertical_lines": (
            "gray dashed line at each phase's refined finite-width m=51,200 "
            "early-stopping iteration"
        ),
        "minimum_refinement": (
            "cubic spline in log iteration and raw centered risk; primary "
            "half-window 3 with half-window 2/3/4 sensitivity gate"
        ),
        "maximum_minimum_window_relative_spread": maximum_refinement_spread,
        "risk": "R_sigma(t,m)-P_sigma^star (clean-target risk after noisy-label training)",
        "curve_interpolation": "linear in log(iteration)-log(centered risk), display only",
        "source_files": {
            "trajectory_curves": [
                str(path.relative_to(ROOT)) for path in trajectory_figure.CURVE_INPUTS
            ],
        },
        "outputs": {
            "png": str(png_output.relative_to(ROOT)),
            "pdf": str(pdf_output.relative_to(ROOT)),
        },
    }
    with OUTPUT_MANIFEST.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return manifest


if __name__ == "__main__":
    print(json.dumps(render(), indent=2, sort_keys=True))
