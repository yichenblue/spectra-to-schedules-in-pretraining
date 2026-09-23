#!/usr/bin/env python3
"""Preview Figure 5 with the regime map and two empirical panels in one row."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import LogNorm
from matplotlib.lines import Line2D
from matplotlib.patches import Polygon, Rectangle
from matplotlib.ticker import FixedLocator, LogFormatterMathtext, NullLocator

import plot_full_adaptive_sigma0_sigma1_overlay as overlay
import plot_full_adaptive_volterra_all_d as full
import plot_representative_subregime_overlays as representative


WORKSTREAM = Path(__file__).resolve().parents[1]
PLOTS = WORKSTREAM / "artifacts" / "plots"

SIGMA2_VALUES = (0.0, 1.0)
COMPOSITE_FRONTIER_INTERVALS = {
    "FB2": (1.0e7, 1.0e9),
    "IM1": (1.0e6, 1.0e8),
}
FIGURE_SIZE = (7.35, 2.92)
FIGURE_DPI = 320


def xcolor_mix(base: tuple[float, float, float], percent: float) -> tuple[float, float, float]:
    """Match xcolor's ``base!percent`` mixing with white."""
    weight = percent / 100.0
    return tuple((1.0 - weight) + weight * channel for channel in base)


def draw_regime_map(axis: plt.Axes) -> None:
    """Use the exact phase geometry and visual hierarchy of main-text Figure 2."""
    purple = (0.75, 0.0, 0.25)
    orange = (1.0, 0.5, 0.0)
    teal = (0.0, 0.5, 0.5)
    gray = (0.5, 0.5, 0.5)

    # Excluded non-finite-energy region.
    axis.add_patch(
        Polygon(
            [(-0.3, 0.0), (0.5, 0.0), (-0.3, 0.8)],
            closed=True,
            facecolor=xcolor_mix(gray, 22),
            edgecolor="none",
        )
    )

    # One hue family per propagation regime; shades separate subregimes.
    axis.add_patch(
        Polygon(
            [(0.5, 0.0), (0.25, 0.25), (0.5, 0.25)],
            closed=True,
            facecolor=xcolor_mix(purple, 27),
            edgecolor="none",
        )
    )
    axis.add_patch(
        Rectangle(
            (0.5, 0.0),
            0.7,
            0.25,
            facecolor=xcolor_mix(purple, 11),
            edgecolor="none",
        )
    )
    axis.add_patch(
        Polygon(
            [(0.25, 0.25), (0.0, 0.5), (0.5, 0.5)],
            closed=True,
            facecolor=xcolor_mix(orange, 34),
            edgecolor="none",
        )
    )
    axis.add_patch(
        Polygon(
            [(0.25, 0.25), (0.5, 0.5), (0.5, 0.25)],
            closed=True,
            facecolor=xcolor_mix(orange, 22),
            edgecolor="none",
        )
    )
    axis.add_patch(
        Rectangle(
            (0.5, 0.25),
            0.7,
            0.25,
            facecolor=xcolor_mix(orange, 10),
            edgecolor="none",
        )
    )
    axis.add_patch(
        Polygon(
            [(0.0, 0.5), (-0.3, 0.8), (-0.3, 1.2), (0.5, 1.2), (0.5, 0.5)],
            closed=True,
            facecolor=xcolor_mix(teal, 31),
            edgecolor="none",
        )
    )
    axis.add_patch(
        Polygon(
            [(0.5, 0.5), (1.2, 1.2), (0.5, 1.2)],
            closed=True,
            facecolor=xcolor_mix(teal, 19),
            edgecolor="none",
        )
    )
    axis.add_patch(
        Polygon(
            [(0.5, 0.5), (1.2, 0.5), (1.2, 1.2)],
            closed=True,
            facecolor=xcolor_mix(teal, 8),
            edgecolor="none",
        )
    )

    # Structural boundaries and finite-energy boundary: exact Figure 2 widths.
    axis.plot([0.25, 1.2], [0.25, 0.25], color="black", linewidth=1.18)
    axis.plot([0.0, 1.2], [0.5, 0.5], color="black", linewidth=1.18)
    axis.plot(
        [0.5, 0.5],
        [0.0, 1.2],
        color="black",
        linewidth=0.72,
        linestyle=(0, (3, 2)),
    )
    axis.plot(
        [0.25, 1.2],
        [0.25, 1.2],
        color="black",
        linewidth=0.72,
        linestyle=(0, (3, 2)),
    )
    axis.plot([-0.3, 0.5], [0.8, 0.0], color="#9c2027", linewidth=1.05)

    # Axes are drawn explicitly to match the arrowed TikZ axes in Figure 2.
    axis.annotate(
        "",
        xy=(1.27, 0.0),
        xytext=(-0.33, 0.0),
        arrowprops={"arrowstyle": "-|>", "linewidth": 0.82, "color": "black"},
        zorder=10,
    )
    axis.annotate(
        "",
        xy=(0.0, 1.25),
        xytext=(0.0, 0.0),
        arrowprops={"arrowstyle": "-|>", "linewidth": 0.82, "color": "black"},
        zorder=10,
    )
    axis.plot([0.5, 0.5], [-0.015, 0.015], color="black", linewidth=0.7)

    label_style = {"fontsize": 7.0, "fontweight": "bold"}
    axis.text(0.405, 0.173, r"$\mathbf{FB}_1$", ha="center", va="center", **label_style)
    axis.text(0.93, 0.125, r"$\mathbf{FB}_2$", ha="center", va="center", **label_style)
    axis.text(0.25, 0.405, r"$\mathbf{LM}_1$", ha="center", va="center", **label_style)
    axis.text(0.415, 0.33, r"$\mathbf{LM}_2$", ha="center", va="center", fontsize=6.6, fontweight="bold")
    axis.text(0.93, 0.375, r"$\mathbf{LM}_3$", ha="center", va="center", **label_style)
    axis.text(0.08, 0.82, r"$\mathbf{IM}_1$", ha="center", va="center", **label_style)
    axis.text(0.72, 0.84, r"$\mathbf{IM}_2$", ha="center", va="center", **label_style)
    axis.text(1.04, 0.78, r"$\mathbf{IM}_3$", ha="center", va="center", **label_style)
    axis.text(-0.14, 0.25, r"$p\leq 0$", ha="center", va="center", fontsize=7.0, color="#707070")

    axis.text(1.295, 0.0, r"$\mathbf{\beta}$", ha="left", va="center", fontsize=7.4)
    axis.text(0.0, 1.275, r"$\mathbf{\alpha}$", ha="center", va="bottom", fontsize=7.4)
    axis.text(0.5, -0.045, r"$\mathbf{1/2}$", ha="center", va="top", fontsize=7.0)
    axis.text(1.22, 0.25, r"$\mathbf{1/4}$", ha="left", va="center", fontsize=7.0)
    axis.text(1.22, 0.5, r"$\mathbf{1/2}$", ha="left", va="center", fontsize=7.0)
    axis.text(1.22, 1.0, r"$\mathbf{1}$", ha="left", va="center", fontsize=7.0)

    axis.set_xlim(-0.36, 1.36)
    axis.set_ylim(-0.06, 1.31)
    axis.set_aspect("auto")
    axis.axis("off")
    axis.set_title("(a) PLRF pullback", fontsize=8.8, fontweight="bold", pad=5)


def draw_empirical_panel(
    axis: plt.Axes,
    subregime: str,
    panel_label: str,
    curves: dict[tuple[str, float, int], dict[str, Any]],
    frontiers: dict[float, dict[str, Any]],
    show_ylabel: bool,
) -> tuple[dict[float, Any], LogNorm]:
    specification = representative.SUBREGIMES[subregime]
    legacy_phase = str(specification["legacy_phase"])
    displayed_widths = overlay.PLOTTED_D_VALUES
    norm = LogNorm(vmin=min(displayed_widths), vmax=max(displayed_widths))
    colormaps = {
        0.0: overlay.truncated_colormap("Blues"),
        1.0: overlay.truncated_colormap("Oranges"),
    }

    for width in reversed(displayed_widths):
        for sigma2 in SIGMA2_VALUES:
            curve = curves[(legacy_phase, sigma2, width)]
            axis.plot(
                curve["flops"],
                curve["centered_loss"],
                color=colormaps[sigma2](norm(width)),
                linewidth=1.10 if sigma2 == 0.0 else 1.05,
                alpha=0.34 if sigma2 == 0.0 else 0.84,
                solid_capstyle="round",
                rasterized=True,
                zorder=1 if sigma2 == 0.0 else 2,
            )

    for sigma2 in SIGMA2_VALUES:
        frontier = frontiers[sigma2]
        frontier_flops_min, frontier_flops_max = COMPOSITE_FRONTIER_INTERVALS[
            subregime
        ]
        mask = (
            (frontier["flops"] >= frontier_flops_min)
            & (frontier["flops"] <= frontier_flops_max)
        )
        displayed_loss = representative.displayed_frontier_loss(
            sigma2, frontier["centered_loss"]
        )
        frontier_color = colormaps[sigma2](norm(max(displayed_widths)))
        axis.plot(
            frontier["flops"][mask],
            displayed_loss[mask],
            color="white",
            linewidth=3.2,
            zorder=8,
        )
        axis.plot(
            frontier["flops"][mask],
            displayed_loss[mask],
            color=frontier_color,
            linewidth=2.15,
            zorder=9,
        )

    axis.set_xscale("log")
    axis.set_yscale("log")
    axis.set_xlim(1.0e2, 1.0e12)
    axis.set_xlabel("FLOPs", fontsize=8.2, labelpad=1.5)
    if show_ylabel:
        axis.set_ylabel("loss", fontsize=8.2, labelpad=1.5)
    axis.grid(True, which="major", linewidth=0.40, alpha=0.30)
    axis.grid(True, which="minor", linewidth=0.25, alpha=0.13)
    axis.tick_params(axis="both", which="both", labelsize=7.0, length=2.2, pad=1.5)
    if subregime == "FB2":
        axis.yaxis.set_major_locator(FixedLocator([1.0]))
        axis.yaxis.set_major_formatter(LogFormatterMathtext(base=10))
        axis.yaxis.set_minor_locator(NullLocator())
    axis.set_title(
        rf"({panel_label}) $\mathrm{{{subregime[:-1]}}}_{{{subregime[-1]}}}$: "
        rf"$\alpha={specification['alpha']:g},\ \beta={specification['beta']:g}$",
        fontsize=8.8,
        fontweight="bold",
        pad=5,
    )
    return colormaps, norm


def run() -> dict[str, str]:
    full.validate_inputs()
    curves = full.prepare_curves()
    frontiers = {
        subregime: representative.compute_full_sweep_frontiers(
            curves, str(specification["legacy_phase"])
        )
        for subregime, specification in representative.SUBREGIMES.items()
    }

    figure = plt.figure(figsize=FIGURE_SIZE, constrained_layout=False)
    outer = figure.add_gridspec(
        1,
        3,
        width_ratios=(1.0, 1.0, 1.0),
        left=0.055,
        right=0.905,
        bottom=0.285,
        top=0.855,
        wspace=0.34,
    )
    map_axis = figure.add_subplot(outer[0, 0])
    fb_axis = figure.add_subplot(outer[0, 1])
    im_axis = figure.add_subplot(outer[0, 2])

    # Keep all three main axes exactly equal; place the two narrow colorbars
    # outside panel (c), so they do not reduce its plotting width.
    im_position = im_axis.get_position()
    clean_colorbar_axis = figure.add_axes(
        (im_position.x1 + 0.010, im_position.y0, 0.008, im_position.height)
    )
    noisy_colorbar_axis = figure.add_axes(
        (im_position.x1 + 0.043, im_position.y0, 0.008, im_position.height)
    )

    draw_regime_map(map_axis)
    fb_colormaps, norm = draw_empirical_panel(
        fb_axis, "FB2", "b", curves, frontiers["FB2"], show_ylabel=True
    )
    im_colormaps, _ = draw_empirical_panel(
        im_axis, "IM1", "c", curves, frontiers["IM1"], show_ylabel=False
    )

    displayed_widths = overlay.PLOTTED_D_VALUES
    for sigma2, colorbar_axis in (
        (0.0, clean_colorbar_axis),
        (1.0, noisy_colorbar_axis),
    ):
        colorbar = figure.colorbar(
            ScalarMappable(norm=norm, cmap=im_colormaps[sigma2]),
            cax=colorbar_axis,
            orientation="vertical",
            ticks=[min(displayed_widths), max(displayed_widths)],
        )
        if sigma2 == 0.0:
            colorbar.ax.set_yticklabels([])
            colorbar.ax.tick_params(length=1.8)
        else:
            colorbar.ax.set_yticklabels(
                [f"{min(displayed_widths):,}", "51k"], fontsize=6.1
            )
            colorbar.ax.tick_params(length=1.8, pad=1.0)
        colorbar.ax.set_title(rf"${sigma2:g}$", fontsize=6.1, pad=3)

    middle_width = overlay.PLOTTED_D_VALUES[len(overlay.PLOTTED_D_VALUES) // 2]
    legend_handles = [
        Line2D(
            [0],
            [0],
            color=fb_colormaps[sigma2](norm(middle_width)),
            linewidth=1.35,
            alpha=0.55 if sigma2 == 0.0 else 0.95,
            label="clean loss" if sigma2 == 0.0 else "noisy loss",
        )
        for sigma2 in SIGMA2_VALUES
    ] + [
        Line2D(
            [0],
            [0],
            color=fb_colormaps[sigma2](norm(max(overlay.PLOTTED_D_VALUES))),
            linewidth=2.3,
            label=(
                "clean compute-optimal"
                if sigma2 == 0.0
                else "noisy compute-optimal"
            ),
        )
        for sigma2 in SIGMA2_VALUES
    ]
    figure.legend(
        handles=legend_handles,
        loc="lower center",
        bbox_to_anchor=(0.64, 0.025),
        ncol=2,
        frameon=False,
        fontsize=7.0,
        handlelength=2.5,
        columnspacing=1.25,
        handletextpad=0.55,
    )

    PLOTS.mkdir(parents=True, exist_ok=True)
    png_output = PLOTS / "figure5_three_panel_horizontal_preview.png"
    pdf_output = PLOTS / "figure5_three_panel_horizontal_preview.pdf"
    figure.savefig(png_output, dpi=FIGURE_DPI)
    figure.savefig(pdf_output)
    plt.close(figure)
    return {"png": str(png_output), "pdf": str(pdf_output)}


if __name__ == "__main__":
    print(json.dumps(run(), indent=2, sort_keys=True))
