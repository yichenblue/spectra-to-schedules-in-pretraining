#!/usr/bin/env python3
"""Plot paper-ready FB2 and IM1 centered-risk/frontier overlays."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.cm import ScalarMappable
from matplotlib.colors import LogNorm
from matplotlib.lines import Line2D
from scipy.signal import savgol_filter

import plot_full_adaptive_sigma0_sigma1_overlay as overlay
import plot_full_adaptive_volterra_all_d as full


WORKSTREAM = Path(__file__).resolve().parents[1]
RESULTS = WORKSTREAM / "artifacts" / "results"
PLOTS = WORKSTREAM / "artifacts" / "plots"

# Current-paper propagation subregime -> legacy data label.
SUBREGIMES = {
    "FB2": {
        "legacy_phase": "Ic",
        "alpha": 0.12,
        "beta": 0.65,
        "frontier_flops_min": 1.0e6,
        "frontier_flops_max": 1.0e10,
        "show_colorbars": False,
    },
    "IM1": {
        "legacy_phase": "Ia",
        "alpha": 0.70,
        "beta": 0.30,
        "frontier_flops_min": 1.0e4,
        "frontier_flops_max": 1.0e9,
        "show_colorbars": True,
    },
}
SIGMA2_VALUES = (0.0, 1.0)
FRONTIER_SMOOTHING_WINDOW = 31
FRONTIER_SMOOTHING_POLYORDER = 3
FONT_SCALE = 4.0 / 3.0
FIGURE_SIZE = (12.0, 7.4)
FIGURE_DPI = 260


def compute_full_sweep_frontiers(
    curves: dict[tuple[str, float, int], dict[str, Any]],
    legacy_phase: str,
) -> dict[float, dict[str, np.ndarray]]:
    """Take the pointwise discrete minimum over the complete accepted width grid."""
    frontiers: dict[float, dict[str, np.ndarray]] = {}
    for sigma2 in SIGMA2_VALUES:
        reference_flops = curves[(legacy_phase, sigma2, full.D_VALUES[0])]["flops"]
        frontier_loss = np.empty_like(reference_flops)
        frontier_width = np.empty_like(reference_flops)
        for index, compute in enumerate(reference_flops):
            eligible = [width for width in full.D_VALUES if width <= compute]
            values = np.asarray(
                [
                    curves[(legacy_phase, sigma2, width)]["centered_loss"][index]
                    for width in eligible
                ],
                dtype=float,
            )
            minimum_index = int(np.argmin(values))
            frontier_loss[index] = values[minimum_index]
            frontier_width[index] = eligible[minimum_index]
        frontiers[sigma2] = {
            "flops": reference_flops,
            "centered_loss": frontier_loss,
            "m_star": frontier_width,
        }
    return frontiers


def displayed_frontier_loss(sigma2: float, centered_loss: np.ndarray) -> np.ndarray:
    """Smooth displayed clean/noisy frontiers; preserve raw values on disk."""
    return np.exp(
        savgol_filter(
            np.log(centered_loss),
            window_length=FRONTIER_SMOOTHING_WINDOW,
            polyorder=FRONTIER_SMOOTHING_POLYORDER,
            mode="interp",
        )
    )


def plot_subregime(
    subregime: str,
    specification: dict[str, Any],
    curves: dict[tuple[str, float, int], dict[str, Any]],
    frontiers: dict[float, dict[str, np.ndarray]],
) -> tuple[Path, Path]:
    legacy_phase = str(specification["legacy_phase"])
    displayed_widths = overlay.PLOTTED_D_VALUES
    norm = LogNorm(vmin=min(displayed_widths), vmax=max(displayed_widths))
    colormaps = {
        0.0: overlay.truncated_colormap("Blues"),
        1.0: overlay.truncated_colormap("Oranges"),
    }

    show_colorbars = bool(specification["show_colorbars"])
    figure = plt.figure(figsize=FIGURE_SIZE)
    axis = figure.add_axes(
        (0.10, 0.22, 0.62, 0.68)
        if show_colorbars
        else (0.10, 0.22, 0.86, 0.68)
    )
    for width in reversed(displayed_widths):
        for sigma2 in SIGMA2_VALUES:
            curve = curves[(legacy_phase, sigma2, width)]
            axis.plot(
                curve["flops"],
                curve["centered_loss"],
                color=colormaps[sigma2](norm(width)),
                linestyle="-",
                linewidth=2.75 if sigma2 == 0.0 else 2.65,
                alpha=0.38 if sigma2 == 0.0 else 0.90,
                solid_capstyle="round",
                rasterized=True,
                zorder=1 if sigma2 == 0.0 else 2,
            )

    for sigma2 in SIGMA2_VALUES:
        frontier = frontiers[sigma2]
        frontier_flops_min = float(specification["frontier_flops_min"])
        frontier_flops_max = float(specification["frontier_flops_max"])
        mask = (
            (frontier["flops"] >= frontier_flops_min)
            & (frontier["flops"] <= frontier_flops_max)
        )
        frontier_loss = displayed_frontier_loss(
            sigma2,
            frontier["centered_loss"],
        )
        frontier_color = colormaps[sigma2](norm(max(displayed_widths)))
        axis.plot(
            frontier["flops"][mask],
            frontier_loss[mask],
            color="white",
            linestyle="-",
            linewidth=8.2,
            alpha=0.95,
            zorder=8,
        )
        axis.plot(
            frontier["flops"][mask],
            frontier_loss[mask],
            color=frontier_color,
            linestyle="-",
            linewidth=5.4,
            alpha=1.0,
            zorder=9,
        )

    axis.set_title(
        rf"Subregime $\mathrm{{{subregime[:-1]}}}_{{{subregime[-1]}}}$: "
        rf"$\alpha={specification['alpha']:g},\ \beta={specification['beta']:g}$",
        fontsize=15 * FONT_SCALE,
    )
    axis.set_xscale("log")
    axis.set_yscale("log")
    axis.set_xlim(1.0e2, 1.0e12)
    axis.set_xlabel(r"FLOPs $\mathfrak{f}=tBm$", fontsize=13 * FONT_SCALE)
    axis.set_ylabel(
        r"$\mathcal{R}_\sigma(t,m)-P_\sigma^\star$",
        fontsize=13 * FONT_SCALE,
    )
    axis.grid(True, which="major", linewidth=0.6, alpha=0.34)
    axis.grid(True, which="minor", linewidth=0.35, alpha=0.16)
    axis.tick_params(axis="both", which="both", labelsize=10.5 * FONT_SCALE)

    fixed_width_handles = [
        Line2D(
            [0],
            [0],
            color=colormaps[sigma2](norm(displayed_widths[len(displayed_widths) // 2])),
            linestyle="-",
            linewidth=2.7,
            alpha=0.5 if sigma2 == 0.0 else 1.0,
            label=rf"fixed $m$, $\sigma^2={sigma2:g}$",
        )
        for sigma2 in SIGMA2_VALUES
    ]
    frontier_handles = [
        Line2D(
            [0],
            [0],
            color=colormaps[sigma2](norm(max(displayed_widths))),
            linestyle="-",
            linewidth=5.4,
            label=rf"full-sweep frontier, $\sigma^2={sigma2:g}$",
        )
        for sigma2 in SIGMA2_VALUES
    ]
    figure.legend(
        handles=fixed_width_handles + frontier_handles,
        loc="lower center",
        bbox_to_anchor=(0.50, 0.015),
        ncol=2,
        frameon=False,
        fontsize=11.5 * FONT_SCALE,
        handlelength=3.2,
    )

    if show_colorbars:
        for sigma2, x_position in ((0.0, 0.78), (1.0, 0.90)):
            colorbar_axis = figure.add_axes((x_position, 0.30, 0.025, 0.47))
            colorbar = figure.colorbar(
                ScalarMappable(norm=norm, cmap=colormaps[sigma2]),
                cax=colorbar_axis,
                orientation="vertical",
                ticks=displayed_widths,
            )
            colorbar.ax.set_yticklabels(
                [f"{width:,}" for width in displayed_widths],
                fontsize=8.5 * FONT_SCALE,
            )
            colorbar.set_label(
                rf"$m$ ($\sigma^2={sigma2:g}$)",
                fontsize=11 * FONT_SCALE,
                labelpad=4,
            )

    stem = f"representative_{subregime}_centered_sigma2_0_vs_1_frontier"
    png_output = PLOTS / f"{stem}.png"
    pdf_output = PLOTS / f"{stem}.pdf"
    png_output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(png_output, dpi=FIGURE_DPI)
    figure.savefig(pdf_output)
    plt.close(figure)
    return png_output, pdf_output


def write_frontier_csv(
    subregime: str,
    frontiers: dict[float, dict[str, np.ndarray]],
    frontier_flops_min: float,
    frontier_flops_max: float,
) -> Path:
    output = RESULTS / f"representative_{subregime}_frontiers.csv"
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "subregime",
                "sigma2",
                "flops",
                "raw_centered_loss",
                "displayed_centered_loss",
                "m_star",
                "display_smoothing",
            ),
        )
        writer.writeheader()
        for sigma2, frontier in frontiers.items():
            displayed_loss = displayed_frontier_loss(
                sigma2,
                frontier["centered_loss"],
            )
            for flops, raw_loss, shown_loss, width in zip(
                frontier["flops"],
                frontier["centered_loss"],
                displayed_loss,
                frontier["m_star"],
                strict=True,
            ):
                if not frontier_flops_min <= flops <= frontier_flops_max:
                    continue
                writer.writerow(
                    {
                        "subregime": subregime,
                        "sigma2": sigma2,
                        "flops": float(flops),
                        "raw_centered_loss": float(raw_loss),
                        "displayed_centered_loss": float(shown_loss),
                        "m_star": int(width),
                        "display_smoothing": "log-space Savitzky-Golay(31,3)",
                    }
                )
    return output


def run() -> dict[str, Any]:
    full.validate_inputs()
    curves = full.prepare_curves()
    outputs: dict[str, Any] = {}
    for subregime, specification in SUBREGIMES.items():
        legacy_phase = str(specification["legacy_phase"])
        if full.PHASE_POINTS[legacy_phase] != (
            specification["alpha"],
            specification["beta"],
        ):
            raise RuntimeError(f"subregime/data mismatch for {subregime}")
        frontiers = compute_full_sweep_frontiers(curves, legacy_phase)
        png_output, pdf_output = plot_subregime(
            subregime,
            specification,
            curves,
            frontiers,
        )
        frontier_output = write_frontier_csv(
            subregime,
            frontiers,
            float(specification["frontier_flops_min"]),
            float(specification["frontier_flops_max"]),
        )
        outputs[subregime] = {
            "legacy_data_phase": legacy_phase,
            "alpha": specification["alpha"],
            "beta": specification["beta"],
            "displayed_frontier_flops_interval": [
                specification["frontier_flops_min"],
                specification["frontier_flops_max"],
            ],
            "show_colorbars": specification["show_colorbars"],
            "png": str(png_output.relative_to(WORKSTREAM)),
            "pdf": str(pdf_output.relative_to(WORKSTREAM)),
            "frontier_csv": str(frontier_output.relative_to(WORKSTREAM)),
        }

    manifest: dict[str, Any] = {
        "status": "COMPLETE",
        "experiment": "representative_FB2_IM1_centered_frontier_overlays",
        "solver": (
            "support-adaptive zero-regularization deterministic-equivalent Volterra"
        ),
        "risk": "R_sigma(t,m) - P_sigma^star",
        "sigma2_values": list(SIGMA2_VALUES),
        "displayed_m_values": list(overlay.PLOTTED_D_VALUES),
        "font_scale_relative_to_previous_version": FONT_SCALE,
        "matched_canvas_inches": list(FIGURE_SIZE),
        "matched_canvas_dpi": FIGURE_DPI,
        "frontier_m_values": list(full.D_VALUES),
        "frontier_definition": (
            "pointwise discrete minimum over the full accepted width sweep, "
            "subject to m <= flops/B with B=1"
        ),
        "frontier_display_smoothing": (
            "log-space Savitzky-Golay(window=31, polyorder=3); raw frontier retained"
        ),
        "outputs": outputs,
    }
    manifest_output = RESULTS / "representative_FB2_IM1_plot_manifest.json"
    with manifest_output.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return manifest


if __name__ == "__main__":
    print(json.dumps(run(), indent=2, sort_keys=True))
