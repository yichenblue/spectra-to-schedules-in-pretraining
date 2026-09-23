#!/usr/bin/env python3
"""Overlay sigma^2=0 and sigma^2=1 on the full accepted width sweep."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import LinearSegmentedColormap, LogNorm
from matplotlib.lines import Line2D

import plot_full_adaptive_volterra_all_d as full


WORKSTREAM = Path(__file__).resolve().parents[1]
RESULTS = WORKSTREAM / "artifacts" / "results"
PLOTS = WORKSTREAM / "artifacts" / "plots"
OUTPUT = (
    PLOTS
    / "full_adaptive_volterra_all_d_centered_sigma2_0_vs_1_overlay.png"
)
OUTPUT_MANIFEST = (
    RESULTS
    / "full_adaptive_volterra_all_d_sigma2_0_vs_1_overlay_manifest.json"
)
SIGMA_STYLES = {
    0.0: {"linestyle": "-", "linewidth": 2.05, "alpha": 0.92},
    1.0: {"linestyle": (0, (4.0, 2.2)), "linewidth": 1.9, "alpha": 0.9},
}
PLOTTED_D_VALUES = (10, 30, 75, 200, 600, 1_600, 4_800, 12_800, 25_600, 51_200)


def truncated_colormap(name: str) -> LinearSegmentedColormap:
    base = plt.get_cmap(name)
    return LinearSegmentedColormap.from_list(
        f"{name}_visible",
        [base(value) for value in (0.30, 0.43, 0.56, 0.69, 0.82, 0.95)],
    )


def plot_overlay(
    curves: dict[tuple[str, float, int], dict[str, Any]],
) -> None:
    norm = LogNorm(vmin=min(PLOTTED_D_VALUES), vmax=max(PLOTTED_D_VALUES))
    colormaps = {
        0.0: truncated_colormap("Blues"),
        1.0: truncated_colormap("Oranges"),
    }
    figure, axes = plt.subplots(
        2,
        4,
        figsize=(20.5, 10.4),
        sharex=True,
        constrained_layout=False,
    )
    flat_axes = axes.ravel()

    for phase_index, phase in enumerate(full.PHASES):
        axis = flat_axes[phase_index]
        for d in reversed(PLOTTED_D_VALUES):
            for sigma2 in (0.0, 1.0):
                curve = curves[(phase, sigma2, d)]
                style = SIGMA_STYLES[sigma2]
                axis.plot(
                    curve["flops"],
                    curve["centered_loss"],
                    color=colormaps[sigma2](norm(d)),
                    linestyle=style["linestyle"],
                    linewidth=style["linewidth"],
                    alpha=style["alpha"],
                    solid_capstyle="round",
                    rasterized=True,
                )
        alpha, beta = full.PHASE_POINTS[phase]
        axis.set_title(
            rf"{phase}: $\alpha={alpha:g},\ \beta={beta:g}$",
            fontsize=12.5,
        )
        axis.set_xscale("log")
        axis.set_yscale("log")
        axis.set_xlim(1.0e2, 1.0e12)
        axis.grid(True, which="major", linewidth=0.55, alpha=0.34)
        axis.grid(True, which="minor", linewidth=0.35, alpha=0.16)
        axis.tick_params(axis="both", which="both", labelsize=9.5)
        if phase_index in (0, 4):
            axis.set_ylabel(
                r"$R_\sigma(r,d)-R_{\sigma,\infty}^{\star}$",
                fontsize=11,
            )
        if phase_index >= 4:
            axis.set_xlabel(r"FLOPs $f=rBd$", fontsize=11)

    flat_axes[-1].axis("off")
    figure.suptitle(
        r"Centered DE Volterra curves: $\sigma^2=0$ versus $\sigma^2=1$",
        fontsize=16,
        y=0.985,
    )
    noise_handles = [
        Line2D(
            [0],
            [0],
            color=(
                colormaps[sigma2](norm(PLOTTED_D_VALUES[len(PLOTTED_D_VALUES) // 2]))
            ),
            linestyle=SIGMA_STYLES[sigma2]["linestyle"],
            linewidth=2.0,
            label=rf"$\sigma^2={sigma2:g}$",
        )
        for sigma2 in (0.0, 1.0)
    ]
    figure.legend(
        handles=noise_handles,
        loc="lower center",
        bbox_to_anchor=(0.50, 0.065),
        ncol=2,
        frameon=False,
        fontsize=10.5,
        handlelength=3.2,
    )

    for sigma2, x_position in ((0.0, 0.795), (1.0, 0.895)):
        colorbar_axis = figure.add_axes((x_position, 0.235, 0.016, 0.225))
        colorbar = figure.colorbar(
            ScalarMappable(norm=norm, cmap=colormaps[sigma2]),
            cax=colorbar_axis,
            orientation="vertical",
            ticks=PLOTTED_D_VALUES,
        )
        colorbar.ax.set_yticklabels(
            [f"{d:,}" for d in PLOTTED_D_VALUES],
            fontsize=8.0,
        )
        colorbar.set_label(
            rf"$d$ ($\sigma^2={sigma2:g}$)",
            fontsize=10.5,
            labelpad=4,
        )

    figure.subplots_adjust(
        left=0.064,
        right=0.985,
        top=0.925,
        bottom=0.15,
        wspace=0.25,
        hspace=0.25,
    )
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(OUTPUT, dpi=220, bbox_inches="tight")
    plt.close(figure)


def run() -> dict[str, Any]:
    full.validate_inputs()
    curves = full.prepare_curves()
    plot_overlay(curves)
    manifest: dict[str, Any] = {
        "status": "COMPLETE",
        "experiment": "full_adaptive_volterra_sigma2_0_vs_1_overlay",
        "solver": (
            "support-adaptive zero-regularization deterministic-equivalent "
            "Volterra"
        ),
        "risk": "R_sigma(r,d) - R_sigma,infinity^star",
        "phases": list(full.PHASES),
        "d_values": list(PLOTTED_D_VALUES),
        "sigma2_values": [0.0, 1.0],
        "flops_min": 1.0e2,
        "flops_max": 1.0e12,
        "points_per_curve": 720,
        "curve_count": len(full.PHASES) * len(PLOTTED_D_VALUES) * 2,
        "encoding": {
            "sigma2=0": "blue heatmap and solid line",
            "sigma2=1": "orange heatmap and dashed line",
            "heatmap_coordinate": "width d on a logarithmic scale",
        },
        "plot": str(OUTPUT.relative_to(WORKSTREAM)),
    }
    with OUTPUT_MANIFEST.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return manifest


if __name__ == "__main__":
    print(json.dumps(run(), indent=2, sort_keys=True))
