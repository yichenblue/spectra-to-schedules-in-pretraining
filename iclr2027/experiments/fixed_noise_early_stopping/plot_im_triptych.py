#!/usr/bin/env python3
"""Render the paper-ready integrable-memory early-stopping triptych.

The plotted trajectories are the accepted support-adaptive, zero-regularization
deterministic-equivalent Volterra curves from ws_443.  The legacy data keys
Ia/II/III are displayed using the manuscript taxonomy IM_1/IM_2/IM_3.
"""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.cm import ScalarMappable
from matplotlib.colors import LinearSegmentedColormap, LogNorm
from matplotlib.ticker import (
    FixedFormatter,
    FixedLocator,
    LogFormatterMathtext,
    NullFormatter,
    NullLocator,
)


ROOT = Path(__file__).resolve().parents[3]
WORKSTREAM = ROOT / "fixed_noise_validation"
RESULTS = WORKSTREAM / "artifacts" / "results"
FIGURES = ROOT / "reproduced_figures"

CURVE_INPUTS = (
    RESULTS / "adaptive_real_axis_d200_to_d3200_curves.csv",
    RESULTS / "adaptive_real_axis_d4800_d6400_d9600_curves.csv",
    RESULTS / "adaptive_real_axis_d12800_curves.csv",
    RESULTS / "adaptive_real_axis_d25600_d51200_curves.csv",
)
INFINITE_WIDTH_INPUT = RESULTS / "trace_class_infinite_plateaus.csv"

OUTPUT_STEM = FIGURES / "fixed_noise_im_early_stopping"
OUTPUT_MANIFEST = FIGURES / "fixed_noise_im_early_stopping_manifest.json"

SIGMA2 = 1.0
BATCH_SIZE = 1
LEARNING_RATE = 0.05
AMBIENT_TO_WIDTH_RATIO = 2
DISPLAYED_WIDTHS = (3_200, 6_400, 12_800, 25_600, 51_200)
T_MIN = 10.0
T_MAX = 1.0e7
T_POINTS = 1_200

SUBREGIMES = (
    {
        "legacy_key": "Ia",
        "paper_name": r"\mathrm{IM}_1",
        "alpha": 0.7,
        "beta": 0.3,
        "panel": "a",
    },
    {
        "legacy_key": "II",
        "paper_name": r"\mathrm{IM}_2",
        "alpha": 0.7,
        "beta": 0.6,
        "panel": "b",
    },
    {
        "legacy_key": "III",
        "paper_name": r"\mathrm{IM}_3",
        "alpha": 0.6,
        "beta": 0.7,
        "panel": "c",
    },
)


def _truncated_colormap(
    name: str,
    lower: float = 0.08,
    upper: float = 0.92,
) -> LinearSegmentedColormap:
    base = plt.get_cmap(name)
    colors = base(np.linspace(lower, upper, 256))
    return LinearSegmentedColormap.from_list(f"{name}_paper", colors)


def _read_curves() -> dict[tuple[str, int], dict[str, np.ndarray]]:
    legacy_keys = {str(specification["legacy_key"]) for specification in SUBREGIMES}
    widths = set(DISPLAYED_WIDTHS)
    grouped: dict[tuple[str, int], list[tuple[float, float]]] = defaultdict(list)

    for path in CURVE_INPUTS:
        if not path.is_file():
            raise FileNotFoundError(path)
        with path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                legacy_key = row["phase"]
                width = int(row["d"])
                if (
                    legacy_key not in legacy_keys
                    or width not in widths
                    or float(row["sigma2"]) != SIGMA2
                ):
                    continue
                if not np.isclose(float(row["gamma"]), LEARNING_RATE):
                    raise RuntimeError(
                        f"unexpected learning rate at {(legacy_key, width)}"
                    )
                grouped[(legacy_key, width)].append(
                    (float(row["r"]), float(row["centered_loss"]))
                )

    expected = {
        (str(specification["legacy_key"]), width)
        for specification in SUBREGIMES
        for width in DISPLAYED_WIDTHS
    }
    if set(grouped) != expected:
        raise RuntimeError(
            f"incomplete curve set: missing={sorted(expected - set(grouped))}"
        )

    curves: dict[tuple[str, int], dict[str, np.ndarray]] = {}
    for key, values in grouped.items():
        values.sort(key=lambda value: value[0])
        training_time = np.asarray([value[0] for value in values], dtype=float)
        centered_risk = np.asarray([value[1] for value in values], dtype=float)
        if not np.all(np.diff(training_time) > 0.0):
            raise RuntimeError(f"non-increasing iteration grid at {key}")
        if np.any(~np.isfinite(centered_risk)) or np.any(centered_risk <= 0.0):
            raise RuntimeError(f"invalid centered risk at {key}")
        if training_time[0] > T_MIN or training_time[-1] < T_MAX:
            raise RuntimeError(f"display interval is outside the source support at {key}")
        minimum_index = int(np.argmin(centered_risk))
        if minimum_index in (0, centered_risk.size - 1):
            raise RuntimeError(f"unresolved fixed-width minimum at {key}")
        curves[key] = {
            "training_time": training_time,
            "centered_risk": centered_risk,
            "sampled_minimum": np.asarray(
                [training_time[minimum_index], centered_risk[minimum_index]],
                dtype=float,
            ),
        }
    return curves


def _read_infinite_width_minima() -> dict[str, float]:
    if not INFINITE_WIDTH_INPUT.is_file():
        raise FileNotFoundError(INFINITE_WIDTH_INPUT)
    minima: dict[str, float] = {}
    with INFINITE_WIDTH_INPUT.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if (
                row["phase"] in {specification["legacy_key"] for specification in SUBREGIMES}
                and float(row["sigma2"]) == SIGMA2
            ):
                if row["accepted"].strip().lower() != "true":
                    raise RuntimeError(f"unaccepted infinite-width result: {row}")
                minima[row["phase"]] = float(row["direct_r_star"])
    expected = {str(specification["legacy_key"]) for specification in SUBREGIMES}
    if set(minima) != expected:
        raise RuntimeError(f"incomplete infinite-width minima: {minima}")
    return minima


def _interpolate_curve(
    training_time: np.ndarray,
    centered_risk: np.ndarray,
    display_time: np.ndarray,
) -> np.ndarray:
    return np.exp(
        np.interp(
            np.log(display_time),
            np.log(training_time),
            np.log(centered_risk),
        )
    )


def _configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 7.0,
            "axes.titlesize": 8.8,
            "axes.labelsize": 8.2,
            "xtick.labelsize": 7.0,
            "ytick.labelsize": 7.0,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.linewidth": 0.72,
        }
    )


def render() -> dict[str, object]:
    curves = _read_curves()
    infinite_width_minima = _read_infinite_width_minima()
    display_time = np.geomspace(T_MIN, T_MAX, T_POINTS)

    _configure_style()
    width_norm = LogNorm(vmin=min(DISPLAYED_WIDTHS), vmax=max(DISPLAYED_WIDTHS))
    width_cmap = _truncated_colormap("viridis")

    figure = plt.figure(figsize=(7.35, 2.66), constrained_layout=False)
    grid = figure.add_gridspec(
        1,
        4,
        width_ratios=(1.0, 1.0, 1.0, 0.050),
        left=0.075,
        right=0.985,
        bottom=0.235,
        top=0.865,
        wspace=0.115,
    )
    axes = [figure.add_subplot(grid[0, index]) for index in range(3)]
    colorbar_axis = figure.add_subplot(grid[0, 3])

    minima_manifest: dict[str, object] = {}
    for axis_index, (axis, specification) in enumerate(zip(axes, SUBREGIMES, strict=True)):
        legacy_key = str(specification["legacy_key"])
        panel_minima: dict[str, dict[str, float]] = {}
        for width in DISPLAYED_WIDTHS:
            curve = curves[(legacy_key, width)]
            color = width_cmap(width_norm(width))
            displayed_risk = _interpolate_curve(
                curve["training_time"],
                curve["centered_risk"],
                display_time,
            )
            axis.plot(
                display_time,
                displayed_risk,
                color=color,
                linewidth=1.25,
                alpha=0.93,
                solid_capstyle="round",
                zorder=2,
            )
            minimum_time, minimum_risk = curve["sampled_minimum"]
            axis.plot(
                minimum_time,
                minimum_risk,
                marker="o",
                markersize=3.5,
                markerfacecolor=color,
                markeredgecolor="white",
                markeredgewidth=0.55,
                linestyle="none",
                zorder=5,
            )
            panel_minima[str(width)] = {
                "sampled_iteration": float(minimum_time),
                "sampled_centered_risk": float(minimum_risk),
            }

        infinite_minimum = infinite_width_minima[legacy_key]
        axis.axvline(
            infinite_minimum,
            color="#3b3b3b",
            linewidth=0.95,
            linestyle=(0, (3.0, 2.2)),
            alpha=0.90,
            zorder=4,
        )

        axis.set_xscale("log")
        axis.set_yscale("log")
        axis.set_xlim(T_MIN, T_MAX)
        axis.set_ylim(4.0e-5, 1.05)
        axis.xaxis.set_major_locator(FixedLocator([1.0e1, 1.0e3, 1.0e5, 1.0e7]))
        axis.xaxis.set_major_formatter(LogFormatterMathtext(base=10))
        axis.xaxis.set_minor_formatter(NullFormatter())
        axis.yaxis.set_major_locator(FixedLocator([1.0e-4, 1.0e-2, 1.0e0]))
        axis.yaxis.set_major_formatter(LogFormatterMathtext(base=10))
        axis.yaxis.set_minor_formatter(NullFormatter())
        axis.grid(True, which="major", linewidth=0.40, alpha=0.22)
        axis.grid(False, which="minor")
        axis.tick_params(axis="both", which="major", length=2.4, width=0.65, pad=1.6)
        axis.tick_params(axis="both", which="minor", length=1.5, width=0.45)
        axis.set_title(
            rf"({specification['panel']}) ${specification['paper_name']}$: "
            rf"$\alpha={specification['alpha']:g},\ \beta={specification['beta']:g}$",
            fontweight="bold",
            pad=4.2,
        )
        if axis_index > 0:
            axis.tick_params(axis="y", which="both", labelleft=False)

        minima_manifest[str(specification["paper_name"])] = {
            "legacy_data_key": legacy_key,
            "infinite_width_iteration": float(infinite_minimum),
            "fixed_width_sampled_minima": panel_minima,
        }

    figure.supxlabel(r"training iterations $t$", fontsize=8.2, y=0.060)
    figure.supylabel("centered risk", fontsize=8.2, x=0.012)

    colorbar = figure.colorbar(
        ScalarMappable(norm=width_norm, cmap=width_cmap),
        cax=colorbar_axis,
        orientation="vertical",
        ticks=DISPLAYED_WIDTHS,
    )
    colorbar.ax.yaxis.set_major_locator(FixedLocator(DISPLAYED_WIDTHS))
    colorbar.ax.yaxis.set_major_formatter(
        FixedFormatter(["3.2k", "6.4k", "12.8k", "25.6k", "51.2k"])
    )
    colorbar.ax.yaxis.set_minor_locator(NullLocator())
    colorbar.ax.yaxis.set_minor_formatter(NullFormatter())
    colorbar.ax.tick_params(labelsize=6.8, length=2.0, width=0.55, pad=1.5)
    colorbar.ax.set_title(r"width $m$", fontsize=7.2, pad=4.0)
    colorbar.outline.set_linewidth(0.65)

    FIGURES.mkdir(parents=True, exist_ok=True)
    png_output = OUTPUT_STEM.with_suffix(".png")
    pdf_output = OUTPUT_STEM.with_suffix(".pdf")
    figure.savefig(png_output, dpi=400, bbox_inches="tight", pad_inches=0.02)
    figure.savefig(pdf_output, bbox_inches="tight", pad_inches=0.02)
    plt.close(figure)

    manifest: dict[str, object] = {
        "status": "COMPLETE",
        "figure": "finite-time early stopping in the integrable-memory subregimes",
        "solver": "support-adaptive zero-regularization deterministic-equivalent Volterra",
        "risk": "R_sigma(t,m)-P_sigma^star (clean-target risk after noisy-label training)",
        "sigma2": SIGMA2,
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "ambient_to_width_ratio": AMBIENT_TO_WIDTH_RATIO,
        "displayed_widths": list(DISPLAYED_WIDTHS),
        "display_interval": [T_MIN, T_MAX],
        "curve_interpolation": "linear in log(iteration)-log(centered risk), display only",
        "minimum_markers": "global minima on each 720-point native source grid",
        "vertical_dashed_lines": "accepted direct infinite-width minima",
        "subregimes": minima_manifest,
        "source_curves": [str(path.relative_to(ROOT)) for path in CURVE_INPUTS],
        "source_infinite_width_minima": str(INFINITE_WIDTH_INPUT.relative_to(ROOT)),
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
