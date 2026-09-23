"""Plot four 124M validation trajectories and two frozen surrogate curves."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np

from experiments.nanogpt_local import e2e_sgd_rpath_factorization_124m_2p5b_v002 as core
from experiments.nanogpt_local import e2e_sgd_unified_phase_surrogate_124m_2p5b_v002 as fit_analysis


Json = dict[str, Any]
DISPLAY_START = 9500.0
EIGHT_ARMS = (
    "eight_one_one__fixed_batch_lr",
    "eight_one_one__fixed_lr_batch",
)
WSD_ARMS = (
    "wsd_exp_80_20__fixed_batch_lr",
    "wsd_exp_80_20__fixed_lr_batch",
)


def _load(path: Path) -> dict[str, dict[str, np.ndarray]]:
    grouped: dict[str, list[dict[str, str]]] = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            grouped.setdefault(str(row["arm_id"]), []).append(row)
    result = {}
    for arm_id, rows in grouped.items():
        result[arm_id] = {
            "intrinsic_time": np.asarray(
                [float(row["intrinsic_time"]) for row in rows],
                dtype=np.float64,
            ),
            "observed": np.asarray(
                [float(row["observed_validation_ce"]) for row in rows],
                dtype=np.float64,
            ),
            "predicted": np.asarray(
                [float(row["predicted_validation_ce"]) for row in rows],
                dtype=np.float64,
            ),
        }
    return result


def _assert_matched_fit(
    curves: dict[str, dict[str, np.ndarray]], arms: tuple[str, str]
) -> None:
    first, second = (curves[arm] for arm in arms)
    if not np.array_equal(first["intrinsic_time"], second["intrinsic_time"]):
        raise ValueError(f"intrinsic-time grids differ: {arms}")
    if not np.array_equal(first["predicted"], second["predicted"]):
        raise ValueError(f"factorization predictions differ: {arms}")


def render(output_directory: Path) -> tuple[Path, Path]:
    eight = _load(output_directory / "predictions.csv")
    wsd = _load(output_directory / "wsd_zero_refit_predictions.csv")
    if set(eight) != set(EIGHT_ARMS):
        raise ValueError(f"unexpected 8-1-1 arms: {set(eight)}")
    if set(wsd) != set(WSD_ARMS):
        raise ValueError(f"unexpected WSD arms: {set(wsd)}")
    _assert_matched_fit(eight, EIGHT_ARMS)
    _assert_matched_fit(wsd, WSD_ARMS)

    curves = {**eight, **wsd}
    styles = {
        WSD_ARMS[0]: {
            "color": "#4B91C6",
            "linewidth": 3.0,
            "zorder": 4,
            "label": "WSD · fixed batch / LR schedule",
        },
        WSD_ARMS[1]: {
            "color": "#B5D2E8",
            "linewidth": 5.2,
            "zorder": 2,
            "label": "WSD · fixed LR / batch schedule",
        },
        EIGHT_ARMS[0]: {
            "color": "#EA914F",
            "linewidth": 3.0,
            "zorder": 4,
            "label": "8-1-1 · fixed batch / LR schedule",
        },
        EIGHT_ARMS[1]: {
            "color": "#F5D0AC",
            "linewidth": 5.2,
            "zorder": 2,
            "label": "8-1-1 · fixed LR / batch schedule",
        },
    }

    figure, axis = plt.subplots(figsize=(13.2, 7.6), constrained_layout=True)
    prefix = eight[EIGHT_ARMS[0]]
    prefix_mask = (
        (prefix["intrinsic_time"] >= DISPLAY_START)
        & (prefix["intrinsic_time"] <= core.PREFIX_INTRINSIC_TIME + 1e-12)
    )
    axis.plot(
        prefix["intrinsic_time"][prefix_mask],
        prefix["observed"][prefix_mask],
        color="#9298A2",
        linewidth=2.3,
        alpha=0.95,
        zorder=1,
    )

    for arm_id in (WSD_ARMS[0], WSD_ARMS[1], EIGHT_ARMS[0], EIGHT_ARMS[1]):
        curve = curves[arm_id]
        tail = curve["intrinsic_time"] >= core.PREFIX_INTRINSIC_TIME - 1e-12
        style = styles[arm_id]
        axis.plot(
            curve["intrinsic_time"][tail],
            curve["observed"][tail],
            color=str(style["color"]),
            linewidth=float(style["linewidth"]),
            alpha=0.94,
            label=str(style["label"]),
            zorder=int(style["zorder"]),
        )

    for arm_id, color, label in (
        (WSD_ARMS[0], "#0B3B6E", "WSD · frozen fit"),
        (EIGHT_ARMS[0], "#8C3C00", "8-1-1 · frozen fit"),
    ):
        curve = curves[arm_id]
        tail = curve["intrinsic_time"] >= core.PREFIX_INTRINSIC_TIME - 1e-12
        axis.plot(
            curve["intrinsic_time"][tail],
            curve["predicted"][tail],
            color=color,
            linewidth=2.2,
            alpha=1.0,
            label=label,
            zorder=6,
        )

    axis.set_title("124M nanoGPT SGD, 2.5B tokens", fontsize=27, pad=18)
    axis.set_xlabel(r"Intrinsic time $\sum\eta$", fontsize=21)
    axis.set_ylabel("Risk", fontsize=21)
    axis.set_xlim(DISPLAY_START, float(wsd[WSD_ARMS[0]]["intrinsic_time"][-1]))
    axis.tick_params(axis="both", labelsize=15)
    axis.grid(alpha=0.16, linewidth=0.8)
    axis.legend(
        loc="upper right",
        frameon=False,
        fontsize=12.5,
        ncol=2,
        columnspacing=1.4,
        handlelength=3.0,
    )
    for spine in axis.spines.values():
        spine.set_linewidth(1.0)

    png = output_directory / "four_validation_curves_and_two_fits_intrinsic.png"
    pdf = output_directory / "four_validation_curves_and_two_fits_intrinsic.pdf"
    figure.savefig(png, dpi=240, bbox_inches="tight")
    figure.savefig(pdf, bbox_inches="tight")
    plt.close(figure)
    return png, pdf


def render_small_multiples(output_directory: Path) -> tuple[Path, Path]:
    """Separate factorizations so no panel contains three overlapping curves."""
    eight = _load(output_directory / "predictions.csv")
    wsd = _load(output_directory / "wsd_zero_refit_predictions.csv")
    if set(eight) != set(EIGHT_ARMS):
        raise ValueError(f"unexpected 8-1-1 arms: {set(eight)}")
    if set(wsd) != set(WSD_ARMS):
        raise ValueError(f"unexpected WSD arms: {set(wsd)}")
    _assert_matched_fit(eight, EIGHT_ARMS)
    _assert_matched_fit(wsd, WSD_ARMS)

    panels = (
        ("WSD", "fixed batch", wsd[WSD_ARMS[0]], "#77ADD2", "#0B3B6E"),
        ("WSD", "fixed LR", wsd[WSD_ARMS[1]], "#77ADD2", "#0B3B6E"),
        ("8-1-1", "fixed batch", eight[EIGHT_ARMS[0]], "#F0A261", "#8C3C00"),
        ("8-1-1", "fixed LR", eight[EIGHT_ARMS[1]], "#F0A261", "#8C3C00"),
    )

    figure, axes = plt.subplots(
        2,
        2,
        figsize=(14.0, 9.0),
        sharex="row",
        sharey=True,
    )
    figure.subplots_adjust(
        left=0.075,
        right=0.99,
        bottom=0.105,
        top=0.79,
        wspace=0.09,
        hspace=0.34,
    )
    prefix = eight[EIGHT_ARMS[0]]
    prefix_mask = (
        (prefix["intrinsic_time"] >= DISPLAY_START)
        & (prefix["intrinsic_time"] <= core.PREFIX_INTRINSIC_TIME + 1e-12)
    )

    for axis, (schedule, factorization, curve, raw_color, fit_color) in zip(
        axes.flat, panels, strict=True
    ):
        axis.plot(
            prefix["intrinsic_time"][prefix_mask],
            prefix["observed"][prefix_mask],
            color="#9298A2",
            linewidth=2.2,
            alpha=0.95,
            zorder=1,
        )
        tail = curve["intrinsic_time"] >= core.PREFIX_INTRINSIC_TIME - 1e-12
        # A broad, light empirical curve remains visible as a halo even when the
        # narrower frozen fit lies directly on top of it.
        axis.plot(
            curve["intrinsic_time"][tail],
            curve["observed"][tail],
            color=raw_color,
            linewidth=5.0,
            alpha=0.88,
            zorder=3,
        )
        axis.plot(
            curve["intrinsic_time"][tail],
            curve["predicted"][tail],
            color=fit_color,
            linewidth=1.65,
            alpha=1.0,
            zorder=5,
        )
        axis.set_title(f"{schedule} · {factorization}", fontsize=18, pad=9)
        axis.set_xlim(DISPLAY_START, float(curve["intrinsic_time"][-1]))
        axis.tick_params(axis="both", labelsize=13)
        axis.grid(alpha=0.15, linewidth=0.8)
        for spine in axis.spines.values():
            spine.set_linewidth(1.0)

    axes[0, 0].set_ylabel("Risk", fontsize=19)
    axes[1, 0].set_ylabel("Risk", fontsize=19)
    figure.supxlabel(r"Intrinsic time $\sum\eta$", fontsize=19, y=0.025)
    figure.suptitle("124M nanoGPT SGD, 2.5B tokens", fontsize=27, y=0.975)
    figure.legend(
        handles=(
            Line2D([], [], color="#83A9C4", linewidth=5.0, label="Validation loss"),
            Line2D([], [], color="#263746", linewidth=1.65, label="Frozen fit"),
            Line2D([], [], color="#9298A2", linewidth=2.2, label="Shared prefix"),
        ),
        loc="upper center",
        bbox_to_anchor=(0.5, 0.905),
        frameon=False,
        fontsize=14,
        ncol=3,
        handlelength=3.0,
        columnspacing=2.0,
    )

    png = output_directory / "four_validation_small_multiples_intrinsic.png"
    pdf = output_directory / "four_validation_small_multiples_intrinsic.pdf"
    figure.savefig(png, dpi=240, bbox_inches="tight")
    figure.savefig(pdf, bbox_inches="tight")
    plt.close(figure)
    return png, pdf


def render_nested_strokes(output_directory: Path) -> tuple[Path, Path]:
    """Show all six curves on one axis using nested, separated solid strokes."""
    eight = _load(output_directory / "predictions.csv")
    wsd = _load(output_directory / "wsd_zero_refit_predictions.csv")
    if set(eight) != set(EIGHT_ARMS):
        raise ValueError(f"unexpected 8-1-1 arms: {set(eight)}")
    if set(wsd) != set(WSD_ARMS):
        raise ValueError(f"unexpected WSD arms: {set(wsd)}")
    _assert_matched_fit(eight, EIGHT_ARMS)
    _assert_matched_fit(wsd, WSD_ARMS)

    schedule_styles = (
        {
            "curves": wsd,
            "arms": WSD_ARMS,
            "name": "WSD",
            "fixed_lr_color": "#C7DFF0",
            "fixed_batch_color": "#5598C8",
            "fit_color": "#0B3B6E",
        },
        {
            "curves": eight,
            "arms": EIGHT_ARMS,
            "name": "8-1-1",
            "fixed_lr_color": "#F7D1AB",
            "fixed_batch_color": "#E98B3D",
            "fit_color": "#8C3C00",
        },
    )

    figure, axis = plt.subplots(figsize=(13.2, 7.6), constrained_layout=True)
    prefix = eight[EIGHT_ARMS[0]]
    prefix_mask = (
        (prefix["intrinsic_time"] >= DISPLAY_START)
        & (prefix["intrinsic_time"] <= core.PREFIX_INTRINSIC_TIME + 1e-12)
    )
    axis.plot(
        prefix["intrinsic_time"][prefix_mask],
        prefix["observed"][prefix_mask],
        color="#9298A2",
        linewidth=2.3,
        alpha=0.95,
        zorder=1,
    )

    # Draw the widest layer for both schedules first, then the middle layer,
    # then the fit. White casings keep every nested stroke visible at exact
    # overlap without moving or transforming any observation.
    for style in schedule_styles:
        curves = style["curves"]
        fixed_lr = curves[style["arms"][1]]
        tail = fixed_lr["intrinsic_time"] >= core.PREFIX_INTRINSIC_TIME - 1e-12
        axis.plot(
            fixed_lr["intrinsic_time"][tail],
            fixed_lr["observed"][tail],
            color=style["fixed_lr_color"],
            linewidth=13.0,
            alpha=0.90,
            solid_capstyle="round",
            solid_joinstyle="round",
            zorder=2,
        )

    for style in schedule_styles:
        curves = style["curves"]
        fixed_batch = curves[style["arms"][0]]
        tail = fixed_batch["intrinsic_time"] >= core.PREFIX_INTRINSIC_TIME - 1e-12
        axis.plot(
            fixed_batch["intrinsic_time"][tail],
            fixed_batch["observed"][tail],
            color=style["fixed_batch_color"],
            linewidth=5.8,
            alpha=0.98,
            solid_capstyle="round",
            solid_joinstyle="round",
            path_effects=(
                path_effects.Stroke(linewidth=8.4, foreground="white", alpha=0.95),
                path_effects.Normal(),
            ),
            zorder=4,
        )

    for style in schedule_styles:
        curves = style["curves"]
        fit = curves[style["arms"][0]]
        tail = fit["intrinsic_time"] >= core.PREFIX_INTRINSIC_TIME - 1e-12
        axis.plot(
            fit["intrinsic_time"][tail],
            fit["predicted"][tail],
            color=style["fit_color"],
            linewidth=2.1,
            alpha=1.0,
            solid_capstyle="round",
            solid_joinstyle="round",
            path_effects=(
                path_effects.Stroke(linewidth=3.8, foreground="white", alpha=0.95),
                path_effects.Normal(),
            ),
            zorder=6,
        )

    legend_handles = []
    for style in schedule_styles:
        legend_handles.extend(
            (
                Line2D(
                    [],
                    [],
                    color=style["fixed_lr_color"],
                    linewidth=8.5,
                    label=f'{style["name"]} · fixed LR',
                ),
                Line2D(
                    [],
                    [],
                    color=style["fixed_batch_color"],
                    linewidth=4.5,
                    label=f'{style["name"]} · fixed batch',
                ),
                Line2D(
                    [],
                    [],
                    color=style["fit_color"],
                    linewidth=2.0,
                    label=f'{style["name"]} · frozen fit',
                ),
            )
        )

    axis.set_title("124M nanoGPT SGD, 2.5B tokens", fontsize=27, pad=18)
    axis.set_xlabel(r"Intrinsic time $\sum\eta$", fontsize=21)
    axis.set_ylabel("Risk", fontsize=21)
    axis.set_xlim(DISPLAY_START, float(wsd[WSD_ARMS[0]]["intrinsic_time"][-1]))
    axis.tick_params(axis="both", labelsize=15)
    axis.grid(alpha=0.16, linewidth=0.8)
    axis.legend(
        handles=legend_handles,
        loc="upper right",
        frameon=False,
        fontsize=11.5,
        ncol=2,
        columnspacing=1.5,
        handlelength=3.0,
        handletextpad=0.8,
    )
    for spine in axis.spines.values():
        spine.set_linewidth(1.0)

    png = output_directory / "four_validation_nested_strokes_intrinsic.png"
    pdf = output_directory / "four_validation_nested_strokes_intrinsic.pdf"
    figure.savefig(png, dpi=240, bbox_inches="tight")
    figure.savefig(pdf, bbox_inches="tight")
    plt.close(figure)
    return png, pdf


def render_3d_categorical_lanes(output_directory: Path) -> tuple[Path, Path]:
    """Place each curve role on a categorical depth lane in one 3D axis."""
    display_start = 0.5 * (DISPLAY_START + core.PREFIX_INTRINSIC_TIME)
    eight = _load(output_directory / "predictions.csv")
    wsd = _load(output_directory / "wsd_zero_refit_predictions.csv")
    if set(eight) != set(EIGHT_ARMS):
        raise ValueError(f"unexpected 8-1-1 arms: {set(eight)}")
    if set(wsd) != set(WSD_ARMS):
        raise ValueError(f"unexpected WSD arms: {set(wsd)}")
    _assert_matched_fit(eight, EIGHT_ARMS)
    _assert_matched_fit(wsd, WSD_ARMS)
    display_end = float(eight[EIGHT_ARMS[0]]["intrinsic_time"][-1])

    fit_lane = 0.0
    fixed_batch_lane = 1.0
    fixed_lr_lane = 2.0
    # Draw orange first and blue last so WSD is the consistent foreground
    # trajectory wherever the two schedule projections cross.
    schedule_styles = (
        {
            "name": "8-1-1",
            "fit_color": "#D66A00",
            "empirical_color": "#F2B678",
            "fixed_lr_color": "#F8D4AE",
            "surface_zorder": 1.0,
            "curves": eight,
            "arms": EIGHT_ARMS,
        },
        {
            "name": "WSD",
            "fit_color": "#2166AC",
            "empirical_color": "#8DB9D8",
            "fixed_lr_color": "#C4DBEC",
            "surface_zorder": 1.1,
            "curves": wsd,
            "arms": WSD_ARMS,
        },
    )

    figure = plt.figure(figsize=(9.2, 9.0))
    axis = figure.add_subplot(111, projection="3d", computed_zorder=False)
    axis.set_proj_type("ortho")

    # Extend each schedule's theory curve across all three depth lanes.  The
    # resulting reference surface is smooth; the observed LR- and batch-schedule
    # trajectories remain unmodified and show their departures from theory.
    # Colored surfaces begin at the fork; the shared prefix is filled separately
    # in gray below.
    for style in schedule_styles:
        curves = style["curves"]
        fixed_batch = curves[style["arms"][0]]
        tail = (
            (fixed_batch["intrinsic_time"] >= core.PREFIX_INTRINSIC_TIME - 1e-12)
            & (fixed_batch["intrinsic_time"] <= display_end + 1e-12)
        )
        time = fixed_batch["intrinsic_time"][tail]
        theory_risk = fixed_batch["predicted"][tail]
        sample_count = min(240, time.size)
        sample_indices = np.unique(
            np.linspace(0, time.size - 1, sample_count, dtype=np.int64)
        )
        sampled_time = time[sample_indices]
        sampled_risk = theory_risk[sample_indices]
        front_edge = np.column_stack(
            (
                sampled_time,
                np.full(sampled_time.shape, fit_lane),
                sampled_risk,
            )
        )
        back_edge = np.column_stack(
            (
                sampled_time[::-1],
                np.full(sampled_time.shape, fixed_lr_lane),
                sampled_risk[::-1],
            )
        )
        surface = Poly3DCollection(
            [np.vstack((front_edge, back_edge))],
            facecolor=style["fit_color"],
            alpha=0.075,
            edgecolor="none",
            linewidth=0,
            antialiased=False,
            zorder=float(style["surface_zorder"]),
        )
        axis.add_collection3d(surface)

    prefix = eight[EIGHT_ARMS[0]]
    prefix_mask = (
        (prefix["intrinsic_time"] >= display_start)
        & (prefix["intrinsic_time"] <= core.PREFIX_INTRINSIC_TIME + 1e-12)
    )
    prefix_time = prefix["intrinsic_time"][prefix_mask]
    prefix_risk = prefix["observed"][prefix_mask]
    # Repeat the one shared observed prefix only on the two empirical lanes.
    # The back lane is reserved for the frozen fitted curve over the full
    # displayed interval, including the prefix.
    for lane, prefix_color, prefix_linewidth in (
        (fixed_lr_lane, "#D0D5DA", 4.2),
        (fixed_batch_lane, "#A7AFB7", 3.2),
    ):
        axis.plot(
            prefix_time,
            np.full(prefix_time.shape, lane),
            prefix_risk,
            color=prefix_color,
            linewidth=prefix_linewidth,
            alpha=0.95,
            zorder=3,
        )

    fitted_prefix_curve = eight[EIGHT_ARMS[0]]
    fitted_prefix_mask = (
        (fitted_prefix_curve["intrinsic_time"] >= display_start)
        & (
            fitted_prefix_curve["intrinsic_time"]
            <= core.PREFIX_INTRINSIC_TIME + 1e-12
        )
    )
    fitted_prefix_time = fitted_prefix_curve["intrinsic_time"][fitted_prefix_mask]
    fitted_prefix_risk = fitted_prefix_curve["predicted"][fitted_prefix_mask]
    prefix_sample_count = min(120, fitted_prefix_time.size)
    prefix_sample_indices = np.unique(
        np.linspace(
            0,
            fitted_prefix_time.size - 1,
            prefix_sample_count,
            dtype=np.int64,
        )
    )
    sampled_prefix_time = fitted_prefix_time[prefix_sample_indices]
    sampled_prefix_risk = fitted_prefix_risk[prefix_sample_indices]
    prefix_front_edge = np.column_stack(
        (
            sampled_prefix_time,
            np.full(sampled_prefix_time.shape, fit_lane),
            sampled_prefix_risk,
        )
    )
    prefix_back_edge = np.column_stack(
        (
            sampled_prefix_time[::-1],
            np.full(sampled_prefix_time.shape, fixed_lr_lane),
            sampled_prefix_risk[::-1],
        )
    )
    prefix_surface = Poly3DCollection(
        [np.vstack((prefix_front_edge, prefix_back_edge))],
        facecolor="#9298A2",
        alpha=0.10,
        edgecolor="none",
        linewidth=0,
        antialiased=False,
        zorder=1.2,
    )
    axis.add_collection3d(prefix_surface)
    axis.plot(
        fitted_prefix_time,
        np.full(fitted_prefix_time.shape, fit_lane),
        fitted_prefix_risk,
        color="#7D858E",
        linewidth=4.6,
        alpha=0.95,
        solid_capstyle="round",
        zorder=4,
    )

    for style in schedule_styles:
        curves = style["curves"]
        fixed_lr = curves[style["arms"][1]]
        fixed_batch = curves[style["arms"][0]]
        fit = curves[style["arms"][0]]

        for curve, lane, values, linewidth, color in (
            (
                fixed_lr,
                fixed_lr_lane,
                "observed",
                4.2,
                style["fixed_lr_color"],
            ),
            (
                fixed_batch,
                fixed_batch_lane,
                "observed",
                3.2,
                style["empirical_color"],
            ),
            (fit, fit_lane, "predicted", 4.6, style["fit_color"]),
        ):
            tail = (
                (curve["intrinsic_time"] >= core.PREFIX_INTRINSIC_TIME - 1e-12)
                & (curve["intrinsic_time"] <= display_end + 1e-12)
            )
            time = curve["intrinsic_time"][tail]
            axis.plot(
                time,
                np.full(time.shape, lane),
                curve[values][tail],
                color=color,
                linewidth=linewidth,
                alpha=1.0,
                solid_capstyle="round",
                zorder=5,
            )

    axis.xaxis.set_rotate_label(False)
    axis.set_xlabel(
        "Intrinsic time",
        fontsize=31.5,
        fontweight="bold",
        labelpad=34,
        rotation=0,
    )
    axis.set_ylabel("")
    axis.zaxis.set_rotate_label(False)
    axis.set_zlabel(
        "Risk",
        fontsize=31.5,
        fontweight="bold",
        labelpad=48,
        rotation=90,
    )
    axis.set_xlim(display_start, display_end)
    axis.set_ylim(-0.18, 2.18)
    axis.set_zlim(3.745, 3.855)
    axis.set_xticks([9700, 9900, 10100])
    axis.set_zticks([3.76, 3.80, 3.84])
    depth_tick_labels = ["Theory", "LR schedule", "Batch schedule"]
    axis.set_yticks(
        [fit_lane, fixed_batch_lane, fixed_lr_lane],
        labels=depth_tick_labels,
    )
    axis.tick_params(axis="x", labelsize=24, pad=5)
    axis.tick_params(axis="y", labelsize=24, pad=7, length=4)
    axis.tick_params(axis="z", labelsize=24, pad=20)
    for tick_label in (
        *axis.get_xticklabels(),
        *axis.get_yticklabels(),
        *axis.get_zticklabels(),
    ):
        tick_label.set_fontweight("bold")
    axis.set_box_aspect((1.45, 1.0, 1.0), zoom=1.06)
    axis.view_init(elev=14, azim=-78)
    axis.grid(False)

    for axis_component in (axis.xaxis, axis.yaxis, axis.zaxis):
        axis_component.pane.set_facecolor((1.0, 1.0, 1.0, 0.0))
        axis_component.pane.set_edgecolor((0.78, 0.80, 0.83, 0.75))
        axis_component._axinfo["grid"]["linewidth"] = 0.0

    axis.legend(
        handles=(
            Line2D([], [], color="#2166AC", linewidth=3.6, label="WSD"),
            Line2D([], [], color="#D66A00", linewidth=3.6, label="8-1-1"),
            Line2D(
                [], [], color="#A7AFB7", linewidth=3.6, label="Shared prefix"
            ),
        ),
        loc="lower left",
        bbox_to_anchor=(0.095, 0.19),
        frameon=False,
        prop={"size": 24, "weight": "bold"},
        handlelength=2.8,
    )
    figure.subplots_adjust(left=0.01, right=0.98, bottom=0.03, top=0.91)

    # mplot3d offsets categorical tick text below its tick-line center even
    # when va="center".  Replace those labels with 2D text anchored to the
    # actual rendered tick-line extensions so the visual alignment is exact.
    figure.canvas.draw()
    depth_label_offset_pixels = 8.0 * figure.dpi / 72.0
    for tick, text_label in zip(
        axis.yaxis.get_major_ticks(), depth_tick_labels, strict=True
    ):
        tick_vertices = tick.tick1line.get_path().transformed(
            tick.tick1line.get_transform()
        ).vertices
        outer_index = int(np.argmax(tick_vertices[:, 0]))
        inner_index = 1 - outer_index
        outer_x, outer_y = tick_vertices[outer_index]
        inner_x, inner_y = tick_vertices[inner_index]
        slope = (outer_y - inner_y) / (outer_x - inner_x)
        anchor_x = outer_x + depth_label_offset_pixels
        anchor_y = outer_y + slope * depth_label_offset_pixels
        figure_x, figure_y = figure.transFigure.inverted().transform(
            (anchor_x, anchor_y)
        )
        tick.label1.set_visible(False)
        figure.text(
            figure_x,
            figure_y,
            text_label,
            fontsize=24,
            fontweight="bold",
            horizontalalignment="left",
            verticalalignment="center",
            transform=figure.transFigure,
            bbox={"facecolor": "white", "edgecolor": "none", "pad": 0.15},
        )

    png = output_directory / "four_validation_3d_categorical_lanes_intrinsic.png"
    pdf = output_directory / "four_validation_3d_categorical_lanes_intrinsic.pdf"
    figure.savefig(png, dpi=240, bbox_inches="tight")
    figure.savefig(pdf, bbox_inches="tight")
    plt.close(figure)
    return png, pdf


def main() -> int:
    root = Path(__file__).resolve().parents[2]
    output = root / "experiments" / "results" / fit_analysis.ANALYSIS_ID
    png, pdf = render(output)
    print(png)
    print(pdf)
    small_png, small_pdf = render_small_multiples(output)
    print(small_png)
    print(small_pdf)
    nested_png, nested_pdf = render_nested_strokes(output)
    print(nested_png)
    print(nested_pdf)
    lanes_png, lanes_pdf = render_3d_categorical_lanes(output)
    print(lanes_png)
    print(lanes_pdf)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
