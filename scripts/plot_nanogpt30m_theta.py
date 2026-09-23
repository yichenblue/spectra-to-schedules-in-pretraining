#!/usr/bin/env python3
"""Reproduce the paper's 30M nanoGPT schedule-response figure."""

from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, Normalize

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "language_model" / "nanogpt30m_theta"
OUTPUT = ROOT / "reproduced_figures" / "nanogpt30m_theta_schedule_response.pdf"

ETA0 = 0.005941406469904266
PREFIX_UPDATES = 196_608
PREFIX_INTRINSIC_TIME = PREFIX_UPDATES * ETA0
TAU_SCALE = 1_024 * ETA0
TAU_MAX = 14 * TAU_SCALE

ARMS = (
    (0.0, "theta0"),
    (0.125, "theta0p125"),
    (0.25, "theta0p25"),
    (0.375, "theta0p375"),
    (0.5, "theta0p5"),
    (0.75, "theta0p75"),
    (1.0, "theta1"),
    (1.25, "theta1p25"),
    (1.5, "theta1p5"),
    (1.75, "theta1p75"),
    (2.0, "theta2"),
)


def main() -> int:
    plt.rcParams.update(
        {
            "font.size": 15,
            "axes.titlesize": 18,
            "axes.labelsize": 17,
            "xtick.labelsize": 14,
            "ytick.labelsize": 14,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "legend.frameon": False,
        }
    )
    figure, axis = plt.subplots(figsize=(8.4, 6.3))
    base_palette = plt.get_cmap("Blues")
    theta_palette = LinearSegmentedColormap.from_list(
        "theta_blues", [base_palette(0.25), base_palette(0.98)]
    )
    theta_norm = Normalize(vmin=0.0, vmax=2.0)

    for theta, folder in ARMS:
        path = DATA / folder / "validation_evaluations.npz"
        with np.load(path, allow_pickle=False) as payload:
            x = np.asarray(payload["actual_x"], dtype=np.float64)
            risk = np.asarray(payload["validation_cross_entropy"], dtype=np.float64)
        if x.shape != risk.shape or x.size != 410 or not np.all(np.isfinite(risk)):
            raise RuntimeError(f"invalid validation curve: {path}")
        intrinsic_time = PREFIX_INTRINSIC_TIME + (x - 1.0) * TAU_SCALE
        axis.plot(
            intrinsic_time,
            risk,
            color=theta_palette(theta_norm(theta)),
            alpha=0.94,
            linewidth=1.9,
        )

    axis.set_xlim(PREFIX_INTRINSIC_TIME, PREFIX_INTRINSIC_TIME + TAU_MAX)
    axis.set_ylim(5.052, 5.19)
    axis.set_xlabel("Intrinsic time")
    axis.set_ylabel("Risk")
    axis.set_title("30M nanoGPT plain-SGD · eleven power-law schedule tails", pad=13)
    axis.grid(True, color="#d1d5db", linewidth=0.7, alpha=0.55)
    scalar_map = plt.cm.ScalarMappable(norm=theta_norm, cmap=theta_palette)
    scalar_map.set_array([])
    colorbar = figure.colorbar(
        scalar_map,
        ax=axis,
        ticks=[0.0, 0.5, 1.0, 1.5, 2.0],
        fraction=0.055,
        pad=0.035,
        aspect=24,
    )
    colorbar.set_label(r"$\vartheta$", rotation=0, labelpad=12)
    colorbar.ax.tick_params(labelsize=14)
    colorbar.outline.set_linewidth(0.6)
    figure.subplots_adjust(left=0.14, right=0.87, top=0.87, bottom=0.15)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(OUTPUT)
    figure.savefig(OUTPUT.with_suffix(".png"), dpi=220)
    plt.close(figure)
    print(OUTPUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
