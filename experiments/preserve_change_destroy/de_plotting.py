"""Visualization for the scalable spectral-quadrature experiment."""

from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Any

import numpy as np

from .de_config import DEProfile
from .de_quadrature import DETrajectory
from .slopes import local_log_slopes


def make_de_figure(
    profile: DEProfile,
    trajectories: dict[float, DETrajectory],
    nominal_fits: dict[float, dict[str, Any]],
    output_path: Path,
) -> tuple[Path, Path]:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(output_path.parent / ".matplotlib"))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {
        profile.representative_thetas[0]: "#c43c39",
        profile.representative_thetas[1]: "#d18700",
        profile.representative_thetas[2]: "#2878b5",
    }
    width = next(iter(trajectories.values())).effective_width
    lower_power, upper_power = profile.fit_window(profile.nominal_window_name)
    lower = math_power(width, lower_power)
    upper = math_power(width, upper_power)
    figure, axes = plt.subplots(1, 3, figsize=(13.5, 3.75))

    axis = axes[0]
    for theta in profile.representative_thetas:
        trajectory = trajectories[theta]
        valid = (trajectory.times > 0.0) & (trajectory.noisy_centered > 0.0)
        nominal = valid & (trajectory.times >= lower) & (trajectory.times <= upper)
        axis.loglog(
            trajectory.times[valid],
            trajectory.noisy_centered[valid],
            color=colors[theta],
            linewidth=0.9,
            alpha=0.28,
        )
        axis.loglog(
            trajectory.times[nominal],
            trajectory.noisy_centered[nominal],
            color=colors[theta],
            linewidth=2.2,
            label=rf"{profile.phase_label(theta)}, $\vartheta={theta:g}$",
        )
        indices = np.flatnonzero(nominal)
        if indices.size:
            anchor = int(indices[indices.size // 2])
            guide = trajectory.noisy_centered[anchor] * (
                trajectory.times[nominal] / trajectory.times[anchor]
            ) ** (-profile.total_exponent(theta))
            axis.loglog(
                trajectory.times[nominal],
                guide,
                color=colors[theta],
                linestyle="--",
                linewidth=1.1,
            )
    axis.set_xlabel(r"intrinsic time $T$")
    axis.set_ylabel(r"centered noisy risk")
    axis.set_title("(a) Triangular pre-saturation window")
    axis.grid(True, which="both", alpha=0.18)
    axis.legend(frameon=False, fontsize=8)

    axis = axes[1]
    for theta in profile.representative_thetas:
        trajectory = trajectories[theta]
        valid = (
            (trajectory.times >= lower)
            & (trajectory.times <= upper)
            & (trajectory.noisy_centered > 0.0)
        )
        slopes = local_log_slopes(
            trajectory.times,
            trajectory.noisy_centered,
            valid=valid,
            half_window_decades=profile.local_slope_half_window_decades,
            minimum_points=7,
        )
        finite = np.isfinite(slopes)
        axis.semilogx(
            trajectory.times[finite],
            slopes[finite],
            color=colors[theta],
            linewidth=2.0,
        )
        axis.axhline(
            profile.total_exponent(theta),
            color=colors[theta],
            linestyle="--",
            linewidth=1.0,
        )
    axis.set_xlabel(r"intrinsic time $T$")
    axis.set_ylabel("local effective exponent")
    axis.set_title("(b) Local slopes inside the joint limit")
    axis.grid(True, which="both", alpha=0.18)

    axis = axes[2]
    theta_grid = np.linspace(min(profile.theta_values), max(profile.theta_values), 400)
    axis.plot(
        theta_grid,
        [profile.total_exponent(value) for value in theta_grid],
        color="black",
        linewidth=2.0,
        label="theory",
    )
    theta_values = np.asarray(profile.theta_values, dtype=float)
    exponents = np.asarray(
        [nominal_fits[value]["exponent"] for value in profile.theta_values],
        dtype=float,
    )
    accepted = np.asarray(
        [nominal_fits[value]["theory_status"] == "PASS" for value in profile.theta_values],
        dtype=bool,
    )
    if np.any(accepted):
        axis.scatter(
            theta_values[accepted],
            exponents[accepted],
            s=46,
            facecolors="#4c956c",
            edgecolors="#2f4858",
            linewidths=1.1,
            label="accepted triangular fit",
            zorder=3,
        )
    if np.any(~accepted):
        axis.scatter(
            theta_values[~accepted],
            exponents[~accepted],
            s=46,
            facecolors="white",
            edgecolors="#2f4858",
            linewidths=1.1,
            label="inconclusive",
            zorder=3,
        )
    axis.axvline(profile.destroy_boundary, color="#c43c39", linestyle=":", linewidth=1.2)
    axis.axvline(profile.preservation_boundary, color="#2878b5", linestyle=":", linewidth=1.2)
    axis.set_xlabel(r"ratio-growth exponent $\vartheta$")
    axis.set_ylabel("total centered-risk exponent")
    axis.set_title("(c) Predicted versus measured")
    axis.set_ylim(bottom=-0.05)
    axis.grid(True, alpha=0.18)
    axis.legend(frameon=False, fontsize=8)

    figure.suptitle(
        rf"Head-preserving spectral proxy: $m=10^{{{int(round(math.log10(float(width))))}}}$, "
        rf"$\alpha={profile.alpha:g}$, $\beta={profile.beta:g}$",
        fontsize=11,
        y=1.02,
    )
    figure.tight_layout()
    png = output_path.with_suffix(".png")
    pdf = output_path.with_suffix(".pdf")
    figure.savefig(png, dpi=220, bbox_inches="tight")
    figure.savefig(pdf, bbox_inches="tight")
    plt.close(figure)
    return png, pdf


def math_power(width: int, exponent: float) -> float:
    return float(np.exp(exponent * np.log(float(width))))
