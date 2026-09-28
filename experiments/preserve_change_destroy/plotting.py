"""Paper-facing three-panel visualization for the experiment."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np

from .config import ExperimentProfile
from .slopes import local_log_slopes


def make_three_panel_figure(
    profile: ExperimentProfile,
    largest_width_curves: dict[float, dict[str, np.ndarray]],
    nominal_fits: dict[float, dict[str, Any]],
    output_path: Path,
) -> tuple[Path, Path]:
    """Plot trajectories, local exponents, and the measured phase law."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault(
        "MPLCONFIGDIR", str(output_path.parent / ".matplotlib")
    )
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {
        profile.representative_thetas[0]: "#c43c39",
        profile.representative_thetas[1]: "#d18700",
        profile.representative_thetas[2]: "#2878b5",
    }
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 3.65))

    # Panel A: the observable that the corollary predicts.
    axis = axes[0]
    for theta in profile.representative_thetas:
        curve = largest_width_curves[theta]
        time = curve["times"]
        value = curve["noisy_centered_mean"]
        mask = (time > 0.0) & (value > 0.0)
        label = rf"{profile.phase_label(theta)}, $\vartheta={theta:g}$"
        axis.loglog(time[mask], value[mask], color=colors[theta], lw=2.0, label=label)

        lower = profile.fit_time_min
        upper = profile.nominal_fit_upper_factor * curve["width"] ** (
            2.0 * profile.alpha
        )
        guide = mask & (time >= lower) & (time <= upper)
        if np.count_nonzero(guide) >= 2:
            guide_indices = np.flatnonzero(guide)
            middle = guide_indices[len(guide_indices) // 2]
            exponent = profile.total_exponent(theta)
            guide_time = time[guide]
            anchor_time = time[middle]
            anchor_value = value[middle]
            guide_value = anchor_value * (guide_time / anchor_time) ** (-exponent)
            axis.loglog(
                guide_time,
                guide_value,
                color=colors[theta],
                lw=1.2,
                ls="--",
                alpha=0.9,
            )
    axis.set_xlabel(r"intrinsic time $T$")
    axis.set_ylabel(r"centered noisy risk $R_\sigma-R_{\mathrm{app}}$")
    axis.set_title("(a) Preserve / change / destroy")
    axis.legend(frameon=False, fontsize=8)
    axis.grid(True, which="both", alpha=0.18)

    # Panel B: local slopes; these are diagnostics, not window selectors.  We
    # show the available finite trajectory even when the preregistered fitting
    # window is too short (as it intentionally is in the smoke profile).
    axis = axes[1]
    for theta in profile.representative_thetas:
        curve = largest_width_curves[theta]
        time = curve["times"]
        value = curve["noisy_centered_mean"]
        diagnostic_lower = max(float(time[1]), 0.05 * float(time[-1]))
        valid = (time >= diagnostic_lower) & (value > 0.0)
        slopes = local_log_slopes(
            time,
            value,
            valid=valid,
            half_window_decades=profile.local_slope_half_window_decades,
            minimum_points=7,
        )
        mask = np.isfinite(slopes)
        axis.semilogx(time[mask], slopes[mask], color=colors[theta], lw=2.0)
        axis.axhline(
            profile.total_exponent(theta),
            color=colors[theta],
            ls="--",
            lw=1.0,
            alpha=0.85,
        )
    axis.set_xlabel(r"intrinsic time $T$")
    axis.set_ylabel("local effective exponent")
    axis.set_title("(b) Local slopes (audit is fail-closed)")
    axis.grid(True, which="both", alpha=0.18)

    # Panel C: theory curve and measured fixed-window exponents.
    axis = axes[2]
    theta_grid = np.linspace(
        min(profile.theta_values), max(profile.theta_values), 400
    )
    theory = np.array([profile.total_exponent(value) for value in theta_grid])
    axis.plot(theta_grid, theory, color="black", lw=2.0, label="theory")
    measured_theta = []
    measured_exponent = []
    accepted = []
    for theta in profile.theta_values:
        fit = nominal_fits[theta]
        exponent = fit.get("exponent")
        if exponent is None or not np.isfinite(exponent):
            # A visible smoke-test diagnostic is useful, but it must remain an
            # open (inconclusive) marker and never enters summary.json as an
            # accepted fixed-window fit.
            curve = largest_width_curves.get(theta)
            if curve is None:
                curve = {
                    "times": np.array([], dtype=float),
                    "noisy_centered_mean": np.array([], dtype=float),
                }
            time = np.asarray(curve["times"], dtype=float)
            value = np.asarray(curve["noisy_centered_mean"], dtype=float)
            last_time = float(time[-1]) if time.size else 0.0
            fallback = (
                (time > 0.0)
                & (time >= 0.25 * last_time)
                & (value > 0.0)
            )
            if np.count_nonzero(fallback) < 3:
                continue
            exponent = -float(
                np.polyfit(
                    np.log10(time[fallback]),
                    np.log10(value[fallback]),
                    1,
                )[0]
            )
            fit = {**fit, "status": "INCONCLUSIVE"}
        measured_theta.append(theta)
        measured_exponent.append(exponent)
        accepted.append(fit["status"] == "PASS")
    measured_theta_array = np.asarray(measured_theta, dtype=float)
    measured_exponent_array = np.asarray(measured_exponent, dtype=float)
    accepted_array = np.asarray(accepted, dtype=bool)
    if np.any(accepted_array):
        axis.scatter(
            measured_theta_array[accepted_array],
            measured_exponent_array[accepted_array],
            s=45,
            facecolors="#4c956c",
            edgecolors="#2f4858",
            linewidths=1.2,
            label="accepted fixed window",
            zorder=3,
        )
    if np.any(~accepted_array):
        axis.scatter(
            measured_theta_array[~accepted_array],
            measured_exponent_array[~accepted_array],
            s=45,
            facecolors="white",
            edgecolors="#2f4858",
            linewidths=1.2,
            label="finite-window diagnostic",
            zorder=3,
        )
    axis.axvline(profile.destroy_boundary, color="#c43c39", ls=":", lw=1.2)
    axis.axvline(profile.preservation_boundary, color="#2878b5", ls=":", lw=1.2)
    axis.set_xlabel(r"batch-growth exponent $\vartheta$")
    axis.set_ylabel("total centered-risk exponent")
    axis.set_title("(c) Predicted versus measured phase law")
    axis.set_ylim(bottom=min(-0.05, axis.get_ylim()[0]))
    axis.legend(frameon=False, fontsize=8)
    axis.grid(True, alpha=0.18)

    fig.suptitle(
        rf"PLRF: $\alpha={profile.alpha:g}$, $\beta={profile.beta:g}$, "
        rf"$q_0={profile.q_clean:g}$, $q_{{\mathrm{{K}}}}={profile.q_kernel:g}$",
        fontsize=11,
        y=1.02,
    )
    fig.tight_layout()
    png_path = output_path.with_suffix(".png")
    pdf_path = output_path.with_suffix(".pdf")
    fig.savefig(png_path, dpi=220, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    return png_path, pdf_path
