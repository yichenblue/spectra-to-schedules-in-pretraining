"""Plots for the true-SGD/discrete/continuum authenticity bridge."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from .bridge_config import BridgeProfile
from .bridge_validation import BridgeCase


def _observable_curves(
    case: BridgeCase,
    observable: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if observable == "clean":
        return (
            case.exact.clean_total,
            case.approximation_floor + case.continuum.clean_centered,
            case.sampled.clean,
            case.sampled.clean_standard_error,
        )
    if observable == "gap":
        return (
            case.exact.noise_gap,
            case.continuum.noise_gap,
            case.sampled.gap,
            case.sampled.gap_standard_error,
        )
    if observable == "total":
        return (
            case.exact.noisy_total,
            case.approximation_floor + case.continuum.noisy_centered,
            case.sampled.total,
            case.sampled.total_standard_error,
        )
    raise KeyError(observable)


def make_bridge_figures(
    profile: BridgeProfile,
    cases: dict[tuple[int, float], BridgeCase],
    output_dir: Path,
) -> list[Path]:
    """Write one 3x4 overlay figure for each phase representative."""

    output_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(output_dir / ".matplotlib"))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    outputs: list[Path] = []
    observables = ("clean", "gap", "total")
    labels = {
        "clean": "clean population risk",
        "gap": "noisy-clean gap",
        "total": "noisy population risk",
    }
    for theta in profile.theta_values:
        figure, axes = plt.subplots(
            len(observables),
            len(profile.widths),
            figsize=(3.2 + 2.75 * len(profile.widths), 8.0),
            sharex=False,
            sharey=False,
            squeeze=False,
        )
        for column, width in enumerate(profile.widths):
            case = cases[(width, theta)]
            times = case.exact.times
            fit_lower = float(width) ** profile.fit_lower_exponent
            fit_upper = float(width) ** profile.fit_upper_exponent
            for row, observable in enumerate(observables):
                axis = axes[row, column]
                exact, continuum, sampled, standard_error = _observable_curves(
                    case, observable
                )
                valid = times > 0.0
                if observable == "gap":
                    valid &= exact > 0.0
                axis.loglog(
                    times[valid],
                    exact[valid],
                    color="#222222",
                    linewidth=2.0,
                    label="discrete exact",
                    zorder=3,
                )
                axis.loglog(
                    times[valid],
                    continuum[valid],
                    color="#2878b5",
                    linewidth=1.8,
                    linestyle="--",
                    label="continuum",
                    zorder=2,
                )
                axis.loglog(
                    times[valid],
                    sampled[valid],
                    color="#4c956c",
                    linewidth=1.1,
                    alpha=0.9,
                    label="true-SGD mean",
                    zorder=4,
                )
                lower = np.maximum(
                    sampled - 2.0 * standard_error,
                    np.finfo(float).tiny,
                )
                upper = sampled + 2.0 * standard_error
                axis.fill_between(
                    times[valid],
                    lower[valid],
                    upper[valid],
                    color="#4c956c",
                    alpha=0.14,
                    linewidth=0.0,
                    zorder=1,
                )
                axis.axvspan(fit_lower, fit_upper, color="#d18700", alpha=0.06)
                axis.grid(True, which="both", alpha=0.16)
                if row == 0:
                    axis.set_title(rf"$m={width}$")
                if column == 0:
                    axis.set_ylabel(labels[observable])
                if row == len(observables) - 1:
                    axis.set_xlabel(r"intrinsic time $T$")
        axes[0, 0].legend(frameon=False, fontsize=8)
        figure.suptitle(
            rf"Authenticity bridge: $\vartheta={theta:g}$, "
            rf"$\eta={profile.eta:g}$, $B_0={profile.initial_batch}$",
            fontsize=12,
            y=1.005,
        )
        figure.tight_layout()
        theta_label = f"{theta:g}".replace(".", "p")
        png = output_dir / f"bridge_theta_{theta_label}.png"
        pdf = output_dir / f"bridge_theta_{theta_label}.pdf"
        figure.savefig(png, dpi=220, bbox_inches="tight")
        figure.savefig(pdf, bbox_inches="tight")
        plt.close(figure)
        outputs.extend([png, pdf])
    return outputs
