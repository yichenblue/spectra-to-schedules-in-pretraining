"""Finite-bulk three-layer bridge for the PLRF online-SGD model.

The experiment compares, on identical discrete batch schedules,

1. Monte Carlo population online SGD;
2. the exact conditional finite-W Volterra recursion; and
3. the support-adaptive resolvent deterministic-equivalent Volterra recursion.

It is deliberately separate from the LM/IM preserve--change--destroy bridge.
In the finite-bulk regime the relevant schedule statistic is the cumulative
injection ``sum Delta T_s / r_s`` rather than a power-law memory convolution.
No manuscript source is edited by this module.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
from typing import Any

import numpy as np

from .bridge_validation import simulate_true_gaussian_sgd
from .dynamics import run_exact_dynamics, solve_two_time_volterra
from .resolvent_de import compute_density_level, density_to_modes
from .run_experiment_0b_finite_w_de_bridge import run_de_schedule
from .spectrum import EmpiricalSpectrum, build_empirical_spectrum


Array = np.ndarray
ROOT = Path(__file__).resolve().parent

AMBIENT_TO_WIDTH_RATIO = 2
INITIAL_BATCH = 16
SIGMA2 = 25.0
ETA_COEFFICIENT = 0.05
TRUE_SGD_SEED = 20260825
ROW_MASS_CAP = 0.8
VOL_TERRA_CROSSCHECK_STEPS = 64

MAXIMUM_TRUE_FINITE_RELATIVE_L2 = 0.04
MAXIMUM_FINITE_DE_RELATIVE_L2 = 0.04
MAXIMUM_TRUE_DE_RELATIVE_L2 = 0.05
MAXIMUM_DE_RESOLUTION_RELATIVE_L2 = 0.01
MAXIMUM_MODAL_RECURRENCE_ERROR = 1.0e-12
MAXIMUM_M_RESIDUAL = 5.0e-11
MAXIMUM_MASS_IDENTITY_ERROR = 1.0e-8


@dataclass(frozen=True)
class FBRegime:
    name: str
    alpha: float
    beta: float

    @property
    def p(self) -> float:
        return 2.0 * self.alpha + 2.0 * self.beta - 1.0

    @property
    def q_clean(self) -> float:
        return self.p / (2.0 * self.alpha)


@dataclass(frozen=True)
class Profile:
    name: str
    widths: tuple[int, ...]
    finite_w_seeds: tuple[int, ...]
    true_sgd_seeds: tuple[int, ...]
    trajectories: int
    de_comparison_level: int
    de_final_level: int
    initial_intervals: int

    @property
    def display_width(self) -> int:
        return max(self.widths)


REGIMES = (
    FBRegime(name="FB1", alpha=0.20, beta=0.40),
    FBRegime(name="FB2", alpha=0.12, beta=0.65),
)
SCHEDULES = (
    (0.0, "linear accumulation"),
    (1.0, "logarithmic accumulation"),
    (2.0, "summable injection"),
)

FULL_PROFILE = Profile(
    name="full",
    widths=(128, 256, 512),
    finite_w_seeds=(11, 23, 37, 53, 71, 89, 107, 131),
    true_sgd_seeds=(11, 23, 37, 53),
    trajectories=256,
    de_comparison_level=1,
    de_final_level=2,
    initial_intervals=40_000,
)
SMOKE_PROFILE = Profile(
    name="smoke",
    widths=(24, 32),
    finite_w_seeds=(11, 23),
    true_sgd_seeds=(11,),
    trajectories=32,
    de_comparison_level=0,
    de_final_level=1,
    initial_intervals=2_000,
)


def learning_rate(regime: FBRegime, width: int) -> float:
    """Finite-bulk peak-scaled learning rate."""

    return (
        ETA_COEFFICIENT
        * INITIAL_BATCH
        * float(width) ** (2.0 * regime.alpha - 1.0)
    )


def maximum_intrinsic_time(regime: FBRegime, width: int) -> float:
    """Triangular strict-window horizon used by both FB subregimes."""

    return 2.0 * float(width) ** (2.0 * regime.alpha) / math.log(float(width))


def horizon_steps(regime: FBRegime, width: int) -> int:
    return max(
        1,
        int(math.ceil(maximum_intrinsic_time(regime, width) / learning_rate(regime, width))),
    )


def _finite(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _finite(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_finite(item) for item in value]
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if math.isfinite(number) else None
    return value


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(_finite(payload), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text.rstrip() + "\n", encoding="utf-8")
    temporary.replace(path)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("cannot write an empty CSV")
    fields: list[str] = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _relative_l2(candidate: Array, reference: Array, *, skip_zero: bool = False) -> float:
    candidate = np.asarray(candidate, dtype=float)
    reference = np.asarray(reference, dtype=float)
    if candidate.shape != reference.shape:
        raise ValueError("curve shapes do not match")
    mask = np.isfinite(candidate) & np.isfinite(reference)
    if skip_zero:
        mask &= reference > 0.0
    denominator = float(np.linalg.norm(reference[mask]))
    if denominator <= 0.0:
        raise ValueError("relative-L2 reference is zero")
    return float(np.linalg.norm(candidate[mask] - reference[mask]) / denominator)


def _mean_trajectory(items: list[Any], field: str) -> Array:
    return np.mean(np.asarray([getattr(item, field) for item in items]), axis=0)


def _aggregate_standard_error(items: list[Any], field: str) -> Array:
    values = np.asarray([getattr(item, field) for item in items], dtype=float)
    return np.sqrt(np.sum(values * values, axis=0)) / len(items)


def _density_modes(
    regime: FBRegime,
    width: int,
    level: int,
    cache_dir: Path,
    initial_intervals: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    density = compute_density_level(
        phase=f"fb_three_layer_{regime.name}",
        alpha=regime.alpha,
        beta=regime.beta,
        width=width,
        level=level,
        cache_dir=cache_dir,
        initial_intervals=initial_intervals,
    )
    modes = density_to_modes(regime.alpha, regime.beta, width, density)
    diagnostics = {
        "regime": regime.name,
        "alpha": regime.alpha,
        "beta": regime.beta,
        "width": width,
        "level": level,
        "density_points": int(np.asarray(density["x"]).size),
        "max_m_residual": float(density["max_m_residual"]),
        "max_m_iterations": int(density["max_m_iterations"]),
        "kernel_mode_count": int(len(modes["kernel_nodes"])),
        "forcing_mode_count": int(len(modes["forcing_nodes"])),
        "raw_trace_mass_relative_error": float(
            modes["raw_trace_mass_relative_error"]
        ),
        "raw_target_mass_relative_error": float(
            modes["raw_target_mass_relative_error"]
        ),
        "raw_trace_first_moment_relative_error": float(
            modes["raw_trace_first_moment_relative_error"]
        ),
        "raw_target_first_moment_relative_error": float(
            modes["raw_target_first_moment_relative_error"]
        ),
        "trace_mass_identity_error": abs(
            float(np.sum(modes["kernel_trace_weights"])) - width
        ),
        "target_mass_identity_error": abs(
            float(np.sum(modes["forcing_weights"]))
            + float(modes["null_weight"])
            - float(modes["target_energy"])
        ),
    }
    return modes, diagnostics


def _build_spectra(profile: Profile) -> dict[tuple[str, int, int], EmpiricalSpectrum]:
    spectra: dict[tuple[str, int, int], EmpiricalSpectrum] = {}
    for regime in REGIMES:
        for width in profile.widths:
            for seed in profile.finite_w_seeds:
                spectra[(regime.name, width, seed)] = build_empirical_spectrum(
                    alpha=regime.alpha,
                    beta=regime.beta,
                    width=width,
                    ambient_to_width_ratio=AMBIENT_TO_WIDTH_RATIO,
                    seed=seed,
                )
    return spectra


def _marker_indices(size: int, target: int = 9) -> Array:
    if size <= target:
        return np.arange(size, dtype=int)
    return np.unique(np.rint(np.geomspace(1, size, target)).astype(int) - 1)


def _plot(
    curve_rows: list[dict[str, Any]], output_dir: Path
) -> tuple[Path, Path]:
    os.environ.setdefault("MPLCONFIGDIR", str(output_dir / ".matplotlib"))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.ticker import FixedFormatter, FixedLocator, NullFormatter

    plt.rcParams.update(
        {
            "font.family": "STIXGeneral",
            "mathtext.fontset": "stix",
            "font.size": 6.7,
            "axes.titlesize": 7.0,
            "axes.labelsize": 7.0,
            "legend.fontsize": 6.35,
            "xtick.labelsize": 6.5,
            "ytick.labelsize": 6.5,
            "axes.linewidth": 0.70,
            "lines.solid_capstyle": "round",
            "lines.dash_capstyle": "round",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    colors = {
        "linear accumulation": "#AA3377",
        "logarithmic accumulation": "#EE7733",
        "summable injection": "#4477AA",
    }
    panels = (
        ("FB1", "total", r"(a) $\mathrm{FB}_1$: centered risk"),
        ("FB2", "total", r"(b) $\mathrm{FB}_2$: centered risk"),
        ("FB1", "gap", r"(c) $\mathrm{FB}_1$: noisy--clean gap"),
        ("FB2", "gap", r"(d) $\mathrm{FB}_2$: noisy--clean gap"),
    )
    figure = plt.figure(figsize=(5.50, 2.48))
    grid = figure.add_gridspec(
        1,
        5,
        left=0.085,
        right=0.992,
        bottom=0.365,
        top=0.895,
        width_ratios=(1.0, 1.0, 0.22, 1.0, 1.0),
        wspace=0.10,
    )
    total_fb1_axis = figure.add_subplot(grid[0, 0])
    total_fb2_axis = figure.add_subplot(
        grid[0, 1], sharex=total_fb1_axis, sharey=total_fb1_axis
    )
    gap_fb1_axis = figure.add_subplot(grid[0, 3], sharex=total_fb1_axis)
    gap_fb2_axis = figure.add_subplot(
        grid[0, 4], sharex=total_fb1_axis, sharey=gap_fb1_axis
    )
    axes = [total_fb1_axis, total_fb2_axis, gap_fb1_axis, gap_fb2_axis]
    for axis, (regime, observable, title) in zip(axes, panels, strict=True):
        for _, schedule_label in SCHEDULES:
            selected = [
                row
                for row in curve_rows
                if row["regime"] == regime and row["schedule_label"] == schedule_label
            ]
            selected.sort(key=lambda row: int(row["iteration"]))
            times = np.asarray([row["intrinsic_time"] for row in selected])
            if observable == "total":
                true = np.asarray([row["true_sgd_noisy_centered"] for row in selected])
                true_se = np.asarray(
                    [row["true_sgd_noisy_centered_se"] for row in selected]
                )
                finite = np.asarray([row["finite_w_noisy_centered"] for row in selected])
                de = np.asarray([row["de_noisy_centered"] for row in selected])
            else:
                true = np.asarray([row["true_sgd_noise_gap"] for row in selected])
                true_se = np.asarray([row["true_sgd_noise_gap_se"] for row in selected])
                finite = np.asarray([row["finite_w_noise_gap"] for row in selected])
                de = np.asarray([row["de_noise_gap"] for row in selected])
            positive = (times > 0.0) & (true > 0.0) & (finite > 0.0) & (de > 0.0)
            plotted_times = times[positive]
            plotted_true = true[positive]
            plotted_se = true_se[positive]
            plotted_finite = finite[positive]
            plotted_de = de[positive]
            color = colors[schedule_label]
            axis.loglog(
                plotted_times,
                plotted_finite,
                color=color,
                linewidth=1.60,
                alpha=0.40,
                zorder=1,
            )
            axis.loglog(
                plotted_times,
                plotted_de,
                color=color,
                linestyle=(0, (5.5, 2.6)),
                linewidth=1.55,
                zorder=2,
            )
            marker_indices = _marker_indices(plotted_times.size)
            axis.errorbar(
                plotted_times[marker_indices],
                plotted_true[marker_indices],
                yerr=2.0 * plotted_se[marker_indices],
                linestyle="none",
                marker="o",
                markersize=3.0,
                markerfacecolor="white",
                markeredgewidth=0.80,
                color=color,
                elinewidth=0.60,
                capsize=1.15,
                zorder=3,
            )
        axis.set_title(title, pad=2.6)
        axis.set_axisbelow(True)
        axis.grid(True, which="major", color="#A0A0A0", alpha=0.20, linewidth=0.38)
        axis.grid(True, which="minor", color="#B8B8B8", alpha=0.08, linewidth=0.28)
        axis.tick_params(which="major", length=2.3, width=0.65, pad=1.2)
        axis.tick_params(which="minor", length=1.4, width=0.45)
    axes[0].set_ylabel(r"$R_{\sigma,t}-R_{\mathrm{app}}$", labelpad=0.8)
    axes[2].set_ylabel(r"$R_{\sigma,t}-R_{0,t}$", labelpad=0.8)
    axes[0].yaxis.set_major_locator(FixedLocator([1.0]))
    axes[0].yaxis.set_major_formatter(FixedFormatter([r"$10^0$"]))
    axes[2].yaxis.set_major_locator(FixedLocator([1.0]))
    axes[2].yaxis.set_major_formatter(FixedFormatter([r"$10^0$"]))
    for axis in axes:
        axis.yaxis.set_minor_formatter(NullFormatter())
    axes[1].yaxis.set_visible(False)
    axes[3].yaxis.set_visible(False)

    schedule_handles = [
        Line2D(
            [0],
            [0],
            color=colors[label],
            linewidth=1.75,
            label=rf"$\vartheta={theta:g}$: {label}",
        )
        for theta, label in SCHEDULES
    ]
    layer_handles = [
        Line2D(
            [0],
            [0],
            color="#333333",
            marker="o",
            markerfacecolor="white",
            linestyle="none",
            markersize=3.5,
            label="true SGD mean",
        ),
        Line2D(
            [0],
            [0],
            color="#333333",
            linewidth=1.60,
            alpha=0.45,
            label=r"finite-$W$ Volterra",
        ),
        Line2D(
            [0],
            [0],
            color="#333333",
            linestyle=(0, (5.5, 2.6)),
            linewidth=1.55,
            label="DE Volterra",
        ),
    ]
    legend_handles = [
        layer_handles[0],
        schedule_handles[0],
        layer_handles[1],
        schedule_handles[1],
        layer_handles[2],
        schedule_handles[2],
    ]
    figure.legend(
        handles=legend_handles,
        frameon=False,
        ncol=3,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.018),
        borderaxespad=0.0,
        handlelength=2.05,
        handletextpad=0.40,
        columnspacing=1.05,
        labelspacing=0.32,
    )
    figure.supxlabel(r"intrinsic time $T_t$", fontsize=7.0, y=0.270)
    output_dir.mkdir(parents=True, exist_ok=True)
    png = output_dir / "fb_volterra_three_layer_bridge.png"
    pdf = output_dir / "fb_volterra_three_layer_bridge.pdf"
    figure.savefig(png, dpi=400)
    figure.savefig(pdf)
    plt.close(figure)
    return png, pdf


def run(profile: Profile = FULL_PROFILE) -> dict[str, Any]:
    output_dir = ROOT / "artifacts" / f"fb_three_layer_bridge_{profile.name}"
    density_cache = output_dir / "cache" / "support_adaptive_density"
    spectra = _build_spectra(profile)
    curve_rows: list[dict[str, Any]] = []
    metric_rows: list[dict[str, Any]] = []
    spectral_rows: list[dict[str, Any]] = []
    maximum_modal_error = 0.0
    all_exact_stable = True
    all_de_stable = True

    for regime_index, regime in enumerate(REGIMES):
        for width in profile.widths:
            eta = learning_rate(regime, width)
            maximum_time = maximum_intrinsic_time(regime, width)
            horizon = horizon_steps(regime, width)
            comparison_modes, comparison_diagnostics = _density_modes(
                regime,
                width,
                profile.de_comparison_level,
                density_cache,
                profile.initial_intervals,
            )
            final_modes, final_diagnostics = _density_modes(
                regime,
                width,
                profile.de_final_level,
                density_cache,
                profile.initial_intervals,
            )
            spectral_rows.extend([comparison_diagnostics, final_diagnostics])

            for schedule_index, (theta, schedule_label) in enumerate(SCHEDULES):
                exact_items = []
                true_items = []
                true_exact_items = []
                true_centered_totals: list[Array] = []
                for seed_index, seed in enumerate(profile.finite_w_seeds):
                    spectrum = spectra[(regime.name, width, seed)]
                    exact = run_exact_dynamics(
                        spectrum=spectrum,
                        eta=eta,
                        theta=theta,
                        initial_batch=INITIAL_BATCH,
                        sigma2=SIGMA2,
                        horizon=horizon,
                        row_mass_cap=ROW_MASS_CAP,
                    )
                    exact_items.append(exact)
                    all_exact_stable &= bool(exact.pointwise_stable and exact.row_stable)
                    if seed in profile.true_sgd_seeds:
                        true = simulate_true_gaussian_sgd(
                            spectrum=spectrum,
                            eta=eta,
                            batches=exact.batches,
                            sigma2=SIGMA2,
                            trajectories=profile.trajectories,
                            seed=(
                                TRUE_SGD_SEED
                                + 10_000 * regime_index
                                + 1_000 * schedule_index
                                + seed_index
                            ),
                            trajectory_chunk_size=64,
                        )
                        true_items.append(true)
                        true_exact_items.append(exact)
                        true_centered_totals.append(
                            true.total - spectrum.approximation_floor
                        )

                probe_spectrum = spectra[
                    (regime.name, width, profile.finite_w_seeds[0])
                ]
                crosscheck_horizon = min(VOL_TERRA_CROSSCHECK_STEPS, horizon)
                explicit = solve_two_time_volterra(
                    spectrum=probe_spectrum,
                    eta=eta,
                    theta=theta,
                    initial_batch=INITIAL_BATCH,
                    sigma2=SIGMA2,
                    horizon=crosscheck_horizon,
                )
                modal_prefix = run_exact_dynamics(
                    spectrum=probe_spectrum,
                    eta=eta,
                    theta=theta,
                    initial_batch=INITIAL_BATCH,
                    sigma2=SIGMA2,
                    horizon=crosscheck_horizon,
                    row_mass_cap=ROW_MASS_CAP,
                )
                modal_scale = max(float(np.max(explicit.noisy_total)), 1.0)
                modal_error = float(
                    np.max(np.abs(explicit.noisy_total - modal_prefix.noisy_total))
                    / modal_scale
                )
                maximum_modal_error = max(maximum_modal_error, modal_error)

                de_comparison = run_de_schedule(
                    comparison_modes,
                    eta,
                    theta,
                    INITIAL_BATCH,
                    SIGMA2,
                    horizon,
                    ROW_MASS_CAP,
                )
                de_final = run_de_schedule(
                    final_modes,
                    eta,
                    theta,
                    INITIAL_BATCH,
                    SIGMA2,
                    horizon,
                    ROW_MASS_CAP,
                )
                all_de_stable &= bool(
                    de_comparison.pointwise_stable
                    and de_comparison.row_stable
                    and de_final.pointwise_stable
                    and de_final.row_stable
                )

                finite_all_total = _mean_trajectory(exact_items, "noisy_centered")
                finite_all_gap = _mean_trajectory(exact_items, "noise_gap")
                finite_de_total = _relative_l2(
                    finite_all_total, de_final.noisy_centered
                )
                finite_de_gap = _relative_l2(
                    finite_all_gap, de_final.noise_gap, skip_zero=True
                )
                resolution_total = _relative_l2(
                    de_comparison.noisy_centered, de_final.noisy_centered
                )
                resolution_gap = _relative_l2(
                    de_comparison.noise_gap, de_final.noise_gap, skip_zero=True
                )

                metric: dict[str, Any] = {
                    "regime": regime.name,
                    "alpha": regime.alpha,
                    "beta": regime.beta,
                    "p": regime.p,
                    "q_clean": regime.q_clean,
                    "width": width,
                    "ambient_dimension": AMBIENT_TO_WIDTH_RATIO * width,
                    "eta": eta,
                    "theta": theta,
                    "schedule_label": schedule_label,
                    "maximum_intrinsic_time": maximum_time,
                    "horizon": horizon,
                    "finite_de_total_relative_l2": finite_de_total,
                    "finite_de_gap_relative_l2": finite_de_gap,
                    "de_resolution_total_relative_l2": resolution_total,
                    "de_resolution_gap_relative_l2": resolution_gap,
                    "modal_recurrence_relative_error": modal_error,
                    "maximum_exact_pointwise_product": max(
                        float(item.maximum_pointwise_product) for item in exact_items
                    ),
                    "maximum_exact_row_mass": max(
                        float(item.maximum_row_mass) for item in exact_items
                    ),
                    "maximum_de_pointwise_product": max(
                        de_comparison.maximum_pointwise_product,
                        de_final.maximum_pointwise_product,
                    ),
                    "maximum_de_row_mass": max(
                        de_comparison.maximum_row_mass, de_final.maximum_row_mass
                    ),
                }

                if width == profile.display_width:
                    finite_plot_total = _mean_trajectory(
                        true_exact_items, "noisy_centered"
                    )
                    finite_plot_gap = _mean_trajectory(true_exact_items, "noise_gap")
                    true_total = np.mean(
                        np.asarray(true_centered_totals, dtype=float), axis=0
                    )
                    true_gap = _mean_trajectory(true_items, "gap")
                    true_total_se = _aggregate_standard_error(
                        true_items, "total_standard_error"
                    )
                    true_gap_se = _aggregate_standard_error(
                        true_items, "gap_standard_error"
                    )
                    metric.update(
                        {
                            "true_finite_total_relative_l2": _relative_l2(
                                true_total, finite_plot_total
                            ),
                            "true_finite_gap_relative_l2": _relative_l2(
                                true_gap, finite_plot_gap, skip_zero=True
                            ),
                            "true_de_total_relative_l2": _relative_l2(
                                true_total, de_final.noisy_centered
                            ),
                            "true_de_gap_relative_l2": _relative_l2(
                                true_gap, de_final.noise_gap, skip_zero=True
                            ),
                            "display_finite_de_total_relative_l2": _relative_l2(
                                finite_plot_total, de_final.noisy_centered
                            ),
                            "display_finite_de_gap_relative_l2": _relative_l2(
                                finite_plot_gap,
                                de_final.noise_gap,
                                skip_zero=True,
                            ),
                        }
                    )
                    for index, time in enumerate(de_final.times):
                        curve_rows.append(
                            {
                                "regime": regime.name,
                                "alpha": regime.alpha,
                                "beta": regime.beta,
                                "p": regime.p,
                                "q_clean": regime.q_clean,
                                "width": width,
                                "ambient_dimension": AMBIENT_TO_WIDTH_RATIO * width,
                                "eta": eta,
                                "theta": theta,
                                "schedule_label": schedule_label,
                                "iteration": index,
                                "intrinsic_time": float(time),
                                "batch": (
                                    int(de_final.batches[index])
                                    if index < horizon
                                    else ""
                                ),
                                "true_sgd_noisy_centered": float(true_total[index]),
                                "true_sgd_noisy_centered_se": float(
                                    true_total_se[index]
                                ),
                                "finite_w_noisy_centered": float(
                                    finite_plot_total[index]
                                ),
                                "de_noisy_centered": float(
                                    de_final.noisy_centered[index]
                                ),
                                "true_sgd_noise_gap": float(true_gap[index]),
                                "true_sgd_noise_gap_se": float(true_gap_se[index]),
                                "finite_w_noise_gap": float(finite_plot_gap[index]),
                                "de_noise_gap": float(de_final.noise_gap[index]),
                            }
                        )
                metric_rows.append(metric)

    maximum_true_finite = max(
        max(
            float(row["true_finite_total_relative_l2"]),
            float(row["true_finite_gap_relative_l2"]),
        )
        for row in metric_rows
        if "true_finite_total_relative_l2" in row
    )
    maximum_audit_finite_de = max(
        max(
            float(row["finite_de_total_relative_l2"]),
            float(row["finite_de_gap_relative_l2"]),
        )
        for row in metric_rows
        if int(row["width"]) == profile.display_width
    )
    maximum_display_finite_de = max(
        max(
            float(row["display_finite_de_total_relative_l2"]),
            float(row["display_finite_de_gap_relative_l2"]),
        )
        for row in metric_rows
        if "display_finite_de_total_relative_l2" in row
    )
    maximum_true_de = max(
        max(
            float(row["true_de_total_relative_l2"]),
            float(row["true_de_gap_relative_l2"]),
        )
        for row in metric_rows
        if "true_de_total_relative_l2" in row
    )
    maximum_de_resolution = max(
        max(
            float(row["de_resolution_total_relative_l2"]),
            float(row["de_resolution_gap_relative_l2"]),
        )
        for row in metric_rows
    )
    maximum_m_residual = max(float(row["max_m_residual"]) for row in spectral_rows)
    maximum_mass_identity_error = max(
        max(
            float(row["trace_mass_identity_error"]),
            float(row["target_mass_identity_error"]),
        )
        for row in spectral_rows
    )
    # The smoke profile checks wiring on tiny widths and a small Monte Carlo
    # ensemble; it is not a scientific acceptance run.  Its looser route
    # tolerances prevent finite-size/Monte-Carlo noise from obscuring that
    # structural purpose.  Every paper-facing value uses the frozen full gates.
    true_finite_tolerance = (
        MAXIMUM_TRUE_FINITE_RELATIVE_L2 if profile.name == "full" else 0.15
    )
    finite_de_tolerance = (
        MAXIMUM_FINITE_DE_RELATIVE_L2 if profile.name == "full" else 0.10
    )
    true_de_tolerance = (
        MAXIMUM_TRUE_DE_RELATIVE_L2 if profile.name == "full" else 0.20
    )
    gate_pass = bool(
        all_exact_stable
        and all_de_stable
        and maximum_true_finite <= true_finite_tolerance
        and maximum_display_finite_de <= finite_de_tolerance
        and maximum_audit_finite_de <= finite_de_tolerance
        and maximum_true_de <= true_de_tolerance
        and maximum_de_resolution <= MAXIMUM_DE_RESOLUTION_RELATIVE_L2
        and maximum_modal_error <= MAXIMUM_MODAL_RECURRENCE_ERROR
        and maximum_m_residual <= MAXIMUM_M_RESIDUAL
        and maximum_mass_identity_error <= MAXIMUM_MASS_IDENTITY_ERROR
    )

    _write_csv(output_dir / "curves.csv", curve_rows)
    _write_csv(output_dir / "metrics.csv", metric_rows)
    _write_csv(output_dir / "spectral_diagnostics.csv", spectral_rows)
    png, pdf = _plot(curve_rows, output_dir)
    caption_path = output_dir / "caption.txt"
    _write_text(
        caption_path,
        (
            "Finite-bulk numerical bridge from true online SGD to the exact "
            "conditional finite-W Volterra recursion and then to resolvent-DE "
            "Volterra. We use FB1 at (alpha,beta)=(0.20,0.40), FB2 at "
            f"(0.12,0.65), d/m=2, sigma^2=25, and display m={profile.display_width}. "
            "The batch "
            "ramps B_t=ceil[16(1+T_t)^theta] with theta=0,1,2 make the "
            "cumulative injection sum_{s<t} Delta T_s/r_s linear, logarithmic, "
            "and summable, respectively. "
            f"Open circles are true-SGD means over {profile.trajectories} "
            f"trajectories for each of {len(profile.true_sgd_seeds)} feature "
            "matrices, averaged over the same matrices; error bars "
            "are +/-2 trajectory Monte Carlo standard errors conditional on "
            "the sampled feature matrices. Light solid curves are the "
            "corresponding exact finite-W Volterra risks, and long-dashed curves "
            "are the DE Volterra risks. Across all six displayed schedules and "
            "both observables, the maximum relative-L2 discrepancies are "
            f"{100.0 * maximum_true_finite:.2f}% (SGD--finite W), "
            f"{100.0 * maximum_display_finite_de:.2f}% (finite W--DE), and "
            f"{100.0 * maximum_true_de:.2f}% (SGD--DE). The learning rate is "
            "eta_m=0.05 B_0 m^(2 alpha-1), and the triangular horizon "
            "T_max(m)=2m^(2 alpha)/log m is o(m^(2 alpha)). This is a "
            "finite-width, finite-horizon authenticity check, not a uniform "
            "random-matrix theorem or an asymptotic-exponent test."
        ),
    )
    summary = {
        "experiment": "fb_specific_three_layer_bridge",
        "status": "PASS" if gate_pass else "FAIL",
        "claim_scope": (
            "finite-width, finite-horizon authenticity bridge for two "
            "finite-bulk PLRF subregimes; not a uniform random-matrix theorem, "
            "an asymptotic-window validation, or an exponent test"
        ),
        "profile": profile.name,
        "parameters": {
            "ambient_to_width_ratio": AMBIENT_TO_WIDTH_RATIO,
            "widths": profile.widths,
            "display_width": profile.display_width,
            "finite_w_seeds": profile.finite_w_seeds,
            "true_sgd_seeds": profile.true_sgd_seeds,
            "trajectories_per_feature_matrix": profile.trajectories,
            "initial_batch": INITIAL_BATCH,
            "sigma2": SIGMA2,
            "eta_schedule": "eta_m=0.05*B0*m^(2*alpha-1)",
            "triangular_horizon": "T_max(m)=2*m^(2*alpha)/log(m)",
            "schedule": "B_t=ceil(B0*(1+T_t)^theta)",
            "regimes": [
                {
                    "name": regime.name,
                    "alpha": regime.alpha,
                    "beta": regime.beta,
                    "p": regime.p,
                    "q_clean": regime.q_clean,
                }
                for regime in REGIMES
            ],
            "schedules": [
                {"theta": theta, "finite_bulk_response": label}
                for theta, label in SCHEDULES
            ],
            "de_comparison_level": profile.de_comparison_level,
            "de_final_level": profile.de_final_level,
            "initial_density_intervals": profile.initial_intervals,
        },
        "gates": {
            "all_exact_stable": all_exact_stable,
            "all_de_stable": all_de_stable,
            "maximum_true_finite_relative_l2": maximum_true_finite,
            "true_finite_tolerance": true_finite_tolerance,
            "maximum_display_finite_de_relative_l2": maximum_display_finite_de,
            "maximum_eight_seed_finite_de_relative_l2": maximum_audit_finite_de,
            "finite_de_tolerance": finite_de_tolerance,
            "maximum_true_de_relative_l2": maximum_true_de,
            "true_de_tolerance": true_de_tolerance,
            "maximum_de_resolution_relative_l2": maximum_de_resolution,
            "de_resolution_tolerance": MAXIMUM_DE_RESOLUTION_RELATIVE_L2,
            "maximum_modal_recurrence_relative_error": maximum_modal_error,
            "modal_recurrence_tolerance": MAXIMUM_MODAL_RECURRENCE_ERROR,
            "maximum_m_residual": maximum_m_residual,
            "m_residual_tolerance": MAXIMUM_M_RESIDUAL,
            "maximum_mass_identity_error": maximum_mass_identity_error,
            "mass_identity_tolerance": MAXIMUM_MASS_IDENTITY_ERROR,
            "all_gates_pass": gate_pass,
        },
        "outputs": {
            "curves": str(output_dir / "curves.csv"),
            "metrics": str(output_dir / "metrics.csv"),
            "spectral_diagnostics": str(output_dir / "spectral_diagnostics.csv"),
            "png": str(png),
            "pdf": str(pdf),
            "caption": str(caption_path),
        },
    }
    _write_json(output_dir / "summary.json", summary)
    if not gate_pass:
        raise RuntimeError(f"FB three-layer bridge failed: {summary['gates']}")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=("smoke", "full"), default="full")
    arguments = parser.parse_args()
    profile = FULL_PROFILE if arguments.profile == "full" else SMOKE_PROFILE
    summary = run(profile)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
