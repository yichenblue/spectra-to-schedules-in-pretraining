"""Build a six-construction summary from completed literal-SGD artifacts."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from .common import log_ols_exponent, write_json
from .spectral_long_horizon_counterexamples import (
    local_exponent_curve,
    periodic_power_fit,
)


Array = np.ndarray
ROOT = Path(__file__).resolve().parent
ARTIFACTS = ROOT / "artifacts"
OUTPUT = ARTIFACTS / "spectral_six_construction_summary_v001"
WIDTH_SOURCE = ARTIFACTS / "spectral_loss_sgd_width_time_confirmation_v003" / "curves.csv"
STRETCHED_SOURCE = ARTIFACTS / "spectral_stretched_t1e6_v001" / "curves.csv"
GEOMETRIC_SOURCE = ARTIFACTS / "spectral_geometric_spacing_sweep_pilot_v001" / "curves.csv"
RAPID_SOURCE = ARTIFACTS / "spectral_rapid_target_mechanism_pilot_v001" / "curves.csv"
MISSING_WIDTH_STRETCHED_SOURCE = (
    ARTIFACTS / "spectral_missing_width_overlays_v001" / "stretched_curves.csv"
)
MISSING_WIDTH_GEOMETRIC_SOURCE = (
    ARTIFACTS / "spectral_missing_width_overlays_v001" / "geometric_curves.csv"
)
POLYNOMIAL_SMALL_WIDTH_SOURCE = (
    ARTIFACTS / "spectral_polynomial_small_width_extension_v001" / "curves.csv"
)
CONSTRUCTION_TITLES = {
    "polynomial_canonical": (
        r"$\lambda_j=j^{-0.8}$" "\n"
        r"$|\theta_j^\star|^2\propto j^{-0.6}$"
    ),
    "polynomial_sparse_target": (
        r"$\lambda_j=j^{-0.8}$" "\n"
        r"$|\theta_j^\star|^2\propto j^{-0.6}(2k-1)\mathbf{1}_{j=k^2}$"
    ),
    "stretched_exponential": (
        r"$\lambda_j=e^{1-\sqrt{j}}$" "\n"
        r"$|\theta_j^\star|^2\propto j^{-3/2}$"
    ),
    "spectral_gap": (
        r"$\lambda_{1:64}=\operatorname{geomspace}(1,0.1)$" "\n"
        r"$\lambda_{j>64}\approx0,\ |\theta_j^\star|^2\propto j^{-1}$"
    ),
    "geometric_spectrum": (
        r"$\lambda_j=e^{-1.4(j-1)}$" "\n"
        r"$|\theta_j^\star|^2\propto\lambda_j^{0.75}$"
    ),
    "polynomial_rapid_target": (
        r"$\lambda_j=j^{-0.8}$" "\n"
        r"$|\theta_j^\star|^2\propto e^{-j}$"
    ),
}


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _stack(
    rows: list[dict[str, str]],
    *,
    filters: dict[str, str],
    value_column: str,
) -> tuple[Array, Array, Array]:
    selected = [
        row for row in rows
        if all(row.get(key) == value for key, value in filters.items())
    ]
    by_seed: dict[str, list[tuple[float, float]]] = {}
    for row in selected:
        by_seed.setdefault(row["seed"], []).append(
            (float(row["intrinsic_time"]), float(row[value_column]))
        )
    if not by_seed:
        raise RuntimeError(f"no rows matched {filters}")
    common_times: Array | None = None
    traces = []
    for points in by_seed.values():
        ordered = sorted(points)
        times = np.asarray([point[0] for point in ordered], dtype=float)
        values = np.asarray([point[1] for point in ordered], dtype=float)
        if common_times is None:
            common_times = times
        elif not np.array_equal(common_times, times):
            raise RuntimeError(f"non-common evaluation grid for {filters}")
        traces.append(values)
    assert common_times is not None
    return common_times, np.stack(traces), np.asarray(list(by_seed), dtype=object)


def _median_band(traces: Array) -> tuple[Array, Array, Array]:
    return (
        np.median(traces, axis=0),
        np.quantile(traces, 0.25, axis=0),
        np.quantile(traces, 0.75, axis=0),
    )


def _power_reference(times: Array, values: Array, exponent: float, anchor: float) -> Array:
    index = int(np.argmin(np.abs(times - anchor)))
    return values[index] * np.power(times / times[index], -exponent)


def _set_plot_style(plt: Any) -> None:
    plt.rcParams.update({
        "font.size": 13,
        "axes.titlesize": 15,
        "axes.labelsize": 14,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "legend.fontsize": 11,
        "figure.titlesize": 18,
        "lines.linewidth": 2.8,
    })


def load_data() -> dict[str, dict[str, Any]]:
    width_rows = _read_rows(WIDTH_SOURCE)
    stretched_rows = _read_rows(STRETCHED_SOURCE)
    geometric_rows = _read_rows(GEOMETRIC_SOURCE)
    rapid_rows = _read_rows(RAPID_SOURCE)
    missing_stretched_rows = _read_rows(MISSING_WIDTH_STRETCHED_SOURCE)
    missing_geometric_rows = _read_rows(MISSING_WIDTH_GEOMETRIC_SOURCE)
    polynomial_small_width_rows = _read_rows(POLYNOMIAL_SMALL_WIDTH_SOURCE)
    data: dict[str, dict[str, Any]] = {}
    for name in (
        "polynomial_canonical", "polynomial_sparse_target",
        "polynomial_rapid_target", "spectral_gap",
    ):
        width_groups: dict[int, dict[str, Any]] = {}
        for width in (4096, 16384, 65536):
            source_rows = (
                polynomial_small_width_rows
                if name in {"polynomial_canonical", "polynomial_sparse_target"}
                and width in {4096, 16384}
                else width_rows
            )
            times, traces, seeds = _stack(
                source_rows,
                filters={"case": name, "width": str(width)},
                value_column="population_excess_risk",
            )
            width_groups[width] = {"times": times, "traces": traces, "seeds": seeds}
        primary = width_groups[65536]
        data[name] = {**primary, "width_groups": width_groups}
    times, traces, seeds = _stack(
        stretched_rows,
        filters={"case": "stretched_exponential", "width": "2048"},
        value_column="population_excess_risk",
    )
    stretched_width_groups = {
        2048: {"times": times, "traces": traces, "seeds": seeds},
    }
    for width in (256, 512):
        times, traces, seeds = _stack(
            missing_stretched_rows,
            filters={"case": "stretched_exponential", "width": str(width)},
            value_column="population_excess_risk",
        )
        stretched_width_groups[width] = {
            "times": times, "traces": traces, "seeds": seeds,
        }
    primary = stretched_width_groups[2048]
    data["stretched_exponential"] = {
        **primary, "width_groups": stretched_width_groups,
    }
    times, traces, seeds = _stack(
        geometric_rows,
        filters={"spacing": "1.4", "width": "2048"},
        value_column="population_excess_risk",
    )
    geometric_width_groups = {
        2048: {"times": times, "traces": traces, "seeds": seeds},
    }
    for width in (16, 64):
        times, traces, seeds = _stack(
            missing_geometric_rows,
            filters={"spacing": "1.4", "width": str(width)},
            value_column="population_excess_risk",
        )
        geometric_width_groups[width] = {
            "times": times, "traces": traces, "seeds": seeds,
        }
    primary = geometric_width_groups[2048]
    data["geometric_spectrum"] = {
        **primary, "width_groups": geometric_width_groups,
    }
    rapid: dict[int, dict[str, Any]] = {}
    for batch_size in (16, 64, 256):
        times, traces, seeds = _stack(
            rapid_rows,
            filters={
                "condition": "sampled_minibatch_sgd",
                "batch_size": str(batch_size),
            },
            value_column="population_risk",
        )
        rapid[batch_size] = {"times": times, "traces": traces, "seeds": seeds}
    data["polynomial_rapid_target"]["batch_groups"] = rapid
    return data


def summarize(data: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for name in ("polynomial_canonical", "polynomial_sparse_target"):
        times = data[name]["times"]
        median = np.median(data[name]["traces"], axis=0)
        mask = (times >= 4.0) & (times <= 24.0)
        fit = log_ols_exponent(times[mask], median[mask])
        rows.append({
            "construction": name,
            "classification": "pure_power",
            "primary_estimate": float(fit["exponent"]),
            "diagnostic": float(fit["r2"]),
            "diagnostic_name": "power_fit_r2",
        })
    times = data["stretched_exponential"]["times"]
    median = np.median(data["stretched_exponential"]["traces"], axis=0)
    early = (times >= 4.0) & (times <= 64.0)
    late = (times >= 100_000.0) & (times <= 800_000.0)
    early_fit = log_ols_exponent(times[early], median[early])
    late_fit = log_ols_exponent(times[late], median[late])
    rows.append({
        "construction": "stretched_exponential",
        "classification": "power_with_log_correction",
        "primary_estimate": float(late_fit["exponent"]),
        "early_exponent": float(early_fit["exponent"]),
        "diagnostic": float(early_fit["exponent"] - late_fit["exponent"]),
        "diagnostic_name": "early_to_late_exponent_drop",
    })
    times = data["spectral_gap"]["times"]
    median = np.median(data["spectral_gap"]["traces"], axis=0)
    mask = (times >= 4.0) & (times <= 24.0)
    design = np.column_stack((np.ones(int(np.sum(mask))), times[mask]))
    coefficients, _, _, _ = np.linalg.lstsq(design, np.log(median[mask]), rcond=None)
    fitted = design @ coefficients
    centered = np.log(median[mask]) - np.mean(np.log(median[mask]))
    r2 = 1.0 - float(np.sum(np.square(np.log(median[mask]) - fitted))) / float(
        np.sum(np.square(centered))
    )
    rows.append({
        "construction": "spectral_gap",
        "classification": "exponential_non_power",
        "primary_estimate": float(-coefficients[1]),
        "diagnostic": r2,
        "diagnostic_name": "exponential_fit_r2",
    })
    times = data["geometric_spectrum"]["times"]
    median = np.median(data["geometric_spectrum"]["traces"], axis=0)
    mask = (times >= 16.0) & (times <= 4096.0)
    periodic = periodic_power_fit(times[mask], median[mask], log_period=1.4)
    rows.append({
        "construction": "geometric_spectrum",
        "classification": "log_periodic_non_power",
        "primary_estimate": float(periodic["power_exponent"]),
        "diagnostic": float(periodic["periodic_residual_fraction_explained"]),
        "diagnostic_name": "known_period_residual_fraction_explained",
    })
    rapid = data["polynomial_rapid_target"]["batch_groups"]
    exponents = []
    endpoint_products = []
    for batch_size, group in rapid.items():
        times = group["times"]
        median = np.median(group["traces"], axis=0)
        mask = (times >= 16.0) & (times <= 64.0)
        exponents.append(float(log_ols_exponent(times[mask], median[mask])["exponent"]))
        endpoint = int(np.argmin(np.abs(times - 64.0)))
        endpoint_products.append(float(batch_size * median[endpoint]))
    rows.append({
        "construction": "polynomial_rapid_target",
        "classification": "rapid_forcing_but_sgd_power_tail",
        "primary_estimate": float(np.mean(exponents)),
        "diagnostic": float(np.std(endpoint_products) / np.mean(endpoint_products)),
        "diagnostic_name": "relative_sd_of_B_times_endpoint_risk",
    })
    return rows


def _risk_panel(
    axis: Any, times: Array, traces: Array, *, color: str, label: str | None
) -> Array:
    median, q25, q75 = _median_band(traces)
    positive = times > 0.0
    axis.loglog(times[positive], median[positive], color=color, label=label)
    axis.fill_between(times[positive], q25[positive], q75[positive], color=color, alpha=0.15)
    axis.grid(alpha=0.16, which="major")
    axis.grid(alpha=0.045, which="minor")
    axis.set(xlabel=r"Intrinsic time $T=\sum\eta$", ylabel="Risk")
    return median


def _width_panel(
    axis: Any,
    width_groups: dict[int, dict[str, Any]],
    *,
    color: str,
    widths_to_plot: list[int] | None = None,
) -> tuple[Array, Array]:
    from matplotlib.colors import to_rgb

    all_widths = sorted(width_groups)
    widths = all_widths if widths_to_plot is None else list(widths_to_plot)
    if not widths or any(width not in width_groups for width in widths):
        raise ValueError("requested plot widths are absent from width_groups")
    base = np.asarray(to_rgb(color), dtype=float)
    strengths = (
        np.asarray([1.0])
        if len(widths) == 1
        else np.linspace(0.48, 1.0, len(widths))
    )
    primary_times: Array | None = None
    primary_median: Array | None = None
    for width, strength in zip(widths, strengths, strict=True):
        group = width_groups[width]
        times = group["times"]
        median = np.median(group["traces"], axis=0)
        positive = times > 0.0
        shade = 1.0 - strength * (1.0 - base)
        axis.loglog(
            times[positive], median[positive], color=shade,
            label=rf"$d={width:,}$", linewidth=2.5,
        )
        if width == widths[-1]:
            primary_times = times
            primary_median = median
    assert primary_times is not None and primary_median is not None
    axis.grid(alpha=0.16, which="major")
    axis.grid(alpha=0.045, which="minor")
    axis.set(xlabel=r"Intrinsic time $T=\sum\eta$", ylabel="Risk")
    return primary_times, primary_median


def make_combined(data: dict[str, dict[str, Any]], output: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    _set_plot_style(plt)
    figure, axes = plt.subplots(2, 3, figsize=(16.0, 9.0), constrained_layout=True)
    specs = (
        ("polynomial_canonical", CONSTRUCTION_TITLES["polynomial_canonical"], "#277DA1"),
        ("polynomial_sparse_target", CONSTRUCTION_TITLES["polynomial_sparse_target"], "#43AA8B"),
        ("stretched_exponential", CONSTRUCTION_TITLES["stretched_exponential"], "#F8961E"),
        ("geometric_spectrum", CONSTRUCTION_TITLES["geometric_spectrum"], "#9B5DE5"),
    )
    for panel_index, (axis, (name, title, color)) in enumerate(
        zip(axes.flat[:4], specs, strict=True)
    ):
        available_widths = sorted(data[name]["width_groups"])
        selected_widths = (
            available_widths
            if name in {"polynomial_canonical", "polynomial_sparse_target"}
            else [available_widths[-1]]
        )
        times, median = _width_panel(
            axis, data[name]["width_groups"], color=color,
            widths_to_plot=selected_widths,
        )
        positive = times > 0.0
        if name in {"polynomial_canonical", "polynomial_sparse_target"}:
            reference = (times >= 1.0) & (times <= 128.0)
            axis.loglog(
                times[reference],
                _power_reference(times[reference], median[reference], 0.5, 8.0),
                color="black", linestyle="--", linewidth=2.0, label=r"$T^{-1/2}$",
            )
            axis.text(
                0.04, 0.08, "pure power", transform=axis.transAxes,
                color="0.22", fontstyle="italic",
            )
        elif name == "stretched_exponential":
            late = times >= 64.0
            axis.loglog(
                times[late], 0.25 * _power_reference(times[late], median[late], 1.0, 4096.0),
                color="black", linestyle="--", linewidth=2.0, label=r"$T^{-1}$",
            )
            axis.text(
                0.04, 0.08, "power law with log correction",
                transform=axis.transAxes, color="0.22", fontstyle="italic",
            )
        else:
            reference = times >= 2.0
            axis.loglog(
                times[reference],
                _power_reference(times[reference], median[reference], 1.75, 64.0),
                color="black", linestyle="--", linewidth=2.0, label=r"$T^{-1.75}$",
            )
            axis.text(
                0.04, 0.08, "log-periodic residual", transform=axis.transAxes,
                color="0.22", fontstyle="italic",
            )
        axis.set_title(title, pad=10)
        axis.text(
            -0.12, 1.10, f"({chr(ord('a') + panel_index)})",
            transform=axis.transAxes, fontsize=13, fontweight="bold",
        )
        handles, labels = axis.get_legend_handles_labels()
        if labels:
            axis.legend(handles, labels, frameon=False, fontsize=10, loc="upper right")
        for spine in axis.spines.values():
            spine.set_color("0.35")
            spine.set_linewidth(0.8)

    rapid_axis = axes.flat[4]
    rapid_colors = {16: "#577590", 64: "#F9844A", 256: "#90BE6D"}
    for batch_size, group in data["polynomial_rapid_target"]["batch_groups"].items():
        times = group["times"]
        median = np.median(group["traces"], axis=0)
        positive = times > 0.0
        rapid_axis.loglog(
            times[positive], median[positive], color=rapid_colors[batch_size],
            label=rf"$B={batch_size}$", linewidth=2.5,
        )
    memory_group = data["polynomial_rapid_target"]["batch_groups"][256]
    memory_times = memory_group["times"]
    memory_median = np.median(memory_group["traces"], axis=0)
    memory_tail = memory_times >= 16.0
    rapid_axis.loglog(
        memory_times[memory_tail],
        0.6 * _power_reference(
            memory_times[memory_tail], memory_median[memory_tail], 0.75, 64.0
        ),
        color="black", linestyle="--", linewidth=2.0,
        label=r"$T^{-3/4}$",
    )
    rapid_axis.set(
        title=CONSTRUCTION_TITLES["polynomial_rapid_target"],
        xlabel=r"Intrinsic time $T=\sum\eta$", ylabel="Risk",
    )
    rapid_axis.text(
        0.04, 0.08, "rapid forcing, power law tail",
        transform=rapid_axis.transAxes, color="0.22", fontstyle="italic",
    )
    rapid_axis.grid(alpha=0.16, which="major")
    rapid_axis.grid(alpha=0.045, which="minor")
    rapid_axis.legend(frameon=False, fontsize=9, loc="upper right")
    rapid_axis.text(
        -0.12, 1.10, "(e)", transform=rapid_axis.transAxes,
        fontsize=13, fontweight="bold",
    )
    for spine in rapid_axis.spines.values():
        spine.set_color("0.35")
        spine.set_linewidth(0.8)

    gap_axis = axes.flat[5]
    gap_widths = sorted(data["spectral_gap"]["width_groups"])
    _width_panel(
        gap_axis, data["spectral_gap"]["width_groups"],
        color="#F94144",
        widths_to_plot=[gap_widths[-1]],
    )
    gap_axis.set_title(CONSTRUCTION_TITLES["spectral_gap"], pad=10)
    gap_axis.text(
        0.04, 0.08, "exponential", transform=gap_axis.transAxes,
        color="0.22", fontstyle="italic",
    )
    gap_axis.text(
        -0.12, 1.10, "(f)", transform=gap_axis.transAxes,
        fontsize=13, fontweight="bold",
    )
    gap_axis.legend(frameon=False, fontsize=9, loc="upper right")
    for spine in gap_axis.spines.values():
        spine.set_color("0.35")
        spine.set_linewidth(0.8)
    figure.text(
        0.5, -0.008,
        r"Dashed lines: theoretical slopes (vertically shifted for clarity).",
        ha="center", va="top", fontsize=12,
    )
    figure.savefig(output / "six_constructions.png", dpi=240, bbox_inches="tight")
    figure.savefig(output / "six_constructions.pdf", bbox_inches="tight")
    figure.savefig(output / "six_constructions_paper_clean.png", dpi=240, bbox_inches="tight")
    plt.close(figure)


def make_non_power_diagnostics(
    data: dict[str, dict[str, Any]], output: Path
) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    _set_plot_style(plt)
    figure, axes = plt.subplots(1, 2, figsize=(11.6, 4.7), constrained_layout=True)

    stretched = data["stretched_exponential"]
    stretched_times = stretched["times"]
    stretched_median = np.median(stretched["traces"], axis=0)
    slope_times, slopes = local_exponent_curve(
        stretched_times, stretched_median, half_window_decades=0.15
    )
    slope_window = slope_times >= 4.0
    axes[0].semilogx(
        slope_times[slope_window], slopes[slope_window],
        color="#F8961E", linewidth=2.8,
    )
    axes[0].axhline(
        1.0, color="black", linestyle="--", linewidth=2.0,
        label=r"limiting $q=1$",
    )
    axes[0].set(
        xlabel=r"Intrinsic time $T=\sum\eta$",
        ylabel=r"Local slope $q(T)$",
        title="Stretched exponential: log correction",
    )
    axes[0].set_ylim(0.95, 1.62)
    axes[0].legend(frameon=False)
    axes[0].grid(alpha=0.2, which="both")

    geometric = data["geometric_spectrum"]
    geometric_times = geometric["times"]
    geometric_median = np.median(geometric["traces"], axis=0)
    residual_window = (geometric_times >= 16.0) & (geometric_times <= 4096.0)
    periodic = periodic_power_fit(
        geometric_times[residual_window], geometric_median[residual_window],
        log_period=1.4,
    )
    axes[1].plot(
        np.log(geometric_times[residual_window]), periodic["power_residual"],
        color="#9B5DE5", linewidth=2.4, label="power-law residual",
    )
    axes[1].plot(
        np.log(geometric_times[residual_window]), periodic["periodic_component"],
        color="black", linewidth=2.0, label="period-1.4 fit",
    )
    axes[1].axhline(0.0, color="0.65", linewidth=1.0)
    axes[1].set(
        xlabel=r"$\log T$", ylabel="Log-risk residual",
        title="Geometric spectrum: log-periodic residual",
    )
    axes[1].legend(frameon=False)
    axes[1].grid(alpha=0.2)

    figure.savefig(output / "non_power_diagnostics.png", dpi=240, bbox_inches="tight")
    figure.savefig(output / "non_power_diagnostics.pdf", bbox_inches="tight")
    plt.close(figure)


def make_individuals(data: dict[str, dict[str, Any]], output: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    _set_plot_style(plt)
    definitions = (
        ("polynomial_canonical", "01_canonical_polynomial", CONSTRUCTION_TITLES["polynomial_canonical"], "#277DA1", 0.15),
        ("polynomial_sparse_target", "02_irregular_sparse_target", CONSTRUCTION_TITLES["polynomial_sparse_target"], "#43AA8B", 0.15),
        ("stretched_exponential", "03_stretched_exponential", CONSTRUCTION_TITLES["stretched_exponential"], "#F8961E", 0.15),
    )
    for name, filename, title, color, window in definitions:
        times = data[name]["times"]
        traces = data[name]["traces"]
        median = np.median(traces, axis=0)
        figure, axes = plt.subplots(1, 2, figsize=(11.2, 4.8), constrained_layout=True)
        _risk_panel(axes[0], times, traces, color=color, label="median ± IQR")
        if name in {"polynomial_canonical", "polynomial_sparse_target"}:
            positive = times > 0.0
            axes[0].loglog(
                times[positive], _power_reference(times[positive], median[positive], 0.5, 8.0),
                color="black", linestyle="--", linewidth=2.0, label=r"$T^{-1/2}$",
            )
        elif name == "stretched_exponential":
            late = times >= 64.0
            axes[0].loglog(
                times[late], 0.6 * _power_reference(times[late], median[late], 1.0, 4096.0),
                color="black", linestyle="--", linewidth=2.0, label=r"$T^{-1}$",
            )
        slope_t, slopes = local_exponent_curve(
            times, median, half_window_decades=window
        )
        axes[1].semilogx(slope_t, slopes, color=color)
        if name in {"polynomial_canonical", "polynomial_sparse_target"}:
            axes[1].axhline(0.5, color="black", linestyle="--", linewidth=2.0)
        else:
            axes[1].axhline(1.0, color="black", linestyle="--", linewidth=2.0, label="limiting exponent 1")
        axes[1].set(
            xlabel=r"Intrinsic time $T=\sum\eta$", ylabel="Local power exponent",
            title="Local-slope diagnostic",
        )
        axes[0].set_title("Risk curve")
        axes[0].legend(frameon=False, fontsize=10)
        axes[1].grid(alpha=0.2, which="both")
        figure.suptitle(title)
        figure.savefig(output / f"{filename}.png", dpi=220, bbox_inches="tight")
        figure.savefig(output / f"{filename}.pdf", bbox_inches="tight")
        plt.close(figure)

    # Spectral gap: curvature on log-log, straightness on semilog-y.
    name = "spectral_gap"
    times = data[name]["times"]
    traces = data[name]["traces"]
    median, q25, q75 = _median_band(traces)
    positive = times > 0.0
    figure, axes = plt.subplots(1, 2, figsize=(11.2, 4.8), constrained_layout=True)
    _risk_panel(axes[0], times, traces, color="#F94144", label="median ± IQR")
    axes[0].set_title("Log-log: curvature")
    axes[1].semilogy(times[positive], median[positive], color="#F94144")
    axes[1].fill_between(times[positive], q25[positive], q75[positive], color="#F94144", alpha=0.15)
    axes[1].set(xlabel=r"Intrinsic time $T=\sum\eta$", ylabel="Risk", title="Semilog-y: exponential tail")
    axes[1].grid(alpha=0.2, which="both")
    figure.suptitle(CONSTRUCTION_TITLES["spectral_gap"])
    figure.savefig(output / "04_spectral_gap.png", dpi=220, bbox_inches="tight")
    figure.savefig(output / "04_spectral_gap.pdf", bbox_inches="tight")
    plt.close(figure)

    # Geometric spectrum: raw risk and its fixed-period residual.
    times = data["geometric_spectrum"]["times"]
    traces = data["geometric_spectrum"]["traces"]
    median = np.median(traces, axis=0)
    mask = (times >= 16.0) & (times <= 4096.0)
    periodic = periodic_power_fit(times[mask], median[mask], log_period=1.4)
    figure, axes = plt.subplots(1, 2, figsize=(11.2, 4.8), constrained_layout=True)
    _risk_panel(axes[0], times, traces, color="#9B5DE5", label=r"$a=1.4$, median ± IQR")
    axes[0].set_title("Power-like envelope")
    axes[1].plot(np.log(times[mask]), periodic["power_residual"], color="#9B5DE5", label="power-law residual")
    axes[1].plot(np.log(times[mask]), periodic["periodic_component"], color="black", linewidth=2.0, label="period-1.4 fit")
    axes[1].set(xlabel=r"$\log T$", ylabel="Log-risk residual", title="Log-periodic diagnostic")
    axes[1].grid(alpha=0.2)
    axes[1].legend(frameon=False, fontsize=10)
    figure.suptitle(CONSTRUCTION_TITLES["geometric_spectrum"])
    figure.savefig(output / "05_geometric_spectrum.png", dpi=220, bbox_inches="tight")
    figure.savefig(output / "05_geometric_spectrum.pdf", bbox_inches="tight")
    plt.close(figure)

    # Rapid target: raw risks and the predicted 1/B amplitude compensation.
    rapid = data["polynomial_rapid_target"]["batch_groups"]
    figure, axes = plt.subplots(1, 2, figsize=(11.2, 4.8), constrained_layout=True)
    colors = {16: "#577590", 64: "#F9844A", 256: "#90BE6D"}
    for batch_size, group in rapid.items():
        times = group["times"]
        median = np.median(group["traces"], axis=0)
        positive = times > 0.0
        axes[0].loglog(times[positive], median[positive], color=colors[batch_size], label=rf"$B={batch_size}$")
        axes[1].loglog(times[positive], batch_size * median[positive], color=colors[batch_size], label=rf"$B={batch_size}$")
    memory_group = rapid[256]
    memory_times = memory_group["times"]
    memory_median = np.median(memory_group["traces"], axis=0)
    memory_tail = memory_times >= 16.0
    memory_reference = _power_reference(
        memory_times[memory_tail], memory_median[memory_tail], 0.75, 64.0
    )
    axes[0].loglog(
        memory_times[memory_tail], 0.6 * memory_reference,
        color="black", linestyle="--", linewidth=2.0,
        label=r"$T^{-3/4}$ memory tail",
    )
    axes[1].loglog(
        memory_times[memory_tail], 0.6 * 256.0 * memory_reference,
        color="black", linestyle="--", linewidth=2.0,
        label=r"$T^{-3/4}$ memory tail",
    )
    axes[0].set(title="Actual SGD risk", xlabel=r"Intrinsic time $T=\sum\eta$", ylabel="Risk")
    axes[1].set(title=r"Tail-amplitude diagnostic", xlabel=r"Intrinsic time $T=\sum\eta$", ylabel=r"$B\,R(T)$")
    for axis in axes:
        axis.grid(alpha=0.2, which="both")
        axis.legend(frameon=False, fontsize=10)
    figure.suptitle(CONSTRUCTION_TITLES["polynomial_rapid_target"])
    figure.savefig(output / "06_polynomial_rapid_target.png", dpi=220, bbox_inches="tight")
    figure.savefig(output / "06_polynomial_rapid_target.pdf", bbox_inches="tight")
    plt.close(figure)


def write_analysis(output: Path, metrics: list[dict[str, Any]]) -> None:
    by_name = {row["construction"]: row for row in metrics}
    text = f"""# Six spectral constructions: literal minibatch-SGD summary

| Construction | Empirical classification | Primary result |
|---|---|---|
| Canonical polynomial | Pure power | fitted exponent {by_name['polynomial_canonical']['primary_estimate']:.3f}, near the predicted 0.5 |
| Irregular sparse target | Same pure power | fitted exponent {by_name['polynomial_sparse_target']['primary_estimate']:.3f}, showing cumulative-mass compensation |
| Stretched-exponential spectrum | Power with slow correction | exponent drifts from {by_name['stretched_exponential']['early_exponent']:.3f} to {by_name['stretched_exponential']['primary_estimate']:.3f} toward 1 |
| Spectral gap | Exponential, not power | semilog rate {by_name['spectral_gap']['primary_estimate']:.3f}, semilog R2 {by_name['spectral_gap']['diagnostic']:.3f} |
| Geometric spectrum, a=1.4 | Log-periodic, not pure power | envelope exponent {by_name['geometric_spectrum']['primary_estimate']:.3f}; known period explains {100*by_name['geometric_spectrum']['diagnostic']:.1f}% of residual |
| Polynomial spectrum, rapid target | Rapid forcing with SGD-induced tail | tail exponent mean {by_name['polynomial_rapid_target']['primary_estimate']:.3f}; endpoint B-times-risk relative SD {100*by_name['polynomial_rapid_target']['diagnostic']:.1f}% |

The panels combine completed exploratory and confirmation artifacts. They are a synthesis, not a new confirmatory run. Every primary curve shown is from literal fresh-dense-minibatch SGD; no expected-SGD recurrence or spectral-binning trajectory is plotted.
"""
    (output / "analysis.md").write_text(text, encoding="utf-8")


def main() -> int:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    data = load_data()
    metrics = summarize(data)
    make_combined(data, OUTPUT)
    make_non_power_diagnostics(data, OUTPUT)
    make_individuals(data, OUTPUT)
    write_json(OUTPUT / "summary.json", {
        "schema_version": "spectral_six_construction_summary_v001",
        "primary_estimand": "literal_sampled_minibatch_sgd_population_excess_risk",
        "is_new_training_run": False,
        "source_artifacts": [
            str(WIDTH_SOURCE.parent), str(STRETCHED_SOURCE.parent),
            str(GEOMETRIC_SOURCE.parent), str(RAPID_SOURCE.parent),
            str(MISSING_WIDTH_STRETCHED_SOURCE.parent),
            str(POLYNOMIAL_SMALL_WIDTH_SOURCE.parent),
        ],
        "metrics": metrics,
    })
    write_analysis(OUTPUT, metrics)
    print(json.dumps({"output": str(OUTPUT), "figure_count": 8}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
