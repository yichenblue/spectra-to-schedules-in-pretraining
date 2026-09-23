"""Same-budget test of the continuum ratio optimizer and integer realization.

The analytic optimizer is exact for the phase-level continuum objective J.
The unit-batch modal recursion is a separate achievability diagnostic; it is
not treated as an exact finite-step minimizer of the conditional risk.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Any

import numpy as np
from scipy import special

from .common import (
    local_exponent_statistics,
    log_binned_indices,
    log_ols_exponent,
    relative_l2,
    sha256_file,
    write_csv,
    write_json,
)


Array = np.ndarray
ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "artifacts" / "ratio_control_phase_iiia"

ALPHA = 0.75
BETA = 0.25
P = 2.0 * ALPHA + 2.0 * BETA - 1.0
Q_FORCING = P / (2.0 * ALPHA)
Q_MEMORY = 2.0 - 1.0 / (2.0 * ALPHA)
THEORY_DATA_EXPONENT = P / (1.0 + P)
SIGMA2 = 1.0
DATA_BUDGETS = (512, 1024, 2048, 4096, 8192, 16384)
WIDTH_SCALE = 64.0
HORIZON_SCALE = 0.10
BASE_RELATIVE_BIN_WIDTH = 2.0e-3
REFINED_RELATIVE_BIN_WIDTH = 1.0e-3
BASE_INTEGRATION_CELLS = 4096
REFINED_INTEGRATION_CELLS = 8192
ROW_MASS_CAP = 0.80
PROFILE_NAMES = ("optimal", "constant", "reversed")


@dataclass(frozen=True)
class ModalSpectrum:
    width: int
    relative_bin_width: float
    eigenvalues: Array
    multiplicities: Array
    initial_clean_mass: Array
    approximation_floor: float
    mode_count: int


def width_and_horizon(data_budget: int) -> tuple[int, float]:
    width = max(
        64,
        int(round(WIDTH_SCALE * data_budget ** (1.0 / (1.0 + P)))),
    )
    horizon = HORIZON_SCALE * data_budget ** (2.0 * ALPHA / (1.0 + P))
    return width, float(horizon)


def build_spectrum(width: int, relative_bin_width: float) -> ModalSpectrum:
    _, _, nodes, multiplicities = log_binned_indices(
        width, relative_bin_width
    )
    target_power = 2.0 * (ALPHA + BETA)
    eigenvalues = nodes ** (-2.0 * ALPHA)
    clean_mass = multiplicities * nodes ** (-target_power)
    floor = float(special.zeta(target_power, float(width) + 1.0))
    return ModalSpectrum(
        width=width,
        relative_bin_width=relative_bin_width,
        eigenvalues=eigenvalues,
        multiplicities=multiplicities,
        initial_clean_mass=clean_mass,
        approximation_floor=floor,
        mode_count=int(nodes.size),
    )


def direct_forcing(spectrum: ModalSpectrum, times: Array) -> Array:
    time = np.asarray(times, dtype=float)
    output = np.empty(time.size, dtype=float)
    for start in range(0, time.size, 64):
        stop = min(start + 64, time.size)
        survival = np.exp(
            -2.0 * spectrum.eigenvalues[:, None] * time[None, start:stop]
        )
        output[start:stop] = (
            spectrum.initial_clean_mass @ survival
            + spectrum.approximation_floor
        )
    return output


def ratio_profiles(
    spectrum: ModalSpectrum,
    data_budget: int,
    horizon: float,
    integration_cells: int,
) -> dict[str, Any]:
    edges = np.linspace(0.0, horizon, integration_cells + 1)
    delta = horizon / integration_cells
    midpoints = 0.5 * (edges[:-1] + edges[1:])
    forcing = direct_forcing(spectrum, midpoints)
    weight = (
        (1.0 + horizon - midpoints) ** (-Q_MEMORY)
        * (forcing + SIGMA2)
    )
    root_weight = np.sqrt(weight)
    optimal = (
        float(data_budget)
        * root_weight
        / (delta * float(np.sum(root_weight)))
    )
    constant = np.full_like(optimal, float(data_budget) / horizon)
    reversed_profile = optimal[::-1].copy()
    profiles = {
        "optimal": optimal,
        "constant": constant,
        "reversed": reversed_profile,
    }
    objectives = {
        name: float(delta * np.sum(weight / profile))
        for name, profile in profiles.items()
    }
    j_star_formula = (
        delta * float(np.sum(root_weight))
    ) ** 2 / float(data_budget)
    budgets = {
        name: float(delta * np.sum(profile))
        for name, profile in profiles.items()
    }
    return {
        "edges": edges,
        "midpoints": midpoints,
        "delta": delta,
        "forcing": forcing,
        "weight": weight,
        "profiles": profiles,
        "objectives": objectives,
        "j_star_formula": j_star_formula,
        "budgets": budgets,
    }


def equal_sample_quantiles(
    edges: Array,
    profile: Array,
    data_budget: int,
) -> tuple[Array, float]:
    edge = np.asarray(edges, dtype=float)
    ratio = np.asarray(profile, dtype=float)
    delta = np.diff(edge)
    cell_mass = ratio * delta
    cell_mass *= float(data_budget) / float(np.sum(cell_mass))
    cumulative = np.concatenate([np.array([0.0]), np.cumsum(cell_mass)])
    cumulative[-1] = float(data_budget)
    quantile_times = np.interp(
        np.arange(data_budget + 1, dtype=float), cumulative, edge
    )
    steps = np.diff(quantile_times)
    if np.any(steps <= 0.0):
        raise RuntimeError("equal-sample quantiles produced a nonpositive step")
    cdf_at_quantiles = np.interp(quantile_times, edge, cumulative) / data_budget
    ideal = np.arange(data_budget + 1, dtype=float) / data_budget
    discrepancy = float(np.max(np.abs(cdf_at_quantiles - ideal)))
    return steps, discrepancy


def run_unit_batch_modal_recursion(
    spectrum: ModalSpectrum,
    schedule_steps: dict[str, Array],
) -> dict[str, Any]:
    names = tuple(schedule_steps)
    lengths = {steps.size for steps in schedule_steps.values()}
    if len(lengths) != 1:
        raise ValueError("all compared unit-batch schedules must use the same data")
    step_count = lengths.pop()
    eta = np.stack([schedule_steps[name] for name in names], axis=0)
    count = len(names)
    modes = spectrum.mode_count
    clean_modes = np.broadcast_to(
        spectrum.initial_clean_mass, (count, modes)
    ).copy()
    gap_modes = np.zeros_like(clean_modes)
    row_modes = np.zeros_like(clean_modes)
    squared = spectrum.eigenvalues * spectrum.eigenvalues
    injection_base = spectrum.multiplicities * squared
    maximum_pointwise = np.zeros(count, dtype=float)
    maximum_row = np.zeros(count, dtype=float)
    for index in range(step_count):
        eta_step = eta[:, index]
        products = 2.0 * eta_step[:, None] * spectrum.eigenvalues[None, :]
        q = (
            1.0
            - products
            + 2.0
            * eta_step[:, None]
            * eta_step[:, None]
            * squared[None, :]
        )
        injection = (
            eta_step[:, None]
            * eta_step[:, None]
            * injection_base[None, :]
        )
        current_clean = spectrum.approximation_floor + np.sum(
            clean_modes, axis=1
        )
        current_gap = np.sum(gap_modes, axis=1)
        clean_modes = q * clean_modes + injection * current_clean[:, None]
        gap_modes = q * gap_modes + injection * (current_gap + SIGMA2)[:, None]
        row_modes = q * row_modes + injection
        maximum_pointwise = np.maximum(
            maximum_pointwise, 2.0 * eta_step * spectrum.eigenvalues[0]
        )
        maximum_row = np.maximum(maximum_row, np.sum(row_modes, axis=1))
        if (
            not np.all(np.isfinite(clean_modes))
            or not np.all(np.isfinite(gap_modes))
            or np.any(clean_modes < -1.0e-13)
            or np.any(gap_modes < -1.0e-13)
        ):
            raise FloatingPointError("unit-batch modal state became invalid")
        clean_modes = np.maximum(clean_modes, 0.0)
        gap_modes = np.maximum(gap_modes, 0.0)
    output: dict[str, Any] = {}
    for position, name in enumerate(names):
        clean_centered = float(np.sum(clean_modes[position]))
        gap = float(np.sum(gap_modes[position]))
        output[name] = {
            "clean_centered": clean_centered,
            "noise_gap": gap,
            "total_risk": spectrum.approximation_floor + clean_centered + gap,
            "maximum_eta": float(np.max(eta[position])),
            "intrinsic_horizon": float(np.sum(eta[position])),
            "processed_samples": int(step_count),
            "maximum_pointwise_product": float(maximum_pointwise[position]),
            "maximum_row_mass": float(maximum_row[position]),
            "pointwise_stable": bool(maximum_pointwise[position] < 2.0),
            "row_stable": bool(maximum_row[position] < ROW_MASS_CAP),
        }
    return output


def evaluate_budget(
    data_budget: int,
    relative_bin_width: float,
    integration_cells: int,
) -> dict[str, Any]:
    width, horizon = width_and_horizon(data_budget)
    spectrum = build_spectrum(width, relative_bin_width)
    profile_data = ratio_profiles(
        spectrum, data_budget, horizon, integration_cells
    )
    schedule_steps: dict[str, Array] = {}
    cdf_discrepancies: dict[str, float] = {}
    for name in PROFILE_NAMES:
        steps, discrepancy = equal_sample_quantiles(
            profile_data["edges"],
            profile_data["profiles"][name],
            data_budget,
        )
        schedule_steps[name] = steps
        cdf_discrepancies[name] = discrepancy
    exact = run_unit_batch_modal_recursion(spectrum, schedule_steps)
    terminal_forcing = float(direct_forcing(spectrum, np.array([horizon]))[0])
    proxy_terminal = terminal_forcing + profile_data["j_star_formula"]
    return {
        "data_budget": data_budget,
        "width": width,
        "horizon": horizon,
        "spectrum": spectrum,
        "profile_data": profile_data,
        "schedule_steps": schedule_steps,
        "cdf_discrepancies": cdf_discrepancies,
        "exact": exact,
        "terminal_forcing": terminal_forcing,
        "proxy_terminal": proxy_terminal,
        "source_window_ratio": horizon / width ** (2.0 * ALPHA),
    }


def metric_rows(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for result in results:
        j_star = result["profile_data"]["j_star_formula"]
        for name in PROFILE_NAMES:
            exact = result["exact"][name]
            rows.append(
                {
                    "schedule": name,
                    "data_budget": result["data_budget"],
                    "width": result["width"],
                    "horizon": result["horizon"],
                    "source_window_ratio": result["source_window_ratio"],
                    "quadrature_modes": result["spectrum"].mode_count,
                    "continuum_budget": result["profile_data"]["budgets"][name],
                    "continuum_objective": result["profile_data"]["objectives"][name],
                    "objective_over_optimum": result["profile_data"]["objectives"][name] / j_star,
                    "j_star_formula": j_star,
                    "cdf_quantile_discrepancy": result["cdf_discrepancies"][name],
                    "processed_samples": exact["processed_samples"],
                    "realized_horizon": exact["intrinsic_horizon"],
                    "maximum_eta": exact["maximum_eta"],
                    "maximum_pointwise_product": exact[
                        "maximum_pointwise_product"
                    ],
                    "maximum_row_mass": exact["maximum_row_mass"],
                    "pointwise_stable": exact["pointwise_stable"],
                    "row_stable": exact["row_stable"],
                    "clean_centered": exact["clean_centered"],
                    "noise_gap": exact["noise_gap"],
                    "total_risk": exact["total_risk"],
                    "terminal_forcing": result["terminal_forcing"],
                    "optimal_proxy_terminal": result["proxy_terminal"],
                }
            )
    return rows


def profile_curve_rows(result: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    average_ratio = result["data_budget"] / result["horizon"]
    profile_data = result["profile_data"]
    for index, u in enumerate(profile_data["midpoints"]):
        row: dict[str, Any] = {
            "u_over_T": float(u / result["horizon"]),
            "u": float(u),
            "weight": float(profile_data["weight"][index]),
        }
        for name in PROFILE_NAMES:
            row[f"ratio_{name}"] = float(
                profile_data["profiles"][name][index]
            )
            row[f"normalized_ratio_{name}"] = float(
                profile_data["profiles"][name][index] / average_ratio
            )
        rows.append(row)
    return rows


def quantile_rows(result: dict[str, Any], maximum_rows: int = 513) -> list[dict[str, Any]]:
    budget = result["data_budget"]
    stride = max(1, int(math.ceil(budget / (maximum_rows - 1))))
    indices = sorted(set(range(0, budget + 1, stride)) | {budget})
    optimal_steps = result["schedule_steps"]["optimal"]
    optimal_times = np.concatenate([np.array([0.0]), np.cumsum(optimal_steps)])
    return [
        {
            "sample_index": index,
            "sample_fraction": index / budget,
            "u": float(optimal_times[index]),
            "u_over_T": float(optimal_times[index] / result["horizon"]),
        }
        for index in indices
    ]


def fit_risk_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for name in PROFILE_NAMES:
        selected = [row for row in rows if row["schedule"] == name]
        data = np.asarray([row["data_budget"] for row in selected], dtype=float)
        risk = np.asarray([row["total_risk"] for row in selected], dtype=float)
        fit = log_ols_exponent(data, risk)
        local = local_exponent_statistics(
            data, risk, half_window_decades=0.50, minimum_points=3
        )
        output[name] = {**fit, "local": local}
    selected = [row for row in rows if row["schedule"] == "optimal"]
    data = np.asarray([row["data_budget"] for row in selected], dtype=float)
    proxy = np.asarray(
        [row["optimal_proxy_terminal"] for row in selected], dtype=float
    )
    output["continuum_proxy"] = {
        **log_ols_exponent(data, proxy),
        "local": local_exponent_statistics(
            data, proxy, half_window_decades=0.50, minimum_points=3
        ),
    }
    return output


def component_resolution_rows(
    reference_rows: list[dict[str, Any]],
    comparison_rows: list[dict[str, Any]],
    comparison_route: str,
) -> list[dict[str, Any]]:
    """Compare clean, direct-gap, and total curves on a common data grid."""

    output: list[dict[str, Any]] = []
    for name in PROFILE_NAMES:
        reference = [row for row in reference_rows if row["schedule"] == name]
        comparison = [row for row in comparison_rows if row["schedule"] == name]
        reference_budgets = np.asarray(
            [row["data_budget"] for row in reference], dtype=float
        )
        comparison_budgets = np.asarray(
            [row["data_budget"] for row in comparison], dtype=float
        )
        if not np.array_equal(reference_budgets, comparison_budgets):
            raise ValueError("resolution routes must share the same data grid")
        for component in ("clean_centered", "noise_gap", "total_risk"):
            reference_values = np.asarray(
                [row[component] for row in reference], dtype=float
            )
            comparison_values = np.asarray(
                [row[component] for row in comparison], dtype=float
            )
            output.append(
                {
                    "comparison_route": comparison_route,
                    "schedule": name,
                    "component": component,
                    "relative_l2": relative_l2(
                        reference_values, comparison_values
                    ),
                    "absolute_slope_difference": abs(
                        log_ols_exponent(
                            reference_budgets, reference_values
                        )["exponent"]
                        - log_ols_exponent(
                            comparison_budgets, comparison_values
                        )["exponent"]
                    ),
                }
            )
    return output


def make_figure(
    results: list[dict[str, Any]],
    rows: list[dict[str, Any]],
    fits: dict[str, Any],
    output_dir: Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    plt.rcParams.update(
        {
            "font.size": 8.6,
            "axes.titlesize": 9.2,
            "axes.labelsize": 8.8,
            "legend.fontsize": 8.1,
            "xtick.labelsize": 8.0,
            "ytick.labelsize": 8.0,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    colors = {
        "optimal": "#dd8452",
        "constant": "#9a9a9a",
        "reversed": "#c44e52",
        "integer": "#4c72b0",
        "theory": "#222222",
    }
    representative = next(
        result for result in results if result["data_budget"] == 8192
    )
    profile_data = representative["profile_data"]
    average_ratio = representative["data_budget"] / representative["horizon"]
    x = profile_data["midpoints"] / representative["horizon"]

    figure, axes = plt.subplots(1, 3, figsize=(7.15, 2.95))
    axis = axes[0]
    for name in PROFILE_NAMES:
        axis.plot(
            x,
            profile_data["profiles"][name] / average_ratio,
            color=colors[name],
            linewidth=2.0,
        )
    j_star = profile_data["j_star_formula"]
    axis.text(
        0.04,
        0.92,
        r"$r^\star\propto\sqrt{w}$",
        transform=axis.transAxes,
        color=colors["optimal"],
        fontsize=8.4,
    )
    for name, y_position in [("optimal", 0.83), ("constant", 0.74), ("reversed", 0.65)]:
        regret = profile_data["objectives"][name] / j_star
        label = {"optimal": "optimal", "constant": "constant", "reversed": "time-reversed"}[name]
        axis.text(
            0.04,
            y_position,
            rf"{label}: $J/J^\star={regret:.3f}$",
            transform=axis.transAxes,
            color=colors[name],
            fontsize=8.3,
        )
    axis.set_title("(a) Same budget, different allocation", pad=4.0)
    axis.set_xlabel(r"intrinsic position $u/T$")
    axis.set_ylabel(r"normalized ratio $r(u)/(D/T)$")
    axis.set_xlim(0.0, 1.0)
    axis.grid(True, alpha=0.16, linewidth=0.55)

    axis = axes[1]
    edges = profile_data["edges"]
    cell_mass = profile_data["profiles"]["optimal"] * np.diff(edges)
    cumulative = np.concatenate([np.array([0.0]), np.cumsum(cell_mass)])
    cumulative /= cumulative[-1]
    axis.plot(
        edges / representative["horizon"],
        cumulative,
        color=colors["optimal"],
        linewidth=2.0,
        label="continuum allocation",
    )
    quantiles = quantile_rows(representative, maximum_rows=129)
    axis.scatter(
        [row["u_over_T"] for row in quantiles],
        [row["sample_fraction"] for row in quantiles],
        s=8.0,
        facecolor="white",
        edgecolor=colors["integer"],
        linewidth=0.65,
        zorder=3,
        label=r"unit batch ($B_s=1$)",
    )
    axis.plot([0.0, 1.0], [0.0, 1.0], color=colors["constant"], linewidth=1.0, linestyle=":")
    axis.text(
        0.96,
        0.08,
        f"exact: {representative['data_budget']:,} samples",
        transform=axis.transAxes,
        ha="right",
        fontsize=8.3,
    )
    axis.set_title("(b) Unit-batch realization", pad=4.0)
    axis.set_xlabel(r"intrinsic position $u/T$")
    axis.set_ylabel("cumulative sample fraction")
    axis.set_xlim(0.0, 1.0)
    axis.set_ylim(0.0, 1.0)
    axis.grid(True, alpha=0.16, linewidth=0.55)
    axis.legend(frameon=False, loc="upper left", handlelength=1.9)

    axis = axes[2]
    for name in PROFILE_NAMES:
        selected = [row for row in rows if row["schedule"] == name]
        axis.plot(
            [row["data_budget"] for row in selected],
            [row["total_risk"] for row in selected],
            color=(colors["integer"] if name == "optimal" else colors[name]),
            linewidth=2.0 if name == "optimal" else 1.35,
            marker="o" if name == "optimal" else None,
            markersize=3.4,
        )
    selected = [row for row in rows if row["schedule"] == "optimal"]
    data = np.asarray([row["data_budget"] for row in selected], dtype=float)
    proxy = np.asarray([row["optimal_proxy_terminal"] for row in selected], dtype=float)
    axis.plot(
        data,
        proxy,
        color=colors["optimal"],
        linewidth=1.55,
        linestyle=(0, (4, 2.2)),
    )
    pivot = 4096.0
    pivot_value = np.exp(
        np.interp(np.log(pivot), np.log(data), np.log(np.asarray([row["total_risk"] for row in selected])))
    )
    guide_data = np.asarray([2048.0, 8192.0])
    guide = pivot_value * (guide_data / pivot) ** (-THEORY_DATA_EXPONENT)
    axis.plot(guide_data, guide * 1.18, color=colors["theory"], linewidth=1.25, linestyle="--")
    axis.text(
        2300.0,
        guide[0] * 1.35,
        rf"theory $D^{{-{THEORY_DATA_EXPONENT:.3f}}}$",
        fontsize=8.3,
        color=colors["theory"],
    )
    measured = fits["optimal"]["exponent"]
    axis.text(
        0.04,
        0.07,
        rf"unit-batch fit: $\widehat{{q}}_D={measured:.3f}$",
        transform=axis.transAxes,
        fontsize=8.3,
        color=colors["integer"],
    )
    axis.set_xscale("log", base=2)
    axis.set_yscale("log")
    axis.set_title("(c) Integer risk scaling", pad=4.0)
    axis.set_xlabel(r"sample budget $D$")
    axis.set_ylabel(r"risk / phase proxy")
    axis.grid(True, alpha=0.16, linewidth=0.55)

    handles = [
        Line2D([0], [0], color=colors["optimal"], linewidth=2.0, label=r"analytic $r^\star$ (proxy)"),
        Line2D([0], [0], color=colors["integer"], linewidth=2.0, marker="o", markersize=3.5, label="unit batch (exact risk)"),
        Line2D([0], [0], color=colors["constant"], linewidth=1.6, label="constant ratio"),
        Line2D([0], [0], color=colors["reversed"], linewidth=1.6, label="time-reversed control"),
    ]
    figure.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.01),
        ncol=4,
        frameon=False,
        columnspacing=1.2,
        handlelength=2.3,
    )
    figure.subplots_adjust(left=0.072, right=0.992, bottom=0.18, top=0.79, wspace=0.37)
    figure.savefig(output_dir / "ratio_control_triptych.png", dpi=300)
    figure.savefig(output_dir / "ratio_control_triptych.pdf", bbox_inches="tight")
    plt.close(figure)


def run(output_dir: Path = OUTPUT_DIR) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    base_results = [
        evaluate_budget(
            budget, BASE_RELATIVE_BIN_WIDTH, BASE_INTEGRATION_CELLS
        )
        for budget in DATA_BUDGETS
    ]
    refined_results = [
        evaluate_budget(
            budget, REFINED_RELATIVE_BIN_WIDTH, REFINED_INTEGRATION_CELLS
        )
        for budget in DATA_BUDGETS
    ]
    spectrum_refined_results = [
        evaluate_budget(
            budget, REFINED_RELATIVE_BIN_WIDTH, BASE_INTEGRATION_CELLS
        )
        for budget in DATA_BUDGETS
    ]
    time_refined_results = [
        evaluate_budget(
            budget, BASE_RELATIVE_BIN_WIDTH, REFINED_INTEGRATION_CELLS
        )
        for budget in DATA_BUDGETS
    ]
    base_rows = metric_rows(base_results)
    refined_rows = metric_rows(refined_results)
    spectrum_refined_rows = metric_rows(spectrum_refined_results)
    time_refined_rows = metric_rows(time_refined_results)
    fits = fit_risk_metrics(refined_rows)
    resolution_rows = (
        component_resolution_rows(refined_rows, base_rows, "base")
        + component_resolution_rows(
            refined_rows, spectrum_refined_rows, "spectrum_refined"
        )
        + component_resolution_rows(
            refined_rows, time_refined_rows, "time_refined"
        )
    )

    representative = next(
        result for result in refined_results if result["data_budget"] == 8192
    )
    write_csv(output_dir / "ratio_control_metrics.csv", refined_rows)
    write_csv(output_dir / "ratio_control_profiles.csv", profile_curve_rows(representative))
    write_csv(output_dir / "ratio_control_quantiles.csv", quantile_rows(representative))
    write_csv(output_dir / "ratio_control_resolution.csv", resolution_rows)
    make_figure(refined_results, refined_rows, fits, output_dir)

    optimal_rows = [row for row in refined_rows if row["schedule"] == "optimal"]
    largest_three = optimal_rows[-3:]
    constant_rows = [row for row in refined_rows if row["schedule"] == "constant"]
    reversed_rows = [row for row in refined_rows if row["schedule"] == "reversed"]
    representative_profile = representative["profile_data"]
    j_formula_error = abs(
        representative_profile["objectives"]["optimal"]
        / representative_profile["j_star_formula"]
        - 1.0
    )
    gates = {
        "continuum_budget": max(
            abs(row["continuum_budget"] - row["data_budget"])
            / row["data_budget"]
            for row in refined_rows
        ) <= 1.0e-10,
        "exact_integer_budget": all(
            row["processed_samples"] == row["data_budget"]
            for row in refined_rows
        ),
        "exact_intrinsic_horizon": max(
            abs(row["realized_horizon"] - row["horizon"])
            / row["horizon"]
            for row in refined_rows
        ) <= 1.0e-12,
        "continuum_kkt_value": j_formula_error <= 1.0e-4,
        "continuum_global_ordering": all(
            result["profile_data"]["objectives"]["optimal"]
            <= result["profile_data"]["objectives"][name] * (1.0 + 1.0e-12)
            for result in refined_results
            for name in ("constant", "reversed")
        ),
        "nontrivial_continuum_improvement": min(
            representative_profile["objectives"][name]
            / representative_profile["j_star_formula"]
            - 1.0
            for name in ("constant", "reversed")
        ) >= 0.05,
        "integer_risk_achievability_order": all(
            optimal["total_risk"]
            <= constant["total_risk"]
            and optimal["total_risk"] <= reversed["total_risk"]
            for optimal, constant, reversed in zip(
                largest_three, constant_rows[-3:], reversed_rows[-3:], strict=True
            )
        ),
        "data_exponent": abs(
            fits["optimal"]["exponent"] - THEORY_DATA_EXPONENT
        ) <= 0.05,
        "local_variation": fits["optimal"]["local"]["variation"] <= 0.15,
        "source_window": max(row["source_window_ratio"] for row in refined_rows) < 1.0,
        "pointwise_stability": all(row["pointwise_stable"] for row in refined_rows),
        "row_stability": all(row["row_stable"] for row in refined_rows),
        "curve_resolution": max(
            row["relative_l2"] for row in resolution_rows
        ) <= 0.003,
        "slope_resolution": max(
            row["absolute_slope_difference"] for row in resolution_rows
        ) <= 0.002,
    }
    gates["overall"] = all(gates.values())
    summary = {
        "experiment": "Phase-IIIa same-budget ratio optimum and unit-batch achievability",
        "status": "PASS" if gates["overall"] else "FAIL",
        "promotion_status": "PILOT_UNPROMOTED",
        "theorems": ["iclr:cor:ratio_optimizer", "iclr:cor:seven_phase_exponents"],
        "scope": (
            "The unconstrained square-root ratio exactly minimizes the displayed "
            "phase-level continuum objective (equivalently, the nonbinding-bounds "
            "case). The unit-batch recursion is a representative "
            "Phase-IIIa achievability diagnostic, not proof of exact finite-step "
            "global optimality or of all six phasewise rate classes."
        ),
        "parameters": {
            "phase": "IIIa",
            "alpha": ALPHA,
            "beta": BETA,
            "p": P,
            "q_forcing": Q_FORCING,
            "q_memory": Q_MEMORY,
            "sigma2": SIGMA2,
            "width_scale": WIDTH_SCALE,
            "horizon_scale": HORIZON_SCALE,
        },
        "data_budgets": DATA_BUDGETS,
        "numerical_resolution": {
            "base_relative_bin_width": BASE_RELATIVE_BIN_WIDTH,
            "refined_relative_bin_width": REFINED_RELATIVE_BIN_WIDTH,
            "base_integration_cells": BASE_INTEGRATION_CELLS,
            "refined_integration_cells": REFINED_INTEGRATION_CELLS,
            "routes": {
                "base": "base spectrum + base time grid",
                "spectrum_refined": "refined spectrum + base time grid",
                "time_refined": "base spectrum + refined time grid",
                "reference": "refined spectrum + refined time grid",
            },
            "audited_components": [
                "clean_centered",
                "noise_gap",
                "total_risk",
            ],
        },
        "theory_data_exponent": THEORY_DATA_EXPONENT,
        "fitted_risk_metrics": fits,
        "representative_continuum_regrets": {
            name: representative_profile["objectives"][name]
            / representative_profile["j_star_formula"]
            for name in PROFILE_NAMES
        },
        "gates": gates,
        "maximum_resolution_relative_l2": max(
            row["relative_l2"] for row in resolution_rows
        ),
        "maximum_resolution_slope_difference": max(
            row["absolute_slope_difference"] for row in resolution_rows
        ),
        "source_sha256": {
            "ratio_control.py": sha256_file(Path(__file__)),
            "common.py": sha256_file(ROOT / "common.py"),
            "phasewise_schedule_design.tex": sha256_file(
                ROOT.parents[1] / "files" / "phasewise_schedule_design.tex"
            ),
            "appendix_c_joint_schedules.tex": sha256_file(
                ROOT.parents[1] / "files" / "appendix_c_joint_schedules.tex"
            ),
            "appendix_h_schedule_proofs.tex": sha256_file(
                ROOT.parents[1] / "files" / "appendix_h_schedule_proofs.tex"
            ),
        },
        "files": {
            "figure_png": str(output_dir / "ratio_control_triptych.png"),
            "figure_pdf": str(output_dir / "ratio_control_triptych.pdf"),
            "metrics": str(output_dir / "ratio_control_metrics.csv"),
            "profiles": str(output_dir / "ratio_control_profiles.csv"),
            "quantiles": str(output_dir / "ratio_control_quantiles.csv"),
            "resolution": str(output_dir / "ratio_control_resolution.csv"),
        },
    }
    write_json(output_dir / "summary.json", summary)
    return summary


if __name__ == "__main__":
    import json

    print(json.dumps(run(), indent=2, sort_keys=True))
