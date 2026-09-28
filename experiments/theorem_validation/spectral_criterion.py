"""Direct numerical certificate for the paper's spectral iff theorem.

The experiment prescribes effective eigenpairs conditional on a frozen
representation.  It compares the two exact objects on either side of the
Tauberian equivalence, including the theorem's Gamma prefactors.  It is not a
population-to-random-feature empirical transfer experiment.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Any, Literal

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
OUTPUT_DIR = ROOT / "artifacts" / "spectral_criterion"

EFFECTIVE_WIDTHS = (10**6, 10**8, 10**10, 10**12)
FORCING_EXPONENTS = (0.30, 0.50, 0.80)
MEMORY_EXPONENTS = (0.50, 0.75, 1.25)
PAPER_ALPHA = 0.40
PAPER_BETA = 0.30
COMMON_EIGENVALUE_POWER = 2.0 * PAPER_ALPHA
ETA = 0.05
BATCH = 128
BASE_RELATIVE_BIN_WIDTH = 5.0e-4
REFINED_RELATIVE_BIN_WIDTH = 2.5e-4
FIT_LOWER_EXPONENT = 0.20
FIT_UPPER_EXPONENT = 0.40
DISPLAY_LOWER_EXPONENT = 0.10
DISPLAY_UPPER_EXPONENT = 0.45
FIT_POINTS = 61
DISPLAY_POINTS = 181


@dataclass(frozen=True)
class Case:
    kind: Literal["forcing", "memory"]
    exponent: float
    eigenvalue_power: float

    @property
    def name(self) -> str:
        letter = "qF" if self.kind == "forcing" else "qK"
        return f"{self.kind}_{letter}_{self.exponent:g}"


def forcing_cases() -> tuple[Case, ...]:
    return tuple(
        Case("forcing", exponent, COMMON_EIGENVALUE_POWER)
        for exponent in FORCING_EXPONENTS
    )


def memory_cases() -> tuple[Case, ...]:
    return tuple(
        Case("memory", exponent, 1.0 / (2.0 - exponent))
        for exponent in MEMORY_EXPONENTS
    )


def all_cases() -> tuple[Case, ...]:
    return forcing_cases() + memory_cases()


def discrete_times(width: int) -> tuple[Array, Array, Array]:
    display_lower = width**DISPLAY_LOWER_EXPONENT
    display_upper = width**DISPLAY_UPPER_EXPONENT
    raw_steps = np.rint(
        np.geomspace(display_lower / ETA, display_upper / ETA, DISPLAY_POINTS)
    ).astype(np.int64)
    steps = np.unique(np.maximum(raw_steps, 1))
    times = ETA * steps.astype(float)
    fit_lower = width**FIT_LOWER_EXPONENT
    fit_upper = width**FIT_UPPER_EXPONENT
    fit_times = np.geomspace(fit_lower, fit_upper, FIT_POINTS)
    return steps, times, fit_times


def one_step_log_filter(eigenvalues: Array) -> Array:
    """Stable log of q_eta(lambda), including modes for which q rounds to one."""

    decrement = (
        2.0 * ETA * eigenvalues
        - (1.0 + 1.0 / BATCH) * ETA * ETA * eigenvalues * eigenvalues
    )
    if np.any(decrement <= 0.0) or np.any(decrement >= 1.0):
        raise RuntimeError("the prescribed spectral filter is not strictly stable")
    return np.log1p(-decrement)


def exact_cutoff_tail(case: Case, width: int, times: Array) -> Array:
    threshold_index = np.maximum(
        np.ceil(np.asarray(times, dtype=float) ** (1.0 / case.eigenvalue_power)),
        1.0,
    )
    if case.kind == "forcing":
        power = 1.0 + case.eigenvalue_power * case.exponent
        amplitude = case.eigenvalue_power * case.exponent
    else:
        power = 2.0 * case.eigenvalue_power
        amplitude = 1.0
    tail = amplitude * (
        special.zeta(power, threshold_index)
        - special.zeta(power, float(width) + 1.0)
    )
    if np.any(tail <= 0.0) or not np.all(np.isfinite(tail)):
        raise FloatingPointError("cutoff tail became nonpositive or nonfinite")
    return np.asarray(tail, dtype=float)


def modal_transform(
    case: Case,
    width: int,
    steps: Array,
    relative_bin_width: float,
) -> tuple[Array, int]:
    _, _, nodes, multiplicity = log_binned_indices(width, relative_bin_width)
    eigenvalues = nodes ** (-case.eigenvalue_power)
    if case.kind == "forcing":
        power = 1.0 + case.eigenvalue_power * case.exponent
        modal_mass = (
            case.eigenvalue_power
            * case.exponent
            * multiplicity
            * nodes ** (-power)
        )
    else:
        modal_mass = multiplicity * eigenvalues * eigenvalues
    log_filter = one_step_log_filter(eigenvalues)
    output = np.empty(steps.size, dtype=float)
    for start in range(0, steps.size, 24):
        stop = min(start + 24, steps.size)
        survival = np.exp(log_filter[:, None] * steps[None, start:stop])
        output[start:stop] = modal_mass @ survival
    if case.kind == "memory":
        output *= ETA * ETA / BATCH
    if np.any(output <= 0.0) or not np.all(np.isfinite(output)):
        raise FloatingPointError("modal transform became nonpositive or nonfinite")
    return output, int(nodes.size)


def theorem_scaled_transform(case: Case, transform: Array) -> Array:
    scale = (2.0**case.exponent) / special.gamma(case.exponent + 1.0)
    if case.kind == "memory":
        scale *= BATCH / (ETA * ETA)
    return np.asarray(transform, dtype=float) * scale


def evaluate_case(
    case: Case,
    width: int,
    relative_bin_width: float,
) -> dict[str, Any]:
    steps, display_times, fit_times = discrete_times(width)
    tail_display = exact_cutoff_tail(case, width, display_times)
    transform, mode_count = modal_transform(
        case, width, steps, relative_bin_width
    )
    scaled_display = theorem_scaled_transform(case, transform)
    tail_fit = np.exp(
        np.interp(
            np.log(fit_times),
            np.log(display_times),
            np.log(tail_display),
        )
    )
    scaled_fit = np.exp(
        np.interp(
            np.log(fit_times),
            np.log(display_times),
            np.log(scaled_display),
        )
    )
    tail_fit_metrics = log_ols_exponent(fit_times, tail_fit)
    dynamics_fit_metrics = log_ols_exponent(fit_times, scaled_fit)
    local = local_exponent_statistics(fit_times, scaled_fit)
    log_ratio = np.abs(np.log(scaled_fit / tail_fit))
    cutoff_at_upper = math.ceil(
        fit_times[-1] ** (1.0 / case.eigenvalue_power)
    )
    return {
        "case": case,
        "width": width,
        "mode_count": mode_count,
        "display_times": display_times,
        "tail_display": tail_display,
        "scaled_display": scaled_display,
        "fit_times": fit_times,
        "tail_fit": tail_fit,
        "scaled_fit": scaled_fit,
        "tail_exponent": tail_fit_metrics["exponent"],
        "tail_r2": tail_fit_metrics["r2"],
        "dynamics_exponent": dynamics_fit_metrics["exponent"],
        "dynamics_r2": dynamics_fit_metrics["r2"],
        "theory_exponent_error": abs(
            dynamics_fit_metrics["exponent"] - case.exponent
        ),
        "tail_dynamics_exponent_error": abs(
            dynamics_fit_metrics["exponent"] - tail_fit_metrics["exponent"]
        ),
        "collapse_median_abs_log_error": float(np.median(log_ratio)),
        "collapse_p90_abs_log_error": float(np.quantile(log_ratio, 0.90)),
        "collapse_max_abs_log_error": float(np.max(log_ratio)),
        "local": local,
        "fit_lower": float(fit_times[0]),
        "fit_upper": float(fit_times[-1]),
        "cutoff_index_at_fit_upper": int(cutoff_at_upper),
        "cutoff_index_fraction_at_fit_upper": float(cutoff_at_upper / width),
    }


def metrics_row(result: dict[str, Any]) -> dict[str, Any]:
    case: Case = result["case"]
    return {
        "kind": case.kind,
        "case": case.name,
        "theory_exponent": case.exponent,
        "eigenvalue_power": case.eigenvalue_power,
        "effective_width": result["width"],
        "quadrature_modes": result["mode_count"],
        "fit_lower": result["fit_lower"],
        "fit_upper": result["fit_upper"],
        "tail_exponent": result["tail_exponent"],
        "dynamics_exponent": result["dynamics_exponent"],
        "tail_theory_exponent_error": abs(
            result["tail_exponent"] - case.exponent
        ),
        "theory_exponent_error": result["theory_exponent_error"],
        "tail_dynamics_exponent_error": result["tail_dynamics_exponent_error"],
        "dynamics_local_median": result["local"]["median"],
        "dynamics_local_variation": result["local"]["variation"],
        "collapse_median_abs_log_error": result[
            "collapse_median_abs_log_error"
        ],
        "collapse_p90_abs_log_error": result["collapse_p90_abs_log_error"],
        "collapse_max_abs_log_error": result["collapse_max_abs_log_error"],
        "cutoff_index_fraction_at_fit_upper": result[
            "cutoff_index_fraction_at_fit_upper"
        ],
    }


def curve_rows(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for result in results:
        case: Case = result["case"]
        fit_lower = result["fit_lower"]
        fit_upper = result["fit_upper"]
        for time, tail, dynamics in zip(
            result["display_times"],
            result["tail_display"],
            result["scaled_display"],
            strict=True,
        ):
            rows.append(
                {
                    "kind": case.kind,
                    "case": case.name,
                    "theory_exponent": case.exponent,
                    "effective_width": result["width"],
                    "T": float(time),
                    "spectral_tail": float(tail),
                    "theorem_scaled_dynamics": float(dynamics),
                    "scaled_dynamics_over_tail": float(dynamics / tail),
                    "inside_frozen_fit": bool(fit_lower <= time <= fit_upper),
                }
            )
    return rows


def make_figure(
    refined_results: list[dict[str, Any]],
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
        0.30: "#c44e52",
        0.50: "#dd8452",
        0.75: "#8172b3",
        0.80: "#4c72b0",
        1.25: "#55a868",
    }
    widths = sorted({int(item["width"]) for item in refined_results})
    largest = max(widths)
    figure, axes = plt.subplots(1, 3, figsize=(7.15, 2.95))
    for axis, kind, title in [
        (axes[0], "forcing", "(a) Target tail → forcing"),
        (axes[1], "memory", "(b) Squared-spectrum tail → memory"),
    ]:
        selected = [
            item
            for item in refined_results
            if item["case"].kind == kind and item["width"] == largest
        ]
        for item in selected:
            case: Case = item["case"]
            color = colors[case.exponent]
            axis.plot(
                item["display_times"],
                item["tail_display"],
                color=color,
                linewidth=2.0,
            )
            axis.plot(
                item["display_times"],
                item["scaled_display"],
                color=color,
                linewidth=1.55,
                linestyle=(0, (4, 2.2)),
                marker=("o" if kind == "forcing" else "s"),
                markevery=18,
                markersize=2.5,
                markerfacecolor="white",
                markeredgewidth=0.65,
            )
            label_time = item["fit_times"][-1] * 0.82
            label_value = np.exp(
                np.interp(
                    np.log(label_time),
                    np.log(item["display_times"]),
                    np.log(item["tail_display"]),
                )
            )
            symbol = (
                r"q_{\mathcal{F}}"
                if kind == "forcing"
                else r"q_{\mathcal{K}}"
            )
            axis.text(
                label_time,
                label_value * 1.14,
                rf"${symbol}={case.exponent:g}$",
                color=color,
                fontsize=8.3,
                ha="right",
            )
        fit_lower = selected[0]["fit_lower"]
        fit_upper = selected[0]["fit_upper"]
        axis.axvspan(
            selected[0]["display_times"][0],
            fit_lower,
            color="#eeeeee",
            alpha=0.70,
            linewidth=0,
        )
        axis.axvspan(
            fit_upper,
            selected[0]["display_times"][-1],
            color="#eeeeee",
            alpha=0.70,
            linewidth=0,
        )
        axis.axvline(fit_lower, color="#888888", linewidth=0.7, linestyle=":")
        axis.axvline(fit_upper, color="#888888", linewidth=0.7, linestyle=":")
        axis.set_xscale("log")
        axis.set_yscale("log")
        axis.set_title(title, pad=4.0)
        axis.set_xlabel(r"intrinsic time $T$")
        axis.grid(True, which="major", alpha=0.16, linewidth=0.55)
    axes[0].set_ylabel("spectral mass / scaled dynamics")
    axes[1].text(
        0.04,
        0.05,
        "gray: excluded from fit",
        transform=axes[1].transAxes,
        color="#666666",
        fontsize=8.3,
    )

    axis = axes[2]
    marker = {"forcing": "o", "memory": "s"}
    for item in refined_results:
        case: Case = item["case"]
        width_index = widths.index(int(item["width"]))
        alpha_value = 0.28 + 0.72 * width_index / max(len(widths) - 1, 1)
        size = 22.0 + 10.0 * width_index
        axis.scatter(
            item["tail_exponent"],
            item["dynamics_exponent"],
            marker=marker[case.kind],
            s=size,
            facecolor=colors[case.exponent],
            edgecolor="white",
            linewidth=0.55,
            alpha=alpha_value,
            zorder=3,
        )
    lower = min(item["tail_exponent"] for item in refined_results) - 0.06
    upper = max(item["tail_exponent"] for item in refined_results) + 0.06
    axis.plot([lower, upper], [lower, upper], color="#222222", linewidth=1.25)
    axis.set_xlim(lower, upper)
    axis.set_ylim(lower, upper)
    axis.set_aspect("equal", adjustable="box")
    axis.set_title("(c) Exponent match", pad=4.0)
    axis.set_xlabel("fitted tail exponent")
    axis.set_ylabel("fitted temporal exponent")
    axis.grid(True, alpha=0.16, linewidth=0.55)
    largest_results = [item for item in refined_results if item["width"] == largest]
    maximum_error = max(
        item["tail_dynamics_exponent_error"] for item in largest_results
    )
    axis.text(
        0.05,
        0.94,
        (
            "finest spectral truncation"
            "\n"
            rf"max $|q_{{\rm time}}-q_{{\rm tail}}|={maximum_error:.3f}$"
        ),
        transform=axis.transAxes,
        va="top",
        fontsize=8.3,
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.82, "pad": 1.0},
    )
    axis.text(
        0.97,
        0.05,
        "lighter: coarser truncation",
        transform=axis.transAxes,
        ha="right",
        color="#666666",
        fontsize=8.3,
    )

    style_handles = [
        Line2D([0], [0], color="#333333", linewidth=2.0, label="cutoff spectral tail"),
        Line2D(
            [0],
            [0],
            color="#333333",
            linewidth=1.55,
            linestyle=(0, (4, 2.2)),
            marker="o",
            markerfacecolor="white",
            markersize=3.0,
            label="Gamma-scaled exact dynamics",
        ),
        Line2D([0], [0], marker="o", color="none", markerfacecolor="#777777", markeredgecolor="white", label="forcing"),
        Line2D([0], [0], marker="s", color="none", markerfacecolor="#777777", markeredgecolor="white", label="memory"),
    ]
    figure.legend(
        handles=style_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.01),
        ncol=4,
        frameon=False,
        columnspacing=1.25,
        handlelength=2.4,
    )
    figure.subplots_adjust(left=0.072, right=0.992, bottom=0.18, top=0.79, wspace=0.36)
    figure.savefig(output_dir / "spectral_criterion_triptych.png", dpi=300)
    figure.savefig(output_dir / "spectral_criterion_triptych.pdf", bbox_inches="tight")
    plt.close(figure)


def run(output_dir: Path = OUTPUT_DIR) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    base_results: list[dict[str, Any]] = []
    refined_results: list[dict[str, Any]] = []
    for width in EFFECTIVE_WIDTHS:
        for case in all_cases():
            base_results.append(
                evaluate_case(case, width, BASE_RELATIVE_BIN_WIDTH)
            )
            refined_results.append(
                evaluate_case(case, width, REFINED_RELATIVE_BIN_WIDTH)
            )

    base_map = {
        (item["case"].name, item["width"]): item for item in base_results
    }
    resolution_rows: list[dict[str, Any]] = []
    for refined in refined_results:
        key = (refined["case"].name, refined["width"])
        base = base_map[key]
        resolution_rows.append(
            {
                "case": key[0],
                "effective_width": key[1],
                "relative_l2_scaled_dynamics": relative_l2(
                    refined["scaled_display"], base["scaled_display"]
                ),
                "absolute_slope_difference": abs(
                    refined["dynamics_exponent"] - base["dynamics_exponent"]
                ),
                "base_quadrature_modes": base["mode_count"],
                "refined_quadrature_modes": refined["mode_count"],
            }
        )

    metric_rows = [metrics_row(item) for item in refined_results]
    write_csv(output_dir / "spectral_criterion_metrics.csv", metric_rows)
    write_csv(output_dir / "spectral_criterion_curves.csv", curve_rows(refined_results))
    write_csv(output_dir / "spectral_criterion_resolution.csv", resolution_rows)
    make_figure(refined_results, output_dir)

    largest = max(EFFECTIVE_WIDTHS)
    primary = [row for row in metric_rows if row["effective_width"] == largest]
    gates = {
        "tail_theory_exponent": max(
            row["tail_theory_exponent_error"] for row in primary
        ) <= 0.05,
        "theory_exponent": max(row["theory_exponent_error"] for row in primary) <= 0.05,
        "tail_dynamics_exponent": max(
            row["tail_dynamics_exponent_error"] for row in primary
        ) <= 0.05,
        "local_variation": max(
            row["dynamics_local_variation"] for row in primary
        ) <= 0.15,
        "gamma_prefactor_median": max(
            row["collapse_median_abs_log_error"] for row in primary
        ) <= 0.05,
        "gamma_prefactor_p90": max(
            row["collapse_p90_abs_log_error"] for row in primary
        ) <= 0.10,
        "curve_resolution": max(
            row["relative_l2_scaled_dynamics"] for row in resolution_rows
        ) <= 0.003,
        "slope_resolution": max(
            row["absolute_slope_difference"] for row in resolution_rows
        ) <= 0.002,
        "pointwise_stability": (
            ETA * (1.0 + 1.0 / BATCH) < 2.0
        ),
    }
    gates["overall"] = all(gates.values())
    summary = {
        "experiment": "spectral iff criterion: cutoff mass versus exact discrete modal transform",
        "status": "PASS" if gates["overall"] else "FAIL",
        "promotion_status": "PILOT_UNPROMOTED",
        "theorem": "iclr:thm:spectral_iff",
        "scope": (
            "Prescribed effective eigenpairs conditional on a frozen representation; "
            "pre-saturation joint-limit component diagnostic. This does not establish "
            "the PLRF population-to-empirical random-feature bridge or a full-loss law."
        ),
        "effective_widths": EFFECTIVE_WIDTHS,
        "forcing_exponents": FORCING_EXPONENTS,
        "memory_exponents": MEMORY_EXPONENTS,
        "paper_point": {
            "alpha": PAPER_ALPHA,
            "beta": PAPER_BETA,
            "q_forcing": 0.50,
            "q_memory": 0.75,
        },
        "constant_schedule": {"eta": ETA, "batch": BATCH},
        "numerical_resolution": {
            "base_relative_bin_width": BASE_RELATIVE_BIN_WIDTH,
            "refined_relative_bin_width": REFINED_RELATIVE_BIN_WIDTH,
        },
        "fit_window": {
            "lower": "m_eff^0.20",
            "upper": "m_eff^0.40",
            "points": FIT_POINTS,
        },
        "display_window": {
            "lower": "m_eff^0.10",
            "upper": "m_eff^0.45",
            "points": DISPLAY_POINTS,
        },
        "gates": gates,
        "primary_metrics": primary,
        "maximum_resolution_relative_l2": max(
            row["relative_l2_scaled_dynamics"] for row in resolution_rows
        ),
        "maximum_resolution_slope_difference": max(
            row["absolute_slope_difference"] for row in resolution_rows
        ),
        "source_sha256": {
            "spectral_criterion.py": sha256_file(Path(__file__)),
            "common.py": sha256_file(ROOT / "common.py"),
            "spectral_foundations.tex": sha256_file(
                ROOT.parents[1] / "files" / "spectral_foundations.tex"
            ),
            "appendix_a_model_and_spectral_characterization.tex": sha256_file(
                ROOT.parents[1]
                / "files"
                / "appendix_a_model_and_spectral_characterization.tex"
            ),
        },
        "files": {
            "figure_png": str(output_dir / "spectral_criterion_triptych.png"),
            "figure_pdf": str(output_dir / "spectral_criterion_triptych.pdf"),
            "curves": str(output_dir / "spectral_criterion_curves.csv"),
            "metrics": str(output_dir / "spectral_criterion_metrics.csv"),
            "resolution": str(output_dir / "spectral_criterion_resolution.csv"),
        },
    }
    write_json(output_dir / "summary.json", summary)
    return summary


if __name__ == "__main__":
    import json

    print(json.dumps(run(), indent=2, sort_keys=True))
