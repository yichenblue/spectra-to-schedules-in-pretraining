"""Create a paper-ready risk--forcing--memory triptych for the rapid target."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from .common import sha256_file, write_json
from .spectral_loss_sgd import CASES, build_case
from .spectral_mass_temporal_bridge import (
    _cutoff_mass,
    _temporal_component,
)
from .spectral_six_construction_summary import (
    _power_reference,
    _read_rows,
    _set_plot_style,
    _stack,
)


Array = np.ndarray
ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "artifacts" / "spectral_rapid_target_triptych_v009"
RAPID_SOURCES = tuple(
    ROOT
    / "artifacts"
    / f"spectral_rapid_target_t1e3_w524288_seed{seed}_v001"
    / "curves.csv"
    for seed in range(31501, 31521)
)
CASE = next(case for case in CASES if case.name == "polynomial_rapid_target")
WIDTH = 524_288
COLOR = "#F9844A"
NU_COLOR = "#F7B58E"


def _positive_curve(group: dict[str, Any]) -> tuple[Array, Array]:
    times = np.asarray(group["times"], dtype=float)
    median = np.median(np.asarray(group["traces"], dtype=float), axis=0)
    positive = (times > 0.0) & (median > 0.0)
    return times[positive], median[positive]


def _load_rapid_group() -> dict[str, Any]:
    rows: list[dict[str, str]] = []
    for source in RAPID_SOURCES:
        rows.extend(_read_rows(source))
    times, traces, seeds = _stack(
        rows,
        filters={
            "condition": "sampled_minibatch_sgd",
            "batch_size": "16",
        },
        value_column="population_risk",
    )
    return {"times": times, "traces": traces, "seeds": seeds}


def _reference_above(
    times: Array,
    values: Array,
    *,
    exponent: float,
    anchor: float,
    clearance: float = 1.30,
) -> Array:
    reference = _power_reference(times, values, exponent, anchor)
    shift = clearance * float(np.max(values / reference))
    return shift * reference


def _style_panel(axis: Any, title: str, ylabel: str) -> None:
    axis.set_xlabel(r"$T=\eta t$")
    axis.set_ylabel(ylabel)
    axis.set_title(title, fontsize=18, fontweight="bold", loc="center", pad=10)
    axis.grid(alpha=0.17, which="major")
    axis.grid(alpha=0.045, which="minor")
    for spine in axis.spines.values():
        spine.set_color("0.35")
        spine.set_linewidth(0.8)


def make_figure(output: Path) -> dict[str, Any]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import NullLocator

    rapid = _load_rapid_group()
    eigenvalues, target = build_case(CASE, WIDTH)
    component_times, _ = _positive_curve(rapid)

    forcing_weights = eigenvalues * np.square(target)
    forcing = _temporal_component(component_times, eigenvalues, forcing_weights)
    forcing_mass = _cutoff_mass(component_times, eigenvalues, forcing_weights)
    forcing_total = float(np.sum(forcing_weights))
    forcing /= forcing_total
    forcing_mass /= forcing_total

    memory_weights = np.square(eigenvalues)
    memory = _temporal_component(component_times, eigenvalues, memory_weights)
    memory_mass = _cutoff_mass(component_times, eigenvalues, memory_weights)
    memory_total = float(np.sum(memory_weights))
    memory /= memory_total
    memory_mass /= memory_total

    _set_plot_style(plt)
    plt.rcParams.update({
        "font.size": 17,
        "axes.labelsize": 18,
        "xtick.labelsize": 16,
        "ytick.labelsize": 16,
        "legend.fontsize": 16,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })
    figure, axes = plt.subplots(
        1, 3, figsize=(15.8, 4.65), constrained_layout=True
    )

    # Literal minibatch-SGD population risk.
    risk_times, risk_median = _positive_curve(rapid)
    fit_window = (100.0, 800.0)
    fit_mask = (risk_times >= fit_window[0]) & (risk_times <= fit_window[1])
    fit_log_time = np.log(risk_times[fit_mask])
    fit_log_risk = np.log(risk_median[fit_mask])
    fit_slope, fit_intercept = np.polyfit(fit_log_time, fit_log_risk, 1)
    fit_prediction = fit_intercept + fit_slope * fit_log_time
    fit_r2 = 1.0 - float(np.sum(np.square(fit_log_risk - fit_prediction))) / float(
        np.sum(np.square(fit_log_risk - np.mean(fit_log_risk)))
    )
    axes[0].loglog(
        risk_times, risk_median, color=COLOR, linewidth=3.4, zorder=3,
    )
    tail = risk_times >= 16.0
    risk_reference = _reference_above(
        risk_times[tail], risk_median[tail], exponent=0.75, anchor=64.0,
        clearance=1.65,
    )
    axes[0].loglog(
        risk_times[tail],
        risk_reference,
        color="black", linestyle="--", linewidth=2.6,
    )
    risk_label_index = int(np.argmin(np.abs(np.log(risk_times[tail] / 95.0))))
    axes[0].text(
        risk_times[tail][risk_label_index], 1.85 * risk_reference[risk_label_index],
        r"$T^{-3/4}$", color="black", fontsize=16.5,
        ha="center", va="bottom",
    )
    _style_panel(
        axes[0], "Loss", "Risk",
    )

    # Exact target-weighted forcing and its low-spectrum cumulative mass.
    forcing_mass_positive = forcing_mass > 0.0
    axes[1].loglog(
        component_times, forcing,
        color=COLOR, linewidth=3.5, label=r"$F(T)$", zorder=3,
    )
    axes[1].loglog(
        component_times[forcing_mass_positive],
        forcing_mass[forcing_mass_positive],
        color=NU_COLOR, linestyle="-", linewidth=2.85,
        drawstyle="steps-post",
        label=r"$\nu^{\mathcal{F}}((0,T^{-1}])$", zorder=2,
    )
    _style_panel(
        axes[1], "Forcing", "magnitude",
    )
    axes[1].legend(frameon=False, loc="upper right")
    axes[1].set_ylim(1e-16, 2.0)
    axes[1].set_yticks([1e0, 1e-4, 1e-8, 1e-12, 1e-16])
    axes[1].yaxis.set_minor_locator(NullLocator())

    # Exact one-injection memory and its low-spectrum cumulative mass.
    axes[2].loglog(
        component_times, memory, color=COLOR, linewidth=3.5,
        label=r"$(B/\eta^2)K(T)$", zorder=3,
    )
    memory_positive = memory_mass > 0.0
    axes[2].loglog(
        component_times[memory_positive], memory_mass[memory_positive],
        color=NU_COLOR, linestyle="-", linewidth=2.85,
        drawstyle="steps-post",
        label=r"$\nu^{\mathcal{K}}((0,T^{-1}])$", zorder=2,
    )
    memory_tail = component_times >= 10.0
    memory_tail_times = component_times[memory_tail]
    memory_tail_values = memory[memory_tail]
    memory_tail_mass = memory_mass[memory_tail]
    memory_coefficient = float(np.max(
        memory_tail_values * np.power(memory_tail_times, 0.75)
    ))
    if memory_tail_times.size > 1:
        # For steps-post, each plateau persists until the next sample; check
        # its right edge, where a decreasing reference is smallest.
        memory_coefficient = max(
            memory_coefficient,
            float(np.max(
                memory_tail_mass[:-1]
                * np.power(memory_tail_times[1:], 0.75)
            )),
        )
    memory_reference = 1.60 * memory_coefficient * np.power(
        memory_tail_times, -0.75
    )
    axes[2].loglog(
        memory_tail_times,
        memory_reference,
        color="black", linestyle="--", linewidth=2.6, zorder=4,
    )
    memory_label_index = int(np.argmin(np.abs(np.log(memory_tail_times / 70.0))))
    axes[2].text(
        memory_tail_times[memory_label_index],
        1.85 * memory_reference[memory_label_index],
        r"$T^{-3/4}$", color="black", fontsize=16.5,
        ha="center", va="bottom",
    )
    _style_panel(
        axes[2], "Memory", "magnitude",
    )
    axes[2].legend(frameon=False, loc="upper right")

    common_xlim = (float(component_times[0]), float(component_times[-1]))
    axes[0].set_xlim(common_xlim)
    axes[2].set_xlim(common_xlim)
    axes[1].set_xlim(common_xlim)

    png = output / "rapid_target_risk_forcing_memory.png"
    pdf = output / "rapid_target_risk_forcing_memory.pdf"
    figure.savefig(png, dpi=300, bbox_inches="tight", facecolor="white")
    figure.savefig(pdf, bbox_inches="tight", facecolor="white")
    plt.close(figure)

    return {
        "construction": "polynomial_rapid_target",
        "width": WIDTH,
        "risk_batches": [16],
        "component_batch_size": 16,
        "max_intrinsic_time": float(component_times[-1]),
        "risk_is_literal_minibatch_sgd": True,
        "forcing_and_memory_are_exact_spectral_sums": True,
        "normalization": "each component and its cumulative spectral mass divided by their shared total spectral mass, equal to the component value at T=0",
        "risk_summary": "median over 20 seeds; IQR retained in raw data but not plotted",
        "risk_endpoint": {
            "intrinsic_time": float(risk_times[-1]),
            "median_population_risk": float(risk_median[-1]),
        },
        "risk_tail_fit": {
            "intrinsic_time_window": list(fit_window),
            "power_exponent": float(-fit_slope),
            "r_squared": fit_r2,
        },
        "risk_sources": [str(source) for source in RAPID_SOURCES],
        "risk_source_sha256": {
            str(source): sha256_file(source) for source in RAPID_SOURCES
        },
        "power_reference": "vertically shifted asymptotic T^-3/4 memory-slope guide",
        "forcing_asymptotic": "raw normalized view clipped below 1e-16: F and its forcing cutoff mass are plotted directly; both are faster than inverse powers",
        "png": str(png),
        "pdf": str(pdf),
    }


def main() -> int:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    summary = make_figure(OUTPUT)
    write_json(OUTPUT / "summary.json", {
        "schema_version": "spectral_rapid_target_triptych_v009",
        "is_new_training_run": False,
        **summary,
    })
    print(json.dumps({"output": str(OUTPUT), "figure_count": 1}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
