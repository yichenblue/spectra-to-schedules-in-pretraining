"""Plot the spectral-mass-to-temporal-component bridge for the six constructions."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from .common import write_json
from .spectral_geometric_spacing_sweep import build_geometric_case
from .spectral_loss_sgd import CASES, build_case
from .spectral_six_construction_summary import (
    CONSTRUCTION_TITLES,
    _power_reference,
    _set_plot_style,
    load_data,
)


Array = np.ndarray
ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "artifacts" / "spectral_mass_temporal_bridge_v001"
ETA = 0.05
BATCH_SIZE = 16
CASE_BY_NAME = {case.name: case for case in CASES}


def _construction(name: str, width: int) -> tuple[Array, Array]:
    if name == "geometric_spectrum":
        return build_geometric_case(width, 1.4)
    return build_case(CASE_BY_NAME[name], width)


def _retention(eigenvalues: Array) -> Array:
    values = (
        1.0 - 2.0 * ETA * eigenvalues
        + (1.0 + 1.0 / BATCH_SIZE) * np.square(ETA * eigenvalues)
    )
    if np.any(values <= 0.0) or np.any(values > 1.0):
        raise RuntimeError("modal retention lies outside (0,1]")
    return values


def _temporal_component(times: Array, eigenvalues: Array, weights: Array) -> Array:
    steps = np.rint(times / ETA).astype(np.int64)
    log_retention = np.log(_retention(eigenvalues))
    return np.asarray([
        float(weights @ np.exp(float(step) * log_retention))
        for step in steps
    ])


def _cutoff_mass(times: Array, eigenvalues: Array, weights: Array) -> Array:
    order = np.argsort(eigenvalues)
    spectrum = eigenvalues[order]
    cumulative = np.cumsum(weights[order])
    indices = np.searchsorted(spectrum, 1.0 / times, side="right") - 1
    values = np.zeros_like(times, dtype=float)
    valid = indices >= 0
    values[valid] = cumulative[indices[valid]]
    return values


def _normalize_pair(
    times: Array, temporal: Array, mass: Array, anchor: float = 1.0
) -> tuple[Array, Array]:
    eligible = (temporal > 0.0) & (mass > 0.0)
    if not np.any(eligible):
        raise RuntimeError("spectral bridge has no common positive support")
    candidates = np.flatnonzero(eligible)
    index = int(candidates[np.argmin(np.abs(np.log(times[candidates] / anchor)))])
    return temporal / temporal[index], mass / mass[index]


def _normalize_single(times: Array, values: Array, anchor: float = 1.0) -> Array:
    eligible = values > 0.0
    if not np.any(eligible):
        raise RuntimeError("curve has no positive support")
    candidates = np.flatnonzero(eligible)
    index = int(candidates[np.argmin(np.abs(np.log(times[candidates] / anchor)))])
    return values / values[index]


def _positive_times(times: Array) -> Array:
    values = np.asarray(times, dtype=float)
    return values[values > 0.0]


def _forcing_specs(data: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "name": "polynomial_canonical", "width": 65536,
            "times": _positive_times(data["polynomial_canonical"]["width_groups"][65536]["times"]),
            "color": "#277DA1",
        },
        {
            "name": "polynomial_sparse_target", "width": 65536,
            "times": _positive_times(data["polynomial_sparse_target"]["width_groups"][65536]["times"]),
            "color": "#43AA8B",
        },
        {
            "name": "stretched_exponential", "width": 2048,
            "times": _positive_times(data["stretched_exponential"]["width_groups"][2048]["times"]),
            "color": "#F8961E",
        },
        {
            "name": "geometric_spectrum", "width": 2048,
            "times": _positive_times(data["geometric_spectrum"]["width_groups"][2048]["times"]),
            "color": "#9B5DE5",
        },
        {
            "name": "polynomial_rapid_target", "width": 16384,
            "times": _positive_times(data["polynomial_rapid_target"]["batch_groups"][16]["times"]),
            "color": "#F9844A",
        },
        {
            "name": "spectral_gap", "width": 65536,
            "times": _positive_times(data["spectral_gap"]["width_groups"][65536]["times"]),
            "color": "#F94144",
        },
    ]


def _memory_specs(data: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "name": "polynomial_canonical", "width": 65536,
            "times": _positive_times(data["polynomial_canonical"]["width_groups"][65536]["times"]),
            "color": "#277DA1",
        },
        {
            "name": "polynomial_sparse_target", "width": 65536,
            "times": _positive_times(data["polynomial_sparse_target"]["width_groups"][65536]["times"]),
            "color": "#43AA8B",
        },
        {
            "name": "stretched_exponential", "width": 2048,
            "times": _positive_times(data["stretched_exponential"]["width_groups"][2048]["times"]),
            "color": "#F8961E",
        },
        {
            "name": "geometric_spectrum", "width": 2048,
            "times": _positive_times(data["geometric_spectrum"]["width_groups"][2048]["times"]),
            "color": "#9B5DE5",
        },
        {
            "name": "polynomial_rapid_target", "width": 16384,
            "times": _positive_times(data["polynomial_rapid_target"]["batch_groups"][16]["times"]),
            "color": "#F9844A",
        },
        {
            "name": "spectral_gap", "width": 65536,
            "times": _positive_times(data["spectral_gap"]["width_groups"][65536]["times"]),
            "color": "#F94144",
        },
    ]


def _add_power_reference(
    axis: Any,
    *,
    name: str,
    times: Array,
    values: Array,
    component: str,
) -> None:
    if component == "forcing":
        settings = {
            "polynomial_canonical": (0.5, r"$T^{-1/2}$", 1.0, 128.0, 8.0, 0.45),
            "polynomial_sparse_target": (0.5, r"$T^{-1/2}$", 1.0, 128.0, 8.0, 0.45),
            "stretched_exponential": (1.0, r"$T^{-1}$", 64.0, np.inf, 4096.0, 0.24),
            "geometric_spectrum": (1.75, r"$T^{-1.75}$", 2.0, np.inf, 64.0, 2.0),
        }
    elif component == "memory":
        settings = {
            "polynomial_canonical": (0.75, r"$T^{-3/4}$", 1.0, 128.0, 8.0, 0.45),
            "polynomial_sparse_target": (0.75, r"$T^{-3/4}$", 1.0, 128.0, 8.0, 0.45),
            "stretched_exponential": (2.0, r"$T^{-2}$", 64.0, np.inf, 4096.0, 4.0),
            "geometric_spectrum": (2.0, r"$T^{-2}$", 2.0, np.inf, 64.0, 2.0),
            "polynomial_rapid_target": (0.75, r"$T^{-3/4}$", 1.0, 64.0, 8.0, 0.45),
        }
    else:
        raise ValueError(f"unknown component: {component}")
    setting = settings.get(name)
    if setting is None:
        return
    exponent, label, lower, upper, anchor, vertical_shift = setting
    mask = (times >= lower) & (times <= upper)
    reference_times = times[mask]
    reference_values = values[mask]
    if reference_times.size == 0:
        return
    reference = _power_reference(
        reference_times, reference_values, exponent, anchor
    )
    vertical_shift = max(
        vertical_shift,
        1.25 * float(np.max(reference_values / reference)),
    )
    axis.loglog(
        reference_times,
        vertical_shift * reference,
        color="black", linestyle="--", linewidth=2.0,
        label=label, zorder=4,
    )


def _risk_group(
    data: dict[str, dict[str, Any]], name: str, width: int | None = None
) -> dict[str, Any]:
    if name == "polynomial_rapid_target":
        return data[name]["batch_groups"][16]
    if width is None:
        return data[name]
    return data[name]["width_groups"][width]


def _risk_curve(group: dict[str, Any]) -> tuple[Array, Array]:
    times = _positive_times(group["times"])
    offset = int(np.sum(np.asarray(group["times"]) <= 0.0))
    median = np.median(group["traces"], axis=0)[offset:]
    return times, _normalize_single(times, median)


def _style_axis(axis: Any, panel_index: int, title: str) -> None:
    axis.set_title(title, pad=10)
    axis.set(xlabel=r"Intrinsic time $T=\sum\eta$", ylabel="Normalized magnitude")
    axis.grid(alpha=0.16, which="major")
    axis.grid(alpha=0.045, which="minor")
    axis.text(
        -0.12, 1.10, f"({chr(ord('a') + panel_index)})",
        transform=axis.transAxes, fontsize=13, fontweight="bold",
    )
    for spine in axis.spines.values():
        spine.set_color("0.35")
        spine.set_linewidth(0.8)


def make_forcing_figure(data: dict[str, dict[str, Any]], output: Path) -> list[dict[str, Any]]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    _set_plot_style(plt)
    figure, axes = plt.subplots(2, 3, figsize=(16.0, 9.0), constrained_layout=True)
    records: list[dict[str, Any]] = []
    for panel_index, (axis, spec) in enumerate(zip(axes.flat, _forcing_specs(data), strict=True)):
        eigenvalues, target = _construction(spec["name"], spec["width"])
        weights = eigenvalues * np.square(target)
        times = spec["times"]
        forcing = _temporal_component(times, eigenvalues, weights)
        mass = _cutoff_mass(times, eigenvalues, weights)
        forcing, mass = _normalize_pair(times, forcing, mass)
        axis.loglog(
            times, forcing, color=spec["color"], linewidth=3.0,
            label=r"$F(T)$", zorder=3,
        )
        positive = mass > 0.0
        axis.loglog(
            times[positive], mass[positive], color=spec["color"], linestyle=":",
            linewidth=2.35, drawstyle="steps-post",
            label=r"$\nu^{\mathcal{F}}((0,T^{-1}])$", zorder=2,
        )
        _add_power_reference(
            axis, name=spec["name"], times=times,
            values=forcing, component="forcing",
        )
        _style_axis(axis, panel_index, CONSTRUCTION_TITLES[spec["name"]])
        axis.legend(frameon=False, fontsize=9, loc="upper right")
        records.append({
            "construction": spec["name"], "width": spec["width"],
            "max_intrinsic_time": float(times[-1]),
        })
    figure.text(
        0.5, -0.008,
        r"Black dashed lines: theoretical slopes (vertically shifted).",
        ha="center", va="top", fontsize=12,
    )
    figure.savefig(output / "forcing_spectral_bridge.png", dpi=240, bbox_inches="tight")
    figure.savefig(output / "forcing_spectral_bridge.pdf", bbox_inches="tight")
    plt.close(figure)
    return records


def make_forcing_risk_figure(
    data: dict[str, dict[str, Any]], output: Path
) -> list[dict[str, Any]]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    _set_plot_style(plt)
    figure, axes = plt.subplots(2, 3, figsize=(16.0, 9.0), constrained_layout=True)
    records: list[dict[str, Any]] = []
    for panel_index, (axis, spec) in enumerate(zip(axes.flat, _forcing_specs(data), strict=True)):
        eigenvalues, target = _construction(spec["name"], spec["width"])
        weights = eigenvalues * np.square(target)
        times = spec["times"]
        forcing = _temporal_component(times, eigenvalues, weights)
        mass = _cutoff_mass(times, eigenvalues, weights)
        forcing, mass = _normalize_pair(times, forcing, mass)
        risk_times, risk = _risk_curve(_risk_group(data, spec["name"], spec["width"]))
        axis.loglog(
            risk_times, risk, color=spec["color"], linewidth=3.0,
            label="Risk", zorder=3,
        )
        axis.loglog(
            times, forcing, color="0.35", linewidth=2.0,
            linestyle=(0, (5, 2, 1.2, 2)), label=r"$F(T)$", zorder=2,
        )
        positive = mass > 0.0
        axis.loglog(
            times[positive], mass[positive], color="black", linestyle="--",
            linewidth=1.9, label=r"$\nu^{\mathcal{F}}((0,T^{-1}])$", zorder=1,
        )
        _style_axis(axis, panel_index, CONSTRUCTION_TITLES[spec["name"]])
        axis.legend(frameon=False, fontsize=8.5, loc="upper right")
        records.append({
            "construction": spec["name"], "width": spec["width"],
            "max_intrinsic_time": float(times[-1]),
        })
    figure.text(
        0.5, -0.008, r"All curves are normalized at $T=1$.",
        ha="center", va="top", fontsize=12,
    )
    figure.savefig(output / "forcing_spectral_bridge_with_risk.png", dpi=240, bbox_inches="tight")
    figure.savefig(output / "forcing_spectral_bridge_with_risk.pdf", bbox_inches="tight")
    plt.close(figure)
    return records


def make_memory_figure(data: dict[str, dict[str, Any]], output: Path) -> list[dict[str, Any]]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    _set_plot_style(plt)
    figure, axes = plt.subplots(2, 3, figsize=(16.0, 9.0), constrained_layout=True)
    records: list[dict[str, Any]] = []
    for panel_index, (axis, spec) in enumerate(zip(axes.flat, _memory_specs(data), strict=True)):
        eigenvalues, _ = _construction(spec["name"], spec["width"])
        weights = np.square(eigenvalues)
        times = spec["times"]
        kernel = _temporal_component(times, eigenvalues, weights)
        mass = _cutoff_mass(times, eigenvalues, weights)
        kernel, mass = _normalize_pair(times, kernel, mass)
        axis.loglog(
            times, kernel, color=spec["color"], linewidth=3.0,
            label=r"$(B/\eta^2)K(T)$", zorder=3,
        )
        positive = mass > 0.0
        axis.loglog(
            times[positive], mass[positive], color=spec["color"], linestyle=":",
            linewidth=2.35, drawstyle="steps-post",
            label=r"$\nu^{\mathcal{K}}((0,T^{-1}])$", zorder=2,
        )
        _add_power_reference(
            axis, name=spec["name"], times=times,
            values=kernel, component="memory",
        )
        _style_axis(axis, panel_index, CONSTRUCTION_TITLES[spec["name"]])
        axis.legend(frameon=False, fontsize=9, loc="upper right")
        records.append({
            "construction": spec["name"], "width": spec["width"],
            "max_intrinsic_time": float(times[-1]),
        })
    figure.text(
        0.5, -0.008,
        r"Black dashed lines: theoretical slopes (vertically shifted).",
        ha="center", va="top", fontsize=12,
    )
    figure.savefig(output / "memory_spectral_bridge.png", dpi=240, bbox_inches="tight")
    figure.savefig(output / "memory_spectral_bridge.pdf", bbox_inches="tight")
    plt.close(figure)
    return records


def make_memory_risk_figure(
    data: dict[str, dict[str, Any]], output: Path
) -> list[dict[str, Any]]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    _set_plot_style(plt)
    figure, axes = plt.subplots(2, 2, figsize=(11.2, 8.4), constrained_layout=True)
    records: list[dict[str, Any]] = []
    polynomial_risks = (
        ("polynomial_canonical", 65536, "canonical", "#277DA1"),
        ("polynomial_sparse_target", 65536, "sparse", "#43AA8B"),
        ("polynomial_rapid_target", None, "rapid", "#F9844A"),
    )
    for panel_index, (axis, spec) in enumerate(zip(axes.flat, _memory_specs(data), strict=True)):
        eigenvalues, _ = _construction(spec["name"], spec["width"])
        weights = np.square(eigenvalues)
        times = spec["times"]
        kernel = _temporal_component(times, eigenvalues, weights)
        mass = _cutoff_mass(times, eigenvalues, weights)
        kernel, mass = _normalize_pair(times, kernel, mass)
        if spec["name"] == "polynomial_canonical":
            for name, width, label, color in polynomial_risks:
                risk_times, risk = _risk_curve(_risk_group(data, name, width))
                axis.loglog(
                    risk_times, risk, color=color, linewidth=2.65,
                    label=rf"Risk: {label}", zorder=3,
                )
        else:
            risk_times, risk = _risk_curve(_risk_group(data, spec["name"], spec["width"]))
            axis.loglog(
                risk_times, risk, color=spec["color"], linewidth=3.0,
                label="Risk", zorder=3,
            )
        axis.loglog(
            times, kernel, color="0.35", linewidth=2.0,
            linestyle=(0, (5, 2, 1.2, 2)), label=r"$(B/\eta^2)K(T)$", zorder=2,
        )
        positive = mass > 0.0
        axis.loglog(
            times[positive], mass[positive], color="black", linestyle="--",
            linewidth=1.9, label=r"$\nu^{\mathcal{K}}((0,T^{-1}])$", zorder=1,
        )
        _style_axis(axis, panel_index, CONSTRUCTION_TITLES[spec["name"]])
        axis.legend(frameon=False, fontsize=8.3, loc="upper right")
        records.append({
            "construction": spec["name"], "width": spec["width"],
            "max_intrinsic_time": float(times[-1]),
        })
    figure.text(
        0.5, -0.008, r"All curves are normalized at $T=1$.",
        ha="center", va="top", fontsize=12,
    )
    figure.savefig(output / "memory_spectral_bridge_with_risk.png", dpi=240, bbox_inches="tight")
    figure.savefig(output / "memory_spectral_bridge_with_risk.pdf", bbox_inches="tight")
    plt.close(figure)
    return records


def main() -> int:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    data = load_data()
    forcing = make_forcing_figure(data, OUTPUT)
    memory = make_memory_figure(data, OUTPUT)
    write_json(OUTPUT / "summary.json", {
        "schema_version": "spectral_mass_temporal_bridge_v001",
        "is_new_training_run": False,
        "eta": ETA, "batch_size": BATCH_SIZE,
        "normalization": "each curve divided by its value at the evaluation closest to T=1",
        "forcing_panels": forcing, "memory_panels": memory,
        "uses_expected_sgd_recurrence": False,
        "quantities_are_exact_component_spectral_sums": True,
    })
    print(json.dumps({"output": str(OUTPUT), "figure_count": 2}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
