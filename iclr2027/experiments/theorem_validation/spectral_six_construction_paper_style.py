"""Restyle the three six-construction mechanism figures for the paper."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from .common import sha256_file, write_json
from .spectral_mass_temporal_bridge import (
    _construction,
    _cutoff_mass,
    _forcing_specs,
    _temporal_component,
)
from .spectral_six_construction_summary import (
    CONSTRUCTION_TITLES,
    _power_reference,
    _set_plot_style,
    load_data,
)


Array = np.ndarray
ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "artifacts" / "spectral_six_construction_paper_style_v002"

COLORS = {
    "polynomial_canonical": "#277DA1",
    "polynomial_sparse_target": "#43AA8B",
    "stretched_exponential": "#F8961E",
    "geometric_spectrum": "#9B5DE5",
    "polynomial_rapid_target": "#F9844A",
    "spectral_gap": "#F94144",
}

PANEL_ORDER = (
    "polynomial_canonical",
    "polynomial_sparse_target",
    "stretched_exponential",
    "geometric_spectrum",
    "polynomial_rapid_target",
    "spectral_gap",
)

ALGEBRAIC_FORCING_ORDER = (
    "polynomial_canonical",
    "polynomial_sparse_target",
    "stretched_exponential",
    "geometric_spectrum",
)

FASTER_THAN_POWER_FORCING_ORDER = (
    "polynomial_rapid_target",
    "spectral_gap",
)

THEORY_GUIDES = {
    "loss": {
        "polynomial_canonical": (0.5, r"$T^{-1/2}$"),
        "polynomial_sparse_target": (0.5, r"$T^{-1/2}$"),
        "stretched_exponential": (1.0, r"$T^{-1}$"),
        "geometric_spectrum": (1.75, r"$T^{-1.75}$"),
        "polynomial_rapid_target": (0.75, r"$T^{-3/4}$"),
        "spectral_gap": None,
    },
    "forcing": {
        "polynomial_canonical": (0.5, r"$T^{-1/2}$"),
        "polynomial_sparse_target": (0.5, r"$T^{-1/2}$"),
        "stretched_exponential": (1.0, r"$T^{-1}$"),
        "geometric_spectrum": (1.75, r"$T^{-1.75}$"),
        "polynomial_rapid_target": None,
        "spectral_gap": None,
    },
    "memory": {
        "polynomial_canonical": (0.75, r"$T^{-3/4}$"),
        "polynomial_sparse_target": (0.75, r"$T^{-3/4}$"),
        "stretched_exponential": (2.0, r"$T^{-2}$"),
        "geometric_spectrum": (2.0, r"$T^{-2}$"),
        "polynomial_rapid_target": (0.75, r"$T^{-3/4}$"),
        "spectral_gap": None,
    },
}

# Manual placement for every panel.  line_x > 1 moves the dashed guide right,
# line_y > 1 moves it up; label_x and label_y move only its T^{-q} label
# relative to the guide.  line_range controls the guide's visible extent.
MANUAL_LAYOUT = {
    "loss": {
        "polynomial_canonical": {
            "line_range": (1.0, 128.0), "line_anchor": 8.0,
            "line_clearance": 1.42, "line_x": 1.50, "line_y": 0.80,
            "label_anchor": 32.0, "label_x": 1.0, "label_y": 1.55,
            "label_ha": "center",
        },
        "polynomial_sparse_target": {
            "line_range": (1.0, 128.0), "line_anchor": 8.0,
            "line_clearance": 1.42, "line_x": 1.50, "line_y": 0.80,
            "label_anchor": 32.0, "label_x": 1.0, "label_y": 1.55,
            "label_ha": "center",
        },
        "stretched_exponential": {
            "line_range": (2048.0, np.inf), "line_anchor": 4096.0,
            "line_clearance": 4.10, "line_x": 1.0, "line_y": 1.0,
            "label_anchor": 80_000.0, "label_x": 1.0, "label_y": 4.80,
            "label_ha": "center",
        },
        "geometric_spectrum": {
            "line_range": (2.0, np.inf), "line_anchor": 64.0,
            "line_clearance": 6.10, "line_x": 1.0, "line_y": 1.0,
            "label_anchor": 260.0, "label_x": 1.0, "label_y": 1.90,
            "label_ha": "left",
        },
        "polynomial_rapid_target": {
            "line_range": (16.0, np.inf), "line_anchor": 40.0,
            "line_clearance": 2.42, "line_x": 1.0, "line_y": 1.0,
            "label_anchor": 42.0, "label_x": 1.0, "label_y": 1.55,
            "label_ha": "center",
        },
        "spectral_gap": None,
    },
    "forcing": {
        "polynomial_canonical": {
            "line_range": (4.0, 128.0), "line_anchor": 8.0,
            "line_clearance": 1.42, "line_x": 1.50, "line_y": 0.80,
            "label_anchor": 64.0, "label_x": 1.0, "label_y": 1.75,
            "label_ha": "center",
        },
        "polynomial_sparse_target": {
            "line_range": (4.0, 128.0), "line_anchor": 8.0,
            "line_clearance": 1.42, "line_x": 1.50, "line_y": 0.80,
            "label_anchor": 64.0, "label_x": 1.0, "label_y": 1.75,
            "label_ha": "center",
        },
        "stretched_exponential": {
            "line_range": (1024.0, np.inf), "line_anchor": 4096.0,
            "line_clearance": 2.42, "line_x": 1.0, "line_y": 1.0,
            "label_anchor": 60_000.0, "label_x": 1.0, "label_y": 4.20,
            "label_ha": "center",
        },
        "geometric_spectrum": {
            "line_range": (2.0, np.inf), "line_anchor": 64.0,
            "line_clearance": 2.42, "line_x": 1.0, "line_y": 1.0,
            "label_anchor": 512.0, "label_x": 1.0, "label_y": 6.50,
            "label_ha": "center",
        },
        "polynomial_rapid_target": None,
        "spectral_gap": None,
    },
    "memory": {
        "polynomial_canonical": {
            "line_range": (4.0, 128.0), "line_anchor": 8.0,
            "line_clearance": 1.42, "line_x": 1.50, "line_y": 0.80,
            "label_anchor": 64.0, "label_x": 1.0, "label_y": 1.75,
            "label_ha": "center",
        },
        "polynomial_sparse_target": {
            "line_range": (4.0, 128.0), "line_anchor": 8.0,
            "line_clearance": 1.42, "line_x": 1.50, "line_y": 0.80,
            "label_anchor":64.0, "label_x": 1.0, "label_y": 1.75,
            "label_ha": "center",
        },
        "stretched_exponential": {
            "line_range": (1024.0, np.inf), "line_anchor": 4096.0,
            "line_clearance": 6.10, "line_x": 1.0, "line_y": 1.0,
            "label_anchor": 80_000.0, "label_x": 1.0, "label_y": 6.20,
            "label_ha": "center",
        },
        "geometric_spectrum": {
            "line_range": (2.0, np.inf), "line_anchor": 64.0,
            "line_clearance": 2.42, "line_x": 1.0, "line_y": 1.0,
            "label_anchor": 512.0, "label_x": 1.0, "label_y": 6.50,
            "label_ha": "center",
        },
        "polynomial_rapid_target": {
            "line_range": (4.0, np.inf), "line_anchor": 8.0,
            "line_clearance": 1.42, "line_x": 1.0, "line_y": 1.0,
            "label_anchor": 32.0, "label_x": 1.0, "label_y": 1.60,
            "label_ha": "center",
        },
        "spectral_gap": None,
    },
}


def _mix_with_white(color: str, fraction: float) -> tuple[float, float, float]:
    from matplotlib.colors import to_rgb

    rgb = np.asarray(to_rgb(color), dtype=float)
    return tuple((1.0 - fraction) * rgb + fraction * np.ones(3))


def _style_axis(axis: Any, *, title: str, ylabel: str) -> None:
    axis.set_xlabel(r"$T=\eta t$")
    axis.set_ylabel(ylabel)
    axis.set_title(title, fontsize=18, fontweight="bold", loc="center", pad=10)
    axis.grid(alpha=0.17, which="major")
    axis.grid(alpha=0.045, which="minor")
    for spine in axis.spines.values():
        spine.set_color("0.35")
        spine.set_linewidth(0.8)


def _configure_style(plt: Any) -> None:
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


def _positive_median(group: dict[str, Any]) -> tuple[Array, Array]:
    times = np.asarray(group["times"], dtype=float)
    median = np.median(np.asarray(group["traces"], dtype=float), axis=0)
    positive = (times > 0.0) & (median > 0.0)
    return times[positive], median[positive]


def _guide_above(
    axis: Any,
    *,
    times: Array,
    anchor_values: Array,
    comparison_values: Iterable[Array],
    theory: tuple[float, str],
    layout: dict[str, Any],
) -> None:
    exponent, label = theory
    lower, upper = layout["line_range"]
    anchor = float(layout["line_anchor"])
    mask = (times >= lower) & (times <= upper) & (anchor_values > 0.0)
    guide_times = times[mask]
    guide_anchor = anchor_values[mask]
    if guide_times.size < 2:
        return

    reference = _power_reference(
        guide_times, guide_anchor, exponent, anchor,
    )
    envelope = np.asarray(guide_anchor, dtype=float)
    for values in comparison_values:
        values = np.asarray(values, dtype=float)
        if values.shape != times.shape:
            raise ValueError("guide comparison curve has a different time grid")
        envelope = np.maximum(envelope, values[mask])
    positive = (reference > 0.0) & np.isfinite(envelope)
    shift = float(layout["line_clearance"]) * float(
        np.max(envelope[positive] / reference[positive])
    )
    shifted = shift * reference
    display_times = float(layout["line_x"]) * guide_times
    display_values = float(layout["line_y"]) * shifted
    axis.loglog(
        display_times, display_values, color="black", linestyle="--",
        linewidth=2.6, zorder=5,
    )
    label_index = int(
        np.argmin(
            np.abs(np.log(guide_times / float(layout["label_anchor"])))
        )
    )
    axis.text(
        float(layout["label_x"]) * display_times[label_index],
        float(layout["label_y"]) * display_values[label_index],
        label,
        color="black", fontsize=16.5,
        ha=str(layout["label_ha"]), va="bottom",
    )


def _loss_groups(
    data: dict[str, dict[str, Any]], name: str,
) -> list[tuple[str, dict[str, Any], Any]]:
    base = COLORS[name]
    if name in {"polynomial_canonical", "polynomial_sparse_target"}:
        widths = (4096, 16384, 65536)
        fractions = (0.55, 0.30, 0.0)
        return [
            (
                rf"$d={width:,}$",
                data[name]["width_groups"][width],
                _mix_with_white(base, fraction),
            )
            for width, fraction in zip(widths, fractions, strict=True)
        ]
    if name == "polynomial_rapid_target":
        fractions = {16: 0.0, 64: 0.30, 256: 0.55}
        return [
            (
                rf"$B={batch_size}$",
                data[name]["batch_groups"][batch_size],
                _mix_with_white(base, fractions[batch_size]),
            )
            for batch_size in (16, 64, 256)
        ]
    width = 2048 if name in {"stretched_exponential", "geometric_spectrum"} else 65536
    return [
        (
            rf"$d={width:,}$",
            data[name]["width_groups"][width],
            base,
        )
    ]


def _plot_loss_panel(
    axis: Any,
    data: dict[str, dict[str, Any]],
    name: str,
) -> dict[str, Any]:
    groups = _loss_groups(data, name)
    medians: list[Array] = []
    common_times: Array | None = None
    for label, group, color in groups:
        times, median = _positive_median(group)
        if common_times is None:
            common_times = times
        elif not np.array_equal(common_times, times):
            raise RuntimeError(f"loss grids differ within {name}")
        medians.append(median)
        axis.loglog(
            times, median, color=color, linewidth=3.4,
            label=label, zorder=3,
        )
    assert common_times is not None
    theory = THEORY_GUIDES["loss"][name]
    layout = MANUAL_LAYOUT["loss"][name]
    if theory is not None and layout is not None:
        _guide_above(
            axis,
            times=common_times,
            anchor_values=medians[0] if name == "polynomial_rapid_target" else medians[-1],
            comparison_values=medians,
            theory=theory,
            layout=layout,
        )
    _style_axis(axis, title=CONSTRUCTION_TITLES[name], ylabel="Risk")
    if len(groups) > 1:
        axis.legend(frameon=False, loc="lower left")
    return {
        "construction": name,
        "curves": [label for label, _, _ in groups],
        "max_intrinsic_time": float(common_times[-1]),
    }


def _plot_component_panel(
    axis: Any,
    data: dict[str, dict[str, Any]],
    name: str,
    *,
    component: str,
    specs: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    from matplotlib.ticker import NullLocator

    spec = specs[name]
    times = np.asarray(spec["times"], dtype=float)
    eigenvalues, target = _construction(name, int(spec["width"]))
    if component == "forcing":
        weights = eigenvalues * np.square(target)
        main_label = r"$F(T)$"
        mass_label = r"$\nu^{\mathcal{F}}((0,T^{-1}])$"
    else:
        weights = np.square(eigenvalues)
        main_label = r"$(B/\eta^2)K(T)$"
        mass_label = r"$\nu^{\mathcal{K}}((0,T^{-1}])$"

    temporal = _temporal_component(times, eigenvalues, weights)
    mass = _cutoff_mass(times, eigenvalues, weights)
    total = float(np.sum(weights))
    if not np.isfinite(total) or total <= 0.0:
        raise RuntimeError(f"invalid component normalization for {name}")
    temporal /= total
    mass /= total

    base = COLORS[name]
    light = "#F7B58E" if name == "polynomial_rapid_target" else _mix_with_white(base, 0.47)
    axis.loglog(
        times, temporal, color=base, linewidth=3.5,
        label=main_label, zorder=3,
    )
    positive = mass > 0.0
    axis.loglog(
        times[positive], mass[positive], color=light,
        linestyle="-", linewidth=2.85, drawstyle="steps-post",
        label=mass_label, zorder=2,
    )
    theory = THEORY_GUIDES[component][name]
    layout = MANUAL_LAYOUT[component][name]
    if theory is not None and layout is not None:
        _guide_above(
            axis,
            times=times,
            anchor_values=temporal,
            comparison_values=(mass,),
            theory=theory,
            layout=layout,
        )
    _style_axis(axis, title=CONSTRUCTION_TITLES[name], ylabel="magnitude")
    axis.legend(frameon=False, loc="lower left")
    if component == "forcing" and name == "polynomial_rapid_target":
        axis.set_ylim(1e-16, 2.0)
        axis.set_yticks([1e0, 1e-4, 1e-8, 1e-12, 1e-16])
        axis.yaxis.set_minor_locator(NullLocator())

    return {
        "construction": name,
        "width": int(spec["width"]),
        "max_intrinsic_time": float(times[-1]),
    }


def make_loss_figure(
    data: dict[str, dict[str, Any]], output: Path,
) -> list[dict[str, Any]]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    _configure_style(plt)
    figure, axes = plt.subplots(
        2, 3, figsize=(15.8, 9.3), constrained_layout=True,
    )
    records: list[dict[str, Any]] = []
    for axis, name in zip(axes.flat, PANEL_ORDER, strict=True):
        records.append(_plot_loss_panel(axis, data, name))

    png = output / "loss_six_constructions.png"
    pdf = output / "loss_six_constructions.pdf"
    figure.savefig(png, dpi=300, bbox_inches="tight", facecolor="white")
    figure.savefig(pdf, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    return records


def make_component_figure(
    data: dict[str, dict[str, Any]],
    output: Path,
    *,
    component: str,
) -> list[dict[str, Any]]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    _configure_style(plt)
    figure, axes = plt.subplots(
        2, 3, figsize=(15.8, 9.3), constrained_layout=True,
    )
    specs = {spec["name"]: spec for spec in _forcing_specs(data)}
    records: list[dict[str, Any]] = []

    for axis, name in zip(axes.flat, PANEL_ORDER, strict=True):
        records.append(_plot_component_panel(
            axis, data, name, component=component, specs=specs,
        ))

    stem = f"{component}_six_constructions"
    png = output / f"{stem}.png"
    pdf = output / f"{stem}.pdf"
    figure.savefig(png, dpi=300, bbox_inches="tight", facecolor="white")
    figure.savefig(pdf, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    return records


def make_combined_figures(
    data: dict[str, dict[str, Any]],
    output: Path,
) -> dict[str, list[dict[str, Any]]]:
    """Split the six Loss/F/K construction rows across two 3x3 figures."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    _configure_style(plt)
    specs = {spec["name"]: spec for spec in _forcing_specs(data)}
    groups = {
        "part1": ALGEBRAIC_FORCING_ORDER,
        "part2": FASTER_THAN_POWER_FORCING_ORDER,
    }
    all_records: dict[str, list[dict[str, Any]]] = {}
    for part, names in groups.items():
        # Preserve the original per-panel geometry: the earlier 2x3 figures
        # used 4.65 inches per row.
        row_count = len(names)
        figure, axes = plt.subplots(
            row_count, 3,
            figsize=(15.8, 4.65 * row_count),
            constrained_layout=True,
        )
        records: list[dict[str, Any]] = []
        for row, name in enumerate(names):
            records.append({
                "construction": name,
                "loss": _plot_loss_panel(axes[row, 0], data, name),
                "forcing": _plot_component_panel(
                    axes[row, 1], data, name, component="forcing", specs=specs,
                ),
                "memory": _plot_component_panel(
                    axes[row, 2], data, name, component="memory", specs=specs,
                ),
            })

        stem = f"six_constructions_loss_forcing_memory_{part}"
        figure.savefig(
            output / f"{stem}.png", dpi=300,
            bbox_inches="tight", facecolor="white",
        )
        figure.savefig(
            output / f"{stem}.pdf",
            bbox_inches="tight", facecolor="white",
        )
        plt.close(figure)
        all_records[part] = records
    return all_records


def main() -> int:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    data = load_data()
    loss = make_loss_figure(data, OUTPUT)
    forcing = make_component_figure(data, OUTPUT, component="forcing")
    memory = make_component_figure(data, OUTPUT, component="memory")
    combined = make_combined_figures(data, OUTPUT)
    outputs = [
        OUTPUT / "loss_six_constructions.pdf",
        OUTPUT / "forcing_six_constructions.pdf",
        OUTPUT / "memory_six_constructions.pdf",
        OUTPUT / "six_constructions_loss_forcing_memory_part1.pdf",
        OUTPUT / "six_constructions_loss_forcing_memory_part2.pdf",
    ]
    write_json(OUTPUT / "summary.json", {
        "schema_version": "spectral_six_construction_paper_style_v002",
        "is_new_training_run": False,
        "data_selection": "preserves the legacy six-construction summary selections",
        "loss": "median literal sampled minibatch-SGD risk; no uncertainty band",
        "forcing_and_memory": "exact finite-spectrum component sums",
        "component_normalization": "F, K, and their cutoff masses are divided by their shared T=0 total spectral mass",
        "panel_order": list(PANEL_ORDER),
        "loss_panels": loss,
        "forcing_panels": forcing,
        "memory_panels": memory,
        "combined_panel_order": (
            list(ALGEBRAIC_FORCING_ORDER)
            + list(FASTER_THAN_POWER_FORCING_ORDER)
        ),
        "combined_panel_groups": [
            list(ALGEBRAIC_FORCING_ORDER),
            list(FASTER_THAN_POWER_FORCING_ORDER),
        ],
        "combined_layout": "one 4x3 figure followed by one 2x3 figure; columns are Loss, Forcing, Memory",
        "combined_group_meaning": {
            "part1": "algebraic forcing envelopes: pure power, logarithmically corrected, or log-periodically modulated",
            "part2": "forcing faster than every inverse power: a gapless memory-dominated tail versus spectrally gapped exponential relaxation",
        },
        "combined_panels": combined["part1"] + combined["part2"],
        "style_reference": str(
            ROOT / "artifacts" / "spectral_rapid_target_triptych_v009"
        ),
        "output_sha256": {
            str(path): sha256_file(path) for path in outputs
        },
    })
    print(json.dumps({"output": str(OUTPUT), "figure_count": 5}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
