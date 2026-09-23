#!/usr/bin/env python3
"""Plot every accepted support-adaptive DE Volterra width in seven phases."""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.cm import ScalarMappable
from matplotlib.colors import LogNorm


WORKSTREAM = Path(__file__).resolve().parents[1]
RESULTS = WORKSTREAM / "artifacts" / "results"
PLOTS = WORKSTREAM / "artifacts" / "plots"

INPUTS = (
    RESULTS / "adaptive_real_axis_d10_to_d40_curves.csv",
    RESULTS / "adaptive_real_axis_d50_to_d150_curves.csv",
    RESULTS / "adaptive_real_axis_d200_to_d3200_curves.csv",
    RESULTS / "adaptive_real_axis_d4800_d6400_d9600_curves.csv",
    RESULTS / "adaptive_real_axis_d12800_curves.csv",
    RESULTS / "adaptive_real_axis_d19200_d38400_curves.csv",
    RESULTS / "adaptive_real_axis_d25600_d51200_curves.csv",
)
INPUT_MANIFESTS = (
    RESULTS / "adaptive_real_axis_d10_to_d40_manifest.json",
    RESULTS / "adaptive_real_axis_d50_to_d150_manifest.json",
    RESULTS / "adaptive_real_axis_d200_to_d3200_manifest.json",
    RESULTS / "adaptive_real_axis_d4800_d6400_d9600_manifest.json",
    RESULTS / "adaptive_real_axis_d12800_manifest.json",
    RESULTS / "adaptive_real_axis_d19200_d38400_manifest.json",
    RESULTS / "adaptive_real_axis_d25600_d51200_manifest.json",
)
OUTPUT_MANIFEST = RESULTS / "full_adaptive_volterra_all_d_plot_manifest.json"

PHASES = ("Ia", "Ib", "Ic", "II", "III", "IVa", "IVb")
PHASE_POINTS = {
    "Ia": (0.7, 0.3),
    "Ib": (0.27, 0.4),
    "Ic": (0.12, 0.65),
    "II": (0.7, 0.6),
    "III": (0.6, 0.7),
    "IVa": (0.3, 0.7),
    "IVb": (0.26, 0.7),
}
D_VALUES = (
    10,
    15,
    20,
    30,
    40,
    50,
    75,
    100,
    150,
    200,
    300,
    400,
    600,
    800,
    1_200,
    1_600,
    2_400,
    3_200,
    4_800,
    6_400,
    9_600,
    12_800,
    19_200,
    25_600,
    38_400,
    51_200,
)
SIGMA2_VALUES = (0.0, 0.1, 1.0, 10.0)
OUTPUTS = {
    0.0: PLOTS / "full_adaptive_volterra_all_d_centered_sigma2_0.png",
    0.1: PLOTS / "full_adaptive_volterra_all_d_centered_sigma2_0p1.png",
    1.0: PLOTS / "full_adaptive_volterra_all_d_centered_sigma2_1.png",
    10.0: PLOTS / "full_adaptive_volterra_all_d_centered_sigma2_10.png",
}


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def validate_inputs() -> None:
    for path in INPUTS + INPUT_MANIFESTS:
        if not path.is_file():
            raise FileNotFoundError(path)
    for path in INPUT_MANIFESTS:
        with path.open(encoding="utf-8") as handle:
            manifest = json.load(handle)
        if manifest.get("status") != "COMPLETE":
            raise RuntimeError(f"input calculation is not complete: {path}")


def prepare_curves() -> dict[tuple[str, float, int], dict[str, np.ndarray]]:
    rows: list[dict[str, str]] = []
    for path in INPUTS:
        rows.extend(read_rows(path))

    grouped: dict[
        tuple[str, float, int],
        list[tuple[float, float, float]],
    ] = defaultdict(list)
    for row in rows:
        sigma2 = float(row["sigma2"])
        if sigma2 not in SIGMA2_VALUES:
            continue
        key = (row["phase"], sigma2, int(row["d"]))
        grouped[key].append(
            (
                float(row["flops"]),
                float(row["centered_loss"]),
                float(row["centering_plateau"]),
            )
        )

    expected = {
        (phase, sigma2, d)
        for phase in PHASES
        for sigma2 in SIGMA2_VALUES
        for d in D_VALUES
    }
    if set(grouped) != expected:
        missing = sorted(expected - set(grouped))[:10]
        extra = sorted(set(grouped) - expected)[:10]
        raise RuntimeError(f"incomplete curve grid: missing={missing}, extra={extra}")

    curves: dict[tuple[str, float, int], dict[str, np.ndarray]] = {}
    for key, values in grouped.items():
        values.sort(key=lambda value: value[0])
        flops = np.asarray([value[0] for value in values], dtype=float)
        centered = np.asarray([value[1] for value in values], dtype=float)
        plateaus = np.asarray([value[2] for value in values], dtype=float)
        if len(flops) != 720:
            raise RuntimeError(f"expected 720 FLOPs points at {key}, found {len(flops)}")
        if not np.all(np.diff(flops) > 0.0):
            raise RuntimeError(f"non-increasing FLOPs grid at {key}")
        if np.any(~np.isfinite(centered)) or np.any(centered <= 0.0):
            raise RuntimeError(f"invalid centered loss at {key}")
        if not np.allclose(plateaus, plateaus[0], rtol=0.0, atol=1.0e-14):
            raise RuntimeError(f"centering plateau varies along curve at {key}")
        curves[key] = {
            "flops": flops,
            "centered_loss": centered,
            "centering_plateau": plateaus[:1],
        }
    return curves


def sigma_label(sigma2: float) -> str:
    return f"{sigma2:g}"


def plot_sigma(
    curves: dict[tuple[str, float, int], dict[str, np.ndarray]],
    sigma2: float,
    output: Path,
) -> None:
    norm = LogNorm(vmin=min(D_VALUES), vmax=max(D_VALUES))
    cmap = plt.get_cmap("viridis")
    figure, axes = plt.subplots(
        2,
        4,
        figsize=(20.5, 10.4),
        sharex=True,
        constrained_layout=False,
    )
    flat_axes = axes.ravel()

    for phase_index, phase in enumerate(PHASES):
        axis = flat_axes[phase_index]
        for d in reversed(D_VALUES):
            curve = curves[(phase, sigma2, d)]
            axis.plot(
                curve["flops"],
                curve["centered_loss"],
                color=cmap(norm(d)),
                linewidth=1.15,
                alpha=0.88,
                solid_capstyle="round",
                rasterized=True,
            )
        alpha, beta = PHASE_POINTS[phase]
        axis.set_title(
            rf"{phase}: $\alpha={alpha:g},\ \beta={beta:g}$",
            fontsize=12.5,
        )
        axis.set_xscale("log")
        axis.set_yscale("log")
        axis.set_xlim(1.0e2, 1.0e12)
        axis.grid(True, which="major", linewidth=0.55, alpha=0.34)
        axis.grid(True, which="minor", linewidth=0.35, alpha=0.16)
        axis.tick_params(axis="both", which="both", labelsize=9.5)
        if phase_index in (0, 4):
            axis.set_ylabel(
                r"$R_\sigma(r,d)-R_{\sigma,\infty}^{\star}$",
                fontsize=11,
            )
        if phase_index >= 4:
            axis.set_xlabel(r"FLOPs $f=rBd$", fontsize=11)

    flat_axes[-1].axis("off")
    figure.suptitle(
        "Latest support-adaptive zero-regularization DE Volterra curves"
        + rf": $\sigma^2={sigma_label(sigma2)}$",
        fontsize=16,
        y=0.985,
    )
    colorbar_axis = figure.add_axes((0.57, 0.116, 0.36, 0.026))
    colorbar = figure.colorbar(
        ScalarMappable(norm=norm, cmap=cmap),
        cax=colorbar_axis,
        orientation="horizontal",
        ticks=D_VALUES,
    )
    colorbar.ax.set_xticklabels(
        [f"{d:,}" for d in D_VALUES],
        rotation=42,
        ha="right",
        fontsize=8.3,
    )
    colorbar.set_label(r"width $d$", fontsize=11, labelpad=2)

    figure.subplots_adjust(
        left=0.064,
        right=0.985,
        top=0.925,
        bottom=0.18,
        wspace=0.25,
        hspace=0.25,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(figure)


def run() -> dict[str, Any]:
    validate_inputs()
    curves = prepare_curves()
    for sigma2, output in OUTPUTS.items():
        plot_sigma(curves, sigma2, output)

    manifest: dict[str, Any] = {
        "status": "COMPLETE",
        "experiment": "full_adaptive_volterra_all_d_centered_plots",
        "solver": "support-adaptive zero-regularization deterministic-equivalent Volterra",
        "risk": "R_sigma(r,d) - R_sigma,infinity^star",
        "phases": list(PHASES),
        "phase_points": {
            phase: {"alpha": point[0], "beta": point[1]}
            for phase, point in PHASE_POINTS.items()
        },
        "d_values": list(D_VALUES),
        "sigma2_values": list(SIGMA2_VALUES),
        "flops_min": 1.0e2,
        "flops_max": 1.0e12,
        "points_per_curve": 720,
        "curve_count": len(curves),
        "source_curves": [str(path.relative_to(WORKSTREAM)) for path in INPUTS],
        "source_manifests": [
            str(path.relative_to(WORKSTREAM)) for path in INPUT_MANIFESTS
        ],
        "plots": {
            sigma_label(sigma2): str(path.relative_to(WORKSTREAM))
            for sigma2, path in OUTPUTS.items()
        },
    }
    with OUTPUT_MANIFEST.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return manifest


if __name__ == "__main__":
    result = run()
    print(json.dumps(result, indent=2, sort_keys=True))
