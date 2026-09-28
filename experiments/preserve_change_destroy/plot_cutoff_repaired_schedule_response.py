"""Plot the reviewed LM/IM schedule-response experiment.

The exponent panels use the cutoff-repaired fixed-infinite-spectrum Volterra
quadrature.  The bridge panel is deliberately separate: it summarizes ten
same-feature comparisons between true Gaussian online SGD and the exact
conditional finite-W Volterra recursion, and is not used to fit exponents.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np


PACKAGE_ROOT = Path(__file__).resolve().parent
DEFAULT_INPUT = PACKAGE_ROOT / "artifacts" / "cutoff_repaired_schedule_response"
DEFAULT_OUTPUT = DEFAULT_INPUT

GAP_COLOR = "#0072B2"
TOTAL_COLOR = "#D55E00"
LM_COLOR = "#009E73"
IM_COLOR = "#7A5195"
GRAY = "#6F6F6F"


@dataclass(frozen=True)
class Regime:
    name: str
    title: str
    boundary: float
    preservation: float
    ceiling: float


REGIMES = {
    "LM": Regime("LM", "Long memory (LM)", 0.25, 0.75, 1.0),
    "IM": Regime("IM", "Integrable memory (IM)", 0.0, 0.5, 7.0 / 6.0),
}


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _float(row: dict[str, str], field: str) -> float:
    return float(row[field])


def _theory(regime: str, theta: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if regime == "LM":
        gap = np.where(theta <= 0.25, 0.0, np.minimum(theta - 0.25, 0.75))
    elif regime == "IM":
        gap = np.where(theta <= 0.0, 0.0, np.minimum(theta, 7.0 / 6.0))
    else:
        raise ValueError(f"unknown regime {regime}")
    return gap, np.minimum(0.5, gap)


def _shade_response_regions(axis: plt.Axes, regime: Regime, xmin: float, xmax: float) -> None:
    if xmin < regime.boundary:
        axis.axvspan(xmin, regime.boundary, color="#BDBDBD", alpha=0.12, lw=0)
    axis.axvspan(
        max(xmin, regime.boundary),
        min(xmax, regime.preservation),
        color=TOTAL_COLOR,
        alpha=0.055,
        lw=0,
    )
    if xmax > regime.preservation:
        axis.axvspan(
            max(xmin, regime.preservation), xmax, color=GAP_COLOR, alpha=0.045, lw=0
        )
    for location in (regime.boundary, regime.preservation, regime.ceiling):
        if xmin <= location <= xmax:
            axis.axvline(location, color="#8A8A8A", lw=0.7, ls=(0, (2, 2)), zorder=1)


def _plot_observable(
    axis: plt.Axes,
    rows: list[dict[str, str]],
    observable: str,
    color: str,
    marker: str,
) -> None:
    selected = [row for row in rows if row["observable"] == observable]
    # Plot every non-structural finite-window estimate with the same visual
    # encoding.  The machine-readable table retains the audit status, while
    # the compact paper panel does not introduce a separate inconclusive class.
    points = [
        row for row in selected if row["status"] in {"PASS", "INCONCLUSIVE"}
    ]
    if points:
        axis.scatter(
            [_float(row, "theta") for row in points],
            [_float(row, "fitted_exponent") for row in points],
            s=22,
            marker=marker,
            facecolor=color,
            edgecolor="white",
            linewidth=0.9,
            zorder=5,
        )
    censored = [row for row in selected if row["status"] == "CENSORED"]
    if censored:
        axis.scatter(
            [_float(row, "theta") for row in censored],
            [_float(row, "fitted_exponent") for row in censored],
            s=24,
            marker="x",
            color=GRAY,
            linewidth=1.0,
            zorder=7,
        )


def _plot_response_panel(
    axis: plt.Axes,
    all_rows: list[dict[str, str]],
    regime_name: str,
    panel: str,
) -> None:
    regime = REGIMES[regime_name]
    rows = [row for row in all_rows if row["regime"] == regime_name]
    theta_values = np.array([_float(row, "theta") for row in rows])
    xmin = float(np.min(theta_values)) - 0.03
    xmax = float(np.max(theta_values)) + 0.03
    _shade_response_regions(axis, regime, xmin, xmax)

    dense_theta = np.linspace(xmin, xmax, 800)
    gap, total = _theory(regime_name, dense_theta)
    axis.plot(dense_theta, gap, color=GAP_COLOR, lw=1.45, zorder=3)
    axis.plot(dense_theta, total, color=TOTAL_COLOR, lw=1.45, zorder=3)
    _plot_observable(axis, rows, "noise_gap", GAP_COLOR, "o")
    _plot_observable(axis, rows, "centered_total", TOTAL_COLOR, "s")

    axis.set_xlim(xmin, xmax)
    axis.set_ylim(-0.075, 1.25 if regime_name == "IM" else 0.82)
    axis.set_title(f"({panel}) {regime.title}", pad=2.5)
    axis.set_xlabel(r"schedule exponent $\vartheta$")
    axis.grid(True, which="major", color="#D8D8D8", lw=0.45, alpha=0.65)
    axis.set_axisbelow(True)


def _plot_bridge_panel(
    axis: plt.Axes,
    rows: list[dict[str, str]],
) -> None:
    for regime_name, marker, color in (
        ("LM", "o", LM_COLOR),
        ("IM", "^", IM_COLOR),
    ):
        selected = sorted(
            (row for row in rows if row["regime"] == regime_name),
            key=lambda row: _float(row, "theta"),
        )
        axis.plot(
            [_float(row, "theta") for row in selected],
            [100.0 * _float(row, "relative_l2_error") for row in selected],
            color=color,
            marker=marker,
            ms=4.0,
            lw=1.0,
            mec="white",
            mew=0.55,
            label=regime_name,
            zorder=4,
        )
    axis.axhspan(0.0, 5.0, color="#BDBDBD", alpha=0.10, lw=0)
    axis.axhline(5.0, color=GRAY, lw=0.85, ls=(0, (3, 2)), zorder=2)
    axis.text(
        0.985,
        0.89,
        "5% gate",
        ha="right",
        va="top",
        transform=axis.transAxes,
        color=GRAY,
        fontsize=6.6,
    )
    axis.set_xlim(-0.055, 1.23)
    axis.set_ylim(0.0, 5.45)
    axis.set_yticks((0.0, 2.5, 5.0))
    axis.set_title("(c) Finite-width SGD bridge", pad=2.5)
    axis.set_xlabel(r"schedule exponent $\vartheta$")
    axis.set_ylabel(r"relative $\ell_2$ error (\%)")
    axis.grid(True, which="major", color="#D8D8D8", lw=0.45, alpha=0.65)
    axis.set_axisbelow(True)
    axis.legend(loc="upper left", frameon=False, fontsize=6.8, handlelength=1.3)


def make_figure(input_dir: Path, output_dir: Path) -> tuple[Path, Path]:
    response_rows = _read_csv(input_dir / "layer_b_response_summary.csv")
    bridge_rows = _read_csv(input_dir / "layer_a_true_sgd_anchor_metrics.csv")

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["STIXGeneral", "Times New Roman", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "font.size": 7.2,
            "axes.titlesize": 7.7,
            "axes.labelsize": 7.4,
            "xtick.labelsize": 6.8,
            "ytick.labelsize": 6.8,
            "legend.fontsize": 6.8,
            "axes.linewidth": 0.7,
            "xtick.major.width": 0.65,
            "ytick.major.width": 0.65,
            "xtick.major.size": 2.6,
            "ytick.major.size": 2.6,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    figure, axes = plt.subplots(1, 3, figsize=(5.5, 2.12), constrained_layout=False)
    _plot_response_panel(axes[0], response_rows, "LM", "a")
    _plot_response_panel(axes[1], response_rows, "IM", "b")
    axes[0].set_ylabel(r"fitted decay exponent $\widehat q$")
    _plot_bridge_panel(axes[2], bridge_rows)

    legend_handles = [
        Line2D([], [], color=GAP_COLOR, marker="o", ms=4.0, lw=1.4, label="noisy--clean gap"),
        Line2D([], [], color=TOTAL_COLOR, marker="s", ms=3.8, lw=1.4, label="centered total"),
        Line2D([], [], color=GRAY, marker="x", ms=4.0, lw=0, label="structural boundary"),
    ]
    figure.legend(
        handles=legend_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.002),
        ncol=3,
        frameon=False,
        handlelength=1.6,
        columnspacing=1.15,
    )
    figure.subplots_adjust(left=0.067, right=0.995, top=0.94, bottom=0.245, wspace=0.34)

    output_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = output_dir / "schedule_response_finite_sgd_bridge.pdf"
    png_path = output_dir / "schedule_response_finite_sgd_bridge.png"
    figure.savefig(pdf_path, bbox_inches="tight", pad_inches=0.015)
    figure.savefig(png_path, dpi=320, bbox_inches="tight", pad_inches=0.015)
    plt.close(figure)
    return pdf_path, png_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    pdf_path, png_path = make_figure(args.input_dir, args.output_dir)
    print(pdf_path)
    print(png_path)


if __name__ == "__main__":
    main()
