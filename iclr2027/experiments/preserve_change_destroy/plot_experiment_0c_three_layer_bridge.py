"""Combine the two audited bridges in one same-object diagnostic figure.

The true-SGD and finite-W Volterra curves are averaged over the same frozen
feature realizations.  The deterministic-equivalent curve is read from
Experiment 0b at the same width, schedule, and discrete time grid.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any

import numpy as np


Array = np.ndarray
ROOT = Path(__file__).resolve().parent
ARTIFACT_ROOT = ROOT / "artifacts"
EXPERIMENT_0 = ARTIFACT_ROOT / "experiment_0_finite_rf_sgd_bridge"
EXPERIMENT_0B = ARTIFACT_ROOT / "experiment_0b_finite_w_de_bridge"
OUTPUT_DIR = ARTIFACT_ROOT / "experiment_0c_three_layer_bridge"
PAPER_FIGURE = ROOT.parents[1] / "figures" / "volterra_three_layer_bridge.pdf"
DISPLAY_WIDTH = 512
FEATURE_SEEDS = (11, 23)

MAXIMUM_TRUE_FINITE_RELATIVE_L2 = 0.04
MAXIMUM_FINITE_DE_RELATIVE_L2 = 0.04
MAXIMUM_TRUE_DE_RELATIVE_L2 = 0.05


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def _relative_l2(candidate: Array, reference: Array) -> float:
    denominator = float(np.linalg.norm(reference))
    if denominator <= 0.0:
        raise ValueError("relative-L2 reference must be nonzero")
    return float(np.linalg.norm(candidate - reference) / denominator)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("cannot write an empty CSV")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _case_key(row: dict[str, str]) -> tuple[str, float, str]:
    return row["regime"], float(row["theta"]), row["response"]


def _sorted_case(
    rows: list[dict[str, str]], key: tuple[str, float, str]
) -> list[dict[str, str]]:
    selected = [row for row in rows if _case_key(row) == key]
    selected.sort(key=lambda row: int(row["iteration"]))
    return selected


def _same_w_mean(
    rows: list[dict[str, str]],
    key: tuple[str, float, str],
    field: str,
) -> Array:
    trajectories = []
    for seed in FEATURE_SEEDS:
        selected = [
            row
            for row in rows
            if _case_key(row) == key and int(row["feature_seed"]) == seed
        ]
        selected.sort(key=lambda row: int(row["iteration"]))
        if not selected:
            raise ValueError(f"missing seed {seed} for {key}")
        trajectories.append([float(row[field]) for row in selected])
    return np.mean(np.asarray(trajectories, dtype=float), axis=0)


def _same_w_monte_carlo_se(
    rows: list[dict[str, str]],
    key: tuple[str, float, str],
    field: str,
) -> Array:
    standard_errors = []
    for seed in FEATURE_SEEDS:
        selected = [
            row
            for row in rows
            if _case_key(row) == key and int(row["feature_seed"]) == seed
        ]
        selected.sort(key=lambda row: int(row["iteration"]))
        standard_errors.append([float(row[field]) for row in selected])
    values = np.asarray(standard_errors, dtype=float)
    return np.sqrt(np.sum(values * values, axis=0)) / len(FEATURE_SEEDS)


def merge_layers() -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    path_0 = EXPERIMENT_0 / "curves.csv"
    path_0b = EXPERIMENT_0B / "curves.csv"
    rows_0 = [
        row
        for row in _read_csv(path_0)
        if int(row["width"]) == DISPLAY_WIDTH
        and int(row["feature_seed"]) in FEATURE_SEEDS
    ]
    rows_0b = [
        row for row in _read_csv(path_0b) if int(row["width"]) == DISPLAY_WIDTH
    ]
    summary_0 = _read_json(EXPERIMENT_0 / "summary.json")
    summary_0b = _read_json(EXPERIMENT_0B / "summary.json")

    parameters_0 = summary_0["parameters"]
    parameters_0b = summary_0b["same_object_contract"]
    contract_checks = {
        "ambient_to_width_ratio": (
            int(parameters_0["ambient_to_width_ratio"])
            == int(parameters_0b["ambient_to_width_ratio"])
        ),
        "eta": math.isclose(
            float(parameters_0["eta"]), float(parameters_0b["eta"]), abs_tol=0.0
        ),
        "initial_batch": (
            int(parameters_0["initial_batch"])
            == int(parameters_0b["initial_batch"])
        ),
        "sigma2": math.isclose(
            float(parameters_0["sigma2"]),
            float(parameters_0b["sigma2"]),
            abs_tol=0.0,
        ),
        "width": (
            DISPLAY_WIDTH in parameters_0["widths"]
            and DISPLAY_WIDTH in parameters_0b["widths"]
        ),
        "source_bridge_pass": bool(summary_0["gates"]["all_cases_pass"]),
        "de_bridge_pass": bool(summary_0b["gates"]["bridge_gate_pass"]),
    }
    if not all(contract_checks.values()):
        raise RuntimeError(f"source experiment contract mismatch: {contract_checks}")

    keys_0 = {_case_key(row) for row in rows_0}
    keys_0b = {_case_key(row) for row in rows_0b}
    if keys_0 != keys_0b or len(keys_0) != 6:
        raise RuntimeError("the source experiments do not contain the same six cases")

    curve_rows: list[dict[str, Any]] = []
    metric_rows: list[dict[str, Any]] = []
    for key in sorted(keys_0):
        regime, theta, response = key
        de_rows = _sorted_case(rows_0b, key)
        seed_reference = [
            row
            for row in rows_0
            if _case_key(row) == key
            and int(row["feature_seed"]) == FEATURE_SEEDS[0]
        ]
        seed_reference.sort(key=lambda row: int(row["iteration"]))
        if len(seed_reference) != len(de_rows):
            raise RuntimeError(f"time-grid length mismatch for {key}")

        iterations_0 = [int(row["iteration"]) for row in seed_reference]
        iterations_0b = [int(row["iteration"]) for row in de_rows]
        times_0 = np.asarray(
            [float(row["intrinsic_time"]) for row in seed_reference]
        )
        times_0b = np.asarray([float(row["intrinsic_time"]) for row in de_rows])
        batches_0 = [row["batch"] for row in seed_reference]
        batches_0b = [row["batch"] for row in de_rows]
        if (
            iterations_0 != iterations_0b
            or not np.array_equal(times_0, times_0b)
            or batches_0 != batches_0b
        ):
            raise RuntimeError(f"discrete schedule mismatch for {key}")

        finite_total = _same_w_mean(
            rows_0, key, "volterra_noisy_centered"
        )
        true_total = _same_w_mean(rows_0, key, "true_sgd_noisy_centered")
        true_total_se = _same_w_monte_carlo_se(
            rows_0, key, "true_sgd_noisy_standard_error"
        )
        finite_gap = _same_w_mean(rows_0, key, "volterra_noise_gap")
        true_gap = _same_w_mean(rows_0, key, "true_sgd_noise_gap")
        true_gap_se = _same_w_monte_carlo_se(
            rows_0, key, "true_sgd_noise_gap_standard_error"
        )
        de_total = np.asarray(
            [float(row["de_noisy_centered"]) for row in de_rows]
        )
        de_gap = np.asarray([float(row["de_noise_gap"]) for row in de_rows])

        for index, intrinsic_time in enumerate(times_0):
            curve_rows.append(
                {
                    "regime": regime,
                    "theta": theta,
                    "response": response,
                    "width": DISPLAY_WIDTH,
                    "iteration": iterations_0[index],
                    "intrinsic_time": float(intrinsic_time),
                    "batch": batches_0[index],
                    "true_sgd_noisy_centered": float(true_total[index]),
                    "true_sgd_noisy_centered_se": float(true_total_se[index]),
                    "finite_w_noisy_centered": float(finite_total[index]),
                    "de_noisy_centered": float(de_total[index]),
                    "true_sgd_noise_gap": float(true_gap[index]),
                    "true_sgd_noise_gap_se": float(true_gap_se[index]),
                    "finite_w_noise_gap": float(finite_gap[index]),
                    "de_noise_gap": float(de_gap[index]),
                }
            )

        metric_rows.append(
            {
                "regime": regime,
                "theta": theta,
                "response": response,
                "width": DISPLAY_WIDTH,
                "feature_seeds": ";".join(str(seed) for seed in FEATURE_SEEDS),
                "true_finite_total_relative_l2": _relative_l2(
                    true_total, finite_total
                ),
                "finite_de_total_relative_l2": _relative_l2(
                    finite_total, de_total
                ),
                "true_de_total_relative_l2": _relative_l2(true_total, de_total),
                "true_finite_gap_relative_l2": _relative_l2(true_gap, finite_gap),
                "finite_de_gap_relative_l2": _relative_l2(finite_gap, de_gap),
                "true_de_gap_relative_l2": _relative_l2(true_gap, de_gap),
            }
        )

    audit = {
        "contract_checks": contract_checks,
        "experiment_0_curves_sha256": _sha256(path_0),
        "experiment_0b_curves_sha256": _sha256(path_0b),
    }
    return curve_rows, metric_rows, audit


def _marker_indices(size: int, count: int = 11) -> Array:
    if size <= 1:
        return np.asarray([0], dtype=int)
    indices = np.unique(
        np.rint(np.geomspace(1.0, float(size - 1), count)).astype(int)
    )
    return indices


def plot(curve_rows: list[dict[str, Any]]) -> tuple[Path, Path, Path]:
    os.environ.setdefault("MPLCONFIGDIR", str(OUTPUT_DIR / ".matplotlib"))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.ticker import FixedFormatter, FixedLocator, NullFormatter

    plt.rcParams.update(
        {
            "font.family": "STIXGeneral",
            "mathtext.fontset": "stix",
            "font.size": 6.8,
            "axes.titlesize": 7.1,
            "axes.labelsize": 7.0,
            "legend.fontsize": 6.7,
            "xtick.labelsize": 6.5,
            "ytick.labelsize": 6.5,
            "axes.linewidth": 0.70,
            "lines.solid_capstyle": "round",
            "lines.dash_capstyle": "round",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    # Paul Tol's colorblind-safe bright palette.  Layer identity is encoded
    # independently by solid/dashed lines and open-circle observations.
    colors = {
        "boundary": "#AA3377",
        "changed": "#EE7733",
        "preserved": "#4477AA",
    }
    panels = (
        ("LM", "total", "(a) LM: centered risk"),
        ("IM", "total", "(b) IM: centered risk"),
        ("LM", "gap", "(c) LM: noisy–clean gap"),
        ("IM", "gap", "(d) IM: noisy–clean gap"),
    )
    # The bundled ICLR template fixes \textwidth=5.5in.  Generate at that
    # physical width so LaTeX need not rescale the typography.  The two pairs
    # share y axes; the narrow central spacer prevents the second y label from
    # colliding with the first pair in this deliberately compact 1x4 layout.
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
    total_lm_axis = figure.add_subplot(grid[0, 0])
    total_im_axis = figure.add_subplot(
        grid[0, 1], sharex=total_lm_axis, sharey=total_lm_axis
    )
    gap_lm_axis = figure.add_subplot(grid[0, 3], sharex=total_lm_axis)
    gap_im_axis = figure.add_subplot(
        grid[0, 4], sharex=total_lm_axis, sharey=gap_lm_axis
    )
    axes = [total_lm_axis, total_im_axis, gap_lm_axis, gap_im_axis]
    for axis, (regime, observable, title) in zip(axes, panels, strict=True):
        responses = ["boundary", "changed", "preserved"]
        for response in responses:
            selected = [
                row
                for row in curve_rows
                if row["regime"] == regime and row["response"] == response
            ]
            selected.sort(key=lambda row: int(row["iteration"]))
            times = np.asarray([row["intrinsic_time"] for row in selected])
            if observable == "total":
                true = np.asarray(
                    [row["true_sgd_noisy_centered"] for row in selected]
                )
                true_se = np.asarray(
                    [row["true_sgd_noisy_centered_se"] for row in selected]
                )
                finite = np.asarray(
                    [row["finite_w_noisy_centered"] for row in selected]
                )
                de = np.asarray([row["de_noisy_centered"] for row in selected])
            else:
                true = np.asarray([row["true_sgd_noise_gap"] for row in selected])
                true_se = np.asarray(
                    [row["true_sgd_noise_gap_se"] for row in selected]
                )
                finite = np.asarray([row["finite_w_noise_gap"] for row in selected])
                de = np.asarray([row["de_noise_gap"] for row in selected])
            positive = (times > 0.0) & (true > 0.0) & (finite > 0.0) & (de > 0.0)
            plotted_times = times[positive]
            plotted_true = true[positive]
            plotted_se = true_se[positive]
            plotted_finite = finite[positive]
            plotted_de = de[positive]
            color = colors[response]
            # Canonical three-layer encoding: observations are unconnected
            # markers, finite-W is a light solid line, and the limiting DE is a
            # saturated long-dashed line.  Drawing the dashed curve last makes
            # both layers visible under near-exact numerical overlap: dark
            # dashes identify DE and the light solid gaps identify finite-W.
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
    # One labeled reference decade is enough for each pair of compact panels.
    # Retain the unlabeled minor ticks/grid so positions within the decade are
    # still visually recoverable without crowding the y axes.
    axes[0].yaxis.set_major_locator(FixedLocator([1.0]))
    axes[0].yaxis.set_major_formatter(FixedFormatter([r"$10^0$"]))
    axes[2].yaxis.set_major_locator(FixedLocator([0.1]))
    axes[2].yaxis.set_major_formatter(FixedFormatter([r"$10^{-1}$"]))
    for axis in axes:
        axis.yaxis.set_minor_formatter(NullFormatter())
    # The right panel in each observable pair repeats exactly the same y scale.
    # Suppress its complete y-axis artist to prevent log tick labels from
    # spilling into the neighboring one-inch-wide panel.
    axes[1].yaxis.set_visible(False)
    axes[3].yaxis.set_visible(False)
    response_handles = [
        Line2D([0], [0], color=colors[name], linewidth=1.75, label=name)
        for name in ("boundary", "changed", "preserved")
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
    # Matplotlib fills multi-row legends column-first.  Interleave the handles
    # so the first row contains the three estimators and the second the three
    # schedule responses.
    legend_handles = [
        layer_handles[0],
        response_handles[0],
        layer_handles[1],
        response_handles[1],
        layer_handles[2],
        response_handles[2],
    ]
    figure.legend(
        handles=legend_handles,
        frameon=False,
        ncol=3,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.025),
        borderaxespad=0.0,
        handlelength=2.05,
        handletextpad=0.40,
        columnspacing=1.30,
        labelspacing=0.32,
    )
    figure.supxlabel(
        r"intrinsic time $T_t$",
        fontsize=7.0,
        y=0.270,
    )
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    png = OUTPUT_DIR / "experiment0c_three_layer_bridge_paper.png"
    pdf = OUTPUT_DIR / "experiment0c_three_layer_bridge_paper.pdf"
    figure.savefig(png, dpi=400)
    figure.savefig(pdf)
    PAPER_FIGURE.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(PAPER_FIGURE)
    plt.close(figure)
    return png, pdf, PAPER_FIGURE


def run() -> dict[str, Any]:
    curve_rows, metric_rows, audit = merge_layers()
    _write_csv(OUTPUT_DIR / "curves.csv", curve_rows)
    _write_csv(OUTPUT_DIR / "metrics.csv", metric_rows)
    png, pdf, paper_pdf = plot(curve_rows)

    maximum_true_finite = max(
        max(
            float(row["true_finite_total_relative_l2"]),
            float(row["true_finite_gap_relative_l2"]),
        )
        for row in metric_rows
    )
    maximum_finite_de = max(
        max(
            float(row["finite_de_total_relative_l2"]),
            float(row["finite_de_gap_relative_l2"]),
        )
        for row in metric_rows
    )
    maximum_true_de = max(
        max(
            float(row["true_de_total_relative_l2"]),
            float(row["true_de_gap_relative_l2"]),
        )
        for row in metric_rows
    )
    gate_pass = bool(
        all(audit["contract_checks"].values())
        and maximum_true_finite <= MAXIMUM_TRUE_FINITE_RELATIVE_L2
        and maximum_finite_de <= MAXIMUM_FINITE_DE_RELATIVE_L2
        and maximum_true_de <= MAXIMUM_TRUE_DE_RELATIVE_L2
    )
    summary = {
        "experiment": "experiment_0c_three_layer_bridge",
        "paper_modified": False,
        "workflow_promoted": False,
        "purpose": (
            "place true-SGD mean, exact same-W conditional Volterra, and the "
            "resolvent deterministic-equivalent Volterra in one audited figure"
        ),
        "same_object_contract": {
            "width": DISPLAY_WIDTH,
            "ambient_to_width_ratio": 2,
            "feature_seeds_for_true_and_finite_w": list(FEATURE_SEEDS),
            "true_and_finite_aggregation": "arithmetic mean over identical W",
            "sigma2": 25.0,
            "schedule": "B_t=ceil(16*(1+eta*t)^theta)",
            "time": "T_t=eta*t",
            **audit,
        },
        "gates": {
            "maximum_true_finite_relative_l2": MAXIMUM_TRUE_FINITE_RELATIVE_L2,
            "maximum_finite_de_relative_l2": MAXIMUM_FINITE_DE_RELATIVE_L2,
            "maximum_true_de_relative_l2": MAXIMUM_TRUE_DE_RELATIVE_L2,
            "three_layer_gate_pass": gate_pass,
        },
        "aggregate_results": {
            "maximum_true_finite_relative_l2": maximum_true_finite,
            "maximum_finite_de_relative_l2": maximum_finite_de,
            "maximum_true_de_relative_l2": maximum_true_de,
        },
        "claim_scope": (
            "finite-width numerical three-layer bridge on the audited schedules "
            "and time windows; not a uniform random-matrix theorem"
        ),
        "artifacts": {
            "curves": str(OUTPUT_DIR / "curves.csv"),
            "metrics": str(OUTPUT_DIR / "metrics.csv"),
            "png": str(png),
            "pdf": str(pdf),
            "paper_pdf": str(paper_pdf),
        },
    }
    _write_json(OUTPUT_DIR / "summary.json", summary)
    return summary


if __name__ == "__main__":
    print(json.dumps(run(), indent=2, sort_keys=True))
