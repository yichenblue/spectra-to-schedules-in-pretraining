"""Run experiment one at m=10,000 with the matched authenticity setting."""

from __future__ import annotations

import csv
import json
import math
import os
from pathlib import Path
from typing import Any

import numpy as np

from .bridge_config import BRIDGE_EXPERIMENT_ONE_M10000_PROFILE
from .bridge_validation import analytic_power_law_spectrum, fit_effective_exponent
from .run_bridge import run_profile
from .slopes import local_log_slopes


ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "artifacts" / "experiment_one_m10000"
BRIDGE_DIR = ROOT / "artifacts" / "bridge_experiment_one_m10000"


def _predicted_total_exponent(theta: float) -> float:
    q_clean = 0.5
    q_kernel = 0.75
    destroy_boundary = 1.0 - q_kernel
    if theta <= destroy_boundary:
        noise = 0.0
    elif theta < 1.0:
        noise = theta + q_kernel - 1.0
    else:
        noise = q_kernel
    return min(q_clean, noise)


def _phase(theta: float) -> str:
    predicted = _predicted_total_exponent(theta)
    if theta <= 0.25:
        return "destroy"
    if predicted < 0.5 - 1.0e-12:
        return "change"
    if abs(theta - 0.75) <= 1.0e-12:
        return "critical"
    return "preserve"


def _load_curves(path: Path) -> dict[float, dict[str, np.ndarray]]:
    grouped: dict[float, list[dict[str, str]]] = {}
    with path.open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            grouped.setdefault(float(row["theta"]), []).append(row)
    fields = (
        "intrinsic_time",
        "exact_total",
        "continuum_total",
        "true_sgd_total",
        "true_sgd_total_se",
    )
    output: dict[float, dict[str, np.ndarray]] = {}
    for theta, rows in grouped.items():
        rows.sort(key=lambda row: int(row["step"]))
        output[theta] = {
            field: np.asarray([float(row[field]) for row in rows], dtype=float)
            for field in fields
        }
    return output


def _local_variation(
    times: np.ndarray,
    values: np.ndarray,
    lower: float,
    upper: float,
) -> tuple[float, float]:
    valid = (
        (times >= lower)
        & (times <= upper)
        & (times > 0.0)
        & (values > 0.0)
    )
    slopes = local_log_slopes(
        times,
        values,
        valid=valid,
        half_window_decades=0.15,
        minimum_points=7,
    )
    selected = slopes[valid & np.isfinite(slopes)]
    if selected.size < 3:
        return float("nan"), float("inf")
    median = float(np.median(selected))
    variation = float(
        (np.quantile(selected, 0.9) - np.quantile(selected, 0.1))
        / max(abs(median), 0.05)
    )
    return median, variation


def _slope_rows(
    curves: dict[float, dict[str, np.ndarray]],
    floor: float,
) -> list[dict[str, Any]]:
    profile = BRIDGE_EXPERIMENT_ONE_M10000_PROFILE
    width = profile.widths[0]
    lower = float(width) ** profile.fit_lower_exponent
    upper = float(width) ** profile.fit_upper_exponent
    maximum_error = 0.05
    maximum_variation = 0.15
    rows: list[dict[str, Any]] = []
    for theta in profile.theta_values:
        data = curves[theta]
        times = data["intrinsic_time"]
        predicted = _predicted_total_exponent(theta)
        for route, field in (
            ("discrete_exact", "exact_total"),
            ("continuum", "continuum_total"),
            ("true_sgd", "true_sgd_total"),
        ):
            centered = np.maximum(data[field] - floor, np.finfo(float).tiny)
            exponent = fit_effective_exponent(times, centered, lower, upper)
            local_median, local_variation = _local_variation(
                times, centered, lower, upper
            )
            exponent_error = abs(exponent - predicted)
            passed = bool(
                np.isfinite(exponent)
                and exponent_error <= maximum_error
                and local_variation <= maximum_variation
            )
            rows.append(
                {
                    "width": width,
                    "theta": theta,
                    "phase": _phase(theta),
                    "route": route,
                    "observable": "centered_noisy_risk",
                    "fit_lower_time": lower,
                    "fit_upper_time": upper,
                    "predicted_exponent": predicted,
                    "measured_exponent": exponent,
                    "absolute_exponent_error": exponent_error,
                    "local_slope_median": local_median,
                    "local_slope_variation": local_variation,
                    "maximum_exponent_error": maximum_error,
                    "maximum_local_slope_variation": maximum_variation,
                    "theory_status": "PASS" if passed else "INCONCLUSIVE",
                }
            )
    return rows


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _make_figure(
    curves: dict[float, dict[str, np.ndarray]],
    slope_rows: list[dict[str, Any]],
    floor: float,
    output_path: Path,
) -> tuple[Path, Path]:
    os.environ.setdefault("MPLCONFIGDIR", str(output_path.parent / ".matplotlib"))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    profile = BRIDGE_EXPERIMENT_ONE_M10000_PROFILE
    width = profile.widths[0]
    lower = float(width) ** profile.fit_lower_exponent
    upper = float(width) ** profile.fit_upper_exponent
    representatives = (0.25, 0.50, 0.90)
    colors = {0.25: "#c43c39", 0.50: "#d18700", 0.90: "#2878b5"}
    figure, axes = plt.subplots(1, 3, figsize=(13.8, 3.9))

    axis = axes[0]
    for theta in representatives:
        data = curves[theta]
        times = data["intrinsic_time"]
        valid = times > 0.0
        exact = np.maximum(data["exact_total"] - floor, np.finfo(float).tiny)
        continuum = np.maximum(
            data["continuum_total"] - floor, np.finfo(float).tiny
        )
        sampled = np.maximum(
            data["true_sgd_total"] - floor, np.finfo(float).tiny
        )
        axis.loglog(
            times[valid], exact[valid], color=colors[theta], linewidth=2.0,
            label=rf"$\vartheta={theta:g}$",
        )
        axis.loglog(
            times[valid], continuum[valid], color=colors[theta],
            linewidth=1.2, linestyle="--",
        )
        axis.loglog(
            times[valid], sampled[valid], color=colors[theta],
            linewidth=0.9, alpha=0.65,
        )
    axis.axvspan(lower, upper, color="#d18700", alpha=0.07)
    axis.set_xlabel(r"intrinsic time $T$")
    axis.set_ylabel("centered noisy risk")
    axis.set_title(r"(a) Matched $m=10^4$ trajectories")
    axis.grid(True, which="both", alpha=0.17)
    theta_legend = axis.legend(frameon=False, fontsize=8, loc="lower left")
    axis.add_artist(theta_legend)
    axis.legend(
        handles=[
            Line2D([0], [0], color="black", linewidth=2.0, label="discrete exact"),
            Line2D([0], [0], color="black", linestyle="--", label="continuum"),
            Line2D([0], [0], color="black", linewidth=0.9, alpha=0.65,
                   label="true-SGD mean"),
        ],
        frameon=False,
        fontsize=7,
        loc="upper right",
    )

    axis = axes[1]
    for theta in representatives:
        data = curves[theta]
        times = data["intrinsic_time"]
        valid = (times >= lower) & (times <= upper)
        for field, linestyle, alpha_value in (
            ("exact_total", "-", 1.0),
            ("continuum_total", "--", 0.9),
            ("true_sgd_total", ":", 0.65),
        ):
            centered = np.maximum(data[field] - floor, np.finfo(float).tiny)
            slopes = local_log_slopes(
                times,
                centered,
                valid=valid,
                half_window_decades=0.15,
                minimum_points=7,
            )
            finite = valid & np.isfinite(slopes)
            axis.semilogx(
                times[finite], slopes[finite], color=colors[theta],
                linestyle=linestyle, linewidth=1.5, alpha=alpha_value,
            )
        axis.axhline(
            _predicted_total_exponent(theta), color=colors[theta],
            linewidth=0.9, linestyle=(0, (2, 2)),
        )
    axis.set_xlabel(r"intrinsic time $T$")
    axis.set_ylabel("local effective exponent")
    axis.set_title("(b) Fixed-window local slopes")
    axis.grid(True, which="both", alpha=0.17)

    axis = axes[2]
    theta_grid = np.linspace(min(profile.theta_values), max(profile.theta_values), 400)
    axis.plot(
        theta_grid,
        [_predicted_total_exponent(value) for value in theta_grid],
        color="black",
        linewidth=2.0,
        label="joint-limit theory",
    )
    markers = {"discrete_exact": "o", "continuum": "^", "true_sgd": "x"}
    route_labels = {
        "discrete_exact": "discrete exact",
        "continuum": "continuum",
        "true_sgd": "true-SGD mean",
    }
    for route in markers:
        selected = [row for row in slope_rows if row["route"] == route]
        axis.scatter(
            [float(row["theta"]) for row in selected],
            [float(row["measured_exponent"]) for row in selected],
            marker=markers[route],
            s=40,
            linewidths=1.1,
            label=route_labels[route],
            zorder=3,
        )
    axis.axvline(0.25, color="#c43c39", linestyle=":", linewidth=1.0)
    axis.axvline(0.75, color="#2878b5", linestyle=":", linewidth=1.0)
    axis.set_xlabel(r"ratio-growth exponent $\vartheta$")
    axis.set_ylabel("centered-risk exponent")
    axis.set_title("(c) Theory versus finite-width measurement")
    axis.set_ylim(bottom=-0.05)
    axis.grid(True, alpha=0.17)
    axis.legend(frameon=False, fontsize=7)

    figure.suptitle(
        r"Experiment one at $m=10^4$: finite-width authenticity diagnostic",
        fontsize=12,
        y=1.02,
    )
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    png = output_path.with_suffix(".png")
    pdf = output_path.with_suffix(".pdf")
    figure.savefig(png, dpi=220, bbox_inches="tight")
    figure.savefig(pdf, bbox_inches="tight")
    plt.close(figure)
    return png, pdf


def main() -> None:
    profile = BRIDGE_EXPERIMENT_ONE_M10000_PROFILE
    bridge_summary = run_profile(profile, BRIDGE_DIR)
    curves = _load_curves(BRIDGE_DIR / "bridge_curves.csv")
    spectrum = analytic_power_law_spectrum(profile.alpha, profile.beta, 10_000)
    slope_rows = _slope_rows(curves, spectrum.approximation_floor)
    _write_csv(OUTPUT_DIR / "experiment_one_slope_metrics.csv", slope_rows)
    png, pdf = _make_figure(
        curves,
        slope_rows,
        spectrum.approximation_floor,
        OUTPUT_DIR / "experiment_one_m10000",
    )
    route_counts = {
        route: sum(
            row["theory_status"] == "PASS"
            for row in slope_rows
            if row["route"] == route
        )
        for route in ("discrete_exact", "continuum", "true_sgd")
    }
    claim_ready = all(row["theory_status"] == "PASS" for row in slope_rows)
    summary = {
        "experiment": "experiment_one_m10000",
        "width": 10_000,
        "object_scope": "finite-width analytic-spectrum Gaussian online SGD",
        "bridge_all_gates_pass": bridge_summary["all_bridge_gates_pass"],
        "phase_claim_ready": claim_ready,
        "theory_pass_counts_out_of_eight": route_counts,
        "approximation_floor": spectrum.approximation_floor,
        "fit_window": [
            10_000 ** profile.fit_lower_exponent,
            10_000 ** profile.fit_upper_exponent,
        ],
        "interpretation": (
            "authenticity evidence is separate from joint-limit phase-law evidence"
        ),
        "artifacts": {
            "figure_png": str(png),
            "figure_pdf": str(pdf),
            "slope_metrics": str(
                OUTPUT_DIR / "experiment_one_slope_metrics.csv"
            ),
            "bridge_curves": str(BRIDGE_DIR / "bridge_curves.csv"),
            "bridge_metrics": str(BRIDGE_DIR / "bridge_metrics.csv"),
        },
    }
    (OUTPUT_DIR / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
