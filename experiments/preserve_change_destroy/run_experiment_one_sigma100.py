"""Generate exploratory experiment one at m=10,000 and sigma squared 100.

This setting was selected after inspecting the preregistered noise sweep, so
the resulting phase plot is explicitly exploratory.  The variance-ten locked
bridge is transformed by the pathwise-exact linear response of the directly
evolved noise component; no slope window or acceptance threshold is changed.
"""

from __future__ import annotations

import csv
import json
import math
import os
from pathlib import Path
from typing import Any

import numpy as np

from .bridge_config import BRIDGE_EXPERIMENT_ONE_M10000_PROFILE
from .bridge_validation import (
    analytic_power_law_spectrum,
    curve_comparison,
    fit_effective_exponent,
)
from .run_experiment_one_m10000 import (
    _local_variation,
    _phase,
    _predicted_total_exponent,
)
from .slopes import local_log_slopes


ROOT = Path(__file__).resolve().parent
SOURCE_DIR = ROOT / "artifacts" / "bridge_experiment_one_m10000"
NOISE_SWEEP_DIR = ROOT / "artifacts" / "noise_sweep"
OUTPUT_DIR = ROOT / "artifacts" / "experiment_one_m10000_sigma100"
SOURCE_SIGMA2 = 10.0
TARGET_SIGMA2 = 100.0
NOISE_MULTIPLIER = TARGET_SIGMA2 / SOURCE_SIGMA2
MAXIMUM_EXPONENT_ERROR = 0.05
MAXIMUM_LOCAL_VARIATION = 0.15


def _finite(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _finite(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_finite(item) for item in value]
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if math.isfinite(number) else None
    return value


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(_finite(payload), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("cannot write an empty CSV")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    fields = list(rows[0])
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _rescale_components(
    clean: np.ndarray,
    gap: np.ndarray,
    multiplier: float = NOISE_MULTIPLIER,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    clean_array = np.asarray(clean, dtype=float)
    gap_array = multiplier * np.asarray(gap, dtype=float)
    return clean_array, gap_array, clean_array + gap_array


def _load_sigma100_curves(path: Path) -> dict[float, dict[str, np.ndarray]]:
    grouped: dict[float, list[dict[str, str]]] = {}
    with path.open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            grouped.setdefault(float(row["theta"]), []).append(row)
    output: dict[float, dict[str, np.ndarray]] = {}
    for theta, rows in grouped.items():
        rows.sort(key=lambda row: int(row["step"]))
        data = {
            field: np.asarray([float(row[field]) for row in rows], dtype=float)
            for field in (
                "intrinsic_time",
                "exact_clean",
                "continuum_clean",
                "true_sgd_clean",
                "true_sgd_clean_se",
                "exact_gap",
                "continuum_gap",
                "true_sgd_gap",
                "true_sgd_gap_se",
            )
        }
        transformed: dict[str, np.ndarray] = {
            "intrinsic_time": data["intrinsic_time"]
        }
        for route in ("exact", "continuum", "true_sgd"):
            clean, gap, total = _rescale_components(
                data[f"{route}_clean"], data[f"{route}_gap"]
            )
            transformed[f"{route}_clean"] = clean
            transformed[f"{route}_gap"] = gap
            transformed[f"{route}_total"] = total
        transformed["true_sgd_clean_se"] = data["true_sgd_clean_se"]
        transformed["true_sgd_gap_se"] = (
            NOISE_MULTIPLIER * data["true_sgd_gap_se"]
        )
        transformed["true_sgd_total_se"] = np.sqrt(
            transformed["true_sgd_clean_se"] ** 2
            + transformed["true_sgd_gap_se"] ** 2
        )
        output[theta] = transformed
    expected = set(BRIDGE_EXPERIMENT_ONE_M10000_PROFILE.theta_values)
    if set(output) != expected:
        raise ValueError("locked bridge artifact does not contain all eight schedules")
    return output


def _curve_rows(curves: dict[float, dict[str, np.ndarray]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for theta in BRIDGE_EXPERIMENT_ONE_M10000_PROFILE.theta_values:
        data = curves[theta]
        for index, intrinsic_time in enumerate(data["intrinsic_time"]):
            rows.append(
                {
                    "width": 10_000,
                    "sigma2": TARGET_SIGMA2,
                    "theta": theta,
                    "step": index,
                    "intrinsic_time": intrinsic_time,
                    "exact_clean": data["exact_clean"][index],
                    "continuum_clean": data["continuum_clean"][index],
                    "true_sgd_clean": data["true_sgd_clean"][index],
                    "true_sgd_clean_se": data["true_sgd_clean_se"][index],
                    "exact_gap": data["exact_gap"][index],
                    "continuum_gap": data["continuum_gap"][index],
                    "true_sgd_gap": data["true_sgd_gap"][index],
                    "true_sgd_gap_se": data["true_sgd_gap_se"][index],
                    "exact_total": data["exact_total"][index],
                    "continuum_total": data["continuum_total"][index],
                    "true_sgd_total": data["true_sgd_total"][index],
                    "true_sgd_total_se": data["true_sgd_total_se"][index],
                }
            )
    return rows


def _slope_rows(
    curves: dict[float, dict[str, np.ndarray]],
    floor: float,
) -> list[dict[str, Any]]:
    profile = BRIDGE_EXPERIMENT_ONE_M10000_PROFILE
    width = profile.widths[0]
    lower = float(width) ** profile.fit_lower_exponent
    upper = float(width) ** profile.fit_upper_exponent
    rows: list[dict[str, Any]] = []
    for theta in profile.theta_values:
        data = curves[theta]
        predicted = _predicted_total_exponent(theta)
        for route in ("exact", "continuum", "true_sgd"):
            values = np.maximum(data[f"{route}_total"] - floor, np.finfo(float).tiny)
            exponent = fit_effective_exponent(
                data["intrinsic_time"], values, lower, upper
            )
            local_median, local_variation = _local_variation(
                data["intrinsic_time"], values, lower, upper
            )
            exponent_error = abs(exponent - predicted)
            error_only_pass = bool(exponent_error <= MAXIMUM_EXPONENT_ERROR)
            strict_pass = bool(
                error_only_pass and local_variation <= MAXIMUM_LOCAL_VARIATION
            )
            rows.append(
                {
                    "width": width,
                    "sigma2": TARGET_SIGMA2,
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
                    "maximum_exponent_error": MAXIMUM_EXPONENT_ERROR,
                    "maximum_local_slope_variation": MAXIMUM_LOCAL_VARIATION,
                    "error_only_status": "PASS" if error_only_pass else "INCONCLUSIVE",
                    "strict_theory_status": "PASS" if strict_pass else "INCONCLUSIVE",
                }
            )
    return rows


def _agreement_rows(
    curves: dict[float, dict[str, np.ndarray]],
) -> list[dict[str, Any]]:
    profile = BRIDGE_EXPERIMENT_ONE_M10000_PROFILE
    lower = 10_000 ** profile.fit_lower_exponent
    upper = 10_000 ** profile.fit_upper_exponent
    rows: list[dict[str, Any]] = []
    for theta in profile.theta_values:
        data = curves[theta]
        times = data["intrinsic_time"]
        mask = (times >= lower) & (times <= upper)
        for observable in ("clean", "gap", "total"):
            exact = data[f"exact_{observable}"]
            exact_slope = fit_effective_exponent(times, exact, lower, upper)
            for route in ("continuum", "true_sgd"):
                candidate = data[f"{route}_{observable}"]
                comparison = curve_comparison(candidate[mask], exact[mask])
                candidate_slope = fit_effective_exponent(
                    times, candidate, lower, upper
                )
                rows.append(
                    {
                        "width": 10_000,
                        "sigma2": TARGET_SIGMA2,
                        "theta": theta,
                        "observable": observable,
                        "route": route,
                        "fit_relative_l2_error": comparison["relative_l2_error"],
                        "exact_effective_exponent": exact_slope,
                        "candidate_effective_exponent": candidate_slope,
                        "absolute_slope_error": abs(candidate_slope - exact_slope),
                    }
                )
    return rows


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
    lower = 10_000 ** profile.fit_lower_exponent
    upper = 10_000 ** profile.fit_upper_exponent
    representatives = (0.25, 0.50, 0.90)
    colors = {0.25: "#c43c39", 0.50: "#d18700", 0.90: "#2878b5"}
    figure, axes = plt.subplots(1, 3, figsize=(14.2, 3.95))

    axis = axes[0]
    for theta in representatives:
        data = curves[theta]
        times = data["intrinsic_time"]
        valid = times > 0.0
        for route, linestyle, linewidth in (
            ("exact", "-", 1.9),
            ("continuum", "--", 1.2),
        ):
            centered = np.maximum(
                data[f"{route}_total"] - floor, np.finfo(float).tiny
            )
            axis.loglog(
                times[valid], centered[valid], color=colors[theta],
                linestyle=linestyle, linewidth=linewidth,
            )
        sampled = np.maximum(data["true_sgd_total"] - floor, np.finfo(float).tiny)
        indices = np.flatnonzero(valid)[::30]
        axis.scatter(
            times[indices], sampled[indices], color=colors[theta], marker="x",
            s=14, linewidths=0.8,
        )
        exact_centered = np.maximum(
            data["exact_total"] - floor, np.finfo(float).tiny
        )
        guide_times = np.geomspace(lower, upper, 120)
        anchor_time = math.sqrt(lower * upper)
        anchor_value = float(
            np.exp(
                np.interp(
                    math.log(anchor_time),
                    np.log(times[valid]),
                    np.log(exact_centered[valid]),
                )
            )
        )
        theory_guide = anchor_value * (
            guide_times / anchor_time
        ) ** (-_predicted_total_exponent(theta))
        axis.loglog(
            guide_times,
            theory_guide,
            color=colors[theta],
            linestyle="-.",
            linewidth=2.15,
            alpha=0.9,
        )
    axis.axvspan(lower, upper, color="#d18700", alpha=0.07)
    axis.set_xlabel(r"intrinsic time $T$")
    axis.set_ylabel("centered total loss")
    axis.set_title(r"(a) $m=10^4$, $\sigma^2=100$")
    axis.grid(True, which="both", alpha=0.17)
    handles = [
        Line2D(
            [0], [0], color=colors[theta],
            label=rf"$\vartheta={theta:g},\ q_{{\rm th}}={_predicted_total_exponent(theta):g}$",
        )
        for theta in representatives
    ] + [
        Line2D([0], [0], color="black", label="discrete exact"),
        Line2D([0], [0], color="black", linestyle="--", label="continuum"),
        Line2D([0], [0], color="black", marker="x", linestyle="None", label="true-SGD mean"),
        Line2D([0], [0], color="black", linestyle="-.", linewidth=2.15,
               label=r"theory slope $T^{-q_{\rm th}}$"),
    ]
    axis.legend(handles=handles, frameon=False, fontsize=6.8, ncol=2, loc="lower left")

    axis = axes[1]
    for theta in representatives:
        data = curves[theta]
        times = data["intrinsic_time"]
        valid = (times >= lower) & (times <= upper)
        for route, linestyle, alpha_value in (
            ("exact", "-", 1.0),
            ("continuum", "--", 0.9),
            ("true_sgd", ":", 0.72),
        ):
            values = np.maximum(data[f"{route}_total"] - floor, np.finfo(float).tiny)
            slopes = local_log_slopes(
                times, values, valid=valid,
                half_window_decades=0.15, minimum_points=7,
            )
            finite = valid & np.isfinite(slopes)
            axis.semilogx(
                times[finite], slopes[finite], color=colors[theta],
                linestyle=linestyle, linewidth=1.4, alpha=alpha_value,
            )
        axis.axhline(
            _predicted_total_exponent(theta), color=colors[theta],
            linestyle=(0, (2, 2)), linewidth=0.9,
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
        color="black", linewidth=2.0, label="joint-limit theory",
    )
    markers = {"exact": "o", "continuum": "^", "true_sgd": "x"}
    labels = {
        "exact": "discrete exact",
        "continuum": "continuum",
        "true_sgd": "true-SGD mean",
    }
    for route in markers:
        selected = [row for row in slope_rows if row["route"] == route]
        axis.scatter(
            [float(row["theta"]) for row in selected],
            [float(row["measured_exponent"]) for row in selected],
            marker=markers[route], s=39, linewidths=1.05,
            label=labels[route], zorder=3,
        )
    true_rows = [row for row in slope_rows if row["route"] == "true_sgd"]
    error_pass = sum(row["error_only_status"] == "PASS" for row in true_rows)
    strict_pass = sum(row["strict_theory_status"] == "PASS" for row in true_rows)
    axis.text(
        0.04, 0.95,
        f"point error: {error_pass}/8; stable plateau: {strict_pass}/8",
        transform=axis.transAxes, va="top", fontsize=7.5,
    )
    axis.axvline(0.25, color="#c43c39", linestyle=":", linewidth=1.0)
    axis.axvline(0.75, color="#2878b5", linestyle=":", linewidth=1.0)
    axis.set_xlabel(r"batch-growth exponent $\vartheta$")
    axis.set_ylabel("centered-total exponent")
    axis.set_title("(c) Improved alignment, unstable slopes")
    axis.set_ylim(bottom=-0.08)
    axis.grid(True, alpha=0.17)
    axis.legend(frameon=False, fontsize=7, loc="lower right")

    figure.suptitle(
        r"Exploratory experiment one after the noise sweep: $m=10^4$, $\sigma^2=100$",
        fontsize=11.5, y=0.995,
    )
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.94))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    png = output_path.with_suffix(".png")
    pdf = output_path.with_suffix(".pdf")
    figure.savefig(png, dpi=220, bbox_inches="tight")
    figure.savefig(pdf, bbox_inches="tight")
    plt.close(figure)
    return png, pdf


def main() -> None:
    source_summary = json.loads(
        (SOURCE_DIR / "summary.json").read_text(encoding="utf-8")
    )
    noise_summary = json.loads(
        (NOISE_SWEEP_DIR / "summary.json").read_text(encoding="utf-8")
    )
    if not source_summary.get("all_bridge_gates_pass", False):
        raise RuntimeError("locked variance-ten bridge did not pass authenticity gates")
    if not noise_summary.get("gates", {}).get(
        "clean_invariance_and_gap_linearity_pass", False
    ):
        raise RuntimeError("noise-response homogeneity gate did not pass")

    curves = _load_sigma100_curves(SOURCE_DIR / "bridge_curves.csv")
    floor = analytic_power_law_spectrum(0.4, 0.3, 10_000).approximation_floor
    curve_rows = _curve_rows(curves)
    slope_rows = _slope_rows(curves, floor)
    agreement_rows = _agreement_rows(curves)
    _write_csv(OUTPUT_DIR / "experiment_one_sigma100_curves.csv", curve_rows)
    _write_csv(OUTPUT_DIR / "experiment_one_sigma100_slope_metrics.csv", slope_rows)
    _write_csv(OUTPUT_DIR / "experiment_one_sigma100_bridge_metrics.csv", agreement_rows)
    png, pdf = _make_figure(
        curves, slope_rows, floor, OUTPUT_DIR / "experiment_one_m10000_sigma100"
    )

    route_summaries: dict[str, dict[str, Any]] = {}
    for route in ("exact", "continuum", "true_sgd"):
        selected = [row for row in slope_rows if row["route"] == route]
        errors = [float(row["absolute_exponent_error"]) for row in selected]
        route_summaries[route] = {
            "mean_absolute_exponent_error": float(np.mean(errors)),
            "maximum_absolute_exponent_error": max(errors),
            "error_only_pass_count_out_of_eight": sum(
                row["error_only_status"] == "PASS" for row in selected
            ),
            "strict_pass_count_out_of_eight": sum(
                row["strict_theory_status"] == "PASS" for row in selected
            ),
        }
    agreement_summary: dict[str, dict[str, float]] = {}
    for route in ("continuum", "true_sgd"):
        selected = [row for row in agreement_rows if row["route"] == route]
        agreement_summary[route] = {
            "maximum_fit_relative_l2_error": max(
                float(row["fit_relative_l2_error"]) for row in selected
            ),
            "maximum_absolute_slope_error": max(
                float(row["absolute_slope_error"]) for row in selected
            ),
        }
    summary = {
        "experiment": "experiment_one_m10000_sigma100",
        "selection_status": "exploratory_post_noise_sweep_selection",
        "width": 10_000,
        "sigma2": TARGET_SIGMA2,
        "noise_response_construction": (
            "locked sigma2=10 clean plus ten times its directly evolved gap"
        ),
        "noise_response_gate_pass": noise_summary["gates"][
            "clean_invariance_and_gap_linearity_pass"
        ],
        "locked_bridge_gate_pass": source_summary["all_bridge_gates_pass"],
        "route_summaries": route_summaries,
        "agreement_with_discrete_exact": agreement_summary,
        "phase_claim_ready": all(
            row["strict_theory_status"] == "PASS" for row in slope_rows
        ),
        "interpretation": (
            "sigma2=100 improves finite-window point estimates, especially in "
            "destroy/change, but no route has a stable eight-schedule slope plateau"
        ),
        "not_claimed": [
            "a preregistered choice of sigma2=100",
            "a strict finite-width validation of the phase law",
            "nonlinear neural-network training",
        ],
        "artifacts": {
            "figure_png": str(png),
            "figure_pdf": str(pdf),
            "curves": str(OUTPUT_DIR / "experiment_one_sigma100_curves.csv"),
            "slope_metrics": str(
                OUTPUT_DIR / "experiment_one_sigma100_slope_metrics.csv"
            ),
            "bridge_metrics": str(
                OUTPUT_DIR / "experiment_one_sigma100_bridge_metrics.csv"
            ),
        },
    }
    _write_json(OUTPUT_DIR / "summary.json", summary)
    print(json.dumps(_finite(summary), indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
