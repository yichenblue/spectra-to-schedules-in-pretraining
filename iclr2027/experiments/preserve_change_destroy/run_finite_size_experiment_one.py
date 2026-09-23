"""Run the locked finite-width validation and finite-size extrapolation protocol.

The stochastic/discrete/scheduled-continuum bridge is trained only on widths
through 4096.  The pre-existing m=10,000 artifact is then treated as a locked
held-out check: it is never used to choose a parameter, window, or correction
law.  Only after both bridge splits pass do we audit the smooth continuum
solver and extrapolate the same equations to larger finite effective widths.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import csv
import json
import math
import os
from pathlib import Path
from typing import Any

import numpy as np

from .bridge_config import (
    BRIDGE_EXPERIMENT_ONE_M10000_PROFILE,
    BRIDGE_FINITE_SIZE_TRAIN_PROFILE,
    BridgeProfile,
)
from .bridge_validation import analytic_power_law_spectrum, fit_effective_exponent
from .de_config import DE_FINITE_SIZE_PROFILE
from .run_bridge import run_profile as run_bridge_profile
from .run_de import run_de_experiment


ROOT = Path(__file__).resolve().parent
ARTIFACT_ROOT = ROOT / "artifacts"
OUTPUT_DIR = ARTIFACT_ROOT / "finite_size_experiment_one"
TRAIN_DIR = ARTIFACT_ROOT / BRIDGE_FINITE_SIZE_TRAIN_PROFILE.name
HELDOUT_DIR = ARTIFACT_ROOT / BRIDGE_EXPERIMENT_ONE_M10000_PROFILE.name
EXTRAPOLATION_DIR = ARTIFACT_ROOT / DE_FINITE_SIZE_PROFILE.name

TRANSITION_MAXIMUM_RELATIVE_L2_ERROR = 0.05
TRANSITION_MAXIMUM_SLOPE_ERROR = 0.05


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
    fields: list[str] = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _load_locked_heldout() -> tuple[dict[str, Any], bool]:
    summary_path = HELDOUT_DIR / "summary.json"
    curves_path = HELDOUT_DIR / "bridge_curves.csv"
    metrics_path = HELDOUT_DIR / "bridge_metrics.csv"
    if summary_path.exists() and curves_path.exists() and metrics_path.exists():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        expected = json.loads(json.dumps(asdict(BRIDGE_EXPERIMENT_ONE_M10000_PROFILE)))
        if summary.get("profile_config") == expected:
            return summary, True
    return (
        run_bridge_profile(
            BRIDGE_EXPERIMENT_ONE_M10000_PROFILE,
            HELDOUT_DIR,
        ),
        False,
    )


def _group_bridge_curves(
    path: Path,
) -> dict[tuple[int, float], dict[str, np.ndarray]]:
    grouped: dict[tuple[int, float], list[dict[str, str]]] = {}
    for row in _read_csv(path):
        key = (int(row["width"]), float(row["theta"]))
        grouped.setdefault(key, []).append(row)
    output: dict[tuple[int, float], dict[str, np.ndarray]] = {}
    for key, rows in grouped.items():
        rows.sort(key=lambda row: int(row["step"]))
        fields = [field for field in rows[0] if field not in {"profile", "batch"}]
        output[key] = {
            field: np.asarray([float(row[field]) for row in rows], dtype=float)
            for field in fields
            if field not in {"width", "theta", "step"}
        }
    return output


def _bridge_component_slopes(
    curves_path: Path,
    profile: BridgeProfile,
    split: str,
) -> list[dict[str, Any]]:
    grouped = _group_bridge_curves(curves_path)
    rows: list[dict[str, Any]] = []
    for width in profile.widths:
        spectrum = analytic_power_law_spectrum(profile.alpha, profile.beta, width)
        floor = spectrum.approximation_floor
        lower = float(width) ** profile.fit_lower_exponent
        upper = float(width) ** profile.fit_upper_exponent
        for theta in profile.theta_values:
            data = grouped[(width, theta)]
            times = data["intrinsic_time"]
            fields = {
                "discrete_exact": {
                    "clean_centered": data["exact_clean"] - floor,
                    "noise_gap": data["exact_gap"],
                    "noisy_centered": data["exact_total"] - floor,
                },
                "scheduled_continuum": {
                    "clean_centered": data["continuum_clean"] - floor,
                    "noise_gap": data["continuum_gap"],
                    "noisy_centered": data["continuum_total"] - floor,
                },
                "true_sgd": {
                    "clean_centered": data["true_sgd_clean"] - floor,
                    "noise_gap": data["true_sgd_gap"],
                    "noisy_centered": data["true_sgd_total"] - floor,
                },
            }
            for route, observables in fields.items():
                for observable, values in observables.items():
                    rows.append(
                        {
                            "split": split,
                            "width": width,
                            "theta": theta,
                            "route": route,
                            "observable": observable,
                            "fit_lower_time": lower,
                            "fit_upper_time": upper,
                            "effective_exponent": fit_effective_exponent(
                                times,
                                np.maximum(values, np.finfo(float).tiny),
                                lower,
                                upper,
                            ),
                        }
                    )
    return rows


def _group_de_curves(path: Path) -> dict[tuple[int, float], dict[str, np.ndarray]]:
    grouped: dict[tuple[int, float], list[dict[str, str]]] = {}
    for row in _read_csv(path):
        key = (int(row["effective_width"]), float(row["theta"]))
        grouped.setdefault(key, []).append(row)
    output: dict[tuple[int, float], dict[str, np.ndarray]] = {}
    for key, rows in grouped.items():
        rows.sort(key=lambda row: float(row["intrinsic_time"]))
        output[key] = {
            field: np.asarray([float(row[field]) for row in rows], dtype=float)
            for field in (
                "intrinsic_time",
                "clean_centered",
                "noise_gap",
                "noisy_centered",
            )
        }
    return output


def _positive_log_interpolate(
    target_times: np.ndarray,
    source_times: np.ndarray,
    source_values: np.ndarray,
) -> np.ndarray:
    valid = (source_times > 0.0) & (source_values > 0.0)
    return np.exp(
        np.interp(
            np.log(target_times),
            np.log(source_times[valid]),
            np.log(source_values[valid]),
        )
    )


def _smooth_transition_rows(
    bridge_curves: dict[tuple[int, float], dict[str, np.ndarray]],
    de_curves: dict[tuple[int, float], dict[str, np.ndarray]],
) -> list[dict[str, Any]]:
    profile = BRIDGE_EXPERIMENT_ONE_M10000_PROFILE
    width = profile.widths[0]
    floor = analytic_power_law_spectrum(profile.alpha, profile.beta, width).approximation_floor
    lower = float(width) ** profile.fit_lower_exponent
    upper = float(width) ** profile.fit_upper_exponent
    rows: list[dict[str, Any]] = []
    for theta in profile.theta_values:
        scheduled = bridge_curves[(width, theta)]
        smooth = de_curves[(width, theta)]
        target_times = scheduled["intrinsic_time"]
        mask = (target_times >= lower) & (target_times <= upper)
        selected_times = target_times[mask]
        observables = {
            "clean_centered": scheduled["continuum_clean"] - floor,
            "noise_gap": scheduled["continuum_gap"],
            "noisy_centered": scheduled["continuum_total"] - floor,
        }
        for observable, reference_full in observables.items():
            reference = reference_full[mask]
            candidate = _positive_log_interpolate(
                selected_times,
                smooth["intrinsic_time"],
                smooth[observable],
            )
            relative_l2 = float(
                np.linalg.norm(candidate - reference) / np.linalg.norm(reference)
            )
            reference_slope = fit_effective_exponent(
                selected_times, reference, lower, upper
            )
            candidate_slope = fit_effective_exponent(
                selected_times, candidate, lower, upper
            )
            slope_error = abs(candidate_slope - reference_slope)
            passed = bool(
                relative_l2 <= TRANSITION_MAXIMUM_RELATIVE_L2_ERROR
                and slope_error <= TRANSITION_MAXIMUM_SLOPE_ERROR
            )
            rows.append(
                {
                    "width": width,
                    "theta": theta,
                    "observable": observable,
                    "reference": "integer-schedule continuum",
                    "candidate": "smooth-schedule continuum",
                    "fit_lower_time": lower,
                    "fit_upper_time": upper,
                    "relative_l2_error": relative_l2,
                    "reference_exponent": reference_slope,
                    "candidate_exponent": candidate_slope,
                    "absolute_slope_error": slope_error,
                    "curve_tolerance": TRANSITION_MAXIMUM_RELATIVE_L2_ERROR,
                    "slope_tolerance": TRANSITION_MAXIMUM_SLOPE_ERROR,
                    "status": "PASS" if passed else "FAIL",
                }
            )
    return rows


def _nominal_de_slopes(
    audit_path: Path,
    de_curves: dict[tuple[int, float], dict[str, np.ndarray]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for raw in _read_csv(audit_path):
        if raw["nominal_window"] != "True":
            continue
        width = int(raw["effective_width"])
        theta = float(raw["theta"])
        observable = raw["observable"]
        lower = float(raw["lower_time"])
        upper = float(raw["upper_time"])
        trajectory = de_curves[(width, theta)]
        descriptive_exponent = fit_effective_exponent(
            trajectory["intrinsic_time"],
            trajectory[observable],
            lower,
            upper,
        )
        predicted = float(raw["predicted_exponent"])
        rows.append(
            {
                "effective_width": width,
                "theta": theta,
                "phase": raw["phase"],
                "observable": observable,
                "fit_lower_time": lower,
                "fit_upper_time": upper,
                "predicted_exponent": predicted,
                "measured_exponent": descriptive_exponent,
                "absolute_exponent_error": abs(descriptive_exponent - predicted),
                "audit_exponent": float(raw["exponent"]),
                "audit_absolute_exponent_error": float(
                    raw["absolute_exponent_error"]
                ),
                "audit_status": raw["audit_status"],
                "theory_status": raw["theory_status"],
                "descriptive_only": raw["audit_status"] != "PASS",
            }
        )
    return rows


def _make_figure(
    heldout_curves: dict[tuple[int, float], dict[str, np.ndarray]],
    de_curves: dict[tuple[int, float], dict[str, np.ndarray]],
    bridge_slopes: list[dict[str, Any]],
    de_slopes: list[dict[str, Any]],
    output_path: Path,
) -> tuple[Path, Path]:
    os.environ.setdefault("MPLCONFIGDIR", str(output_path.parent / ".matplotlib"))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    profile = BRIDGE_EXPERIMENT_ONE_M10000_PROFILE
    width = 10_000
    floor = analytic_power_law_spectrum(profile.alpha, profile.beta, width).approximation_floor
    lower = width ** profile.fit_lower_exponent
    upper = width ** profile.fit_upper_exponent
    representatives = (0.25, 0.50, 0.90)
    colors = {0.25: "#c43c39", 0.50: "#d18700", 0.90: "#2878b5"}
    figure, axes = plt.subplots(1, 3, figsize=(14.6, 4.0))

    axis = axes[0]
    for theta in representatives:
        bridge = heldout_curves[(width, theta)]
        smooth = de_curves[(width, theta)]
        times = bridge["intrinsic_time"]
        positive = times > 0.0
        exact = np.maximum(bridge["exact_total"] - floor, np.finfo(float).tiny)
        sampled = np.maximum(bridge["true_sgd_total"] - floor, np.finfo(float).tiny)
        interpolated = _positive_log_interpolate(
            times[positive], smooth["intrinsic_time"], smooth["noisy_centered"]
        )
        axis.loglog(times[positive], exact[positive], color=colors[theta], linewidth=1.8)
        axis.loglog(
            times[positive], interpolated, color=colors[theta], linewidth=1.2,
            linestyle="--",
        )
        indices = np.flatnonzero(positive)[::32]
        axis.scatter(
            times[indices], sampled[indices], color=colors[theta], marker="x",
            s=13, linewidths=0.8,
        )
    axis.axvspan(lower, upper, color="#d18700", alpha=0.07)
    axis.set_xlabel(r"intrinsic time $T$")
    axis.set_ylabel("centered total loss")
    axis.set_title(r"(a) Locked holdout at $m=10^4$")
    axis.grid(True, which="both", alpha=0.17)
    theta_handles = [
        Line2D([0], [0], color=colors[value], label=rf"$\vartheta={value:g}$")
        for value in representatives
    ]
    route_handles = [
        Line2D([0], [0], color="black", linewidth=1.8, label="discrete exact"),
        Line2D([0], [0], color="black", linestyle="--", label="smooth continuum"),
        Line2D([0], [0], color="black", marker="x", linestyle="None", label="true-SGD mean"),
    ]
    axis.legend(
        handles=theta_handles + route_handles,
        frameon=False,
        fontsize=6.8,
        ncol=2,
        loc="lower left",
    )

    axis = axes[1]
    width_values = sorted({int(row["effective_width"]) for row in de_slopes})
    line_styles = {
        "clean_centered": "--",
        "noise_gap": ":",
        "noisy_centered": "-",
    }
    for theta in representatives:
        for observable, line_style in line_styles.items():
            selected = [
                row for row in de_slopes
                if float(row["theta"]) == theta and row["observable"] == observable
            ]
            selected.sort(key=lambda row: int(row["effective_width"]))
            axis.semilogx(
                [float(row["effective_width"]) for row in selected],
                [float(row["measured_exponent"]) for row in selected],
                color=colors[theta], linestyle=line_style, marker="o", markersize=2.8,
                linewidth=1.35,
            )
    axis.set_xscale("log", base=10)
    axis.set_xlim(min(width_values) / 3.0, max(width_values) * 3.0)
    axis.set_xlabel(r"finite effective width $m$")
    axis.set_ylabel("fixed-window exponent")
    axis.set_title("(b) Componentwise drift")
    axis.grid(True, which="both", alpha=0.17)
    component_handles = [
        Line2D([0], [0], color="black", linestyle=style, label=label)
        for label, style in (
            ("clean", "--"), ("noise gap", ":"), ("total", "-")
        )
    ]
    theta_legend = axis.legend(handles=theta_handles, frameon=False, fontsize=7, loc="upper right")
    axis.add_artist(theta_legend)
    axis.legend(handles=component_handles, frameon=False, fontsize=7, loc="lower right")

    axis = axes[2]
    theta_grid = np.linspace(min(profile.theta_values), max(profile.theta_values), 400)
    axis.plot(
        theta_grid,
        [DE_FINITE_SIZE_PROFILE.total_exponent(value) for value in theta_grid],
        color="black", linewidth=2.0, label="joint-limit theory",
    )
    selected_widths = (10**4, 10**12, 10**48)
    width_styles = {
        10**4: ("#9e9e9e", "o", r"continuum $10^4$"),
        10**12: ("#7b5ea7", "s", r"continuum $10^{12}$"),
        10**48: ("#2f855a", "^", r"continuum $10^{48}$"),
    }
    for candidate_width in selected_widths:
        selected = [
            row for row in de_slopes
            if int(row["effective_width"]) == candidate_width
            and row["observable"] == "noisy_centered"
        ]
        selected.sort(key=lambda row: float(row["theta"]))
        color, marker, label = width_styles[candidate_width]
        axis.plot(
            [float(row["theta"]) for row in selected],
            [float(row["measured_exponent"]) for row in selected],
            color=color, marker=marker, markersize=4.0, linewidth=1.0, label=label,
        )
    heldout_total = [
        row for row in bridge_slopes
        if row["split"] == "locked_holdout"
        and row["route"] == "true_sgd"
        and row["observable"] == "noisy_centered"
    ]
    heldout_total.sort(key=lambda row: float(row["theta"]))
    axis.scatter(
        [float(row["theta"]) for row in heldout_total],
        [float(row["effective_exponent"]) for row in heldout_total],
        color="#c43c39", marker="x", s=36, linewidths=1.1,
        label=r"true SGD $m=10^4$", zorder=4,
    )
    axis.axvline(DE_FINITE_SIZE_PROFILE.destroy_boundary, color="#c43c39", linestyle=":", linewidth=1.0)
    axis.axvline(DE_FINITE_SIZE_PROFILE.preservation_boundary, color="#2878b5", linestyle=":", linewidth=1.0)
    axis.set_xlabel(r"batch-growth exponent $\vartheta$")
    axis.set_ylabel("centered-total exponent")
    axis.set_title("(c) Phase law recovered with width")
    axis.set_ylim(bottom=-0.08)
    axis.grid(True, alpha=0.17)
    axis.legend(frameon=False, fontsize=7, loc="best")

    figure.suptitle(
        "Finite-width authenticity first; fixed-equation extrapolation second",
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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--postprocess-only",
        action="store_true",
        help="reuse completed numerical artifacts and rebuild tables/figure/summary",
    )
    arguments = parser.parse_args()
    if arguments.postprocess_only:
        training_summary = json.loads(
            (TRAIN_DIR / "summary.json").read_text(encoding="utf-8")
        )
        heldout_summary = json.loads(
            (HELDOUT_DIR / "summary.json").read_text(encoding="utf-8")
        )
        de_summary = json.loads(
            (EXTRAPOLATION_DIR / "summary.json").read_text(encoding="utf-8")
        )
        heldout_reused = True
    else:
        print(json.dumps({"stage": "training_bridge", "widths": BRIDGE_FINITE_SIZE_TRAIN_PROFILE.widths}), flush=True)
        training_summary = run_bridge_profile(BRIDGE_FINITE_SIZE_TRAIN_PROFILE, TRAIN_DIR)

        print(json.dumps({"stage": "locked_holdout", "width": 10_000}), flush=True)
        heldout_summary, heldout_reused = _load_locked_heldout()

        print(json.dumps({"stage": "fixed_continuum_extrapolation", "widths": DE_FINITE_SIZE_PROFILE.effective_widths}), flush=True)
        de_summary = run_de_experiment(DE_FINITE_SIZE_PROFILE, EXTRAPOLATION_DIR)

    heldout_curves = _group_bridge_curves(HELDOUT_DIR / "bridge_curves.csv")
    de_curves = _group_de_curves(EXTRAPOLATION_DIR / "curves.csv")
    bridge_slopes = _bridge_component_slopes(
        TRAIN_DIR / "bridge_curves.csv", BRIDGE_FINITE_SIZE_TRAIN_PROFILE, "training"
    )
    bridge_slopes.extend(
        _bridge_component_slopes(
            HELDOUT_DIR / "bridge_curves.csv",
            BRIDGE_EXPERIMENT_ONE_M10000_PROFILE,
            "locked_holdout",
        )
    )
    transition_rows = _smooth_transition_rows(heldout_curves, de_curves)
    de_slopes = _nominal_de_slopes(
        EXTRAPOLATION_DIR / "slope_audits.csv", de_curves
    )
    _write_csv(OUTPUT_DIR / "bridge_component_slopes.csv", bridge_slopes)
    _write_csv(OUTPUT_DIR / "smooth_transition_metrics.csv", transition_rows)
    _write_csv(OUTPUT_DIR / "finite_size_component_slopes.csv", de_slopes)

    figure_png, figure_pdf = _make_figure(
        heldout_curves,
        de_curves,
        bridge_slopes,
        de_slopes,
        OUTPUT_DIR / "finite_size_experiment_one",
    )

    transition_pass = all(row["status"] == "PASS" for row in transition_rows)
    largest = max(DE_FINITE_SIZE_PROFILE.effective_widths)
    largest_rows = [
        row for row in de_slopes if int(row["effective_width"]) == largest
    ]
    largest_component_pass = all(row["theory_status"] == "PASS" for row in largest_rows)
    largest_total_rows = [
        row for row in largest_rows if row["observable"] == "noisy_centered"
    ]
    largest_total_pass = all(row["theory_status"] == "PASS" for row in largest_total_rows)
    finite_width_total_rows = [
        row for row in bridge_slopes
        if row["split"] == "locked_holdout"
        and row["route"] == "true_sgd"
        and row["observable"] == "noisy_centered"
    ]
    finite_width_errors = [
        abs(
            float(row["effective_exponent"])
            - DE_FINITE_SIZE_PROFILE.total_exponent(float(row["theta"]))
        )
        for row in finite_width_total_rows
    ]
    claim_ready = bool(
        training_summary["all_bridge_gates_pass"]
        and heldout_summary["all_bridge_gates_pass"]
        and transition_pass
        and de_summary["numerical_contract_pass"]
        and largest_component_pass
        and largest_total_pass
    )
    summary = {
        "experiment": "finite_size_experiment_one",
        "protocol": {
            "training_widths": list(BRIDGE_FINITE_SIZE_TRAIN_PROFILE.widths),
            "locked_holdout_width": 10_000,
            "heldout_artifact_reused": heldout_reused,
            "finite_effective_widths": list(DE_FINITE_SIZE_PROFILE.effective_widths),
            "retuned_parameters_after_holdout": False,
            "fitted_finite_width_correction": False,
        },
        "gates": {
            "training_bridge_pass": training_summary["all_bridge_gates_pass"],
            "locked_holdout_bridge_pass": heldout_summary["all_bridge_gates_pass"],
            "smooth_transition_pass": transition_pass,
            "extrapolation_numerical_contract_pass": de_summary["numerical_contract_pass"],
            "largest_width_all_components_pass": largest_component_pass,
            "largest_width_total_phase_pass": largest_total_pass,
        },
        "maximum_errors": {
            "training_bridge": training_summary["maximum_errors"],
            "locked_holdout_bridge": heldout_summary["maximum_errors"],
            "smooth_transition_maximum_relative_l2_error": max(
                float(row["relative_l2_error"]) for row in transition_rows
            ),
            "smooth_transition_maximum_absolute_slope_error": max(
                float(row["absolute_slope_error"]) for row in transition_rows
            ),
            "locked_holdout_total_maximum_theory_slope_error": max(finite_width_errors),
        },
        "finite_width_phase_claim_ready": bool(max(finite_width_errors) <= 0.05),
        "fixed_equation_extrapolation_claim_ready": claim_ready,
        "interpretation": (
            "Finite m=10,000 validates the stochastic and continuum dynamics but is "
            "pre-asymptotic for the phase curve; the unchanged continuum equations "
            "recover the componentwise and total phase law as finite effective width grows."
        ),
        "scope": (
            "analytic-spectrum Gaussian online SGD and its validated continuum spectral proxy"
        ),
        "not_claimed": [
            "nonlinear neural-network training",
            "finite-dataset sample-reuse SGD",
            "empirical random-feature spectrum equivalence",
        ],
        "artifacts": {
            "figure_png": str(figure_png),
            "figure_pdf": str(figure_pdf),
            "bridge_component_slopes": str(OUTPUT_DIR / "bridge_component_slopes.csv"),
            "smooth_transition_metrics": str(OUTPUT_DIR / "smooth_transition_metrics.csv"),
            "finite_size_component_slopes": str(OUTPUT_DIR / "finite_size_component_slopes.csv"),
            "training_summary": str(TRAIN_DIR / "summary.json"),
            "locked_holdout_summary": str(HELDOUT_DIR / "summary.json"),
            "extrapolation_summary": str(EXTRAPOLATION_DIR / "summary.json"),
        },
    }
    _write_json(OUTPUT_DIR / "summary.json", summary)
    print(json.dumps(_finite(summary), indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
