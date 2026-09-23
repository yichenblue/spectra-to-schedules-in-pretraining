"""Run the preregistered label-noise crossover sweep.

The stochastic bridge is evaluated at m=10,000 for unit label-noise variance
and compared with the already locked variance-ten bridge under common random
numbers.  Exact homogeneity then supplies variances 100 and 1000.  Crossover
ordering is measured on the already validated m_eff=10^30 smooth-continuum
trajectory, the first width in the preceding finite-size experiment at which
all phase-law audits passed.  No window, width, or noise level is selected
from this sweep's outcomes.
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
    BRIDGE_NOISE_SWEEP_UNIT_PROFILE,
)
from .bridge_validation import analytic_power_law_spectrum
from .run_bridge import run_profile as run_bridge_profile


ROOT = Path(__file__).resolve().parent
ARTIFACT_ROOT = ROOT / "artifacts"
OUTPUT_DIR = ARTIFACT_ROOT / "noise_sweep"
UNIT_DIR = ARTIFACT_ROOT / BRIDGE_NOISE_SWEEP_UNIT_PROFILE.name
TEN_DIR = ARTIFACT_ROOT / BRIDGE_EXPERIMENT_ONE_M10000_PROFILE.name
DE_DIR = ARTIFACT_ROOT / "de_finite_size_extrapolation"

NOISE_VARIANCES = (1.0, 10.0, 100.0, 1000.0)
THETA_VALUES = (0.25, 0.50, 0.90)
BRIDGE_WIDTH = 10_000
CONTINUUM_WIDTH = 10**30
DIRECT_RESPONSE_TOLERANCE = 2.0e-12


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


def _group_bridge(path: Path) -> dict[float, dict[str, np.ndarray]]:
    grouped: dict[float, list[dict[str, str]]] = {}
    for row in _read_csv(path):
        if int(row["width"]) == BRIDGE_WIDTH and float(row["theta"]) in THETA_VALUES:
            grouped.setdefault(float(row["theta"]), []).append(row)
    output: dict[float, dict[str, np.ndarray]] = {}
    numeric_fields = (
        "intrinsic_time",
        "exact_clean",
        "continuum_clean",
        "true_sgd_clean",
        "exact_gap",
        "continuum_gap",
        "true_sgd_gap",
    )
    for theta, rows in grouped.items():
        rows.sort(key=lambda row: int(row["step"]))
        output[theta] = {
            field: np.asarray([float(row[field]) for row in rows], dtype=float)
            for field in numeric_fields
        }
    if set(output) != set(THETA_VALUES):
        raise ValueError("bridge artifact does not contain all preregistered theta values")
    return output


def _group_de(path: Path) -> dict[float, dict[str, np.ndarray]]:
    grouped: dict[float, list[dict[str, str]]] = {}
    for row in _read_csv(path):
        if (
            int(row["effective_width"]) == CONTINUUM_WIDTH
            and float(row["theta"]) in THETA_VALUES
        ):
            grouped.setdefault(float(row["theta"]), []).append(row)
    output: dict[float, dict[str, np.ndarray]] = {}
    for theta, rows in grouped.items():
        rows.sort(key=lambda row: float(row["intrinsic_time"]))
        output[theta] = {
            field: np.asarray([float(row[field]) for row in rows], dtype=float)
            for field in ("intrinsic_time", "clean_centered", "noise_gap")
        }
    if set(output) != set(THETA_VALUES):
        raise ValueError("DE artifact does not contain all preregistered theta values")
    return output


def _relative_l2(candidate: np.ndarray, reference: np.ndarray) -> float:
    candidate = np.asarray(candidate, dtype=float)
    reference = np.asarray(reference, dtype=float)
    valid = np.isfinite(candidate) & np.isfinite(reference)
    denominator = float(np.linalg.norm(reference[valid]))
    if denominator == 0.0:
        return float(np.linalg.norm(candidate[valid] - reference[valid]))
    return float(np.linalg.norm(candidate[valid] - reference[valid]) / denominator)


def _bridge_routes(
    data: dict[str, np.ndarray], floor: float
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    return {
        "discrete_exact": (data["exact_clean"] - floor, data["exact_gap"]),
        "scheduled_continuum": (
            data["continuum_clean"] - floor,
            data["continuum_gap"],
        ),
        "true_sgd": (data["true_sgd_clean"] - floor, data["true_sgd_gap"]),
    }


def _direct_response_checks(
    unit: dict[float, dict[str, np.ndarray]],
    ten: dict[float, dict[str, np.ndarray]],
    floor: float,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for theta in THETA_VALUES:
        unit_routes = _bridge_routes(unit[theta], floor)
        ten_routes = _bridge_routes(ten[theta], floor)
        for route in unit_routes:
            clean_unit, gap_unit = unit_routes[route]
            clean_ten, gap_ten = ten_routes[route]
            clean_error = _relative_l2(clean_ten, clean_unit)
            gap_error = _relative_l2(gap_ten / 10.0, gap_unit)
            rows.extend(
                [
                    {
                        "theta": theta,
                        "route": route,
                        "check": "clean_invariance_sigma2_1_vs_10",
                        "relative_l2_error": clean_error,
                        "tolerance": DIRECT_RESPONSE_TOLERANCE,
                        "status": "PASS" if clean_error <= DIRECT_RESPONSE_TOLERANCE else "FAIL",
                    },
                    {
                        "theta": theta,
                        "route": route,
                        "check": "gap_linearity_sigma2_1_vs_10",
                        "relative_l2_error": gap_error,
                        "tolerance": DIRECT_RESPONSE_TOLERANCE,
                        "status": "PASS" if gap_error <= DIRECT_RESPONSE_TOLERANCE else "FAIL",
                    },
                ]
            )
    return rows


def _log_crossing(
    times: np.ndarray,
    ratio: np.ndarray,
    direction: str,
    *,
    after_index: int = 0,
) -> float | None:
    if direction not in {"up", "down"}:
        raise ValueError("direction must be up or down")
    times = np.asarray(times, dtype=float)
    ratio = np.asarray(ratio, dtype=float)
    start = max(1, int(after_index) + 1)
    for index in range(start, times.size):
        previous = ratio[index - 1]
        current = ratio[index]
        if times[index - 1] <= 0.0 or previous <= 0.0 or current <= 0.0:
            continue
        crossed = (
            previous < 1.0 <= current
            if direction == "up"
            else previous > 1.0 >= current
        )
        if not crossed:
            continue
        x0 = math.log(float(times[index - 1]))
        x1 = math.log(float(times[index]))
        y0 = math.log(float(previous))
        y1 = math.log(float(current))
        if y1 == y0:
            return float(times[index])
        return float(math.exp(x0 - y0 * (x1 - x0) / (y1 - y0)))
    return None


def _crossover_row(
    *,
    width: int,
    theta: float,
    sigma2: float,
    route: str,
    times: np.ndarray,
    clean: np.ndarray,
    gap: np.ndarray,
) -> dict[str, Any]:
    positive = (times > 0.0) & (clean > 0.0) & (gap > 0.0)
    selected_times = times[positive]
    ratio = gap[positive] / clean[positive]
    if selected_times.size < 2:
        raise ValueError("crossover calculation needs positive clean and gap curves")
    peak_index = int(np.argmax(ratio))
    if theta < 0.75:
        kind = "gap_takeover"
        crossing = _log_crossing(selected_times, ratio, "up")
        if crossing is None:
            status = "right_censored" if ratio[-1] < 1.0 else "left_censored"
        else:
            status = "observed"
        effective_time = crossing if crossing is not None else float("inf")
    else:
        kind = "clean_takeover_after_gap_peak"
        crossing = _log_crossing(
            selected_times, ratio, "down", after_index=peak_index
        )
        if float(np.max(ratio)) < 1.0:
            status = "clean_dominant_throughout"
            effective_time = 0.0
        elif crossing is None:
            status = "right_censored"
            effective_time = float("inf")
        else:
            status = "observed"
            effective_time = crossing
    return {
        "effective_width": width,
        "theta": theta,
        "sigma2": sigma2,
        "route": route,
        "crossover_kind": kind,
        "crossover_time": crossing,
        "ordering_time": effective_time,
        "censoring_status": status,
        "maximum_gap_to_clean_ratio": float(np.max(ratio)),
        "terminal_gap_to_clean_ratio": float(ratio[-1]),
        "peak_time": float(selected_times[peak_index]),
        "time_horizon": float(selected_times[-1]),
    }


def _assemble_bridge_sweep(
    unit: dict[float, dict[str, np.ndarray]],
    ten: dict[float, dict[str, np.ndarray]],
    floor: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    curve_rows: list[dict[str, Any]] = []
    crossover_rows: list[dict[str, Any]] = []
    for theta in THETA_VALUES:
        times = unit[theta]["intrinsic_time"]
        unit_routes = _bridge_routes(unit[theta], floor)
        ten_routes = _bridge_routes(ten[theta], floor)
        for sigma2 in NOISE_VARIANCES:
            source = "direct" if sigma2 in {1.0, 10.0} else "unit_response_rescale"
            for route in unit_routes:
                if sigma2 == 10.0:
                    clean, gap = ten_routes[route]
                else:
                    clean_unit, gap_unit = unit_routes[route]
                    clean = clean_unit
                    gap = sigma2 * gap_unit
                crossover_rows.append(
                    _crossover_row(
                        width=BRIDGE_WIDTH,
                        theta=theta,
                        sigma2=sigma2,
                        route=route,
                        times=times,
                        clean=clean,
                        gap=gap,
                    )
                )
                for index, intrinsic_time in enumerate(times):
                    curve_rows.append(
                        {
                            "object": "finite_width_bridge",
                            "effective_width": BRIDGE_WIDTH,
                            "theta": theta,
                            "sigma2": sigma2,
                            "route": route,
                            "source": source,
                            "intrinsic_time": intrinsic_time,
                            "clean_centered": clean[index],
                            "noise_gap": gap[index],
                            "noisy_centered": clean[index] + gap[index],
                            "gap_to_clean_ratio": (
                                gap[index] / clean[index]
                                if clean[index] > 0.0 else float("nan")
                            ),
                        }
                    )
    return curve_rows, crossover_rows


def _assemble_continuum_sweep(
    de: dict[float, dict[str, np.ndarray]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    curve_rows: list[dict[str, Any]] = []
    crossover_rows: list[dict[str, Any]] = []
    for theta in THETA_VALUES:
        data = de[theta]
        for sigma2 in NOISE_VARIANCES:
            clean = data["clean_centered"]
            gap = (sigma2 / 10.0) * data["noise_gap"]
            crossover_rows.append(
                _crossover_row(
                    width=CONTINUUM_WIDTH,
                    theta=theta,
                    sigma2=sigma2,
                    route="smooth_continuum",
                    times=data["intrinsic_time"],
                    clean=clean,
                    gap=gap,
                )
            )
            for index, intrinsic_time in enumerate(data["intrinsic_time"]):
                curve_rows.append(
                    {
                        "object": "validated_continuum_proxy",
                        "effective_width": CONTINUUM_WIDTH,
                        "theta": theta,
                        "sigma2": sigma2,
                        "route": "smooth_continuum",
                        "source": "variance_ten_linear_response",
                        "intrinsic_time": intrinsic_time,
                        "clean_centered": clean[index],
                        "noise_gap": gap[index],
                        "noisy_centered": clean[index] + gap[index],
                        "gap_to_clean_ratio": (
                            gap[index] / clean[index]
                            if clean[index] > 0.0 else float("nan")
                        ),
                    }
                )
    return curve_rows, crossover_rows


def _strict_order(values: list[float], direction: str) -> bool:
    if not all(math.isfinite(value) for value in values):
        return False
    if direction == "decreasing":
        return all(left > right for left, right in zip(values, values[1:]))
    if direction == "increasing":
        return all(left <= right for left, right in zip(values, values[1:]))
    raise ValueError(direction)


def _make_figure(
    bridge_curves: list[dict[str, Any]],
    continuum_curves: list[dict[str, Any]],
    continuum_crossovers: list[dict[str, Any]],
    response_checks: list[dict[str, Any]],
    output_path: Path,
) -> tuple[Path, Path]:
    os.environ.setdefault("MPLCONFIGDIR", str(output_path.parent / ".matplotlib"))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    colors = {1.0: "#4c78a8", 10.0: "#59a14f", 100.0: "#f28e2b", 1000.0: "#e15759"}
    figure, axes = plt.subplots(1, 3, figsize=(14.7, 4.05))

    axis = axes[0]
    for sigma2 in NOISE_VARIANCES:
        for theta, linestyle in ((0.25, "-"), (0.50, "--")):
            selected = [
                row for row in continuum_curves
                if row["theta"] == theta and row["sigma2"] == sigma2
            ]
            axis.loglog(
                [row["intrinsic_time"] for row in selected if row["intrinsic_time"] > 0.0],
                [row["gap_to_clean_ratio"] for row in selected if row["intrinsic_time"] > 0.0],
                color=colors[sigma2], linestyle=linestyle, linewidth=1.55,
            )
            crossing = next(
                row for row in continuum_crossovers
                if row["theta"] == theta and row["sigma2"] == sigma2
            )
            if crossing["censoring_status"] == "observed":
                axis.scatter(
                    [crossing["crossover_time"]], [1.0],
                    color=colors[sigma2],
                    marker="o" if theta == 0.25 else "s",
                    s=24,
                    zorder=4,
                )
    axis.axhline(1.0, color="black", linewidth=1.0)
    axis.set_xlabel(r"intrinsic time $T$")
    axis.set_ylabel(r"gap/clean ratio $G/C$")
    axis.set_title("(a) Earlier gap takeover")
    axis.grid(True, which="both", alpha=0.17)
    sigma_handles = [
        Line2D([0], [0], color=colors[value], label=rf"$\sigma^2={value:g}$")
        for value in NOISE_VARIANCES
    ]
    phase_handles = [
        Line2D([0], [0], color="black", linestyle="-", label=r"destroy, $\vartheta=.25$"),
        Line2D([0], [0], color="black", linestyle="--", label=r"change, $\vartheta=.50$"),
    ]
    legend = axis.legend(handles=sigma_handles, frameon=False, fontsize=7, loc="lower right")
    axis.add_artist(legend)
    axis.legend(handles=phase_handles, frameon=False, fontsize=7, loc="upper left")

    axis = axes[1]
    for sigma2 in NOISE_VARIANCES:
        selected = [
            row for row in continuum_curves
            if row["theta"] == 0.90 and row["sigma2"] == sigma2
        ]
        axis.loglog(
            [row["intrinsic_time"] for row in selected if row["intrinsic_time"] > 0.0],
            [row["gap_to_clean_ratio"] for row in selected if row["intrinsic_time"] > 0.0],
            color=colors[sigma2], linewidth=1.7,
        )
        crossing = next(
            row for row in continuum_crossovers
            if row["theta"] == 0.90 and row["sigma2"] == sigma2
        )
        if crossing["censoring_status"] == "observed":
            axis.scatter(
                [crossing["crossover_time"]], [1.0], color=colors[sigma2],
                s=32, marker="o", zorder=4,
            )
    axis.axhline(1.0, color="black", linewidth=1.0)
    axis.set_xlabel(r"intrinsic time $T$")
    axis.set_ylabel(r"gap/clean ratio $G/C$")
    axis.set_title("(b) Delayed clean takeover")
    axis.grid(True, which="both", alpha=0.17)
    axis.legend(handles=sigma_handles, frameon=False, fontsize=7, loc="best")

    axis = axes[2]
    theta = 0.50
    for noise_index, sigma2 in enumerate(NOISE_VARIANCES):
        selected = [
            row for row in bridge_curves
            if row["theta"] == theta
            and row["sigma2"] == sigma2
            and row["route"] == "true_sgd"
            and row["intrinsic_time"] > 0.0
        ]
        times = np.asarray([row["intrinsic_time"] for row in selected])
        clean = np.asarray([row["clean_centered"] for row in selected])
        normalized_gap = np.asarray([row["noise_gap"] for row in selected]) / sigma2
        axis.loglog(
            times, clean, color=colors[sigma2], linewidth=1.0,
            marker="o", markersize=2.8, markevery=(6 + 7 * noise_index, 112),
        )
        axis.loglog(
            times, normalized_gap, color=colors[sigma2], linestyle="--",
            linewidth=1.0, marker="x", markersize=3.0,
            markevery=(13 + 7 * noise_index, 112),
        )
    maximum_error = max(float(row["relative_l2_error"]) for row in response_checks)
    axis.text(
        0.48, 0.05, rf"max normalized residual $={maximum_error:.1e}$",
        transform=axis.transAxes, fontsize=8,
    )
    axis.set_xlabel(r"intrinsic time $T$")
    axis.set_ylabel("centered response")
    axis.set_title(r"(c) $C$ invariant; $G/\sigma^2$ invariant")
    axis.grid(True, which="both", alpha=0.17)
    response_handles = [
        Line2D([0], [0], color="black", linestyle="-", label="clean component"),
        Line2D([0], [0], color="black", linestyle="--", label=r"normalized gap $G/\sigma^2$"),
    ]
    first = axis.legend(handles=sigma_handles, frameon=False, fontsize=7, loc="lower left")
    axis.add_artist(first)
    axis.legend(handles=response_handles, frameon=False, fontsize=7, loc="upper right")

    figure.suptitle(
        r"Preregistered noise sweep: $m=10^4$ bridge and $m_{\rm eff}=10^{30}$ crossover resolution",
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
    parser.add_argument("--postprocess-only", action="store_true")
    arguments = parser.parse_args()

    if arguments.postprocess_only:
        unit_summary = json.loads((UNIT_DIR / "summary.json").read_text(encoding="utf-8"))
    else:
        print(json.dumps({"stage": "unit_noise_true_sgd", "sigma2": 1.0}), flush=True)
        unit_summary = run_bridge_profile(BRIDGE_NOISE_SWEEP_UNIT_PROFILE, UNIT_DIR)
    ten_summary = json.loads((TEN_DIR / "summary.json").read_text(encoding="utf-8"))
    expected_ten = json.loads(json.dumps(asdict(BRIDGE_EXPERIMENT_ONE_M10000_PROFILE)))
    if ten_summary.get("profile_config") != expected_ten:
        raise RuntimeError("locked sigma2=10 bridge artifact does not match its profile")
    de_summary = json.loads((DE_DIR / "summary.json").read_text(encoding="utf-8"))
    if not de_summary.get("numerical_contract_pass", False):
        raise RuntimeError("finite-size continuum artifact failed its numerical contract")

    unit = _group_bridge(UNIT_DIR / "bridge_curves.csv")
    ten = _group_bridge(TEN_DIR / "bridge_curves.csv")
    de = _group_de(DE_DIR / "curves.csv")
    floor = analytic_power_law_spectrum(0.4, 0.3, BRIDGE_WIDTH).approximation_floor
    response_checks = _direct_response_checks(unit, ten, floor)
    bridge_curves, bridge_crossovers = _assemble_bridge_sweep(unit, ten, floor)
    continuum_curves, continuum_crossovers = _assemble_continuum_sweep(de)
    all_curves = bridge_curves + continuum_curves
    all_crossovers = bridge_crossovers + continuum_crossovers
    _write_csv(OUTPUT_DIR / "noise_sweep_curves.csv", all_curves)
    _write_csv(OUTPUT_DIR / "crossover_metrics.csv", all_crossovers)
    _write_csv(OUTPUT_DIR / "response_checks.csv", response_checks)

    figure_png, figure_pdf = _make_figure(
        bridge_curves,
        continuum_curves,
        continuum_crossovers,
        response_checks,
        OUTPUT_DIR / "noise_sweep",
    )

    continuum_order: dict[str, list[float]] = {}
    direction_pass = True
    for theta in THETA_VALUES:
        selected = [
            row for row in continuum_crossovers if float(row["theta"]) == theta
        ]
        selected.sort(key=lambda row: float(row["sigma2"]))
        values = [float(row["ordering_time"]) for row in selected]
        label = str(theta)
        continuum_order[label] = values
        if theta < 0.75:
            direction_pass = direction_pass and _strict_order(values, "decreasing")
        else:
            direction_pass = direction_pass and _strict_order(values, "increasing")
    response_pass = all(row["status"] == "PASS" for row in response_checks)
    overall_pass = bool(
        unit_summary["all_bridge_gates_pass"]
        and ten_summary["all_bridge_gates_pass"]
        and de_summary["numerical_contract_pass"]
        and response_pass
        and direction_pass
    )
    summary = {
        "experiment": "preregistered_noise_sweep",
        "protocol": {
            "noise_variances": list(NOISE_VARIANCES),
            "theta_values": list(THETA_VALUES),
            "true_sgd_width": BRIDGE_WIDTH,
            "crossover_resolution_effective_width": CONTINUUM_WIDTH,
            "crossover_definition": "first G/C upcrossing for destroy/change; last G/C downcrossing after its peak for preserve",
            "retuned_after_sweep": False,
            "sigma2_100_and_1000_construction": "exact unit-response homogeneity",
        },
        "gates": {
            "unit_noise_bridge_pass": unit_summary["all_bridge_gates_pass"],
            "locked_variance_ten_bridge_pass": ten_summary["all_bridge_gates_pass"],
            "continuum_numerical_contract_pass": de_summary["numerical_contract_pass"],
            "clean_invariance_and_gap_linearity_pass": response_pass,
            "crossover_direction_pass": direction_pass,
            "overall_pass": overall_pass,
        },
        "maximum_direct_response_relative_l2_error": max(
            float(row["relative_l2_error"]) for row in response_checks
        ),
        "continuum_crossover_ordering_times": continuum_order,
        "interpretation": {
            "destroy_change": "increasing label-noise variance moves gap takeover earlier",
            "preserve": "increasing label-noise variance delays clean takeover; if G/C never exceeds one, clean is marked dominant throughout",
            "response": "the directly evolved clean component is invariant and the gap is linear in sigma squared",
        },
        "scope": "analytic-spectrum Gaussian online SGD plus its previously validated smooth-continuum spectral proxy",
        "not_claimed": [
            "a noise-tuned validation of the phase law",
            "nonlinear neural-network training",
            "a true-SGD run at m=10^30",
        ],
        "artifacts": {
            "figure_png": str(figure_png),
            "figure_pdf": str(figure_pdf),
            "curves": str(OUTPUT_DIR / "noise_sweep_curves.csv"),
            "crossovers": str(OUTPUT_DIR / "crossover_metrics.csv"),
            "response_checks": str(OUTPUT_DIR / "response_checks.csv"),
            "unit_bridge_summary": str(UNIT_DIR / "summary.json"),
            "locked_ten_bridge_summary": str(TEN_DIR / "summary.json"),
            "continuum_summary": str(DE_DIR / "summary.json"),
        },
    }
    _write_json(OUTPUT_DIR / "summary.json", summary)
    print(json.dumps(_finite(summary), indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
