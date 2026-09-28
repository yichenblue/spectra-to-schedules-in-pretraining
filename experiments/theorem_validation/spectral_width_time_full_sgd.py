"""Joint width--time extension for the complete clean minibatch-SGD risk.

For Gaussian covariates with diagonal covariance, the expected squared-error
risk obeys an exact scalar Volterra recursion.  The forcing and kernel modal
sums are evaluated with converged geometric spectral bins, while the Volterra
feedback is retained rather than relabeled as forcing.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import time
from typing import Any, Sequence

import numpy as np

from .common import log_ols_exponent, sha256_file, write_csv, write_json
from .spectral_loss_sgd import local_exponent_curve
from .spectral_width_time import CASES, evaluation_steps, modal_data, read_config


Array = np.ndarray
ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG = ROOT / "spectral_width_time_full_sgd_v001.json"
DEFAULT_OUTPUT = ROOT / "artifacts" / "spectral_width_time_full_sgd_v001"


def geometric_bin_edges(size: int, bin_count: int) -> Array:
    raw = np.rint(np.geomspace(1.0, float(size + 1), bin_count + 1)).astype(np.int64) - 1
    return np.unique(np.concatenate((np.asarray([0]), raw, np.asarray([size]))))


def aggregate_exponentials(weights: Array, log_q: Array, bin_count: int) -> tuple[Array, Array]:
    if weights.shape != log_q.shape or np.any(weights < 0.0):
        raise ValueError("modal weights and filters must have matching valid shapes")
    edges = geometric_bin_edges(weights.size, min(bin_count, weights.size))
    starts = edges[:-1]
    total_weight = np.add.reduceat(weights, starts)
    weighted_log_q = np.add.reduceat(weights * log_q, starts)
    positive = total_weight > 0.0
    return total_weight[positive], np.exp(weighted_log_q[positive] / total_weight[positive])


def exponential_series(weights: Array, q: Array, max_step: int) -> Array:
    state = weights.copy()
    output = np.empty(max_step + 1, dtype=np.float64)
    output[0] = float(np.sum(state))
    for step in range(1, max_step + 1):
        state *= q
        output[step] = float(np.sum(state))
    return output


def full_risk_from_volterra(forcing: Array, kernel: Array, coefficient: float) -> Array:
    if forcing.shape != kernel.shape or coefficient < 0.0:
        raise ValueError("forcing and kernel must share a valid grid")
    risk = np.empty_like(forcing)
    risk[0] = forcing[0]
    for step in range(1, forcing.size):
        risk[step] = forcing[step] + coefficient * float(
            risk[:step] @ kernel[step - 1 :: -1]
        )
    if np.any(~np.isfinite(risk)) or np.any(risk <= 0.0):
        raise FloatingPointError("full SGD risk is nonfinite or nonpositive")
    return risk


def modal_sequences(case: str, width: int, eta: float, batch_size: int, max_step: int, bins: int) -> tuple[Array, Array, float]:
    full_index = np.arange(1, width + 1, dtype=np.float64)
    full_lambda = full_index ** (-0.8)
    q_full = 1.0 - 2.0 * eta * full_lambda + (1.0 + 1.0 / batch_size) * eta**2 * full_lambda**2
    log_q_full = np.log(q_full)

    kernel_weights, kernel_q = aggregate_exponentials(full_lambda**2, log_q_full, bins)
    kernel = exponential_series(kernel_weights, kernel_q, max_step)

    eigenvalues, forcing_weights = modal_data(case, width)
    q_forcing = 1.0 - 2.0 * eta * eigenvalues + (1.0 + 1.0 / batch_size) * eta**2 * eigenvalues**2
    forcing_bin_weights, forcing_bin_q = aggregate_exponentials(
        forcing_weights, np.log(q_forcing), bins
    )
    forcing = exponential_series(forcing_bin_weights, forcing_bin_q, max_step)

    row_mass = float(
        (eta**2 / batch_size)
        * np.sum(full_lambda**2 / (1.0 - q_full))
    )
    return forcing, kernel, row_mass


def fit_metric(times: Array, values: Array, window: tuple[float, float]) -> dict[str, float | int]:
    mask = (times >= window[0]) & (times <= window[1])
    if int(np.sum(mask)) < 7:
        raise RuntimeError("primary fit window has fewer than seven evaluations")
    fit = log_ols_exponent(times[mask], values[mask])
    _, slopes = local_exponent_curve(times[mask], values[mask])
    return {
        "fit_lower": float(times[mask][0]),
        "fit_upper": float(times[mask][-1]),
        "fit_points": int(np.sum(mask)),
        "power_exponent": float(fit["exponent"]),
        "power_r2": float(fit["r2"]),
        "local_slope_median": float(np.median(slopes)),
        "local_slope_first": float(slopes[0]),
        "local_slope_last": float(slopes[-1]),
        "local_slope_variation": float(np.ptp(slopes)),
    }


def run(config: dict[str, Any], output_dir: Path) -> dict[str, Any]:
    eta = float(config["eta"])
    batch_size = int(config["batch_size"])
    widths = tuple(int(value) for value in config["widths"])
    width_power = float(config["width_power"])
    cutoff_ratio = float(config["max_time_over_width_power"])
    bins = int(config["spectral_bins"])
    check_bins = int(config["convergence_check_bins"])
    primary_width = int(config["primary_width"])
    primary_window = tuple(float(value) for value in config["primary_fit_window"])
    output_dir.mkdir(parents=True, exist_ok=True)
    began = time.monotonic()

    rows: list[dict[str, Any]] = []
    metrics: list[dict[str, Any]] = []
    payload: dict[tuple[str, int], tuple[Array, Array, Array]] = {}
    primary_high: dict[str, Array] = {}
    primary_low: dict[str, Array] = {}
    row_masses: dict[str, float] = {}

    for width in widths:
        max_time = cutoff_ratio * width**width_power
        eval_steps = evaluation_steps(max_time, eta, int(config["evaluation_points"]))
        max_step = int(eval_steps[-1])
        times = eta * eval_steps.astype(np.float64)
        for case in CASES:
            forcing, kernel, row_mass = modal_sequences(
                case, width, eta, batch_size, max_step, bins
            )
            full_risk = full_risk_from_volterra(forcing, kernel, eta**2 / batch_size)
            sampled_forcing = forcing[eval_steps]
            sampled_risk = full_risk[eval_steps]
            payload[(case, width)] = (times, sampled_forcing, sampled_risk)
            row_masses[f"{case}:{width}"] = row_mass
            for step, intrinsic_time, force, risk in zip(
                eval_steps, times, sampled_forcing, sampled_risk, strict=True
            ):
                rows.append(
                    {
                        "case": case,
                        "width": width,
                        "step": int(step),
                        "intrinsic_time": float(intrinsic_time),
                        "forcing": float(force),
                        "full_minibatch_sgd_risk": float(risk),
                        "feedback_fraction": float((risk - force) / risk),
                        "row_mass": row_mass,
                        "estimated_missing_tail_fraction": float(
                            math.sqrt(intrinsic_time / width**width_power)
                        ),
                    }
                )
            if width == primary_width:
                primary_high[case] = sampled_risk
                low_forcing, low_kernel, _ = modal_sequences(
                    case, width, eta, batch_size, max_step, check_bins
                )
                low_risk = full_risk_from_volterra(
                    low_forcing, low_kernel, eta**2 / batch_size
                )[eval_steps]
                primary_low[case] = low_risk
                metrics.append(
                    {
                        "case": case,
                        "width": width,
                        "theory_exponent": 0.5,
                        **fit_metric(times, sampled_risk, primary_window),
                        "forcing_power_exponent": float(
                            fit_metric(times, sampled_forcing, primary_window)["power_exponent"]
                        ),
                        "max_feedback_fraction_in_fit": float(
                            np.max(
                                ((sampled_risk - sampled_forcing) / sampled_risk)[
                                    (times >= primary_window[0]) & (times <= primary_window[1])
                                ]
                            )
                        ),
                        "spectral_bin_relative_error": float(
                            np.max(np.abs(sampled_risk - low_risk) / sampled_risk)
                        ),
                        "row_mass": row_mass,
                    }
                )

    primary_by_case = {row["case"]: row for row in metrics}
    max_theory_error = max(
        abs(float(row["power_exponent"]) - 0.5) for row in metrics
    )
    pair_difference = abs(
        float(primary_by_case[CASES[0]]["power_exponent"])
        - float(primary_by_case[CASES[1]]["power_exponent"])
    )
    max_bin_error = max(float(row["spectral_bin_relative_error"]) for row in metrics)
    max_row_mass = max(row_masses.values())
    gates = {
        "primary_max_theory_error": max_theory_error,
        "primary_pair_exponent_difference": pair_difference,
        "max_spectral_bin_relative_error": max_bin_error,
        "max_row_mass": max_row_mass,
        "theory_exponent_gate_passed": max_theory_error
        <= float(config["theory_error_tolerance"]),
        "pair_agreement_gate_passed": pair_difference
        <= float(config["pair_difference_tolerance"]),
        "spectral_bin_gate_passed": max_bin_error
        <= float(config["spectral_bin_relative_tolerance"]),
        "row_stability_gate_passed": max_row_mass
        <= float(config["max_row_mass"]),
        "finite_width_gate_passed": math.sqrt(
            primary_window[1] / primary_width**width_power
        )
        <= float(config["max_missing_tail_fraction"]),
    }
    gates["all_primary_gates_passed"] = all(
        bool(value) for key, value in gates.items() if key.endswith("_gate_passed")
    )

    write_csv(output_dir / "curves.csv", rows)
    write_csv(output_dir / "metrics.csv", metrics)
    make_figures(payload, widths, output_dir)
    summary = {
        "schema_version": "spectral_width_time_full_sgd_v001",
        "run_id": config["run_id"],
        "stage": config["stage"],
        "primary_prediction_id": config["primary_prediction_id"],
        "claim_bearing": bool(config.get("claim_bearing", False)),
        "estimand": "exact_expected_complete_clean_minibatch_sgd_risk",
        "config": config,
        "metrics": metrics,
        "gates": gates,
        "elapsed_seconds": time.monotonic() - began,
        "source_sha256": sha256_file(Path(__file__)),
    }
    write_json(output_dir / "summary.json", summary)
    return summary


def make_figures(
    payload: dict[tuple[str, int], tuple[Array, Array, Array]],
    widths: tuple[int, ...],
    output_dir: Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = plt.cm.viridis(np.linspace(0.12, 0.88, len(widths)))
    figure, axes = plt.subplots(1, 2, figsize=(10.8, 4.2), constrained_layout=True)
    slope_figure, slope_axes = plt.subplots(1, 2, figsize=(10.8, 4.2), constrained_layout=True)
    for axis, slope_axis, case in zip(axes, slope_axes, CASES, strict=True):
        for color, width in zip(colors, widths, strict=True):
            times, forcing, risk = payload[(case, width)]
            positive = times > 0.0
            axis.loglog(times[positive], risk[positive], color=color, label=f"d={width:,}")
            slope_time, slopes = local_exponent_curve(times, risk)
            slope_axis.semilogx(slope_time, slopes, color=color, label=f"d={width:,}")
        primary_times, primary_forcing, _ = payload[(case, widths[-1])]
        positive = primary_times > 0.0
        axis.loglog(
            primary_times[positive],
            primary_forcing[positive],
            color="black",
            linewidth=1.0,
            linestyle="--",
            label="forcing at max width",
        )
        slope_axis.axhline(0.5, color="black", linewidth=1.0, linestyle="--", label="theory 0.5")
        axis.set_title(case.replace("_", " "))
        axis.set_xlabel(r"Intrinsic time $T=\sum\eta$")
        axis.set_ylabel("Expected complete minibatch-SGD risk")
        axis.grid(alpha=0.2, which="both")
        slope_axis.set_title(case.replace("_", " "))
        slope_axis.set_xlabel(r"Intrinsic time $T=\sum\eta$")
        slope_axis.set_ylabel("Local power exponent")
        slope_axis.grid(alpha=0.2, which="both")
    axes[0].legend(frameon=False, fontsize=8)
    slope_axes[0].legend(frameon=False, fontsize=8)
    figure.suptitle("Complete clean minibatch-SGD risk · joint width–time extension")
    slope_figure.suptitle("Complete-risk local slopes · finite-width-safe windows")
    figure.savefig(output_dir / "full_risk_curves.png", dpi=220, bbox_inches="tight")
    figure.savefig(output_dir / "full_risk_curves.pdf", bbox_inches="tight")
    slope_figure.savefig(output_dir / "full_risk_local_slopes.png", dpi=220, bbox_inches="tight")
    slope_figure.savefig(output_dir / "full_risk_local_slopes.pdf", bbox_inches="tight")
    plt.close(figure)
    plt.close(slope_figure)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    summary = run(read_config(args.config), args.output_dir)
    print(json.dumps({"output": str(args.output_dir), "gates": summary["gates"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
