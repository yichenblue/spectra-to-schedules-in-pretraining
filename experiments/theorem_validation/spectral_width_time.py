"""Joint width--time test of the polynomial spectral forcing exponent.

The paper's spectral examples make a statement about the learnable forcing,
not an unseparated minibatch-SGD loss.  This experiment therefore evaluates
the exact discrete modal forcing under the same constant-step filter used by
the theory.  Width and terminal intrinsic time are co-scaled so that every
reported point remains before the finite-spectrum cutoff.
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


Array = np.ndarray
ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG = ROOT / "spectral_width_time_v001.json"
DEFAULT_OUTPUT = ROOT / "artifacts" / "spectral_width_time_v001"
CASES = ("polynomial_canonical", "polynomial_sparse_target")


def read_config(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("configuration root must be an object")
    required = {
        "run_id",
        "stage",
        "primary_prediction_id",
        "eta",
        "batch_size",
        "widths",
        "width_power",
        "max_time_over_width_power",
        "evaluation_points",
        "primary_width",
        "primary_fit_window",
    }
    missing = sorted(required - payload.keys())
    if missing:
        raise ValueError(f"configuration misses required keys: {missing}")
    return payload


def modal_data(case: str, width: int) -> tuple[Array, Array]:
    index = np.arange(1, width + 1, dtype=np.float64)
    eigenvalues = index ** (-0.8)
    if case == "polynomial_canonical":
        weights = index ** (-1.4)
    elif case == "polynomial_sparse_target":
        roots = np.arange(1, int(math.isqrt(width)) + 1, dtype=np.int64)
        square_index = roots * roots
        eigenvalues = square_index.astype(np.float64) ** (-0.8)
        weights = square_index.astype(np.float64) ** (-1.4) * (2.0 * roots - 1.0)
    else:
        raise ValueError(f"unknown case: {case}")
    weights /= float(np.sum(weights))
    return eigenvalues, weights


def evaluation_steps(max_time: float, eta: float, count: int) -> Array:
    max_step = int(math.floor(max_time / eta))
    if max_step < 2:
        raise ValueError("co-scaled horizon contains fewer than two updates")
    positive = np.unique(
        np.rint(np.geomspace(1.0, float(max_step), count - 1)).astype(np.int64)
    )
    return np.concatenate((np.asarray([0], dtype=np.int64), positive))


def exact_forcing(
    eigenvalues: Array,
    weights: Array,
    *,
    eta: float,
    batch_size: int,
    steps: Array,
) -> Array:
    q = 1.0 - 2.0 * eta * eigenvalues + (1.0 + 1.0 / batch_size) * eta**2 * eigenvalues**2
    if np.any(q <= 0.0) or np.any(q >= 1.0):
        raise ValueError("modal filter is outside the stable interval (0, 1)")
    log_q = np.log(q)
    return np.asarray(
        [float(weights @ np.exp(float(step) * log_q)) for step in steps],
        dtype=np.float64,
    )


def fit_curve(times: Array, values: Array, lower: float, upper: float) -> dict[str, float | int]:
    mask = (times >= lower) & (times <= upper)
    if int(np.sum(mask)) < 7:
        raise RuntimeError("frozen fit window has fewer than seven points")
    fit = log_ols_exponent(times[mask], values[mask])
    slope_time, slopes = local_exponent_curve(times[mask], values[mask])
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
        "local_slope_points": int(slope_time.size),
    }


def run(config: dict[str, Any], output_dir: Path) -> dict[str, Any]:
    eta = float(config["eta"])
    batch_size = int(config["batch_size"])
    widths = tuple(int(value) for value in config["widths"])
    cutoff_ratio = float(config["max_time_over_width_power"])
    width_power = float(config["width_power"])
    count = int(config["evaluation_points"])
    primary_width = int(config["primary_width"])
    primary_window = tuple(float(value) for value in config["primary_fit_window"])
    sensitivity_windows = tuple(
        tuple(float(value) for value in window)
        for window in config["sensitivity_fit_windows"]
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    began = time.monotonic()

    rows: list[dict[str, Any]] = []
    plot_payload: dict[tuple[str, int], tuple[Array, Array]] = {}
    for case in CASES:
        for width in widths:
            max_time = cutoff_ratio * width**width_power
            steps = evaluation_steps(max_time, eta, count)
            times = eta * steps.astype(np.float64)
            eigenvalues, weights = modal_data(case, width)
            forcing = exact_forcing(
                eigenvalues,
                weights,
                eta=eta,
                batch_size=batch_size,
                steps=steps,
            )
            plot_payload[(case, width)] = (times, forcing)
            for step, intrinsic_time, value in zip(steps, times, forcing, strict=True):
                rows.append(
                    {
                        "case": case,
                        "width": width,
                        "step": int(step),
                        "intrinsic_time": float(intrinsic_time),
                        "forcing": float(value),
                        "time_over_width_power": float(intrinsic_time / width**width_power),
                        "estimated_missing_tail_fraction": float(
                            math.sqrt(intrinsic_time / width**width_power)
                        ),
                    }
                )

    metric_rows: list[dict[str, Any]] = []
    for case in CASES:
        times, forcing = plot_payload[(case, primary_width)]
        for label, window in (
            ("primary", primary_window),
            *((f"sensitivity_{index + 1}", value) for index, value in enumerate(sensitivity_windows)),
        ):
            metric_rows.append(
                {
                    "case": case,
                    "width": primary_width,
                    "window_label": label,
                    "theory_exponent": 0.5,
                    **fit_curve(times, forcing, *window),
                }
            )

    primary = {
        row["case"]: row
        for row in metric_rows
        if row["window_label"] == "primary"
    }
    exponent_difference = abs(
        float(primary[CASES[0]]["power_exponent"])
        - float(primary[CASES[1]]["power_exponent"])
    )
    max_theory_error = max(
        abs(float(row["power_exponent"]) - 0.5) for row in primary.values()
    )
    gates = {
        "primary_max_theory_error": max_theory_error,
        "primary_pair_exponent_difference": exponent_difference,
        "theory_error_tolerance": float(config["theory_error_tolerance"]),
        "pair_difference_tolerance": float(config["pair_difference_tolerance"]),
        "theory_exponent_gate_passed": max_theory_error
        <= float(config["theory_error_tolerance"]),
        "pair_agreement_gate_passed": exponent_difference
        <= float(config["pair_difference_tolerance"]),
        "finite_width_gate_passed": math.sqrt(
            primary_window[1] / primary_width**width_power
        )
        <= float(config["max_missing_tail_fraction"]),
    }
    gates["all_primary_gates_passed"] = all(
        bool(value) for key, value in gates.items() if key.endswith("_gate_passed")
    )

    write_csv(output_dir / "curves.csv", rows)
    write_csv(output_dir / "metrics.csv", metric_rows)
    make_figures(plot_payload, widths, output_dir)
    summary = {
        "schema_version": "spectral_width_time_v001",
        "run_id": config["run_id"],
        "stage": config["stage"],
        "primary_prediction_id": config["primary_prediction_id"],
        "claim_bearing": bool(config.get("claim_bearing", False)),
        "estimand": "exact_discrete_modal_forcing",
        "optimizer_filter": "plain_sgd_constant_step",
        "config": config,
        "metrics": metric_rows,
        "gates": gates,
        "elapsed_seconds": time.monotonic() - began,
        "source_sha256": sha256_file(Path(__file__)),
    }
    write_json(output_dir / "summary.json", summary)
    return summary


def make_figures(
    payload: dict[tuple[str, int], tuple[Array, Array]],
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
            times, forcing = payload[(case, width)]
            positive = times > 0.0
            axis.loglog(times[positive], forcing[positive], color=color, label=f"d={width:,}")
            slope_time, slopes = local_exponent_curve(times, forcing)
            slope_axis.semilogx(slope_time, slopes, color=color, label=f"d={width:,}")
        axis.set_title(case.replace("_", " "))
        axis.set_xlabel(r"Intrinsic time $T=\sum\eta$")
        axis.set_ylabel("Exact learnable forcing")
        axis.grid(alpha=0.2, which="both")
        slope_axis.axhline(0.5, color="black", linewidth=1.0, linestyle="--", label="theory 0.5")
        slope_axis.set_title(case.replace("_", " "))
        slope_axis.set_xlabel(r"Intrinsic time $T=\sum\eta$")
        slope_axis.set_ylabel("Local power exponent")
        slope_axis.grid(alpha=0.2, which="both")
    axes[0].legend(frameon=False, fontsize=8)
    slope_axes[0].legend(frameon=False, fontsize=8)
    figure.suptitle("Polynomial forcing · joint width–time extension")
    slope_figure.suptitle("Local slopes · finite-width-safe windows")
    figure.savefig(output_dir / "forcing_curves.png", dpi=220, bbox_inches="tight")
    figure.savefig(output_dir / "forcing_curves.pdf", bbox_inches="tight")
    slope_figure.savefig(output_dir / "local_slopes.png", dpi=220, bbox_inches="tight")
    slope_figure.savefig(output_dir / "local_slopes.pdf", bbox_inches="tight")
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
