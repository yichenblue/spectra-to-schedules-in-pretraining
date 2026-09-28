"""True-minibatch-SGD sweep for resolvable geometric-spectrum spacing.

The previous a=0.35 realization remains the weak-signal baseline.  This run
changes only the log-eigenvalue spacing a in lambda_j=exp(-a(j-1)); all primary
curves use fresh dense Gaussian minibatches and full-vector SGD updates.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import time
from typing import Any, Sequence

import numpy as np

from .common import sha256_file, write_csv, write_json
from .spectral_loss_sgd import evaluation_steps, train_one
from .spectral_long_horizon_counterexamples import (
    local_exponent_curve,
    periodic_power_fit,
)


Array = np.ndarray
ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG = ROOT / "spectral_geometric_spacing_sweep_pilot_v001.json"
DEFAULT_OUTPUT = ROOT / "artifacts" / "spectral_geometric_spacing_sweep_pilot_v001"


def read_config(path: Path) -> dict[str, Any]:
    config = json.loads(path.read_text(encoding="utf-8"))
    required = {
        "run_id", "stage", "primary_prediction_id", "spacings", "width",
        "eta", "batch_size", "steps", "seeds", "evaluation_points",
        "fit_intrinsic_time", "local_window_fraction_of_period", "gates",
    }
    missing = sorted(required - config.keys())
    if missing:
        raise ValueError(f"configuration misses required keys: {missing}")
    spacings = [float(value) for value in config["spacings"]]
    if any(value <= 0.0 for value in spacings) or spacings != sorted(spacings):
        raise ValueError("spacings must be positive and increasing")
    return config


def build_geometric_case(width: int, spacing: float) -> tuple[Array, Array]:
    if width < 16 or spacing <= 0.0:
        raise ValueError("width and spacing must be positive")
    index = np.arange(width, dtype=float)
    eigenvalues = np.maximum(
        np.exp(-spacing * index), np.finfo(np.float64).tiny
    )
    target_squared = np.power(eigenvalues, 0.75)
    energy = float(eigenvalues @ target_squared)
    target = np.sqrt(target_squared / energy)
    if (
        not np.all(np.isfinite(eigenvalues))
        or not np.all(eigenvalues > 0.0)
        or not np.all(np.diff(eigenvalues) <= 0.0)
        or not np.all(np.isfinite(target))
        or not math.isclose(
            float(eigenvalues @ np.square(target)), 1.0,
            rel_tol=0.0, abs_tol=2e-13,
        )
    ):
        raise RuntimeError("invalid geometric spectrum or target")
    return eigenvalues, target


def _fit_mask(times: Array, bounds: Sequence[float]) -> Array:
    mask = (times >= float(bounds[0])) & (times <= float(bounds[1]))
    if int(np.sum(mask)) < 7:
        raise RuntimeError("fit window contains fewer than seven points")
    return mask


def run(config: dict[str, Any], output_dir: Path) -> dict[str, Any]:
    spacings = tuple(float(value) for value in config["spacings"])
    width = int(config["width"])
    eta = float(config["eta"])
    batch_size = int(config["batch_size"])
    steps = int(config["steps"])
    seeds = tuple(int(value) for value in config["seeds"])
    eval_steps = evaluation_steps(steps, int(config["evaluation_points"]))
    times = eta * eval_steps.astype(float)
    fit_mask = _fit_mask(times, config["fit_intrinsic_time"])
    output_dir.mkdir(parents=True, exist_ok=True)
    began = time.monotonic()

    curves: list[dict[str, Any]] = []
    seed_metrics: list[dict[str, Any]] = []
    aggregate_metrics: list[dict[str, Any]] = []
    payload: dict[float, Array] = {}
    for spacing in spacings:
        eigenvalues, target = build_geometric_case(width, spacing)
        traces: list[Array] = []
        for seed_index, seed in enumerate(seeds, start=1):
            trace = train_one(
                eigenvalues, target, eta=eta, batch_size=batch_size,
                steps=steps, seed=seed, eval_steps=eval_steps,
            )
            traces.append(trace)
            seed_fit = periodic_power_fit(
                times[fit_mask], trace[fit_mask], log_period=spacing
            )
            seed_metrics.append({
                "spacing": spacing, "seed": seed,
                "power_exponent": seed_fit["power_exponent"],
                "periodic_amplitude": seed_fit["periodic_amplitude"],
                "periodic_residual_fraction_explained": seed_fit[
                    "periodic_residual_fraction_explained"
                ],
                "power_residual_peak_to_peak": seed_fit[
                    "power_residual_peak_to_peak"
                ],
            })
            for step, intrinsic_time, risk in zip(
                eval_steps, times, trace, strict=True
            ):
                curves.append({
                    "spacing": spacing, "width": width, "seed": seed,
                    "step": int(step), "intrinsic_time": float(intrinsic_time),
                    "population_excess_risk": float(risk),
                })
            print(json.dumps({
                "event": "seed_complete", "spacing": spacing, "seed": seed,
                "seed_index": seed_index, "seed_count": len(seeds),
                "elapsed_seconds": time.monotonic() - began,
            }), flush=True)
        stacked = np.stack(traces)
        payload[spacing] = stacked
        median = np.median(stacked, axis=0)
        fit = periodic_power_fit(
            times[fit_mask], median[fit_mask], log_period=spacing
        )
        half_window_decades = (
            spacing / math.log(10.0)
            * float(config["local_window_fraction_of_period"])
        )
        slope_times, slopes = local_exponent_curve(
            times[fit_mask], median[fit_mask],
            half_window_decades=half_window_decades,
        )
        rows = [row for row in seed_metrics if row["spacing"] == spacing]
        threshold = float(config["gates"]["minimum_periodic_fraction_explained"])
        aggregate_metrics.append({
            "spacing": spacing, "log10_period": spacing / math.log(10.0),
            "observed_cycles": math.log(
                float(config["fit_intrinsic_time"][1])
                / float(config["fit_intrinsic_time"][0])
            ) / spacing,
            "power_exponent": fit["power_exponent"],
            "periodic_amplitude": fit["periodic_amplitude"],
            "periodic_residual_fraction_explained": fit[
                "periodic_residual_fraction_explained"
            ],
            "power_residual_peak_to_peak": fit[
                "power_residual_peak_to_peak"
            ],
            "local_exponent_peak_to_peak": float(np.ptp(slopes)),
            "local_exponent_points": int(slope_times.size),
            "seed_detection_count": sum(
                float(row["periodic_residual_fraction_explained"]) >= threshold
                for row in rows
            ),
            "seed_count": len(seeds),
        })
        write_csv(output_dir / "curves.csv", curves)
        write_csv(output_dir / "seed_metrics.csv", seed_metrics)

    amplitudes = [float(row["periodic_amplitude"]) for row in aggregate_metrics]
    fractions = [
        float(row["periodic_residual_fraction_explained"])
        for row in aggregate_metrics
    ]
    threshold = float(config["gates"]["minimum_periodic_fraction_explained"])
    minimum_detected = int(config["gates"]["minimum_detected_spacings"])
    minimum_seeds = int(config["gates"]["minimum_detecting_seeds_at_largest_spacing"])
    gates = {
        "all_values_finite_positive": all(
            np.all(np.isfinite(traces)) and np.all(traces > 0.0)
            for traces in payload.values()
        ),
        "periodic_amplitude_increases_with_spacing": all(
            right > left for left, right in zip(amplitudes[:-1], amplitudes[1:])
        ),
        "enough_spacings_detected": sum(value >= threshold for value in fractions)
        >= minimum_detected,
        "largest_spacing_seed_replication": (
            int(aggregate_metrics[-1]["seed_detection_count"]) >= minimum_seeds
        ),
    }
    gates["all_spacing_sweep_gates_passed"] = all(gates.values())

    write_csv(output_dir / "metrics.csv", aggregate_metrics)
    make_figure(times, payload, config, output_dir)
    summary = {
        "schema_version": "spectral_geometric_spacing_sweep_v001",
        "run_id": config["run_id"], "stage": config["stage"],
        "primary_prediction_id": config["primary_prediction_id"],
        "claim_bearing": bool(config.get("claim_bearing", False)),
        "primary_estimand": "literal_sampled_minibatch_sgd_population_excess_risk",
        "uses_expected_sgd_recurrence": False,
        "uses_spectral_binning_to_train": False,
        "config": config, "metrics": aggregate_metrics, "gates": gates,
        "elapsed_seconds": time.monotonic() - began,
        "source_sha256": sha256_file(Path(__file__)),
    }
    write_json(output_dir / "summary.json", summary)
    return summary


def make_figure(
    times: Array, payload: dict[float, Array], config: dict[str, Any], output_dir: Path
) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    spacings = tuple(float(value) for value in config["spacings"])
    fit_mask = _fit_mask(times, config["fit_intrinsic_time"])
    colors = plt.cm.viridis(np.linspace(0.15, 0.85, len(spacings)))
    figure, axes = plt.subplots(1, 3, figsize=(13.0, 4.0), constrained_layout=True)
    positive = times > 0.0
    residual_offsets = np.arange(len(spacings), dtype=float) * 0.32
    for color, spacing, offset in zip(colors, spacings, residual_offsets, strict=True):
        traces = payload[spacing]
        median = np.median(traces, axis=0)
        q25, q75 = np.quantile(traces, [0.25, 0.75], axis=0)
        label = rf"$a={spacing:g}$"
        axes[0].loglog(times[positive], median[positive], color=color, label=label)
        axes[0].fill_between(
            times[positive], q25[positive], q75[positive], color=color, alpha=0.12
        )
        half_window = (
            spacing / math.log(10.0)
            * float(config["local_window_fraction_of_period"])
        )
        slope_t, slopes = local_exponent_curve(
            times[fit_mask], median[fit_mask],
            half_window_decades=half_window,
        )
        axes[1].semilogx(slope_t, slopes, color=color, label=label)
        fit = periodic_power_fit(
            times[fit_mask], median[fit_mask], log_period=spacing
        )
        axes[2].plot(
            np.log(times[fit_mask]), fit["power_residual"] + offset,
            color=color, label=label,
        )
        axes[2].plot(
            np.log(times[fit_mask]), fit["periodic_component"] + offset,
            color="black", linewidth=0.65, alpha=0.75,
        )
    axes[0].set(
        xlabel=r"Intrinsic time $T=\sum\eta$", ylabel="Population excess risk",
        title="Geometric-spectrum risk",
    )
    axes[1].set(
        xlabel=r"Intrinsic time $T=\sum\eta$", ylabel="Local power exponent",
        title="Resolved local slope",
    )
    axes[2].set(
        xlabel=r"$\log T$", ylabel="Power-law residual + offset",
        title="Fixed-period fits (black)",
    )
    for axis in axes:
        axis.grid(alpha=0.2, which="both")
        axis.legend(frameon=False, fontsize=8)
    figure.suptitle("Increasing geometric spectral spacing · true minibatch-SGD")
    figure.savefig(output_dir / "geometric_spacing_sweep.png", dpi=220, bbox_inches="tight")
    figure.savefig(output_dir / "geometric_spacing_sweep.pdf", bbox_inches="tight")
    plt.close(figure)


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
