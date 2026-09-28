"""Mechanism pilot for the rapid-target sampled-minibatch crossover.

The primary curves are literal sampled minibatch-SGD trajectories. A
deterministic population-GD curve is retained only as a forcing control.
Actual parameter risk is decomposed into fixed spectral-index bands at every
evaluation; no modal recurrence or expected-SGD proxy is used.
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
from .spectral_loss_sgd import CASES, build_case, evaluation_steps, local_exponent_curve


Array = np.ndarray
ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG = ROOT / "spectral_rapid_target_mechanism_pilot_v001.json"
DEFAULT_OUTPUT = ROOT / "artifacts" / "spectral_rapid_target_mechanism_pilot_v001"
RAPID_CASE = next(case for case in CASES if case.name == "polynomial_rapid_target")


def read_config(path: Path) -> dict[str, Any]:
    config = json.loads(path.read_text(encoding="utf-8"))
    required = {
        "run_id", "stage", "primary_prediction_id", "width", "steps", "eta",
        "batch_sizes", "seeds", "evaluation_points", "tail_fit_window",
        "forcing_transform_fit_window", "spectral_band_edges",
    }
    missing = sorted(required - config.keys())
    if missing:
        raise ValueError(f"configuration misses required keys: {missing}")
    return config


def spectral_slices(width: int, edges: Sequence[int]) -> tuple[tuple[int, int], ...]:
    clipped = sorted(set(max(0, min(width, int(edge))) for edge in edges) | {0, width})
    return tuple((left, right) for left, right in zip(clipped[:-1], clipped[1:]) if right > left)


def sampled_train_with_bands(
    eigenvalues: Array,
    target: Array,
    *,
    eta: float,
    batch_size: int,
    steps: int,
    seed: int,
    eval_steps: Array,
    bands: tuple[tuple[int, int], ...],
) -> tuple[Array, Array]:
    stability = eta * (
        (1.0 + 1.0 / batch_size) * float(eigenvalues[0])
        + float(np.sum(eigenvalues)) / batch_size
    )
    if not 0.0 < stability < 2.0:
        raise ValueError(f"unstable or invalid nominal step: {stability}")
    rng = np.random.default_rng(seed)
    error = -target.copy()
    sqrt_spectrum = np.sqrt(eigenvalues)
    risks = np.empty(eval_steps.size, dtype=np.float64)
    band_risks = np.empty((eval_steps.size, len(bands)), dtype=np.float64)
    cursor = 0
    for step in range(steps + 1):
        if step == int(eval_steps[cursor]):
            coordinate_risk = eigenvalues * np.square(error)
            risks[cursor] = float(np.sum(coordinate_risk))
            band_risks[cursor] = [
                float(np.sum(coordinate_risk[left:right])) for left, right in bands
            ]
            if not math.isclose(
                risks[cursor], float(np.sum(band_risks[cursor])), rel_tol=2e-13, abs_tol=2e-15
            ):
                raise RuntimeError("spectral risk bands do not sum to total risk")
            cursor += 1
            if cursor == eval_steps.size:
                break
        samples = rng.standard_normal((batch_size, eigenvalues.size))
        samples *= sqrt_spectrum
        predictions = samples @ error
        error -= eta * (samples.T @ predictions) / float(batch_size)
        if not np.all(np.isfinite(error)):
            raise FloatingPointError("sampled SGD parameter error became nonfinite")
    return risks, band_risks


def population_gd_control(
    eigenvalues: Array, target: Array, *, eta: float, steps: int, eval_steps: Array
) -> Array:
    error = -target.copy()
    output = np.empty(eval_steps.size, dtype=np.float64)
    cursor = 0
    for step in range(steps + 1):
        if step == int(eval_steps[cursor]):
            output[cursor] = float(eigenvalues @ np.square(error))
            cursor += 1
            if cursor == eval_steps.size:
                break
        error *= 1.0 - eta * eigenvalues
    if np.any(~np.isfinite(output)) or np.any(output <= 0.0):
        raise FloatingPointError("population-GD control became nonfinite or nonpositive")
    return output


def forcing_transform_slope(times: Array, risks: Array, window: tuple[float, float]) -> dict[str, float | int]:
    transformed = -np.log(risks / risks[0])
    mask = (times >= window[0]) & (times <= window[1]) & (transformed > 0.0)
    fit = log_ols_exponent(times[mask], 1.0 / transformed[mask])
    # log_ols_exponent fits y ~ T^-q. Since 1/z ~ T^-gamma, q=gamma.
    return {
        "fit_lower": float(times[mask][0]),
        "fit_upper": float(times[mask][-1]),
        "fit_points": int(np.sum(mask)),
        "forcing_transform_exponent": float(fit["exponent"]),
        "forcing_transform_r2": float(fit["r2"]),
    }


def run(config: dict[str, Any], output_dir: Path) -> dict[str, Any]:
    width = int(config["width"])
    steps = int(config["steps"])
    eta = float(config["eta"])
    batch_sizes = tuple(int(value) for value in config["batch_sizes"])
    seeds = tuple(int(value) for value in config["seeds"])
    tail_window = tuple(float(value) for value in config["tail_fit_window"])
    forcing_window = tuple(float(value) for value in config["forcing_transform_fit_window"])
    bands = spectral_slices(width, config["spectral_band_edges"])
    eval_steps = evaluation_steps(steps, int(config["evaluation_points"]))
    times = eta * eval_steps.astype(np.float64)
    eigenvalues, target = build_case(RAPID_CASE, width)
    output_dir.mkdir(parents=True, exist_ok=True)
    began = time.monotonic()

    curve_rows: list[dict[str, Any]] = []
    band_rows: list[dict[str, Any]] = []
    metrics: list[dict[str, Any]] = []
    payload: dict[int, Array] = {}
    band_payload: dict[int, Array] = {}

    for batch_size in batch_sizes:
        all_risks = []
        all_band_risks = []
        seed_exponents = []
        for seed in seeds:
            risks, band_risks = sampled_train_with_bands(
                eigenvalues, target, eta=eta, batch_size=batch_size, steps=steps,
                seed=seed, eval_steps=eval_steps, bands=bands,
            )
            all_risks.append(risks)
            all_band_risks.append(band_risks)
            mask = (times >= tail_window[0]) & (times <= tail_window[1])
            seed_exponents.append(float(log_ols_exponent(times[mask], risks[mask])["exponent"]))
            for eval_step, intrinsic_time, risk in zip(eval_steps, times, risks, strict=True):
                curve_rows.append({
                    "condition": "sampled_minibatch_sgd", "batch_size": batch_size,
                    "seed": seed, "step": int(eval_step),
                    "intrinsic_time": float(intrinsic_time), "population_risk": float(risk),
                })
            for time_index, (eval_step, intrinsic_time) in enumerate(zip(eval_steps, times, strict=True)):
                for band_index, (left, right) in enumerate(bands):
                    band_rows.append({
                        "batch_size": batch_size, "seed": seed,
                        "step": int(eval_step), "intrinsic_time": float(intrinsic_time),
                        "band_left_index": left + 1, "band_right_index": right,
                        "band_population_risk": float(band_risks[time_index, band_index]),
                        "band_risk_fraction": float(band_risks[time_index, band_index] / risks[time_index]),
                    })
        stacked = np.stack(all_risks)
        stacked_bands = np.stack(all_band_risks)
        payload[batch_size] = stacked
        band_payload[batch_size] = stacked_bands
        median = np.median(stacked, axis=0)
        median_bands = np.median(stacked_bands, axis=0)
        tail_mask = (times >= tail_window[0]) & (times <= tail_window[1])
        tail_fit = log_ols_exponent(times[tail_mask], median[tail_mask])
        final_index = int(np.argmin(np.abs(times - tail_window[1])))
        low_start = next(index for index, (left, _) in enumerate(bands) if left >= 256)
        metrics.append({
            "condition": "sampled_minibatch_sgd", "batch_size": batch_size,
            "seed_count": len(seeds), "tail_fit_lower": tail_window[0],
            "tail_fit_upper": tail_window[1], "tail_power_exponent": float(tail_fit["exponent"]),
            "tail_power_r2": float(tail_fit["r2"]),
            "seed_exponent_mean": float(np.mean(seed_exponents)),
            "seed_exponent_sd": float(np.std(seed_exponents, ddof=1)),
            "risk_at_tail_endpoint": float(median[final_index]),
            "weak_mode_fraction_at_tail_endpoint": float(
                np.sum(median_bands[final_index, low_start:]) / median[final_index]
            ),
        })
        print(json.dumps({
            "event": "batch_complete", "batch_size": batch_size,
            "elapsed_seconds": time.monotonic() - began,
        }), flush=True)

    control = population_gd_control(
        eigenvalues, target, eta=eta, steps=steps, eval_steps=eval_steps
    )
    transform = forcing_transform_slope(times, control, forcing_window)
    for eval_step, intrinsic_time, risk in zip(eval_steps, times, control, strict=True):
        curve_rows.append({
            "condition": "population_gd_forcing_control", "batch_size": "",
            "seed": "", "step": int(eval_step), "intrinsic_time": float(intrinsic_time),
            "population_risk": float(risk),
        })

    ordered_endpoint_risks = [float(row["risk_at_tail_endpoint"]) for row in metrics]
    predicted = float(config["predicted_forcing_transform_exponent"])
    gates = {
        "all_sampled_values_finite_positive": all(
            np.all(np.isfinite(values)) and np.all(values > 0.0) for values in payload.values()
        ),
        "tail_risk_monotone_with_batch": all(
            left >= right for left, right in zip(ordered_endpoint_risks[:-1], ordered_endpoint_risks[1:])
        ),
        "forcing_transform_exponent_error": abs(float(transform["forcing_transform_exponent"]) - predicted),
    }
    gates["forcing_transform_gate_passed"] = (
        gates["forcing_transform_exponent_error"] <= float(config["forcing_transform_tolerance"])
    )
    gates["all_primary_gates_passed"] = all(
        bool(gates[key]) for key in (
            "all_sampled_values_finite_positive", "tail_risk_monotone_with_batch",
            "forcing_transform_gate_passed",
        )
    )

    write_csv(output_dir / "curves.csv", curve_rows)
    write_csv(output_dir / "spectral_band_risks.csv", band_rows)
    write_csv(output_dir / "metrics.csv", metrics)
    make_figure(times, payload, band_payload, control, bands, batch_sizes, output_dir)
    summary = {
        "schema_version": "spectral_rapid_target_mechanism_v001",
        "run_id": config["run_id"], "stage": config["stage"],
        "primary_prediction_id": config["primary_prediction_id"],
        "claim_bearing": bool(config.get("claim_bearing", False)),
        "primary_estimand": "literal_sampled_minibatch_sgd_population_risk",
        "control_estimand": "deterministic_population_gd_forcing",
        "uses_expected_sgd_recurrence": False, "uses_spectral_binning_to_train": False,
        "config": config, "metrics": metrics, "forcing_control": transform,
        "gates": gates, "elapsed_seconds": time.monotonic() - began,
        "source_sha256": sha256_file(Path(__file__)),
    }
    write_json(output_dir / "summary.json", summary)
    return summary


def make_figure(
    times: Array, payload: dict[int, Array], band_payload: dict[int, Array],
    control: Array, bands: tuple[tuple[int, int], ...], batch_sizes: tuple[int, ...],
    output_dir: Path,
) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = plt.cm.viridis(np.linspace(0.12, 0.82, len(batch_sizes)))
    positive = times > 0.0
    figure, axes = plt.subplots(1, 3, figsize=(13.2, 3.9), constrained_layout=True)
    for color, batch_size in zip(colors, batch_sizes, strict=True):
        values = payload[batch_size]
        median = np.median(values, axis=0)
        q25, q75 = np.quantile(values, [0.25, 0.75], axis=0)
        axes[0].loglog(times[positive], median[positive], color=color, label=f"minibatch B={batch_size}")
        axes[0].fill_between(times[positive], q25[positive], q75[positive], color=color, alpha=0.14)
        slope_times, slopes = local_exponent_curve(times, median)
        axes[1].semilogx(slope_times, slopes, color=color, label=f"B={batch_size}")
    axes[0].loglog(times[positive], control[positive], color="black", linewidth=1.5, label="population GD control")
    axes[0].set(xlabel=r"Intrinsic time $T=\sum\eta$", ylabel="Population risk", title="Rapid target: complete risk")
    axes[1].set(xlabel=r"Intrinsic time $T=\sum\eta$", ylabel="Local power exponent", title="Crossover diagnostic")

    selected_batch = batch_sizes[0]
    median_band = np.median(band_payload[selected_batch], axis=0)
    fractions = median_band / np.sum(median_band, axis=1, keepdims=True)
    labels = [f"{left + 1}–{right}" for left, right in bands]
    axes[2].stackplot(times, fractions.T, labels=labels, alpha=0.85)
    axes[2].set_xscale("log")
    axes[2].set(xlabel=r"Intrinsic time $T=\sum\eta$", ylabel="Fraction of actual risk", title=f"Spectral migration, B={selected_batch}")
    for axis in axes:
        axis.grid(alpha=0.2, which="both")
    axes[0].legend(frameon=False, fontsize=8)
    axes[1].legend(frameon=False, fontsize=8)
    axes[2].legend(frameon=False, fontsize=6, ncol=2)
    figure.suptitle("True sampled minibatch-SGD rapid-target mechanism pilot")
    figure.savefig(output_dir / "mechanism.png", dpi=220, bbox_inches="tight")
    figure.savefig(output_dir / "mechanism.pdf", bbox_inches="tight")
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
