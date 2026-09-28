"""Long-horizon literal-minibatch-SGD discriminator for two non-power cases.

This exploratory experiment distinguishes a slowly drifting logarithmically
corrected curve from a log-periodic curve.  Every primary trajectory draws a
fresh dense Gaussian minibatch and updates the complete parameter vector; no
expected-SGD recurrence, spectral binning, or training proxy is used.
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
from .spectral_loss_sgd import CASES, build_case, evaluation_steps, train_one


Array = np.ndarray
ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG = ROOT / "spectral_long_horizon_counterexamples_pilot_v001.json"
DEFAULT_OUTPUT = ROOT / "artifacts" / "spectral_long_horizon_counterexamples_pilot_v001"
CASE_BY_NAME = {case.name: case for case in CASES}


def read_config(path: Path) -> dict[str, Any]:
    config = json.loads(path.read_text(encoding="utf-8"))
    required = {
        "run_id", "stage", "primary_prediction_id", "cases", "width",
        "eta", "batch_size", "steps", "seeds", "evaluation_points",
        "stretched_early_window", "stretched_late_window",
        "geometric_fit_window", "geometric_log_period",
        "stretched_local_half_window_decades",
        "geometric_local_half_window_decades", "gates",
    }
    missing = sorted(required - config.keys())
    if missing:
        raise ValueError(f"configuration misses required keys: {missing}")
    if tuple(config["cases"]) != ("stretched_exponential", "geometric_spectrum"):
        raise ValueError("this discriminator requires the two frozen counterexample cases")
    return config


def local_exponent_curve(
    times: Array,
    values: Array,
    *,
    half_window_decades: float,
    minimum_points: int = 7,
) -> tuple[Array, Array]:
    """Estimate local log-log slopes with an explicitly chosen window."""

    positive = (times > 0.0) & (values > 0.0)
    time = np.asarray(times[positive], dtype=float)
    value = np.asarray(values[positive], dtype=float)
    log_time = np.log10(time)
    centers: list[float] = []
    exponents: list[float] = []
    for index, center in enumerate(log_time):
        mask = np.abs(log_time - center) <= half_window_decades
        if int(np.sum(mask)) < minimum_points:
            continue
        centers.append(float(time[index]))
        exponents.append(float(log_ols_exponent(time[mask], value[mask])["exponent"]))
    return np.asarray(centers), np.asarray(exponents)


def periodic_power_fit(
    times: Array,
    values: Array,
    *,
    log_period: float,
) -> dict[str, Any]:
    """Fit a power law plus a sinusoid with a preregistered ln(time) period."""

    time = np.asarray(times, dtype=float)
    value = np.asarray(values, dtype=float)
    if time.size < 7 or np.any(time <= 0.0) or np.any(value <= 0.0):
        raise ValueError("periodic fit requires at least seven positive observations")
    x = np.log(time)
    y = np.log(value)
    base = np.column_stack((np.ones_like(x), x))
    base_coefficients, _, _, _ = np.linalg.lstsq(base, y, rcond=None)
    base_residual = y - base @ base_coefficients
    omega = 2.0 * math.pi / float(log_period)
    full = np.column_stack((base, np.sin(omega * x), np.cos(omega * x)))
    coefficients, _, _, _ = np.linalg.lstsq(full, y, rcond=None)
    full_residual = y - full @ coefficients
    base_sse = float(base_residual @ base_residual)
    full_sse = float(full_residual @ full_residual)
    return {
        "power_exponent": float(-base_coefficients[1]),
        "periodic_amplitude": float(np.hypot(coefficients[2], coefficients[3])),
        "periodic_residual_fraction_explained": float(
            1.0 - full_sse / max(base_sse, 1e-300)
        ),
        "power_residual_peak_to_peak": float(np.ptp(base_residual)),
        "power_residual": base_residual,
        "periodic_component": full[:, 2:] @ coefficients[2:],
    }


def _window(times: Array, values: Array, bounds: Sequence[float]) -> tuple[Array, Array]:
    mask = (times >= float(bounds[0])) & (times <= float(bounds[1]))
    if int(np.sum(mask)) < 7:
        raise RuntimeError(f"analysis window {bounds} contains fewer than seven points")
    return times[mask], values[mask]


def run(config: dict[str, Any], output_dir: Path) -> dict[str, Any]:
    width = int(config["width"])
    eta = float(config["eta"])
    batch_size = int(config["batch_size"])
    steps = int(config["steps"])
    seeds = tuple(int(seed) for seed in config["seeds"])
    eval_steps = evaluation_steps(steps, int(config["evaluation_points"]))
    times = eta * eval_steps.astype(float)
    output_dir.mkdir(parents=True, exist_ok=True)
    began = time.monotonic()

    curve_rows: list[dict[str, Any]] = []
    payload: dict[str, Array] = {}
    metrics: dict[str, dict[str, Any]] = {}
    for case_name in config["cases"]:
        eigenvalues, target = build_case(CASE_BY_NAME[case_name], width)
        traces = []
        for seed_index, seed in enumerate(seeds, start=1):
            trace = train_one(
                eigenvalues, target, eta=eta, batch_size=batch_size,
                steps=steps, seed=seed, eval_steps=eval_steps,
            )
            traces.append(trace)
            for step, intrinsic_time, risk in zip(eval_steps, times, trace, strict=True):
                curve_rows.append({
                    "case": case_name, "width": width, "seed": seed,
                    "step": int(step), "intrinsic_time": float(intrinsic_time),
                    "population_excess_risk": float(risk),
                })
            print(json.dumps({
                "event": "seed_complete", "case": case_name, "seed": seed,
                "seed_index": seed_index, "seed_count": len(seeds),
                "elapsed_seconds": time.monotonic() - began,
            }), flush=True)
        stacked = np.stack(traces)
        payload[case_name] = stacked
        median = np.median(stacked, axis=0)
        if case_name == "stretched_exponential":
            early_t, early_y = _window(times, median, config["stretched_early_window"])
            late_t, late_y = _window(times, median, config["stretched_late_window"])
            early = log_ols_exponent(early_t, early_y)
            late = log_ols_exponent(late_t, late_y)
            slope_t, slopes = local_exponent_curve(
                times, median,
                half_window_decades=float(config["stretched_local_half_window_decades"]),
            )
            metrics[case_name] = {
                "early_power_exponent": float(early["exponent"]),
                "early_power_r2": float(early["r2"]),
                "late_power_exponent": float(late["exponent"]),
                "late_power_r2": float(late["r2"]),
                "early_to_late_exponent_drop": float(early["exponent"] - late["exponent"]),
                "late_local_exponent": float(slopes[np.argmin(np.abs(slope_t - late_t[-1]))]),
            }
        else:
            fit_t, fit_y = _window(times, median, config["geometric_fit_window"])
            periodic = periodic_power_fit(
                fit_t, fit_y, log_period=float(config["geometric_log_period"])
            )
            slope_t, slopes = local_exponent_curve(
                fit_t, fit_y,
                half_window_decades=float(config["geometric_local_half_window_decades"]),
            )
            metrics[case_name] = {
                key: value for key, value in periodic.items()
                if key not in {"power_residual", "periodic_component"}
            }
            metrics[case_name].update({
                "local_exponent_min": float(np.min(slopes)),
                "local_exponent_max": float(np.max(slopes)),
                "local_exponent_peak_to_peak": float(np.ptp(slopes)),
                "local_exponent_points": int(slope_t.size),
            })

        # Preserve complete cases even if a later case is interrupted.
        write_csv(output_dir / "curves.csv", curve_rows)

    thresholds = config["gates"]
    stretched = metrics["stretched_exponential"]
    geometric = metrics["geometric_spectrum"]
    gates = {
        "all_values_finite_positive": all(
            np.all(np.isfinite(traces)) and np.all(traces > 0.0)
            for traces in payload.values()
        ),
        "stretched_exponent_drift_detected": (
            stretched["early_to_late_exponent_drop"]
            >= float(thresholds["minimum_stretched_exponent_drop"])
            and abs(stretched["late_power_exponent"] - 1.0)
            < abs(stretched["early_power_exponent"] - 1.0)
        ),
        "geometric_local_oscillation_detected": (
            geometric["local_exponent_peak_to_peak"]
            >= float(thresholds["minimum_geometric_local_peak_to_peak"])
        ),
        "geometric_known_period_detected": (
            geometric["periodic_residual_fraction_explained"]
            >= float(thresholds["minimum_geometric_periodic_fraction_explained"])
        ),
    }
    gates["all_discriminator_gates_passed"] = all(gates.values())

    metric_rows = [
        {"case": name, "width": width, "seed_count": len(seeds), **values}
        for name, values in metrics.items()
    ]
    write_csv(output_dir / "metrics.csv", metric_rows)
    make_figure(times, payload, config, output_dir)
    summary = {
        "schema_version": "spectral_long_horizon_counterexamples_v001",
        "run_id": config["run_id"], "stage": config["stage"],
        "primary_prediction_id": config["primary_prediction_id"],
        "claim_bearing": bool(config.get("claim_bearing", False)),
        "primary_estimand": "literal_sampled_minibatch_sgd_population_excess_risk",
        "uses_expected_sgd_recurrence": False,
        "uses_spectral_binning_to_train": False,
        "config": config, "metrics": metrics, "gates": gates,
        "elapsed_seconds": time.monotonic() - began,
        "source_sha256": sha256_file(Path(__file__)),
    }
    write_json(output_dir / "summary.json", summary)
    write_analysis(output_dir, summary)
    return summary


def make_figure(times: Array, payload: dict[str, Array], config: dict[str, Any], output_dir: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(2, 3, figsize=(12.2, 7.0), constrained_layout=True)
    colors = {"stretched_exponential": "#276FBF", "geometric_spectrum": "#D95F02"}
    positive = times > 0.0
    for row, case_name in enumerate(config["cases"]):
        traces = payload[case_name]
        median = np.median(traces, axis=0)
        q25, q75 = np.quantile(traces, [0.25, 0.75], axis=0)
        color = colors[case_name]
        axes[row, 0].loglog(times[positive], median[positive], color=color)
        axes[row, 0].fill_between(times[positive], q25[positive], q75[positive], color=color, alpha=0.16)
        half_window = float(config[
            "stretched_local_half_window_decades" if row == 0
            else "geometric_local_half_window_decades"
        ])
        slope_t, slopes = local_exponent_curve(
            times, median, half_window_decades=half_window
        )
        axes[row, 1].semilogx(slope_t, slopes, color=color)
        axes[row, 0].set(ylabel="Population excess risk")
        axes[row, 1].set(ylabel="Local power exponent")
        axes[row, 0].set_title(case_name.replace("_", " "))

    stretched = np.median(payload["stretched_exponential"], axis=0)
    axes[0, 2].semilogx(times[positive], times[positive] * stretched[positive], color=colors["stretched_exponential"])
    axes[0, 2].set(ylabel=r"Compensated risk $T R(T)$", title="Log-correction diagnostic")

    geometric = np.median(payload["geometric_spectrum"], axis=0)
    fit_t, fit_y = _window(times, geometric, config["geometric_fit_window"])
    periodic = periodic_power_fit(fit_t, fit_y, log_period=float(config["geometric_log_period"]))
    axes[1, 2].plot(np.log(fit_t), periodic["power_residual"], color=colors["geometric_spectrum"], alpha=0.72, label="power-law residual")
    axes[1, 2].plot(np.log(fit_t), periodic["periodic_component"], color="black", linewidth=1.1, label="fixed-period fit")
    axes[1, 2].set(xlabel=r"$\log T$", ylabel="Log-risk residual", title="Log-periodic diagnostic")
    axes[1, 2].legend(frameon=False, fontsize=8)
    for row in range(2):
        for column in range(2):
            axes[row, column].set_xlabel(r"Intrinsic time $T=\sum\eta$")
        for axis in axes[row]:
            axis.grid(alpha=0.2, which="both")
    figure.suptitle("Long-horizon counterexample discriminator · true minibatch-SGD")
    figure.savefig(output_dir / "counterexample_discriminator.png", dpi=220, bbox_inches="tight")
    figure.savefig(output_dir / "counterexample_discriminator.pdf", bbox_inches="tight")
    plt.close(figure)


def write_analysis(output_dir: Path, summary: dict[str, Any]) -> None:
    stretched = summary["metrics"]["stretched_exponential"]
    geometric = summary["metrics"]["geometric_spectrum"]
    gates = summary["gates"]
    text = f"""# Long-horizon counterexample discriminator

- Estimand: literal fresh-dense-minibatch SGD population excess risk.
- Stretched-exponential early/late exponents: {stretched['early_power_exponent']:.4f} / {stretched['late_power_exponent']:.4f}.
- Geometric local-slope peak-to-peak: {geometric['local_exponent_peak_to_peak']:.4f}.
- Geometric fixed-period residual fraction explained: {geometric['periodic_residual_fraction_explained']:.4f}.
- All preregistered pilot gates passed: {gates['all_discriminator_gates_passed']}.

This exploratory pilot is not by itself a claim-bearing confirmation. Windows and thresholds were frozen in the checked-in configuration before execution.
"""
    (output_dir / "analysis.md").write_text(text, encoding="utf-8")


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
