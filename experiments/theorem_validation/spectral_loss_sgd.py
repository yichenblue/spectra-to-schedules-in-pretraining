"""Plain-SGD pilot for classifying spectral power and non-power loss curves.

The experiment trains a linear predictor on fresh Gaussian covariates with a
prescribed diagonal covariance.  Population excess risk is evaluated exactly
from the parameter error, so validation noise cannot create or erase a power
law.  The default checked-in configuration is a pilot; ``--smoke`` is only an
implementation check and is never claim-bearing.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import csv
import json
import math
from pathlib import Path
import time
from typing import Any, Sequence

import numpy as np

from .common import (
    local_exponent_statistics,
    log_ols_exponent,
    sha256_file,
    write_csv,
    write_json,
)


Array = np.ndarray
ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG = ROOT / "spectral_loss_sgd_pilot_v001.json"
DEFAULT_OUTPUT = ROOT / "artifacts" / "spectral_loss_sgd_pilot_v001"


@dataclass(frozen=True)
class Case:
    name: str
    expected_behavior: str


CASES = (
    Case("polynomial_canonical", "pure_power"),
    Case("polynomial_sparse_target", "same_power_from_cumulative_mass"),
    Case("stretched_exponential", "power_with_log_correction"),
    Case("spectral_gap", "exponential"),
    Case("geometric_spectrum", "log_periodic_non_power"),
    Case("polynomial_rapid_target", "rapid_forcing_full_risk_to_test"),
)


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
        "steps",
        "widths",
        "seeds",
        "evaluation_points",
        "fit_intrinsic_time",
    }
    missing = sorted(required - payload.keys())
    if missing:
        raise ValueError(f"configuration misses required keys: {missing}")
    return payload


def smoke_config() -> dict[str, Any]:
    return {
        "run_id": "spectral-loss-sgd-smoke-v001",
        "stage": "smoke_non_claim_bearing",
        "primary_prediction_id": "P2_PHASE",
        "eta": 0.05,
        "batch_size": 8,
        "steps": 384,
        "widths": [128],
        "seeds": [31001, 31002],
        "evaluation_points": 49,
        "fit_intrinsic_time": [1.0, 12.0],
    }


def build_case(case: Case, width: int) -> tuple[Array, Array]:
    if width < 16:
        raise ValueError("width must be at least 16")
    index = np.arange(1, width + 1, dtype=np.float64)
    if case.name == "polynomial_canonical":
        eigenvalues = index ** (-0.8)
        target_squared = index ** (-0.6)
    elif case.name == "polynomial_sparse_target":
        eigenvalues = index ** (-0.8)
        target_squared = np.zeros(width, dtype=np.float64)
        roots = np.arange(1, int(math.isqrt(width)) + 1, dtype=np.int64)
        square_indices = roots * roots - 1
        target_squared[square_indices] = (
            index[square_indices] ** (-0.6) * (2.0 * roots - 1.0)
        )
    elif case.name == "stretched_exponential":
        eigenvalues = np.exp(-np.sqrt(index))
        eigenvalues /= eigenvalues[0]
        target_squared = index ** (-1.5)
    elif case.name == "spectral_gap":
        # Keep the gapped family independent of ambient width.  Tiny padded
        # modes are inactive and only retain a common vector representation.
        active_rank = min(64, width)
        eigenvalues = np.full(width, np.finfo(np.float64).tiny)
        eigenvalues[:active_rank] = np.geomspace(1.0, 0.1, active_rank)
        target_squared = np.zeros(width, dtype=np.float64)
        target_squared[:active_rank] = index[:active_rank] ** (-1.0)
    elif case.name == "geometric_spectrum":
        eigenvalues = np.exp(-0.35 * (index - 1.0))
        # Float64 underflows only in dynamically irrelevant far-tail modes at
        # large width.  Keeping the smallest representable positive value
        # preserves a valid covariance without changing the resolved curve.
        eigenvalues = np.maximum(eigenvalues, np.finfo(np.float64).tiny)
        target_squared = np.power(eigenvalues, 0.75)
    elif case.name == "polynomial_rapid_target":
        eigenvalues = index ** (-0.8)
        target_squared = np.exp(-index)
    else:
        raise ValueError(f"unknown case: {case.name}")

    energy = float(eigenvalues @ target_squared)
    if not math.isfinite(energy) or energy <= 0.0:
        raise FloatingPointError(f"invalid target energy for {case.name}")
    target = np.sqrt(target_squared / energy)
    if (
        not np.all(np.isfinite(eigenvalues))
        or not np.all(eigenvalues > 0.0)
        or not np.all(np.diff(eigenvalues) <= 0.0)
        or not np.all(np.isfinite(target))
        or not math.isclose(float(eigenvalues @ np.square(target)), 1.0, abs_tol=2e-13)
    ):
        raise RuntimeError(f"invalid normalized spectrum or target for {case.name}")
    return eigenvalues, target


def evaluation_steps(steps: int, count: int) -> Array:
    if steps < 2 or count < 3:
        raise ValueError("steps and evaluation count are too small")
    positive = np.unique(
        np.rint(np.geomspace(1.0, float(steps), count - 1)).astype(np.int64)
    )
    return np.concatenate((np.asarray([0], dtype=np.int64), positive))


def train_one(
    eigenvalues: Array,
    target: Array,
    *,
    eta: float,
    batch_size: int,
    steps: int,
    seed: int,
    eval_steps: Array,
) -> Array:
    # Gaussian least-squares mean-square stability includes stochastic
    # covariance fluctuations through trace(Sigma)/B.
    stability = eta * (
        (1.0 + 1.0 / batch_size) * float(eigenvalues[0])
        + float(np.sum(eigenvalues)) / batch_size
    )
    if not 0.0 < stability < 2.0:
        raise ValueError(f"unstable or invalid nominal step: {stability}")
    rng = np.random.default_rng(seed)
    error = -np.asarray(target, dtype=np.float64).copy()
    sqrt_spectrum = np.sqrt(eigenvalues)
    risks = np.empty(eval_steps.size, dtype=np.float64)
    eval_cursor = 0
    for step in range(steps + 1):
        if step == int(eval_steps[eval_cursor]):
            risks[eval_cursor] = float(eigenvalues @ np.square(error))
            eval_cursor += 1
            if eval_cursor == eval_steps.size:
                break
        samples = rng.standard_normal((batch_size, eigenvalues.size))
        samples *= sqrt_spectrum
        predictions = samples @ error
        gradient = samples.T @ predictions / float(batch_size)
        error -= eta * gradient
        if not np.all(np.isfinite(error)):
            raise FloatingPointError("SGD parameter error became nonfinite")
    if eval_cursor != eval_steps.size or np.any(risks <= 0.0):
        raise RuntimeError("evaluation trace is incomplete or nonpositive")
    return risks


def _linear_fit_r2(x: Array, log_values: Array) -> tuple[float, float]:
    design = np.column_stack((np.ones_like(x), x))
    coefficients, _, _, _ = np.linalg.lstsq(design, log_values, rcond=None)
    residual = log_values - design @ coefficients
    centered = log_values - np.mean(log_values)
    denominator = max(float(centered @ centered), 1e-300)
    return float(coefficients[1]), float(1.0 - (residual @ residual) / denominator)


def analyze_curve(times: Array, risks: Array, fit_window: tuple[float, float]) -> dict[str, Any]:
    mask = (times >= fit_window[0]) & (times <= fit_window[1]) & (risks > 0.0)
    if int(np.sum(mask)) < 7:
        raise RuntimeError("fit window contains fewer than seven evaluations")
    fit_times = times[mask]
    fit_risks = risks[mask]
    power = log_ols_exponent(fit_times, fit_risks)
    exponential_slope, exponential_r2 = _linear_fit_r2(fit_times, np.log(fit_risks))
    local = local_exponent_statistics(
        fit_times,
        fit_risks,
        half_window_decades=0.25,
        minimum_points=5,
    )
    local_times, local_values = local_exponent_curve(fit_times, fit_risks)
    log_time = np.log(fit_times)
    log_risk = np.log(fit_risks)
    design = np.column_stack((np.ones_like(log_time), log_time))
    coefficients, _, _, _ = np.linalg.lstsq(design, log_risk, rcond=None)
    power_residual = log_risk - design @ coefficients
    return {
        "fit_lower": float(fit_times[0]),
        "fit_upper": float(fit_times[-1]),
        "fit_points": int(fit_times.size),
        "power_exponent": power["exponent"],
        "power_r2": power["r2"],
        "exponential_rate": float(-exponential_slope),
        "exponential_r2": exponential_r2,
        "local_power_median": local["median"],
        "local_power_variation": local["variation"],
        "local_power_first": float(local_values[0]),
        "local_power_last": float(local_values[-1]),
        "local_power_net_drift": float(local_values[-1] - local_values[0]),
        "power_log_residual_peak_to_peak": float(np.ptp(power_residual)),
    }


def local_exponent_curve(times: Array, values: Array) -> tuple[Array, Array]:
    positive = (times > 0.0) & (values > 0.0)
    time = times[positive]
    value = values[positive]
    log_time = np.log10(time)
    centers: list[float] = []
    exponents: list[float] = []
    for index, center in enumerate(log_time):
        mask = np.abs(log_time - center) <= 0.25
        if int(np.sum(mask)) < 5:
            continue
        centers.append(float(time[index]))
        exponents.append(log_ols_exponent(time[mask], value[mask])["exponent"])
    return np.asarray(centers), np.asarray(exponents)


def run(config: dict[str, Any], output_dir: Path) -> dict[str, Any]:
    eta = float(config["eta"])
    batch_size = int(config["batch_size"])
    steps = int(config["steps"])
    widths = tuple(int(item) for item in config["widths"])
    seeds = tuple(int(item) for item in config["seeds"])
    fit_window = tuple(float(item) for item in config["fit_intrinsic_time"])
    eval_steps = evaluation_steps(steps, int(config["evaluation_points"]))
    times = eta * eval_steps.astype(np.float64)
    output_dir.mkdir(parents=True, exist_ok=True)

    began = time.monotonic()
    curve_rows: list[dict[str, Any]] = []
    metric_rows: list[dict[str, Any]] = []
    plot_payload: dict[tuple[str, int], tuple[Array, Array, Array]] = {}
    for case in CASES:
        for width in widths:
            spectrum, target = build_case(case, width)
            seed_risks = np.stack(
                [
                    train_one(
                        spectrum,
                        target,
                        eta=eta,
                        batch_size=batch_size,
                        steps=steps,
                        seed=seed,
                        eval_steps=eval_steps,
                    )
                    for seed in seeds
                ],
                axis=0,
            )
            median = np.median(seed_risks, axis=0)
            q25, q75 = np.quantile(seed_risks, [0.25, 0.75], axis=0)
            plot_payload[(case.name, width)] = (median, q25, q75)
            metrics = analyze_curve(times, median, fit_window)
            metric_rows.append(
                {
                    "case": case.name,
                    "expected_behavior": case.expected_behavior,
                    "width": width,
                    "seed_count": len(seeds),
                    **metrics,
                }
            )
            print(
                json.dumps(
                    {
                        "event": "case_complete",
                        "case": case.name,
                        "width": width,
                        "completed": len(metric_rows),
                        "total": len(CASES) * len(widths),
                        "elapsed_seconds": time.monotonic() - began,
                    }
                ),
                flush=True,
            )
            for seed_index, seed in enumerate(seeds):
                for step, intrinsic_time, risk in zip(
                    eval_steps, times, seed_risks[seed_index], strict=True
                ):
                    curve_rows.append(
                        {
                            "case": case.name,
                            "expected_behavior": case.expected_behavior,
                            "width": width,
                            "seed": seed,
                            "step": int(step),
                            "intrinsic_time": float(intrinsic_time),
                            "population_excess_risk": float(risk),
                        }
                    )

    write_csv(output_dir / "curves.csv", curve_rows)
    write_csv(output_dir / "metrics.csv", metric_rows)
    make_figures(times, widths, plot_payload, output_dir, stage=str(config["stage"]))
    summary = {
        "schema_version": "spectral_loss_sgd_v001",
        "run_id": config["run_id"],
        "stage": config["stage"],
        "primary_prediction_id": config["primary_prediction_id"],
        "claim_bearing": bool(config.get("claim_bearing", False)),
        "optimizer": "plain_sgd",
        "label_noise_variance": 0.0,
        "population_risk_evaluation": True,
        "config": config,
        "cases": [asdict(case) for case in CASES],
        "metrics": metric_rows,
        "elapsed_seconds": time.monotonic() - began,
        "source_sha256": sha256_file(Path(__file__)),
    }
    write_json(output_dir / "summary.json", summary)
    return summary


def make_figures(
    times: Array,
    widths: tuple[int, ...],
    payload: dict[tuple[str, int], tuple[Array, Array, Array]],
    output_dir: Path,
    *,
    stage: str,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 9, "axes.titlesize": 10})
    colors = plt.cm.viridis(np.linspace(0.18, 0.85, len(widths)))
    figure, axes = plt.subplots(2, 3, figsize=(11.7, 6.7), constrained_layout=True)
    positive = times > 0.0
    for axis, case in zip(axes.flat, CASES, strict=True):
        for color, width in zip(colors, widths, strict=True):
            median, q25, q75 = payload[(case.name, width)]
            axis.loglog(times[positive], median[positive], color=color, label=f"d={width}")
            if np.any(q25 != q75):
                axis.fill_between(
                    times[positive], q25[positive], q75[positive], color=color, alpha=0.16
                )
        axis.set_title(case.name.replace("_", " "))
        axis.set_xlabel(r"Intrinsic time $T=\sum\eta$")
        axis.set_ylabel("Population excess risk")
        axis.grid(alpha=0.2, which="both")
    axes[0, 0].legend(frameon=False)
    figure.suptitle(f"Plain-SGD spectral loss classification · {stage}")
    figure.savefig(output_dir / "risk_curves.png", dpi=220, bbox_inches="tight")
    figure.savefig(output_dir / "risk_curves.pdf", bbox_inches="tight")
    plt.close(figure)

    slope_figure, slope_axes = plt.subplots(
        2, 3, figsize=(11.7, 6.7), constrained_layout=True
    )
    for axis, case in zip(slope_axes.flat, CASES, strict=True):
        for color, width in zip(colors, widths, strict=True):
            median, _, _ = payload[(case.name, width)]
            slope_times, slopes = local_exponent_curve(times, median)
            axis.semilogx(slope_times, slopes, color=color, label=f"d={width}")
        axis.set_title(case.name.replace("_", " "))
        axis.set_xlabel(r"Intrinsic time $T=\sum\eta$")
        axis.set_ylabel("Local power exponent")
        axis.grid(alpha=0.2, which="both")
    slope_axes[0, 0].legend(frameon=False)
    slope_figure.suptitle("Local-slope diagnostic · drift rejects a pure power")
    slope_figure.savefig(output_dir / "local_power_slopes.png", dpi=220, bbox_inches="tight")
    slope_figure.savefig(output_dir / "local_power_slopes.pdf", bbox_inches="tight")
    plt.close(slope_figure)


def replay_figures(config: dict[str, Any], output_dir: Path) -> None:
    grouped: dict[tuple[str, int], dict[int, list[tuple[float, float]]]] = {}
    with (output_dir / "curves.csv").open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            key = (row["case"], int(row["width"]))
            grouped.setdefault(key, {}).setdefault(int(row["seed"]), []).append(
                (float(row["intrinsic_time"]), float(row["population_excess_risk"]))
            )
    widths = tuple(int(item) for item in config["widths"])
    payload: dict[tuple[str, int], tuple[Array, Array, Array]] = {}
    common_times: Array | None = None
    for key, by_seed in grouped.items():
        traces = []
        for points in by_seed.values():
            ordered = sorted(points)
            times = np.asarray([point[0] for point in ordered])
            traces.append(np.asarray([point[1] for point in ordered]))
            if common_times is None:
                common_times = times
            elif not np.array_equal(common_times, times):
                raise RuntimeError("stored curves do not share evaluation times")
        stacked = np.stack(traces)
        payload[key] = (
            np.median(stacked, axis=0),
            np.quantile(stacked, 0.25, axis=0),
            np.quantile(stacked, 0.75, axis=0),
        )
    if common_times is None:
        raise RuntimeError("no stored curves found")
    make_figures(
        common_times,
        widths,
        payload,
        output_dir,
        stage=str(config["stage"]),
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--replay-existing", action="store_true")
    args = parser.parse_args(argv)
    config = smoke_config() if args.smoke else read_config(args.config)
    output = (
        ROOT / "artifacts" / "spectral_loss_sgd_smoke_v001"
        if args.smoke and args.output_dir == DEFAULT_OUTPUT
        else args.output_dir
    )
    if args.replay_existing:
        replay_figures(config, output)
        print(json.dumps({"output": str(output), "replayed": True}))
        return 0
    summary = run(config, output)
    print(json.dumps({"output": str(output), "elapsed_seconds": summary["elapsed_seconds"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
