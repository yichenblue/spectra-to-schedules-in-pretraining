"""True sampled minibatch-SGD width/time sweep over all six spectral cases.

Every update draws a fresh dense Gaussian minibatch, forms its empirical
gradient, and updates the complete parameter-error vector. No modal forcing,
Volterra recursion, spectral binning, or expected-risk proxy is used.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import time
from typing import Any, Sequence

import numpy as np

from .common import sha256_file, write_csv, write_json
from .spectral_loss_sgd import (
    CASES,
    analyze_curve,
    build_case,
    evaluation_steps,
    local_exponent_curve,
    train_one,
)


Array = np.ndarray
ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG = ROOT / "spectral_loss_sgd_width_time_b0_v001.json"
DEFAULT_OUTPUT = ROOT / "artifacts" / "spectral_loss_sgd_width_time_b0_v001"


def read_config(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    required = {
        "run_id", "stage", "primary_prediction_id", "eta", "batch_size",
        "width_steps", "seeds", "evaluation_points", "fit_intrinsic_time",
    }
    missing = sorted(required - payload.keys())
    if missing:
        raise ValueError(f"configuration misses required keys: {missing}")
    pairs = payload["width_steps"]
    if not pairs or any(int(row["width"]) < 16 or int(row["steps"]) < 2 for row in pairs):
        raise ValueError("width_steps must contain valid width/step pairs")
    return payload


def run(config: dict[str, Any], output_dir: Path) -> dict[str, Any]:
    eta = float(config["eta"])
    batch_size = int(config["batch_size"])
    seeds = tuple(int(seed) for seed in config["seeds"])
    width_steps = tuple(
        (int(row["width"]), int(row["steps"])) for row in config["width_steps"]
    )
    fit_window = tuple(float(value) for value in config["fit_intrinsic_time"])
    output_dir.mkdir(parents=True, exist_ok=True)
    began = time.monotonic()
    curve_rows: list[dict[str, Any]] = []
    metric_rows: list[dict[str, Any]] = []
    plot_payload: dict[tuple[str, int], tuple[Array, Array, Array, Array]] = {}
    total_updates = 0
    total_examples = 0

    for case in CASES:
        for width, steps in width_steps:
            spectrum, target = build_case(case, width)
            eval_steps = evaluation_steps(steps, int(config["evaluation_points"]))
            times = eta * eval_steps.astype(np.float64)
            seed_risks = np.stack([
                train_one(
                    spectrum, target, eta=eta, batch_size=batch_size,
                    steps=steps, seed=seed, eval_steps=eval_steps,
                )
                for seed in seeds
            ])
            total_updates += steps * len(seeds)
            total_examples += steps * batch_size * len(seeds)
            median = np.median(seed_risks, axis=0)
            q25, q75 = np.quantile(seed_risks, [0.25, 0.75], axis=0)
            plot_payload[(case.name, width)] = (times, median, q25, q75)
            metric_rows.append({
                "case": case.name,
                "expected_behavior": case.expected_behavior,
                "width": width,
                "steps": steps,
                "max_intrinsic_time": float(times[-1]),
                "seed_count": len(seeds),
                **analyze_curve(times, median, fit_window),
            })
            for seed_index, seed in enumerate(seeds):
                for step, intrinsic_time, risk in zip(
                    eval_steps, times, seed_risks[seed_index], strict=True
                ):
                    curve_rows.append({
                        "case": case.name,
                        "expected_behavior": case.expected_behavior,
                        "width": width,
                        "seed": seed,
                        "step": int(step),
                        "intrinsic_time": float(intrinsic_time),
                        "population_excess_risk": float(risk),
                    })
            print(json.dumps({
                "event": "sampled_case_width_complete",
                "case": case.name,
                "width": width,
                "steps": steps,
                "elapsed_seconds": time.monotonic() - began,
            }), flush=True)

    write_csv(output_dir / "curves.csv", curve_rows)
    write_csv(output_dir / "metrics.csv", metric_rows)
    make_figures(plot_payload, width_steps, output_dir)
    summary = {
        "schema_version": "spectral_loss_sgd_width_time_v001",
        "run_id": config["run_id"],
        "stage": config["stage"],
        "primary_prediction_id": config["primary_prediction_id"],
        "claim_bearing": bool(config.get("claim_bearing", False)),
        "estimand": "sampled_complete_population_excess_risk",
        "training_mode": "fresh_dense_gaussian_minibatches_and_explicit_parameter_updates",
        "uses_analytic_recurrence": False,
        "uses_spectral_binning": False,
        "optimizer": "plain_sgd",
        "cases": [asdict(case) for case in CASES],
        "config": config,
        "metrics": metric_rows,
        "total_sampled_updates": total_updates,
        "total_sampled_examples": total_examples,
        "elapsed_seconds": time.monotonic() - began,
        "source_sha256": sha256_file(Path(__file__)),
    }
    write_json(output_dir / "summary.json", summary)
    return summary


def make_figures(
    payload: dict[tuple[str, int], tuple[Array, Array, Array, Array]],
    width_steps: tuple[tuple[int, int], ...], output_dir: Path,
) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = plt.cm.viridis(np.linspace(0.15, 0.88, len(width_steps)))
    figure, axes = plt.subplots(2, 3, figsize=(11.7, 6.7), constrained_layout=True)
    for axis, case in zip(axes.flat, CASES, strict=True):
        for color, (width, _) in zip(colors, width_steps, strict=True):
            times, median, q25, q75 = payload[(case.name, width)]
            positive = times > 0.0
            axis.loglog(times[positive], median[positive], color=color, label=f"d={width:,}")
            axis.fill_between(times[positive], q25[positive], q75[positive], color=color, alpha=0.16)
        axis.set_title(case.name.replace("_", " "))
        axis.set_xlabel(r"Intrinsic time $T=\sum\eta$")
        axis.set_ylabel("Population excess risk")
        axis.grid(alpha=0.2, which="both")
    axes[0, 0].legend(frameon=False)
    figure.suptitle("True sampled minibatch-SGD · all six spectral constructions")
    figure.savefig(output_dir / "risk_curves.png", dpi=220, bbox_inches="tight")
    figure.savefig(output_dir / "risk_curves.pdf", bbox_inches="tight")
    plt.close(figure)

    slope_figure, slope_axes = plt.subplots(2, 3, figsize=(11.7, 6.7), constrained_layout=True)
    for axis, case in zip(slope_axes.flat, CASES, strict=True):
        for color, (width, _) in zip(colors, width_steps, strict=True):
            times, median, _, _ = payload[(case.name, width)]
            slope_times, slopes = local_exponent_curve(times, median)
            axis.semilogx(slope_times, slopes, color=color, label=f"d={width:,}")
        axis.set_title(case.name.replace("_", " "))
        axis.set_xlabel(r"Intrinsic time $T=\sum\eta$")
        axis.set_ylabel("Local power exponent")
        axis.grid(alpha=0.2, which="both")
    slope_axes[0, 0].legend(frameon=False)
    slope_figure.suptitle("True sampled minibatch-SGD · local-slope diagnostic")
    slope_figure.savefig(output_dir / "local_power_slopes.png", dpi=220, bbox_inches="tight")
    slope_figure.savefig(output_dir / "local_power_slopes.pdf", bbox_inches="tight")
    plt.close(slope_figure)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    summary = run(read_config(args.config), args.output_dir)
    print(json.dumps({"output": str(args.output_dir), "elapsed_seconds": summary["elapsed_seconds"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
