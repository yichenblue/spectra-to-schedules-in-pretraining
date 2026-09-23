"""Extend the stretched-exponential construction to intrinsic time 1e6.

Every update is literal fresh dense Gaussian minibatch SGD on the complete
parameter vector.  Per-seed checkpoints preserve the parameter error and the
NumPy generator state, so resuming does not alter the training tape.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import math
import os
from pathlib import Path
import time
from typing import Any, Sequence

import numpy as np

from .common import log_ols_exponent, sha256_file, write_csv, write_json
from .spectral_long_horizon_counterexamples import local_exponent_curve
from .spectral_loss_sgd import CASES, build_case, evaluation_steps


Array = np.ndarray
ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG = ROOT / "spectral_stretched_t1e6_v001.json"
DEFAULT_OUTPUT = ROOT / "artifacts" / "spectral_stretched_t1e6_v001"
CASE = next(case for case in CASES if case.name == "stretched_exponential")


def read_config(path: Path) -> dict[str, Any]:
    config = json.loads(path.read_text(encoding="utf-8"))
    required = {
        "run_id", "stage", "primary_prediction_id", "width", "eta",
        "batch_size", "steps", "seeds", "evaluation_points",
        "checkpoint_interval", "fit_windows",
        "local_half_window_decades", "workers",
    }
    missing = sorted(required - config.keys())
    if missing:
        raise ValueError(f"configuration misses required keys: {missing}")
    expected_steps = int(round(1_000_000.0 / float(config["eta"])))
    if int(config["steps"]) != expected_steps:
        raise ValueError(
            f"steps must give T_max=1e6: expected {expected_steps}, got {config['steps']}"
        )
    return config


def _save_checkpoint(
    path: Path,
    *,
    step: int,
    eval_cursor: int,
    error: Array,
    risks: Array,
    rng: np.random.Generator,
) -> None:
    temporary = path.with_suffix(".tmp.npz")
    np.savez_compressed(
        temporary,
        step=np.asarray(step, dtype=np.int64),
        eval_cursor=np.asarray(eval_cursor, dtype=np.int64),
        error=error,
        risks=risks,
        rng_state=np.asarray(json.dumps(rng.bit_generator.state)),
    )
    os.replace(temporary, path)


def train_seed(
    seed: int,
    config: dict[str, Any],
    eval_steps: Array,
    checkpoint_dir: str,
) -> dict[str, Any]:
    began = time.monotonic()
    width = int(config["width"])
    eta = float(config["eta"])
    batch_size = int(config["batch_size"])
    steps = int(config["steps"])
    checkpoint_interval = int(config["checkpoint_interval"])
    eigenvalues, target = build_case(CASE, width)
    stability = eta * (
        (1.0 + 1.0 / batch_size) * float(eigenvalues[0])
        + float(np.sum(eigenvalues)) / batch_size
    )
    if not 0.0 < stability < 2.0:
        raise ValueError(f"unstable or invalid nominal step: {stability}")

    checkpoint = Path(checkpoint_dir) / f"seed_{seed}.npz"
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    error = -np.asarray(target, dtype=np.float64).copy()
    risks = np.full(eval_steps.size, np.nan, dtype=np.float64)
    step = 0
    eval_cursor = 0
    resumed = False
    if checkpoint.exists():
        with np.load(checkpoint, allow_pickle=False) as saved:
            step = int(saved["step"])
            eval_cursor = int(saved["eval_cursor"])
            error = np.asarray(saved["error"], dtype=np.float64)
            risks = np.asarray(saved["risks"], dtype=np.float64)
            rng.bit_generator.state = json.loads(str(saved["rng_state"]))
        resumed = True

    sqrt_spectrum = np.sqrt(eigenvalues)
    while step <= steps:
        if eval_cursor < eval_steps.size and step == int(eval_steps[eval_cursor]):
            risks[eval_cursor] = float(eigenvalues @ np.square(error))
            eval_cursor += 1
        if step == steps:
            break
        samples = rng.standard_normal((batch_size, width))
        samples *= sqrt_spectrum
        predictions = samples @ error
        gradient = samples.T @ predictions / float(batch_size)
        error -= eta * gradient
        if not np.all(np.isfinite(error)):
            raise FloatingPointError(f"seed {seed}: SGD parameter error became nonfinite")
        step += 1
        if step % checkpoint_interval == 0:
            _save_checkpoint(
                checkpoint, step=step, eval_cursor=eval_cursor,
                error=error, risks=risks, rng=rng,
            )
            print(json.dumps({
                "event": "checkpoint", "seed": seed, "step": step,
                "intrinsic_time": eta * step,
                "elapsed_seconds": time.monotonic() - began,
            }), flush=True)

    if eval_cursor != eval_steps.size or np.any(~np.isfinite(risks)) or np.any(risks <= 0.0):
        raise RuntimeError(f"seed {seed}: evaluation trace is incomplete or nonpositive")
    _save_checkpoint(
        checkpoint, step=step, eval_cursor=eval_cursor,
        error=error, risks=risks, rng=rng,
    )
    return {
        "seed": seed,
        "risks": risks,
        "elapsed_seconds": time.monotonic() - began,
        "resumed": resumed,
    }


def run(config: dict[str, Any], output_dir: Path) -> dict[str, Any]:
    eta = float(config["eta"])
    steps = int(config["steps"])
    seeds = tuple(int(seed) for seed in config["seeds"])
    eval_steps = evaluation_steps(steps, int(config["evaluation_points"]))
    times = eta * eval_steps.astype(float)
    output_dir.mkdir(parents=True, exist_ok=True)
    began = time.monotonic()

    results: dict[int, dict[str, Any]] = {}
    with ProcessPoolExecutor(max_workers=min(int(config["workers"]), len(seeds))) as pool:
        futures = {
            pool.submit(
                train_seed, seed, config, eval_steps,
                str(output_dir / "checkpoints"),
            ): seed
            for seed in seeds
        }
        for future in as_completed(futures):
            result = future.result()
            results[int(result["seed"])] = result
            print(json.dumps({
                "event": "seed_complete", "seed": result["seed"],
                "seed_count_complete": len(results), "seed_count": len(seeds),
                "elapsed_seconds": time.monotonic() - began,
            }), flush=True)

    traces = np.stack([results[seed]["risks"] for seed in seeds])
    median = np.median(traces, axis=0)
    curve_rows = []
    for seed_index, seed in enumerate(seeds):
        for step, intrinsic_time, risk in zip(eval_steps, times, traces[seed_index], strict=True):
            curve_rows.append({
                "case": CASE.name, "width": int(config["width"]), "seed": seed,
                "step": int(step), "intrinsic_time": float(intrinsic_time),
                "population_excess_risk": float(risk),
            })
    write_csv(output_dir / "curves.csv", curve_rows)

    fits: dict[str, Any] = {}
    for name, bounds in config["fit_windows"].items():
        mask = (times >= float(bounds[0])) & (times <= float(bounds[1]))
        if int(np.sum(mask)) < 7:
            raise RuntimeError(f"fit window {name} has fewer than seven points")
        fits[name] = {
            **log_ols_exponent(times[mask], median[mask]),
            "lower": float(times[mask][0]),
            "upper": float(times[mask][-1]),
            "points": int(np.sum(mask)),
        }
    local_t, local_q = local_exponent_curve(
        times, median,
        half_window_decades=float(config["local_half_window_decades"]),
    )
    metrics = {
        "fits": fits,
        "endpoint_local_exponent": float(local_q[-1]),
        "endpoint_local_time": float(local_t[-1]),
        "per_seed_elapsed_seconds": {
            str(seed): float(results[seed]["elapsed_seconds"]) for seed in seeds
        },
    }
    make_figure(times, traces, local_t, local_q, output_dir)
    summary = {
        "schema_version": "spectral_stretched_t1e6_v001",
        "run_id": config["run_id"], "stage": config["stage"],
        "primary_prediction_id": config["primary_prediction_id"],
        "claim_bearing": bool(config.get("claim_bearing", False)),
        "primary_estimand": "literal_sampled_minibatch_sgd_population_excess_risk",
        "uses_expected_sgd_recurrence": False,
        "uses_spectral_binning_to_train": False,
        "config": config, "metrics": metrics,
        "elapsed_seconds": time.monotonic() - began,
        "source_sha256": sha256_file(Path(__file__)),
        "config_sha256": sha256_file(DEFAULT_CONFIG),
    }
    write_json(output_dir / "summary.json", summary)
    return summary


def make_figure(
    times: Array,
    traces: Array,
    local_t: Array,
    local_q: Array,
    output_dir: Path,
) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    positive = times > 0.0
    median = np.median(traces, axis=0)
    q25, q75 = np.quantile(traces, [0.25, 0.75], axis=0)
    figure, axes = plt.subplots(1, 3, figsize=(13.2, 3.8), constrained_layout=True)
    axes[0].loglog(times[positive], median[positive], color="#F8961E", linewidth=2.2)
    axes[0].fill_between(times[positive], q25[positive], q75[positive], color="#F8961E", alpha=0.16)
    axes[0].set(xlabel=r"Intrinsic time $T=\sum\eta$", ylabel="Risk", title="Stretched-exponential spectrum")
    axes[1].semilogx(local_t, local_q, color="#F8961E", linewidth=2.2)
    axes[1].axhline(1.0, color="black", linestyle="--", linewidth=1.5)
    axes[1].set(xlabel=r"Intrinsic time $T=\sum\eta$", ylabel=r"Local exponent $q(T)$", title="Slope drift toward 1")
    axes[2].semilogx(times[positive], times[positive] * median[positive], color="#F8961E", linewidth=2.2)
    axes[2].set(xlabel=r"Intrinsic time $T=\sum\eta$", ylabel=r"$T R(T)$", title="Log-correction diagnostic")
    for axis in axes:
        axis.grid(alpha=0.2, which="both")
    figure.savefig(output_dir / "stretched_t1e6.png", dpi=220, bbox_inches="tight")
    figure.savefig(output_dir / "stretched_t1e6.pdf", bbox_inches="tight")
    plt.close(figure)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    config = read_config(args.config)
    summary = run(config, args.output_dir)
    print(json.dumps({
        "output": str(args.output_dir),
        "endpoint_local_exponent": summary["metrics"]["endpoint_local_exponent"],
    }))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
