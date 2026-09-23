"""Generate only the missing literal-SGD widths for the six-panel figure."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
from pathlib import Path
import time
from typing import Any, Sequence

import numpy as np

from .common import sha256_file, write_csv, write_json
from .spectral_geometric_spacing_sweep import build_geometric_case
from .spectral_loss_sgd import evaluation_steps, train_one
from .spectral_stretched_t1e6 import train_seed


Array = np.ndarray
ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG = ROOT / "spectral_missing_width_overlays_v001.json"
DEFAULT_OUTPUT = ROOT / "artifacts" / "spectral_missing_width_overlays_v001"


def read_config(path: Path) -> dict[str, Any]:
    config = json.loads(path.read_text(encoding="utf-8"))
    required = {"run_id", "stage", "primary_prediction_id", "workers", "stretched", "geometric"}
    missing = sorted(required - config.keys())
    if missing:
        raise ValueError(f"configuration misses required keys: {missing}")
    stretched = config["stretched"]
    expected_steps = int(round(1_000_000.0 / float(stretched["eta"])))
    if int(stretched["steps"]) != expected_steps:
        raise ValueError("stretched section must end at intrinsic time 1e6")
    return config


def _stretched_task(
    width: int,
    seed: int,
    section: dict[str, Any],
    eval_steps: Array,
    output_dir: str,
) -> dict[str, Any]:
    task_config = dict(section)
    task_config["width"] = int(width)
    result = train_seed(
        int(seed), task_config, eval_steps,
        str(Path(output_dir) / "checkpoints" / f"d{width}"),
    )
    result["width"] = int(width)
    return result


def _geometric_task(
    width: int,
    seed: int,
    section: dict[str, Any],
    eval_steps: Array,
) -> dict[str, Any]:
    began = time.monotonic()
    eigenvalues, target = build_geometric_case(int(width), float(section["spacing"]))
    risks = train_one(
        eigenvalues, target,
        eta=float(section["eta"]),
        batch_size=int(section["batch_size"]),
        steps=int(section["steps"]),
        seed=int(seed),
        eval_steps=eval_steps,
    )
    return {
        "width": int(width), "seed": int(seed), "risks": risks,
        "elapsed_seconds": time.monotonic() - began,
    }


def _curve_rows(
    results: dict[tuple[int, int], dict[str, Any]],
    widths: tuple[int, ...],
    seeds: tuple[int, ...],
    eval_steps: Array,
    eta: float,
    extra: dict[str, Any],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    times = eta * eval_steps.astype(float)
    for width in widths:
        for seed in seeds:
            for step, intrinsic_time, risk in zip(
                eval_steps, times, results[(width, seed)]["risks"], strict=True
            ):
                rows.append({
                    **extra, "width": width, "seed": seed, "step": int(step),
                    "intrinsic_time": float(intrinsic_time),
                    "population_excess_risk": float(risk),
                })
    return rows


def run(config: dict[str, Any], output_dir: Path) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    began = time.monotonic()
    worker_count = int(config["workers"])

    stretched = config["stretched"]
    stretched_widths = tuple(int(value) for value in stretched["widths"])
    stretched_seeds = tuple(int(value) for value in stretched["seeds"])
    stretched_eval = evaluation_steps(
        int(stretched["steps"]), int(stretched["evaluation_points"])
    )
    stretched_results: dict[tuple[int, int], dict[str, Any]] = {}
    with ProcessPoolExecutor(max_workers=worker_count) as pool:
        futures = {
            pool.submit(
                _stretched_task, width, seed, stretched, stretched_eval,
                str(output_dir),
            ): (width, seed)
            for width in stretched_widths for seed in stretched_seeds
        }
        for future in as_completed(futures):
            width, seed = futures[future]
            stretched_results[(width, seed)] = future.result()
            print(json.dumps({
                "event": "stretched_seed_complete", "width": width, "seed": seed,
                "complete": len(stretched_results), "total": len(futures),
                "elapsed_seconds": time.monotonic() - began,
            }), flush=True)
    write_csv(
        output_dir / "stretched_curves.csv",
        _curve_rows(
            stretched_results, stretched_widths, stretched_seeds,
            stretched_eval, float(stretched["eta"]),
            {"case": "stretched_exponential"},
        ),
    )

    geometric = config["geometric"]
    geometric_widths = tuple(int(value) for value in geometric["widths"])
    geometric_seeds = tuple(int(value) for value in geometric["seeds"])
    geometric_eval = evaluation_steps(
        int(geometric["steps"]), int(geometric["evaluation_points"])
    )
    geometric_results: dict[tuple[int, int], dict[str, Any]] = {}
    with ProcessPoolExecutor(max_workers=worker_count) as pool:
        futures = {
            pool.submit(
                _geometric_task, width, seed, geometric, geometric_eval,
            ): (width, seed)
            for width in geometric_widths for seed in geometric_seeds
        }
        for future in as_completed(futures):
            width, seed = futures[future]
            geometric_results[(width, seed)] = future.result()
            print(json.dumps({
                "event": "geometric_seed_complete", "width": width, "seed": seed,
                "complete": len(geometric_results), "total": len(futures),
                "elapsed_seconds": time.monotonic() - began,
            }), flush=True)
    write_csv(
        output_dir / "geometric_curves.csv",
        _curve_rows(
            geometric_results, geometric_widths, geometric_seeds,
            geometric_eval, float(geometric["eta"]),
            {"case": "geometric_spectrum", "spacing": float(geometric["spacing"])},
        ),
    )

    summary = {
        "schema_version": "spectral_missing_width_overlays_v001",
        "run_id": config["run_id"], "stage": config["stage"],
        "primary_prediction_id": config["primary_prediction_id"],
        "claim_bearing": bool(config.get("claim_bearing", False)),
        "primary_estimand": "literal_sampled_minibatch_sgd_population_excess_risk",
        "uses_expected_sgd_recurrence": False,
        "uses_spectral_binning_to_train": False,
        "config": config,
        "elapsed_seconds": time.monotonic() - began,
        "source_sha256": sha256_file(Path(__file__)),
        "config_sha256": sha256_file(DEFAULT_CONFIG),
    }
    write_json(output_dir / "summary.json", summary)
    return summary


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
