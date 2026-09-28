"""Extend the two polynomial constructions at smaller widths to T=235.2."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
import json
from pathlib import Path
import time
from typing import Any, Sequence

import numpy as np

from .common import sha256_file, write_csv, write_json
from .spectral_loss_sgd import CASES, build_case, evaluation_steps, train_one


Array = np.ndarray
ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG = ROOT / "spectral_polynomial_small_width_extension_v001.json"
DEFAULT_OUTPUT = ROOT / "artifacts" / "spectral_polynomial_small_width_extension_v001"
CASE_BY_NAME = {case.name: case for case in CASES}


def read_config(path: Path) -> dict[str, Any]:
    config = json.loads(path.read_text(encoding="utf-8"))
    required = {
        "run_id", "stage", "primary_prediction_id", "cases", "widths",
        "steps", "eta", "batch_size", "seeds", "evaluation_points", "workers",
    }
    missing = sorted(required - config.keys())
    if missing:
        raise ValueError(f"configuration misses required keys: {missing}")
    expected = {"polynomial_canonical", "polynomial_sparse_target"}
    if set(config["cases"]) != expected:
        raise ValueError(f"cases must be exactly {sorted(expected)}")
    return config


def _task(
    case_name: str,
    width: int,
    seed: int,
    config: dict[str, Any],
    eval_steps: Array,
) -> dict[str, Any]:
    began = time.monotonic()
    eigenvalues, target = build_case(CASE_BY_NAME[case_name], int(width))
    risks = train_one(
        eigenvalues, target,
        eta=float(config["eta"]), batch_size=int(config["batch_size"]),
        steps=int(config["steps"]), seed=int(seed), eval_steps=eval_steps,
    )
    return {
        "case": case_name, "width": int(width), "seed": int(seed),
        "risks": risks, "elapsed_seconds": time.monotonic() - began,
    }


def run(config: dict[str, Any], output_dir: Path) -> dict[str, Any]:
    cases = tuple(str(value) for value in config["cases"])
    widths = tuple(int(value) for value in config["widths"])
    seeds = tuple(int(value) for value in config["seeds"])
    steps = int(config["steps"])
    eta = float(config["eta"])
    eval_steps = evaluation_steps(steps, int(config["evaluation_points"]))
    times = eta * eval_steps.astype(float)
    began = time.monotonic()
    output_dir.mkdir(parents=True, exist_ok=True)

    results: dict[tuple[str, int, int], dict[str, Any]] = {}
    with ProcessPoolExecutor(max_workers=int(config["workers"])) as pool:
        futures = {
            pool.submit(_task, case, width, seed, config, eval_steps):
            (case, width, seed)
            for case in cases for width in widths for seed in seeds
        }
        for future in as_completed(futures):
            key = futures[future]
            results[key] = future.result()
            print(json.dumps({
                "event": "seed_complete", "case": key[0], "width": key[1],
                "seed": key[2], "complete": len(results), "total": len(futures),
                "elapsed_seconds": time.monotonic() - began,
            }), flush=True)

    rows: list[dict[str, Any]] = []
    for case in cases:
        for width in widths:
            for seed in seeds:
                for step, intrinsic_time, risk in zip(
                    eval_steps, times, results[(case, width, seed)]["risks"], strict=True
                ):
                    rows.append({
                        "case": case,
                        "expected_behavior": CASE_BY_NAME[case].expected_behavior,
                        "width": width, "seed": seed, "step": int(step),
                        "intrinsic_time": float(intrinsic_time),
                        "population_excess_risk": float(risk),
                    })
    write_csv(output_dir / "curves.csv", rows)
    summary = {
        "schema_version": "spectral_polynomial_small_width_extension_v001",
        "run_id": config["run_id"], "stage": config["stage"],
        "primary_prediction_id": config["primary_prediction_id"],
        "claim_bearing": bool(config.get("claim_bearing", False)),
        "primary_estimand": "literal_sampled_minibatch_sgd_population_excess_risk",
        "uses_expected_sgd_recurrence": False,
        "uses_spectral_binning_to_train": False,
        "cases": [asdict(CASE_BY_NAME[name]) for name in cases],
        "config": config, "elapsed_seconds": time.monotonic() - began,
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
