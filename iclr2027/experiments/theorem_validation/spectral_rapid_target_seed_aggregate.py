"""Aggregate the frozen 20-seed rapid-target replication.

This module never trains a model or modifies a seed artifact.  It requires the
exact seed set 31501--31520, recomputes each log--log exponent on the frozen
intrinsic-time window [100, 800], and reports the literal T=1000 endpoints.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .common import log_ols_exponent, sha256_file, write_csv, write_json


Array = np.ndarray
ROOT = Path(__file__).resolve().parent
ARTIFACTS = ROOT / "artifacts"
DEFAULT_OUTPUT = ARTIFACTS / "spectral_rapid_target_t1e3_w524288_20seed_v001"
EXPECTED_SEEDS = tuple(range(31501, 31521))
FIT_WINDOW = (100.0, 800.0)
ENDPOINT_TIME = 1000.0
T_975_DF19 = 2.093024054408263


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _source_paths(seed: int, artifacts: Path) -> dict[str, Path]:
    stem = f"spectral_rapid_target_t1e3_w524288_seed{seed}_v001"
    directory = artifacts / stem
    return {
        "config": ROOT / f"{stem}.json",
        "curves": directory / "curves.csv",
        "metrics": directory / "metrics.csv",
        "bands": directory / "spectral_band_risks.csv",
        "summary": directory / "summary.json",
    }


def _validate_config(config: dict[str, Any], seed: int) -> None:
    expected = {
        "stage": "exploratory_width_time_extension",
        "primary_prediction_id": "P2_PHASE",
        "claim_bearing": False,
        "width": 524_288,
        "steps": 20_000,
        "eta": 0.05,
        "batch_sizes": [16],
        "seeds": [seed],
        "evaluation_points": 321,
        "tail_fit_window": [100.0, 800.0],
        "forcing_transform_fit_window": [4.0, 64.0],
        "predicted_forcing_transform_exponent": 5.0 / 9.0,
        "forcing_transform_tolerance": 0.08,
        "spectral_band_edges": [
            0, 4, 16, 64, 256, 1024, 4096, 16384, 65536, 262144, 524288,
        ],
    }
    for key, value in expected.items():
        _require(config.get(key) == value, f"seed {seed}: unexpected config field {key}")
    _require(
        config.get("run_id") == f"spectral-rapid-target-t1e3-w524288-seed{seed}-v001",
        f"seed {seed}: unexpected run_id",
    )


def _load_seed(seed: int, artifacts: Path) -> tuple[dict[str, Any], Array]:
    paths = _source_paths(seed, artifacts)
    for name, path in paths.items():
        _require(path.is_file(), f"seed {seed}: missing {name} file {path}")

    config = _read_json(paths["config"])
    summary = _read_json(paths["summary"])
    _validate_config(config, seed)
    _require(summary.get("config") == config, f"seed {seed}: config/summary mismatch")
    _require(
        summary.get("schema_version") == "spectral_rapid_target_mechanism_v001",
        f"seed {seed}: unexpected summary schema",
    )
    _require(summary.get("uses_expected_sgd_recurrence") is False, f"seed {seed}: proxy SGD")
    _require(summary.get("uses_spectral_binning_to_train") is False, f"seed {seed}: binned training")

    rows = [
        row for row in _read_csv(paths["curves"])
        if row["condition"] == "sampled_minibatch_sgd"
    ]
    _require(rows, f"seed {seed}: no sampled-minibatch rows")
    _require(
        all(int(row["batch_size"]) == 16 and int(row["seed"]) == seed for row in rows),
        f"seed {seed}: curve identity mismatch",
    )
    times = np.asarray([float(row["intrinsic_time"]) for row in rows], dtype=float)
    risks = np.asarray([float(row["population_risk"]) for row in rows], dtype=float)
    _require(np.all(np.diff(times) > 0.0), f"seed {seed}: time grid is not strictly increasing")
    _require(np.all(np.isfinite(risks)) and np.all(risks > 0.0), f"seed {seed}: invalid risk")
    _require(math.isclose(float(times[-1]), ENDPOINT_TIME), f"seed {seed}: no T=1000 endpoint")

    fit_mask = (times >= FIT_WINDOW[0]) & (times <= FIT_WINDOW[1])
    fit = log_ols_exponent(times[fit_mask], risks[fit_mask])
    endpoint_indices = np.flatnonzero(np.isclose(times, ENDPOINT_TIME, rtol=0.0, atol=1.0e-12))
    _require(endpoint_indices.size == 1, f"seed {seed}: T=1000 endpoint is not unique")

    metric_rows = _read_csv(paths["metrics"])
    _require(len(metric_rows) == 1, f"seed {seed}: expected one metrics row")
    stored_exponent = float(metric_rows[0]["tail_power_exponent"])
    _require(
        math.isclose(float(fit["exponent"]), stored_exponent, rel_tol=0.0, abs_tol=5.0e-13),
        f"seed {seed}: recomputed/stored exponent mismatch",
    )

    gates = summary.get("gates", {})
    record = {
        "seed": seed,
        "run_id": config["run_id"],
        "fit_requested_lower": FIT_WINDOW[0],
        "fit_requested_upper": FIT_WINDOW[1],
        "fit_actual_lower": float(times[fit_mask][0]),
        "fit_actual_upper": float(times[fit_mask][-1]),
        "fit_points": int(np.sum(fit_mask)),
        "exponent": float(fit["exponent"]),
        "exponent_r2": float(fit["r2"]),
        "risk_t1000": float(risks[int(endpoint_indices[0])]),
        "all_primary_gates_passed": bool(gates.get("all_primary_gates_passed", False)),
        "all_sampled_values_finite_positive": bool(
            gates.get("all_sampled_values_finite_positive", False)
        ),
        "forcing_transform_gate_passed": bool(gates.get("forcing_transform_gate_passed", False)),
        "tail_risk_monotone_with_batch": bool(gates.get("tail_risk_monotone_with_batch", False)),
        "training_source_sha256": str(summary.get("source_sha256", "")),
        "config_sha256": sha256_file(paths["config"]),
        "curves_sha256": sha256_file(paths["curves"]),
        "summary_sha256": sha256_file(paths["summary"]),
    }
    return {"record": record, "summary": summary}, times


def _statistics(values: Array) -> dict[str, Any]:
    data = np.asarray(values, dtype=float)
    _require(data.shape == (20,), "the frozen Student-t summary requires exactly 20 seeds")
    mean = float(np.mean(data))
    sample_sd = float(np.std(data, ddof=1))
    half_width = T_975_DF19 * sample_sd / math.sqrt(data.size)
    return {
        "n": int(data.size),
        "median": float(np.median(data)),
        "mean": mean,
        "sample_sd": sample_sd,
        "coefficient_of_variation": sample_sd / abs(mean),
        "minimum": float(np.min(data)),
        "maximum": float(np.max(data)),
        "mean_student_t_95_ci": [mean - half_width, mean + half_width],
        "ci_method": "two-sided Student-t interval for the across-seed mean, df=19",
    }


def run(artifacts: Path, output: Path) -> dict[str, Any]:
    payloads: list[dict[str, Any]] = []
    common_times: Array | None = None
    for seed in EXPECTED_SEEDS:
        payload, times = _load_seed(seed, artifacts)
        if common_times is None:
            common_times = times
        else:
            _require(np.array_equal(times, common_times), f"seed {seed}: evaluation-grid mismatch")
        payloads.append(payload)

    records = [payload["record"] for payload in payloads]
    exponents = np.asarray([row["exponent"] for row in records], dtype=float)
    endpoints = np.asarray([row["risk_t1000"] for row in records], dtype=float)
    exponent_statistics = _statistics(exponents)
    endpoint_statistics = _statistics(endpoints)
    reference_exponent = 0.75

    gate_names = (
        "all_primary_gates_passed",
        "all_sampled_values_finite_positive",
        "forcing_transform_gate_passed",
        "tail_risk_monotone_with_batch",
    )
    gate_counts = {
        name: sum(bool(row[name]) for row in records) for name in gate_names
    }
    unique_forcing_controls = {
        json.dumps(payload["summary"].get("forcing_control", {}), sort_keys=True)
        for payload in payloads
    }
    source_hashes = {str(row["training_source_sha256"]) for row in records}
    summary = {
        "schema_version": "spectral_rapid_target_seed_aggregate_v001",
        "is_new_training_run": False,
        "source_artifacts_are_modified": False,
        "expected_seeds": list(EXPECTED_SEEDS),
        "fit_window_requested": list(FIT_WINDOW),
        "fit_grid_realized": {
            "lower": records[0]["fit_actual_lower"],
            "upper": records[0]["fit_actual_upper"],
            "points": records[0]["fit_points"],
        },
        "exponent_statistics": exponent_statistics,
        "endpoint_t1000_statistics": endpoint_statistics,
        "reference_exponent_diagnostic": {
            "reference": reference_exponent,
            "mean_minus_reference": exponent_statistics["mean"] - reference_exponent,
            "median_minus_reference": exponent_statistics["median"] - reference_exponent,
            "reference_inside_mean_95_ci": (
                exponent_statistics["mean_student_t_95_ci"][0]
                <= reference_exponent
                <= exponent_statistics["mean_student_t_95_ci"][1]
            ),
            "is_preregistered_acceptance_gate": False,
        },
        "gate_audit": {
            "complete_exact_seed_set": True,
            "frozen_configs_match": True,
            "shared_evaluation_grid": True,
            "recomputed_exponents_match_seed_metrics": True,
            "training_source_hash_consistent": len(source_hashes) == 1,
            "source_gate_pass_counts_out_of_20": gate_counts,
            "forcing_control_unique_payload_count": len(unique_forcing_controls),
            "tail_risk_monotone_with_batch_is_informative": False,
            "tail_risk_monotone_with_batch_note": (
                "Each shard contains only B=16, so this source gate is vacuous."
            ),
            "forcing_control_replication_note": (
                "The forcing control is deterministic and repeated in each shard; it is one unique control, "
                "not 20 independent confirmations."
            ),
        },
        "scope": {
            "stage": "exploratory_width_time_extension",
            "claim_bearing": False,
            "primary_prediction_id": "P2_PHASE",
        },
        "finite_width_caveat": (
            "All 20 seeds use only d=524288. Replication estimates sampled-SGD seed variability "
            "conditional on that width; it does not estimate finite-width bias or establish an "
            "asymptotic exponent. The width was pre-calibrated so the exact memory local slope "
            "remains within 0.75 +/- 0.10 through T=1000, but a multi-width check is still needed "
            "for a width-robust claim."
        ),
        "sources": records,
    }

    output.mkdir(parents=True, exist_ok=True)
    write_csv(output / "seed_metrics.csv", records)
    write_json(output / "summary.json", summary)
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts", type=Path, default=ARTIFACTS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    summary = run(args.artifacts, args.output)
    print(json.dumps({
        "output": str(args.output),
        "seed_count": summary["exponent_statistics"]["n"],
        "mean_exponent": summary["exponent_statistics"]["mean"],
    }))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
