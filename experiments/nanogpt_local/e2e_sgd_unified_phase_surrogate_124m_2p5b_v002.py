"""Fit the seven-parameter ordinary-memory surrogate to 124M SGD v002.

The fit uses the completed fixed-batch 8-1-1 trajectory (shared prefix plus
tail) over the previously frozen terminal intrinsic-time decade.  The matched
fixed-learning-rate / batch-schedule trajectory is evaluated with no parameter
refit.  This is a post-hoc, single-seed A2_EXTERNAL diagnostic; it is not an
independent impulse-response estimate of the paper's memory kernel.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import csv
import json
import math
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import least_squares

from experiments.nanogpt_local import e2e_sgd_im_qk_profile_v001 as fixed_q
from experiments.nanogpt_local import e2e_sgd_phase_surrogate_transfer_124m_2p5b_v001 as transfer
from experiments.nanogpt_local import e2e_sgd_phase_surrogate_transfer_v001 as phase
from experiments.nanogpt_local import e2e_sgd_rpath_factorization_124m_2p5b_analysis_v002 as validator
from experiments.nanogpt_local import e2e_sgd_rpath_factorization_124m_2p5b_v002 as core


Json = dict[str, Any]
ANALYSIS_ID = "nanogpt124m-e2e-sgd-unified-phase-surrogate-v002"
Q_K_BOUNDS = (0.02, 40.0)
Q_F_BOUNDS = (0.02, 8.0)
LOG_C_BOUNDS = (-14.0, 14.0)
BOUNDARY_TOLERANCE = 0.05
FIT_WINDOW_RULE = (
    "last evaluation at or below one tenth of terminal 8-1-1 intrinsic time, "
    "through the terminal evaluation"
)
FIT_ARM = "eight_one_one__fixed_batch_lr"
TRANSFER_ARM = "eight_one_one__fixed_lr_batch"


def _canonical_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty CSV: {path}")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


@contextmanager
def _use_v002_schedule_core() -> Iterator[None]:
    previous = phase.core
    phase._macro_quadrature.cache_clear()
    phase._causal_grid.cache_clear()
    phase.core = core
    try:
        yield
    finally:
        phase._macro_quadrature.cache_clear()
        phase._causal_grid.cache_clear()
        phase.core = previous


def _load_arrays(directory: Path) -> tuple[Json, dict[str, np.ndarray]]:
    report = validator.validate_result_directory(directory)
    if report["status"] != "passed":
        raise validator.AnalysisError(f"source validation failed: {directory}")
    with np.load(directory / core.EVALUATION_FILENAME, allow_pickle=False) as payload:
        arrays = {
            name: np.ascontiguousarray(payload[name])
            for name in (
                "macro_index",
                "global_update",
                "intrinsic_time",
                "validation_cross_entropy",
                "context_cross_entropy",
            )
        }
    return report, arrays


def _combine_prefix_tail(prefix_directory: Path, tail_directory: Path) -> tuple[phase.EvaluationCurve, Json]:
    prefix_report, prefix = _load_arrays(prefix_directory)
    tail_report, tail = _load_arrays(tail_directory)
    if prefix_report["phase"] != "prefix" or tail_report["phase"] != "tail":
        raise validator.AnalysisError("expected one prefix and one tail result")
    linkage_fields = (
        "campaign_contract_sha256",
        "numerics_id",
        "data_identity_sha256",
        "schedule_manifest_sha256",
        "resource_calibration_sha256",
        "prefix_checkpoint_sha256",
        "prefix_model_state_sha256",
        "prefix_validation_anchor_sha256",
    )
    if any(prefix_report[name] != tail_report[name] for name in linkage_fields):
        raise validator.AnalysisError("prefix/tail lineage differs")
    for name in (
        "macro_index",
        "global_update",
        "intrinsic_time",
        "validation_cross_entropy",
        "context_cross_entropy",
    ):
        if not np.array_equal(prefix[name][-1], tail[name][0]):
            raise validator.AnalysisError(f"prefix/tail anchor differs: {name}")

    def concatenate(name: str) -> np.ndarray:
        return np.concatenate([prefix[name][:-1], tail[name]], axis=0)

    contexts = np.asarray(concatenate("context_cross_entropy"), dtype=np.float64)
    standard_error = np.std(contexts, axis=1, ddof=1) / math.sqrt(core.VALIDATION_CONTEXTS)
    arm_id = str(tail_report["arm_id"])
    shape_id, _ = arm_id.split("__", 1)
    curve = phase.EvaluationCurve(
        run_id=core.CAMPAIGN_ID,
        arm_id=arm_id,
        shape_id=shape_id,
        update=np.asarray(concatenate("global_update"), dtype=np.int64),
        macro_index=np.asarray(concatenate("macro_index"), dtype=np.int64),
        intrinsic_time=np.asarray(concatenate("intrinsic_time"), dtype=np.float64),
        loss=np.asarray(concatenate("validation_cross_entropy"), dtype=np.float64),
        standard_error=np.asarray(standard_error, dtype=np.float64),
    )
    if not np.all(np.diff(curve.intrinsic_time) > 0.0):
        raise validator.AnalysisError("combined validation clock is not strict")
    return curve, {"prefix": prefix_report, "tail": tail_report}


def _fit_mask(curve: phase.EvaluationCurve) -> np.ndarray:
    positive = np.flatnonzero(curve.intrinsic_time > 0.0)
    threshold = float(curve.intrinsic_time[-1] / 10.0)
    eligible = positive[curve.intrinsic_time[positive] <= threshold]
    if eligible.size == 0:
        raise ValueError("8-1-1 curve does not expose a full intrinsic-time decade")
    return np.arange(curve.loss.size) >= int(eligible[-1])


def _fit_unified(
    curve: phase.EvaluationCurve,
    mask: np.ndarray,
    q_f_bounds: tuple[float, float] = Q_F_BOUNDS,
    noise_amplitude_upper: float = 1e5,
    objective_sqrt_weights: np.ndarray | None = None,
) -> Json:
    evaluator = transfer._FftFreeQEvaluator(curve, mask)
    if objective_sqrt_weights is None:
        objective_sqrt_weights = np.ones(int(np.sum(mask)), dtype=np.float64)
        objective_weighting = "one weight per recorded validation evaluation"
    else:
        objective_sqrt_weights = np.asarray(
            objective_sqrt_weights, dtype=np.float64
        )
        if objective_sqrt_weights.shape != (int(np.sum(mask)),):
            raise ValueError("objective weights do not match the fit window")
        if not np.all(np.isfinite(objective_sqrt_weights)) or np.any(
            objective_sqrt_weights <= 0.0
        ):
            raise ValueError("objective weights must be finite and positive")
        objective_weighting = "trapezoidal equal-mass integration in log intrinsic time"
    lower = np.asarray(
        [0.0, 0.0, 0.0, 0.0, LOG_C_BOUNDS[0], q_f_bounds[0], math.log(Q_K_BOUNDS[0])],
        dtype=np.float64,
    )
    upper = np.asarray(
        [12.0, 80.0, noise_amplitude_upper, noise_amplitude_upper, LOG_C_BOUNDS[1], q_f_bounds[1], math.log(Q_K_BOUNDS[1])],
        dtype=np.float64,
    )

    def evaluate(transformed: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        ordinary = transformed.copy()
        ordinary[-1] = math.exp(float(transformed[-1]))
        residual, jacobian = evaluator.evaluate(ordinary)
        transformed_jacobian = jacobian.copy()
        transformed_jacobian[:, -1] *= ordinary[-1]
        return (
            residual * objective_sqrt_weights,
            transformed_jacobian * objective_sqrt_weights[:, None],
        )

    starts: list[np.ndarray] = []
    q_starts = (0.10, 0.35, 0.70, 0.95, 1.01, 1.50, 3.0, 8.0, 20.0)
    q_f_starts = (0.10, 0.20, 0.35, 0.55, 0.80, 1.20, 2.0, 3.5, 5.0)
    c_starts = (0.002, 0.01, 0.05, 0.2, 0.5, 1.0, 3.0, 10.0, 30.0)
    for q_k, q_f, c_k in zip(q_starts, q_f_starts, c_starts, strict=True):
        level, amplitude = phase._initial_clean_coefficients(curve, mask, q_f)
        starts.append(
            np.asarray(
                [level, amplitude, 0.1, 0.1, math.log(c_k), q_f, math.log(q_k)],
                dtype=np.float64,
            )
        )

    solutions = []
    for start in starts:
        fit = least_squares(
            lambda vector: evaluate(vector)[0],
            np.clip(start, lower + 1e-10, upper - 1e-10),
            jac=lambda vector: evaluate(vector)[1],
            bounds=(lower, upper),
            loss="huber",
            f_scale=1e-3,
            max_nfev=2_400,
            xtol=1e-9,
            ftol=1e-9,
            gtol=1e-9,
            x_scale="jac",
        )
        rss = float(np.sum(np.square(evaluate(fit.x)[0])))
        solutions.append((rss, fit))

    rss, fit = min(solutions, key=lambda item: item[0])
    ordinary = fit.x.copy()
    ordinary[-1] = math.exp(float(fit.x[-1]))
    parameters = fixed_q._parameters(ordinary[:6], float(ordinary[6]))
    near_q = [
        math.exp(float(candidate.x[-1]))
        for score, candidate in solutions
        if score <= rss * (1.0 + 1e-4) + 1e-12
    ]
    q_k = float(parameters["q_K"])
    raw_phase = "LM" if q_k < 1.0 else "IM"
    boundary_unresolved = (
        abs(q_k - 1.0) <= BOUNDARY_TOLERANCE
        or (min(near_q) < 1.0 < max(near_q))
    )
    return {
        "model": "ordinary_memory_unified",
        "parameter_count": 7,
        "parameters": parameters,
        "optimizer": {
            "success": bool(fit.success),
            "status": int(fit.status),
            "message": str(fit.message),
            "function_evaluations": int(fit.nfev),
            "multistarts": len(solutions),
            "log_residual_rss": rss,
            "objective_weighting": objective_weighting,
        },
        "phase_classification": {
            "raw_phase_from_point_estimate": raw_phase,
            "decision": "boundary_unresolved" if boundary_unresolved else raw_phase,
            "boundary": 1.0,
            "exploratory_boundary_tolerance": BOUNDARY_TOLERANCE,
            "near_optimal_multistart_q_K_min": float(min(near_q)),
            "near_optimal_multistart_q_K_max": float(max(near_q)),
        },
    }


def _log_time_objective_sqrt_weights(
    curve: phase.EvaluationCurve, mask: np.ndarray
) -> np.ndarray:
    log_time = np.log(np.asarray(curve.intrinsic_time[mask], dtype=np.float64))
    if log_time.size < 3 or not np.all(np.diff(log_time) > 0.0):
        raise ValueError("log-time weighting requires at least three strict times")
    weights = np.empty_like(log_time)
    weights[0] = 0.5 * (log_time[1] - log_time[0])
    weights[-1] = 0.5 * (log_time[-1] - log_time[-2])
    weights[1:-1] = 0.5 * (log_time[2:] - log_time[:-2])
    weights *= float(weights.size) / float(np.sum(weights))
    return np.sqrt(weights)


def _log_time_subset_mask(
    curve: phase.EvaluationCurve,
    full_mask: np.ndarray,
    target_count: int = 256,
) -> np.ndarray:
    eligible = np.flatnonzero(full_mask)
    log_time = np.log(np.asarray(curve.intrinsic_time[eligible], dtype=np.float64))
    targets = np.linspace(log_time[0], log_time[-1], target_count)
    right = np.searchsorted(log_time, targets, side="left")
    right = np.clip(right, 0, log_time.size - 1)
    left = np.maximum(right - 1, 0)
    choose_left = np.abs(targets - log_time[left]) <= np.abs(
        log_time[right] - targets
    )
    nearest = np.where(choose_left, left, right)
    chosen = np.unique(np.concatenate(([0], nearest, [log_time.size - 1])))
    subset = np.zeros_like(full_mask, dtype=bool)
    subset[eligible[chosen]] = True
    if int(np.sum(subset)) != target_count:
        raise ValueError(
            f"log-time target grid produced {int(np.sum(subset))}, expected {target_count}"
        )
    return subset


def _render(
    output_directory: Path,
    fit_curve: phase.EvaluationCurve,
    transfer_curve: phase.EvaluationCurve,
    prediction: np.ndarray,
    transfer_prediction: np.ndarray,
    fit_mask: np.ndarray,
) -> None:
    visible = fit_curve.intrinsic_time >= fit_curve.intrinsic_time[fit_mask][0]
    tail = fit_curve.intrinsic_time >= core.PREFIX_INTRINSIC_TIME - 1e-12
    figure, axes = plt.subplots(1, 2, figsize=(14.2, 5.2), constrained_layout=True)
    axes[0].plot(
        fit_curve.intrinsic_time[visible],
        fit_curve.loss[visible],
        color="#2878B5",
        linewidth=1.0,
        alpha=0.80,
        label="Observed: fixed batch / LR schedule",
    )
    axes[0].plot(
        fit_curve.intrinsic_time[visible],
        prediction[visible],
        color="#111111",
        linewidth=2.0,
        label="Seven-parameter fit",
    )
    axes[0].axvline(core.PREFIX_INTRINSIC_TIME, color="0.45", linestyle=":", linewidth=1.1)
    axes[0].set_title("Fit trajectory")
    axes[0].set_xlabel(r"Intrinsic time, $T=\sum_s\eta_s$")
    axes[0].set_ylabel("Fixed-probe validation CE")
    axes[0].legend(frameon=False)

    axes[1].plot(
        fit_curve.intrinsic_time[tail],
        fit_curve.loss[tail],
        color="#2878B5",
        linewidth=1.05,
        label="Fixed batch / LR schedule",
    )
    axes[1].plot(
        transfer_curve.intrinsic_time[tail],
        transfer_curve.loss[tail],
        color="#E56B2F",
        linewidth=1.05,
        label="Fixed LR / batch schedule",
    )
    axes[1].plot(
        transfer_curve.intrinsic_time[tail],
        transfer_prediction[tail],
        color="#111111",
        linewidth=2.0,
        label="Zero-refit surrogate prediction",
    )
    axes[1].set_title("8-1-1 tail factorization check")
    axes[1].set_xlabel(r"Intrinsic time, $T=\sum_s\eta_s$")
    axes[1].set_ylabel("Fixed-probe validation CE")
    axes[1].legend(frameon=False)
    for axis in axes:
        axis.grid(alpha=0.22, linewidth=0.7)
    figure.suptitle("124M GPT · 2.5B tokens · plain SGD v002")
    figure.savefig(output_directory / "unified_fit_811_pair.png", dpi=220, bbox_inches="tight")
    figure.savefig(output_directory / "unified_fit_811_pair.pdf", bbox_inches="tight")
    plt.close(figure)


def analyze(prefix_directory: Path, fit_tail_directory: Path, transfer_tail_directory: Path, output_directory: Path) -> Json:
    output_directory.mkdir(parents=True, exist_ok=True)
    fit_curve, fit_validation = _combine_prefix_tail(prefix_directory, fit_tail_directory)
    transfer_curve, transfer_validation = _combine_prefix_tail(prefix_directory, transfer_tail_directory)
    if fit_curve.arm_id != FIT_ARM or transfer_curve.arm_id != TRANSFER_ARM:
        raise validator.AnalysisError("8-1-1 arm identities changed")
    if not (
        np.array_equal(fit_curve.macro_index, transfer_curve.macro_index)
        and np.array_equal(fit_curve.intrinsic_time, transfer_curve.intrinsic_time)
    ):
        raise validator.AnalysisError("matched 8-1-1 intrinsic-time grids differ")

    mask = _fit_mask(fit_curve)
    tail_mask = fit_curve.intrinsic_time >= core.PREFIX_INTRINSIC_TIME - 1e-12
    with _use_v002_schedule_core():
        fit = _fit_unified(fit_curve, mask)
        prediction = fixed_q._predict_curve(fit_curve, fit["parameters"], force_fft=True)
        transfer_prediction = fixed_q._predict_curve(
            transfer_curve, fit["parameters"], force_fft=True
        )

    fit_metrics = phase._metrics(
        fit_curve.loss[mask], prediction[mask], parameter_count=fit["parameter_count"]
    )
    transfer_metrics = phase._metrics(
        transfer_curve.loss[mask], transfer_prediction[mask]
    )
    fit_tail_metrics = phase._metrics(fit_curve.loss[tail_mask], prediction[tail_mask])
    transfer_tail_metrics = phase._metrics(
        transfer_curve.loss[tail_mask], transfer_prediction[tail_mask]
    )
    observed_difference = transfer_curve.loss[tail_mask] - fit_curve.loss[tail_mask]
    median_evaluation_se = float(np.median(fit_curve.standard_error[mask]))
    fit["metrics"] = {
        "fit_arm_primary": fit_metrics,
        "transfer_arm_primary_zero_refit": transfer_metrics,
        "fit_arm_tail": fit_tail_metrics,
        "transfer_arm_tail_zero_refit": transfer_tail_metrics,
        "observed_factorization_tail_rmse_ce": float(
            np.sqrt(np.mean(np.square(observed_difference)))
        ),
        "observed_factorization_tail_max_abs_ce": float(
            np.max(np.abs(observed_difference))
        ),
        "median_primary_evaluation_standard_error_ce": median_evaluation_se,
        "fit_primary_rmse_in_median_evaluation_se_units": float(
            fit_metrics["rmse_ce"] / median_evaluation_se
        ),
        "transfer_primary_rmse_in_median_evaluation_se_units": float(
            transfer_metrics["rmse_ce"] / median_evaluation_se
        ),
    }

    _render(
        output_directory,
        fit_curve,
        transfer_curve,
        prediction,
        transfer_prediction,
        mask,
    )
    rows = []
    for arm_id, curve, predicted in (
        (FIT_ARM, fit_curve, prediction),
        (TRANSFER_ARM, transfer_curve, transfer_prediction),
    ):
        for index in range(curve.loss.size):
            rows.append(
                {
                    "arm_id": arm_id,
                    "global_update": int(curve.update[index]),
                    "macro_index": int(curve.macro_index[index]),
                    "intrinsic_time": float(curve.intrinsic_time[index]),
                    "observed_validation_ce": float(curve.loss[index]),
                    "predicted_validation_ce": float(predicted[index]),
                    "residual_ce": float(predicted[index] - curve.loss[index]),
                    "inside_fit_window": bool(mask[index]),
                }
            )
    _write_csv(output_directory / "predictions.csv", rows)
    _write_csv(
        output_directory / "fit_parameters.csv",
        [{"parameter": key, "value": value} for key, value in fit["parameters"].items()],
    )

    summary = {
        "schema_version": "nanogpt124m_e2e_sgd_unified_phase_surrogate_v002",
        "analysis_id": ANALYSIS_ID,
        "status": "completed",
        "classification": "post_hoc_exploratory_single_seed_A2_EXTERNAL_partial_two_tail_analysis",
        "new_gpu_hours_consumed_by_analysis": 0.0,
        "source_validation": {
            "fit_arm": fit_validation,
            "transfer_arm": transfer_validation,
        },
        "fit_protocol": {
            "fit_arm": FIT_ARM,
            "zero_refit_arm": TRANSFER_ARM,
            "fit_window_rule": FIT_WINDOW_RULE,
            "fit_start_intrinsic_time": float(fit_curve.intrinsic_time[mask][0]),
            "fit_end_intrinsic_time": float(fit_curve.intrinsic_time[-1]),
            "fit_evaluation_rows": int(np.sum(mask)),
            "tail_evaluation_rows_including_anchor": int(np.sum(tail_mask)),
            "clock": "intrinsic_time",
            "objective": "Huber robust least squares on log validation CE",
            "q_K_bounds": list(Q_K_BOUNDS),
            "q_F_bounds": list(Q_F_BOUNDS),
            "phase_not_conditioned_during_fit": True,
            "parameter_refit_on_transfer_curve": False,
        },
        "fit": fit,
        "claim_boundary": {
            "allowed": [
                "report the finite-window seven-parameter point optimum",
                "compare the matched 8-1-1 factorization without refitting",
                "diagnose whether the point optimum lies away from q_K=1",
            ],
            "prohibited": [
                "report an independently measured stationary memory kernel",
                "claim theorem-facing LM or IM identification",
                "claim WSD schedule transfer before both v002 WSD artifacts are collected",
            ],
        },
    }
    (output_directory / "summary.json").write_bytes(_canonical_bytes(summary))
    return summary


def analyze_sensitivity(
    prefix_directory: Path,
    fit_tail_directory: Path,
    output_directory: Path,
) -> Json:
    fit_curve, source_validation = _combine_prefix_tail(
        prefix_directory, fit_tail_directory
    )
    if fit_curve.arm_id != FIT_ARM:
        raise validator.AnalysisError("8-1-1 fit arm identity changed")
    mask = _fit_mask(fit_curve)
    with _use_v002_schedule_core():
        lower_q_f = _fit_unified(fit_curve, mask, q_f_bounds=(0.001, 8.0))
        enlarged_amplitude = _fit_unified(
            fit_curve, mask, noise_amplitude_upper=1e7
        )
    result = {
        "schema_version": "nanogpt124m_e2e_sgd_unified_phase_surrogate_sensitivity_v002",
        "analysis_id": ANALYSIS_ID + "-sensitivity",
        "status": "completed",
        "classification": "post_hoc_exploratory_parameter_bound_sensitivity",
        "source_validation": source_validation,
        "fit_window_rule": FIT_WINDOW_RULE,
        "fit_start_intrinsic_time": float(fit_curve.intrinsic_time[mask][0]),
        "fit_end_intrinsic_time": float(fit_curve.intrinsic_time[-1]),
        "fit_evaluation_rows": int(np.sum(mask)),
        "lower_q_F_bound_sensitivity": {
            "q_F_bounds": [0.001, 8.0],
            "fit": lower_q_f,
        },
        "enlarged_noise_amplitude_bound_sensitivity": {
            "noise_amplitude_upper": 1e7,
            "fit": enlarged_amplitude,
        },
        "claim_boundary": "Numerical bound sensitivity only; neither fit is an independent memory-kernel estimate.",
    }
    output_directory.mkdir(parents=True, exist_ok=True)
    (output_directory / "parameter_sensitivity.json").write_bytes(
        _canonical_bytes(result)
    )
    return result


def analyze_sampling_density_sensitivity(
    prefix_directory: Path,
    fit_tail_directory: Path,
    output_directory: Path,
) -> Json:
    fit_curve, source_validation = _combine_prefix_tail(
        prefix_directory, fit_tail_directory
    )
    if fit_curve.arm_id != FIT_ARM:
        raise validator.AnalysisError("8-1-1 fit arm identity changed")
    mask = _fit_mask(fit_curve)
    sqrt_weights = _log_time_objective_sqrt_weights(fit_curve, mask)
    subset_mask = _log_time_subset_mask(fit_curve, mask)
    with _use_v002_schedule_core():
        log_time_balanced = _fit_unified(
            fit_curve, mask, objective_sqrt_weights=sqrt_weights
        )
        log_time_subset = _fit_unified(fit_curve, subset_mask)
    q_weighted = float(log_time_balanced["parameters"]["q_K"])
    q_subset = float(log_time_subset["parameters"]["q_K"])
    result = {
        "schema_version": "nanogpt124m_e2e_sgd_unified_phase_surrogate_sampling_sensitivity_v002",
        "analysis_id": ANALYSIS_ID + "-sampling-density-sensitivity",
        "status": "completed",
        "classification": "post_hoc_exploratory_sampling_density_sensitivity",
        "source_validation": source_validation,
        "fit_window_rule": FIT_WINDOW_RULE,
        "fit_start_intrinsic_time": float(fit_curve.intrinsic_time[mask][0]),
        "fit_end_intrinsic_time": float(fit_curve.intrinsic_time[-1]),
        "fit_evaluation_rows": int(np.sum(mask)),
        "weighting": (
            "trapezoidal integration weights in log intrinsic time, "
            "normalized to mean one"
        ),
        "sqrt_weight_min": float(np.min(sqrt_weights)),
        "sqrt_weight_max": float(np.max(sqrt_weights)),
        "fit": log_time_balanced,
        "log_time_256_point_subset": {
            "target_count": 256,
            "selected_evaluation_rows": int(np.sum(subset_mask)),
            "tie_break": "nearest evaluation; ties choose the earlier row",
            "fit": log_time_subset,
        },
        "predeclared_stability_check": {
            "absolute_q_K_difference_between_sensitivities": abs(
                q_weighted - q_subset
            ),
            "both_boundary_unresolved": bool(
                log_time_balanced["phase_classification"]["decision"]
                == "boundary_unresolved"
                and log_time_subset["phase_classification"]["decision"]
                == "boundary_unresolved"
            ),
            "passes": bool(
                abs(q_weighted - q_subset) <= BOUNDARY_TOLERANCE
                and log_time_balanced["phase_classification"]["decision"]
                == "boundary_unresolved"
                and log_time_subset["phase_classification"]["decision"]
                == "boundary_unresolved"
            ),
        },
        "claim_boundary": (
            "Sampling-density sensitivity only; this is not an independent "
            "memory-kernel estimate."
        ),
    }
    output_directory.mkdir(parents=True, exist_ok=True)
    (output_directory / "sampling_density_sensitivity.json").write_bytes(
        _canonical_bytes(result)
    )
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sensitivity-only", action="store_true")
    parser.add_argument("--sampling-density-sensitivity-only", action="store_true")
    args = parser.parse_args(argv)
    root = Path(__file__).resolve().parents[2]
    campaign = root / "experiments" / "generated" / core.CAMPAIGN_ID
    output = root / "experiments" / "results" / ANALYSIS_ID
    if args.sampling_density_sensitivity_only:
        sensitivity = analyze_sampling_density_sensitivity(
            campaign / "prefix-launch-a001" / "collected_output",
            campaign
            / "tails-launch-a001"
            / "collected_output"
            / "tail_output_811_fblr",
            output,
        )
        print(
            json.dumps(
                {
                    "log_time_balanced_parameters": sensitivity["fit"][
                        "parameters"
                    ],
                    "log_time_256_subset_parameters": sensitivity[
                        "log_time_256_point_subset"
                    ]["fit"]["parameters"],
                    "stability_check": sensitivity[
                        "predeclared_stability_check"
                    ],
                    "phase": sensitivity["fit"]["phase_classification"],
                    "output": str(output / "sampling_density_sensitivity.json"),
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    if args.sensitivity_only:
        sensitivity = analyze_sensitivity(
            campaign / "prefix-launch-a001" / "collected_output",
            campaign
            / "tails-launch-a001"
            / "collected_output"
            / "tail_output_811_fblr",
            output,
        )
        print(
            json.dumps(
                {
                    "lower_q_F_bound_parameters": sensitivity[
                        "lower_q_F_bound_sensitivity"
                    ]["fit"]["parameters"],
                    "enlarged_amplitude_parameters": sensitivity[
                        "enlarged_noise_amplitude_bound_sensitivity"
                    ]["fit"]["parameters"],
                    "output": str(output / "parameter_sensitivity.json"),
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    summary = analyze(
        campaign / "prefix-launch-a001" / "collected_output",
        campaign / "tails-launch-a001" / "collected_output" / "tail_output_811_fblr",
        campaign / "tails-launch-a001" / "collected_output" / "tail_output_811_flrbs",
        output,
    )
    print(
        json.dumps(
            {
                "parameters": summary["fit"]["parameters"],
                "phase": summary["fit"]["phase_classification"],
                "metrics": summary["fit"]["metrics"],
                "output": str(output),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
