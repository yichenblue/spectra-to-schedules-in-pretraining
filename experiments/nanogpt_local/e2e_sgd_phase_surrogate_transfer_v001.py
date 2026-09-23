"""Exploratory phase-surrogate fit on 8-1-1 and zero-refit WSD transfer.

This analysis reuses the completed 30M end-to-end plain-SGD fixed-batch runs.
It is intentionally post-hoc and external-validity only: the source run did
not preregister LM/IM/FB identification, and one trajectory cannot identify a
stationary memory kernel.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import least_squares

from experiments.nanogpt_local import e2e_sgd_full_rpath_factorization_v001 as core


Json = dict[str, Any]

ANALYSIS_ID = "nanogpt30m-e2e-sgd-phase-surrogate-transfer-v001"
PHASES = ("LM", "IM", "FB")
PHASE_COLORS = {"LM": "#2a6fbb", "IM": "#d1495b", "FB": "#2f8f5b"}
PRIMARY_WINDOW_RULE = (
    "last evaluation at or below one tenth of terminal 8-1-1 intrinsic time, "
    "through the terminal evaluation"
)


@dataclass(frozen=True)
class EvaluationCurve:
    run_id: str
    arm_id: str
    shape_id: str
    update: np.ndarray
    macro_index: np.ndarray
    intrinsic_time: np.ndarray
    loss: np.ndarray
    standard_error: np.ndarray


def _canonical_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> Json:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if type(value) is not dict:
        raise ValueError(f"expected a JSON object: {path}")
    return value


def _load_curve(path: Path, expected_arm: str) -> EvaluationCurve:
    payload = _read_json(path)
    result = payload.get("measurements", {}).get("experiment_result", {})
    if payload.get("completed") is not True or result.get("status") != "completed":
        raise ValueError(f"source result is not completed: {path}")
    if result.get("arm_id") != expected_arm:
        raise ValueError(
            f"arm mismatch for {path}: {result.get('arm_id')!r} != {expected_arm!r}"
        )
    rows = result.get("measurements", {}).get("evaluations", {}).get("rows")
    if type(rows) is not list or not rows:
        raise ValueError(f"missing evaluation rows: {path}")
    fields = {
        "update": np.int64,
        "macro_index": np.int64,
        "intrinsic_time": np.float64,
        "validation_cross_entropy": np.float64,
        "validation_standard_error": np.float64,
    }
    arrays = {
        key: np.asarray([row[key] for row in rows], dtype=dtype)
        for key, dtype in fields.items()
    }
    for key, values in arrays.items():
        if not np.all(np.isfinite(values)):
            raise ValueError(f"nonfinite {key} in {path}")
    if not np.all(np.diff(arrays["update"]) > 0):
        raise ValueError(f"evaluation updates are not strictly increasing: {path}")
    if not np.all(np.diff(arrays["intrinsic_time"]) > 0):
        raise ValueError(f"evaluation intrinsic times are not strictly increasing: {path}")
    shape_id = str(rows[0]["shape_id"])
    if any(row.get("shape_id") != shape_id for row in rows):
        raise ValueError(f"mixed shape ids in {path}")
    return EvaluationCurve(
        run_id=str(payload["run_id"]),
        arm_id=expected_arm,
        shape_id=shape_id,
        update=arrays["update"],
        macro_index=arrays["macro_index"],
        intrinsic_time=arrays["intrinsic_time"],
        loss=arrays["validation_cross_entropy"],
        standard_error=arrays["validation_standard_error"],
    )


def _validate_pair(eight: EvaluationCurve, wsd: EvaluationCurve) -> None:
    if eight.shape_id != "eight_one_one" or wsd.shape_id != "wsd_exp_80_20":
        raise ValueError("unexpected source schedule shapes")
    prefix = (eight.update <= core.CONSTANT_SOURCE_UPDATES) & np.isin(
        eight.update, wsd.update
    )
    common_updates = eight.update[prefix]
    for update in common_updates:
        left = eight.loss[eight.update == update][0]
        right = wsd.loss[wsd.update == update][0]
        if left != right:
            raise ValueError(f"shared-prefix validation loss differs at update {update}")


@lru_cache(maxsize=2)
def _macro_quadrature(shape_id: str) -> tuple[np.ndarray, np.ndarray]:
    """Return intrinsic-time midpoints and exact per-macro injection masses.

    For fixed-batch/LR realization, one macro contains ``g`` updates with
    ``eta=.02/g`` and ``B=4``.  Thus its exact sum of eta^2/B is
    ``.02/(200*g) = DeltaT/r``.  Only the source/kernel values use midpoint
    quadrature on the fine DeltaT=.02 macro grid.
    """

    plan = core._schedule_plan(shape_id)
    g = np.asarray(plan["g"], dtype=np.float64)
    index = np.arange(g.size, dtype=np.float64)
    midpoints = (index + 0.5) * core.MACRO_INTRINSIC_TIME
    injection = core.MACRO_INTRINSIC_TIME / (core.EXECUTED_RATIO_UNIT * g)
    return midpoints, injection


@lru_cache(maxsize=4)
def _causal_grid(
    shape_id: str,
    macro_indices: tuple[int, ...],
    intrinsic_times: tuple[float, ...],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    source_time, injection = _macro_quadrature(shape_id)
    maximum = max(macro_indices, default=0)
    source_time = source_time[:maximum]
    injection = injection[:maximum]
    counts = np.asarray(macro_indices, dtype=np.int64)
    targets = np.asarray(intrinsic_times, dtype=np.float64)
    causal = np.arange(maximum, dtype=np.int64)[None, :] < counts[:, None]
    weights = causal * injection[None, :]
    ages = np.maximum(targets[:, None] - source_time[None, :], 0.0)
    return source_time, weights, ages


def _feature_matrix(
    curve: EvaluationCurve,
    q_f: float,
    phase: str,
    c_k: float | None = None,
    q_k: float | None = None,
) -> np.ndarray:
    source_time, weights, ages = _causal_grid(
        curve.shape_id,
        tuple(int(value) for value in curve.macro_index),
        tuple(float(value) for value in curve.intrinsic_time),
    )
    clean = np.power(1.0 + curve.intrinsic_time, -q_f)
    source = np.power(1.0 + source_time, -q_f)
    if phase == "FB":
        s0 = np.sum(weights, axis=1)
        return np.column_stack([np.ones(curve.loss.size), clean, s0])
    if c_k is None or q_k is None:
        raise ValueError("ordinary-memory features require c_k and q_k")
    weighted = weights * np.power(1.0 + c_k * ages, -q_k)
    s0 = np.sum(weighted, axis=1)
    s1 = np.sum(weighted * source[None, :], axis=1)
    return np.column_stack([np.ones(curve.loss.size), clean, s0, s1])


def _predict(curve: EvaluationCurve, phase: str, parameters: Mapping[str, float]) -> np.ndarray:
    if phase == "FB":
        matrix = _feature_matrix(curve, parameters["q_F"], phase)
        coefficients = np.asarray(
            [parameters["L_inf"], parameters["clean_amplitude"], parameters["noise_amplitude"]],
            dtype=np.float64,
        )
    else:
        matrix = _feature_matrix(
            curve,
            parameters["q_F"],
            phase,
            parameters["kernel_scale"],
            parameters["q_K"],
        )
        coefficients = np.asarray(
            [
                parameters["L_inf"],
                parameters["clean_amplitude"],
                parameters["noise_floor_amplitude"],
                parameters["fit_dependent_noise_amplitude"],
            ],
            dtype=np.float64,
        )
    return matrix @ coefficients


def _ordinary_parameters(vector: Sequence[float]) -> Json:
    return {
        "L_inf": float(vector[0]),
        "clean_amplitude": float(vector[1]),
        "noise_floor_amplitude": float(vector[2]),
        "fit_dependent_noise_amplitude": float(vector[3]),
        "kernel_scale": float(math.exp(vector[4])),
        "q_F": float(vector[5]),
        "q_K": float(vector[6]),
    }


def _fb_parameters(vector: Sequence[float]) -> Json:
    return {
        "L_inf": float(vector[0]),
        "clean_amplitude": float(vector[1]),
        "noise_amplitude": float(vector[2]),
        "q_F": float(vector[3]),
    }


def _log_residual(prediction: np.ndarray, target: np.ndarray) -> np.ndarray:
    if np.any(prediction <= 0.0):
        return np.full(target.size, 1e6, dtype=np.float64)
    return np.log(prediction) - np.log(target)


def _initial_clean_coefficients(curve: EvaluationCurve, mask: np.ndarray, q_f: float) -> tuple[float, float]:
    clean = np.power(1.0 + curve.intrinsic_time[mask], -q_f)
    matrix = np.column_stack([np.ones(clean.size), clean])
    coefficients = np.linalg.lstsq(matrix, curve.loss[mask], rcond=None)[0]
    return max(float(coefficients[0]), 0.1), max(float(coefficients[1]), 0.1)


def _fit_ordinary(curve: EvaluationCurve, mask: np.ndarray, phase: str) -> Json:
    if phase == "LM":
        q_bounds = (0.02, 0.98)
        q_starts = (0.15, 0.50, 0.85)
    elif phase == "IM":
        q_bounds = (1.02, 6.0)
        q_starts = (1.10, 1.75, 3.50)
    else:
        raise ValueError(phase)
    lower = np.asarray([0.0, 0.0, 0.0, 0.0, -9.0, 0.02, q_bounds[0]])
    upper = np.asarray([12.0, 40.0, 1e5, 1e5, 9.0, 4.0, q_bounds[1]])
    best = None
    starts = []
    start_grid = (
        (0.20, 0.05, q_starts[0]),
        (0.40, 0.30, q_starts[1]),
        (0.80, 1.00, q_starts[2]),
        (0.40, 3.00, q_starts[1]),
    )
    for q_f, c_k, q_k in start_grid:
        base, amplitude = _initial_clean_coefficients(curve, mask, q_f)
        starts.append(
            np.asarray(
                [base, amplitude, 0.1, 0.1, math.log(c_k), q_f, q_k],
                dtype=np.float64,
            )
        )

    target = curve.loss[mask]

    def residual(vector: np.ndarray) -> np.ndarray:
        parameters = _ordinary_parameters(vector)
        return _log_residual(_predict(curve, phase, parameters)[mask], target)

    solutions = []
    for start in starts:
        start = np.clip(start, lower + 1e-10, upper - 1e-10)
        fit = least_squares(
            residual,
            start,
            bounds=(lower, upper),
            loss="huber",
            f_scale=1e-3,
            max_nfev=500,
            xtol=1e-8,
            ftol=1e-8,
            gtol=1e-8,
        )
        objective = float(np.sum(np.square(residual(fit.x))))
        solutions.append((objective, fit))
        if best is None or objective < best[0]:
            best = (objective, fit)
    assert best is not None
    objective, fit = best
    parameters = _ordinary_parameters(fit.x)
    near = [
        _ordinary_parameters(candidate.x)
        for score, candidate in solutions
        if score <= objective * (1.0 + 1e-4) + 1e-12
    ]
    return {
        "phase": phase,
        "parameter_count": 7,
        "parameters": parameters,
        "optimizer": {
            "success": bool(fit.success),
            "status": int(fit.status),
            "message": str(fit.message),
            "function_evaluations": int(fit.nfev),
            "multistarts": len(starts),
            "log_residual_rss": objective,
        },
        "boundary_diagnostics": {
            "q_K_lower": q_bounds[0],
            "q_K_upper": q_bounds[1],
            "q_K_fraction_through_interval": float(
                (parameters["q_K"] - q_bounds[0]) / (q_bounds[1] - q_bounds[0])
            ),
            "near_optimal_solution_count": len(near),
            "near_optimal_q_K_min": min(item["q_K"] for item in near),
            "near_optimal_q_K_max": max(item["q_K"] for item in near),
            "noise_floor_contribution_active": parameters["noise_floor_amplitude"] > 1e-8,
            "fit_dependent_contribution_active": parameters["fit_dependent_noise_amplitude"] > 1e-8,
        },
    }


def _fit_fb(curve: EvaluationCurve, mask: np.ndarray) -> Json:
    lower = np.asarray([0.0, 0.0, 0.0, 0.02])
    upper = np.asarray([12.0, 40.0, 1e5, 4.0])
    target = curve.loss[mask]

    def parameters(vector: Sequence[float]) -> Json:
        return _fb_parameters(vector)

    def residual(vector: np.ndarray) -> np.ndarray:
        return _log_residual(_predict(curve, "FB", parameters(vector))[mask], target)

    solutions = []
    for q_f in (0.10, 0.30, 0.70, 1.50):
        base, amplitude = _initial_clean_coefficients(curve, mask, q_f)
        for noise in (0.05, 1.0):
            start = np.asarray([base, amplitude, noise, q_f], dtype=np.float64)
            start = np.clip(start, lower + 1e-10, upper - 1e-10)
            fit = least_squares(
                residual,
                start,
                bounds=(lower, upper),
                loss="huber",
                f_scale=1e-3,
                max_nfev=500,
                xtol=1e-8,
                ftol=1e-8,
                gtol=1e-8,
            )
            score = float(np.sum(np.square(residual(fit.x))))
            solutions.append((score, fit))
    score, fit = min(solutions, key=lambda item: item[0])
    fitted = parameters(fit.x)
    return {
        "phase": "FB",
        "parameter_count": 4,
        "parameters": fitted,
        "optimizer": {
            "success": bool(fit.success),
            "status": int(fit.status),
            "message": str(fit.message),
            "function_evaluations": int(fit.nfev),
            "multistarts": len(solutions),
            "log_residual_rss": score,
        },
        "boundary_diagnostics": {
            "noise_contribution_active": fitted["noise_amplitude"] > 1e-8,
        },
    }


def _metrics(target: np.ndarray, prediction: np.ndarray, parameter_count: int | None = None) -> Json:
    residual = prediction - target
    log_residual = np.log(prediction) - np.log(target)
    result = {
        "count": int(target.size),
        "rmse_ce": float(np.sqrt(np.mean(np.square(residual)))),
        "mae_ce": float(np.mean(np.abs(residual))),
        "max_abs_ce": float(np.max(np.abs(residual))),
        "rmse_log_ce": float(np.sqrt(np.mean(np.square(log_residual)))),
        "bias_ce": float(np.mean(residual)),
    }
    if parameter_count is not None:
        n = target.size
        rss = max(float(np.sum(np.square(log_residual))), np.finfo(float).tiny)
        aic = n * math.log(rss / n) + 2.0 * parameter_count
        result["aic_log_residual"] = float(aic)
        result["aicc_log_residual"] = float(
            aic
            + 2.0
            * parameter_count
            * (parameter_count + 1)
            / (n - parameter_count - 1)
        )
        result["bic_log_residual"] = float(
            n * math.log(rss / n) + parameter_count * math.log(n)
        )
    return result


def _window_masks(eight: EvaluationCurve, wsd: EvaluationCurve) -> dict[str, np.ndarray]:
    positive = np.flatnonzero(eight.intrinsic_time > 0.0)
    threshold = float(eight.intrinsic_time[-1] / 10.0)
    eligible = positive[eight.intrinsic_time[positive] <= threshold]
    if eligible.size == 0:
        raise ValueError("8-1-1 curve does not expose a full intrinsic-time decade")
    start_index = int(eligible[-1])
    primary = np.arange(eight.loss.size) >= start_index
    fork_time = core.CONSTANT_SOURCE_UPDATES * core.CANONICAL_BASE_LEARNING_RATE
    return {
        "eight_primary": primary,
        "eight_all_positive": eight.intrinsic_time > 0.0,
        "wsd_full": np.ones(wsd.loss.size, dtype=bool),
        "wsd_tail": wsd.intrinsic_time >= fork_time - 1e-12,
        "wsd_tail_overlap": (wsd.intrinsic_time >= fork_time - 1e-12)
        & (wsd.intrinsic_time <= eight.intrinsic_time[-1] + 1e-12),
        "wsd_tail_extrapolation": wsd.intrinsic_time > eight.intrinsic_time[-1] + 1e-12,
    }


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty CSV: {path}")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _render(
    output_dir: Path,
    eight: EvaluationCurve,
    wsd: EvaluationCurve,
    fits: Mapping[str, Json],
    masks: Mapping[str, np.ndarray],
) -> None:
    fig, axes = plt.subplots(3, 2, figsize=(13.0, 11.0), sharex="col")
    for row, phase in enumerate(PHASES):
        fitted = _predict(eight, phase, fits[phase]["parameters"])
        transfer = _predict(wsd, phase, fits[phase]["parameters"])
        left, right = axes[row]
        left.errorbar(
            eight.intrinsic_time,
            eight.loss,
            yerr=eight.standard_error,
            fmt="o",
            ms=2.8,
            lw=0.7,
            color="#222222",
            alpha=0.75,
            label="8-1-1 observed",
        )
        left.plot(eight.intrinsic_time, fitted, color=PHASE_COLORS[phase], lw=2.0, label=f"{phase} fit")
        left.axvspan(
            eight.intrinsic_time[0],
            eight.intrinsic_time[masks["eight_primary"]][0],
            color="#bbbbbb",
            alpha=0.12,
            label="outside primary fit window" if row == 0 else None,
        )
        right.errorbar(
            wsd.intrinsic_time,
            wsd.loss,
            yerr=wsd.standard_error,
            fmt="o",
            ms=2.8,
            lw=0.7,
            color="#222222",
            alpha=0.75,
            label="WSD observed",
        )
        right.plot(wsd.intrinsic_time, transfer, color=PHASE_COLORS[phase], lw=2.0, label=f"{phase} zero-refit prediction")
        for axis in (left, right):
            axis.axvline(409.6, color="#777777", ls="--", lw=1.0)
            axis.grid(alpha=0.22)
            axis.set_ylabel("Validation CE")
            axis.legend(loc="best", fontsize=8)
        left.set_title(f"{phase}: fit on 8-1-1")
        right.set_title(f"{phase}: transfer to WSD")
    axes[-1, 0].set_xlabel("Intrinsic time")
    axes[-1, 1].set_xlabel("Intrinsic time")
    fig.suptitle("30M fixed-batch phase surrogates: 8-1-1 fit and zero-refit WSD transfer")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(output_dir / "phase_surrogate_fit_and_transfer.png", dpi=180)
    fig.savefig(output_dir / "phase_surrogate_fit_and_transfer.pdf")
    plt.close(fig)

    tail = masks["wsd_tail"]
    fig, axis = plt.subplots(figsize=(9.4, 5.6))
    axis.errorbar(
        wsd.intrinsic_time[tail],
        wsd.loss[tail],
        yerr=wsd.standard_error[tail],
        fmt="o",
        ms=4.0,
        lw=0.8,
        color="#111111",
        label="WSD observed",
    )
    for phase in PHASES:
        prediction = _predict(wsd, phase, fits[phase]["parameters"])
        axis.plot(
            wsd.intrinsic_time[tail],
            prediction[tail],
            lw=2.0,
            color=PHASE_COLORS[phase],
            label=f"{phase} zero-refit",
        )
    axis.axvline(eight.intrinsic_time[-1], color="#777777", ls=":", lw=1.2, label="8-1-1 terminal T")
    axis.set_xlabel("Intrinsic time")
    axis.set_ylabel("Validation CE")
    axis.set_title("WSD tail: zero-refit predictions from the 8-1-1 fit")
    axis.grid(alpha=0.22)
    axis.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "wsd_zero_refit_tail_comparison.png", dpi=180)
    fig.savefig(output_dir / "wsd_zero_refit_tail_comparison.pdf")
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(9.4, 5.6))
    for phase in PHASES:
        prediction = _predict(wsd, phase, fits[phase]["parameters"])
        axis.plot(
            wsd.intrinsic_time[tail],
            (prediction - wsd.loss)[tail],
            marker="o",
            ms=3.0,
            lw=1.5,
            color=PHASE_COLORS[phase],
            label=phase,
        )
    axis.axhline(0.0, color="#111111", lw=1.0)
    axis.axvline(eight.intrinsic_time[-1], color="#777777", ls=":", lw=1.2)
    axis.set_xlabel("Intrinsic time")
    axis.set_ylabel("Prediction minus observed validation CE")
    axis.set_title("WSD tail zero-refit residuals")
    axis.grid(alpha=0.22)
    axis.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "wsd_zero_refit_tail_residuals.png", dpi=180)
    fig.savefig(output_dir / "wsd_zero_refit_tail_residuals.pdf")
    plt.close(fig)


def _load_existing_fits(path: Path) -> dict[str, Json]:
    values: dict[str, dict[str, float]] = {phase: {} for phase in PHASES}
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            phase = str(row["phase"])
            if phase not in values:
                raise ValueError(f"unexpected phase in {path}: {phase}")
            values[phase][str(row["parameter"])] = float(row["value"])
    expected = {
        "LM": {
            "L_inf", "clean_amplitude", "noise_floor_amplitude",
            "fit_dependent_noise_amplitude", "kernel_scale", "q_F", "q_K",
        },
        "IM": {
            "L_inf", "clean_amplitude", "noise_floor_amplitude",
            "fit_dependent_noise_amplitude", "kernel_scale", "q_F", "q_K",
        },
        "FB": {"L_inf", "clean_amplitude", "noise_amplitude", "q_F"},
    }
    for phase in PHASES:
        if set(values[phase]) != expected[phase]:
            raise ValueError(f"incomplete {phase} parameters in {path}")
    return {
        phase: {
            "phase": phase,
            "parameter_count": 4 if phase == "FB" else 7,
            "parameters": values[phase],
            "optimizer": {"reused_completed_fit_parameters_from": str(path)},
            "boundary_diagnostics": (
                {"noise_contribution_active": values[phase]["noise_amplitude"] > 1e-8}
                if phase == "FB"
                else {
                    "q_K_lower": 0.02 if phase == "LM" else 1.02,
                    "q_K_upper": 0.98 if phase == "LM" else 6.0,
                    "q_K_fraction_through_interval": (
                        (values[phase]["q_K"] - (0.02 if phase == "LM" else 1.02))
                        / ((0.98 if phase == "LM" else 6.0) - (0.02 if phase == "LM" else 1.02))
                    ),
                    "noise_floor_contribution_active": values[phase]["noise_floor_amplitude"] > 1e-8,
                    "fit_dependent_contribution_active": values[phase]["fit_dependent_noise_amplitude"] > 1e-8,
                }
            ),
        }
        for phase in PHASES
    }


def analyze(
    eight_path: Path,
    wsd_path: Path,
    output_dir: Path,
    reuse_existing_fits: bool = False,
) -> Json:
    output_dir.mkdir(parents=True, exist_ok=True)
    eight = _load_curve(eight_path, "eight_one_one__fixed_batch_lr")
    wsd = _load_curve(wsd_path, "wsd_exp_80_20__fixed_batch_lr")
    _validate_pair(eight, wsd)
    masks = _window_masks(eight, wsd)

    parameter_path = output_dir / "fit_parameters.csv"
    if reuse_existing_fits:
        if not parameter_path.is_file():
            raise ValueError(f"existing fit parameters not found: {parameter_path}")
        fits = _load_existing_fits(parameter_path)
    else:
        fits = {
            "LM": _fit_ordinary(eight, masks["eight_primary"], "LM"),
            "IM": _fit_ordinary(eight, masks["eight_primary"], "IM"),
            "FB": _fit_fb(eight, masks["eight_primary"]),
        }
    metric_rows = []
    prediction_rows = []
    for phase in PHASES:
        fit_prediction = _predict(eight, phase, fits[phase]["parameters"])
        transfer_prediction = _predict(wsd, phase, fits[phase]["parameters"])
        fits[phase]["metrics"] = {
            "eight_primary_fit": _metrics(
                eight.loss[masks["eight_primary"]],
                fit_prediction[masks["eight_primary"]],
                fits[phase]["parameter_count"],
            ),
            "eight_all_positive_sensitivity": _metrics(
                eight.loss[masks["eight_all_positive"]],
                fit_prediction[masks["eight_all_positive"]],
            ),
            "wsd_full_zero_refit": _metrics(wsd.loss, transfer_prediction),
            "wsd_tail_zero_refit": _metrics(
                wsd.loss[masks["wsd_tail"]], transfer_prediction[masks["wsd_tail"]]
            ),
            "wsd_tail_overlap_zero_refit": _metrics(
                wsd.loss[masks["wsd_tail_overlap"]],
                transfer_prediction[masks["wsd_tail_overlap"]],
            ),
            "wsd_tail_extrapolation_zero_refit": _metrics(
                wsd.loss[masks["wsd_tail_extrapolation"]],
                transfer_prediction[masks["wsd_tail_extrapolation"]],
            ),
        }
        for window, metrics in fits[phase]["metrics"].items():
            metric_rows.append({"phase": phase, "window": window, **metrics})
        for schedule, curve, prediction in (
            ("eight_one_one", eight, fit_prediction),
            ("wsd_exp_80_20", wsd, transfer_prediction),
        ):
            for index in range(curve.loss.size):
                prediction_rows.append(
                    {
                        "phase": phase,
                        "schedule": schedule,
                        "update": int(curve.update[index]),
                        "macro_index": int(curve.macro_index[index]),
                        "intrinsic_time": float(curve.intrinsic_time[index]),
                        "observed_validation_ce": float(curve.loss[index]),
                        "predicted_validation_ce": float(prediction[index]),
                        "residual_ce": float(prediction[index] - curve.loss[index]),
                        "validation_standard_error": float(curve.standard_error[index]),
                        "in_primary_fit_window": bool(
                            schedule == "eight_one_one" and masks["eight_primary"][index]
                        ),
                    }
                )

    ranked_fit = sorted(
        PHASES,
        key=lambda phase: fits[phase]["metrics"]["eight_primary_fit"]["aicc_log_residual"],
    )
    ranked_transfer = sorted(
        PHASES,
        key=lambda phase: fits[phase]["metrics"]["wsd_tail_zero_refit"]["rmse_ce"],
    )
    best_transfer = fits[ranked_transfer[0]]["metrics"]["wsd_tail_zero_refit"]["rmse_ce"]
    second_transfer = fits[ranked_transfer[1]]["metrics"]["wsd_tail_zero_refit"]["rmse_ce"]
    median_tail_se = float(np.median(wsd.standard_error[masks["wsd_tail"]]))
    summary = {
        "schema_version": "nanogpt30m_e2e_sgd_phase_surrogate_transfer_v001",
        "analysis_id": ANALYSIS_ID,
        "status": "completed",
        "classification": "post_hoc_exploratory_A2_EXTERNAL",
        "claim_boundary": {
            "allowed": [
                "compare constrained empirical surrogate fits",
                "compare zero-refit WSD predictive errors",
            ],
            "prohibited": [
                "theorem confirmation",
                "LM/IM/FB phase identification from this result alone",
                "stationary q_K identification",
            ],
        },
        "source": {
            "eight_one_one": {
                "path": str(eight_path),
                "sha256": _sha256_file(eight_path),
                "run_id": eight.run_id,
                "arm_id": eight.arm_id,
                "evaluation_rows": int(eight.loss.size),
                "terminal_intrinsic_time": float(eight.intrinsic_time[-1]),
            },
            "wsd": {
                "path": str(wsd_path),
                "sha256": _sha256_file(wsd_path),
                "run_id": wsd.run_id,
                "arm_id": wsd.arm_id,
                "evaluation_rows": int(wsd.loss.size),
                "terminal_intrinsic_time": float(wsd.intrinsic_time[-1]),
            },
        },
        "fit_protocol": {
            "target": "fixed_validation_probe_cross_entropy",
            "clock": "intrinsic_time",
            "objective": "Huber robust least squares on log validation CE",
            "huber_f_scale": 1e-3,
            "primary_window_rule": PRIMARY_WINDOW_RULE,
            "primary_start_update": int(eight.update[masks["eight_primary"]][0]),
            "primary_start_intrinsic_time": float(eight.intrinsic_time[masks["eight_primary"]][0]),
            "primary_end_intrinsic_time": float(eight.intrinsic_time[-1]),
            "primary_evaluation_rows": int(np.sum(masks["eight_primary"])),
            "schedule_quadrature": "DeltaT=0.02 macro midpoint; exact per-macro sum eta^2/B",
            "parameter_refit_on_wsd": False,
        },
        "fits": fits,
        "comparison": {
            "in_sample_aicc_ranking": ranked_fit,
            "zero_refit_wsd_tail_rmse_ranking": ranked_transfer,
            "best_to_second_wsd_tail_rmse_ratio": float(best_transfer / second_transfer),
            "median_wsd_tail_validation_standard_error": median_tail_se,
            "best_wsd_tail_rmse_in_median_evaluation_se_units": float(
                best_transfer / median_tail_se
            ),
            "decision_label": "inconclusive",
            "decision_rule_note": (
                "This single-seed post-hoc end-to-end run was not preregistered for phase "
                "identification. Rankings are descriptive; out-of-schedule separation and "
                "parameter-boundary behavior determine whether a follow-up is worthwhile."
            ),
        },
    }

    parameter_rows = []
    for phase in PHASES:
        for name, value in fits[phase]["parameters"].items():
            parameter_rows.append({"phase": phase, "parameter": name, "value": value})
    _write_csv(parameter_path, parameter_rows)
    _write_csv(output_dir / "fit_metrics.csv", metric_rows)
    _write_csv(output_dir / "predictions.csv", prediction_rows)

    summary_bytes = _canonical_bytes(summary)
    (output_dir / "summary.json").write_bytes(summary_bytes)
    readme = f"""# {ANALYSIS_ID}

Post-hoc exploratory A2 external-validity analysis. Three constrained phase
surrogates are fitted only to the completed fixed-batch 8-1-1 validation-CE
trajectory. Their frozen parameters are then used to predict the completed
fixed-batch WSD trajectory. No GPU execution or WSD parameter refit occurs.

Primary fit window: `{PRIMARY_WINDOW_RULE}`.

This artifact cannot identify an LM, IM, or FB phase by itself and cannot be
used as an independent estimate of a stationary memory kernel.
"""
    (output_dir / "README.md").write_text(readme, encoding="utf-8")
    _render(output_dir, eight, wsd, fits, masks)
    artifact_names = (
        "fit_parameters.csv",
        "fit_metrics.csv",
        "predictions.csv",
        "phase_surrogate_fit_and_transfer.png",
        "phase_surrogate_fit_and_transfer.pdf",
        "wsd_zero_refit_tail_comparison.png",
        "wsd_zero_refit_tail_comparison.pdf",
        "wsd_zero_refit_tail_residuals.png",
        "wsd_zero_refit_tail_residuals.pdf",
        "README.md",
    )
    summary["artifacts"] = {
        name: {"sha256": _sha256_file(output_dir / name), "bytes": (output_dir / name).stat().st_size}
        for name in artifact_names
    }
    (output_dir / "summary.json").write_bytes(_canonical_bytes(summary))
    return summary


def _default_paths(root: Path) -> tuple[Path, Path, Path]:
    runs = root / "experiments" / "runs"
    eight = (
        runs
        / "nanogpt30m-e2e-sgd-full-rpath-factorization-811-fblr-v001"
        / "results"
        / "a001.json"
    )
    wsd = (
        runs
        / "nanogpt30m-e2e-sgd-full-rpath-factorization-wsd-fblr-v001"
        / "results"
        / "a002.json"
    )
    output = root / "experiments" / "results" / ANALYSIS_ID
    return eight, wsd, output


def main(argv: Sequence[str] | None = None) -> int:
    root = Path(__file__).resolve().parents[2]
    default_eight, default_wsd, default_output = _default_paths(root)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eight-result", type=Path, default=default_eight)
    parser.add_argument("--wsd-result", type=Path, default=default_wsd)
    parser.add_argument("--output-dir", type=Path, default=default_output)
    parser.add_argument("--reuse-existing-fits", action="store_true")
    args = parser.parse_args(argv)
    result = analyze(
        args.eight_result.expanduser().resolve(),
        args.wsd_result.expanduser().resolve(),
        args.output_dir.expanduser().resolve(),
        reuse_existing_fits=args.reuse_existing_fits,
    )
    print(json.dumps(result["comparison"], indent=2, sort_keys=True))
    print(args.output_dir.expanduser().resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
