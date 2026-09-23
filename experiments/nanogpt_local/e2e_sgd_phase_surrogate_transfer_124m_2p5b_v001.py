"""Compare LM, IM, and FB empirical surrogates on the 124M campaign.

All models are selected using only the fixed-batch 8-1-1 validation-CE curve.
Their frozen parameters are then evaluated on WSD and on the matched
fixed-LR/batch factorizations.  This is a post-hoc, single-seed A2_EXTERNAL
diagnostic; it is not a theorem-facing phase identification experiment.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.fft import irfft, next_fast_len, rfft
from scipy.optimize import least_squares

from experiments.nanogpt_local import e2e_sgd_im_qk_profile_124m_2p5b_v001 as im124
from experiments.nanogpt_local import e2e_sgd_im_qk_profile_v001 as fixed_q
from experiments.nanogpt_local import e2e_sgd_phase_surrogate_transfer_v001 as phase
from experiments.nanogpt_local import e2e_sgd_rpath_factorization_124m_2p5b_analysis_v001 as validator
from experiments.nanogpt_local import e2e_sgd_rpath_factorization_124m_2p5b_v001 as core


Json = dict[str, Any]

ANALYSIS_ID = "nanogpt124m-e2e-sgd-phase-surrogate-transfer-v001"
LM_Q_K_BOUNDS = (0.02, 0.98)
MODEL_NAMES = ("LM", "IM", "FB")
MODEL_COLORS = {"LM": "#2a6fbb", "IM": "#d1495b", "FB": "#2f8f5b"}
MODEL_LINE_STYLES = {
    "LM": {"linestyle": "--", "linewidth": 3.0, "zorder": 5},
    "IM": {"linestyle": "-", "linewidth": 1.8, "zorder": 4},
    "FB": {"linestyle": "-.", "linewidth": 2.0, "zorder": 3},
}


def _read_csv(path: Path) -> list[Json]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty CSV: {path}")
    fieldnames = list(rows[0])
    for row in rows[1:]:
        fieldnames.extend(key for key in row if key not in fieldnames)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _fb_features(curve: phase.EvaluationCurve, q_f: float) -> np.ndarray:
    _, injection = phase._macro_quadrature(curve.shape_id)
    counts = np.asarray(curve.macro_index, dtype=np.int64)
    maximum = int(np.max(counts, initial=0))
    if maximum > injection.size or np.any(counts < 0):
        raise ValueError("invalid macro count for FB features")
    cumulative = np.concatenate(
        [np.zeros(1, dtype=np.float64), np.cumsum(injection[:maximum])]
    )
    clean = np.power(1.0 + curve.intrinsic_time, -q_f)
    return np.column_stack(
        [np.ones(curve.loss.size, dtype=np.float64), clean, cumulative[counts]]
    )


def _fb_parameters(vector: Sequence[float]) -> Json:
    return {
        "L_inf": float(vector[0]),
        "clean_amplitude": float(vector[1]),
        "noise_amplitude": float(vector[2]),
        "q_F": float(vector[3]),
    }


def _predict_fb(curve: phase.EvaluationCurve, parameters: Mapping[str, float]) -> np.ndarray:
    coefficients = np.asarray(
        [
            parameters["L_inf"],
            parameters["clean_amplitude"],
            parameters["noise_amplitude"],
        ],
        dtype=np.float64,
    )
    return _fb_features(curve, float(parameters["q_F"])) @ coefficients


def _fit_fb(curve: phase.EvaluationCurve, mask: np.ndarray) -> Json:
    target = np.asarray(curve.loss[mask], dtype=np.float64)
    target_time = np.asarray(curve.intrinsic_time[mask], dtype=np.float64)
    target_log_time = np.log1p(target_time)
    _, injection = phase._macro_quadrature(curve.shape_id)
    counts = np.asarray(curve.macro_index[mask], dtype=np.int64)
    maximum = int(np.max(counts, initial=0))
    cumulative = np.concatenate(
        [np.zeros(1, dtype=np.float64), np.cumsum(injection[:maximum])]
    )
    accumulated_noise = cumulative[counts]
    lower = np.asarray([0.0, 0.0, 0.0, 0.02], dtype=np.float64)
    upper = np.asarray([12.0, 40.0, 1e5, 4.0], dtype=np.float64)

    cache_vector: np.ndarray | None = None
    cache_residual: np.ndarray | None = None
    cache_jacobian: np.ndarray | None = None

    def evaluate(vector: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        nonlocal cache_vector, cache_residual, cache_jacobian
        if cache_vector is not None and np.array_equal(vector, cache_vector):
            assert cache_residual is not None and cache_jacobian is not None
            return cache_residual, cache_jacobian
        level, clean_amplitude, noise_amplitude, q_f = vector
        clean = np.exp(-q_f * target_log_time)
        prediction = level + clean_amplitude * clean + noise_amplitude * accumulated_noise
        if np.any(prediction <= 0.0):
            residual = np.full(target.size, 1e6, dtype=np.float64)
            jacobian = np.zeros((target.size, vector.size), dtype=np.float64)
        else:
            derivative = np.column_stack(
                [
                    np.ones(target.size, dtype=np.float64),
                    clean,
                    accumulated_noise,
                    -clean_amplitude * target_log_time * clean,
                ]
            )
            residual = np.log(prediction) - np.log(target)
            jacobian = derivative / prediction[:, None]
        cache_vector = vector.copy()
        cache_residual = residual
        cache_jacobian = jacobian
        return residual, jacobian

    solutions = []
    for q_f in (0.10, 0.30, 0.70, 1.50):
        level, amplitude = phase._initial_clean_coefficients(curve, mask, q_f)
        for noise in (0.05, 1.0):
            start = np.asarray([level, amplitude, noise, q_f], dtype=np.float64)
            fit = least_squares(
                lambda vector: evaluate(vector)[0],
                np.clip(start, lower + 1e-10, upper - 1e-10),
                jac=lambda vector: evaluate(vector)[1],
                bounds=(lower, upper),
                loss="huber",
                f_scale=1e-3,
                max_nfev=600,
                xtol=1e-8,
                ftol=1e-8,
                gtol=1e-8,
            )
            rss = float(np.sum(np.square(evaluate(fit.x)[0])))
            solutions.append((rss, fit))
    rss, fit = min(solutions, key=lambda item: item[0])
    parameters = _fb_parameters(fit.x)
    return {
        "model": "FB",
        "parameter_count": 4,
        "parameters": parameters,
        "optimizer": {
            "success": bool(fit.success),
            "status": int(fit.status),
            "message": str(fit.message),
            "function_evaluations": int(fit.nfev),
            "multistarts": len(solutions),
            "log_residual_rss": rss,
        },
        "boundary_diagnostics": {
            "noise_contribution_active": parameters["noise_amplitude"] > 1e-8,
            "q_F_at_lower_bound": parameters["q_F"] <= 0.02001,
            "q_F_at_upper_bound": parameters["q_F"] >= 3.999,
        },
    }


def _ordinary_parameters_from_profile_row(row: Mapping[str, Any]) -> Json:
    return {
        "L_inf": float(row["L_inf"]),
        "clean_amplitude": float(row["clean_amplitude"]),
        "noise_floor_amplitude": float(row["noise_floor_amplitude"]),
        "fit_dependent_noise_amplitude": float(row["fit_dependent_noise_amplitude"]),
        "kernel_scale": float(row["kernel_scale"]),
        "q_F": float(row["q_F"]),
        "q_K": float(row["q_K"]),
    }


class _FftFreeQEvaluator:
    """Exact ordinary-memory residual and Jacobian with free q_K."""

    def __init__(self, curve: phase.EvaluationCurve, mask: np.ndarray) -> None:
        source_time, injection = phase._macro_quadrature(curve.shape_id)
        self.counts = np.asarray(curve.macro_index[mask], dtype=np.int64)
        maximum = int(np.max(self.counts, initial=0))
        if maximum <= 0 or maximum > source_time.size:
            raise ValueError("invalid macro counts for FFT convolution")
        self.source_time = np.asarray(source_time[:maximum], dtype=np.float64)
        self.injection = np.asarray(injection[:maximum], dtype=np.float64)
        self.target_time = np.asarray(curve.intrinsic_time[mask], dtype=np.float64)
        macro_dt = float(2.0 * self.source_time[0])
        if not np.allclose(
            self.target_time,
            self.counts.astype(np.float64) * macro_dt,
            rtol=0.0,
            atol=1e-12,
        ):
            raise ValueError("FFT convolution requires exact macro coordinates")
        self.kernel_age = (np.arange(maximum, dtype=np.float64) + 0.5) * macro_dt
        self.source_log_time = np.log1p(self.source_time)
        self.target_log_time = np.log1p(self.target_time)
        self.nfft = int(next_fast_len(2 * maximum - 1, real=True))
        self.injection_fft = rfft(self.injection, n=self.nfft)
        self.target = np.asarray(curve.loss[mask], dtype=np.float64)
        self.cache_vector: np.ndarray | None = None
        self.cache_residual: np.ndarray | None = None
        self.cache_jacobian: np.ndarray | None = None

    def evaluate(self, vector: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if self.cache_vector is not None and np.array_equal(vector, self.cache_vector):
            assert self.cache_residual is not None and self.cache_jacobian is not None
            return self.cache_residual, self.cache_jacobian

        level, clean_amplitude, noise_floor, fit_noise, log_c_k, q_f, q_k = vector
        c_k = math.exp(float(log_c_k))
        clean = np.exp(-q_f * self.target_log_time)
        source = np.exp(-q_f * self.source_log_time)
        scaled_age = c_k * self.kernel_age
        log_kernel_base = np.log1p(scaled_age)
        kernel = np.exp(-q_k * log_kernel_base)
        kernel_fft = rfft(kernel, n=self.nfft)
        source_injection_fft = rfft(self.injection * source, n=self.nfft)
        s0 = fixed_q._fft_sample(
            self.injection_fft * kernel_fft, self.nfft, self.counts
        )
        s1 = fixed_q._fft_sample(
            source_injection_fft * kernel_fft, self.nfft, self.counts
        )
        prediction = level + clean_amplitude * clean + noise_floor * s0 + fit_noise * s1
        if np.any(prediction <= 0.0):
            residual = np.full(self.target.size, 1e6, dtype=np.float64)
            jacobian = np.zeros((self.target.size, vector.size), dtype=np.float64)
        else:
            combined_noise_fft = (
                noise_floor * self.injection_fft + fit_noise * source_injection_fft
            )
            dkernel_dlog_c = -q_k * kernel * scaled_age / (1.0 + scaled_age)
            dprediction_dlog_c = fixed_q._fft_sample(
                combined_noise_fft * rfft(dkernel_dlog_c, n=self.nfft),
                self.nfft,
                self.counts,
            )
            dkernel_dq_k = -kernel * log_kernel_base
            dprediction_dq_k = fixed_q._fft_sample(
                combined_noise_fft * rfft(dkernel_dq_k, n=self.nfft),
                self.nfft,
                self.counts,
            )
            dclean_dq_f = -self.target_log_time * clean
            dsource_dq_f_fft = rfft(
                self.injection * (-self.source_log_time * source), n=self.nfft
            )
            ds1_dq_f = fixed_q._fft_sample(
                dsource_dq_f_fft * kernel_fft, self.nfft, self.counts
            )
            prediction_jacobian = np.column_stack(
                [
                    np.ones(self.target.size, dtype=np.float64),
                    clean,
                    s0,
                    s1,
                    dprediction_dlog_c,
                    clean_amplitude * dclean_dq_f + fit_noise * ds1_dq_f,
                    dprediction_dq_k,
                ]
            )
            residual = np.log(prediction) - np.log(self.target)
            jacobian = prediction_jacobian / prediction[:, None]
        self.cache_vector = vector.copy()
        self.cache_residual = residual
        self.cache_jacobian = jacobian
        return residual, jacobian


def _fit_lm(
    curve: phase.EvaluationCurve,
    mask: np.ndarray,
    warm_parameters: Mapping[str, float],
) -> Json:
    evaluator = _FftFreeQEvaluator(curve, mask)
    lower = np.asarray(
        [0.0, 0.0, 0.0, 0.0, im124.LOG_C_BOUNDS[0], 0.02, LM_Q_K_BOUNDS[0]],
        dtype=np.float64,
    )
    upper = np.asarray(
        [12.0, 40.0, 1e5, 1e5, im124.LOG_C_BOUNDS[1], 4.0, LM_Q_K_BOUNDS[1]],
        dtype=np.float64,
    )
    reference_q = float(warm_parameters["q_K"])
    projected_q = LM_Q_K_BOUNDS[1] - 1e-4
    starts = [
        np.asarray(
            [
                warm_parameters["L_inf"],
                warm_parameters["clean_amplitude"],
                warm_parameters["noise_floor_amplitude"],
                warm_parameters["fit_dependent_noise_amplitude"],
                math.log(
                    float(warm_parameters["kernel_scale"])
                    * reference_q
                    / projected_q
                ),
                warm_parameters["q_F"],
                projected_q,
            ],
            dtype=np.float64,
        )
    ]
    for q_f, c_k, q_k, noise in (
        (0.10, 0.002, 0.20, 5.0),
        (0.20, 0.02, 0.50, 0.1),
        (0.40, 0.30, 0.85, 0.1),
        (0.80, 1.00, 0.95, 0.1),
    ):
        level, amplitude = phase._initial_clean_coefficients(curve, mask, q_f)
        starts.append(
            np.asarray(
                [level, amplitude, 0.1, noise, math.log(c_k), q_f, q_k],
                dtype=np.float64,
            )
        )

    solutions = []
    for start in starts:
        fit = least_squares(
            lambda vector: evaluator.evaluate(vector)[0],
            np.clip(start, lower + 1e-10, upper - 1e-10),
            jac=lambda vector: evaluator.evaluate(vector)[1],
            bounds=(lower, upper),
            loss="huber",
            f_scale=1e-3,
            max_nfev=600,
            xtol=1e-8,
            ftol=1e-8,
            gtol=1e-8,
            x_scale="jac",
        )
        rss = float(np.sum(np.square(evaluator.evaluate(fit.x)[0])))
        solutions.append((rss, fit))
    rss, fit = min(solutions, key=lambda item: item[0])
    parameters = fixed_q._parameters(fit.x[:6], float(fit.x[6]))
    near = [
        fixed_q._parameters(candidate.x[:6], float(candidate.x[6]))
        for score, candidate in solutions
        if score <= rss * (1.0 + 1e-4) + 1e-12
    ]
    return {
        "model": "LM",
        "parameter_count": 7,
        "parameters": parameters,
        "optimizer": {
            "success": bool(fit.success),
            "status": int(fit.status),
            "message": str(fit.message),
            "function_evaluations": int(fit.nfev),
            "multistarts": len(solutions),
            "log_residual_rss": rss,
        },
        "boundary_diagnostics": {
            "q_K_lower": LM_Q_K_BOUNDS[0],
            "q_K_upper": LM_Q_K_BOUNDS[1],
            "q_K_fraction_through_interval": float(
                (parameters["q_K"] - LM_Q_K_BOUNDS[0])
                / (LM_Q_K_BOUNDS[1] - LM_Q_K_BOUNDS[0])
            ),
            "q_K_at_lower_bound": parameters["q_K"] <= LM_Q_K_BOUNDS[0] + 1e-4,
            "q_K_at_upper_bound": parameters["q_K"] >= LM_Q_K_BOUNDS[1] - 1e-4,
            "near_optimal_solution_count": len(near),
            "near_optimal_q_K_min": min(item["q_K"] for item in near),
            "near_optimal_q_K_max": max(item["q_K"] for item in near),
            "noise_floor_contribution_active": parameters["noise_floor_amplitude"] > 1e-8,
            "fit_dependent_contribution_active": parameters[
                "fit_dependent_noise_amplitude"
            ] > 1e-8,
        },
    }


def _prediction(
    model: str,
    curve: phase.EvaluationCurve,
    parameters: Mapping[str, float],
) -> np.ndarray:
    if model == "FB":
        return _predict_fb(curve, parameters)
    return fixed_q._predict_curve(curve, parameters)


def _trapezoid_weights(coordinate: np.ndarray) -> np.ndarray:
    values = np.asarray(coordinate, dtype=np.float64)
    if values.ndim != 1 or values.size < 2 or not np.all(np.diff(values) > 0.0):
        raise ValueError("trapezoid coordinate must be a strict vector of length >= 2")
    weights = np.empty(values.size, dtype=np.float64)
    weights[0] = 0.5 * (values[1] - values[0])
    weights[-1] = 0.5 * (values[-1] - values[-2])
    weights[1:-1] = 0.5 * (values[2:] - values[:-2])
    weights /= np.sum(weights)
    return weights


def _weighted_metrics(
    target: np.ndarray,
    prediction: np.ndarray,
    weights: np.ndarray,
) -> Json:
    residual = np.asarray(prediction) - np.asarray(target)
    log_residual = np.log(prediction) - np.log(target)
    normalized = np.asarray(weights, dtype=np.float64) / np.sum(weights)
    return {
        "count": int(target.size),
        "effective_weight_sum": float(np.sum(normalized)),
        "rmse_ce": float(np.sqrt(np.sum(normalized * np.square(residual)))),
        "mae_ce": float(np.sum(normalized * np.abs(residual))),
        "rmse_log_ce": float(
            np.sqrt(np.sum(normalized * np.square(log_residual)))
        ),
        "bias_ce": float(np.sum(normalized * residual)),
    }


def _metric_windows(
    model: str,
    parameters: Mapping[str, float],
    parameter_count: int,
    curves: Mapping[str, phase.EvaluationCurve],
    masks: Mapping[str, np.ndarray],
) -> tuple[Json, dict[str, np.ndarray]]:
    predictions = {
        arm_id: _prediction(model, curve, parameters)
        for arm_id, curve in curves.items()
    }
    eight = curves["eight_one_one__fixed_batch_lr"]
    eight_factorized = curves["eight_one_one__fixed_lr_batch"]
    wsd = curves["wsd_exp_80_20__fixed_batch_lr"]
    wsd_factorized = curves["wsd_exp_80_20__fixed_lr_batch"]
    primary_time = eight.intrinsic_time[masks["eight_primary"]]
    log_time_weights = _trapezoid_weights(np.log(primary_time))
    return (
        {
            "eight_primary_fit": phase._metrics(
                eight.loss[masks["eight_primary"]],
                predictions["eight_one_one__fixed_batch_lr"][masks["eight_primary"]],
                parameter_count=parameter_count,
            ),
            "eight_primary_frozen_fit_log_time_balanced_sensitivity": _weighted_metrics(
                eight.loss[masks["eight_primary"]],
                predictions["eight_one_one__fixed_batch_lr"][masks["eight_primary"]],
                log_time_weights,
            ),
            "eight_all_positive_sensitivity": phase._metrics(
                eight.loss[masks["eight_all_positive"]],
                predictions["eight_one_one__fixed_batch_lr"][masks["eight_all_positive"]],
            ),
            "eight_fixed_lr_batch_primary_zero_refit": phase._metrics(
                eight_factorized.loss[masks["eight_primary"]],
                predictions["eight_one_one__fixed_lr_batch"][masks["eight_primary"]],
            ),
            "wsd_tail_zero_refit": phase._metrics(
                wsd.loss[masks["wsd_tail"]],
                predictions["wsd_exp_80_20__fixed_batch_lr"][masks["wsd_tail"]],
            ),
            "wsd_tail_overlap_zero_refit": phase._metrics(
                wsd.loss[masks["wsd_tail_overlap"]],
                predictions["wsd_exp_80_20__fixed_batch_lr"][masks["wsd_tail_overlap"]],
            ),
            "wsd_tail_extrapolation_zero_refit": phase._metrics(
                wsd.loss[masks["wsd_tail_extrapolation"]],
                predictions["wsd_exp_80_20__fixed_batch_lr"][masks["wsd_tail_extrapolation"]],
            ),
            "wsd_fixed_lr_batch_tail_zero_refit": phase._metrics(
                wsd_factorized.loss[masks["wsd_tail"]],
                predictions["wsd_exp_80_20__fixed_lr_batch"][masks["wsd_tail"]],
            ),
        },
        predictions,
    )


def _render_boundary_diagnostic(
    output_directory: Path,
    lm_fit: Mapping[str, Any],
    im_rows: Sequence[Mapping[str, Any]],
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11.4, 4.6))
    q = np.asarray([float(row["q_K"]) for row in im_rows])
    fit = np.asarray([float(row["eight_primary_rmse_ce"]) for row in im_rows])
    transfer = np.asarray([float(row["wsd_tail_rmse_ce"]) for row in im_rows])
    axes[0].plot(q, fit, "o-", color=MODEL_COLORS["IM"], label="IM: 8-1-1 fit")
    axes[0].plot(
        q,
        transfer,
        "s--",
        color=MODEL_COLORS["IM"],
        alpha=0.8,
        label="IM: WSD zero-refit",
    )
    lm_q = float(lm_fit["parameters"]["q_K"])
    axes[0].plot(
        [lm_q],
        [lm_fit["metrics"]["eight_primary_fit"]["rmse_ce"]],
        marker="*",
        ms=13,
        color=MODEL_COLORS["LM"],
        ls="none",
        label="LM constrained optimum: fit",
    )
    axes[0].plot(
        [lm_q],
        [lm_fit["metrics"]["wsd_tail_zero_refit"]["rmse_ce"]],
        marker="P",
        ms=9,
        color=MODEL_COLORS["LM"],
        ls="none",
        label="LM constrained optimum: WSD",
    )
    axes[1].plot(
        q,
        [float(row["kernel_scale"]) for row in im_rows],
        "o-",
        color=MODEL_COLORS["IM"],
        label="IM profile",
    )
    axes[1].plot(
        [lm_q],
        [lm_fit["parameters"]["kernel_scale"]],
        marker="*",
        ms=13,
        color=MODEL_COLORS["LM"],
        ls="none",
        label="LM constrained optimum",
    )
    for axis in axes:
        axis.axvline(1.0, color="#333333", lw=1.0, ls=":", label=r"$q_{\mathcal{K}}=1$")
        axis.set_xscale("log")
        axis.set_xlabel(r"Fixed $q_{\mathcal{K}}$")
        axis.grid(alpha=0.22)
    axes[0].set_ylabel("Validation CE RMSE")
    axes[0].set_title("Fit and schedule-transfer error")
    axes[1].set_yscale("log")
    axes[1].set_ylabel(r"Fitted kernel scale $c_{\mathcal{K}}$")
    axes[1].set_title("Exponent-scale compensation")
    axes[0].legend(fontsize=8)
    axes[1].legend(fontsize=8)
    fig.suptitle("124M/2.5B ordinary-memory diagnostic near the LM/IM boundary")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(output_directory / "ordinary_phase_boundary.png", dpi=180)
    fig.savefig(output_directory / "ordinary_phase_boundary.pdf")
    plt.close(fig)


def _render_models(
    output_directory: Path,
    curves: Mapping[str, phase.EvaluationCurve],
    masks: Mapping[str, np.ndarray],
    predictions: Mapping[str, Mapping[str, np.ndarray]],
) -> None:
    eight = curves["eight_one_one__fixed_batch_lr"]
    wsd = curves["wsd_exp_80_20__fixed_batch_lr"]
    panels = (
        (eight, masks["eight_primary"], "eight_one_one__fixed_batch_lr", "8-1-1 fit window"),
        (wsd, masks["wsd_tail"], "wsd_exp_80_20__fixed_batch_lr", "WSD zero-refit tail"),
    )
    fig, axes = plt.subplots(2, 2, figsize=(12.4, 8.0), sharex="col")
    for column, (curve, mask, arm_id, title) in enumerate(panels):
        x = curve.intrinsic_time[mask]
        observed = curve.loss[mask]
        axes[0, column].plot(
            x,
            observed,
            "o",
            color="#111111",
            ms=3.0,
            alpha=0.75,
            zorder=6,
            label="observed",
        )
        for model in MODEL_NAMES:
            value = predictions[model][arm_id][mask]
            axes[0, column].plot(
                x,
                value,
                color=MODEL_COLORS[model],
                label=model,
                **MODEL_LINE_STYLES[model],
            )
            axes[1, column].plot(
                x,
                value - observed,
                color=MODEL_COLORS[model],
                label=model,
                **MODEL_LINE_STYLES[model],
            )
        axes[0, column].set_title(title)
        axes[0, column].set_ylabel("Validation CE")
        axes[1, column].axhline(0.0, color="#111111", lw=0.8)
        axes[1, column].set_xlabel("Intrinsic time")
        axes[1, column].set_ylabel("Prediction - observed CE")
        for row in range(2):
            axes[row, column].grid(alpha=0.22)
            axes[row, column].legend(fontsize=8)
    fig.suptitle("124M/2.5B LM, IM, and FB empirical surrogates")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(output_directory / "phase_fit_and_zero_refit_transfer.png", dpi=180)
    fig.savefig(output_directory / "phase_fit_and_zero_refit_transfer.pdf")
    plt.close(fig)


def analyze(
    prefix_directory: Path,
    tail_directories: Mapping[str, Path],
    im_profile_directory: Path,
    output_directory: Path,
    reuse_existing_fits: bool = False,
) -> Json:
    output_directory.mkdir(parents=True, exist_ok=True)
    campaign_validation = validator.validate_campaign(
        prefix_directory,
        [tail_directories[arm_id] for arm_id in core.ARM_IDS],
    )
    if campaign_validation["status"] != "passed":
        raise validator.AnalysisError("124M campaign validation failed")
    curves = {
        arm_id: im124._combine_prefix_tail(prefix_directory, directory)
        for arm_id, directory in tail_directories.items()
    }
    eight = curves["eight_one_one__fixed_batch_lr"]
    wsd = curves["wsd_exp_80_20__fixed_batch_lr"]
    if not np.array_equal(
        eight.intrinsic_time,
        curves["eight_one_one__fixed_lr_batch"].intrinsic_time,
    ) or not np.array_equal(
        wsd.intrinsic_time,
        curves["wsd_exp_80_20__fixed_lr_batch"].intrinsic_time,
    ):
        raise validator.AnalysisError("same-shape factorization grids differ")

    im_profile_path = im_profile_directory / "im_qk_profile.csv"
    im_rows = im124._read_profile_csv(im_profile_path)
    best_im_row = min(im_rows, key=lambda row: float(row["eight_primary_rmse_ce"]))
    im_parameters = _ordinary_parameters_from_profile_row(best_im_row)

    with im124._use_124m_schedule_core():
        phase._validate_pair(eight, wsd)
        masks = phase._window_masks(eight, wsd)
        fit_checkpoint = output_directory / "fit_checkpoint.json"
        if reuse_existing_fits:
            checkpoint = phase._read_json(fit_checkpoint)
            fits = checkpoint["fits"]
            if set(fits) != set(MODEL_NAMES):
                raise ValueError("fit checkpoint has unexpected models")
        else:
            lm_fit = _fit_lm(eight, masks["eight_primary"], im_parameters)
            fb_fit = _fit_fb(eight, masks["eight_primary"])
            fits: dict[str, Json] = {
                "LM": lm_fit,
                "IM": {
                    "model": "IM",
                    "parameter_count": 7,
                    "parameters": im_parameters,
                    "selection": {
                        "rule": "minimum 8-1-1 primary-window RMSE over precomputed IM grid",
                        "q_K_grid": list(fixed_q.Q_K_GRID),
                        "selected_q_K": float(best_im_row["q_K"]),
                        "selected_at_grid_boundary": float(best_im_row["q_K"])
                        in (fixed_q.Q_K_GRID[0], fixed_q.Q_K_GRID[-1]),
                        "source_profile": str(im_profile_path),
                        "source_profile_sha256": phase._sha256_file(im_profile_path),
                    },
                    "optimizer": {
                        "reused_precomputed_fixed_q_fit": True,
                        "log_residual_rss": float(best_im_row["log_residual_rss"]),
                    },
                },
                "FB": fb_fit,
            }
            fit_checkpoint.write_bytes(phase._canonical_bytes({"fits": fits}))

        all_predictions: dict[str, dict[str, np.ndarray]] = {}
        metric_rows: list[Json] = []
        prediction_rows: list[Json] = []
        for model in MODEL_NAMES:
            metrics, model_predictions = _metric_windows(
                model,
                fits[model]["parameters"],
                int(fits[model]["parameter_count"]),
                curves,
                masks,
            )
            fits[model]["metrics"] = metrics
            all_predictions[model] = model_predictions
            for window, values in metrics.items():
                metric_rows.append({"model": model, "window": window, **values})
            for arm_id, curve in curves.items():
                prediction = model_predictions[arm_id]
                for index in range(curve.loss.size):
                    prediction_rows.append(
                        {
                            "model": model,
                            "arm_id": arm_id,
                            "update": int(curve.update[index]),
                            "macro_index": int(curve.macro_index[index]),
                            "intrinsic_time": float(curve.intrinsic_time[index]),
                            "observed_validation_ce": float(curve.loss[index]),
                            "predicted_validation_ce": float(prediction[index]),
                            "residual_ce": float(prediction[index] - curve.loss[index]),
                            "validation_standard_error": float(
                                curve.standard_error[index]
                            ),
                        }
                    )

        for row in im_rows:
            if "wsd_tail_rmse_ce" not in row:
                raise ValueError("IM profile lacks WSD transfer metrics")
        _render_boundary_diagnostic(output_directory, fits["LM"], im_rows)
        _render_models(output_directory, curves, masks, all_predictions)

    fit_ranking = sorted(
        MODEL_NAMES,
        key=lambda model: fits[model]["metrics"]["eight_primary_fit"][
            "aicc_log_residual"
        ],
    )
    transfer_ranking = sorted(
        MODEL_NAMES,
        key=lambda model: fits[model]["metrics"]["wsd_tail_zero_refit"][
            "rmse_ce"
        ],
    )
    balanced_fit_ranking = sorted(
        MODEL_NAMES,
        key=lambda model: fits[model]["metrics"][
            "eight_primary_frozen_fit_log_time_balanced_sensitivity"
        ]["rmse_ce"],
    )
    overlap_ranking = sorted(
        MODEL_NAMES,
        key=lambda model: fits[model]["metrics"]["wsd_tail_overlap_zero_refit"][
            "rmse_ce"
        ],
    )
    extrapolation_ranking = sorted(
        MODEL_NAMES,
        key=lambda model: fits[model]["metrics"][
            "wsd_tail_extrapolation_zero_refit"
        ]["rmse_ce"],
    )
    best_fit_model = fit_ranking[0]
    best_transfer_model = transfer_ranking[0]
    lm_q = float(fits["LM"]["parameters"]["q_K"])
    im_q = float(fits["IM"]["parameters"]["q_K"])
    ordinary_boundary_unresolved = (
        lm_q >= LM_Q_K_BOUNDS[1] - 1e-4 and im_q == fixed_q.Q_K_GRID[0]
    )
    fit_times = eight.intrinsic_time[masks["eight_primary"]]
    last_log_decile_threshold = float(
        math.exp(
            math.log(float(fit_times[0]))
            + 0.9 * (math.log(float(fit_times[-1])) - math.log(float(fit_times[0])))
        )
    )
    fork_time = core.CONSTANT_SOURCE_UPDATES * core.CANONICAL_BASE_LEARNING_RATE
    eight_tail = eight.intrinsic_time >= fork_time - 1e-12
    eight_factorized = curves["eight_one_one__fixed_lr_batch"]
    wsd_factorized = curves["wsd_exp_80_20__fixed_lr_batch"]

    def factorization_metrics(
        left: phase.EvaluationCurve,
        right: phase.EvaluationCurve,
        mask: np.ndarray,
    ) -> Json:
        return phase._metrics(left.loss[mask], right.loss[mask])

    summary: Json = {
        "schema_version": "nanogpt124m_e2e_sgd_phase_surrogate_transfer_v001",
        "analysis_id": ANALYSIS_ID,
        "status": "completed",
        "classification": "post_hoc_exploratory_single_seed_A2_EXTERNAL",
        "new_gpu_hours_consumed_by_analysis": 0.0,
        "claim_boundary": {
            "allowed": [
                "compare constrained empirical LM, IM, and FB surrogate fits",
                "compare zero-refit schedule and factorization transfer errors",
                "diagnose whether the ordinary-memory optimum is resolved across q_K=1",
            ],
            "prohibited": [
                "claim theorem-facing LM, IM, or FB phase identification",
                "report a stationary q_K estimate",
                "treat validation ticks sharing fixed contexts as independent replicates",
            ],
        },
        "fit_protocol": {
            "fit_arm": "eight_one_one__fixed_batch_lr",
            "fit_target": "fixed_1024_context_validation_probe_cross_entropy",
            "clock": "intrinsic_time",
            "primary_window_rule": phase.PRIMARY_WINDOW_RULE,
            "primary_start_intrinsic_time": float(
                eight.intrinsic_time[masks["eight_primary"]][0]
            ),
            "primary_end_intrinsic_time": float(eight.intrinsic_time[-1]),
            "primary_evaluation_rows": int(np.sum(masks["eight_primary"])),
            "sampling_density_diagnostic": {
                "rows_at_or_after_fork": int(np.sum(eight_tail & masks["eight_primary"])),
                "last_log_time_decile_start": last_log_decile_threshold,
                "rows_in_last_log_time_decile": int(
                    np.sum(fit_times >= last_log_decile_threshold)
                ),
                "note": (
                    "The primary optimizer remains the legacy unweighted row-wise fit. "
                    "A frozen-fit log-time trapezoid metric is reported as a sensitivity."
                ),
            },
            "objective": "Huber robust least squares on log validation CE",
            "model_parameter_counts_for_descriptive_AICc": {
                "LM": 7,
                "IM": 7,
                "FB": 4,
            },
            "ordinary_kernel_log_scale_bounds": list(im124.LOG_C_BOUNDS),
            "convolution": "exact FFT replay of DeltaT=0.02 midpoint surrogate",
            "parameter_refit_on_transfer_curves": False,
        },
        "fits": fits,
        "comparison": {
            "descriptive_in_sample_aicc_ranking": fit_ranking,
            "frozen_fit_log_time_balanced_rmse_ranking": balanced_fit_ranking,
            "zero_refit_wsd_tail_rmse_ranking": transfer_ranking,
            "zero_refit_wsd_overlap_rmse_ranking": overlap_ranking,
            "zero_refit_wsd_extrapolation_rmse_ranking": extrapolation_ranking,
            "best_in_sample_model": best_fit_model,
            "best_zero_refit_wsd_tail_model": best_transfer_model,
            "ordinary_optimum_brackets_q_K_1": ordinary_boundary_unresolved,
            "absolute_lm_im_fit_rmse_difference": float(
                abs(
                    fits["LM"]["metrics"]["eight_primary_fit"]["rmse_ce"]
                    - fits["IM"]["metrics"]["eight_primary_fit"]["rmse_ce"]
                )
            ),
            "absolute_lm_im_wsd_tail_rmse_difference": float(
                abs(
                    fits["LM"]["metrics"]["wsd_tail_zero_refit"]["rmse_ce"]
                    - fits["IM"]["metrics"]["wsd_tail_zero_refit"]["rmse_ce"]
                )
            ),
            "median_wsd_tail_validation_standard_error": float(
                np.median(wsd.standard_error[masks["wsd_tail"]])
            ),
            "decision_label": "inconclusive_phase_identification",
            "interpretation": (
                "LM/IM/FB rankings are descriptive only. Boundary selection on both "
                "sides of q_K=1 means the ordinary-memory phase boundary is unresolved; "
                "one shared-context, single-seed trajectory cannot identify a theorem phase."
                if ordinary_boundary_unresolved
                else "LM/IM/FB rankings are descriptive only; one shared-context, "
                "single-seed trajectory cannot identify a theorem phase."
            ),
        },
        "factorization_diagnostics": {
            "eight_one_one_observed_curve_rmse_ce": float(
                np.sqrt(
                    np.mean(
                        np.square(
                            curves["eight_one_one__fixed_batch_lr"].loss
                            - curves["eight_one_one__fixed_lr_batch"].loss
                        )
                    )
                )
            ),
            "wsd_observed_curve_rmse_ce": float(
                np.sqrt(
                    np.mean(
                        np.square(
                            curves["wsd_exp_80_20__fixed_batch_lr"].loss
                            - curves["wsd_exp_80_20__fixed_lr_batch"].loss
                        )
                    )
                )
            ),
            "eight_one_one_tail": factorization_metrics(
                eight, eight_factorized, eight_tail
            ),
            "wsd_tail": factorization_metrics(
                wsd, wsd_factorized, masks["wsd_tail"]
            ),
            "wsd_overlap": factorization_metrics(
                wsd, wsd_factorized, masks["wsd_tail_overlap"]
            ),
            "wsd_extrapolation": factorization_metrics(
                wsd, wsd_factorized, masks["wsd_tail_extrapolation"]
            ),
        },
        "source": {
            "prefix_directory": str(prefix_directory),
            "tail_directories": {
                arm_id: str(path) for arm_id, path in tail_directories.items()
            },
            "im_profile_directory": str(im_profile_directory),
        },
        "campaign_validation": campaign_validation,
    }

    parameter_rows = []
    for model in MODEL_NAMES:
        for name, value in fits[model]["parameters"].items():
            parameter_rows.append({"model": model, "parameter": name, "value": value})
    _write_csv(output_directory / "selected_fit_parameters.csv", parameter_rows)
    _write_csv(output_directory / "phase_metrics.csv", metric_rows)
    _write_csv(output_directory / "predictions.csv", prediction_rows)
    (output_directory / "campaign_validation.json").write_bytes(
        phase._canonical_bytes(campaign_validation)
    )
    readme = f"""# {ANALYSIS_ID}

Post-hoc, single-seed A2 external-validity comparison on the completed 124M,
2.5B-token plain-SGD campaign. LM, IM, and FB are selected only on the same
fixed-batch 8-1-1 validation-CE window. WSD and both fixed-LR/batch curves are
strictly zero-refit diagnostics.

Selected results under the legacy unweighted fit objective:

- LM: `q_K={fits['LM']['parameters']['q_K']:.6g}` (upper boundary),
  8-1-1 RMSE `{fits['LM']['metrics']['eight_primary_fit']['rmse_ce']:.8g}`,
  WSD-tail RMSE `{fits['LM']['metrics']['wsd_tail_zero_refit']['rmse_ce']:.8g}`.
- IM: `q_K={fits['IM']['parameters']['q_K']:.6g}` (lower grid boundary),
  8-1-1 RMSE `{fits['IM']['metrics']['eight_primary_fit']['rmse_ce']:.8g}`,
  WSD-tail RMSE `{fits['IM']['metrics']['wsd_tail_zero_refit']['rmse_ce']:.8g}`.
- FB: 8-1-1 RMSE `{fits['FB']['metrics']['eight_primary_fit']['rmse_ce']:.8g}`,
  WSD-tail RMSE `{fits['FB']['metrics']['wsd_tail_zero_refit']['rmse_ce']:.8g}`.

The ordinary-memory optimum is bracketed by `q_K=0.98` and `q_K=1.02`.
Moreover, 145/156 fit rows lie in the last log-time decile; evaluating the
frozen fits with log-time trapezoid weights reverses the tiny LM/IM ordering.
The result therefore disfavors this FB empirical surrogate but does not
separate LM from IM.

The AICc values are descriptive because all evaluation ticks reuse the same
1,024 validation contexts and are temporally correlated. This artifact cannot
identify a theorem-facing phase or a stationary `q_K` by itself.
"""
    (output_directory / "README.md").write_text(readme, encoding="utf-8")
    artifact_names = (
        "campaign_validation.json",
        "fit_checkpoint.json",
        "selected_fit_parameters.csv",
        "phase_metrics.csv",
        "predictions.csv",
        "ordinary_phase_boundary.png",
        "ordinary_phase_boundary.pdf",
        "phase_fit_and_zero_refit_transfer.png",
        "phase_fit_and_zero_refit_transfer.pdf",
        "README.md",
    )
    summary["artifacts"] = {
        name: {
            "sha256": phase._sha256_file(output_directory / name),
            "bytes": (output_directory / name).stat().st_size,
        }
        for name in artifact_names
    }
    (output_directory / "summary.json").write_bytes(phase._canonical_bytes(summary))
    return summary


def _default_paths(root: Path) -> tuple[Path, dict[str, Path], Path, Path]:
    prefix, tails, _ = im124._default_paths(root)
    im_profile = (
        root
        / "experiments"
        / "results"
        / "nanogpt124m-e2e-sgd-im-qk-profile-v001"
    )
    output = root / "experiments" / "results" / ANALYSIS_ID
    return prefix, tails, im_profile, output


def main(argv: Sequence[str] | None = None) -> int:
    root = Path(__file__).resolve().parents[2]
    default_prefix, default_tails, default_im_profile, default_output = _default_paths(root)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prefix-directory", type=Path, default=default_prefix)
    parser.add_argument("--im-profile-directory", type=Path, default=default_im_profile)
    parser.add_argument("--output-directory", type=Path, default=default_output)
    parser.add_argument("--reuse-existing-fits", action="store_true")
    args = parser.parse_args(argv)
    result = analyze(
        args.prefix_directory.expanduser().resolve(),
        {arm_id: path.resolve() for arm_id, path in default_tails.items()},
        args.im_profile_directory.expanduser().resolve(),
        args.output_directory.expanduser().resolve(),
        reuse_existing_fits=args.reuse_existing_fits,
    )
    print(json.dumps(result["comparison"], indent=2, sort_keys=True))
    print(args.output_directory.expanduser().resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
