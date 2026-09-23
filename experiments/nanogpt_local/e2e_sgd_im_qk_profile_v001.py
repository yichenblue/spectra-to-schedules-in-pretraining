"""Profile the IM kernel exponent on the 30M 8-1-1/WSD transfer pair.

For every frozen q_K in a declared grid, refit the other six ordinary-memory
surrogate parameters on the same 8-1-1 validation-CE window used by the
phase-surrogate analysis.  Apply those parameters to WSD without refitting.
This is a post-hoc A2_EXTERNAL identifiability diagnostic, not an independent
kernel estimate or theorem-facing phase test.
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

from experiments.nanogpt_local import e2e_sgd_phase_surrogate_transfer_v001 as base


Json = dict[str, Any]

ANALYSIS_ID = "nanogpt30m-e2e-sgd-im-qk-profile-v001"
Q_K_GRID = (1.02, 1.10, 1.25, 1.50, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0, 20.0, 40.0)
SELECTED_PREDICTION_Q = (1.02, 1.50, 2.0, 6.0, 12.0, 40.0)
DENSE_CONVOLUTION_ELEMENT_LIMIT = 5_000_000


def _parameters(vector: Sequence[float], q_k: float) -> Json:
    return {
        "L_inf": float(vector[0]),
        "clean_amplitude": float(vector[1]),
        "noise_floor_amplitude": float(vector[2]),
        "fit_dependent_noise_amplitude": float(vector[3]),
        "kernel_scale": float(math.exp(vector[4])),
        "q_F": float(vector[5]),
        "q_K": float(q_k),
    }


def _fft_sample(product: np.ndarray, nfft: int, counts: np.ndarray) -> np.ndarray:
    convolution = irfft(product, n=nfft)
    result = np.zeros(counts.size, dtype=np.float64)
    positive = counts > 0
    result[positive] = convolution[counts[positive] - 1]
    return result


class _FftFixedQEvaluator:
    """Exact uniform-macro convolution and Jacobian for a fixed q_K."""

    def __init__(
        self,
        curve: base.EvaluationCurve,
        mask: np.ndarray,
        q_k: float,
    ) -> None:
        source_time, injection = base._macro_quadrature(curve.shape_id)
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
            raise ValueError("FFT convolution requires exact uniform macro coordinates")
        self.kernel_age = (np.arange(maximum, dtype=np.float64) + 0.5) * macro_dt
        self.source_log_time = np.log1p(self.source_time)
        self.target_log_time = np.log1p(self.target_time)
        self.nfft = int(next_fast_len(2 * maximum - 1, real=True))
        self.injection_fft = rfft(self.injection, n=self.nfft)
        self.q_k = float(q_k)
        self.target = np.asarray(curve.loss[mask], dtype=np.float64)
        self.cache_vector: np.ndarray | None = None
        self.cache_residual: np.ndarray | None = None
        self.cache_jacobian: np.ndarray | None = None

    def evaluate(self, vector: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if self.cache_vector is not None and np.array_equal(vector, self.cache_vector):
            assert self.cache_residual is not None and self.cache_jacobian is not None
            return self.cache_residual, self.cache_jacobian

        level, clean_amplitude, noise_floor, fit_noise, log_c_k, q_f = vector
        c_k = math.exp(float(log_c_k))
        clean = np.exp(-q_f * self.target_log_time)
        source = np.exp(-q_f * self.source_log_time)
        scaled_age = c_k * self.kernel_age
        kernel = np.exp(-self.q_k * np.log1p(scaled_age))
        kernel_fft = rfft(kernel, n=self.nfft)
        source_injection_fft = rfft(self.injection * source, n=self.nfft)
        s0 = _fft_sample(self.injection_fft * kernel_fft, self.nfft, self.counts)
        s1 = _fft_sample(source_injection_fft * kernel_fft, self.nfft, self.counts)
        prediction = level + clean_amplitude * clean + noise_floor * s0 + fit_noise * s1
        if np.any(prediction <= 0.0):
            residual_value = np.full(self.target.size, 1e6, dtype=np.float64)
            jacobian_value = np.zeros((self.target.size, vector.size), dtype=np.float64)
        else:
            dkernel_dlog_c = (
                -self.q_k * kernel * scaled_age / (1.0 + scaled_age)
            )
            dkernel_fft = rfft(dkernel_dlog_c, n=self.nfft)
            ds0_dlog_c = _fft_sample(
                self.injection_fft * dkernel_fft, self.nfft, self.counts
            )
            ds1_dlog_c = _fft_sample(
                source_injection_fft * dkernel_fft, self.nfft, self.counts
            )
            dclean_dq_f = -self.target_log_time * clean
            dsource_dq_f_fft = rfft(
                self.injection * (-self.source_log_time * source), n=self.nfft
            )
            ds1_dq_f = _fft_sample(
                dsource_dq_f_fft * kernel_fft, self.nfft, self.counts
            )
            prediction_jacobian = np.column_stack(
                [
                    np.ones(self.target.size),
                    clean,
                    s0,
                    s1,
                    noise_floor * ds0_dlog_c + fit_noise * ds1_dlog_c,
                    clean_amplitude * dclean_dq_f + fit_noise * ds1_dq_f,
                ]
            )
            residual_value = np.log(prediction) - np.log(self.target)
            jacobian_value = prediction_jacobian / prediction[:, None]
        self.cache_vector = vector.copy()
        self.cache_residual = residual_value
        self.cache_jacobian = jacobian_value
        return residual_value, jacobian_value


def _feature_matrix_fft(
    curve: base.EvaluationCurve,
    q_f: float,
    c_k: float,
    q_k: float,
) -> np.ndarray:
    source_time, injection = base._macro_quadrature(curve.shape_id)
    counts = np.asarray(curve.macro_index, dtype=np.int64)
    maximum = int(np.max(counts, initial=0))
    if maximum <= 0 or maximum > source_time.size:
        raise ValueError("invalid macro counts for FFT prediction")
    source_time = np.asarray(source_time[:maximum], dtype=np.float64)
    injection = np.asarray(injection[:maximum], dtype=np.float64)
    macro_dt = float(2.0 * source_time[0])
    if not np.allclose(
        curve.intrinsic_time,
        counts.astype(np.float64) * macro_dt,
        rtol=0.0,
        atol=1e-12,
    ):
        raise ValueError("FFT prediction requires exact uniform macro coordinates")
    kernel_age = (np.arange(maximum, dtype=np.float64) + 0.5) * macro_dt
    kernel = np.exp(-q_k * np.log1p(c_k * kernel_age))
    source = np.power(1.0 + source_time, -q_f)
    nfft = int(next_fast_len(2 * maximum - 1, real=True))
    kernel_fft = rfft(kernel, n=nfft)
    s0 = _fft_sample(rfft(injection, n=nfft) * kernel_fft, nfft, counts)
    s1 = _fft_sample(rfft(injection * source, n=nfft) * kernel_fft, nfft, counts)
    clean = np.power(1.0 + curve.intrinsic_time, -q_f)
    return np.column_stack([np.ones(curve.loss.size), clean, s0, s1])


def _predict_curve(
    curve: base.EvaluationCurve,
    parameters: Mapping[str, float],
    force_fft: bool | None = None,
) -> np.ndarray:
    maximum = int(np.max(curve.macro_index, initial=0))
    use_fft = (
        maximum * curve.loss.size > DENSE_CONVOLUTION_ELEMENT_LIMIT
        if force_fft is None
        else force_fft
    )
    if use_fft:
        matrix = _feature_matrix_fft(
            curve,
            parameters["q_F"],
            parameters["kernel_scale"],
            parameters["q_K"],
        )
    else:
        matrix = base._feature_matrix(
            curve,
            parameters["q_F"],
            "IM",
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


def _fit_fixed_q(
    curve: base.EvaluationCurve,
    mask: np.ndarray,
    q_k: float,
    warm_parameters: Mapping[str, float] | None = None,
    log_c_bounds: tuple[float, float] = (-9.0, 9.0),
    minimum_q_k: float = 1.0,
) -> Json:
    if not math.isfinite(minimum_q_k) or minimum_q_k < 0.0:
        raise ValueError("minimum_q_k must be finite and nonnegative")
    if not math.isfinite(q_k) or q_k <= minimum_q_k:
        raise ValueError(
            f"fixed q_K fit requires a finite q_K > {minimum_q_k:g}"
        )
    if not (
        len(log_c_bounds) == 2
        and math.isfinite(log_c_bounds[0])
        and math.isfinite(log_c_bounds[1])
        and log_c_bounds[0] < log_c_bounds[1]
    ):
        raise ValueError("log_c_bounds must be a finite increasing pair")
    lower = np.asarray(
        [0.0, 0.0, 0.0, 0.0, log_c_bounds[0], 0.02], dtype=np.float64
    )
    upper = np.asarray(
        [12.0, 40.0, 1e5, 1e5, log_c_bounds[1], 4.0], dtype=np.float64
    )
    target = curve.loss[mask]
    dense_elements = int(np.max(curve.macro_index, initial=0)) * int(np.sum(mask))
    fft_evaluator = (
        _FftFixedQEvaluator(curve, mask, q_k)
        if dense_elements > DENSE_CONVOLUTION_ELEMENT_LIMIT
        else None
    )
    if fft_evaluator is None:
        source_time, weights_all, ages_all = base._causal_grid(
            curve.shape_id,
            tuple(int(value) for value in curve.macro_index),
            tuple(float(value) for value in curve.intrinsic_time),
        )
        weights = weights_all[mask]
        ages = ages_all[mask]
        target_time = curve.intrinsic_time[mask]
        source_log_time = np.log1p(source_time)
        target_log_time = np.log1p(target_time)

    starts: list[np.ndarray] = []
    if warm_parameters is not None:
        reference_q = float(warm_parameters["q_K"])
        scaled_kernel = float(warm_parameters["kernel_scale"]) * reference_q / q_k
        starts.append(
            np.asarray(
                [
                    warm_parameters["L_inf"],
                    warm_parameters["clean_amplitude"],
                    warm_parameters["noise_floor_amplitude"],
                    warm_parameters["fit_dependent_noise_amplitude"],
                    math.log(max(scaled_kernel, math.exp(log_c_bounds[0]))),
                    warm_parameters["q_F"],
                ],
                dtype=np.float64,
            )
        )
    # These starts span clean-decay and kernel scales without adapting the grid
    # after seeing the profile result.
    for q_f, c_k, noise in (
        (0.10, 0.002, 5.0),
        (0.20, 0.02, 0.1),
        (0.40, 0.30, 0.1),
        (0.80, 1.00, 0.1),
    ):
        base_level, clean_amplitude = base._initial_clean_coefficients(curve, mask, q_f)
        starts.append(
            np.asarray(
                [base_level, clean_amplitude, 0.1, noise, math.log(c_k), q_f],
                dtype=np.float64,
            )
        )

    cache_vector: np.ndarray | None = None
    cache_residual: np.ndarray | None = None
    cache_jacobian: np.ndarray | None = None

    def evaluate(vector: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return log residual and its analytic Jacobian.

        SciPy otherwise evaluates this million-element convolution once per
        finite-difference column.  The exact Jacobian preserves the objective
        while removing those redundant kernel constructions.
        """

        if fft_evaluator is not None:
            return fft_evaluator.evaluate(vector)

        nonlocal cache_vector, cache_residual, cache_jacobian
        if cache_vector is not None and np.array_equal(vector, cache_vector):
            assert cache_residual is not None and cache_jacobian is not None
            return cache_residual, cache_jacobian

        level, clean_amplitude, noise_floor, fit_noise, log_c_k, q_f = vector
        c_k = math.exp(float(log_c_k))
        clean = np.exp(-q_f * target_log_time)
        source = np.exp(-q_f * source_log_time)
        scaled_age = c_k * ages
        kernel = np.exp(-q_k * np.log1p(scaled_age))
        weighted_kernel = weights * kernel
        s0 = np.sum(weighted_kernel, axis=1)
        s1 = weighted_kernel @ source
        prediction = level + clean_amplitude * clean + noise_floor * s0 + fit_noise * s1
        if np.any(prediction <= 0.0):
            residual_value = np.full(target.size, 1e6, dtype=np.float64)
            jacobian_value = np.zeros((target.size, vector.size), dtype=np.float64)
        else:
            dkernel_dlog_c = -q_k * kernel * scaled_age / (1.0 + scaled_age)
            weighted_dkernel = weights * dkernel_dlog_c
            ds0_dlog_c = np.sum(weighted_dkernel, axis=1)
            ds1_dlog_c = weighted_dkernel @ source
            dclean_dq_f = -target_log_time * clean
            dsource_dq_f = -source_log_time * source
            ds1_dq_f = weighted_kernel @ dsource_dq_f
            prediction_jacobian = np.column_stack(
                [
                    np.ones(target.size),
                    clean,
                    s0,
                    s1,
                    noise_floor * ds0_dlog_c + fit_noise * ds1_dlog_c,
                    clean_amplitude * dclean_dq_f + fit_noise * ds1_dq_f,
                ]
            )
            residual_value = np.log(prediction) - np.log(target)
            jacobian_value = prediction_jacobian / prediction[:, None]
        cache_vector = vector.copy()
        cache_residual = residual_value
        cache_jacobian = jacobian_value
        return residual_value, jacobian_value

    def residual(vector: np.ndarray) -> np.ndarray:
        return evaluate(vector)[0]

    def jacobian(vector: np.ndarray) -> np.ndarray:
        return evaluate(vector)[1]

    solutions = []
    for start in starts:
        start = np.clip(start, lower + 1e-10, upper - 1e-10)
        fit = least_squares(
            residual,
            start,
            jac=jacobian,
            bounds=(lower, upper),
            loss="huber",
            f_scale=1e-3,
            max_nfev=600,
            xtol=1e-8,
            ftol=1e-8,
            gtol=1e-8,
        )
        rss = float(np.sum(np.square(residual(fit.x))))
        solutions.append((rss, fit))
    rss, fit = min(solutions, key=lambda item: item[0])
    fitted = _parameters(fit.x, q_k)
    return {
        "parameters": fitted,
        "optimizer": {
            "success": bool(fit.success),
            "status": int(fit.status),
            "message": str(fit.message),
            "function_evaluations": int(fit.nfev),
            "multistarts": len(starts),
            "log_residual_rss": rss,
        },
    }


def _load_reference_im(path: Path) -> Json:
    fits = base._load_existing_fits(path)
    return fits["IM"]["parameters"]


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty CSV: {path}")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _read_profile_csv(path: Path) -> list[Json]:
    rows: list[Json] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        for raw in csv.DictReader(handle):
            row: Json = {}
            for key, value in raw.items():
                if key == "optimizer_success":
                    row[key] = value == "True"
                elif key == "optimizer_function_evaluations":
                    row[key] = int(value)
                else:
                    row[key] = float(value)
            rows.append(row)
    observed_grid = tuple(float(row["q_K"]) for row in rows)
    if observed_grid != Q_K_GRID:
        raise ValueError(f"profile grid mismatch in {path}: {observed_grid!r}")
    return rows


def _fits_from_profile_rows(rows: Sequence[Mapping[str, Any]]) -> dict[float, Json]:
    return {
        float(row["q_K"]): {
            "parameters": {
                "L_inf": float(row["L_inf"]),
                "clean_amplitude": float(row["clean_amplitude"]),
                "noise_floor_amplitude": float(row["noise_floor_amplitude"]),
                "fit_dependent_noise_amplitude": float(row["fit_dependent_noise_amplitude"]),
                "kernel_scale": float(row["kernel_scale"]),
                "q_F": float(row["q_F"]),
                "q_K": float(row["q_K"]),
            },
            "optimizer": {"replayed_from_profile_csv": True},
        }
        for row in rows
    }


def _render_profile(
    output_dir: Path,
    rows: Sequence[Mapping[str, Any]],
    title: str = "30M IM fixed-exponent profile: 8-1-1 fit and WSD transfer",
) -> None:
    q = np.asarray([row["q_K"] for row in rows], dtype=np.float64)
    fit_rmse = np.asarray([row["eight_primary_rmse_ce"] for row in rows], dtype=np.float64)
    wsd_rmse = np.asarray([row["wsd_tail_rmse_ce"] for row in rows], dtype=np.float64)
    c_k = np.asarray([row["kernel_scale"] for row in rows], dtype=np.float64)
    effective_rate = np.asarray([row["q_K_times_kernel_scale"] for row in rows], dtype=np.float64)

    fig, axes = plt.subplots(2, 2, figsize=(11.4, 8.0))
    axis = axes[0, 0]
    axis.plot(q, fit_rmse, "o-", label="8-1-1 fit")
    axis.plot(q, wsd_rmse, "s-", label="WSD zero-refit tail")
    axis.set_ylabel("Validation CE RMSE")
    axis.legend()

    axis = axes[0, 1]
    profile_rss = np.asarray([row["log_residual_rss"] for row in rows], dtype=np.float64)
    axis.plot(q, profile_rss / np.min(profile_rss) - 1.0, "o-", color="#7b2cbf")
    axis.set_ylabel("Relative log-residual RSS above best")

    axis = axes[1, 0]
    axis.plot(q, c_k, "o-", color="#d1495b")
    axis.set_ylabel(r"Kernel scale $c_{\mathcal{K}}$")
    axis.set_yscale("log")

    axis = axes[1, 1]
    axis.plot(q, effective_rate, "o-", color="#2f8f5b")
    axis.set_ylabel(r"Effective rate $q_{\mathcal{K}}c_{\mathcal{K}}$")
    axis.set_yscale("log")

    for axis in axes.flat:
        axis.set_xscale("log")
        axis.set_xlabel(r"Fixed $q_{\mathcal{K}}$")
        axis.grid(alpha=0.22)
    fig.suptitle(title)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(output_dir / "im_qk_profile.png", dpi=180)
    fig.savefig(output_dir / "im_qk_profile.pdf")
    plt.close(fig)


def _render_selected_predictions(
    output_dir: Path,
    wsd: base.EvaluationCurve,
    tail: np.ndarray,
    fits: Mapping[float, Json],
    title: str = "WSD tail: zero-refit predictions across fixed IM exponents",
) -> None:
    fig, axis = plt.subplots(figsize=(9.6, 5.8))
    axis.errorbar(
        wsd.intrinsic_time[tail],
        wsd.loss[tail],
        yerr=wsd.standard_error[tail],
        fmt="o",
        color="#111111",
        ms=4.0,
        lw=0.8,
        label="WSD observed",
    )
    colors = plt.cm.viridis(np.linspace(0.08, 0.92, len(SELECTED_PREDICTION_Q)))
    for q_k, color in zip(SELECTED_PREDICTION_Q, colors):
        prediction = _predict_curve(wsd, fits[q_k]["parameters"])
        axis.plot(
            wsd.intrinsic_time[tail],
            prediction[tail],
            lw=1.8,
            color=color,
            label=fr"$q_{{\mathcal{{K}}}}={q_k:g}$",
        )
    axis.set_xlabel("Intrinsic time")
    axis.set_ylabel("Validation CE")
    axis.set_title(title)
    axis.grid(alpha=0.22)
    axis.legend(ncol=2)
    fig.tight_layout()
    fig.savefig(output_dir / "wsd_tail_selected_qk_predictions.png", dpi=180)
    fig.savefig(output_dir / "wsd_tail_selected_qk_predictions.pdf")
    plt.close(fig)


def analyze(
    eight_path: Path,
    wsd_path: Path,
    reference_parameters: Path,
    output_dir: Path,
    reuse_existing_profile: bool = False,
) -> Json:
    output_dir.mkdir(parents=True, exist_ok=True)
    eight = base._load_curve(eight_path, "eight_one_one__fixed_batch_lr")
    wsd = base._load_curve(wsd_path, "wsd_exp_80_20__fixed_batch_lr")
    base._validate_pair(eight, wsd)
    masks = base._window_masks(eight, wsd)
    reference_im = _load_reference_im(reference_parameters)

    profile_path = output_dir / "im_qk_profile.csv"
    fits: dict[float, Json] = {}
    rows: list[Json] = []
    prediction_rows = []
    if reuse_existing_profile:
        if not profile_path.is_file():
            raise ValueError(f"existing profile CSV not found: {profile_path}")
        rows = _read_profile_csv(profile_path)
        fits = _fits_from_profile_rows(rows)
    else:
        for q_k in Q_K_GRID:
            fit = _fit_fixed_q(eight, masks["eight_primary"], q_k, reference_im)
            fits[q_k] = fit
            fit_prediction = _predict_curve(eight, fit["parameters"])
            transfer_prediction = _predict_curve(wsd, fit["parameters"])
            fit_metrics = base._metrics(
                eight.loss[masks["eight_primary"]],
                fit_prediction[masks["eight_primary"]],
                parameter_count=6,
            )
            tail_metrics = base._metrics(
                wsd.loss[masks["wsd_tail"]], transfer_prediction[masks["wsd_tail"]]
            )
            overlap_metrics = base._metrics(
                wsd.loss[masks["wsd_tail_overlap"]],
                transfer_prediction[masks["wsd_tail_overlap"]],
            )
            extrapolation_metrics = base._metrics(
                wsd.loss[masks["wsd_tail_extrapolation"]],
                transfer_prediction[masks["wsd_tail_extrapolation"]],
            )
            parameters = fit["parameters"]
            c_k = parameters["kernel_scale"]
            row = {
                "q_K": q_k,
                "kernel_scale": c_k,
                "q_K_times_kernel_scale": q_k * c_k,
                "L_inf": parameters["L_inf"],
                "clean_amplitude": parameters["clean_amplitude"],
                "noise_floor_amplitude": parameters["noise_floor_amplitude"],
                "fit_dependent_noise_amplitude": parameters["fit_dependent_noise_amplitude"],
                "q_F": parameters["q_F"],
                "log_residual_rss": fit["optimizer"]["log_residual_rss"],
                "eight_primary_rmse_ce": fit_metrics["rmse_ce"],
                "eight_primary_aicc_log_residual": fit_metrics["aicc_log_residual"],
                "wsd_tail_rmse_ce": tail_metrics["rmse_ce"],
                "wsd_tail_bias_ce": tail_metrics["bias_ce"],
                "wsd_overlap_rmse_ce": overlap_metrics["rmse_ce"],
                "wsd_extrapolation_rmse_ce": extrapolation_metrics["rmse_ce"],
                "optimizer_success": fit["optimizer"]["success"],
                "optimizer_function_evaluations": fit["optimizer"]["function_evaluations"],
            }
            rows.append(row)
            print(
                f"q_K={q_k:g} fit_rmse={fit_metrics['rmse_ce']:.8f} "
                f"wsd_tail_rmse={tail_metrics['rmse_ce']:.8f} "
                f"c_K={c_k:.8g}",
                flush=True,
            )

    for q_k in SELECTED_PREDICTION_Q:
        transfer_prediction = _predict_curve(wsd, fits[q_k]["parameters"])
        for index in np.flatnonzero(masks["wsd_tail"]):
            prediction_rows.append(
                {
                    "q_K": q_k,
                    "update": int(wsd.update[index]),
                    "intrinsic_time": float(wsd.intrinsic_time[index]),
                    "observed_validation_ce": float(wsd.loss[index]),
                    "predicted_validation_ce": float(transfer_prediction[index]),
                    "residual_ce": float(transfer_prediction[index] - wsd.loss[index]),
                }
            )

    best_fit = min(rows, key=lambda row: row["eight_primary_rmse_ce"])
    best_transfer = min(rows, key=lambda row: row["wsd_tail_rmse_ce"])
    fit_min = float(best_fit["eight_primary_rmse_ce"])
    transfer_min = float(best_transfer["wsd_tail_rmse_ce"])
    fit_near = [float(row["q_K"]) for row in rows if row["eight_primary_rmse_ce"] <= 1.01 * fit_min]
    transfer_near = [float(row["q_K"]) for row in rows if row["wsd_tail_rmse_ce"] <= 1.01 * transfer_min]
    last = rows[-1]
    penultimate = rows[-2]
    summary = {
        "schema_version": "nanogpt30m_e2e_sgd_im_qk_profile_v001",
        "analysis_id": ANALYSIS_ID,
        "status": "completed",
        "classification": "post_hoc_exploratory_A2_EXTERNAL",
        "claim_boundary": {
            "allowed": [
                "diagnose finite-window IM q_K identifiability",
                "compare fixed-q_K zero-refit WSD prediction errors",
            ],
            "prohibited": [
                "report a stationary q_K estimate",
                "claim theorem-facing IM identification",
                "use the WSD curve to refit any parameter",
            ],
        },
        "q_K_grid": list(Q_K_GRID),
        "fit_protocol": {
            "fit_schedule": "eight_one_one fixed_batch_lr",
            "transfer_schedule": "wsd_exp_80_20 fixed_batch_lr",
            "target": "fixed_validation_probe_cross_entropy",
            "clock": "intrinsic_time",
            "primary_window_rule": base.PRIMARY_WINDOW_RULE,
            "objective": "Huber robust least squares on log validation CE",
            "free_parameter_count_per_grid_point": 6,
            "parameter_refit_on_wsd": False,
        },
        "profile_summary": {
            "best_eight_primary_q_K": float(best_fit["q_K"]),
            "best_eight_primary_rmse_ce": fit_min,
            "q_K_within_one_percent_of_best_fit_rmse": fit_near,
            "best_wsd_tail_q_K": float(best_transfer["q_K"]),
            "best_wsd_tail_rmse_ce": transfer_min,
            "q_K_within_one_percent_of_best_transfer_rmse": transfer_near,
            "q_K_40_vs_20_fit_rmse_ratio": float(
                last["eight_primary_rmse_ce"] / penultimate["eight_primary_rmse_ce"]
            ),
            "q_K_40_vs_20_transfer_rmse_ratio": float(
                last["wsd_tail_rmse_ce"] / penultimate["wsd_tail_rmse_ce"]
            ),
            "median_wsd_tail_validation_standard_error": float(
                np.median(wsd.standard_error[masks["wsd_tail"]])
            ),
            "decision_label": "inconclusive_q_K_not_identified",
            "interpretation": (
                "8-1-1 fit error continues to decrease through q_K=40 while the "
                "high-q_K WSD transfer error is nearly flat and q_K*c_K stabilizes; "
                "the finite window identifies an approximately exponential effective "
                "rate more readily than a finite power-law exponent."
            ),
        },
        "source": {
            "eight_one_one": {"path": str(eight_path), "sha256": base._sha256_file(eight_path)},
            "wsd": {"path": str(wsd_path), "sha256": base._sha256_file(wsd_path)},
            "reference_parameters": {
                "path": str(reference_parameters),
                "sha256": base._sha256_file(reference_parameters),
            },
        },
        "rows": rows,
    }

    _write_csv(profile_path, rows)
    _write_csv(output_dir / "wsd_tail_selected_qk_predictions.csv", prediction_rows)
    _render_profile(output_dir, rows)
    _render_selected_predictions(output_dir, wsd, masks["wsd_tail"], fits)
    readme = f"""# {ANALYSIS_ID}

Post-hoc exploratory A2 external-validity profile. At each declared IM
`q_K`, the other six ordinary-memory surrogate parameters are fitted only to
the original 8-1-1 primary window. WSD is evaluated strictly zero-refit.

This artifact diagnoses finite-window identifiability. It is not an independent
kernel measurement and cannot establish an IM phase or a stationary `q_K`.
"""
    (output_dir / "README.md").write_text(readme, encoding="utf-8")
    artifact_names = (
        "im_qk_profile.csv",
        "wsd_tail_selected_qk_predictions.csv",
        "im_qk_profile.png",
        "im_qk_profile.pdf",
        "wsd_tail_selected_qk_predictions.png",
        "wsd_tail_selected_qk_predictions.pdf",
        "README.md",
    )
    summary["artifacts"] = {
        name: {
            "sha256": base._sha256_file(output_dir / name),
            "bytes": (output_dir / name).stat().st_size,
        }
        for name in artifact_names
    }
    (output_dir / "summary.json").write_bytes(base._canonical_bytes(summary))
    return summary


def _default_paths(root: Path) -> tuple[Path, Path, Path, Path]:
    eight, wsd, original_output = base._default_paths(root)
    reference_parameters = original_output / "fit_parameters.csv"
    output = root / "experiments" / "results" / ANALYSIS_ID
    return eight, wsd, reference_parameters, output


def main(argv: Sequence[str] | None = None) -> int:
    root = Path(__file__).resolve().parents[2]
    default_eight, default_wsd, default_parameters, default_output = _default_paths(root)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eight-result", type=Path, default=default_eight)
    parser.add_argument("--wsd-result", type=Path, default=default_wsd)
    parser.add_argument("--reference-parameters", type=Path, default=default_parameters)
    parser.add_argument("--output-dir", type=Path, default=default_output)
    parser.add_argument("--reuse-existing-profile", action="store_true")
    args = parser.parse_args(argv)
    result = analyze(
        args.eight_result.expanduser().resolve(),
        args.wsd_result.expanduser().resolve(),
        args.reference_parameters.expanduser().resolve(),
        args.output_dir.expanduser().resolve(),
        reuse_existing_profile=args.reuse_existing_profile,
    )
    print(json.dumps(result["profile_summary"], indent=2, sort_keys=True))
    print(args.output_dir.expanduser().resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
