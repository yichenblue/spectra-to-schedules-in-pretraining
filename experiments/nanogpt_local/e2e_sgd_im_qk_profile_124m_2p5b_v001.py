"""Fixed-q_K IM profile on the completed 124M/2.5B ratio-path campaign.

Fit each declared q_K only on the fixed-batch 8-1-1 validation-CE trajectory,
then evaluate the fixed-batch WSD trajectory without refitting.  The matched
fixed-LR/batch trajectories are retained as secondary factorization checks.
This is a post-hoc single-seed A2_EXTERNAL identifiability diagnostic.
"""

from __future__ import annotations

import argparse
import csv
from contextlib import contextmanager
import json
import math
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

import numpy as np

from experiments.nanogpt_local import e2e_sgd_im_qk_profile_v001 as profile
from experiments.nanogpt_local import e2e_sgd_phase_surrogate_transfer_v001 as phase
from experiments.nanogpt_local import e2e_sgd_rpath_factorization_124m_2p5b_analysis_v001 as validator
from experiments.nanogpt_local import e2e_sgd_rpath_factorization_124m_2p5b_v001 as core


Json = dict[str, Any]

ANALYSIS_ID = "nanogpt124m-e2e-sgd-im-qk-profile-v001"
LOG_C_BOUNDS = (-14.0, 9.0)


@contextmanager
def _use_124m_schedule_core() -> Iterator[None]:
    """Temporarily bind the shared surrogate quadrature to the 124M schedule."""

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


def _load_verified_arrays(directory: Path) -> tuple[Json, dict[str, np.ndarray]]:
    report = validator.validate_result_directory(directory)
    if report["status"] != "passed":
        raise validator.AnalysisError(f"source validation failed: {directory}")
    if report["phase"] == "prefix":
        expected = validator._expected_prefix_evaluations()
    else:
        expected = validator._expected_tail_evaluations(str(report["arm_id"]))
    arrays = validator._load_evaluations(directory / core.EVALUATION_FILENAME, expected)
    return report, arrays


def _combine_prefix_tail(
    prefix_directory: Path,
    tail_directory: Path,
) -> phase.EvaluationCurve:
    prefix_report, prefix = _load_verified_arrays(prefix_directory)
    tail_report, tail = _load_verified_arrays(tail_directory)
    if prefix_report["phase"] != "prefix" or tail_report["phase"] != "tail":
        raise validator.AnalysisError("expected one prefix and one tail result")
    if not (
        prefix["macro_index"][-1] == tail["macro_index"][0]
        and prefix["global_update"][-1] == tail["global_update"][0]
        and prefix["intrinsic_time"][-1] == tail["intrinsic_time"][0]
        and np.array_equal(
            prefix["context_cross_entropy"][-1], tail["context_cross_entropy"][0]
        )
    ):
        raise validator.AnalysisError("prefix/tail validation anchor differs")

    # Retain the tail copy of the shared anchor so every tail source contributes
    # all of its validated rows while the concatenated clock stays strict.
    def concatenate(name: str) -> np.ndarray:
        return np.concatenate([prefix[name][:-1], tail[name]], axis=0)

    contexts = np.asarray(concatenate("context_cross_entropy"), dtype=np.float64)
    standard_error = np.std(contexts, axis=1, ddof=1) / math.sqrt(
        core.VALIDATION_CONTEXTS
    )
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
    return curve


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
    if tuple(float(row["q_K"]) for row in rows) != profile.Q_K_GRID:
        raise ValueError(f"profile grid mismatch in {path}")
    return rows


def _fits_from_rows(rows: Sequence[Mapping[str, Any]]) -> dict[float, Json]:
    return {
        float(row["q_K"]): {
            "parameters": {
                "L_inf": float(row["L_inf"]),
                "clean_amplitude": float(row["clean_amplitude"]),
                "noise_floor_amplitude": float(row["noise_floor_amplitude"]),
                "fit_dependent_noise_amplitude": float(
                    row["fit_dependent_noise_amplitude"]
                ),
                "kernel_scale": float(row["kernel_scale"]),
                "q_F": float(row["q_F"]),
                "q_K": float(row["q_K"]),
            },
            "optimizer": {
                "success": bool(row["optimizer_success"]),
                "function_evaluations": int(row["optimizer_function_evaluations"]),
                "log_residual_rss": float(row["log_residual_rss"]),
                "replayed_from_profile_csv": True,
            },
        }
        for row in rows
    }


def _metric_row(
    q_k: float,
    fit: Json,
    eight: phase.EvaluationCurve,
    wsd: phase.EvaluationCurve,
    eight_factorized: phase.EvaluationCurve,
    wsd_factorized: phase.EvaluationCurve,
    masks: Mapping[str, np.ndarray],
) -> tuple[Json, np.ndarray, np.ndarray]:
    parameters = fit["parameters"]
    eight_prediction = profile._predict_curve(eight, parameters)
    wsd_prediction = profile._predict_curve(wsd, parameters)
    fit_metrics = phase._metrics(
        eight.loss[masks["eight_primary"]],
        eight_prediction[masks["eight_primary"]],
        parameter_count=6,
    )
    tail_metrics = phase._metrics(
        wsd.loss[masks["wsd_tail"]], wsd_prediction[masks["wsd_tail"]]
    )
    overlap_metrics = phase._metrics(
        wsd.loss[masks["wsd_tail_overlap"]],
        wsd_prediction[masks["wsd_tail_overlap"]],
    )
    extrapolation_metrics = phase._metrics(
        wsd.loss[masks["wsd_tail_extrapolation"]],
        wsd_prediction[masks["wsd_tail_extrapolation"]],
    )
    factorized_eight_metrics = phase._metrics(
        eight_factorized.loss[masks["eight_primary"]],
        eight_prediction[masks["eight_primary"]],
    )
    factorized_wsd_metrics = phase._metrics(
        wsd_factorized.loss[masks["wsd_tail"]],
        wsd_prediction[masks["wsd_tail"]],
    )
    c_k = float(parameters["kernel_scale"])
    return (
        {
            "q_K": q_k,
            "kernel_scale": c_k,
            "q_K_times_kernel_scale": q_k * c_k,
            "L_inf": parameters["L_inf"],
            "clean_amplitude": parameters["clean_amplitude"],
            "noise_floor_amplitude": parameters["noise_floor_amplitude"],
            "fit_dependent_noise_amplitude": parameters[
                "fit_dependent_noise_amplitude"
            ],
            "q_F": parameters["q_F"],
            "log_residual_rss": fit["optimizer"]["log_residual_rss"],
            "eight_primary_rmse_ce": fit_metrics["rmse_ce"],
            "eight_primary_aicc_log_residual": fit_metrics[
                "aicc_log_residual"
            ],
            "wsd_tail_rmse_ce": tail_metrics["rmse_ce"],
            "wsd_tail_bias_ce": tail_metrics["bias_ce"],
            "wsd_overlap_rmse_ce": overlap_metrics["rmse_ce"],
            "wsd_extrapolation_rmse_ce": extrapolation_metrics["rmse_ce"],
            "eight_fixed_lr_batch_zero_refit_rmse_ce": factorized_eight_metrics[
                "rmse_ce"
            ],
            "wsd_fixed_lr_batch_zero_refit_rmse_ce": factorized_wsd_metrics[
                "rmse_ce"
            ],
            "optimizer_success": fit["optimizer"]["success"],
            "optimizer_function_evaluations": fit["optimizer"][
                "function_evaluations"
            ],
        },
        eight_prediction,
        wsd_prediction,
    )


def analyze(
    prefix_directory: Path,
    tail_directories: Mapping[str, Path],
    output_directory: Path,
    reuse_existing_profile: bool = False,
) -> Json:
    output_directory.mkdir(parents=True, exist_ok=True)
    campaign_validation = validator.validate_campaign(
        prefix_directory,
        [tail_directories[arm_id] for arm_id in core.ARM_IDS],
    )
    if campaign_validation["status"] != "passed":
        raise validator.AnalysisError("124M campaign validation failed")
    (output_directory / "campaign_validation.json").write_bytes(
        phase._canonical_bytes(campaign_validation)
    )

    curves = {
        arm_id: _combine_prefix_tail(prefix_directory, directory)
        for arm_id, directory in tail_directories.items()
    }
    eight = curves["eight_one_one__fixed_batch_lr"]
    wsd = curves["wsd_exp_80_20__fixed_batch_lr"]
    eight_factorized = curves["eight_one_one__fixed_lr_batch"]
    wsd_factorized = curves["wsd_exp_80_20__fixed_lr_batch"]
    if not (
        np.array_equal(eight.intrinsic_time, eight_factorized.intrinsic_time)
        and np.array_equal(wsd.intrinsic_time, wsd_factorized.intrinsic_time)
    ):
        raise validator.AnalysisError("same-shape factorization grids differ")

    profile_path = output_directory / "im_qk_profile.csv"
    rows: list[Json] = []
    fits: dict[float, Json]
    prediction_rows: list[Json] = []
    with _use_124m_schedule_core():
        phase._validate_pair(eight, wsd)
        masks = phase._window_masks(eight, wsd)
        if reuse_existing_profile:
            if not profile_path.is_file():
                raise ValueError(f"existing profile CSV not found: {profile_path}")
            existing_rows = _read_profile_csv(profile_path)
            fits = _fits_from_rows(existing_rows)
            rows = []
            for q_k in profile.Q_K_GRID:
                row, _, _ = _metric_row(
                    q_k,
                    fits[q_k],
                    eight,
                    wsd,
                    eight_factorized,
                    wsd_factorized,
                    masks,
                )
                rows.append(row)
        else:
            initial_level, initial_clean = phase._initial_clean_coefficients(
                eight, masks["eight_primary"], 0.10
            )
            small_scale_warm = {
                "L_inf": initial_level,
                "clean_amplitude": initial_clean,
                "noise_floor_amplitude": 0.1,
                "fit_dependent_noise_amplitude": 5.0,
                "kernel_scale": 1e-5,
                "q_F": 0.10,
                "q_K": 6.0,
            }
            reference_fit = profile._fit_fixed_q(
                eight,
                masks["eight_primary"],
                6.0,
                warm_parameters=small_scale_warm,
                log_c_bounds=LOG_C_BOUNDS,
            )
            reference_parameters = reference_fit["parameters"]
            fits = {}
            for q_k in profile.Q_K_GRID:
                fit = (
                    reference_fit
                    if q_k == 6.0
                    else profile._fit_fixed_q(
                        eight,
                        masks["eight_primary"],
                        q_k,
                        warm_parameters=reference_parameters,
                        log_c_bounds=LOG_C_BOUNDS,
                    )
                )
                fits[q_k] = fit
                row, _, _ = _metric_row(
                    q_k,
                    fit,
                    eight,
                    wsd,
                    eight_factorized,
                    wsd_factorized,
                    masks,
                )
                rows.append(row)
                print(
                    f"q_K={q_k:g} fit_rmse={row['eight_primary_rmse_ce']:.8f} "
                    f"wsd_tail_rmse={row['wsd_tail_rmse_ce']:.8f} "
                    f"c_K={row['kernel_scale']:.8g}",
                    flush=True,
                )

        for q_k in profile.SELECTED_PREDICTION_Q:
            prediction = profile._predict_curve(wsd, fits[q_k]["parameters"])
            for index in np.flatnonzero(masks["wsd_tail"]):
                prediction_rows.append(
                    {
                        "q_K": q_k,
                        "update": int(wsd.update[index]),
                        "macro_index": int(wsd.macro_index[index]),
                        "intrinsic_time": float(wsd.intrinsic_time[index]),
                        "observed_validation_ce": float(wsd.loss[index]),
                        "predicted_validation_ce": float(prediction[index]),
                        "residual_ce": float(prediction[index] - wsd.loss[index]),
                    }
                )

        profile._render_profile(
            output_directory,
            rows,
            title="124M/2.5B IM fixed-exponent profile: 8-1-1 fit and WSD transfer",
        )
        profile._render_selected_predictions(
            output_directory,
            wsd,
            masks["wsd_tail"],
            fits,
            title="124M/2.5B WSD tail: zero-refit fixed-IM-exponent predictions",
        )

    best_fit = min(rows, key=lambda row: row["eight_primary_rmse_ce"])
    best_transfer = min(rows, key=lambda row: row["wsd_tail_rmse_ce"])
    fit_min = float(best_fit["eight_primary_rmse_ce"])
    transfer_min = float(best_transfer["wsd_tail_rmse_ce"])
    by_q = {float(row["q_K"]): row for row in rows}
    factorization = {
        "eight_one_one_validation_curve_rmse_ce": float(
            np.sqrt(np.mean(np.square(eight.loss - eight_factorized.loss)))
        ),
        "wsd_validation_curve_rmse_ce": float(
            np.sqrt(np.mean(np.square(wsd.loss - wsd_factorized.loss)))
        ),
    }
    summary = {
        "schema_version": "nanogpt124m_e2e_sgd_im_qk_profile_v001",
        "analysis_id": ANALYSIS_ID,
        "status": "completed",
        "classification": "post_hoc_exploratory_single_seed_A2_EXTERNAL",
        "new_gpu_hours_consumed_by_analysis": 0.0,
        "claim_boundary": {
            "allowed": [
                "diagnose finite-window IM q_K identifiability",
                "compare fixed-q_K zero-refit schedule-transfer errors",
                "compare transfer to matched ratio-path factorizations",
            ],
            "prohibited": [
                "report a stationary q_K estimate",
                "claim theorem-facing IM identification",
                "use WSD or fixed-LR/batch curves to refit parameters",
            ],
        },
        "fit_protocol": {
            "q_K_grid": list(profile.Q_K_GRID),
            "fit_schedule": "eight_one_one__fixed_batch_lr",
            "transfer_schedule": "wsd_exp_80_20__fixed_batch_lr",
            "target": "fixed_1024_context_validation_probe_cross_entropy",
            "clock": "intrinsic_time",
            "primary_window_rule": phase.PRIMARY_WINDOW_RULE,
            "primary_start_intrinsic_time": float(
                eight.intrinsic_time[masks["eight_primary"]][0]
            ),
            "primary_end_intrinsic_time": float(eight.intrinsic_time[-1]),
            "primary_evaluation_rows": int(np.sum(masks["eight_primary"])),
            "objective": "Huber robust least squares on log validation CE",
            "free_parameter_count_per_grid_point": 6,
            "kernel_log_scale_bounds": list(LOG_C_BOUNDS),
            "convolution": "exact FFT replay of DeltaT=0.02 midpoint surrogate",
            "parameter_refit_on_transfer_curves": False,
        },
        "profile_summary": {
            "best_eight_primary_q_K": float(best_fit["q_K"]),
            "best_eight_primary_rmse_ce": fit_min,
            "q_K_within_one_percent_of_best_fit_rmse": [
                float(row["q_K"])
                for row in rows
                if row["eight_primary_rmse_ce"] <= 1.01 * fit_min
            ],
            "best_wsd_tail_q_K": float(best_transfer["q_K"]),
            "best_wsd_tail_rmse_ce": transfer_min,
            "q_K_within_one_percent_of_best_transfer_rmse": [
                float(row["q_K"])
                for row in rows
                if row["wsd_tail_rmse_ce"] <= 1.01 * transfer_min
            ],
            "q_K_40_vs_20_fit_rmse_ratio": float(
                by_q[40.0]["eight_primary_rmse_ce"]
                / by_q[20.0]["eight_primary_rmse_ce"]
            ),
            "q_K_40_vs_20_transfer_rmse_ratio": float(
                by_q[40.0]["wsd_tail_rmse_ce"]
                / by_q[20.0]["wsd_tail_rmse_ce"]
            ),
            "median_wsd_tail_validation_standard_error": float(
                np.median(wsd.standard_error[masks["wsd_tail"]])
            ),
            "decision_label": "inconclusive",
            "decision_scope": (
                "single-seed 124M end-to-end plain-SGD CE, fixed-batch "
                "8-1-1 fit over T=[983.04,10273.70], zero-refit WSD tail"
            ),
            "interpretation": (
                "The 8-1-1 in-sample optimum is the lowest IM grid point "
                "q_K=1.02, while WSD transfer selects the adjacent q_K=1.10. "
                "The strong c_K/amplitude compensation near q_K=1 and the "
                "absence of an interior optimum prevent a stationary finite "
                "q_K estimate. The large-q branch reaches a worse plateau."
            ),
        },
        "factorization_diagnostics": factorization,
        "campaign_validation": campaign_validation,
        "source": {
            "prefix_directory": str(prefix_directory),
            "tail_directories": {
                arm_id: str(path) for arm_id, path in tail_directories.items()
            },
        },
        "rows": rows,
    }

    profile._write_csv(profile_path, rows)
    profile._write_csv(
        output_directory / "wsd_tail_selected_qk_predictions.csv", prediction_rows
    )
    readme = f"""# {ANALYSIS_ID}

Post-hoc, single-seed A2 external-validity profile on the completed 124M,
2.5B-token H200 campaign. Each fixed `q_K` refits six parameters only on the
fixed-batch 8-1-1 curve. WSD and both fixed-LR/batch curves are zero-refit.

The exact macro-midpoint surrogate convolution is evaluated by FFT. This
artifact diagnoses finite-window identifiability; it is not an independent
kernel measurement and cannot establish an IM phase or stationary `q_K`.
"""
    (output_directory / "README.md").write_text(readme, encoding="utf-8")
    artifact_names = (
        "campaign_validation.json",
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
            "sha256": phase._sha256_file(output_directory / name),
            "bytes": (output_directory / name).stat().st_size,
        }
        for name in artifact_names
    }
    (output_directory / "summary.json").write_bytes(phase._canonical_bytes(summary))
    return summary


def _default_paths(root: Path) -> tuple[Path, dict[str, Path], Path]:
    campaign = (
        root
        / "experiments"
        / "generated"
        / "nanogpt124m-e2e-sgd-rpath-factorization-2p5b-v001"
    )
    prefix = campaign / "prefix-launch-a001" / "collected_output"
    tail_root = campaign / "tails-launch-a001" / "collected_output"
    tails = {
        "eight_one_one__fixed_batch_lr": tail_root / "tail_output_811_fblr",
        "eight_one_one__fixed_lr_batch": tail_root / "tail_output_811_flrbs",
        "wsd_exp_80_20__fixed_batch_lr": tail_root / "tail_output_wsd_fblr",
        "wsd_exp_80_20__fixed_lr_batch": tail_root / "tail_output_wsd_flrbs",
    }
    output = root / "experiments" / "results" / ANALYSIS_ID
    return prefix, tails, output


def main(argv: Sequence[str] | None = None) -> int:
    root = Path(__file__).resolve().parents[2]
    default_prefix, default_tails, default_output = _default_paths(root)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prefix-directory", type=Path, default=default_prefix)
    parser.add_argument("--output-directory", type=Path, default=default_output)
    parser.add_argument("--reuse-existing-profile", action="store_true")
    args = parser.parse_args(argv)
    result = analyze(
        args.prefix_directory.expanduser().resolve(),
        {arm_id: path.resolve() for arm_id, path in default_tails.items()},
        args.output_directory.expanduser().resolve(),
        reuse_existing_profile=args.reuse_existing_profile,
    )
    print(json.dumps(result["profile_summary"], indent=2, sort_keys=True))
    print(args.output_directory.expanduser().resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
