"""Zero-refit WSD validation for the frozen 124M SGD v002 surrogate.

The seven parameters are loaded verbatim from the completed fixed-batch
8-1-1 fit.  This module validates the newly collected WSD artifacts and then
evaluates both WSD factorizations without invoking an optimizer.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from experiments.nanogpt_local import e2e_sgd_im_qk_profile_v001 as fixed_q
from experiments.nanogpt_local import e2e_sgd_phase_surrogate_transfer_124m_2p5b_v001 as transfer
from experiments.nanogpt_local import e2e_sgd_phase_surrogate_transfer_v001 as phase
from experiments.nanogpt_local import e2e_sgd_rpath_factorization_124m_2p5b_v002 as core
from experiments.nanogpt_local import e2e_sgd_unified_phase_surrogate_124m_2p5b_v002 as fit_analysis


Json = dict[str, Any]
ANALYSIS_ID = fit_analysis.ANALYSIS_ID + "-wsd-zero-refit"
WSD_ARMS = (
    "wsd_exp_80_20__fixed_batch_lr",
    "wsd_exp_80_20__fixed_lr_batch",
)
EIGHT_ARMS = (
    "eight_one_one__fixed_batch_lr",
    "eight_one_one__fixed_lr_batch",
)
MATCHED_SCHEDULE_ARMS = dict(zip(WSD_ARMS, EIGHT_ARMS, strict=True))
LINKAGE_FIELDS = (
    "campaign_contract_sha256",
    "numerics_id",
    "data_identity_sha256",
    "schedule_manifest_sha256",
    "resource_calibration_sha256",
    "prefix_checkpoint_sha256",
    "prefix_model_state_sha256",
    "prefix_validation_anchor_sha256",
)


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty CSV: {path}")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _weighted_metrics(
    target: np.ndarray, prediction: np.ndarray, intrinsic_time: np.ndarray
) -> Json:
    weights = transfer._trapezoid_weights(np.log(intrinsic_time))
    return transfer._weighted_metrics(target, prediction, weights)


def _window_metrics(
    curve: phase.EvaluationCurve,
    prediction: np.ndarray,
    masks: Mapping[str, np.ndarray],
) -> Json:
    result: Json = {}
    for name, mask in masks.items():
        if int(np.sum(mask)) == 0:
            result[name] = {"count": 0, "status": "empty"}
            continue
        result[name] = {
            "recorded_tick_weighted": phase._metrics(
                curve.loss[mask], prediction[mask]
            ),
            "log_intrinsic_time_balanced": _weighted_metrics(
                curve.loss[mask],
                prediction[mask],
                curve.intrinsic_time[mask],
            ),
            "median_fixed_probe_context_standard_error_ce": float(
                np.median(curve.standard_error[mask])
            ),
        }
    return result


def _difference_metrics(difference: np.ndarray) -> Json:
    return {
        "count": int(difference.size),
        "rmse_ce": float(np.sqrt(np.mean(np.square(difference)))),
        "mae_ce": float(np.mean(np.abs(difference))),
        "max_abs_ce": float(np.max(np.abs(difference))),
        "bias_ce": float(np.mean(difference)),
    }


def _contrast_metrics(
    observed_contrast: np.ndarray, predicted_contrast: np.ndarray
) -> Json:
    residual = predicted_contrast - observed_contrast
    observed_rms = float(np.sqrt(np.mean(np.square(observed_contrast))))
    correlation = float(
        np.corrcoef(observed_contrast, predicted_contrast)[0, 1]
    )
    return {
        "count": int(observed_contrast.size),
        "observed_contrast_rms_ce": observed_rms,
        "predicted_contrast_rms_ce": float(
            np.sqrt(np.mean(np.square(predicted_contrast)))
        ),
        "rmse_ce": float(np.sqrt(np.mean(np.square(residual)))),
        "relative_rmse_to_observed_contrast_rms": float(
            np.sqrt(np.mean(np.square(residual))) / observed_rms
        ),
        "mae_ce": float(np.mean(np.abs(residual))),
        "max_abs_ce": float(np.max(np.abs(residual))),
        "bias_ce": float(np.mean(residual)),
        "pearson_correlation": correlation,
    }


def _render(
    output_directory: Path,
    curves: Mapping[str, phase.EvaluationCurve],
    predictions: Mapping[str, np.ndarray],
    masks: Mapping[str, np.ndarray],
    fit_end: float,
) -> None:
    colors = {
        WSD_ARMS[0]: "#2878B5",
        WSD_ARMS[1]: "#E56B2F",
    }
    titles = {
        WSD_ARMS[0]: "Fixed batch / LR schedule",
        WSD_ARMS[1]: "Fixed LR / batch schedule",
    }
    figure, axes = plt.subplots(
        2,
        2,
        figsize=(14.2, 8.0),
        sharex="col",
        gridspec_kw={"height_ratios": [2.5, 1.0]},
        constrained_layout=True,
    )
    for column, arm_id in enumerate(WSD_ARMS):
        curve = curves[arm_id]
        prediction = predictions[arm_id]
        tail = masks["tail_full"]
        top = axes[0, column]
        residual = axes[1, column]
        top.plot(
            curve.intrinsic_time[tail],
            curve.loss[tail],
            color=colors[arm_id],
            linewidth=1.25,
            label="Observed WSD validation CE",
        )
        top.plot(
            curve.intrinsic_time[tail],
            prediction[tail],
            color="#111111",
            linewidth=2.0,
            label="Frozen 8-1-1 fit: zero refit",
        )
        top.axvline(fit_end, color="0.55", linewidth=1.0, alpha=0.8)
        top.axvspan(
            fit_end,
            float(curve.intrinsic_time[tail][-1]),
            color="0.92",
            alpha=0.45,
            linewidth=0.0,
            label="Beyond 8-1-1 fit T range",
        )
        top.set_title(titles[arm_id])
        top.set_ylabel("Validation CE")
        top.grid(alpha=0.20, linewidth=0.7)
        top.legend(frameon=False)

        residual.plot(
            curve.intrinsic_time[tail],
            prediction[tail] - curve.loss[tail],
            color=colors[arm_id],
            linewidth=1.1,
        )
        residual.axhline(0.0, color="#111111", linewidth=0.9)
        residual.axvline(fit_end, color="0.55", linewidth=1.0, alpha=0.8)
        residual.axvspan(
            fit_end,
            float(curve.intrinsic_time[tail][-1]),
            color="0.92",
            alpha=0.45,
            linewidth=0.0,
        )
        residual.set_xlabel(r"Intrinsic time, $T=\sum_s\eta_s$")
        residual.set_ylabel("Predicted − observed")
        residual.grid(alpha=0.20, linewidth=0.7)
    figure.suptitle("124M SGD · WSD zero-refit validation")
    figure.savefig(
        output_directory / "wsd_zero_refit_validation.png",
        dpi=220,
        bbox_inches="tight",
    )
    figure.savefig(
        output_directory / "wsd_zero_refit_validation.pdf",
        bbox_inches="tight",
    )
    plt.close(figure)


def analyze(
    prefix_directory: Path,
    eight_fixed_batch_directory: Path,
    eight_fixed_lr_directory: Path,
    wsd_fixed_batch_directory: Path,
    wsd_fixed_lr_directory: Path,
    frozen_summary_path: Path,
    output_directory: Path,
) -> Json:
    frozen = json.loads(frozen_summary_path.read_text(encoding="utf-8"))
    if frozen.get("analysis_id") != fit_analysis.ANALYSIS_ID:
        raise ValueError("frozen summary is not the 124M SGD v002 fit")
    if frozen.get("status") != "completed":
        raise ValueError("frozen fit is incomplete")
    parameters = {
        name: float(value)
        for name, value in frozen["fit"]["parameters"].items()
    }
    parameter_sha256 = hashlib.sha256(
        fit_analysis._canonical_bytes(parameters)
    ).hexdigest()

    campaign_validation = fit_analysis.validator.validate_campaign(
        prefix_directory,
        (
            eight_fixed_batch_directory,
            eight_fixed_lr_directory,
            wsd_fixed_batch_directory,
            wsd_fixed_lr_directory,
        ),
    )
    if campaign_validation["status"] != "passed":
        raise ValueError("complete v002 campaign validation failed")

    eight_fixed_batch, eight_fixed_batch_validation = (
        fit_analysis._combine_prefix_tail(
            prefix_directory, eight_fixed_batch_directory
        )
    )
    eight_fixed_lr, eight_fixed_lr_validation = (
        fit_analysis._combine_prefix_tail(
            prefix_directory, eight_fixed_lr_directory
        )
    )
    fixed_batch, fixed_batch_validation = fit_analysis._combine_prefix_tail(
        prefix_directory, wsd_fixed_batch_directory
    )
    fixed_lr, fixed_lr_validation = fit_analysis._combine_prefix_tail(
        prefix_directory, wsd_fixed_lr_directory
    )
    curves = {
        fixed_batch.arm_id: fixed_batch,
        fixed_lr.arm_id: fixed_lr,
    }
    eight_curves = {
        eight_fixed_batch.arm_id: eight_fixed_batch,
        eight_fixed_lr.arm_id: eight_fixed_lr,
    }
    if tuple(curves) != WSD_ARMS:
        raise ValueError(f"unexpected WSD arm identities: {tuple(curves)}")
    if not (
        np.array_equal(fixed_batch.macro_index, fixed_lr.macro_index)
        and np.array_equal(fixed_batch.intrinsic_time, fixed_lr.intrinsic_time)
    ):
        raise ValueError("WSD factorization validation grids differ")
    if tuple(eight_curves) != EIGHT_ARMS:
        raise ValueError(f"unexpected 8-1-1 arm identities: {tuple(eight_curves)}")

    frozen_lineage = frozen["source_validation"]["fit_arm"]["prefix"]
    for validation in (
        eight_fixed_batch_validation,
        eight_fixed_lr_validation,
        fixed_batch_validation,
        fixed_lr_validation,
    ):
        for report in validation.values():
            for field in LINKAGE_FIELDS:
                if report[field] != frozen_lineage[field]:
                    raise ValueError(f"frozen-fit/WSD lineage mismatch: {field}")

    fit_end = float(frozen["fit_protocol"]["fit_end_intrinsic_time"])
    masks = {
        "tail_full": fixed_batch.intrinsic_time
        >= core.PREFIX_INTRINSIC_TIME - 1e-12,
        "tail_overlap_with_8_1_1_range": (
            fixed_batch.intrinsic_time >= core.PREFIX_INTRINSIC_TIME - 1e-12
        )
        & (fixed_batch.intrinsic_time <= fit_end + 1e-12),
        "tail_extrapolation_beyond_8_1_1_range": fixed_batch.intrinsic_time
        > fit_end + 1e-12,
    }
    with fit_analysis._use_v002_schedule_core():
        predictions = {
            arm_id: fixed_q._predict_curve(
                curve, parameters, force_fft=True
            )
            for arm_id, curve in curves.items()
        }
        eight_predictions = {
            arm_id: fixed_q._predict_curve(
                curve, parameters, force_fft=True
            )
            for arm_id, curve in eight_curves.items()
        }

    metrics = {
        arm_id: _window_metrics(curve, predictions[arm_id], masks)
        for arm_id, curve in curves.items()
    }
    factorization = {}
    for name, mask in masks.items():
        observed_difference = fixed_lr.loss[mask] - fixed_batch.loss[mask]
        predicted_difference = (
            predictions[fixed_lr.arm_id][mask]
            - predictions[fixed_batch.arm_id][mask]
        )
        factorization[name] = {
            "observed_fixed_lr_minus_fixed_batch": _difference_metrics(
                observed_difference
            ),
            "predicted_fixed_lr_minus_fixed_batch": _difference_metrics(
                predicted_difference
            ),
        }

    schedule_contrast = {}
    for wsd_arm, eight_arm in MATCHED_SCHEDULE_ARMS.items():
        wsd_curve = curves[wsd_arm]
        eight_curve = eight_curves[eight_arm]
        overlap = masks["tail_overlap_with_8_1_1_range"]
        eight_tail = (
            eight_curve.intrinsic_time
            >= core.PREFIX_INTRINSIC_TIME - 1e-12
        )
        if not (
            np.array_equal(
                wsd_curve.macro_index[overlap],
                eight_curve.macro_index[eight_tail],
            )
            and np.array_equal(
                wsd_curve.intrinsic_time[overlap],
                eight_curve.intrinsic_time[eight_tail],
            )
        ):
            raise ValueError(
                f"WSD/8-1-1 overlap grid differs for {wsd_arm}"
            )
        observed_contrast = (
            wsd_curve.loss[overlap] - eight_curve.loss[eight_tail]
        )
        predicted_contrast = (
            predictions[wsd_arm][overlap]
            - eight_predictions[eight_arm][eight_tail]
        )
        schedule_contrast[wsd_arm] = {
            "reference_arm": eight_arm,
            "definition": "WSD validation CE minus matched 8-1-1 validation CE",
            "metrics": _contrast_metrics(
                observed_contrast, predicted_contrast
            ),
        }

    rows = []
    for arm_id, curve in curves.items():
        prediction = predictions[arm_id]
        for index in range(curve.loss.size):
            rows.append(
                {
                    "arm_id": arm_id,
                    "global_update": int(curve.update[index]),
                    "macro_index": int(curve.macro_index[index]),
                    "intrinsic_time": float(curve.intrinsic_time[index]),
                    "observed_validation_ce": float(curve.loss[index]),
                    "predicted_validation_ce": float(prediction[index]),
                    "residual_ce": float(prediction[index] - curve.loss[index]),
                    "inside_wsd_tail": bool(masks["tail_full"][index]),
                    "inside_8_1_1_T_overlap": bool(
                        masks["tail_overlap_with_8_1_1_range"][index]
                    ),
                    "beyond_8_1_1_T_range": bool(
                        masks["tail_extrapolation_beyond_8_1_1_range"][index]
                    ),
                }
            )
    output_directory.mkdir(parents=True, exist_ok=True)
    _write_csv(output_directory / "wsd_zero_refit_predictions.csv", rows)
    _render(output_directory, curves, predictions, masks, fit_end)

    summary = {
        "schema_version": "nanogpt124m_e2e_sgd_unified_phase_surrogate_wsd_zero_refit_v002",
        "analysis_id": ANALYSIS_ID,
        "status": "completed",
        "classification": "post_hoc_exploratory_single_seed_A2_EXTERNAL_zero_refit_WSD_validation",
        "new_gpu_hours_consumed_by_analysis": 0.0,
        "frozen_fit": {
            "analysis_id": frozen["analysis_id"],
            "summary_path": str(frozen_summary_path.resolve()),
            "parameters": parameters,
            "parameters_sha256": parameter_sha256,
            "phase_classification": frozen["fit"]["phase_classification"],
            "parameter_refit_on_WSD": False,
            "optimizer_calls": 0,
        },
        "source_validation": {
            "complete_campaign": campaign_validation,
            eight_fixed_batch.arm_id: eight_fixed_batch_validation,
            eight_fixed_lr.arm_id: eight_fixed_lr_validation,
            fixed_batch.arm_id: fixed_batch_validation,
            fixed_lr.arm_id: fixed_lr_validation,
        },
        "windows": {
            "prefix_fork_intrinsic_time": core.PREFIX_INTRINSIC_TIME,
            "frozen_8_1_1_fit_end_intrinsic_time": fit_end,
            "WSD_terminal_intrinsic_time": float(fixed_batch.intrinsic_time[-1]),
            "tail_rows_including_anchor": int(np.sum(masks["tail_full"])),
            "tail_overlap_rows": int(
                np.sum(masks["tail_overlap_with_8_1_1_range"])
            ),
            "tail_extrapolation_rows": int(
                np.sum(masks["tail_extrapolation_beyond_8_1_1_range"])
            ),
        },
        "metrics": metrics,
        "factorization_comparison": factorization,
        "WSD_minus_8_1_1_schedule_contrast": schedule_contrast,
        "claim_boundary": {
            "allowed": [
                "assess zero-refit transfer of the frozen empirical surrogate to WSD",
                "compare the two matched WSD factorizations at common intrinsic time",
                "test the WSD-minus-8-1-1 schedule contrast without parameter refitting",
                "separate interpolation-range and modest extrapolation-range errors",
            ],
            "prohibited": [
                "reinterpret the frozen q_K as an independent memory-kernel estimate",
                "claim theorem-facing LM or IM identification",
                "treat a single shared-seed WSD transfer as independent replication",
            ],
        },
    }
    (output_directory / "wsd_zero_refit_summary.json").write_bytes(
        fit_analysis._canonical_bytes(summary)
    )
    return summary


def main() -> int:
    root = Path(__file__).resolve().parents[2]
    campaign = root / "experiments" / "generated" / core.CAMPAIGN_ID
    output = root / "experiments" / "results" / fit_analysis.ANALYSIS_ID
    summary = analyze(
        campaign / "prefix-launch-a001" / "collected_output",
        campaign
        / "tails-launch-a001"
        / "collected_output"
        / "tail_output_811_fblr",
        campaign
        / "tails-launch-a001"
        / "collected_output"
        / "tail_output_811_flrbs",
        campaign
        / "tails-launch-a001"
        / "collected_output"
        / "tail_output_wsd_fblr",
        campaign
        / "tails-launch-a001"
        / "collected_output"
        / "tail_output_wsd_flrbs",
        output / "summary.json",
        output,
    )
    print(
        json.dumps(
            {
                "frozen_parameters": summary["frozen_fit"]["parameters"],
                "phase": summary["frozen_fit"]["phase_classification"],
                "windows": summary["windows"],
                "metrics": summary["metrics"],
                "factorization_comparison": summary[
                    "factorization_comparison"
                ],
                "schedule_contrast": summary[
                    "WSD_minus_8_1_1_schedule_contrast"
                ],
                "output": str(output),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
