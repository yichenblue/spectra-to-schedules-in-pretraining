"""Prospective from-initialization 300M Muon LR calibration, with exact resume."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch

from . import e2e_300m_b0_v001 as b0
from . import e2e_300m_6p5b_plan_v001 as scale
from . import e2e_300m_muon_calibration_v001 as prior
from .muon_300m_v001 import HybridMuon, optimizer_state_fp32_and_finite


RUN_ID = "nanogpt300m-e2e-muon-calibration-v002"
INPUT_SCHEMA = "nanogpt300m_e2e_muon_calibration_inputs_v002"
RESULT_SCHEMA = "nanogpt300m_e2e_muon_calibration_result_v002"
CHECKPOINT_SCHEMA = "nanogpt300m_e2e_muon_calibration_checkpoint_v002"
WARMUP_UPDATES = 256
STAGE_UPDATES = 8_192
LR_CANDIDATES = (7.5e-5, 1.0e-4)
CONTROL_CLIP = 33.504
STARTUP_CLIP_CONTROL = 10.0
LATE_TIE_BAND_CE = 0.01
CONFIG_SCHEMA = "nanogpt300m_e2e_muon_calibration_config_v002"


class CalibrationError(RuntimeError):
    """The prospective calibration's pinned-input or numerical gate failed."""


def validate_config(root: Path, plan: dict[str, Any]) -> dict[str, Any]:
    path = root / "experiments/configs/nanogpt300m_e2e_muon_calibration_v002.json"
    config = json.loads(path.read_text(encoding="utf-8"))
    expected = {
        "schema_version": CONFIG_SCHEMA,
        "run_id": RUN_ID,
        "status": "prepared_launch_false",
        "primary_prediction_id": "A2_EXTERNAL",
        "parent_run_id": prior.RUN_ID,
        "campaign_id": plan["campaign_id"],
        "campaign_plan_path": "experiments/configs/nanogpt300m_e2e_6p5b_v001.json",
        "source_data_sha256": _source_identity(plan),
        "source_tape_offsets_sha256": plan["data"]["training_tape_offsets_sha256"],
        "initialization_seed": b0.INITIALIZATION_SEED,
        "model_parameters": plan["model"]["expected_parameter_count"],
        "context_tokens": 256,
        "global_batch_sequences": prior.GLOBAL_BATCH,
        "micro_batch_sequences": prior.MICRO_BATCH,
        "optimizer": "hybrid_muon_with_aux_adamw",
        "nominal_learning_rates": list(LR_CANDIDATES),
        "lr_warmup_updates": WARMUP_UPDATES,
        "fixed_lr_control_gradient_clip": CONTROL_CLIP,
        "conditional_startup_clip_control": STARTUP_CLIP_CONTROL,
        "initial_updates_per_arm": STAGE_UPDATES,
        "conditional_extension_updates_per_arm": STAGE_UPDATES,
        "validation_probe_contexts": 1_024,
        "validation_regular_stride_updates": 512,
        "additional_initial_validation_tick": 256,
        "late_trend_first_four_offsets_from_stage_end": [-4096, -3584, -3072, -2560],
        "late_trend_last_four_offsets_from_stage_end": [-1536, -1024, -512, 0],
        "post_warmup_max_clipping_fraction": .10,
        "lr_tie_band_validation_ce": LATE_TIE_BAND_CE,
        "save_complete_model_optimizer_rng_checkpoint": True,
        "hardware": "one_H200_per_independent_arm",
        "predicted_initial_training_gpu_hours_total": 1.7,
        "predicted_initial_plus_extension_training_gpu_hours_total": 3.39,
        "hard_job_gpu_hours_per_arm": 1.5,
        "cluster_submission_authorized": False,
        "formal_prefix_submission_authorized": False,
    }
    if config != expected:
        raise CalibrationError("v002 prospective config changed from the frozen contract")
    return config


def eval_ticks(start_update: int) -> tuple[int, ...]:
    if start_update not in (0, STAGE_UPDATES):
        raise CalibrationError("calibration stage must start at 0 or 8192")
    regular = set(range(start_update, start_update + STAGE_UPDATES + 1, 512))
    if start_update == 0:
        regular.add(WARMUP_UPDATES)
    return tuple(sorted(regular))


def _source_identity(plan: dict[str, Any]) -> dict[str, str]:
    return {key: plan["data"][key] for key in (
        "train_sha256", "training_tape_file_sha256", "validation_sha256", "validation_probe_sha256",
    )}


def _inputs(
    folder: Path, manifest_path: Path, plan: dict[str, Any], start_update: int,
    resume_checkpoint: Path | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        manifest.get("schema_version") != INPUT_SCHEMA
        or manifest.get("run_id") != RUN_ID
        or manifest.get("campaign_id") != plan["campaign_id"]
        or manifest.get("source_data_sha256") != _source_identity(plan)
        or manifest.get("source_update_begin") != start_update
        or manifest.get("packed_tape_updates") != STAGE_UPDATES
        or (resume_checkpoint is None) != (start_update == 0)
    ):
        raise CalibrationError("300M v002 input provenance or stage changed")
    expected_checkpoint_hash = manifest.get("resume_checkpoint_sha256")
    if start_update == 0:
        if expected_checkpoint_hash is not None:
            raise CalibrationError("initialization stage cannot load a checkpoint")
    elif (
        resume_checkpoint is None or not resume_checkpoint.is_file()
        or resume_checkpoint.is_symlink()
        or b0._sha256(resume_checkpoint) != expected_checkpoint_hash
    ):
        raise CalibrationError("extension checkpoint changed")
    names = ("pretrain_train.bin", "training_tape.npy", "pretrain_validation.bin", "validation_probe.npy")
    if set(manifest.get("files", {})) != set(names):
        raise CalibrationError("300M v002 packed input inventory changed")
    for name in names:
        path = folder / name
        row = manifest["files"][name]
        if (
            not path.is_file() or path.is_symlink()
            or path.stat().st_size != row.get("size_bytes")
            or b0._sha256(path) != row.get("sha256")
        ):
            raise CalibrationError(f"300M v002 packed input changed: {name}")
    train = np.memmap(folder / names[0], dtype="<u2", mode="r")
    tape = np.load(folder / names[1], mmap_mode="r", allow_pickle=False)
    validation = np.memmap(folder / names[2], dtype="<u2", mode="r")
    probe = np.load(folder / names[3], allow_pickle=False)
    if (
        tape.shape != (STAGE_UPDATES * prior.GLOBAL_BATCH,)
        or tape.dtype.str != "<i8"
        or probe.shape != (1_024,)
        or probe.dtype.str != "<i8"
        or int(tape[0]) != 0
        or int(tape[-1]) + 257 != len(train)
        or int(probe[0]) != 0
        or int(probe[-1]) + 257 != len(validation)
    ):
        raise CalibrationError("300M v002 packed tape/probe geometry changed")
    return train, tape, validation, probe, manifest


def _load_checkpoint(
    path: Path, manifest: dict[str, Any], plan: dict[str, Any],
    model: torch.nn.Module, optimizer: HybridMuon,
    learning_rate: float, clip: float,
) -> float:
    state = torch.load(path, map_location=next(model.parameters()).device, weights_only=False)
    if (
        type(state) is not dict
        or state.get("schema_version") != CHECKPOINT_SCHEMA
        or state.get("run_id") != RUN_ID
        or state.get("campaign_id") != plan["campaign_id"]
        or state.get("completed_updates") != STAGE_UPDATES
        or state.get("learning_rate") != learning_rate
        or state.get("gradient_clip") != clip
        or state.get("source_data_sha256") != manifest["source_data_sha256"]
        or state.get("optimizer_partition_sha256") != optimizer.partition["sha256"]
    ):
        raise CalibrationError("v002 checkpoint cannot resume this arm or tape")
    model.load_state_dict(state["model_state_dict"])
    optimizer.load_state_dict(state["optimizer_state_dict"])
    torch.set_rng_state(state["cpu_rng_state"].cpu())
    torch.cuda.set_rng_state_all([value.cpu() for value in state["cuda_rng_state_all"]])
    if not optimizer_state_fp32_and_finite(optimizer):
        raise CalibrationError("resumed hybrid optimizer state is not finite FP32")
    return float(state["final_validation_ce"])


def _save_checkpoint(
    path: Path, plan: dict[str, Any], manifest: dict[str, Any],
    model: torch.nn.Module, optimizer: HybridMuon,
    learning_rate: float, clip: float, completed_updates: int, final_ce: float,
) -> str:
    if path.exists() or path.is_symlink():
        raise CalibrationError("refusing to overwrite v002 checkpoint")
    state = {
        "schema_version": CHECKPOINT_SCHEMA,
        "run_id": RUN_ID,
        "campaign_id": plan["campaign_id"],
        "source_data_sha256": manifest["source_data_sha256"],
        "completed_updates": completed_updates,
        "learning_rate": learning_rate,
        "gradient_clip": clip,
        "final_validation_ce": final_ce,
        "optimizer_partition_sha256": optimizer.partition["sha256"],
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "cpu_rng_state": torch.get_rng_state(),
        "cuda_rng_state_all": torch.cuda.get_rng_state_all(),
    }
    torch.save(state, path)
    return b0._sha256(path)


def _arm(
    train: np.ndarray, tape: np.ndarray, validation: np.ndarray, probe: np.ndarray,
    device: torch.device, model: torch.nn.Module, optimizer: HybridMuon,
    start_update: int, learning_rate: float, clip: float,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    ticks = set(eval_ticks(start_update))
    ce: list[float] = []
    norms: list[float] = []
    clipped: list[bool] = []
    durations: list[float] = []
    evaluations = [(start_update, b0._validation_ce(model, validation, probe, device))]
    for local_update in range(1, STAGE_UPDATES + 1):
        global_update = start_update + local_update
        rate = learning_rate * min(1.0, global_update / WARMUP_UPDATES)
        row = prior._step(
            model, optimizer, train, tape, device, local_update - 1, rate, clip,
        )
        ce.append(row[0])
        norms.append(row[1])
        clipped.append(row[2])
        durations.append(row[3])
        if global_update in ticks:
            evaluations.append((global_update, b0._validation_ce(model, validation, probe, device)))
        if local_update % 512 == 0:
            print(json.dumps({"event": "progress", "lr": learning_rate, "clip": clip,
                              "completed_updates": global_update}), flush=True)
    if not optimizer_state_fp32_and_finite(optimizer):
        raise CalibrationError("v002 hybrid optimizer state is not finite FP32")
    if not all(bool(torch.all(torch.isfinite(parameter)).detach().cpu()) for parameter in model.parameters()):
        raise CalibrationError("v002 model parameter is nonfinite")
    row = {
        "learning_rate": learning_rate,
        "gradient_clip": clip,
        "start_update": start_update,
        "completed_updates": start_update + STAGE_UPDATES,
        "initial_validation_ce": evaluations[0][1],
        "final_validation_ce": evaluations[-1][1],
        "startup_clipping_fraction": float(np.mean(clipped[:WARMUP_UPDATES])) if start_update == 0 else None,
        "post_warmup_clipping_fraction": float(np.mean(clipped[WARMUP_UPDATES:])) if start_update == 0 else float(np.mean(clipped)),
        "preclip_gradient_norm_p50": float(np.quantile(norms, .5)),
        "preclip_gradient_norm_p95": float(np.quantile(norms, .95)),
        "tokens_per_second": STAGE_UPDATES * prior.GLOBAL_BATCH * 256 / sum(durations),
        "model_and_optimizer_fp32_finite": True,
    }
    trace = {
        "training_ce": np.asarray(ce, dtype=np.float32),
        "preclip_gradient_norm": np.asarray(norms, dtype=np.float32),
        "gradient_clipped": np.asarray(clipped, dtype=np.bool_),
        "step_seconds": np.asarray(durations, dtype=np.float32),
        "evaluation_update": np.asarray([tick for tick, _ in evaluations], dtype=np.int64),
        "validation_ce": np.asarray([value for _, value in evaluations], dtype=np.float64),
    }
    return row, trace


def run(
    plan_path: Path, input_dir: Path, manifest_path: Path, output_dir: Path,
    learning_rate: float, clip: float = CONTROL_CLIP,
    start_update: int = 0, resume_checkpoint: Path | None = None,
) -> dict[str, Any]:
    if learning_rate not in LR_CANDIDATES or clip not in (CONTROL_CLIP, STARTUP_CLIP_CONTROL):
        raise CalibrationError("v002 arm changed from the prospective grid")
    if clip == STARTUP_CLIP_CONTROL and start_update != 0:
        raise CalibrationError("clipping control must run from initialization")
    root = Path(__file__).resolve().parents[2]
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    scale.validate_plan(plan, root, verify_data=False)
    validate_config(root, plan)
    train, tape, validation, probe, manifest = _inputs(
        input_dir, manifest_path, plan, start_update, resume_checkpoint,
    )
    if output_dir.exists() or output_dir.is_symlink():
        raise CalibrationError(f"refusing to overwrite v002 output: {output_dir}")
    device = b0._runtime()
    model = b0._new_model(device)
    optimizer = HybridMuon(model, learning_rate / WARMUP_UPDATES)
    expected_resume_ce = None
    if resume_checkpoint is not None:
        expected_resume_ce = _load_checkpoint(
            resume_checkpoint, manifest, plan, model, optimizer, learning_rate, clip,
        )
    output_dir.mkdir(parents=True)
    row, trace = _arm(
        train, tape, validation, probe, device, model, optimizer,
        start_update, learning_rate, clip,
    )
    if expected_resume_ce is not None and abs(row["initial_validation_ce"] - expected_resume_ce) > 1e-4:
        raise CalibrationError("resumed fixed probe CE differs from checkpoint anchor")
    trace_path = output_dir / "trace.npz"
    np.savez_compressed(trace_path, **trace)
    checkpoint_path = output_dir / "checkpoint.pt"
    checkpoint_sha = _save_checkpoint(
        checkpoint_path, plan, manifest, model, optimizer, learning_rate, clip,
        row["completed_updates"], row["final_validation_ce"],
    )
    result = {
        "schema_version": RESULT_SCHEMA,
        "run_id": RUN_ID,
        "campaign_id": plan["campaign_id"],
        "status": "completed_prospective_arm_no_formal_freeze",
        "phase": "initial" if start_update == 0 else "extension",
        "arm": row,
        "fixed_probe_evaluation_ticks": list(eval_ticks(start_update)),
        "source_data_sha256": manifest["source_data_sha256"],
        "input_manifest_sha256": b0._sha256(manifest_path),
        "trace_sha256": b0._sha256(trace_path),
        "checkpoint_sha256": checkpoint_sha,
        "formal_prefix_gpu_submission_authorized": False,
    }
    (output_dir / "result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8",
    )
    return result


def _late_means(trace: dict[str, np.ndarray], completed_updates: int) -> tuple[float, float]:
    ticks = dict(zip(trace["evaluation_update"].tolist(), trace["validation_ce"].tolist()))
    first = tuple(completed_updates - 4096 + 512 * index for index in range(4))
    last = tuple(completed_updates - 1536 + 512 * index for index in range(4))
    if any(tick not in ticks for tick in first + last):
        raise CalibrationError("v002 decision-window validation tick missing")
    return float(np.mean([ticks[tick] for tick in first])), float(np.mean([ticks[tick] for tick in last]))


def decide(result_dirs: tuple[Path, Path]) -> dict[str, Any]:
    arms = []
    source_identity = None
    for learning_rate, folder in zip(LR_CANDIDATES, result_dirs):
        result = json.loads((folder / "result.json").read_text(encoding="utf-8"))
        trace_path = folder / "trace.npz"
        if (
            result.get("schema_version") != RESULT_SCHEMA
            or result.get("run_id") != RUN_ID
            or result.get("arm", {}).get("learning_rate") != learning_rate
            or result["arm"].get("gradient_clip") != CONTROL_CLIP
            or result.get("trace_sha256") != b0._sha256(trace_path)
            or result.get("source_data_sha256") is None
        ):
            raise CalibrationError("prospective LR decision input changed")
        if source_identity is None:
            source_identity = result["source_data_sha256"]
        elif source_identity != result["source_data_sha256"]:
            raise CalibrationError("prospective LR arms used different source tapes")
        with np.load(trace_path, allow_pickle=False) as data:
            trace = {name: data[name] for name in data.files}
        completed = result["arm"]["completed_updates"]
        expected_start = completed - STAGE_UPDATES
        if (
            expected_start not in (0, STAGE_UPDATES)
            or result["arm"].get("start_update") != expected_start
            or trace["training_ce"].shape != (STAGE_UPDATES,)
            or trace["validation_ce"].shape != (len(eval_ticks(expected_start)),)
            or trace["evaluation_update"].tolist() != list(eval_ticks(expected_start))
        ):
            raise CalibrationError("prospective LR arm duration or evaluation grid changed")
        first, last = _late_means(trace, completed)
        passed = bool(
            result["arm"].get("model_and_optimizer_fp32_finite") is True
            and math.isfinite(first) and math.isfinite(last)
            and last < first
            and result["arm"].get("post_warmup_clipping_fraction", 1) <= .10
            and np.all(np.isfinite(trace["training_ce"]))
            and np.all(np.isfinite(trace["validation_ce"]))
        )
        arms.append({"learning_rate": learning_rate, "completed_updates": completed,
                     "first_four_late_probe_mean": first, "last_four_probe_mean": last,
                     "post_warmup_clipping_fraction": result["arm"]["post_warmup_clipping_fraction"],
                     "passed_late_trend_gate": passed})
    if arms[0]["completed_updates"] != arms[1]["completed_updates"]:
        raise CalibrationError("LR candidates were not compared at equal updates")
    passed = [arm for arm in arms if arm["passed_late_trend_gate"]]
    if not passed:
        status, chosen = "no_go", None
    elif len(passed) == 2 and abs(passed[0]["last_four_probe_mean"] - passed[1]["last_four_probe_mean"]) <= LATE_TIE_BAND_CE and arms[0]["completed_updates"] == STAGE_UPDATES:
        status, chosen = "extend_both_to_16384", None
    elif len(passed) == 2 and abs(passed[0]["last_four_probe_mean"] - passed[1]["last_four_probe_mean"]) <= LATE_TIE_BAND_CE:
        status, chosen = "provisional_lr_near_tie_clip_unresolved", LR_CANDIDATES[0]
    else:
        status = "provisional_lr_candidate_clip_unresolved"
        chosen = min(passed, key=lambda arm: arm["last_four_probe_mean"])["learning_rate"]
    return {
        "schema_version": "nanogpt300m_e2e_muon_calibration_decision_v002",
        "run_id": RUN_ID,
        "status": status,
        "arms": arms,
        "provisional_learning_rate": chosen,
        "gradient_clip_frozen": False,
        "formal_prefix_gpu_submission_authorized": False,
        "rule": "finite_states_and_late_four_point_mean_decrease_and_postwarmup_clip_le_10pct;tie_le_0p01_extend_both",
    }


def main() -> int:
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, default=root / "experiments/configs/nanogpt300m_e2e_6p5b_v001.json")
    parser.add_argument("--input-dir", type=Path)
    parser.add_argument("--input-manifest", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--gradient-clip", type=float, default=CONTROL_CLIP)
    parser.add_argument("--start-update", type=int, default=0)
    parser.add_argument("--resume-checkpoint", type=Path)
    parser.add_argument("--decide", action="store_true")
    parser.add_argument("--low-result-dir", type=Path)
    parser.add_argument("--mid-result-dir", type=Path)
    args = parser.parse_args()
    if args.decide:
        if args.low_result_dir is None or args.mid_result_dir is None:
            parser.error("--decide requires both result directories")
        print(json.dumps(decide((args.low_result_dir, args.mid_result_dir)), indent=2, sort_keys=True))
    else:
        if any(value is None for value in (args.input_dir, args.input_manifest, args.output_dir, args.learning_rate)):
            parser.error("an arm requires input, manifest, output, and learning rate")
        print(json.dumps(run(
            args.plan, args.input_dir, args.input_manifest, args.output_dir,
            args.learning_rate, args.gradient_clip, args.start_update, args.resume_checkpoint,
        ), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
