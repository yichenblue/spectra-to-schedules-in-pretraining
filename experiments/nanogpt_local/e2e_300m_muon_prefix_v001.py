"""Single-H200 300M hybrid-Muon shared prefix; preparation is launch-false."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import time
from typing import Any

import numpy as np
import torch

from . import e2e_300m_b0_v001 as b0
from . import e2e_300m_6p5b_plan_v001 as scale
from . import e2e_300m_muon_calibration_v001 as calibration_step
from . import e2e_300m_muon_calibration_v002 as calibration
from .muon_300m_v001 import HybridMuon, optimizer_state_fp32_and_finite


RUN_ID = "nanogpt300m-e2e-muon-prefix-v001"
CONFIG_SCHEMA = "nanogpt300m_e2e_muon_prefix_config_v001"
RESULT_SCHEMA = "nanogpt300m_e2e_muon_prefix_result_v001"
CHECKPOINT_SCHEMA = "nanogpt300m_e2e_muon_prefix_checkpoint_v001"
PREFIX_UPDATES = 79_346
GLOBAL_BATCH = 256
MICRO_BATCH = 32
LEARNING_RATE = 1.0e-4
GRADIENT_CLIP = 33.504
WARMUP_UPDATES = 256


class PrefixError(RuntimeError):
    """The pinned input, runtime, or prefix result did not meet its contract."""


def _is_nonempty_model_state_dict(value: Any) -> bool:
    return isinstance(value, dict) and bool(value)


def _root() -> Path:
    return Path(__file__).resolve().parents[2]


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if type(value) is not dict:
        raise PrefixError(f"JSON root is not an object: {path}")
    return value


def _grid() -> tuple[int, ...]:
    return scale.prefix_validation_updates(1)


def _grid_sha256() -> str:
    return hashlib.sha256(np.asarray(_grid(), dtype="<i8").tobytes()).hexdigest()


def _intrinsic_time() -> float:
    return LEARNING_RATE * ((WARMUP_UPDATES + 1) / 2 + PREFIX_UPDATES - WARMUP_UPDATES)


def validate_config(config: dict[str, Any], plan: dict[str, Any], root: Path) -> dict[str, Any]:
    scale.validate_plan(plan, root, verify_data=False)
    frozen = _read_json(root / "experiments/configs/nanogpt300m_e2e_muon_prefix_v001.json")
    if config != frozen:
        raise PrefixError("formal prefix config differs from its bundled frozen copy")
    if (
        config.get("schema_version") != CONFIG_SCHEMA
        or config.get("run_id") != RUN_ID
        or config.get("campaign_id") != plan["campaign_id"]
        or config.get("primary_prediction_id") != "A2_EXTERNAL"
        or config.get("theorem_facing") is not False
        or config.get("launch_control") != {
            "cluster_submission_authorized": False, "shared_data_uploaded": False,
        }
        or config.get("initialization_seed") != b0.INITIALIZATION_SEED
        or config.get("model_parameters") != plan["model"]["expected_parameter_count"]
        or config.get("prefix_updates") != plan["horizon"]["shared_prefix_updates"]
        or config.get("prefix_tokens") != plan["horizon"]["shared_prefix_tokens"]
        or config.get("global_batch_sequences") != GLOBAL_BATCH
        or config.get("micro_batch_sequences") != MICRO_BATCH
        or config.get("optimizer", {}).get("nominal_learning_rate") != LEARNING_RATE
        or config["optimizer"].get("gradient_clip") != GRADIENT_CLIP
        or config["optimizer"].get("warmup_updates") != WARMUP_UPDATES
        or config.get("validation", {}).get("sample_count") != len(_grid())
        or config["validation"].get("tail_regular_stride_macros") != 1
        or config["validation"].get("fixed_probe_contexts") != 1_024
        or config.get("hard_job_duration_seconds") != 43_200
    ):
        raise PrefixError("formal prefix identity, schedule, or launch boundary changed")
    for role, source_key in (
        ("train", "train_sha256"),
        ("training_tape", "training_tape_file_sha256"),
        ("validation", "validation_sha256"),
        ("validation_probe", "validation_probe_sha256"),
    ):
        record = config["data_files"][role]
        if record.get("sha256") != plan["data"][source_key]:
            raise PrefixError(f"formal prefix {role} source identity changed")
    if config["source_data_sha256"] != {
        key: plan["data"][key] for key in config["source_data_sha256"]
    }:
        raise PrefixError("formal prefix source-tape identity changed")
    return config


def verify_calibration(config: dict[str, Any], root: Path) -> dict[str, Any]:
    provenance = config["calibration_provenance"]
    b0_path = root / "experiments/results/nanogpt300m-e2e-muon-b0-v001/a001/result.json"
    low = root / "experiments/results/nanogpt300m-e2e-muon-calibration-v002/a001/lr7p5e05"
    high = root / "experiments/results/nanogpt300m-e2e-muon-calibration-v002/a001/lr1e04"
    submission = _read_json(root / "experiments/generated/nanogpt300m-e2e-muon-calibration-v002/a001/submission_record.json")
    b0_submission = _read_json(root / "experiments/generated/nanogpt300m-e2e-muon-b0-v001/a001/submission_record.json")
    if (
        b0._sha256(b0_path) != provenance["b0_result_sha256"]
        or b0._sha256(low / "result.json") != provenance["v002_low_result_sha256"]
        or b0._sha256(high / "result.json") != provenance["v002_high_result_sha256"]
        or b0_submission.get("job_id") != provenance["b0_job_id"]
        or submission.get("jobs", {}).get("lr7p5e05", {}).get("job_id") != provenance["v002_low_job_id"]
        or submission.get("jobs", {}).get("lr1e04", {}).get("job_id") != provenance["v002_high_job_id"]
    ):
        raise PrefixError("B0 or prospective LR calibration provenance changed")
    b0_result = _read_json(b0_path)
    candidate = b0_result.get("candidates", [{}])[0]
    if (
        b0_result.get("selected_micro_batch_sequences") != MICRO_BATCH
        or not math.isclose(candidate.get("fixed_probe_evaluation_seconds", -1),
                            provenance["b0_fixed_probe_evaluation_seconds"], abs_tol=1e-9)
    ):
        raise PrefixError("300M B0 micro-batch or probe timing changed")
    decision = calibration.decide((low, high))
    if (
        decision["status"] != "provisional_lr_candidate_clip_unresolved"
        or decision["provisional_learning_rate"] != LEARNING_RATE
        or not all(arm["passed_late_trend_gate"] for arm in decision["arms"])
        or not all(arm["post_warmup_clipping_fraction"] == 0.0 for arm in decision["arms"])
        or not math.isclose(decision["arms"][0]["last_four_probe_mean"],
                            provenance["v002_low_last_four_validation_ce"], abs_tol=1e-9)
        or not math.isclose(decision["arms"][1]["last_four_probe_mean"],
                            provenance["v002_high_last_four_validation_ce"], abs_tol=1e-9)
    ):
        raise PrefixError("prospective decision does not support the pinned settings")
    return decision


def preflight(config_path: Path, *, verify_data: bool = True) -> dict[str, Any]:
    root = _root()
    plan = _read_json(root / "experiments/configs/nanogpt300m_e2e_6p5b_v001.json")
    config = validate_config(_read_json(config_path), plan, root)
    decision = verify_calibration(config, root)
    data = scale.validate_plan(plan, root, verify_data=verify_data)
    ticks = _grid()
    near = [tick for tick in ticks if tick >= PREFIX_UPDATES - 4_096]
    if len(near) != 4_097 or near != list(range(PREFIX_UPDATES - 4_096, PREFIX_UPDATES + 1)):
        raise PrefixError("fork-side validation density changed")
    return {
        "schema_version": "nanogpt300m_e2e_muon_prefix_preflight_v001",
        "run_id": RUN_ID,
        "status": "offline_ready_pending_shared_upload_and_submission_authorization",
        "data_gate": data["status"],
        "calibration_gate": decision["status"],
        "validation_samples": len(ticks),
        "first_validation_update": ticks[0],
        "last_validation_update": ticks[-1],
        "near_fork_validation_samples": len(near),
        "validation_grid_sha256": _grid_sha256(),
        "estimated_gpu_hours_before_transfer_and_checkpoint": config["estimated_total_gpu_hours_before_transfer_and_checkpoint"],
        "shared_data_uploaded": False,
        "cluster_submission_authorized": False,
    }


def _load_inputs(config: dict[str, Any], workdir: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    files = {}
    for role, record in config["data_files"].items():
        path = workdir / record["path"]
        if (
            not path.is_file() or path.is_symlink()
            or path.stat().st_size != record["size_bytes"]
            or b0._sha256(path) != record["sha256"]
        ):
            raise PrefixError(f"formal prefix transferred {role} changed")
        files[role] = path
    train = np.memmap(files["train"], dtype="<u2", mode="r")
    tape = np.load(files["training_tape"], mmap_mode="r", allow_pickle=False)
    validation = np.memmap(files["validation"], dtype="<u2", mode="r")
    probe = np.load(files["validation_probe"], allow_pickle=False)
    if (
        len(train) != 6_500_057_344
        or tape.shape != (99_183 * GLOBAL_BATCH,)
        or tape.dtype.str != "<i8"
        or len(validation) != 16_777_216
        or probe.shape != (1_024,)
        or probe.dtype.str != "<i8"
        or int(np.min(tape)) < 0
        or int(np.max(tape)) + 257 > len(train)
        or int(np.min(probe)) < 0
        or int(np.max(probe)) + 257 > len(validation)
    ):
        raise PrefixError("formal prefix loaded tape/probe geometry changed")
    return train, tape, validation, probe


def run(config_path: Path, output_dir: Path) -> dict[str, Any]:
    root = _root()
    plan = _read_json(root / "experiments/configs/nanogpt300m_e2e_6p5b_v001.json")
    config = validate_config(_read_json(config_path), plan, root)
    if output_dir.exists() or output_dir.is_symlink():
        raise PrefixError(f"refusing to overwrite formal prefix output: {output_dir}")
    train, tape, validation, probe = _load_inputs(config, Path.cwd())
    device = b0._runtime()
    model = b0._new_model(device)
    optimizer = HybridMuon(model, LEARNING_RATE / WARMUP_UPDATES)
    output_dir.mkdir(parents=True)
    ticks = _grid()
    scheduled = set(ticks)
    trace_ce = np.empty(PREFIX_UPDATES, dtype=np.float32)
    trace_norm = np.empty(PREFIX_UPDATES, dtype=np.float32)
    trace_clipped = np.empty(PREFIX_UPDATES, dtype=np.bool_)
    trace_seconds = np.empty(PREFIX_UPDATES, dtype=np.float32)
    trace_rate = np.empty(PREFIX_UPDATES, dtype=np.float64)
    trace_time = np.empty(PREFIX_UPDATES, dtype=np.float64)
    validation_ce = np.empty(len(ticks), dtype=np.float64)
    validation_time = np.empty(len(ticks), dtype=np.float64)
    validation_intrinsic = np.empty(len(ticks), dtype=np.float64)
    torch.cuda.synchronize(device)
    initial_evaluation_began = time.monotonic()
    validation_ce[0] = b0._validation_ce(model, validation, probe, device)
    torch.cuda.synchronize(device)
    validation_time[0] = time.monotonic() - initial_evaluation_began
    validation_intrinsic[0] = 0.0
    position = 1
    intrinsic = 0.0
    started = time.monotonic()
    for update in range(1, PREFIX_UPDATES + 1):
        rate = LEARNING_RATE * min(1.0, update / WARMUP_UPDATES)
        ce, norm, clipped, seconds = calibration_step._step(
            model, optimizer, train, tape, device, update - 1, rate, GRADIENT_CLIP,
        )
        intrinsic += rate
        trace_ce[update - 1] = ce
        trace_norm[update - 1] = norm
        trace_clipped[update - 1] = clipped
        trace_seconds[update - 1] = seconds
        trace_rate[update - 1] = rate
        trace_time[update - 1] = intrinsic
        if update in scheduled:
            torch.cuda.synchronize(device)
            began = time.monotonic()
            validation_ce[position] = b0._validation_ce(model, validation, probe, device)
            torch.cuda.synchronize(device)
            validation_time[position] = time.monotonic() - began
            validation_intrinsic[position] = intrinsic
            position += 1
        if update % 1_024 == 0 or update == PREFIX_UPDATES:
            print(json.dumps({"event": "progress", "run_id": RUN_ID,
                              "completed_updates": update, "total_updates": PREFIX_UPDATES}), flush=True)
    if (
        position != len(ticks)
        or not math.isclose(intrinsic, _intrinsic_time(), rel_tol=0.0, abs_tol=1e-9)
        or not optimizer_state_fp32_and_finite(optimizer)
        or not all(bool(torch.all(torch.isfinite(parameter)).detach().cpu()) for parameter in model.parameters())
        or not np.all(np.isfinite(trace_ce))
        or not np.all(np.isfinite(validation_ce))
    ):
        raise PrefixError("formal prefix completed with a nonfinite or incomplete state")
    trace_path = output_dir / "training_trace.npz"
    np.savez_compressed(
        trace_path,
        optimizer_update=np.arange(1, PREFIX_UPDATES + 1, dtype=np.int64),
        source_update=np.arange(1, PREFIX_UPDATES + 1, dtype=np.int64),
        tokens_seen=np.arange(1, PREFIX_UPDATES + 1, dtype=np.int64) * 65_536,
        batch_size=np.full(PREFIX_UPDATES, GLOBAL_BATCH, dtype=np.int32),
        learning_rate=trace_rate,
        nominal_ratio_B_over_eta=GLOBAL_BATCH / trace_rate,
        intrinsic_time_after=trace_time,
        training_cross_entropy=trace_ce,
        preclip_gradient_norm=trace_norm,
        gradient_clipped=trace_clipped,
        step_time_seconds=trace_seconds,
    )
    evaluations_path = output_dir / "validation_evaluations.npz"
    np.savez_compressed(
        evaluations_path,
        optimizer_update=np.asarray(ticks, dtype=np.int64),
        source_update=np.asarray(ticks, dtype=np.int64),
        tokens_seen=np.asarray(ticks, dtype=np.int64) * 65_536,
        intrinsic_time=validation_intrinsic,
        validation_cross_entropy=validation_ce,
        evaluation_time_seconds=validation_time,
    )
    checkpoint_path = output_dir / "prefix_checkpoint.pt"
    torch.save({
        "schema_version": CHECKPOINT_SCHEMA,
        "run_id": RUN_ID,
        "campaign_id": plan["campaign_id"],
        "source_data_sha256": config["source_data_sha256"],
        "config_sha256": b0._sha256(config_path),
        "validation_grid_sha256": _grid_sha256(),
        "optimizer_partition_sha256": optimizer.partition["sha256"],
        "initialization_seed": b0.INITIALIZATION_SEED,
        "optimizer_update": PREFIX_UPDATES,
        "source_update": PREFIX_UPDATES,
        "tape_context_cursor": PREFIX_UPDATES * GLOBAL_BATCH,
        "tokens_seen": PREFIX_UPDATES * GLOBAL_BATCH * 256,
        "intrinsic_time": intrinsic,
        "nominal_learning_rate": LEARNING_RATE,
        "gradient_clip": GRADIENT_CLIP,
        "validation_anchor_ce": float(validation_ce[-1]),
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "cpu_rng_state": torch.get_rng_state(),
        "cuda_rng_state_all": torch.cuda.get_rng_state_all(),
    }, checkpoint_path)
    result = {
        "schema_version": RESULT_SCHEMA,
        "run_id": RUN_ID,
        "campaign_id": plan["campaign_id"],
        "phase": "shared_constant_r_prefix",
        "status": "passed",
        "evidence": {"primary_prediction_id": "A2_EXTERNAL", "theorem_facing": False},
        "optimizer_updates": PREFIX_UPDATES,
        "source_updates": PREFIX_UPDATES,
        "training_tokens": PREFIX_UPDATES * GLOBAL_BATCH * 256,
        "final_nominal_intrinsic_time": intrinsic,
        "validation_samples": len(ticks),
        "validation_grid_sha256": _grid_sha256(),
        "initial_validation_ce": float(validation_ce[0]),
        "final_validation_ce": float(validation_ce[-1]),
        "startup_clipping_fraction": float(np.mean(trace_clipped[:WARMUP_UPDATES])),
        "post_warmup_clipping_fraction": float(np.mean(trace_clipped[WARMUP_UPDATES:])),
        "model_and_optimizer_fp32_finite": True,
        "source_data_sha256": config["source_data_sha256"],
        "config_sha256": b0._sha256(config_path),
        "calibration_provenance": config["calibration_provenance"],
        "runtime": {
            "device_name": torch.cuda.get_device_name(device),
            "torch_version": torch.__version__,
            "elapsed_seconds": time.monotonic() - started,
            "training_tokens_per_second": PREFIX_UPDATES * GLOBAL_BATCH * 256 / float(np.sum(trace_seconds)),
            "mean_fixed_probe_evaluation_seconds": float(np.mean(validation_time[1:])),
        },
        "artifacts": {
            "training_trace": {"path": trace_path.name, "sha256": b0._sha256(trace_path)},
            "validation_evaluations": {"path": evaluations_path.name, "sha256": b0._sha256(evaluations_path)},
            "prefix_checkpoint": {"path": checkpoint_path.name, "sha256": b0._sha256(checkpoint_path)},
        },
    }
    (output_dir / "result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8",
    )
    return result


def validate_result(config_path: Path, output_dir: Path) -> dict[str, Any]:
    root = _root()
    plan = _read_json(root / "experiments/configs/nanogpt300m_e2e_6p5b_v001.json")
    config = validate_config(_read_json(config_path), plan, root)
    result = _read_json(output_dir / "result.json")
    if (
        result.get("schema_version") != RESULT_SCHEMA
        or result.get("run_id") != RUN_ID
        or result.get("status") != "passed"
        or result.get("optimizer_updates") != PREFIX_UPDATES
        or result.get("source_updates") != PREFIX_UPDATES
        or result.get("training_tokens") != plan["horizon"]["shared_prefix_tokens"]
        or result.get("model_and_optimizer_fp32_finite") is not True
        or result.get("validation_samples") != len(_grid())
        or result.get("validation_grid_sha256") != _grid_sha256()
        or result.get("source_data_sha256") != config["source_data_sha256"]
        or result.get("config_sha256") != b0._sha256(config_path)
        or result.get("calibration_provenance") != config["calibration_provenance"]
        or not math.isclose(result.get("final_nominal_intrinsic_time", -1), _intrinsic_time(), abs_tol=1e-9)
    ):
        raise PrefixError("formal prefix result ledger changed")
    for role, filename in (
        ("training_trace", "training_trace.npz"),
        ("validation_evaluations", "validation_evaluations.npz"),
        ("prefix_checkpoint", "prefix_checkpoint.pt"),
    ):
        record = result["artifacts"][role]
        path = output_dir / filename
        if record.get("path") != filename or b0._sha256(path) != record.get("sha256"):
            raise PrefixError(f"formal prefix {role} bytes changed")
    with np.load(output_dir / "training_trace.npz", allow_pickle=False) as trace:
        if (
            trace["training_cross_entropy"].shape != (PREFIX_UPDATES,)
            or trace["optimizer_update"].tolist() != list(range(1, PREFIX_UPDATES + 1))
            or not np.all(np.isfinite(trace["training_cross_entropy"]))
            or not np.all(np.isfinite(trace["intrinsic_time_after"]))
        ):
            raise PrefixError("formal prefix per-update training trace changed")
    with np.load(output_dir / "validation_evaluations.npz", allow_pickle=False) as evaluations:
        if (
            evaluations["optimizer_update"].tolist() != list(_grid())
            or not np.all(np.isfinite(evaluations["validation_cross_entropy"]))
            or not math.isclose(float(evaluations["validation_cross_entropy"][-1]),
                                result["final_validation_ce"], abs_tol=1e-9)
        ):
            raise PrefixError("formal prefix fixed-probe evaluation grid changed")
    checkpoint = torch.load(output_dir / "prefix_checkpoint.pt", map_location="cpu", weights_only=False)
    optimizer_state = checkpoint.get("optimizer_state_dict", {}) if type(checkpoint) is dict else {}
    if (
        type(checkpoint) is not dict
        or checkpoint.get("schema_version") != CHECKPOINT_SCHEMA
        or checkpoint.get("run_id") != RUN_ID
        or checkpoint.get("optimizer_update") != PREFIX_UPDATES
        or checkpoint.get("source_update") != PREFIX_UPDATES
        or checkpoint.get("tape_context_cursor") != PREFIX_UPDATES * GLOBAL_BATCH
        or checkpoint.get("tokens_seen") != PREFIX_UPDATES * GLOBAL_BATCH * 256
        or checkpoint.get("source_data_sha256") != config["source_data_sha256"]
        or checkpoint.get("config_sha256") != b0._sha256(config_path)
        or checkpoint.get("validation_grid_sha256") != _grid_sha256()
        or checkpoint.get("nominal_learning_rate") != LEARNING_RATE
        or checkpoint.get("gradient_clip") != GRADIENT_CLIP
        or not math.isclose(checkpoint.get("intrinsic_time", -1), _intrinsic_time(), abs_tol=1e-9)
        or not math.isclose(checkpoint.get("validation_anchor_ce", -1), result["final_validation_ce"], abs_tol=1e-9)
        or not _is_nonempty_model_state_dict(checkpoint.get("model_state_dict"))
        or type(optimizer_state) is not dict
        or optimizer_state.get("schema_version") != "nanogpt300m_hybrid_muon_state_v001"
        or optimizer_state.get("partition_sha256") != checkpoint.get("optimizer_partition_sha256")
        or type(optimizer_state.get("muon")) is not dict
        or type(optimizer_state.get("aux_adamw")) is not dict
        or not optimizer_state["muon"].get("state")
        or not optimizer_state["aux_adamw"].get("state")
        or not torch.is_tensor(checkpoint.get("cpu_rng_state"))
        or type(checkpoint.get("cuda_rng_state_all")) is not list
        or not checkpoint["cuda_rng_state_all"]
    ):
        raise PrefixError("formal prefix checkpoint cannot seed matched tails")
    return {"schema_version": "nanogpt300m_e2e_muon_prefix_validation_v001",
            "run_id": RUN_ID, "status": "passed", "validation_samples": len(_grid()),
            "checkpoint_sha256": result["artifacts"]["prefix_checkpoint"]["sha256"]}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--preflight", action="store_true")
    mode.add_argument("--run-prefix", action="store_true")
    mode.add_argument("--validate-result", action="store_true")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--skip-full-data-hash", action="store_true")
    args = parser.parse_args()
    if args.preflight:
        result = preflight(args.config, verify_data=not args.skip_full_data_hash)
    elif args.run_prefix:
        if args.output_dir is None:
            parser.error("--run-prefix requires --output-dir")
        result = run(args.config, args.output_dir)
    else:
        if args.output_dir is None:
            parser.error("--validate-result requires --output-dir")
        result = validate_result(args.config, args.output_dir)
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
