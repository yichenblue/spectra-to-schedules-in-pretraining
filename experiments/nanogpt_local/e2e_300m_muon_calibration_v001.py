"""One-H200, maturity-aware 300M Muon LR/clipping stability calibration."""

from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path
import time
from typing import Any

import numpy as np
import torch

from . import e2e_300m_b0_v001 as b0
from . import e2e_300m_6p5b_plan_v001 as scale
from .muon_300m_v001 import HybridMuon, optimizer_state_fp32_and_finite


RUN_ID = "nanogpt300m-e2e-muon-calibration-v001"
RESULT_SCHEMA = "nanogpt300m_e2e_muon_calibration_result_v001"
INPUT_SCHEMA = "nanogpt300m_e2e_muon_calibration_inputs_v001"
GLOBAL_BATCH = 256
MICRO_BATCH = 32
DIAGNOSTIC_UPDATES = 128
PREFIX_WARMUP_UPDATES = 256
SHARED_MATURITY_UPDATES = 2_048
BRANCH_UPDATES = 256
PACKED_TAPE_UPDATES = SHARED_MATURITY_UPDATES + BRANCH_UPDATES
REFERENCE_LR = 1.5e-4
LR_GRID = (7.5e-5, REFERENCE_LR, 3.0e-4)
INITIAL_CLIP = 1.0


class CalibrationError(RuntimeError):
    """The input, GPU, or shared-maturity stability gate failed."""


def _inputs(folder: Path, manifest_path: Path, plan: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    source = {key: plan["data"][key] for key in (
        "train_sha256", "training_tape_file_sha256", "validation_sha256", "validation_probe_sha256",
    )}
    if (
        manifest.get("schema_version") != INPUT_SCHEMA
        or manifest.get("run_id") != RUN_ID
        or manifest.get("campaign_id") != plan["campaign_id"]
        or manifest.get("source_data_sha256") != source
        or manifest.get("packed_tape_updates") != PACKED_TAPE_UPDATES
    ):
        raise CalibrationError("300M calibration input provenance changed")
    names = ("pretrain_train.bin", "training_tape.npy", "pretrain_validation.bin", "validation_probe.npy")
    if set(manifest.get("files", {})) != set(names):
        raise CalibrationError("300M calibration input inventory changed")
    for name in names:
        path = folder / name
        record = manifest["files"][name]
        if (
            not path.is_file() or path.is_symlink()
            or path.stat().st_size != record.get("size_bytes")
            or b0._sha256(path) != record.get("sha256")
        ):
            raise CalibrationError(f"300M calibration packed input changed: {name}")
    train = np.memmap(folder / names[0], dtype="<u2", mode="r")
    tape = np.load(folder / names[1], mmap_mode="r", allow_pickle=False)
    validation = np.memmap(folder / names[2], dtype="<u2", mode="r")
    probe = np.load(folder / names[3], allow_pickle=False)
    if (
        tape.shape != (PACKED_TAPE_UPDATES * GLOBAL_BATCH,)
        or probe.shape != (1_024,)
        or tape.dtype.str != "<i8"
        or probe.dtype.str != "<i8"
        or int(tape[0]) != 0
        or int(tape[-1]) + 257 != len(train)
        or int(probe[0]) != 0
        or int(probe[-1]) + 257 != len(validation)
    ):
        raise CalibrationError("300M calibration packed tape/probe geometry changed")
    return train, tape, validation, probe


def _step(
    model: torch.nn.Module,
    optimizer: HybridMuon,
    train: np.ndarray,
    tape: np.ndarray,
    device: torch.device,
    update_index: int,
    learning_rate: float,
    clip: float,
) -> tuple[float, float, bool, float]:
    for group in optimizer.param_groups:
        group["lr"] = learning_rate
    optimizer.zero_grad(set_to_none=True)
    contexts = tape[update_index * GLOBAL_BATCH : (update_index + 1) * GLOBAL_BATCH]
    if len(contexts) != GLOBAL_BATCH:
        raise CalibrationError("calibration tape ended before an update")
    torch.cuda.synchronize(device)
    started = time.monotonic()
    ce = 0.0
    for begin in range(0, GLOBAL_BATCH, MICRO_BATCH):
        x, y = b0._batch(train, contexts[begin : begin + MICRO_BATCH], device)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            _, loss, _ = model(x, y)
        if loss is None or not bool(torch.isfinite(loss).detach().cpu()):
            raise CalibrationError("nonfinite 300M calibration training CE")
        ce += float(loss.detach().cpu()) * MICRO_BATCH / GLOBAL_BATCH
        (loss * MICRO_BATCH / GLOBAL_BATCH).backward()
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), clip, foreach=True)
    if not bool(torch.isfinite(norm).detach().cpu()):
        raise CalibrationError("nonfinite 300M calibration preclip gradient norm")
    norm_value = float(norm.detach().cpu())
    optimizer.step()
    torch.cuda.synchronize(device)
    return ce, norm_value, norm_value > clip, time.monotonic() - started


def _segment(
    model: torch.nn.Module,
    optimizer: HybridMuon,
    train: np.ndarray,
    tape: np.ndarray,
    validation: np.ndarray,
    probe: np.ndarray,
    device: torch.device,
    begin: int,
    updates: int,
    lr: float,
    clip: float,
    eval_ticks: tuple[int, ...],
    warmup: bool = False,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    ce: list[float] = []
    norms: list[float] = []
    clipped: list[bool] = []
    durations: list[float] = []
    evaluations: list[tuple[int, float]] = []
    if 0 in eval_ticks:
        evaluations.append((0, b0._validation_ce(model, validation, probe, device)))
    for local_update in range(1, updates + 1):
        rate = lr * min(1.0, (begin + local_update) / PREFIX_WARMUP_UPDATES) if warmup else lr
        row = _step(model, optimizer, train, tape, device, begin + local_update - 1, rate, clip)
        ce.append(row[0])
        norms.append(row[1])
        clipped.append(row[2])
        durations.append(row[3])
        if local_update in eval_ticks:
            evaluations.append((local_update, b0._validation_ce(model, validation, probe, device)))
        if local_update % 256 == 0:
            print(json.dumps({"event": "progress", "begin": begin, "completed": local_update, "total": updates, "lr": lr, "clip": clip}), flush=True)
    if not optimizer_state_fp32_and_finite(optimizer):
        raise CalibrationError("300M calibration hybrid optimizer state is not finite FP32")
    result = {
        "learning_rate": lr,
        "gradient_clip": clip,
        "completed_updates": updates,
        "initial_validation_ce": evaluations[0][1] if evaluations else None,
        "final_validation_ce": evaluations[-1][1] if evaluations else None,
        "clipping_fraction": float(np.mean(clipped)),
        "preclip_gradient_norm_p50": float(np.quantile(norms, 0.5)),
        "preclip_gradient_norm_p95": float(np.quantile(norms, 0.95)),
        "tokens_per_second": updates * GLOBAL_BATCH * 256 / sum(durations),
        "optimizer_state_fp32_and_finite": True,
    }
    trace = {
        "training_ce": np.asarray(ce, dtype=np.float32),
        "preclip_gradient_norm": np.asarray(norms, dtype=np.float32),
        "gradient_clipped": np.asarray(clipped, dtype=np.bool_),
        "step_seconds": np.asarray(durations, dtype=np.float32),
        "evaluation_update": np.asarray([row[0] for row in evaluations], dtype=np.int64),
        "validation_ce": np.asarray([row[1] for row in evaluations], dtype=np.float64),
    }
    return result, trace


def recommend(warmup: dict[str, Any], branches: list[dict[str, Any]], adaptive_clip: float) -> dict[str, Any]:
    reference = next((arm for arm in branches if arm["learning_rate"] == REFERENCE_LR and arm["gradient_clip"] == adaptive_clip), None)
    stable = bool(
        reference is not None
        and reference.get("status") == "completed"
        and reference.get("optimizer_state_fp32_and_finite") is True
        and math.isfinite(float(reference["final_validation_ce"]))
        and reference["final_validation_ce"] < reference["initial_validation_ce"]
        and reference["clipping_fraction"] <= 0.10
        and warmup["final_validation_ce"] < warmup["initial_validation_ce"]
        and warmup["clipping_fraction"] <= 0.10
    )
    return {
        "status": "short_window_go" if stable else "short_window_no_go",
        "nominal_learning_rate": REFERENCE_LR if stable else None,
        "gradient_clip": adaptive_clip if stable else None,
        "rule": "retain_reference_lr_if_shared_maturity_and_reference_branch_decrease_with_at_most_10pct_clipping; outer_lrs_are_guardrails",
    }


def run(plan_path: Path, input_dir: Path, manifest_path: Path, output_dir: Path) -> dict[str, Any]:
    root = Path(__file__).resolve().parents[2]
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    scale.validate_plan(plan, root, verify_data=False)
    train, tape, validation, probe = _inputs(input_dir, manifest_path, plan)
    if output_dir.exists() or output_dir.is_symlink():
        raise CalibrationError(f"refusing to overwrite calibration output: {output_dir}")
    device = b0._runtime()
    output_dir.mkdir(parents=True)

    diagnostic_model = b0._new_model(device)
    diagnostic_optimizer = HybridMuon(diagnostic_model, REFERENCE_LR / PREFIX_WARMUP_UPDATES)
    diagnostic, diagnostic_trace = _segment(
        diagnostic_model, diagnostic_optimizer, train, tape, validation, probe, device,
        0, DIAGNOSTIC_UPDATES, REFERENCE_LR, INITIAL_CLIP, (0, DIAGNOSTIC_UPDATES), warmup=True,
    )
    adaptive_clip = float(round(max(1.0, 1.25 * diagnostic["preclip_gradient_norm_p95"]), 3))
    if not math.isfinite(adaptive_clip) or adaptive_clip <= INITIAL_CLIP:
        raise CalibrationError("adaptive pilot clipping threshold did not separate from 1.0")
    del diagnostic_model, diagnostic_optimizer
    torch.cuda.empty_cache()

    model = b0._new_model(device)
    optimizer = HybridMuon(model, REFERENCE_LR / PREFIX_WARMUP_UPDATES)
    shared, shared_trace = _segment(
        model, optimizer, train, tape, validation, probe, device,
        0, SHARED_MATURITY_UPDATES, REFERENCE_LR, adaptive_clip,
        (0, 256, 1_024, SHARED_MATURITY_UPDATES), warmup=True,
    )
    model_state = copy.deepcopy(model.state_dict())
    optimizer_state = copy.deepcopy(optimizer.state_dict())
    branch_rows: list[dict[str, Any]] = []
    trace_files: dict[str, dict[str, Any]] = {}
    for learning_rate in LR_GRID:
        for clip in (INITIAL_CLIP, adaptive_clip):
            model.load_state_dict(model_state)
            optimizer.load_state_dict(optimizer_state)
            name = f"lr{learning_rate:.1e}-clip{clip:.3f}"
            try:
                row, trace = _segment(
                    model, optimizer, train, tape, validation, probe, device,
                    SHARED_MATURITY_UPDATES, BRANCH_UPDATES, learning_rate, clip,
                    (0, 128, BRANCH_UPDATES),
                )
                row["status"] = "completed"
                target = output_dir / f"{name}.npz"
                np.savez_compressed(target, **trace)
                trace_files[name] = {"path": target.name, "sha256": b0._sha256(target)}
            except CalibrationError as error:
                row = {"learning_rate": learning_rate, "gradient_clip": clip, "status": "failed_numerical_gate", "reason": str(error)}
            branch_rows.append(row)
    result = {
        "schema_version": RESULT_SCHEMA,
        "run_id": RUN_ID,
        "campaign_id": plan["campaign_id"],
        "status": "completed_short_window_calibration",
        "diagnostic_updates": DIAGNOSTIC_UPDATES,
        "shared_maturity_updates": SHARED_MATURITY_UPDATES,
        "branch_updates_per_arm": BRANCH_UPDATES,
        "selected_adaptive_clip": adaptive_clip,
        "diagnostic": diagnostic,
        "shared_maturity": shared,
        "branches": branch_rows,
        "recommendation": recommend(shared, branch_rows, adaptive_clip),
        "trace_files": trace_files,
        "formal_prefix_gpu_submission_authorized": False,
    }
    for name, trace in (("diagnostic", diagnostic_trace), ("shared-maturity", shared_trace)):
        target = output_dir / f"{name}.npz"
        np.savez_compressed(target, **trace)
        trace_files[name] = {"path": target.name, "sha256": b0._sha256(target)}
    (output_dir / "result.json").write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    return result


def main() -> int:
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, default=root / "experiments/configs/nanogpt300m_e2e_6p5b_v001.json")
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--input-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.plan, args.input_dir, args.input_manifest, args.output_dir), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
