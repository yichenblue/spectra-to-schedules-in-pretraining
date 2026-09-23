"""Short one-H200 300M hybrid-Muon BF16/Flash runtime pilot.

This response-free pilot checks that the new model/data/evaluation path runs.
Its four-update nominal learning rate is provisional; it does not freeze
production Muon settings and contains no upload or submission code.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time
from typing import Any

import numpy as np
import torch

from . import build_openwebtext_cooldown_6p5b_append_v001 as corpus
from . import e2e_300m_6p5b_plan_v001 as scale
from .muon_300m_v001 import HybridMuon, optimizer_state_fp32_and_finite
from .model import GPT, GPTConfig


MICRO_BATCH_CANDIDATES = (32, 16, 8)
GLOBAL_BATCH = 256
UPDATES_PER_CANDIDATE = 4
PILOT_LR = 1.5e-4
PILOT_CLIP = 1.0
VALIDATION_BATCH = 32
INITIALIZATION_SEED = 2_026_091_502


class B0Error(RuntimeError):
    """The 300M pilot failed an input, runtime, or numerical gate."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _packed_inputs(input_dir: Path, manifest_path: Path, plan: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        manifest.get("schema_version") != "nanogpt300m_e2e_muon_b0_inputs_v001"
        or manifest.get("campaign_id") != plan["campaign_id"]
        or manifest.get("source_data_sha256") != {
            key: plan["data"][key] for key in (
                "train_sha256", "training_tape_file_sha256",
                "validation_sha256", "validation_probe_sha256",
            )
        }
    ):
        raise B0Error("packed B0 input provenance changed")
    names = ("pretrain_train.bin", "training_tape.npy", "pretrain_validation.bin", "validation_probe.npy")
    if set(manifest.get("files", {})) != set(names):
        raise B0Error("packed B0 input inventory changed")
    for name in names:
        path = input_dir / name
        record = manifest["files"][name]
        if (
            not path.is_file() or path.is_symlink()
            or path.stat().st_size != record.get("size_bytes")
            or _sha256(path) != record.get("sha256")
        ):
            raise B0Error(f"packed B0 input changed: {name}")
    train = np.memmap(input_dir / names[0], dtype="<u2", mode="r")
    tape = np.load(input_dir / names[1], mmap_mode="r", allow_pickle=False)
    validation = np.memmap(input_dir / names[2], dtype="<u2", mode="r")
    probe = np.load(input_dir / names[3], allow_pickle=False)
    if (
        tape.shape != (UPDATES_PER_CANDIDATE * GLOBAL_BATCH,)
        or probe.shape != (1_024,)
        or tape.dtype.str != "<i8"
        or probe.dtype.str != "<i8"
        or int(np.min(tape)) < 0
        or int(np.max(tape)) + 257 > len(train)
        or int(np.min(probe)) < 0
        or int(np.max(probe)) + 257 > len(validation)
    ):
        raise B0Error("packed B0 input geometry changed")
    return train, tape, validation, probe


def _runtime() -> torch.device:
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise B0Error("B0 requires one visible CUDA GPU")
    device = torch.device("cuda", 0)
    if "H200" not in torch.cuda.get_device_name(device):
        raise B0Error("B0 requires an H200")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cuda.enable_flash_sdp(True)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_cudnn_sdp(False)
    torch.backends.cuda.enable_math_sdp(False)
    if not torch.backends.cuda.flash_sdp_enabled() or torch.backends.cuda.math_sdp_enabled():
        raise B0Error("Flash-only SDPA configuration failed")
    return device


def _new_model(device: torch.device) -> GPT:
    torch.manual_seed(INITIALIZATION_SEED)
    torch.cuda.manual_seed_all(INITIALIZATION_SEED)
    model = GPT(
        GPTConfig(
            block_size=256,
            vocab_size=50_304,
            n_layer=20,
            n_head=16,
            n_embd=1_024,
            dropout=0.0,
            bias=False,
        )
    )
    if sum(parameter.numel() for parameter in model.parameters()) != corpus.PARAMETERS:
        raise B0Error("instantiated 300M parameter count changed")
    return model.to(device)


def _batch(stream: np.ndarray, offsets: np.ndarray, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    positions = np.asarray(offsets, dtype=np.int64)[:, None] + np.arange(257, dtype=np.int64)[None, :]
    values = np.asarray(stream[positions], dtype=np.int64)
    x = torch.from_numpy(np.ascontiguousarray(values[:, :-1])).to(device)
    y = torch.from_numpy(np.ascontiguousarray(values[:, 1:])).to(device)
    return x, y


@torch.inference_mode()
def _validation_ce(
    model: GPT, validation: np.ndarray, probe: np.ndarray, device: torch.device
) -> float:
    model.eval()
    values = []
    for start in range(0, len(probe), VALIDATION_BATCH):
        x, y = _batch(validation, probe[start : start + VALIDATION_BATCH], device)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            _, loss, _ = model(x, y)
        if loss is None or not bool(torch.isfinite(loss).detach().cpu()):
            raise B0Error("nonfinite fixed-probe validation CE")
        values.append(float(loss.detach().cpu()))
    model.train()
    return float(np.mean(values, dtype=np.float64))


def _candidate(
    micro_batch: int,
    train: np.ndarray,
    tape: np.ndarray,
    device: torch.device,
) -> tuple[dict[str, Any], GPT]:
    if GLOBAL_BATCH % micro_batch:
        raise B0Error("micro-batch must divide global batch")
    model = _new_model(device)
    optimizer = HybridMuon(model, PILOT_LR)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    durations = []
    losses = []
    clipped = []
    for update in range(UPDATES_PER_CANDIDATE):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        contexts = tape[update * GLOBAL_BATCH : (update + 1) * GLOBAL_BATCH]
        torch.cuda.synchronize(device)
        started = time.monotonic()
        ce_sum = 0.0
        for begin in range(0, GLOBAL_BATCH, micro_batch):
            x, y = _batch(train, contexts[begin : begin + micro_batch], device)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                _, loss, _ = model(x, y)
            if loss is None or not bool(torch.isfinite(loss).detach().cpu()):
                raise B0Error("nonfinite training CE")
            ce_sum += float(loss.detach().cpu())
            (loss * (micro_batch / GLOBAL_BATCH)).backward()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), PILOT_CLIP, foreach=True)
        if not bool(torch.isfinite(norm).detach().cpu()):
            raise B0Error("nonfinite gradient norm")
        clipped.append(float(norm.detach().cpu()) > PILOT_CLIP)
        optimizer.step()
        torch.cuda.synchronize(device)
        durations.append(time.monotonic() - started)
        losses.append(ce_sum * micro_batch / GLOBAL_BATCH)
    if not optimizer_state_fp32_and_finite(optimizer):
        raise B0Error("hybrid Muon/auxiliary AdamW state is not finite FP32")
    throughput = (UPDATES_PER_CANDIDATE - 1) * GLOBAL_BATCH * 256 / sum(durations[1:])
    return {
        "micro_batch_sequences": micro_batch,
        "global_batch_sequences": GLOBAL_BATCH,
        "completed_updates": UPDATES_PER_CANDIDATE,
        "training_tokens": UPDATES_PER_CANDIDATE * GLOBAL_BATCH * 256,
        "training_ce": losses,
        "gradient_clipping_fraction": float(np.mean(clipped)),
        "step_seconds": durations,
        "steady_tokens_per_second": throughput,
        "peak_allocated_gpu_bytes": torch.cuda.max_memory_allocated(device),
        "hybrid_muon_partition_sha256": optimizer.partition["sha256"],
        "hybrid_optimizer_state_fp32_and_finite": True,
    }, model


def run(plan_path: Path, output_dir: Path, input_dir: Path | None = None, input_manifest: Path | None = None) -> dict[str, Any]:
    root = Path(__file__).resolve().parents[2]
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    scale.validate_plan(plan, root, verify_data=input_dir is None)
    if plan["status"] != "offline_data_ready_b0_pending_launch_false":
        raise B0Error("6.5B corpus must be complete before B0")
    if input_dir is not None:
        if input_manifest is None:
            raise B0Error("packed B0 inputs require a manifest")
        train, tape, validation, probe = _packed_inputs(input_dir, input_manifest, plan)
    else:
        if input_manifest is not None:
            raise B0Error("input manifest without packed inputs")
        folder = root / str(plan["data"]["expanded_corpus"])
        train = np.memmap(folder / "pretrain_train.bin", dtype="<u2", mode="r")
        validation = np.memmap(folder / "pretrain_validation.bin", dtype="<u2", mode="r")
        probe = np.load(folder / "validation_probe.npy", allow_pickle=False)
        tape = np.load(folder / "training_tape.npy", mmap_mode="r", allow_pickle=False)
    device = _runtime()
    tried = []
    selected = None
    for micro_batch in MICRO_BATCH_CANDIDATES:
        try:
            result, model = _candidate(micro_batch, train, tape, device)
            torch.cuda.synchronize(device)
            started = time.monotonic()
            result["fixed_probe_validation_ce_after_4_updates"] = _validation_ce(
                model, validation, probe, device
            )
            torch.cuda.synchronize(device)
            result["fixed_probe_evaluation_seconds"] = time.monotonic() - started
            tried.append(result)
            selected = micro_batch
            break
        except torch.cuda.OutOfMemoryError as error:
            tried.append({"micro_batch_sequences": micro_batch, "status": "cuda_oom", "reason": str(error)[:160]})
            torch.cuda.empty_cache()
    if selected is None:
        raise B0Error("none of the micro-batch candidates fits H200")
    result = {
        "schema_version": "nanogpt300m_e2e_b0_result_v001",
        "campaign_id": plan["campaign_id"],
        "status": "muon_runtime_pilot_passed_nominal_lr_calibration_still_pending",
        "selected_micro_batch_sequences": selected,
        "candidates": tried,
        "optimizer": "hybrid_muon_from_initialization",
        "production_optimizer_settings_frozen": False,
        "formal_gpu_training_authorized": False,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    target = output_dir / "result.json"
    if target.exists() or target.is_symlink():
        raise B0Error("B0 result already exists")
    target.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    return result


def main() -> int:
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, default=root / "experiments/configs/nanogpt300m_e2e_6p5b_v001.json")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--input-dir", type=Path)
    parser.add_argument("--input-manifest", type=Path)
    args = parser.parse_args()
    print(json.dumps(run(args.plan, args.output_dir, args.input_dir, args.input_manifest), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
