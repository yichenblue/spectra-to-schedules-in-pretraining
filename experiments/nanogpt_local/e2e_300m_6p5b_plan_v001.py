"""Check the launch-false 300M/6.5B model and corpus token ledger."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from . import build_openwebtext_cooldown_6p5b_append_v001 as corpus
from . import muon_300m_v001 as muon


SCHEMA = "nanogpt300m_e2e_6p5b_plan_v001"
CAMPAIGN = "nanogpt300m-e2e-6p5b-v001"
PREFIX_VALIDATION_SAMPLES = 6_000
CHECKPOINT_PLOT_WINDOW_UPDATES = 4_096
PREFIX_UPDATES = 79_346


class PlanError(RuntimeError):
    """The 300M model, data, or horizon is internally inconsistent."""


def prefix_validation_updates(tail_regular_stride_macros: int) -> tuple[int, ...]:
    """Exactly 6,000 ticks, with the pre-fork window matched to tail ΔT density.

    On the constant-r segment, one prefix update and one tail macro each add
    the same nominal intrinsic-time increment. Freeze the stride after the
    H200 pilot.
    """
    if type(tail_regular_stride_macros) is not int or tail_regular_stride_macros < 1:
        raise PlanError("tail regular validation stride must be a positive integer")
    stride = tail_regular_stride_macros
    near = tuple(
        range(
            PREFIX_UPDATES - (CHECKPOINT_PLOT_WINDOW_UPDATES // stride) * stride,
            PREFIX_UPDATES + 1,
            stride,
        )
    )
    early_count = PREFIX_VALIDATION_SAMPLES - len(near)
    if early_count < 2:
        raise PlanError("tail stride leaves too few prefix validation ticks")
    early_end = near[0] - 1
    early = tuple(
        (index * early_end + (early_count - 1) // 2) // (early_count - 1)
        for index in range(early_count)
    )
    ticks = early + near
    if len(ticks) != PREFIX_VALIDATION_SAMPLES or len(set(ticks)) != len(ticks):
        raise PlanError("prefix validation grid is not exactly 6,000 unique ticks")
    return ticks


def tail_regular_validation_macros(macro_count: int, stride: int) -> tuple[int, ...]:
    """Regular nominal-intrinsic-time grid, including the fork anchor and end."""
    if type(macro_count) is not int or macro_count < 0:
        raise PlanError("tail macro count must be nonnegative")
    if type(stride) is not int or stride < 1:
        raise PlanError("tail regular validation stride must be a positive integer")
    return tuple(sorted(set(range(0, macro_count + 1, stride)) | {macro_count}))


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_plan(plan: Mapping[str, Any], root: Path, verify_data: bool = False) -> dict[str, Any]:
    model = plan.get("model", {})
    horizon = plan.get("horizon", {})
    data = plan.get("data", {})
    runtime = plan.get("runtime", {})
    optimizer = plan.get("optimizer", {})
    if (
        plan.get("schema_version") != SCHEMA
        or plan.get("campaign_id") != CAMPAIGN
        or plan.get("status") not in {
            "corpus_materializing_b0_pending_launch_false",
            "offline_data_ready_b0_pending_launch_false",
        }
        or plan.get("launch_control") != {"cluster_submission_authorized": False}
        or plan.get("evidence") != {
            "primary_prediction_id": "A2_EXTERNAL",
            "theorem_facing": False,
        }
    ):
        raise PlanError("identity or launch-false classification changed")
    required_model = {
        "family": "nanogpt",
        "n_layer": 20,
        "n_head": 16,
        "n_embd": 1024,
        "block_size": 256,
        "vocab_size": 50304,
        "bias": False,
        "dropout": 0.0,
        "tied_token_embedding_and_output": True,
        "expected_parameter_count": corpus.PARAMETERS,
    }
    if model != required_model:
        raise PlanError("300M architecture changed")
    width = model["n_embd"]
    actual = (
        model["vocab_size"] * width
        + model["block_size"] * width
        + model["n_layer"] * (12 * width * width + 2 * width)
        + width
    )
    if actual != corpus.PARAMETERS:
        raise PlanError("analytic parameter count changed")
    updates = horizon.get("canonical_updates")
    prefix = horizon.get("shared_prefix_updates")
    tail = horizon.get("tail_source_updates")
    batch = horizon.get("global_batch_sequences")
    if (
        updates != 99_183
        or prefix != 79_346
        or tail != updates - prefix
        or batch != 256
        or horizon.get("tokens_per_canonical_update") != batch * model["block_size"]
        or horizon.get("executed_tokens_per_trajectory")
        != updates * batch * model["block_size"]
        or horizon.get("executed_tokens_per_trajectory") != corpus.EXECUTED_TOKENS
        or horizon.get("stored_train_tokens") != corpus.TRAIN_TOKENS
        or horizon.get("shared_prefix_tokens") != prefix * batch * model["block_size"]
        or horizon.get("tail_tokens_per_arm") != tail * batch * model["block_size"]
        or abs(horizon.get("tokens_per_parameter", 0) - corpus.EXECUTED_TOKENS / actual)
        > 1e-12
        or corpus.EXECUTED_TOKENS <= 20 * actual
        or horizon["stored_train_tokens"] - horizon["executed_tokens_per_trajectory"]
        != 256
    ):
        raise PlanError("executed-token ledger or >20 token/parameter gate failed")
    expected_optimizer = muon.optimizer_manifest(None, None)
    expected_optimizer["calibration_status"] = "pending_300m_muon_b0"
    experiment = plan.get("experiment", {})
    tail_stride = experiment.get("tail_regular_validation_stride_macros")
    if (
        runtime.get("short_300m_h200_calibration_required") is not True
        or runtime.get("micro_batch_sequences") is not None
        or optimizer != expected_optimizer
        or experiment.get("reuse_124m_checkpoint") is not False
        or experiment.get("reuse_adamw_prefix_checkpoint") is not False
        or experiment.get("muon_from_initialization_to_end") is not True
        or experiment.get("shared_prefix_optimizer") != "hybrid_muon"
        or experiment.get("resume_same_hybrid_muon_state_in_all_tails") is not True
        or experiment.get("primary_metric") != "fixed_validation_probe_cross_entropy"
        or experiment.get("fixed_validation_probe_contexts") != 1_024
        or experiment.get("record_per_update_training_ce") is not True
        or experiment.get("prefix_validation_samples") != PREFIX_VALIDATION_SAMPLES
        or experiment.get("prefix_checkpoint_plot_window_updates") != CHECKPOINT_PLOT_WINDOW_UPDATES
        or experiment.get("prefix_near_checkpoint_stride_rule") != "match_tail_regular_nominal_T_stride"
        or experiment.get("dense_tail_validation_at_schedule_knots") is not True
    ):
        raise PlanError("Muon prefix/state or validation-grid design changed")
    # A null stride denotes an unfrozen H200 pilot choice; a frozen integer
    # must generate the exact paired grid before either run is launched.
    prefix_validation_updates(1 if tail_stride is None else tail_stride)
    source_path = root / str(data.get("source_plan"))
    if (
        not source_path.is_file()
        or source_path.is_symlink()
        or _file_sha256(source_path) != data.get("source_plan_sha256")
    ):
        raise PlanError("pinned 6.5B source plan changed")
    corpus.validate_plan(json.loads(source_path.read_text(encoding="utf-8")))
    if data.get("validation_sha256") != corpus.VALIDATION_SHA256 or data.get("validation_probe_sha256") != corpus.PROBE_SHA256:
        raise PlanError("fixed validation identities changed")
    checked_data = False
    if verify_data:
        folder = root / str(data.get("expanded_corpus"))
        train = folder / "pretrain_train.bin"
        tape = folder / "training_tape.npy"
        validation = folder / "pretrain_validation.bin"
        probe = folder / "validation_probe.npy"
        metadata = folder / "metadata.json"
        values = np.load(tape, mmap_mode="r", allow_pickle=False) if tape.is_file() else None
        if (
            not all(path.is_file() and not path.is_symlink() for path in (train, tape, validation, probe, metadata))
            or train.stat().st_size != 2 * corpus.TRAIN_TOKENS
            or _file_sha256(train) != data.get("train_sha256")
            or _file_sha256(tape) != data.get("training_tape_file_sha256")
            or _file_sha256(validation) != data.get("validation_sha256")
            or _file_sha256(probe) != data.get("validation_probe_sha256")
            or values is None
            or values.shape != (corpus.TAPE_CONTEXTS,)
            or values.dtype.str != "<i8"
            or hashlib.sha256(np.ascontiguousarray(values, dtype="<i8").tobytes()).hexdigest()
            != data.get("training_tape_offsets_sha256")
            or json.loads(metadata.read_text(encoding="utf-8")).get("completed") is not True
        ):
            raise PlanError("materialized corpus/tape identity failed")
        checked_data = True
    return {
        "campaign_id": CAMPAIGN,
        "status": "passed_data_verified" if checked_data else "passed_offline_plan",
        "parameter_count": actual,
        "executed_tokens_per_trajectory": corpus.EXECUTED_TOKENS,
        "tokens_per_parameter": corpus.EXECUTED_TOKENS / actual,
        "shared_prefix_updates": prefix,
        "prefix_validation_samples": PREFIX_VALIDATION_SAMPLES,
        "prefix_checkpoint_plot_window_updates": CHECKPOINT_PLOT_WINDOW_UPDATES,
        "tail_source_updates": tail,
        "b0_required_before_gpu_training": True,
        "cluster_submission_authorized": False,
    }


def main() -> int:
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, default=root / "experiments/configs/nanogpt300m_e2e_6p5b_v001.json")
    parser.add_argument("--verify-data", action="store_true")
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text(encoding="utf-8"))
    print(json.dumps(validate_plan(plan, root, args.verify_data), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
