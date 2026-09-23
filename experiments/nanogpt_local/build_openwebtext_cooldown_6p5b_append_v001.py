"""Extend the frozen 5B OpenWebText stream to 6.5B without repeating text.

The old train stream is copied byte-for-byte.  Its boundary document is
resumed at the exact token where the 5B build stopped; later documents are
read from contiguous, SHA-pinned shards.  Validation and its fixed probe are
reused byte-for-byte.  No upload or job submission happens here.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
from typing import Any, Mapping

import numpy as np

from . import build_openwebtext_cooldown_corpus_v001 as base


SCHEMA = "openwebtext_cooldown_6p5b_append_corpus_v001"
PLAN_SCHEMA = "openwebtext_cooldown_6p5b_expansion_source_plan_v001"
TRAIN_TOKENS = 6_500_057_344
EXECUTED_TOKENS = 6_500_057_088
PARAMETERS = 303_473_664
TAPE_CONTEXTS = EXECUTED_TOKENS // base.MODEL_CONTEXT_TOKENS
TAPE_SEED = 2_026_091_501
BASE_TOKENS = 5_000_007_936
BASE_TRAIN_SHA256 = "b5b3bf08ae0ffecdeda46c28273af995d4c5af6934090eb9904f51bf6c17f84b"
VALIDATION_SHA256 = "52085210f2da47797efd93401d8b890c6efdb41055c266d499649e1d82ee20be"
PROBE_SHA256 = "0295b25ce763f30570d4a268491d5a829e244ade33c602c32423e66aea278155"
REVISION = "79d93d786212f7344586290adb811d4ae6a1762c"


class ExpansionError(RuntimeError):
    """A pinned source or corpus-continuity contract failed."""


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        value = json.load(stream)
    if type(value) is not dict:
        raise ExpansionError("source plan must be a JSON object")
    return value


def _sha256_file(path: Path) -> str:
    return base._sha256_file(path)


def _verified(path: Path, size: int, digest: str, label: str) -> Path:
    if path.is_symlink() or not path.is_file():
        raise ExpansionError(f"{label} is missing or linked: {path}")
    if path.stat().st_size != size or _sha256_file(path) != digest:
        raise ExpansionError(f"{label} size or SHA-256 changed: {path}")
    return path


def validate_plan(plan: Mapping[str, Any]) -> dict[str, Any]:
    base_record = plan.get("base_corpus", {})
    target = plan.get("target", {})
    dataset = plan.get("dataset", {})
    boundary = base_record.get("boundary_document", {})
    boundary_source = base_record.get("boundary_source", {})
    sources = plan.get("sources")
    if (
        plan.get("schema_version") != PLAN_SCHEMA
        or plan.get("status") != "planned_unmaterialized"
        or dataset != {
            "id": base.DATASET_ID,
            "configuration": "plain_text",
            "split": "train",
            "revision": REVISION,
        }
        or base_record.get("train_tokens") != BASE_TOKENS
        or base_record.get("train_sha256") != BASE_TRAIN_SHA256
        or base_record.get("validation_sha256") != VALIDATION_SHA256
        or base_record.get("validation_probe_sha256") != PROBE_SHA256
        or boundary.get("source_shard_index") != 45
        or boundary.get("source_row") != 83_359
        or boundary.get("document_sha256")
        != "5e5df06f2db7a116d216aa7ec9691af5c18d0c7d1c50e7c54bbe1285dc8c258d"
        or boundary.get("full_tokens_including_eot") != 1_980
        or boundary.get("written_prefix_tokens") != 1_161
        or boundary_source.get("bytes") != 303_334_655
        or boundary_source.get("sha256")
        != "c34b7d448fd193eb2d9c8a6fbe4f775ef6cb65b4510b778007c8151a26b7f5a6"
        or target.get("stored_train_tokens") != TRAIN_TOKENS
        or target.get("executed_train_tokens") != EXECUTED_TOKENS
        or target.get("tokens_per_parameter_lower_bound") != 20
        or not isinstance(sources, list)
        or len(sources) != 17
    ):
        raise ExpansionError("6.5B source-plan identity changed")
    indices = []
    for record in sources:
        if (
            type(record) is not list
            or len(record) != 3
            or type(record[0]) is not int
            or type(record[1]) is not int
            or record[1] <= 0
            or type(record[2]) is not str
            or len(record[2]) != 64
            or any(character not in "0123456789abcdef" for character in record[2])
        ):
            raise ExpansionError("a shard binding is malformed")
        indices.append(record[0])
    if indices != list(range(46, 63)) or EXECUTED_TOKENS <= 20 * PARAMETERS:
        raise ExpansionError("source order or token/parameter floor changed")
    return {
        "source_indices": indices,
        "stored_train_tokens": TRAIN_TOKENS,
        "executed_train_tokens": EXECUTED_TOKENS,
        "parameter_count": PARAMETERS,
        "tokens_per_parameter": EXECUTED_TOKENS / PARAMETERS,
        "training_contexts": TAPE_CONTEXTS,
    }


def _copy(source: Path, target: Path) -> None:
    with source.open("rb") as reader, target.open("xb") as writer:
        shutil.copyfileobj(reader, writer, length=16 * 1024 * 1024)
        writer.flush()
        os.fsync(writer.fileno())


def _spec(path: Path, index: int, size: int, digest: str) -> base.ShardSpec:
    try:
        import pyarrow.parquet as parquet
    except ImportError as error:
        raise ExpansionError("repository environment requires pyarrow") from error
    artifact = parquet.ParquetFile(path)
    return base.ShardSpec(
        index=index,
        relative_path=str(path),
        remote_relative_path=f"plain_text/train-{index:05d}-of-00080.parquet",
        bytes=size,
        sha256=digest,
        xet_hash="",
        rows=int(artifact.metadata.num_rows),
        row_groups=int(artifact.metadata.num_row_groups),
    )


def _append_text(
    writer: Any, tokenizer: Any, text: str, skip_tokens: int, remaining: int
) -> tuple[int, int]:
    values = np.asarray([*tokenizer.encode_ordinary(text), base.EOT_TOKEN], dtype="<u2")
    if not 0 <= skip_tokens < len(values):
        raise ExpansionError("boundary token offset is outside its document")
    take = min(len(values) - skip_tokens, remaining)
    values[skip_tokens : skip_tokens + take].tofile(writer)
    return len(values), take


def materialize(plan_path: Path, project_root: Path, output_dir: Path) -> dict[str, Any]:
    plan = _load_json(plan_path)
    report = validate_plan(plan)
    root = project_root.resolve()
    if output_dir.exists() and output_dir.is_symlink():
        raise ExpansionError("output directory must not be a symlink")
    output = output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    names = {
        "train": "pretrain_train.bin",
        "validation": "pretrain_validation.bin",
        "probe": "validation_probe.npy",
        "metadata": "metadata.json",
    }
    finals = {key: output / name for key, name in names.items()}
    parts = {key: path.with_name(path.name + ".part") for key, path in finals.items()}
    if any(path.exists() or path.is_symlink() for path in (*finals.values(), *parts.values())):
        raise ExpansionError("final or partial 6.5B corpus already exists")
    base_dir = root / str(plan["base_corpus"]["path"])
    old_train = _verified(base_dir / names["train"], 2 * BASE_TOKENS, BASE_TRAIN_SHA256, "5B train")
    old_validation = _verified(
        base_dir / names["validation"], 2 * base.VALIDATION_TOKEN_TARGET,
        VALIDATION_SHA256, "5B validation",
    )
    old_probe = _verified(base_dir / names["probe"], 8_320, PROBE_SHA256, "5B probe")
    boundary = plan["base_corpus"]["boundary_document"]
    source = plan["base_corpus"]["boundary_source"]
    boundary_path = _verified(
        root / str(source["relative_path"]), int(source["bytes"]),
        str(source["sha256"]), "shard 45",
    )
    tokenizer = base._local_gpt2_encoding(root)
    written = BASE_TOKENS
    rows_scanned = 0
    used_sources: list[dict[str, Any]] = []
    endpoint: dict[str, Any] | None = None
    try:
        _copy(old_train, parts["train"])
        with parts["train"].open("ab") as writer:
            records = [
                (45, int(source["bytes"]), str(source["sha256"]), boundary_path)
            ] + [
                (
                    int(index), int(size), str(digest),
                    root / "data/openwebtext-cooldown-6p5b-source-v001/source/plain_text"
                    / f"train-{int(index):05d}-of-00080.parquet",
                )
                for index, size, digest in plan["sources"]
            ]
            for index, size, digest, unverified_path in records:
                path = _verified(unverified_path, size, digest, f"shard {index}")
                spec = _spec(path, index, size, digest)
                source_written = 0
                for row, text in base._parquet_documents(path, spec):
                    if index == 45 and row < int(boundary["source_row"]):
                        continue
                    rows_scanned += 1
                    identity = base.document_digest(text)
                    if base.assign_split(identity) != "train":
                        if index == 45 and row == int(boundary["source_row"]):
                            raise ExpansionError("5B boundary document changed split")
                        continue
                    skip = int(boundary["written_prefix_tokens"]) if index == 45 and row == int(boundary["source_row"]) else 0
                    full, take = _append_text(writer, tokenizer, text, skip, TRAIN_TOKENS - written)
                    if skip and (identity.hex() != boundary["document_sha256"] or full != boundary["full_tokens_including_eot"]):
                        raise ExpansionError("5B boundary document changed")
                    written += take
                    source_written += take
                    if take < full - skip:
                        endpoint = {
                            "source_shard_index": index,
                            "source_row": row,
                            "document_sha256": identity.hex(),
                            "full_tokens_including_eot": full,
                            "written_prefix_tokens": skip + take,
                        }
                    if written == TRAIN_TOKENS:
                        break
                used_sources.append({"index": index, "sha256": digest, "training_tokens_written": source_written})
                print(json.dumps({"event": "6p5b_append_shard_complete", "index": index, "train_tokens": written}), flush=True)
                if written == TRAIN_TOKENS:
                    break
            writer.flush()
            os.fsync(writer.fileno())
        if written != TRAIN_TOKENS:
            raise ExpansionError(f"pinned source prefix provided only {written} train tokens")
        _copy(old_validation, parts["validation"])
        _copy(old_probe, parts["probe"])
        artifacts = {
            name: {"bytes": parts[key].stat().st_size, "sha256": _sha256_file(parts[key])}
            for key, name in names.items() if key != "metadata"
        }
        if (
            artifacts[names["train"]]["bytes"] != 2 * TRAIN_TOKENS
            or artifacts[names["validation"]]["sha256"] != VALIDATION_SHA256
            or artifacts[names["probe"]]["sha256"] != PROBE_SHA256
        ):
            raise ExpansionError("final corpus integrity check failed")
        metadata = {
            "schema_version": SCHEMA,
            "completed": True,
            "dataset": plan["dataset"],
            "construction": {
                "method": "copy_frozen_5b_train_resume_boundary_then_append_new_train_documents",
                "base_train_tokens": BASE_TOKENS,
                "target_train_tokens": TRAIN_TOKENS,
                "validation_and_probe_reused_byte_for_byte": True,
                "training_contexts_without_replacement": TAPE_CONTEXTS,
            },
            "scan": {"rows_scanned": rows_scanned, "used_sources": used_sources, "boundary_document": endpoint},
            "source_plan": {"path": plan_path.name, **report},
            "artifacts": artifacts,
        }
        with parts["metadata"].open("xb") as writer:
            writer.write((json.dumps(metadata, sort_keys=True, indent=2, allow_nan=False) + "\n").encode("utf-8"))
            writer.flush()
            os.fsync(writer.fileno())
        for key in names:
            os.replace(parts[key], finals[key])
        return metadata
    except BaseException:
        for path in parts.values():
            try:
                path.unlink()
            except FileNotFoundError:
                pass
        raise


def materialize_tape(path: Path) -> dict[str, Any]:
    path = path.resolve()
    part = path.with_name(path.name + ".part")
    if any(value.exists() or value.is_symlink() for value in (path, part)):
        raise ExpansionError("6.5B tape already exists")
    eligible = (TRAIN_TOKENS - base.PROBE_WINDOW_TOKENS) // base.MODEL_CONTEXT_TOKENS + 1
    if eligible != TAPE_CONTEXTS:
        raise ExpansionError("corpus/tape integer geometry changed")
    values = np.random.default_rng(TAPE_SEED).permutation(eligible).astype("<i8", copy=False)
    values *= base.MODEL_CONTEXT_TOKENS
    path.parent.mkdir(parents=True, exist_ok=True)
    with part.open("xb") as writer:
        np.save(writer, values, allow_pickle=False)
        writer.flush()
        os.fsync(writer.fileno())
    checked = np.load(part, mmap_mode="r", allow_pickle=False)
    if (
        checked.shape != (TAPE_CONTEXTS,)
        or checked.dtype.str != "<i8"
        or int(np.min(checked)) != 0
        or int(np.max(checked)) + base.PROBE_WINDOW_TOKENS > TRAIN_TOKENS
        or len(np.unique(checked)) != TAPE_CONTEXTS
    ):
        part.unlink()
        raise ExpansionError("6.5B tape failed permutation validation")
    file_digest = _sha256_file(part)
    offsets_digest = hashlib.sha256(np.ascontiguousarray(checked, dtype="<i8").tobytes()).hexdigest()
    os.replace(part, path)
    return {
        "context_count": TAPE_CONTEXTS,
        "seed": TAPE_SEED,
        "size_bytes": path.stat().st_size,
        "sha256": file_digest,
        "offsets_sha256": offsets_digest,
    }


def main() -> int:
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, default=root / "experiments/configs/openwebtext_cooldown_6p5b_expansion_source_plan_v001.json")
    parser.add_argument("--project-root", type=Path, default=root)
    parser.add_argument("--output-dir", type=Path, default=root / "data/openwebtext-cooldown-6p5b-v001")
    action = parser.add_mutually_exclusive_group()
    action.add_argument("--validate-contract", action="store_true")
    action.add_argument("--materialize-tape", action="store_true")
    args = parser.parse_args()
    plan = _load_json(args.plan)
    if args.validate_contract:
        result = validate_plan(plan)
    elif args.materialize_tape:
        validate_plan(plan)
        result = materialize_tape(args.output_dir / "training_tape.npy")
    else:
        result = materialize(args.plan, args.project_root, args.output_dir)
    print(json.dumps(result, sort_keys=True, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
