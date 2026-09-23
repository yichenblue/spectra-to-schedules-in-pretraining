#!/usr/bin/env python3
"""Build the frozen local OpenWebText corpus for 30M cooldown calibration.

This is a deliberately single-purpose, offline data builder.  It reads the
seven already-materialized, hash-pinned OpenWebText parquet shards from a
separate project tree, tokenizes one document at a time with local pinned GPT-2
assets, and publishes exact-size train/validation uint16 streams plus fixed
validation probe offsets.  It never downloads data, imports a model, trains,
submits a job, or writes into the source project tree.

Run it from the repository root with that repository's ``./.venv/bin/python``.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import re
from typing import Any, Iterator, Sequence
import unicodedata

import numpy as np


SCHEMA_VERSION = "nanogpt30m_openwebtext_cooldown_corpus_v001"
DATASET_ID = "Skylion007/openwebtext"
DATASET_REVISION = "79d93d786212f7344586290adb811d4ae6a1762c"
OUTPUT_RELATIVE = "data/openwebtext-cooldown-700m-v001"

TRAIN_TOKEN_TARGET = 700_000_000
VALIDATION_TOKEN_TARGET = 16_777_216
VALIDATION_PROBE_DOCUMENTS = 1_024
MODEL_CONTEXT_TOKENS = 256
# A probe start must provide both a 256-token input and its next-token targets.
PROBE_WINDOW_TOKENS = MODEL_CONTEXT_TOKENS + 1

SPLIT_DOMAIN = b"nanogpt30m-openwebtext-cooldown-split-v001\0"
PROBE_OFFSET_DOMAIN = b"nanogpt30m-openwebtext-cooldown-probe-offset-v001\0"
VALIDATION_MODULUS = 32
VALIDATION_RESIDUE = 0
PARQUET_BATCH_SIZE = 512
EOT_TOKEN = 50_256
VOCAB_SIZE = 50_257

TOKENIZER_ASSETS = (
    {
        "role": "gpt2_encoder_json",
        "relative_path": (
            "experiments/chtc/nanogpt124m_e2e_memory_h100_scale_pilot_v001/"
            "tokenizer_assets/encoder.json"
        ),
        "bytes": 1_042_301,
        "sha256": "196139668be63f3b5d6574427317ae82f612a97c5d1cdaf36ed2256dbf636783",
    },
    {
        "role": "gpt2_vocab_bpe",
        "relative_path": (
            "experiments/chtc/nanogpt124m_e2e_memory_h100_scale_pilot_v001/"
            "tokenizer_assets/vocab.bpe"
        ),
        "bytes": 456_318,
        "sha256": "1ce1664773c50f3e0cc8842619a93edc4624525b728b188a9e0be33b7726adc5",
    },
)


@dataclass(frozen=True)
class ShardSpec:
    index: int
    relative_path: str
    remote_relative_path: str
    bytes: int
    sha256: str
    xet_hash: str
    rows: int = 100_173
    row_groups: int = 101


SOURCE_SHARDS = (
    ShardSpec(
        0,
        "data/openwebtext-p2p1-fresh-v001/source/openwebtext-train-00000-of-00080.parquet",
        "plain_text/train-00000-of-00080.parquet",
        302_848_326,
        "caed9f4b7053d7cd4d1a13ce9ec9224d84a3bba1f11579193562a7e31ebe656e",
        "32c33412107782cc1f8eba9fdeddba2ca2924532272bafeb2c9aa1a6fe6f4a9c",
    ),
    ShardSpec(
        1,
        "data/openwebtext-p2p1-fresh-v001/source/openwebtext-train-00001-of-00080.parquet",
        "plain_text/train-00001-of-00080.parquet",
        305_678_548,
        "70307b7749e10cb0bfb895fde024dd82d96e6bb2710bca1886b952abe45bf3ea",
        "0b4af751e5851622bc3994fdf900f382b3a9ad0b14d33bb3913585352ca21e54",
    ),
    ShardSpec(
        2,
        "data/openwebtext-p22a-im-fresh-v001/source/openwebtext-train-00002-of-00080.parquet",
        "plain_text/train-00002-of-00080.parquet",
        304_419_240,
        "0d02a67f73464d383c8df2059f27b36b4b1652c855051be3c65930baef12bd2d",
        "c20ce7ccb7d8d1bb39979d2b06a30837beaecb6a9692fd0e3f387f6abe94a4fe",
    ),
    ShardSpec(
        3,
        "data/openwebtext-p22a-im-fresh-v001/source/openwebtext-train-00003-of-00080.parquet",
        "plain_text/train-00003-of-00080.parquet",
        304_332_042,
        "437833e29326c3d88792e834d36368d2c4836e5c5f3535233a88f3cf7dd46aad",
        "14f133f516ccf56991f363bdf74035cbbcd4806ccbc893764ea9cc0c69f1f838",
    ),
    ShardSpec(
        4,
        "data/openwebtext-p22a-im-fresh-v001/source/openwebtext-train-00004-of-00080.parquet",
        "plain_text/train-00004-of-00080.parquet",
        301_175_397,
        "7b3cef26e2d4e2b041e6e13a0102cb4d03277e8c6418b8ff1b865544a4695de3",
        "ee0d172a006be8c8163fd9f903eeca6e2c845f2d95df7f66ce48cda1c83c27e2",
    ),
    ShardSpec(
        5,
        "data/openwebtext-p22a-im-fresh-v001/source/openwebtext-train-00005-of-00080.parquet",
        "plain_text/train-00005-of-00080.parquet",
        301_755_705,
        "2d3200a936b021c9abafdda6714e4d678387e8854e56abaac5d2b62882fc6044",
        "3dd0367355e90ae9ef4a28dee5eee42cc4e87213dbe5560b3fe0d49a4b64aa01",
    ),
    ShardSpec(
        6,
        "data/openwebtext-p22a-im-fresh-v001/source/openwebtext-train-00006-of-00080.parquet",
        "plain_text/train-00006-of-00080.parquet",
        303_536_715,
        "76c0b96ce0250c3a32f2e8878172a4b79ac7970fede78edfcfc4d98d87ec5a6c",
        "f94cf6bba5c7abfc4946b4aba53c81b881202ab7d414594c3b4234045b960976",
    ),
)


class CorpusBuildError(RuntimeError):
    """A fail-closed corpus materialization error."""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _normalise_text(text: str) -> str:
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", str(text))).strip()


def document_digest(text: str) -> bytes:
    """Return the frozen content identity used by the split rule."""

    return hashlib.sha256(_normalise_text(text).encode("utf-8")).digest()


def assign_split(text_or_digest: str | bytes) -> str:
    """Assign a normalized document identity to one deterministic split."""

    digest = (
        document_digest(text_or_digest)
        if isinstance(text_or_digest, str)
        else bytes(text_or_digest)
    )
    if len(digest) != 32:
        raise ValueError("document digest must contain exactly 32 bytes")
    value = int.from_bytes(hashlib.sha256(SPLIT_DOMAIN + digest).digest()[:8], "big")
    return "validation" if value % VALIDATION_MODULUS == VALIDATION_RESIDUE else "train"


def fixed_probe_offset(document_hash: bytes, token_count: int, window_tokens: int) -> int:
    """Choose one reproducible complete-window start inside a document."""

    digest = bytes(document_hash)
    if len(digest) != 32:
        raise ValueError("document hash must contain exactly 32 bytes")
    choices = int(token_count) - int(window_tokens) + 1
    if choices <= 0:
        raise ValueError("document is too short for a complete probe window")
    value = int.from_bytes(
        hashlib.sha256(PROBE_OFFSET_DOMAIN + digest).digest()[:8], "big"
    )
    return value % choices


def _resolve_regular_file(root: Path, relative: str, label: str) -> Path:
    value = Path(relative)
    if value.is_absolute() or ".." in value.parts:
        raise CorpusBuildError(f"{label} must be a source-project-relative path")
    path = root.joinpath(value)
    if path.is_symlink() or not path.is_file():
        raise CorpusBuildError(f"pinned {label} is missing or is not a regular file: {path}")
    return path


def _verify_bound_file(
    root: Path, relative: str, expected_bytes: int, expected_sha256: str, label: str
) -> Path:
    path = _resolve_regular_file(root, relative, label)
    observed_bytes = path.stat().st_size
    if observed_bytes != int(expected_bytes):
        raise CorpusBuildError(
            f"pinned {label} size changed: expected {expected_bytes}, observed {observed_bytes}"
        )
    observed_sha256 = _sha256_file(path)
    if observed_sha256 != expected_sha256:
        raise CorpusBuildError(
            f"pinned {label} SHA-256 changed: expected {expected_sha256}, "
            f"observed {observed_sha256}"
        )
    return path


def verify_sources(
    source_project_root: Path, shard_specs: Sequence[ShardSpec] = SOURCE_SHARDS
) -> list[tuple[ShardSpec, Path]]:
    """Verify every parquet binding before reading any document content."""

    root = source_project_root.resolve()
    expected_indices = list(range(len(shard_specs)))
    observed_indices = [int(spec.index) for spec in shard_specs]
    if observed_indices != expected_indices:
        raise CorpusBuildError(
            f"source shard order must be contiguous from zero, found {observed_indices}"
        )
    result: list[tuple[ShardSpec, Path]] = []
    for spec in shard_specs:
        path = _verify_bound_file(
            root,
            spec.relative_path,
            spec.bytes,
            spec.sha256,
            f"OpenWebText shard {spec.index}",
        )
        result.append((spec, path))
    return result


def _verify_tokenizer_assets(source_project_root: Path) -> dict[str, Path]:
    root = source_project_root.resolve()
    paths: dict[str, Path] = {}
    for record in TOKENIZER_ASSETS:
        paths[str(record["role"])] = _verify_bound_file(
            root,
            str(record["relative_path"]),
            int(record["bytes"]),
            str(record["sha256"]),
            str(record["role"]),
        )
    return paths


def _local_gpt2_encoding(source_project_root: Path) -> Any:
    """Construct GPT-2 BPE only from the two pinned local assets."""

    assets = _verify_tokenizer_assets(source_project_root)
    try:
        import tiktoken
        from tiktoken.load import data_gym_to_mergeable_bpe_ranks
        from tiktoken_ext.openai_public import ENDOFTEXT, r50k_pat_str
    except ImportError as error:
        raise CorpusBuildError("the repository environment must provide tiktoken") from error
    version = str(getattr(tiktoken, "__version__", ""))
    if version != "0.14.0":
        raise CorpusBuildError(f"requires pinned local tiktoken==0.14.0, found {version!r}")
    ranks = data_gym_to_mergeable_bpe_ranks(
        vocab_bpe_file=str(assets["gpt2_vocab_bpe"]),
        encoder_json_file=str(assets["gpt2_encoder_json"]),
        vocab_bpe_hash=str(TOKENIZER_ASSETS[1]["sha256"]),
        encoder_json_hash=str(TOKENIZER_ASSETS[0]["sha256"]),
    )
    encoding = tiktoken.Encoding(
        name="nanogpt30m-openwebtext-cooldown-local-gpt2-v001",
        explicit_n_vocab=VOCAB_SIZE,
        pat_str=r50k_pat_str,
        mergeable_ranks=ranks,
        special_tokens={ENDOFTEXT: EOT_TOKEN},
    )
    if int(encoding.eot_token) != EOT_TOKEN or int(encoding.n_vocab) != VOCAB_SIZE:
        raise CorpusBuildError("local GPT-2 tokenizer vocabulary contract changed")
    return encoding


def _parquet_documents(path: Path, spec: ShardSpec) -> Iterator[tuple[int, str]]:
    """Yield text one document at a time from a structurally pinned parquet."""

    try:
        import pyarrow.parquet as parquet
    except ImportError as error:
        raise CorpusBuildError("the repository environment must provide pyarrow") from error
    artifact = parquet.ParquetFile(path)
    metadata = artifact.metadata
    schema = artifact.schema_arrow
    if metadata.num_rows != int(spec.rows):
        raise CorpusBuildError(f"source parquet row count changed: {path}")
    if metadata.num_row_groups != int(spec.row_groups):
        raise CorpusBuildError(f"source parquet row-group count changed: {path}")
    if schema.names != ["text"] or str(schema.field("text").type) != "string":
        raise CorpusBuildError(f"source parquet text schema changed: {path}")

    source_row = 0
    for batch in artifact.iter_batches(batch_size=PARQUET_BATCH_SIZE, columns=["text"]):
        column = batch.column(batch.schema.get_field_index("text"))
        if column.null_count:
            raise CorpusBuildError(f"source parquet contains a null text value: {path}")
        for value in column.to_pylist():
            yield source_row, str(value)
            source_row += 1
    if source_row != int(spec.rows):
        raise CorpusBuildError(f"source parquet streaming row count changed: {path}")


def _ordered_document_update(
    digest: Any,
    *,
    shard_index: int,
    source_row: int,
    document_hash: bytes,
    full_tokens: int,
    written_tokens: int,
) -> None:
    digest.update(int(shard_index).to_bytes(2, "little", signed=False))
    digest.update(int(source_row).to_bytes(8, "little", signed=False))
    digest.update(document_hash)
    digest.update(int(full_tokens).to_bytes(8, "little", signed=False))
    digest.update(int(written_tokens).to_bytes(8, "little", signed=False))


def _artifact_record(path: Path) -> dict[str, Any]:
    return {
        "path": path.name,
        "bytes": path.stat().st_size,
        "sha256": _sha256_file(path),
    }


def _part_path(path: Path) -> Path:
    return path.with_name(path.name + ".part")


def _write_json_part(path: Path, value: Any) -> None:
    payload = json.dumps(
        value, sort_keys=True, indent=2, ensure_ascii=True, allow_nan=False
    ).encode("utf-8") + b"\n"
    with path.open("xb") as handle:
        handle.write(payload)


def _require_output_location(source_root: Path, output_dir: Path) -> None:
    source = source_root.resolve()
    output = output_dir.resolve()
    if output == source or source in output.parents:
        raise CorpusBuildError("output directory must not be inside the read-only source tree")


def build_corpus(
    source_project_root: str | Path,
    output_dir: str | Path,
    *,
    force: bool = False,
    shard_specs: Sequence[ShardSpec] = SOURCE_SHARDS,
    encoding: Any | None = None,
    train_token_target: int = TRAIN_TOKEN_TARGET,
    validation_token_target: int = VALIDATION_TOKEN_TARGET,
    probe_documents: int = VALIDATION_PROBE_DOCUMENTS,
    probe_window_tokens: int = PROBE_WINDOW_TOKENS,
) -> dict[str, Any]:
    """Materialize the frozen corpus; small overrides exist only for unit tests."""

    source_root = Path(source_project_root).resolve()
    destination = Path(output_dir).resolve()
    _require_output_location(source_root, destination)
    if train_token_target <= 0 or validation_token_target <= 0:
        raise ValueError("train and validation token targets must be positive")
    if probe_documents <= 0 or probe_window_tokens <= 1:
        raise ValueError("probe dimensions must be positive")

    verified_shards = verify_sources(source_root, shard_specs)
    tokenizer_assets = _verify_tokenizer_assets(source_root) if encoding is None else None
    tokenizer = _local_gpt2_encoding(source_root) if encoding is None else encoding
    if int(tokenizer.eot_token) != EOT_TOKEN:
        raise CorpusBuildError("tokenizer EOT token differs from the frozen GPT-2 contract")

    destination.mkdir(parents=True, exist_ok=True)
    finals = {
        "pretrain_train": destination / "pretrain_train.bin",
        "pretrain_validation": destination / "pretrain_validation.bin",
        "validation_probe": destination / "validation_probe.npy",
        "metadata": destination / "metadata.json",
    }
    if not force and any(path.exists() for path in finals.values()):
        raise CorpusBuildError("output artifact already exists; pass --force to replace it")
    for path in finals.values():
        if path.is_symlink():
            raise CorpusBuildError(f"refusing symlink output: {path}")
    parts = {name: _part_path(path) for name, path in finals.items()}
    for path in parts.values():
        if path.exists() or path.is_symlink():
            path.unlink()

    targets = {
        "train": int(train_token_target),
        "validation": int(validation_token_target),
    }
    written = {"train": 0, "validation": 0}
    contributing = {"train": 0, "validation": 0}
    complete = {"train": 0, "validation": 0}
    partial = {"train": 0, "validation": 0}
    assigned_scanned = {"train": 0, "validation": 0}
    ordered_digests = {"train": hashlib.sha256(), "validation": hashlib.sha256()}
    boundary_documents: dict[str, dict[str, Any] | None] = {
        "train": None,
        "validation": None,
    }
    probe_offsets: list[int] = []
    probe_lineage: list[dict[str, Any]] = []
    probe_document_hashes: set[bytes] = set()
    documents_scanned = 0
    shards_scanned = 0

    try:
        with parts["pretrain_train"].open("xb") as train_handle, parts[
            "pretrain_validation"
        ].open("xb") as validation_handle:
            handles = {"train": train_handle, "validation": validation_handle}
            for spec, path in verified_shards:
                for source_row, text in _parquet_documents(path, spec):
                    documents_scanned += 1
                    identity = document_digest(text)
                    split = assign_split(identity)
                    assigned_scanned[split] += 1
                    if written[split] >= targets[split]:
                        if all(written[name] >= targets[name] for name in targets):
                            break
                        continue

                    token_ids = tokenizer.encode_ordinary(text)
                    values = np.asarray([*token_ids, EOT_TOKEN], dtype="<u2")
                    if values.ndim != 1 or np.any(values >= VOCAB_SIZE):
                        raise CorpusBuildError("tokenizer produced an invalid GPT-2 token id")
                    remaining = targets[split] - written[split]
                    take = min(len(values), remaining)
                    if take <= 0:
                        continue
                    document_corpus_offset = written[split]
                    values[:take].tofile(handles[split])
                    written[split] += int(take)
                    contributing[split] += 1
                    is_complete = take == len(values)
                    if is_complete:
                        complete[split] += 1
                    else:
                        partial[split] += 1
                        boundary_documents[split] = {
                            "source_shard_index": int(spec.index),
                            "source_row": int(source_row),
                            "document_sha256": identity.hex(),
                            "full_tokens_including_eot": int(len(values)),
                            "written_prefix_tokens": int(take),
                        }
                    _ordered_document_update(
                        ordered_digests[split],
                        shard_index=spec.index,
                        source_row=source_row,
                        document_hash=identity,
                        full_tokens=len(values),
                        written_tokens=take,
                    )

                    if (
                        split == "validation"
                        and is_complete
                        and len(values) >= probe_window_tokens
                        and identity not in probe_document_hashes
                        and len(probe_offsets) < probe_documents
                    ):
                        within = fixed_probe_offset(
                            identity, len(values), probe_window_tokens
                        )
                        absolute = int(document_corpus_offset + within)
                        probe_document_hashes.add(identity)
                        probe_offsets.append(absolute)
                        probe_lineage.append(
                            {
                                "probe_index": len(probe_offsets) - 1,
                                "source_shard_index": int(spec.index),
                                "source_row": int(source_row),
                                "document_sha256": identity.hex(),
                                "document_corpus_offset": int(document_corpus_offset),
                                "document_tokens_including_eot": int(len(values)),
                                "context_offset_within_document": int(within),
                                "absolute_corpus_offset": absolute,
                            }
                        )

                    if all(written[name] >= targets[name] for name in targets):
                        break
                shards_scanned += 1
                if all(written[name] >= targets[name] for name in targets):
                    break

        if written != targets:
            raise CorpusBuildError(
                f"pinned shards contain insufficient assigned tokens: {written}, need {targets}"
            )
        if len(probe_offsets) != probe_documents:
            raise CorpusBuildError(
                f"only {len(probe_offsets)} complete, distinct validation documents can "
                f"supply a probe; need {probe_documents}"
            )
        if partial["train"] > 1 or partial["validation"] > 1:
            raise CorpusBuildError("each exact split may truncate at most one boundary document")

        probe_array = np.asarray(probe_offsets, dtype="<i8")
        if probe_array.shape != (probe_documents,) or probe_array.dtype != np.dtype("<i8"):
            raise CorpusBuildError("validation probe serialization contract changed")
        with parts["validation_probe"].open("xb") as handle:
            np.save(handle, probe_array, allow_pickle=False)

        part_artifacts = {
            name: _artifact_record(parts[name])
            for name in ("pretrain_train", "pretrain_validation", "validation_probe")
        }
        for name, record in part_artifacts.items():
            record["path"] = finals[name].name

        source_records = []
        for spec, _ in verified_shards:
            source_records.append(
                {
                    "shard_index": int(spec.index),
                    "relative_path": spec.relative_path,
                    "remote_relative_path": spec.remote_relative_path,
                    "bytes": int(spec.bytes),
                    "sha256": spec.sha256,
                    "xet_hash": spec.xet_hash,
                    "parquet_rows": int(spec.rows),
                    "parquet_row_groups": int(spec.row_groups),
                    "parquet_schema": {"text": "string"},
                }
            )
        tokenizer_records = []
        for record in TOKENIZER_ASSETS:
            tokenizer_records.append(
                {
                    "role": record["role"],
                    "relative_path": record["relative_path"],
                    "bytes": int(record["bytes"]),
                    "sha256": record["sha256"],
                    "verified": tokenizer_assets is not None,
                }
            )

        metadata = {
            "schema_version": SCHEMA_VERSION,
            "completed": True,
            "dataset": {
                "id": DATASET_ID,
                "revision": DATASET_REVISION,
                "source_shard_indices": [int(spec.index) for spec in shard_specs],
            },
            "tokenizer": {
                "name": "gpt2",
                "implementation": "tiktoken local data-gym assets",
                "required_tiktoken_version": "0.14.0",
                "vocab_size": VOCAB_SIZE,
                "eot_token": EOT_TOKEN,
                "document_encoding": "encode_ordinary(text) followed by one EOT",
                "assets": tokenizer_records,
            },
            "construction": {
                "offline_only": True,
                "parquet_iteration": "shard_index_ascending_then_source_row; iter_batches; one_document_tokenized_at_a_time",
                "source_tree_access": "read_only",
                "serialization": "little_endian_uint16_token_ids",
                "exact_target_rule": "write complete assigned documents, then truncate only the final boundary document prefix per split",
                "document_identity": "sha256(NFKC_then_collapse_whitespace_UTF8)",
                "split_rule": {
                    "domain_hex": SPLIT_DOMAIN.hex(),
                    "hash": "sha256(domain || 32_byte_document_identity)",
                    "integer": "first_8_digest_bytes_big_endian",
                    "validation_condition": f"integer mod {VALIDATION_MODULUS} == {VALIDATION_RESIDUE}",
                    "train_condition": f"integer mod {VALIDATION_MODULUS} != {VALIDATION_RESIDUE}",
                    "content_duplicates_cannot_cross_splits": True,
                },
                "targets": {
                    "train_tokens": int(train_token_target),
                    "validation_tokens": int(validation_token_target),
                },
            },
            "sources": source_records,
            "scan": {
                "documents_scanned": int(documents_scanned),
                "shards_scanned": int(shards_scanned),
                "documents_assigned_while_scanned": assigned_scanned,
            },
            "splits": {
                split: {
                    "tokens_written": int(written[split]),
                    "contributing_documents": int(contributing[split]),
                    "complete_documents": int(complete[split]),
                    "partial_boundary_documents": int(partial[split]),
                    "boundary_document": boundary_documents[split],
                    "ordered_contributing_document_ledger_sha256": ordered_digests[
                        split
                    ].hexdigest(),
                }
                for split in ("train", "validation")
            },
            "validation_probe": {
                "rows": int(probe_documents),
                "shape": [int(probe_documents)],
                "dtype": "<i8",
                "offset_unit": "uint16_token_index_in_pretrain_validation.bin",
                "model_context_tokens": MODEL_CONTEXT_TOKENS,
                "required_contiguous_tokens_per_offset": int(probe_window_tokens),
                "selection": "first distinct complete eligible validation documents in frozen scan order",
                "offset_rule": {
                    "domain_hex": PROBE_OFFSET_DOMAIN.hex(),
                    "hash": "sha256(domain || 32_byte_document_identity)",
                    "integer": "first_8_digest_bytes_big_endian",
                    "within_document_offset": "integer mod (document_tokens_including_eot - required_contiguous_tokens_per_offset + 1)",
                },
                "all_documents_distinct": len(probe_document_hashes) == probe_documents,
                "all_documents_complete_in_validation_stream": True,
                "lineage": probe_lineage,
            },
            "artifacts": part_artifacts,
        }
        _write_json_part(parts["metadata"], metadata)

        # Metadata is the commit record and is therefore published last.
        for name in ("pretrain_train", "pretrain_validation", "validation_probe"):
            os.replace(parts[name], finals[name])
        os.replace(parts["metadata"], finals["metadata"])
        return metadata
    except BaseException:
        for path in parts.values():
            try:
                path.unlink()
            except FileNotFoundError:
                pass
        raise


def main() -> None:
    project_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-project-root",
        type=Path,
        required=True,
        help="Read-only project root containing the seven pinned shards and tokenizer assets.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=project_root / OUTPUT_RELATIVE,
        help=f"Destination directory (default: {OUTPUT_RELATIVE}).",
    )
    parser.add_argument("--force", action="store_true")
    arguments = parser.parse_args()
    result = build_corpus(
        arguments.source_project_root,
        arguments.output_dir,
        force=arguments.force,
    )
    print(json.dumps(result, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
