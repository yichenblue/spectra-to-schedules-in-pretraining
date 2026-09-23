#!/usr/bin/env python3
"""Prepare the launch-false three-arm high-theta extension package."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Sequence

PACKAGE = Path(__file__).resolve().parent
REPOSITORY = Path(__file__).resolve().parents[3]
if str(REPOSITORY) not in sys.path:
    sys.path.insert(0, str(REPOSITORY))

from experiments.chtc.nanogpt30m_e2e_sgd_theta_powerlaw_v001 import (
    materialize_campaign as base,
)
from experiments.nanogpt_local import e2e_sgd_theta_powerlaw_30m_high_v001 as high


THETAS = (
    ("theta1p25", "1.25"),
    ("theta1p75", "1.75"),
    ("theta2", "2.0"),
)
TRAIN_REMOTE_URI = (
    "osdf:///chtc/staging/w/wang3587/nanogpt124m-e2e-adamw-5b-v001/"
    "b5b3bf08ae0ffecdeda46c28273af995d4c5af6934090eb9904f51bf6c17f84b/"
    "pretrain_train.bin"
)
VALIDATION_REMOTE_URI = (
    "osdf:///chtc/staging/w/wang3587/"
    "nanogpt124m-e2e-sgd-rpath-factorization-2p5b-v001/"
    "52085210f2da47797efd93401d8b890c6efdb41055c266d499649e1d82ee20be/"
    "pretrain_validation.bin"
)
PROBE_REMOTE_URI = (
    "osdf:///chtc/staging/w/wang3587/"
    "nanogpt124m-e2e-sgd-rpath-factorization-2p5b-v001/"
    "0295b25ce763f30570d4a268491d5a829e244ade33c602c32423e66aea278155/"
    "validation_probe.npy"
)
REMOTE_DATA = {
    "train": TRAIN_REMOTE_URI,
    "validation": VALIDATION_REMOTE_URI,
    "validation_probe": PROBE_REMOTE_URI,
}
REQUEST_DISK_MB = 16_384


def _configure_base() -> None:
    base.PACKAGE = PACKAGE
    base.CONFIG = (
        REPOSITORY
        / "experiments/configs/nanogpt30m_e2e_sgd_theta_powerlaw_high_v001.json"
    )
    base.THETAS = THETAS
    base.ALLOWED_JOB_DURATION = 38_700
    base.DATA = {
        "train": (
            "b5b3bf08ae0ffecdeda46c28273af995d4c5af6934090eb9904f51bf6c17f84b",
            10_000_015_872,
            "pretrain_train.bin",
        ),
        "validation": (
            "52085210f2da47797efd93401d8b890c6efdb41055c266d499649e1d82ee20be",
            33_554_432,
            "pretrain_validation.bin",
        ),
        "validation_probe": (
            "0295b25ce763f30570d4a268491d5a829e244ade33c602c32423e66aea278155",
            8_320,
            "validation_probe.npy",
        ),
    }
    base.CODE_FILES = (
        *base.CODE_FILES,
        "experiments/nanogpt_local/e2e_sgd_theta_powerlaw_30m_high_v001.py",
    )

    def validate_checkpoint(path: Path) -> dict[str, object]:
        checkpoint = path.expanduser().resolve()
        if not checkpoint.is_file() or checkpoint.is_symlink():
            raise base.MaterializationError("prefix checkpoint is absent or linked")
        payload = base.torch.load(checkpoint, map_location="cpu", weights_only=True)
        data_identity = high.core._canonical_sha256(
            {
                "train": {
                    "sha256": base.DATA["train"][0],
                    "token_count": high.HIGH_CORPUS_TOKENS,
                    "bytes": base.DATA["train"][1],
                },
                "validation": {
                    "sha256": base.DATA["validation"][0],
                    "token_count": 16_777_216,
                    "bytes": base.DATA["validation"][1],
                },
                "validation_probe": {"sha256": base.DATA["validation_probe"][0]},
            }
        )
        try:
            high.core._validate_checkpoint_payload(
                payload,
                data_identity_sha256=data_identity,
                validation_probe_sha256=base.DATA["validation_probe"][0],
            )
        except (RuntimeError, TypeError, ValueError) as error:
            raise base.MaterializationError(
                f"prefix checkpoint failed the full frozen contract: {error}"
            ) from error
        return {
            "path": str(checkpoint),
            "size_bytes": checkpoint.stat().st_size,
            "sha256": base._sha256(checkpoint),
            "model_state_sha256": base.EXPECTED_PREFIX_STATE_SHA256,
        }

    base._validate_checkpoint = validate_checkpoint


def _rewrite_remote_data(output_dir: Path) -> None:
    submit_path = output_dir / "tails.sub"
    text = submit_path.read_text(encoding="utf-8")
    for role, (digest, _, filename) in base.DATA.items():
        generated = f"{base.OSDF_ROOT}/{digest}/{filename}"
        if text.count(generated) != 1:
            raise base.MaterializationError(
                f"expected exactly one generated remote URI for {role}"
            )
        text = text.replace(generated, REMOTE_DATA[role])
    original_disk = "request_disk = 8192MB"
    if text.count(original_disk) != 1:
        raise base.MaterializationError(
            "expected exactly one inherited 8 GiB request_disk declaration"
        )
    text = text.replace(original_disk, f"request_disk = {REQUEST_DISK_MB}MB")
    submit_path.write_text(text, encoding="utf-8")

    manifest_path = output_dir / "prepared_manifest.json"
    manifest = base._json(manifest_path)
    manifest["data_remote_cache"] = {
        role: {
            "path": REMOTE_DATA[role],
            "sha256": digest,
            "size_bytes": size,
            "status": "uploaded_remote_size_and_sha256_verified",
        }
        for role, (digest, size, _) in base.DATA.items()
    }
    manifest["execution_plan"]["request_disk_mb"] = REQUEST_DISK_MB
    manifest["replaces_failed_attempt"] = {
        "attempt": "a005",
        "cluster_id": 6197317,
        "failure": "disk usage exceeded request_disk",
        "scientific_training_executed": False,
    }
    manifest_path.write_text(
        json.dumps(
            manifest,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--attempt", default="a006")
    args = parser.parse_args(argv)
    _configure_base()
    result = base.prepare_tails(args.output_dir, args.checkpoint, args.attempt)
    _rewrite_remote_data(args.output_dir.expanduser().resolve())
    result = base._json(args.output_dir.expanduser().resolve() / "prepared_manifest.json")
    print(json.dumps(result, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
