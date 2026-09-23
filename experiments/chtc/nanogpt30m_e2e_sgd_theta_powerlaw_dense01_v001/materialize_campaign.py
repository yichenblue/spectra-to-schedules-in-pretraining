#!/usr/bin/env python3
"""Prepare the launch-false three-arm dense-theta extension package."""

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


THETAS = (
    ("theta0p125", "0.125"),
    ("theta0p25", "0.25"),
    ("theta0p375", "0.375"),
)


def _configure_base() -> None:
    base.PACKAGE = PACKAGE
    base.CONFIG = (
        REPOSITORY
        / "experiments/configs/nanogpt30m_e2e_sgd_theta_powerlaw_dense01_v001.json"
    )
    base.THETAS = THETAS
    base.ALLOWED_JOB_DURATION = 3_000
    base.CODE_FILES = (
        *base.CODE_FILES,
        "experiments/nanogpt_local/e2e_sgd_theta_powerlaw_30m_dense01_v001.py",
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--attempt", default="a003")
    args = parser.parse_args(argv)
    _configure_base()
    result = base.prepare_tails(args.output_dir, args.checkpoint, args.attempt)
    print(json.dumps(result, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
