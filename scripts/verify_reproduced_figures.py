#!/usr/bin/env python3
"""Render reference/reproduced PDFs and require pixel-identical first pages."""

from __future__ import annotations

import hashlib
import shutil
import subprocess
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REFERENCE = ROOT / "paper_figures"
REPRODUCED = ROOT / "reproduced_figures"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def render(source: Path, destination_stem: Path) -> Path:
    subprocess.run(
        [
            "pdftoppm",
            "-png",
            "-f",
            "1",
            "-singlefile",
            "-r",
            "100",
            str(source),
            str(destination_stem),
        ],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return Path(f"{destination_stem}.png")


def main() -> int:
    if shutil.which("pdftoppm") is None:
        raise SystemExit("pdftoppm is required for rendered-PDF comparison")
    names = sorted(path.name for path in REFERENCE.glob("*.pdf"))
    failures: list[str] = []
    with tempfile.TemporaryDirectory(prefix="paper-figure-check-") as temporary:
        directory = Path(temporary)
        for name in names:
            reproduced = REPRODUCED / name
            if not reproduced.is_file():
                failures.append(f"missing reproduced figure: {name}")
                continue
            reference_png = render(REFERENCE / name, directory / f"reference-{name}")
            reproduced_png = render(reproduced, directory / f"reproduced-{name}")
            if digest(reference_png) != digest(reproduced_png):
                failures.append(f"rendered pixels differ: {name}")
    if failures:
        print("REPRODUCTION CHECK FAILED")
        for failure in failures:
            print(f"- {failure}")
        return 1
    print(f"REPRODUCTION CHECK PASSED: {len(names)} rendered PDFs are pixel-identical")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
