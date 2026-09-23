#!/usr/bin/env python3
"""Verify the release structure, figure coverage, and publication hygiene."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EXPECTED_FIGURES = {
    "nanogpt30m_theta_schedule_response.pdf",
    "nanogpt_factorization_collapse_sgd_muon.pdf",
    "nanogpt124m_sgd_surrogate_fit_transfer.pdf",
    "nanogpt124m_intrinsic_time_theory_transfer.pdf",
    "spectral_six_constructions_powerlike.pdf",
    "spectral_six_constructions_boundaries.pdf",
    "preserve_change_destroy_continuum_diagnostic.pdf",
    "spectral_iff_rapid_target.pdf",
    "fixed_noise_im_early_stopping_composite.pdf",
    "fixed_noise_compute_optimal_representative.pdf",
    "fixed_noise_im_noise_width_stopping_dynamics.pdf",
    "volterra_three_layer_bridge.pdf",
    "fb_volterra_three_layer_bridge.pdf",
}
FORBIDDEN_SUFFIXES = {".pt", ".pth", ".ckpt", ".safetensors"}
FORBIDDEN_TEXT = ("/Users/", "yichenwang", "sk-proj-", "sk_live_", "ghp_")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    failures: list[str] = []
    figures = {path.name for path in (ROOT / "paper_figures").glob("*.pdf")}
    if figures != EXPECTED_FIGURES:
        failures.append(
            f"paper figure mismatch: missing={sorted(EXPECTED_FIGURES - figures)}, "
            f"extra={sorted(figures - EXPECTED_FIGURES)}"
        )

    for path in ROOT.rglob("*"):
        if not path.is_file() or ".git" in path.parts:
            continue
        if path.resolve() == Path(__file__).resolve():
            continue
        if path.suffix.lower() in FORBIDDEN_SUFFIXES:
            failures.append(f"checkpoint-like file included: {path.relative_to(ROOT)}")
        if path.stat().st_size > 100_000_000:
            failures.append(f"file exceeds GitHub 100 MB limit: {path.relative_to(ROOT)}")
        if path.suffix.lower() in {".py", ".md", ".json", ".sh", ".txt", ".toml", ".yaml", ".yml"}:
            text = path.read_text(encoding="utf-8", errors="replace")
            for marker in FORBIDDEN_TEXT:
                if marker in text:
                    failures.append(f"forbidden marker {marker!r}: {path.relative_to(ROOT)}")

    hashes = {
        path.name: sha256(path)
        for path in sorted((ROOT / "paper_figures").glob("*.pdf"))
    }
    expected_hashes = json.loads((ROOT / "paper_figures.sha256.json").read_text())
    if hashes != expected_hashes:
        failures.append("paper figure checksums do not match paper_figures.sha256.json")

    if failures:
        print("RELEASE CHECK FAILED")
        for failure in failures:
            print(f"- {failure}")
        return 1
    print(f"RELEASE CHECK PASSED: {len(figures)} paper figures covered")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
