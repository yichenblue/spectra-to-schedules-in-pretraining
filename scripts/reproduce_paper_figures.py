#!/usr/bin/env python3
"""Re-create all paper figures from bundled results and plotting code."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "reproduced_figures"


def run(*arguments: str, cwd: Path | None = None) -> None:
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(ROOT)
    environment.setdefault("MPLCONFIGDIR", str(ROOT / ".mplconfig"))
    subprocess.run(
        [sys.executable, *arguments],
        cwd=cwd or ROOT,
        env=environment,
        check=True,
    )


def reproduce_spectral() -> None:
    package = ROOT / "iclr2027" / "experiments" / "theorem_validation"
    run("-m", "iclr2027.experiments.theorem_validation.spectral_six_construction_paper_style")
    run("-m", "iclr2027.experiments.theorem_validation.spectral_rapid_target_triptych")
    six = package / "artifacts" / "spectral_six_construction_paper_style_v002"
    rapid = package / "artifacts" / "spectral_rapid_target_triptych_v009"
    shutil.copy2(
        six / "six_constructions_loss_forcing_memory_part1.pdf",
        OUTPUT / "spectral_six_constructions_powerlike.pdf",
    )
    shutil.copy2(
        six / "six_constructions_loss_forcing_memory_part2.pdf",
        OUTPUT / "spectral_six_constructions_boundaries.pdf",
    )
    shutil.copy2(
        rapid / "rapid_target_risk_forcing_memory.pdf",
        OUTPUT / "spectral_iff_rapid_target.pdf",
    )


def reproduce_preserve_change_destroy() -> None:
    artifacts = ROOT / "iclr2027" / "experiments" / "preserve_change_destroy" / "artifacts"
    run(str(ROOT / "scripts" / "plot_preserve_change_destroy_cached.py"))
    sources = {
        "volterra_three_layer_bridge.pdf": (
            artifacts
            / "experiment_0c_three_layer_bridge"
            / "experiment0c_three_layer_bridge_paper.pdf"
        ),
        "fb_volterra_three_layer_bridge.pdf": (
            artifacts
            / "fb_three_layer_bridge_full"
            / "fb_volterra_three_layer_bridge.pdf"
        ),
    }
    for name, source in sources.items():
        shutil.copy2(source, OUTPUT / name)


def reproduce_fixed_noise() -> None:
    package = ROOT / "iclr2027" / "experiments" / "fixed_noise_early_stopping"
    run("plot_im_early_stopping_composite.py", cwd=package)
    run("plot_im_noise_width_stopping_dynamics.py", cwd=package)
    source = ROOT / "fixed_noise_validation" / "src"
    run("plot_figure5_three_panel_preview.py", cwd=source)
    shutil.copy2(
        ROOT
        / "fixed_noise_validation"
        / "artifacts"
        / "plots"
        / "figure5_three_panel_horizontal_preview.pdf",
        OUTPUT / "fixed_noise_compute_optimal_representative.pdf",
    )


def reproduce_language_model() -> None:
    data = ROOT / "data" / "language_model"
    run(str(ROOT / "scripts" / "plot_nanogpt30m_theta.py"))

    factor_script = ROOT / "experiments" / "nanogpt_local" / "plot_124m_sgd_300m_muon_risk_four_panel_v001.py"
    run(
        str(factor_script),
        "--sgd-root",
        str(data / "nanogpt124m_sgd_factorization" / "tails"),
        "--sgd-prefix",
        str(data / "nanogpt124m_sgd_factorization" / "prefix" / "validation_evaluations.npz"),
        "--muon-root",
        str(data / "nanogpt300m_muon_factorization" / "tails"),
        "--muon-prefix",
        str(data / "nanogpt300m_muon_factorization" / "prefix" / "validation_evaluations.npz"),
        "--output-dir",
        str(OUTPUT),
    )
    shutil.copy2(
        OUTPUT / "124m-sgd-300m-muon-validation-risk-four-panel.pdf",
        OUTPUT / "nanogpt_factorization_collapse_sgd_muon.pdf",
    )

    surrogate = data / "nanogpt124m_surrogate"
    run(
        "-m",
        "experiments.nanogpt_local.plot_124m_sgd_unified_full_prefix_v002",
        "--directory",
        str(surrogate),
    )
    shutil.copy2(
        surrogate / "validation_fits_full_prefix.pdf",
        OUTPUT / "nanogpt124m_sgd_surrogate_fit_transfer.pdf",
    )

    script = (
        "from pathlib import Path; "
        "from experiments.nanogpt_local.plot_124m_v002_four_validation_and_fits "
        "import render_3d_categorical_lanes; "
        f"render_3d_categorical_lanes(Path({str(surrogate)!r}))"
    )
    run("-c", script)
    shutil.copy2(
        surrogate / "four_validation_3d_categorical_lanes_intrinsic.pdf",
        OUTPUT / "nanogpt124m_intrinsic_time_theory_transfer.pdf",
    )


GROUPS = {
    "spectral": reproduce_spectral,
    "preserve": reproduce_preserve_change_destroy,
    "fixed-noise": reproduce_fixed_noise,
    "language-model": reproduce_language_model,
}


def sanitize_generated_json_paths() -> None:
    """Remove machine-local absolute paths from generated JSON summaries."""

    def clean(value: object) -> object:
        if isinstance(value, dict):
            cleaned: dict[str, object] = {}
            for key, item in value.items():
                key_path = Path(key)
                if key_path.is_absolute():
                    key = f"SOURCE_SNAPSHOT/{key_path.parent.name}/{key_path.name}"
                cleaned[key] = clean(item)
            return cleaned
        if isinstance(value, list):
            return [clean(item) for item in value]
        if isinstance(value, str) and Path(value).is_absolute():
            value_path = Path(value)
            return f"SOURCE_SNAPSHOT/{value_path.parent.name}/{value_path.name}"
        return value

    for path in ROOT.rglob("*.json"):
        payload = json.loads(path.read_text(encoding="utf-8"))
        sanitized = clean(payload)
        if sanitized != payload:
            path.write_text(
                json.dumps(sanitized, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--group",
        action="append",
        choices=tuple(GROUPS) + ("all",),
        help="Figure group to reproduce; repeat the option to select several.",
    )
    arguments = parser.parse_args()
    selected = arguments.group or ["all"]
    if "all" in selected:
        selected = list(GROUPS)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    for name in selected:
        print(f"== {name} ==")
        GROUPS[name]()
    sanitize_generated_json_paths()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
