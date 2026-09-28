#!/usr/bin/env python3
"""Redraw the preserve/change/destroy triptych from frozen accepted curves."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from experiments.preserve_change_destroy import (
    run_exact_power_continuum_m100000 as experiment,
)


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = (
    ROOT
    / "experiments"
    / "preserve_change_destroy"
    / "artifacts"
    / "exact_power_continuum_m100000_sigma25"
)
OUTPUT = ROOT / "reproduced_figures" / "preserve_change_destroy_continuum_diagnostic"


def rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def main() -> int:
    thetas = experiment.PAPER_THETAS
    fit_rows = [row for row in rows(ARTIFACT / "curves.csv") if row["route"] == "refined"]
    display_rows = [
        row
        for row in rows(ARTIFACT / "joint_schedule_triptych_curves.csv")
        if row["route"] == "refined"
    ]
    metric_rows = [
        row
        for row in rows(ARTIFACT / "joint_schedule_triptych_metrics.csv")
        if row["route"] == "refined"
    ]

    fit_times = np.asarray(
        [float(row["intrinsic_time"]) for row in fit_rows if float(row["theta"]) == thetas[0]],
        dtype=np.float64,
    )
    display_times = np.asarray(
        [float(row["intrinsic_time"]) for row in display_rows if float(row["theta"]) == thetas[0]],
        dtype=np.float64,
    )
    refined = {
        "target_times": fit_times,
        "metrics": {
            theta: {
                "measured_phase_envelope_exponent": float(
                    next(
                        row["measured_component_envelope_exponent"]
                        for row in metric_rows
                        if float(row["theta"]) == theta
                    )
                )
            }
            for theta in thetas
        },
    }
    display_refined = {
        "target_times": display_times,
        "centered_clean": {},
        "gap": {},
        "envelopes": {},
    }
    for theta in thetas:
        selected = [row for row in display_rows if float(row["theta"]) == theta]
        total = np.asarray([float(row["floor_centered_total"]) for row in selected])
        gap = np.asarray([float(row["noise_clean_gap"]) for row in selected])
        display_refined["centered_clean"][theta] = total - gap
        display_refined["gap"][theta] = gap
        display_refined["envelopes"][theta] = np.asarray(
            [float(row["normalized_component_envelope"]) for row in selected]
        )

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    _, pdf = experiment._make_compact_paper_triptych(
        refined,
        display_refined,
        25.0,
        OUTPUT,
    )
    print(pdf)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
