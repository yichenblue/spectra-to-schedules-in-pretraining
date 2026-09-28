"""Run the m=100,000 exact-power protocol and compact sigma-25 paper figure.

This wrapper reuses the audited frozen gates, changes the noise variance and
artifact directory, and opts into theta=2 solely as a separately audited,
display-only trajectory for the compact triptych.
"""

from __future__ import annotations

import json
from pathlib import Path

from .run_exact_power_continuum_m100000 import run_experiment


ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "artifacts" / "exact_power_continuum_m100000_sigma25"
SIGMA2 = 25.0
DISPLAY_ONLY_THETAS = (2.0,)
MAKE_COMPACT_TRIPTYCH = True


def main() -> None:
    summary = run_experiment(
        output_dir=OUTPUT_DIR,
        sigma2=SIGMA2,
        display_only_thetas=DISPLAY_ONLY_THETAS,
        make_compact_triptych=MAKE_COMPACT_TRIPTYCH,
    )
    print(
        json.dumps(
            {
                "status": summary["status"],
                "sigma2": summary["protocol"]["sigma2"],
                "numerical_contract_pass": summary["numerical_contract_pass"],
                "primary_phase_gate_pass": summary["primary_phase_gate_pass"],
                "sentinel_phase_gate_pass": summary["sentinel_phase_gate_pass"],
                "overall_observable_gate_pass": summary[
                    "overall_observable_gate_pass"
                ],
                "destroy_control": summary["destroy_control"],
                "elapsed_seconds": summary["elapsed_seconds"],
                "artifacts": summary["artifacts"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
