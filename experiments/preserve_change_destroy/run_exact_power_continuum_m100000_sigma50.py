"""Run the frozen m=100,000 exact-power protocol at label noise variance 50.

This wrapper reuses the audited parameterized implementation and changes only
the noise variance and artifact directory.
"""

from __future__ import annotations

import json
from pathlib import Path

from .run_exact_power_continuum_m100000 import run_experiment


ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "artifacts" / "exact_power_continuum_m100000_sigma50"
SIGMA2 = 50.0


def main() -> None:
    summary = run_experiment(output_dir=OUTPUT_DIR, sigma2=SIGMA2)
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
