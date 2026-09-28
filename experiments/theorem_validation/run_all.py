"""Run both theorem-validation experiments."""

from __future__ import annotations

import json

from .ratio_control import run as run_ratio_control
from .spectral_criterion import run as run_spectral_criterion


def main() -> None:
    result = {
        "spectral_criterion": run_spectral_criterion(),
        "ratio_control": run_ratio_control(),
    }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
