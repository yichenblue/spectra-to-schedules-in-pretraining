"""Exploratory dense extension of the 30M theta grid on 0 < theta < 0.5.

This wrapper changes only the enumerated theta values and their exact discrete
update counts.  Training, data, checkpoint, tape, evaluation, optimizer, and
resource contracts remain those of ``e2e_sgd_theta_powerlaw_30m_v001``.
"""

from __future__ import annotations

from typing import Sequence

from experiments.nanogpt_local import e2e_sgd_theta_powerlaw_30m_v001 as core


DENSE_THETAS = (0.125, 0.25, 0.375)
DENSE_TAIL_UPDATES = {
    0.125: 18_244,
    0.25: 23_364,
    0.375: 30_096,
}
DENSE_TAGS = {
    0.125: "theta0p125",
    0.25: "theta0p25",
    0.375: "theta0p375",
}


def _install_dense_grid() -> None:
    core.THETAS = DENSE_THETAS
    core.EXPECTED_TAIL_UPDATES = DENSE_TAIL_UPDATES

    def theta_tag(theta: float) -> str:
        return DENSE_TAGS[core._theta(theta)]

    core._theta_tag = theta_tag
    core._tail_schedule.cache_clear()


_install_dense_grid()


def main(argv: Sequence[str] | None = None) -> int:
    return core.main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
