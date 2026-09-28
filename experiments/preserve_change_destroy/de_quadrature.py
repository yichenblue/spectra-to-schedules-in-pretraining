"""Head-preserving spectral quadrature and continuum modal dynamics.

The implementation follows the retained high-width route in co-mathematician
workstream ws_443: every leading spectral mode is retained, and only the
smooth power-law tail is compressed into logarithmic index bins.  It is a
continuum spectral proxy, not a contour deterministic equivalent for a finite
random feature matrix.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
from scipy import special


Array = np.ndarray


@dataclass(frozen=True)
class DEQuadrature:
    alpha: float
    beta: float
    effective_width: int
    relative_bin_width: float
    index_nodes: Array
    multiplicities: Array
    eigenvalues: Array
    initial_clean_mass: Array
    approximation_floor: float
    total_teacher_energy: float

    @property
    def mode_count(self) -> int:
        return int(self.eigenvalues.size)


@dataclass(frozen=True)
class DETrajectory:
    effective_width: int
    theta: float
    times: Array
    clean_centered: Array
    noise_gap: Array
    noisy_centered: Array
    row_mass: Array
    maximum_cell_coupling: float
    maximum_row_mass: float
    stable: bool


def head_preserving_quadrature(
    alpha: float,
    beta: float,
    effective_width: int,
    relative_bin_width: float,
) -> DEQuadrature:
    """Return logarithmic index bins while retaining every leading mode."""

    if alpha <= 0.0 or beta <= 0.0:
        raise ValueError("alpha and beta must be positive")
    if effective_width <= 0:
        raise ValueError("effective_width must be positive")
    if not (0.0 < relative_bin_width < 1.0):
        raise ValueError("relative_bin_width must lie in (0,1)")
    target_power = 2.0 * (alpha + beta)
    if target_power <= 1.0:
        raise ValueError("finite target energy requires 2(alpha+beta)>1")

    starts: list[int] = []
    cursor = 1
    while cursor <= effective_width:
        starts.append(cursor)
        cursor += max(1, int(math.floor(relative_bin_width * cursor)))
    lower = np.asarray(starts, dtype=float)
    upper = np.minimum(
        float(effective_width),
        np.append(lower[1:] - 1.0, float(effective_width)),
    )
    multiplicities = upper - lower + 1.0
    index_nodes = np.exp(0.5 * (np.log(lower) + np.log(upper)))
    eigenvalues = index_nodes ** (-2.0 * alpha)
    initial_clean_mass = multiplicities * index_nodes ** (-target_power)
    approximation_floor = float(
        special.zeta(target_power, float(effective_width) + 1.0)
    )
    total_teacher_energy = float(special.zeta(target_power, 1.0))
    if (
        not np.all(np.isfinite(eigenvalues))
        or not np.all(eigenvalues > 0.0)
        or approximation_floor < 0.0
    ):
        raise FloatingPointError("non-finite spectral quadrature")
    return DEQuadrature(
        alpha=alpha,
        beta=beta,
        effective_width=effective_width,
        relative_bin_width=relative_bin_width,
        index_nodes=index_nodes,
        multiplicities=multiplicities,
        eigenvalues=eigenvalues,
        initial_clean_mass=initial_clean_mass,
        approximation_floor=approximation_floor,
        total_teacher_energy=total_teacher_energy,
    )


def triangular_time_grid(
    effective_width: int,
    horizon_exponent: float,
    time_min: float,
    points_per_decade: int,
) -> Array:
    """Log grid extending to a strict sub-saturation power of the width."""

    if effective_width <= 1 or not (0.0 < horizon_exponent < 1.0):
        raise ValueError("require width>1 and horizon_exponent in (0,1)")
    if time_min <= 0.0 or points_per_decade < 2:
        raise ValueError("invalid time-grid controls")
    maximum = math.exp(horizon_exponent * math.log(float(effective_width)))
    decades = math.log10(maximum / time_min)
    intervals = max(2, int(math.ceil(decades * points_per_decade)))
    positive = np.geomspace(time_min, maximum, intervals + 1)
    return np.concatenate([np.array([0.0]), positive])


def run_continuum_modal_dynamics(
    quadrature: DEQuadrature,
    times: Array,
    theta_values: tuple[float, ...],
    sigma2: float,
    inverse_ratio_amplitude: float,
    row_mass_cap: float,
) -> dict[float, DETrajectory]:
    """Integrate the positive continuum modal system on a log-time mesh.

    Modal survival is exact within each cell.  The scalar clean/noise
    feedback is trapezoidal and solved implicitly through its rank-one sum.
    Resolution doubling is therefore required for every paper-facing run.
    """

    time = np.asarray(times, dtype=float)
    if time.ndim != 1 or time.size < 2 or time[0] != 0.0:
        raise ValueError("times must be a one-dimensional grid starting at zero")
    if np.any(np.diff(time) <= 0.0):
        raise ValueError("times must be strictly increasing")
    if sigma2 < 0.0 or inverse_ratio_amplitude <= 0.0:
        raise ValueError("invalid noise or ratio amplitude")
    if not (0.0 < row_mass_cap < 1.0):
        raise ValueError("row_mass_cap must lie in (0,1)")

    theta = np.asarray(theta_values, dtype=float)
    if np.any(theta < 0.0) or np.unique(theta).size != theta.size:
        raise ValueError("theta values must be distinct and nonnegative")
    count = theta.size
    modes = quadrature.mode_count
    eigenvalues = quadrature.eigenvalues
    multiplicities = quadrature.multiplicities
    clean_modes = np.broadcast_to(
        quadrature.initial_clean_mass, (count, modes)
    ).copy()
    gap_modes = np.zeros((count, modes), dtype=float)
    row_modes = np.zeros((count, modes), dtype=float)

    clean = np.empty((count, time.size), dtype=float)
    gap = np.empty_like(clean)
    row = np.empty_like(clean)
    clean[:, 0] = np.sum(clean_modes, axis=1)
    gap[:, 0] = 0.0
    row[:, 0] = 0.0
    maximum_cell = np.zeros(count, dtype=float)

    for index in range(time.size - 1):
        left = float(time[index])
        right = float(time[index + 1])
        delta = right - left
        midpoint = math.sqrt((1.0 + left) * (1.0 + right)) - 1.0
        survival = np.exp(-2.0 * eigenvalues * delta)
        one_minus_survival = -np.expm1(-2.0 * eigenvalues * delta)
        # Integral of multiplicity * lambda^2 * exp(-2 lambda age)
        # over this cell, before multiplying by 1/r(midpoint).
        base_injection = (
            0.5 * multiplicities * eigenvalues * one_minus_survival
        )
        inverse_ratio = inverse_ratio_amplitude * (
            1.0 + midpoint
        ) ** (-theta)
        coefficients = inverse_ratio[:, None] * base_injection[None, :]
        cell_coupling = np.sum(coefficients, axis=1)
        maximum_cell = np.maximum(maximum_cell, cell_coupling)
        denominator = 1.0 - 0.5 * cell_coupling
        if np.any(denominator <= 0.0):
            raise RuntimeError("continuum cell lost its rank-one stability margin")

        old_clean = clean[:, index]
        clean_base = np.sum(clean_modes * survival[None, :], axis=1)
        new_clean = (
            clean_base
            + cell_coupling
            * (quadrature.approximation_floor + 0.5 * old_clean)
        ) / denominator
        clean_source = quadrature.approximation_floor + 0.5 * (
            old_clean + new_clean
        )
        clean_modes = (
            clean_modes * survival[None, :]
            + coefficients * clean_source[:, None]
        )

        old_gap = gap[:, index]
        gap_base = np.sum(gap_modes * survival[None, :], axis=1)
        new_gap = (
            gap_base
            + cell_coupling * (sigma2 + 0.5 * old_gap)
        ) / denominator
        gap_source = sigma2 + 0.5 * (old_gap + new_gap)
        gap_modes = (
            gap_modes * survival[None, :]
            + coefficients * gap_source[:, None]
        )

        row_modes = (
            row_modes * survival[None, :] + coefficients
        )
        clean[:, index + 1] = np.sum(clean_modes, axis=1)
        gap[:, index + 1] = np.sum(gap_modes, axis=1)
        row[:, index + 1] = np.sum(row_modes, axis=1)
        if (
            not np.all(np.isfinite(clean[:, index + 1]))
            or not np.all(np.isfinite(gap[:, index + 1]))
            or np.any(clean[:, index + 1] < 0.0)
            or np.any(gap[:, index + 1] < 0.0)
        ):
            raise FloatingPointError("continuum modal state became invalid")

    output: dict[float, DETrajectory] = {}
    for theta_index, theta_value in enumerate(theta_values):
        maximum_row = float(np.max(row[theta_index]))
        output[float(theta_value)] = DETrajectory(
            effective_width=quadrature.effective_width,
            theta=float(theta_value),
            times=time.copy(),
            clean_centered=clean[theta_index].copy(),
            noise_gap=gap[theta_index].copy(),
            noisy_centered=(clean[theta_index] + gap[theta_index]).copy(),
            row_mass=row[theta_index].copy(),
            maximum_cell_coupling=float(maximum_cell[theta_index]),
            maximum_row_mass=maximum_row,
            stable=bool(maximum_row < row_mass_cap),
        )
    return output


def run_scheduled_continuum_modal_dynamics(
    quadrature: DEQuadrature,
    times: Array,
    batches: Array,
    eta: float,
    theta: float,
    sigma2: float,
    row_mass_cap: float,
) -> DETrajectory:
    """Run the continuum modal proxy on an explicit discrete SGD schedule.

    The time cells and integer mini-batches are exactly those used by the
    finite discrete recursion and Monte Carlo bridge.  On cell ``t`` the
    continuum inverse ratio is ``eta / B_t``.  This isolates the continuous-
    time approximation from spectrum, schedule, and time-grid differences.
    """

    time = np.asarray(times, dtype=float)
    batch = np.asarray(batches, dtype=np.int64)
    if eta <= 0.0 or sigma2 < 0.0:
        raise ValueError("require eta>0 and sigma2>=0")
    if not (0.0 < row_mass_cap < 1.0):
        raise ValueError("row_mass_cap must lie in (0,1)")
    if time.ndim != 1 or batch.ndim != 1 or time.size != batch.size + 1:
        raise ValueError("times must have exactly one more entry than batches")
    if time.size < 2 or time[0] != 0.0 or np.any(batch <= 0):
        raise ValueError("invalid scheduled continuum inputs")
    if not np.allclose(np.diff(time), eta, rtol=1.0e-12, atol=1.0e-14):
        raise ValueError("scheduled continuum requires the common grid T_t=eta*t")

    eigenvalues = quadrature.eigenvalues
    multiplicities = quadrature.multiplicities
    clean_modes = quadrature.initial_clean_mass.copy()
    gap_modes = np.zeros_like(clean_modes)
    row_modes = np.zeros_like(clean_modes)
    clean = np.empty(time.size, dtype=float)
    gap = np.empty_like(clean)
    row = np.empty_like(clean)
    clean[0] = float(np.sum(clean_modes))
    gap[0] = 0.0
    row[0] = 0.0
    maximum_cell = 0.0

    survival = np.exp(-2.0 * eigenvalues * eta)
    one_minus_survival = -np.expm1(-2.0 * eigenvalues * eta)
    base_injection = 0.5 * multiplicities * eigenvalues * one_minus_survival
    for index, batch_value in enumerate(batch):
        coefficients = (eta / float(batch_value)) * base_injection
        cell_coupling = float(np.sum(coefficients))
        maximum_cell = max(maximum_cell, cell_coupling)
        denominator = 1.0 - 0.5 * cell_coupling
        if denominator <= 0.0:
            raise RuntimeError("scheduled continuum cell lost its stability margin")

        old_clean = clean[index]
        clean_base = float(np.sum(clean_modes * survival))
        new_clean = (
            clean_base
            + cell_coupling
            * (quadrature.approximation_floor + 0.5 * old_clean)
        ) / denominator
        clean_source = quadrature.approximation_floor + 0.5 * (
            old_clean + new_clean
        )
        clean_modes = clean_modes * survival + coefficients * clean_source

        old_gap = gap[index]
        gap_base = float(np.sum(gap_modes * survival))
        new_gap = (
            gap_base + cell_coupling * (sigma2 + 0.5 * old_gap)
        ) / denominator
        gap_source = sigma2 + 0.5 * (old_gap + new_gap)
        gap_modes = gap_modes * survival + coefficients * gap_source

        row_modes = row_modes * survival + coefficients
        clean[index + 1] = float(np.sum(clean_modes))
        gap[index + 1] = float(np.sum(gap_modes))
        row[index + 1] = float(np.sum(row_modes))
        if (
            not np.isfinite(clean[index + 1])
            or not np.isfinite(gap[index + 1])
            or clean[index + 1] < 0.0
            or gap[index + 1] < 0.0
        ):
            raise FloatingPointError("scheduled continuum state became invalid")

    maximum_row = float(np.max(row))
    return DETrajectory(
        effective_width=quadrature.effective_width,
        theta=float(theta),
        times=time.copy(),
        clean_centered=clean,
        noise_gap=gap,
        noisy_centered=clean + gap,
        row_mass=row,
        maximum_cell_coupling=maximum_cell,
        maximum_row_mass=maximum_row,
        stable=bool(maximum_row < row_mass_cap),
    )


def spectral_profiles(
    quadrature: DEQuadrature,
    times: Array,
    chunk_size: int = 32,
) -> tuple[Array, Array]:
    """Return the bare memory kernel and learnable clean forcing."""

    time = np.asarray(times, dtype=float)
    kernel = np.empty_like(time)
    forcing = np.empty_like(time)
    kernel_weights = quadrature.multiplicities * quadrature.eigenvalues**2
    for start in range(0, time.size, chunk_size):
        stop = min(start + chunk_size, time.size)
        survival = np.exp(
            -2.0
            * quadrature.eigenvalues[:, None]
            * time[None, start:stop]
        )
        kernel[start:stop] = kernel_weights @ survival
        forcing[start:stop] = quadrature.initial_clean_mass @ survival
    return kernel, forcing
