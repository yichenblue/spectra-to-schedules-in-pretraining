"""Continuum modal dynamics for an exact intrinsic-time power ratio path.

This module is deliberately separate from :mod:`de_quadrature`.  Historical
artifacts use the shifted path ``(1 + T)**theta`` or an explicit integer batch
schedule; the experiment here instead freezes

    r_theta(T) = ratio_amplitude * max(1, T)**theta.

The inverse ratio is integrated exactly over every time cell.  Modal survival
is exact for a cell-constant source, while the scalar feedback source is still
trapezoidal.  Consequently paper-facing runs must retain a time-resolution
doubling check.
"""

from __future__ import annotations

import math

import numpy as np

from .de_quadrature import DEQuadrature, DETrajectory


Array = np.ndarray


def _positive_power_integral(left: float, right: float, theta: float) -> float:
    """Return integral_left^right u**(-theta) du for 1 <= left < right."""

    if not (1.0 <= left < right) or theta < 0.0:
        raise ValueError("require 1 <= left < right and theta >= 0")
    exponent = 1.0 - theta
    logarithmic_width = math.log(right / left)
    if abs(exponent) <= 1.0e-12:
        return logarithmic_width
    # This form stays accurate both near theta=1 and on narrow log cells.
    return (
        math.exp(exponent * math.log(left))
        * math.expm1(exponent * logarithmic_width)
        / exponent
    )


def exact_power_inverse_ratio_cell_integrals(
    times: Array,
    theta_values: tuple[float, ...],
    inverse_ratio_amplitude: float,
) -> Array:
    """Integrate ``amplitude * max(1,T)**(-theta)`` on every time cell.

    The returned array has shape ``(len(theta_values), len(times)-1)``.  The
    cell crossing ``T=1`` is split explicitly; no midpoint sampling or integer
    batch rounding enters this calculation.
    """

    time = np.asarray(times, dtype=float)
    theta = np.asarray(theta_values, dtype=float)
    if time.ndim != 1 or time.size < 2 or time[0] != 0.0:
        raise ValueError("times must be a one-dimensional grid starting at zero")
    if np.any(np.diff(time) <= 0.0):
        raise ValueError("times must be strictly increasing")
    if theta.ndim != 1 or np.any(theta < 0.0):
        raise ValueError("theta values must be one-dimensional and nonnegative")
    if np.unique(theta).size != theta.size:
        raise ValueError("theta values must be distinct")
    if inverse_ratio_amplitude <= 0.0:
        raise ValueError("inverse_ratio_amplitude must be positive")

    integrals = np.empty((theta.size, time.size - 1), dtype=float)
    for cell_index, (left_value, right_value) in enumerate(
        zip(time[:-1], time[1:], strict=True)
    ):
        left = float(left_value)
        right = float(right_value)
        constant_mass = max(0.0, min(right, 1.0) - left)
        power_left = max(left, 1.0)
        for theta_index, theta_value in enumerate(theta):
            power_mass = 0.0
            if right > power_left:
                power_mass = _positive_power_integral(
                    power_left, right, float(theta_value)
                )
            integrals[theta_index, cell_index] = (
                inverse_ratio_amplitude * (constant_mass + power_mass)
            )
    if not np.all(np.isfinite(integrals)) or np.any(integrals <= 0.0):
        raise FloatingPointError("invalid exact-power cell integral")
    return integrals


def run_exact_power_ratio_continuum_modal_dynamics(
    quadrature: DEQuadrature,
    times: Array,
    theta_values: tuple[float, ...],
    sigma2: float,
    inverse_ratio_amplitude: float,
    row_mass_cap: float,
) -> dict[float, DETrajectory]:
    """Integrate the modal system with a cell-averaged pure-power ratio path.

    On cell ``[T_i,T_{i+1}]`` the scalar inverse-ratio coefficient is

    ``mean_i = integral_i / (T_{i+1} - T_i)``.

    It multiplies the exact constant-coefficient modal survival filter.  This
    exactly matches the schedule injection mass in each cell, but it is not a
    claim that the power-weighted modal survival integral is closed-form exact;
    time-grid refinement controls that remaining approximation.
    """

    time = np.asarray(times, dtype=float)
    if sigma2 < 0.0:
        raise ValueError("sigma2 must be nonnegative")
    if not (0.0 < row_mass_cap < 1.0):
        raise ValueError("row_mass_cap must lie in (0,1)")
    cell_integrals = exact_power_inverse_ratio_cell_integrals(
        time,
        theta_values,
        inverse_ratio_amplitude,
    )

    theta = np.asarray(theta_values, dtype=float)
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

    for cell_index in range(time.size - 1):
        delta = float(time[cell_index + 1] - time[cell_index])
        survival = np.exp(-2.0 * eigenvalues * delta)
        one_minus_survival = -np.expm1(-2.0 * eigenvalues * delta)
        base_injection = 0.5 * multiplicities * eigenvalues * one_minus_survival
        mean_inverse_ratio = cell_integrals[:, cell_index] / delta
        coefficients = mean_inverse_ratio[:, None] * base_injection[None, :]
        cell_coupling = np.sum(coefficients, axis=1)
        maximum_cell = np.maximum(maximum_cell, cell_coupling)
        denominator = 1.0 - 0.5 * cell_coupling
        if np.any(denominator <= 0.0):
            raise RuntimeError("pure-power continuum cell lost stability margin")

        old_clean = clean[:, cell_index]
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

        old_gap = gap[:, cell_index]
        gap_base = np.sum(gap_modes * survival[None, :], axis=1)
        new_gap = (
            gap_base + cell_coupling * (sigma2 + 0.5 * old_gap)
        ) / denominator
        gap_source = sigma2 + 0.5 * (old_gap + new_gap)
        gap_modes = (
            gap_modes * survival[None, :]
            + coefficients * gap_source[:, None]
        )

        row_modes = row_modes * survival[None, :] + coefficients
        clean[:, cell_index + 1] = np.sum(clean_modes, axis=1)
        gap[:, cell_index + 1] = np.sum(gap_modes, axis=1)
        row[:, cell_index + 1] = np.sum(row_modes, axis=1)
        if (
            not np.all(np.isfinite(clean[:, cell_index + 1]))
            or not np.all(np.isfinite(gap[:, cell_index + 1]))
            or not np.all(np.isfinite(row[:, cell_index + 1]))
            or np.any(clean[:, cell_index + 1] < 0.0)
            or np.any(gap[:, cell_index + 1] < 0.0)
            or np.any(row[:, cell_index + 1] < 0.0)
        ):
            raise FloatingPointError("pure-power continuum state became invalid")

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
