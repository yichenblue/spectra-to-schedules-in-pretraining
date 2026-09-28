"""Exact empirical varying-batch dynamics and independent validation routes."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from .spectrum import EmpiricalSpectrum


Array = np.ndarray


@dataclass(frozen=True)
class ExactTrajectory:
    """Exact conditional second-moment trajectory on one empirical spectrum."""

    times: Array
    batches: Array
    clean_centered: Array
    noise_gap: Array
    noisy_centered: Array
    clean_total: Array
    noisy_total: Array
    row_mass: Array
    maximum_pointwise_product: float
    minimum_pointwise_margin: float
    maximum_row_mass: float
    pointwise_stable: bool
    row_stable: bool


@dataclass(frozen=True)
class VolterraTrajectory:
    """Independent finite two-time Volterra solution used by validation tests."""

    forcing: Array
    kernel: Array
    clean_total: Array
    noisy_total: Array
    noise_gap: Array


def batch_schedule(
    eta: float,
    theta: float,
    initial_batch: int,
    horizon: int,
) -> tuple[Array, Array]:
    """Return intrinsic grid T_t and integer B_t for t=0,...,horizon-1."""

    if eta <= 0.0:
        raise ValueError("eta must be positive")
    if theta < 0.0:
        raise ValueError("theta must be nonnegative")
    if initial_batch <= 0 or horizon < 0:
        raise ValueError("initial_batch must be positive and horizon nonnegative")
    times = eta * np.arange(horizon + 1, dtype=float)
    raw = initial_batch * (1.0 + times[:-1]) ** theta
    batches = np.ceil(raw).astype(np.int64)
    return times, batches


def horizon_for_width(
    width: int,
    alpha: float,
    eta: float,
    horizon_factor: float,
) -> int:
    """Largest integer horizon covering the declared pre-saturation window."""

    if width <= 0 or alpha <= 0.0 or eta <= 0.0 or horizon_factor <= 0.0:
        raise ValueError("width, alpha, eta, and horizon_factor must be positive")
    intrinsic_horizon = horizon_factor * width ** (2.0 * alpha)
    return max(1, int(math.ceil(intrinsic_horizon / eta)))


def one_step_polynomial(eigenvalues: Array, eta: float, batch: int) -> Array:
    """Current-paper averaged-gradient one-step polynomial."""

    values = np.asarray(eigenvalues, dtype=float)
    if eta <= 0.0 or batch <= 0:
        raise ValueError("eta and batch must be positive")
    return (
        1.0
        - 2.0 * eta * values
        + (1.0 + 1.0 / batch) * eta * eta * values * values
    )


def run_exact_dynamics(
    spectrum: EmpiricalSpectrum,
    eta: float,
    theta: float,
    initial_batch: int,
    sigma2: float,
    horizon: int,
    row_mass_cap: float = 0.8,
) -> ExactTrajectory:
    """Evolve clean modes, the noisy--clean gap, and exact kernel row mass.

    The gap is evolved directly.  No subtraction of two nearly equal risks is
    used.  The approximation floor is kept explicitly in the clean scalar
    feedback but is never subtracted numerically from a full risk curve.
    """

    if sigma2 < 0.0:
        raise ValueError("sigma2 must be nonnegative")
    if not (0.0 < row_mass_cap < 1.0):
        raise ValueError("row_mass_cap must lie in (0,1)")

    eigenvalues = np.asarray(spectrum.eigenvalues, dtype=float)
    clean_modes = np.asarray(spectrum.teacher_overlaps, dtype=float).copy()
    gap_modes = np.zeros_like(clean_modes)
    row_modes = np.zeros_like(clean_modes)
    squared_eigenvalues = eigenvalues * eigenvalues
    times, batches = batch_schedule(eta, theta, initial_batch, horizon)

    clean_centered = np.empty(horizon + 1, dtype=float)
    noise_gap = np.empty(horizon + 1, dtype=float)
    clean_total = np.empty(horizon + 1, dtype=float)
    row_mass = np.empty(horizon + 1, dtype=float)
    pointwise_products = np.empty(horizon, dtype=float)

    clean_centered[0] = float(np.sum(clean_modes))
    noise_gap[0] = 0.0
    clean_total[0] = spectrum.approximation_floor + clean_centered[0]
    row_mass[0] = 0.0

    for step, batch in enumerate(batches):
        batch_int = int(batch)
        q_values = one_step_polynomial(eigenvalues, eta, batch_int)
        injection = (eta * eta / batch_int) * squared_eigenvalues
        pointwise_products[step] = (
            eta * (1.0 + 1.0 / batch_int) * spectrum.lambda_max
        )

        current_clean_total = spectrum.approximation_floor + float(
            np.sum(clean_modes)
        )
        current_gap = float(np.sum(gap_modes))
        clean_modes = q_values * clean_modes + injection * current_clean_total
        gap_modes = q_values * gap_modes + injection * (current_gap + sigma2)
        row_modes = q_values * row_modes + injection

        numerical_scale = max(1.0, spectrum.teacher_energy, sigma2)
        negativity_tolerance = 1.0e-13 * numerical_scale
        if (
            float(np.min(clean_modes, initial=0.0)) < -negativity_tolerance
            or float(np.min(gap_modes, initial=0.0)) < -negativity_tolerance
            or float(np.min(row_modes, initial=0.0)) < -negativity_tolerance
        ):
            raise FloatingPointError("a positive modal state became materially negative")
        clean_modes = np.maximum(clean_modes, 0.0)
        gap_modes = np.maximum(gap_modes, 0.0)
        row_modes = np.maximum(row_modes, 0.0)

        clean_centered[step + 1] = float(np.sum(clean_modes))
        noise_gap[step + 1] = float(np.sum(gap_modes))
        clean_total[step + 1] = (
            spectrum.approximation_floor + clean_centered[step + 1]
        )
        row_mass[step + 1] = float(np.sum(row_modes))

    noisy_centered = clean_centered + noise_gap
    noisy_total = clean_total + noise_gap
    maximum_pointwise_product = float(
        np.max(pointwise_products, initial=0.0)
    )
    minimum_pointwise_margin = 2.0 - maximum_pointwise_product
    maximum_row_mass = float(np.max(row_mass, initial=0.0))
    return ExactTrajectory(
        times=times,
        batches=batches,
        clean_centered=clean_centered,
        noise_gap=noise_gap,
        noisy_centered=noisy_centered,
        clean_total=clean_total,
        noisy_total=noisy_total,
        row_mass=row_mass,
        maximum_pointwise_product=maximum_pointwise_product,
        minimum_pointwise_margin=minimum_pointwise_margin,
        maximum_row_mass=maximum_row_mass,
        pointwise_stable=minimum_pointwise_margin > 0.0,
        row_stable=maximum_row_mass < row_mass_cap,
    )


def solve_two_time_volterra(
    spectrum: EmpiricalSpectrum,
    eta: float,
    theta: float,
    initial_batch: int,
    sigma2: float,
    horizon: int,
) -> VolterraTrajectory:
    """Build and solve the exact nonstationary Volterra system explicitly.

    This O(horizon^2 * width) implementation is intentionally independent of
    the O(horizon * width) modal solver and is only for small validation cases.
    """

    eigenvalues = np.asarray(spectrum.eigenvalues, dtype=float)
    overlaps = np.asarray(spectrum.teacher_overlaps, dtype=float)
    squared_eigenvalues = eigenvalues * eigenvalues
    _, batches = batch_schedule(eta, theta, initial_batch, horizon)
    q_steps = np.stack(
        [one_step_polynomial(eigenvalues, eta, int(batch)) for batch in batches],
        axis=0,
    ) if horizon else np.empty((0, eigenvalues.size), dtype=float)

    forcing = np.empty(horizon + 1, dtype=float)
    forcing[0] = spectrum.approximation_floor + float(np.sum(overlaps))
    propagated_initial = np.ones_like(eigenvalues)
    for time in range(1, horizon + 1):
        propagated_initial *= q_steps[time - 1]
        forcing[time] = spectrum.approximation_floor + float(
            overlaps @ propagated_initial
        )

    kernel = np.zeros((horizon + 1, horizon + 1), dtype=float)
    for source in range(horizon):
        survival = np.ones_like(eigenvalues)
        coefficient = eta * eta / int(batches[source])
        for time in range(source + 1, horizon + 1):
            kernel[time, source] = coefficient * float(
                squared_eigenvalues @ survival
            )
            if time < horizon:
                survival *= q_steps[time]

    def forward(noise_variance: float) -> Array:
        result = np.empty(horizon + 1, dtype=float)
        result[0] = forcing[0]
        for time in range(1, horizon + 1):
            result[time] = forcing[time] + float(
                kernel[time, :time] @ (result[:time] + noise_variance)
            )
        return result

    clean = forward(0.0)
    noisy = forward(sigma2)
    return VolterraTrajectory(
        forcing=forcing,
        kernel=kernel,
        clean_total=clean,
        noisy_total=noisy,
        noise_gap=noisy - clean,
    )


def dense_covariance_risk(
    spectrum: EmpiricalSpectrum,
    eta: float,
    theta: float,
    initial_batch: int,
    sigma2: float,
    horizon: int,
) -> Array:
    """Independent original-coordinate covariance recursion in a modal basis."""

    eigenvalues = np.concatenate(
        [np.asarray(spectrum.eigenvalues, dtype=float), np.array([0.0])]
    )
    initial_components = np.concatenate(
        [
            np.sqrt(np.asarray(spectrum.teacher_overlaps, dtype=float)),
            np.array([math.sqrt(spectrum.approximation_floor)]),
        ]
    )
    covariance = np.outer(initial_components, initial_components)
    operator = np.diag(eigenvalues)
    operator_squared = np.diag(eigenvalues * eigenvalues)
    _, batches = batch_schedule(eta, theta, initial_batch, horizon)
    risk = np.empty(horizon + 1, dtype=float)
    risk[0] = float(np.trace(covariance))

    for step, batch in enumerate(batches):
        batch_int = int(batch)
        trace = float(np.trace(covariance))
        left = operator @ covariance
        covariance = (
            covariance
            - eta * (left + left.T)
            + eta
            * eta
            * (1.0 + 1.0 / batch_int)
            * (left @ operator)
            + (eta * eta / batch_int)
            * (trace + sigma2)
            * operator_squared
        )
        covariance = 0.5 * (covariance + covariance.T)
        risk[step + 1] = float(np.trace(covariance))
    return risk
