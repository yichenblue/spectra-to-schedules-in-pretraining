"""Authenticity bridge from true Gaussian SGD to the continuum proxy."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
from scipy import special

from .de_quadrature import (
    DETrajectory,
    head_preserving_quadrature,
    run_scheduled_continuum_modal_dynamics,
)
from .dynamics import ExactTrajectory, batch_schedule, run_exact_dynamics
from .spectrum import EmpiricalSpectrum


Array = np.ndarray


@dataclass(frozen=True)
class GaussianSGDTrajectory:
    """Monte Carlo means for population clean, gap, and total risks."""

    times: Array
    batches: Array
    clean: Array
    clean_standard_error: Array
    gap: Array
    gap_standard_error: Array
    total: Array
    total_standard_error: Array
    trajectories: int


@dataclass(frozen=True)
class BridgeCase:
    width: int
    theta: float
    exact: ExactTrajectory
    continuum: DETrajectory
    sampled: GaussianSGDTrajectory
    approximation_floor: float


def analytic_power_law_spectrum(
    alpha: float,
    beta: float,
    width: int,
) -> EmpiricalSpectrum:
    """Return the deterministic truncated spectrum shared by all routes."""

    if alpha <= 0.0 or beta <= 0.0 or width <= 0:
        raise ValueError("alpha, beta, and width must be positive")
    indices = np.arange(1, width + 1, dtype=float)
    eigenvalues = indices ** (-2.0 * alpha)
    target_power = 2.0 * (alpha + beta)
    if target_power <= 1.0:
        raise ValueError("finite teacher energy requires 2(alpha+beta)>1")
    overlaps = indices ** (-target_power)
    floor = float(special.zeta(target_power, float(width) + 1.0))
    teacher_energy = float(special.zeta(target_power, 1.0))
    residual = abs(teacher_energy - floor - float(np.sum(overlaps)))
    return EmpiricalSpectrum(
        alpha=alpha,
        beta=beta,
        ambient_dimension=width,
        width=width,
        seed=-1,
        eigenvalues=eigenvalues,
        teacher_overlaps=overlaps,
        approximation_floor=floor,
        teacher_energy=teacher_energy,
        trace=float(np.sum(eigenvalues)),
        lambda_max=float(np.max(eigenvalues)),
        rank_tolerance=0.0,
        parseval_residual=residual,
    )


def _collapsed_minibatch_step(
    states: Array,
    eigenvalues: Array,
    residual_noise_std: float,
    eta: float,
    batch: int,
    rng: np.random.Generator,
    trajectory_chunk_size: int,
) -> None:
    """Apply one distributionally exact Gaussian mini-batch SGD step.

    For residual vector ``h=[state,-noise_std]`` and a standard Gaussian
    mini-batch X, this samples the first coordinates of ``X.T @ X @ h / B``
    exactly in law.  Rotational invariance reduces the cost from O(B*m) to
    O(m) per trajectory without replacing the stochastic update by its mean.
    """

    if batch <= 0 or eta <= 0.0 or residual_noise_std < 0.0:
        raise ValueError("invalid collapsed mini-batch controls")
    noise_variance = residual_noise_std * residual_noise_std
    for start in range(0, states.shape[0], trajectory_chunk_size):
        stop = min(start + trajectory_chunk_size, states.shape[0])
        state = states[start:stop]
        norm_squared = np.einsum("tm,tm->t", state, state) + noise_variance
        active = norm_squared > 0.0
        if not np.any(active):
            continue
        radius = np.sqrt(norm_squared)
        chi_squared = rng.chisquare(float(batch), size=state.shape[0])
        gaussian = rng.standard_normal(state.shape)
        gaussian_noise = rng.standard_normal(state.shape[0])
        projection = (
            np.einsum("tm,tm->t", gaussian, state)
            - residual_noise_std * gaussian_noise
        ) / norm_squared
        gaussian -= projection[:, None] * state
        gaussian *= (
            radius * np.sqrt(chi_squared) / float(batch)
        )[:, None]
        gaussian += (chi_squared / float(batch))[:, None] * state
        state -= eta * eigenvalues[None, :] * gaussian


def _record_population_risk(
    states: Array,
    irreducible_floor: float,
) -> tuple[float, float]:
    samples = np.einsum("tm,tm->t", states, states) + irreducible_floor
    mean = float(np.mean(samples))
    standard_error = float(np.std(samples, ddof=1) / math.sqrt(samples.size))
    return mean, standard_error


def _simulate_risk_component(
    eigenvalues: Array,
    initial_state: Array,
    residual_noise_variance: float,
    irreducible_floor: float,
    eta: float,
    batches: Array,
    trajectories: int,
    seed_sequence: np.random.SeedSequence,
    trajectory_chunk_size: int,
) -> tuple[Array, Array]:
    if residual_noise_variance < 0.0 or irreducible_floor < 0.0:
        raise ValueError("risk variances must be nonnegative")
    if trajectories < 2:
        raise ValueError("at least two trajectories are required")
    state0 = np.asarray(initial_state, dtype=float)
    states = np.broadcast_to(
        state0, (trajectories, state0.size)
    ).copy()
    means = np.empty(len(batches) + 1, dtype=float)
    standard_errors = np.empty_like(means)
    means[0], standard_errors[0] = _record_population_risk(
        states, irreducible_floor
    )
    rng = np.random.default_rng(seed_sequence)
    noise_std = math.sqrt(residual_noise_variance)
    for index, batch in enumerate(batches):
        _collapsed_minibatch_step(
            states=states,
            eigenvalues=eigenvalues,
            residual_noise_std=noise_std,
            eta=eta,
            batch=int(batch),
            rng=rng,
            trajectory_chunk_size=trajectory_chunk_size,
        )
        means[index + 1], standard_errors[index + 1] = _record_population_risk(
            states, irreducible_floor
        )
    return means, standard_errors


def simulate_true_gaussian_sgd(
    spectrum: EmpiricalSpectrum,
    eta: float,
    batches: Array,
    sigma2: float,
    trajectories: int,
    seed: int,
    trajectory_chunk_size: int = 64,
) -> GaussianSGDTrajectory:
    """Simulate clean and noise-response population risks under true SGD."""

    batches = np.asarray(batches, dtype=np.int64)
    if sigma2 < 0.0 or np.any(batches <= 0):
        raise ValueError("invalid SGD bridge configuration")
    seed_root = np.random.SeedSequence(seed)
    clean_seed, gap_seed = seed_root.spawn(2)
    initial_clean = np.sqrt(np.maximum(spectrum.teacher_overlaps, 0.0))
    clean, clean_se = _simulate_risk_component(
        eigenvalues=spectrum.eigenvalues,
        initial_state=initial_clean,
        residual_noise_variance=spectrum.approximation_floor,
        irreducible_floor=spectrum.approximation_floor,
        eta=eta,
        batches=batches,
        trajectories=trajectories,
        seed_sequence=clean_seed,
        trajectory_chunk_size=trajectory_chunk_size,
    )
    gap, gap_se = _simulate_risk_component(
        eigenvalues=spectrum.eigenvalues,
        initial_state=np.zeros_like(spectrum.eigenvalues),
        residual_noise_variance=sigma2,
        irreducible_floor=0.0,
        eta=eta,
        batches=batches,
        trajectories=trajectories,
        seed_sequence=gap_seed,
        trajectory_chunk_size=trajectory_chunk_size,
    )
    total = clean + gap
    total_se = np.sqrt(clean_se * clean_se + gap_se * gap_se)
    return GaussianSGDTrajectory(
        times=eta * np.arange(batches.size + 1, dtype=float),
        batches=batches.copy(),
        clean=clean,
        clean_standard_error=clean_se,
        gap=gap,
        gap_standard_error=gap_se,
        total=total,
        total_standard_error=total_se,
        trajectories=trajectories,
    )


def run_bridge_case(
    alpha: float,
    beta: float,
    width: int,
    theta: float,
    eta: float,
    initial_batch: int,
    sigma2: float,
    horizon_exponent: float,
    trajectories: int,
    seed: int,
    row_mass_cap: float,
) -> BridgeCase:
    """Run all three routes with identical spectrum, schedule, and time grid."""

    spectrum = analytic_power_law_spectrum(alpha, beta, width)
    maximum_time = float(width) ** horizon_exponent
    horizon = max(1, int(math.ceil(maximum_time / eta)))
    times, batches = batch_schedule(eta, theta, initial_batch, horizon)
    exact = run_exact_dynamics(
        spectrum=spectrum,
        eta=eta,
        theta=theta,
        initial_batch=initial_batch,
        sigma2=sigma2,
        horizon=horizon,
        row_mass_cap=row_mass_cap,
    )
    # relative_bin_width < 1/width makes every mode a singleton, so the
    # bridge changes only time dynamics and never changes the shared spectrum.
    quadrature = head_preserving_quadrature(
        alpha=alpha,
        beta=beta,
        effective_width=width,
        relative_bin_width=0.5 / float(width),
    )
    continuum = run_scheduled_continuum_modal_dynamics(
        quadrature=quadrature,
        times=times,
        batches=batches,
        eta=eta,
        theta=theta,
        sigma2=sigma2,
        row_mass_cap=row_mass_cap,
    )
    sampled = simulate_true_gaussian_sgd(
        spectrum=spectrum,
        eta=eta,
        batches=batches,
        sigma2=sigma2,
        trajectories=trajectories,
        seed=seed,
    )
    return BridgeCase(
        width=width,
        theta=theta,
        exact=exact,
        continuum=continuum,
        sampled=sampled,
        approximation_floor=spectrum.approximation_floor,
    )


def fit_effective_exponent(
    times: Array,
    values: Array,
    lower_time: float,
    upper_time: float,
) -> float:
    times = np.asarray(times, dtype=float)
    values = np.asarray(values, dtype=float)
    mask = (
        (times >= lower_time)
        & (times <= upper_time)
        & (times > 0.0)
        & (values > 0.0)
        & np.isfinite(values)
    )
    if int(np.count_nonzero(mask)) < 3:
        return float("nan")
    slope = np.polyfit(np.log(times[mask]), np.log(values[mask]), 1)[0]
    return -float(slope)


def curve_comparison(
    candidate: Array,
    reference: Array,
    standard_error: Array | None = None,
) -> dict[str, float]:
    candidate = np.asarray(candidate, dtype=float)
    reference = np.asarray(reference, dtype=float)
    if candidate.shape != reference.shape:
        raise ValueError("candidate and reference curves must have the same shape")
    valid = (
        np.isfinite(candidate)
        & np.isfinite(reference)
        & (candidate > 0.0)
        & (reference > 0.0)
    )
    if not np.any(valid):
        return {
            "relative_l2_error": float("nan"),
            "log_rmse": float("nan"),
            "maximum_standardized_error": float("nan"),
            "fraction_within_three_standard_errors": float("nan"),
        }
    difference = candidate[valid] - reference[valid]
    relative_l2 = float(
        np.linalg.norm(difference) / np.linalg.norm(reference[valid])
    )
    log_rmse = float(
        np.sqrt(np.mean(np.log(candidate[valid] / reference[valid]) ** 2))
    )
    maximum_z = float("nan")
    fraction_three = float("nan")
    if standard_error is not None:
        error = np.asarray(standard_error, dtype=float)
        if error.shape != reference.shape:
            raise ValueError("standard errors must match the reference curve")
        z_valid = valid & (error > 0.0) & np.isfinite(error)
        if np.any(z_valid):
            z = np.abs(candidate[z_valid] - reference[z_valid]) / error[z_valid]
            maximum_z = float(np.max(z))
            fraction_three = float(np.mean(z <= 3.0))
    return {
        "relative_l2_error": relative_l2,
        "log_rmse": log_rmse,
        "maximum_standardized_error": maximum_z,
        "fraction_within_three_standard_errors": fraction_three,
    }
