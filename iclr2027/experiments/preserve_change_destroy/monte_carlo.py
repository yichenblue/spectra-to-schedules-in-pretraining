"""Paired antithetic true-SGD validation for the exact noisy--clean gap."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np


Array = np.ndarray


@dataclass(frozen=True)
class MonteCarloGap:
    iterations: Array
    mean: Array
    standard_error: Array


@dataclass(frozen=True)
class MonteCarloComparison:
    maximum_standardized_error: float
    fraction_within_three_standard_errors: float
    passed: bool


def paired_antithetic_gap(
    eigenvalues: Array,
    eta: float,
    batches: Array,
    sigma2: float,
    trajectories: int,
    seed: int,
) -> MonteCarloGap:
    """Estimate R_sigma-R_0 by evolving the paired residual difference.

    With common Gaussian mini-batches and antithetic +/- label noise, the
    signal/noise cross term cancels exactly.  In the empirical eigenbasis the
    paired population-risk increment is ||delta_t||^2, so the clean trajectory
    never needs to be simulated or subtracted.
    """

    eigenvalues = np.asarray(eigenvalues, dtype=float)
    batches = np.asarray(batches, dtype=np.int64)
    if eta <= 0.0 or sigma2 < 0.0 or trajectories < 2:
        raise ValueError("require eta>0, sigma2>=0, and at least two trajectories")
    if np.any(batches <= 0):
        raise ValueError("all batches must be positive")

    rng = np.random.default_rng(seed)
    delta = np.zeros((trajectories, eigenvalues.size), dtype=float)
    means = np.empty(batches.size + 1, dtype=float)
    standard_errors = np.empty(batches.size + 1, dtype=float)
    noise_scale = math.sqrt(sigma2)

    def record(index: int) -> None:
        risk = np.einsum("tm,tm->t", delta, delta)
        means[index] = float(np.mean(risk))
        standard_errors[index] = float(
            np.std(risk, ddof=1) / math.sqrt(trajectories)
        )

    record(0)
    for step, batch in enumerate(batches):
        batch_int = int(batch)
        gaussian = rng.standard_normal(
            (trajectories, batch_int, eigenvalues.size)
        )
        label_noise = noise_scale * rng.standard_normal(
            (trajectories, batch_int)
        )
        residual_difference = (
            np.einsum("tbm,tm->tb", gaussian, delta) - label_noise
        )
        update = np.einsum(
            "m,tbm,tb->tm",
            eigenvalues,
            gaussian,
            residual_difference,
        )
        delta -= (eta / batch_int) * update
        record(step + 1)

    return MonteCarloGap(
        iterations=np.arange(batches.size + 1, dtype=np.int64),
        mean=means,
        standard_error=standard_errors,
    )


def compare_to_exact(
    sampled: MonteCarloGap,
    exact_gap: Array,
    maximum_allowed_z: float = 4.0,
) -> MonteCarloComparison:
    exact_gap = np.asarray(exact_gap, dtype=float)
    if exact_gap.shape != sampled.mean.shape:
        raise ValueError("sampled and exact trajectories must have the same shape")
    positive_error = sampled.standard_error > 0.0
    positive_error[0] = False
    standardized = np.zeros_like(exact_gap)
    standardized[positive_error] = (
        np.abs(sampled.mean[positive_error] - exact_gap[positive_error])
        / sampled.standard_error[positive_error]
    )
    maximum = float(np.max(standardized, initial=0.0))
    fraction = float(
        np.mean(standardized[positive_error] <= 3.0)
    ) if np.any(positive_error) else 1.0
    return MonteCarloComparison(
        maximum_standardized_error=maximum,
        fraction_within_three_standard_errors=fraction,
        passed=maximum <= maximum_allowed_z,
    )
