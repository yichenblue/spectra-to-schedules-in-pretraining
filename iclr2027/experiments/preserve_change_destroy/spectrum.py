"""Finite PLRF empirical spectrum in the current paper normalization."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np


Array = np.ndarray


@dataclass(frozen=True)
class EmpiricalSpectrum:
    """Positive spectrum and teacher overlaps conditional on frozen features."""

    alpha: float
    beta: float
    ambient_dimension: int
    width: int
    seed: int
    eigenvalues: Array
    teacher_overlaps: Array
    approximation_floor: float
    teacher_energy: float
    trace: float
    lambda_max: float
    rank_tolerance: float
    parseval_residual: float


@dataclass(frozen=True)
class GlobalLearningRate:
    """One learning rate shared by every width and feature realization."""

    eta: float
    maximum_lambda: float
    maximum_trace: float
    pointwise_product: float
    pointwise_margin: float
    row_mass_certificate: float
    limiting_constraint: str


def build_empirical_spectrum(
    alpha: float,
    beta: float,
    width: int,
    ambient_to_width_ratio: int,
    seed: int,
) -> EmpiricalSpectrum:
    """Construct the exact current-paper finite PLRF object.

    Current notation is used throughout: the ambient dimension is ``d`` and
    the feature width is ``m``.  Thus W has shape d x m and entries N(0,1/m).
    The nonzero spectrum is evaluated through the smaller Gram matrix
    A.T @ A, where A = Lambda^(1/2) W.
    """

    if alpha <= 0.0 or beta <= 0.0:
        raise ValueError("alpha and beta must be positive")
    if width <= 0 or ambient_to_width_ratio <= 1:
        raise ValueError("require width > 0 and ambient_to_width_ratio > 1")

    ambient_dimension = int(ambient_to_width_ratio * width)
    indices = np.arange(1, ambient_dimension + 1, dtype=float)
    population_eigenvalues = indices ** (-2.0 * alpha)
    teacher = indices ** (-beta)
    weighted_teacher = np.sqrt(population_eigenvalues) * teacher

    rng = np.random.default_rng(seed)
    random_features = rng.standard_normal((ambient_dimension, width))
    random_features /= math.sqrt(width)
    weighted_features = np.sqrt(population_eigenvalues)[:, None] * random_features

    gram = weighted_features.T @ weighted_features
    gram = 0.5 * (gram + gram.T)
    eigenvalues, eigenvectors = np.linalg.eigh(gram)
    scale = max(1.0, float(eigenvalues[-1]))
    rank_tolerance = 256.0 * np.finfo(float).eps * scale * width
    positive = eigenvalues > rank_tolerance
    if int(np.count_nonzero(positive)) != width:
        raise RuntimeError(
            f"expected empirical rank {width}, found {int(np.count_nonzero(positive))}"
        )

    eigenvalues = eigenvalues[positive]
    eigenvectors = eigenvectors[:, positive]
    target_projection = weighted_features.T @ weighted_teacher
    teacher_overlaps = (
        (eigenvectors.T @ target_projection) ** 2 / eigenvalues
    )
    teacher_energy = float(weighted_teacher @ weighted_teacher)
    raw_floor = teacher_energy - float(np.sum(teacher_overlaps))
    floor_tolerance = 2.0e-10 * max(1.0, teacher_energy)
    if raw_floor < -floor_tolerance:
        raise RuntimeError(
            "teacher overlap decomposition violates Parseval beyond tolerance: "
            f"floor={raw_floor:.3e}, tolerance={floor_tolerance:.3e}"
        )
    approximation_floor = max(0.0, raw_floor)
    parseval_residual = abs(
        teacher_energy
        - approximation_floor
        - float(np.sum(teacher_overlaps))
    )

    return EmpiricalSpectrum(
        alpha=alpha,
        beta=beta,
        ambient_dimension=ambient_dimension,
        width=width,
        seed=seed,
        eigenvalues=eigenvalues,
        teacher_overlaps=teacher_overlaps,
        approximation_floor=approximation_floor,
        teacher_energy=teacher_energy,
        trace=float(np.sum(eigenvalues)),
        lambda_max=float(eigenvalues[-1]),
        rank_tolerance=rank_tolerance,
        parseval_residual=parseval_residual,
    )


def select_global_learning_rate(
    spectra: list[EmpiricalSpectrum],
    initial_batch: int,
    pointwise_product_cap: float,
    row_certificate_cap: float,
) -> GlobalLearningRate:
    """Choose one conservative eta from the worst frozen-feature realization."""

    if not spectra:
        raise ValueError("at least one spectrum is required")
    if initial_batch <= 0:
        raise ValueError("initial_batch must be positive")
    if not (0.0 < pointwise_product_cap < 2.0):
        raise ValueError("pointwise_product_cap must lie in (0,2)")
    if not (0.0 < row_certificate_cap < 1.0):
        raise ValueError("row_certificate_cap must lie in (0,1)")

    maximum_lambda = max(item.lambda_max for item in spectra)
    maximum_trace = max(item.trace for item in spectra)
    spectral_eta = pointwise_product_cap / (
        (1.0 + 1.0 / initial_batch) * maximum_lambda
    )
    row_eta = row_certificate_cap * initial_batch / maximum_trace
    eta = min(spectral_eta, row_eta)
    limiting_constraint = "pointwise" if spectral_eta <= row_eta else "row_certificate"
    pointwise_product = (
        eta * (1.0 + 1.0 / initial_batch) * maximum_lambda
    )
    pointwise_margin = 2.0 - pointwise_product
    row_mass_certificate = (
        maximum_trace * eta / (initial_batch * pointwise_margin)
    )
    if pointwise_margin <= 0.0 or row_mass_certificate >= 1.0:
        raise RuntimeError("global learning-rate construction failed its own gates")

    return GlobalLearningRate(
        eta=eta,
        maximum_lambda=maximum_lambda,
        maximum_trace=maximum_trace,
        pointwise_product=pointwise_product,
        pointwise_margin=pointwise_margin,
        row_mass_certificate=row_mass_certificate,
        limiting_constraint=limiting_constraint,
    )
