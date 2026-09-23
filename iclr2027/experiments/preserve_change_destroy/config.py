"""Configuration and theory values for the preserve--change--destroy run."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ExperimentProfile:
    """A completely explicit experiment profile.

    The smoke profile exercises every code path.  It is not intended to
    establish an asymptotic slope.  The main profile is the preregistered
    paper-facing sweep.
    """

    name: str
    alpha: float
    beta: float
    ambient_to_width_ratio: int
    widths: tuple[int, ...]
    seeds: tuple[int, ...]
    theta_values: tuple[float, ...]
    representative_thetas: tuple[float, ...]
    initial_batch: int
    horizon_factor: float
    fit_time_min: float
    nominal_fit_upper_factor: float
    fit_upper_sensitivity: tuple[float, ...]
    minimum_fit_decades: float
    minimum_fit_points: int
    local_slope_half_window_decades: float
    maximum_local_slope_variation: float
    pointwise_product_cap: float
    row_certificate_cap: float
    empirical_row_mass_cap: float
    calibration_theta: float
    monte_carlo_width: int
    monte_carlo_steps: int
    monte_carlo_trajectories: int

    @property
    def q_clean(self) -> float:
        return (2.0 * self.alpha + 2.0 * self.beta - 1.0) / (
            2.0 * self.alpha
        )

    @property
    def q_kernel(self) -> float:
        return 2.0 - 1.0 / (2.0 * self.alpha)

    @property
    def destroy_boundary(self) -> float:
        return 1.0 - self.q_kernel

    @property
    def preservation_boundary(self) -> float:
        return self.destroy_boundary + self.q_clean

    def noise_exponent(self, theta: float) -> float:
        """The predicted label-noise exponent on the studied boundary/right side."""

        if theta <= self.destroy_boundary:
            return 0.0
        if theta < 1.0:
            return theta + self.q_kernel - 1.0
        return self.q_kernel

    def total_exponent(self, theta: float) -> float:
        return min(self.q_clean, self.noise_exponent(theta))

    def phase_label(self, theta: float, tolerance: float = 1.0e-12) -> str:
        if theta <= self.destroy_boundary + tolerance:
            return "destroy"
        noise = self.noise_exponent(theta)
        if noise < self.q_clean - tolerance:
            return "change"
        if abs(noise - self.q_clean) <= tolerance:
            return "critical"
        return "preserve"


COMMON = dict(
    alpha=0.4,
    beta=0.3,
    ambient_to_width_ratio=2,
    theta_values=(0.25, 0.35, 0.50, 0.65, 0.75, 0.90, 1.00, 1.10),
    representative_thetas=(0.25, 0.50, 0.90),
    initial_batch=16,
    horizon_factor=0.35,
    fit_time_min=10.0,
    nominal_fit_upper_factor=0.20,
    fit_upper_sensitivity=(0.10, 0.20, 0.30),
    minimum_fit_decades=1.0,
    minimum_fit_points=12,
    local_slope_half_window_decades=0.25,
    maximum_local_slope_variation=0.10,
    pointwise_product_cap=0.80,
    row_certificate_cap=0.20,
    empirical_row_mass_cap=0.80,
    calibration_theta=0.75,
)


SMOKE_PROFILE = ExperimentProfile(
    name="smoke",
    widths=(32, 64, 128),
    seeds=(11, 23),
    monte_carlo_width=16,
    monte_carlo_steps=12,
    monte_carlo_trajectories=2048,
    **COMMON,
)


MAIN_PROFILE = ExperimentProfile(
    name="main",
    widths=(512, 1024, 2048, 4096),
    seeds=(11, 23, 37, 53, 71),
    monte_carlo_width=16,
    monte_carlo_steps=24,
    monte_carlo_trajectories=8192,
    **COMMON,
)


PROFILES = {profile.name: profile for profile in (SMOKE_PROFILE, MAIN_PROFILE)}
