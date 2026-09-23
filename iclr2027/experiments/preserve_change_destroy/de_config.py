"""Profiles for the scalable head-preserving spectral-quadrature backend."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DEProfile:
    """Completely explicit configuration for a triangular-limit run."""

    name: str
    alpha: float
    beta: float
    effective_widths: tuple[int, ...]
    theta_values: tuple[float, ...]
    representative_thetas: tuple[float, ...]
    sigma2: float
    inverse_ratio_amplitude: float
    spectral_relative_bin_width: float
    time_points_per_decade: int
    time_min: float
    horizon_exponent: float
    fit_windows: tuple[tuple[str, float, float], ...]
    nominal_window_name: str
    minimum_fit_decades: float
    minimum_fit_points: int
    local_slope_half_window_decades: float
    maximum_local_slope_variation: float
    maximum_exponent_error: float
    row_mass_cap: float
    maximum_resolution_curve_error: float
    maximum_resolution_slope_error: float

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

    def fit_window(self, name: str) -> tuple[float, float]:
        for candidate, lower, upper in self.fit_windows:
            if candidate == name:
                return lower, upper
        raise KeyError(name)


COMMON = dict(
    alpha=0.4,
    beta=0.3,
    theta_values=(0.25, 0.35, 0.50, 0.65, 0.75, 0.90, 1.00, 1.10),
    representative_thetas=(0.25, 0.50, 0.90),
    # Fixed across every effective width.  This is intentionally different
    # from the finite-W profile's width-dependent calibration.
    sigma2=10.0,
    inverse_ratio_amplitude=1.0 / 128.0,
    time_min=1.0e-3,
    horizon_exponent=0.56,
    fit_windows=(
        ("relaxed", 0.15, 0.53),
        ("nominal", 0.20, 0.50),
        ("strict", 0.25, 0.45),
    ),
    nominal_window_name="nominal",
    minimum_fit_decades=2.0,
    minimum_fit_points=24,
    local_slope_half_window_decades=0.30,
    maximum_local_slope_variation=0.10,
    maximum_exponent_error=0.035,
    row_mass_cap=0.20,
)


DE_SMOKE_PROFILE = DEProfile(
    name="de_smoke",
    effective_widths=(10**4, 10**8),
    spectral_relative_bin_width=2.0e-2,
    time_points_per_decade=18,
    maximum_resolution_curve_error=4.0e-2,
    maximum_resolution_slope_error=2.0e-2,
    **COMMON,
)


DE_MAIN_PROFILE = DEProfile(
    name="de_main",
    effective_widths=(10**18, 10**30, 10**48),
    spectral_relative_bin_width=2.0e-3,
    time_points_per_decade=36,
    maximum_resolution_curve_error=3.0e-3,
    maximum_resolution_slope_error=2.0e-3,
    **COMMON,
)


DE_FINITE_SIZE_PROFILE = DEProfile(
    name="de_finite_size_extrapolation",
    effective_widths=(10**4, 10**6, 10**8, 10**12, 10**18, 10**30, 10**48),
    spectral_relative_bin_width=2.0e-3,
    time_points_per_decade=36,
    maximum_resolution_curve_error=3.0e-3,
    maximum_resolution_slope_error=2.0e-3,
    **COMMON,
)


DE_PROFILES = {
    profile.name.removeprefix("de_"): profile
    for profile in (DE_SMOKE_PROFILE, DE_MAIN_PROFILE, DE_FINITE_SIZE_PROFILE)
}
