"""Frozen configurations for the discrete/continuum/true-SGD bridge."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class BridgeProfile:
    name: str
    widths: tuple[int, ...]
    theta_values: tuple[float, ...]
    alpha: float
    beta: float
    eta: float
    initial_batch: int
    sigma2: float
    horizon_exponent: float
    fit_lower_exponent: float
    fit_upper_exponent: float
    trajectories: int
    seed: int
    row_mass_cap: float
    maximum_monte_carlo_relative_l2_error: float
    maximum_continuum_relative_l2_error: float
    maximum_monte_carlo_slope_error: float
    maximum_continuum_slope_error: float


COMMON = dict(
    theta_values=(0.25, 0.50, 0.90),
    alpha=0.4,
    beta=0.3,
    # eta/B_0 = 1/128, matching the inverse-ratio amplitude of the scalable
    # experiment without width- or phase-dependent calibration.
    eta=1.0 / 8.0,
    initial_batch=16,
    sigma2=10.0,
    horizon_exponent=0.50,
    fit_lower_exponent=0.20,
    fit_upper_exponent=0.45,
    seed=20270819,
    row_mass_cap=0.20,
    maximum_monte_carlo_relative_l2_error=0.10,
    maximum_continuum_relative_l2_error=0.05,
    maximum_monte_carlo_slope_error=0.08,
    maximum_continuum_slope_error=0.05,
)


BRIDGE_SMOKE_PROFILE = BridgeProfile(
    name="bridge_smoke",
    widths=(32, 64),
    trajectories=64,
    **COMMON,
)


BRIDGE_MAIN_PROFILE = BridgeProfile(
    name="bridge_main",
    widths=(256, 512, 1024, 2048),
    trajectories=256,
    **COMMON,
)


BRIDGE_LARGE_PROFILE = BridgeProfile(
    name="bridge_large",
    widths=(4096, 10_000),
    # At these widths every stochastic trajectory contains up to 10,000
    # modal coordinates.  The longer fit window offsets the reduced ensemble
    # size; Monte Carlo standard errors remain explicit in every artifact.
    trajectories=128,
    **COMMON,
)


BRIDGE_EXPERIMENT_ONE_M10000_PROFILE = BridgeProfile(
    name="bridge_experiment_one_m10000",
    widths=(10_000,),
    trajectories=128,
    **{
        **COMMON,
        "theta_values": (0.25, 0.35, 0.50, 0.65, 0.75, 0.90, 1.00, 1.10),
    },
)


BRIDGE_FINITE_SIZE_TRAIN_PROFILE = BridgeProfile(
    name="bridge_finite_size_train",
    widths=(512, 1024, 2048, 4096),
    trajectories=128,
    **{
        **COMMON,
        "theta_values": (0.25, 0.35, 0.50, 0.65, 0.75, 0.90, 1.00, 1.10),
    },
)


BRIDGE_NOISE_SWEEP_UNIT_PROFILE = BridgeProfile(
    name="bridge_noise_sweep_unit",
    widths=(10_000,),
    trajectories=128,
    **{
        **COMMON,
        "theta_values": (0.25, 0.50, 0.90),
        "sigma2": 1.0,
    },
)


BRIDGE_PROFILES = {
    profile.name.removeprefix("bridge_"): profile
    for profile in (
        BRIDGE_SMOKE_PROFILE,
        BRIDGE_MAIN_PROFILE,
        BRIDGE_LARGE_PROFILE,
        BRIDGE_EXPERIMENT_ONE_M10000_PROFILE,
        BRIDGE_FINITE_SIZE_TRAIN_PROFILE,
        BRIDGE_NOISE_SWEEP_UNIT_PROFILE,
    )
}
