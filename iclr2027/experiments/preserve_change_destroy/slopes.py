"""Preregistered power-law slope diagnostics without theory-guided window search."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np


Array = np.ndarray


@dataclass(frozen=True)
class FitAudit:
    observable: str
    lower_time: float
    upper_time: float
    point_count: int
    span_decades: float
    exponent: float
    intercept: float
    r_squared: float
    local_slope_median: float
    local_slope_variation: float
    status: str
    reason: str

    def to_dict(self) -> dict[str, float | int | str]:
        return asdict(self)


def local_log_slopes(
    times: Array,
    values: Array,
    valid: Array | None = None,
    half_window_decades: float = 0.25,
    minimum_points: int = 7,
) -> Array:
    """Estimate -d log(value) / d log(time) on fixed local neighborhoods."""

    times = np.asarray(times, dtype=float)
    values = np.asarray(values, dtype=float)
    if times.shape != values.shape:
        raise ValueError("times and values must have the same shape")
    if valid is None:
        valid = np.ones_like(times, dtype=bool)
    else:
        valid = np.asarray(valid, dtype=bool)
    valid = valid & (times > 0.0) & (values > 0.0) & np.isfinite(values)
    slopes = np.full_like(times, np.nan, dtype=float)
    if not np.any(valid):
        return slopes
    log_time = np.full_like(times, np.nan, dtype=float)
    log_value = np.full_like(values, np.nan, dtype=float)
    log_time[valid] = np.log10(times[valid])
    log_value[valid] = np.log10(values[valid])
    for index in np.flatnonzero(valid):
        neighborhood = (
            valid
            & (np.abs(log_time - log_time[index]) <= half_window_decades)
        )
        if int(np.count_nonzero(neighborhood)) >= minimum_points:
            slopes[index] = -float(
                np.polyfit(log_time[neighborhood], log_value[neighborhood], 1)[0]
            )
    return slopes


def audit_fixed_window(
    times: Array,
    values: Array,
    observable: str,
    lower_time: float,
    upper_time: float,
    minimum_decades: float,
    minimum_points: int,
    half_window_decades: float,
    maximum_local_variation: float,
) -> FitAudit:
    """Fit exactly the declared window and fail closed when it is not stable."""

    times = np.asarray(times, dtype=float)
    values = np.asarray(values, dtype=float)
    mask = (
        (times >= lower_time)
        & (times <= upper_time)
        & (times > 0.0)
        & (values > 0.0)
        & np.isfinite(values)
    )
    point_count = int(np.count_nonzero(mask))
    if point_count:
        selected_times = times[mask]
        span_decades = float(
            np.log10(selected_times[-1] / selected_times[0])
        )
    else:
        span_decades = 0.0

    if point_count < minimum_points:
        return _inconclusive(
            observable,
            lower_time,
            upper_time,
            point_count,
            span_decades,
            "too_few_points",
        )
    if span_decades < minimum_decades:
        return _inconclusive(
            observable,
            lower_time,
            upper_time,
            point_count,
            span_decades,
            "insufficient_decade_span",
        )

    log_time = np.log10(times[mask])
    log_value = np.log10(values[mask])
    slope, intercept = np.polyfit(log_time, log_value, 1)
    fitted = slope * log_time + intercept
    residual_sum = float(np.sum((log_value - fitted) ** 2))
    centered_sum = float(np.sum((log_value - np.mean(log_value)) ** 2))
    r_squared = 1.0 if centered_sum == 0.0 else 1.0 - residual_sum / centered_sum

    local = local_log_slopes(
        times,
        values,
        valid=mask,
        half_window_decades=half_window_decades,
        minimum_points=max(5, min(9, minimum_points // 2)),
    )
    local_values = local[mask & np.isfinite(local)]
    if local_values.size < 3:
        return FitAudit(
            observable=observable,
            lower_time=lower_time,
            upper_time=upper_time,
            point_count=point_count,
            span_decades=span_decades,
            exponent=-float(slope),
            intercept=float(intercept),
            r_squared=r_squared,
            local_slope_median=float("nan"),
            local_slope_variation=float("inf"),
            status="INCONCLUSIVE",
            reason="too_few_local_slopes",
        )

    median = float(np.median(local_values))
    variation = float(
        (np.quantile(local_values, 0.9) - np.quantile(local_values, 0.1))
        / max(abs(median), 5.0e-2)
    )
    status = "PASS" if variation <= maximum_local_variation else "INCONCLUSIVE"
    reason = "stable_preregistered_window" if status == "PASS" else "local_slope_unstable"
    return FitAudit(
        observable=observable,
        lower_time=lower_time,
        upper_time=upper_time,
        point_count=point_count,
        span_decades=span_decades,
        exponent=-float(slope),
        intercept=float(intercept),
        r_squared=r_squared,
        local_slope_median=median,
        local_slope_variation=variation,
        status=status,
        reason=reason,
    )

def _inconclusive(
    observable: str,
    lower_time: float,
    upper_time: float,
    point_count: int,
    span_decades: float,
    reason: str,
) -> FitAudit:
    return FitAudit(
        observable=observable,
        lower_time=lower_time,
        upper_time=upper_time,
        point_count=point_count,
        span_decades=span_decades,
        exponent=float("nan"),
        intercept=float("nan"),
        r_squared=float("nan"),
        local_slope_median=float("nan"),
        local_slope_variation=float("inf"),
        status="INCONCLUSIVE",
        reason=reason,
    )
