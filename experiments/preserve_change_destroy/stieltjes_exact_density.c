#include <math.h>
#include <stddef.h>
#include <stdlib.h>

typedef struct {
    double re;
    double im;
} complex_pair;

static inline complex_pair reciprocal(complex_pair value) {
    const double denominator =
        value.re * value.re + value.im * value.im;
    complex_pair result = {
        value.re / denominator,
        -value.im / denominator
    };
    return result;
}

static void residual_and_derivative(
    complex_pair m,
    double z_re,
    double z_im,
    const double *restrict covariance,
    size_t v,
    double d,
    complex_pair *residual,
    complex_pair *derivative
) {
    double covariance_sum_re = 0.0;
    double covariance_sum_im = 0.0;
    double covariance_squared_sum_re = 0.0;
    double covariance_squared_sum_im = 0.0;
    for (size_t index = 0; index < v; ++index) {
        const double lambda = covariance[index];
        const double denominator_re = lambda * m.re - z_re;
        const double denominator_im = lambda * m.im - z_im;
        const double denominator_squared =
            denominator_re * denominator_re
            + denominator_im * denominator_im;
        const double inverse_re = denominator_re / denominator_squared;
        const double inverse_im = -denominator_im / denominator_squared;
        covariance_sum_re += lambda * inverse_re;
        covariance_sum_im += lambda * inverse_im;
        const double lambda_squared = lambda * lambda;
        covariance_squared_sum_re += lambda_squared * (
            inverse_re * inverse_re - inverse_im * inverse_im
        );
        covariance_squared_sum_im += lambda_squared * (
            2.0 * inverse_re * inverse_im
        );
    }
    const complex_pair inverse_m = reciprocal(m);
    const double m_squared_re = m.re * m.re - m.im * m.im;
    const double m_squared_im = 2.0 * m.re * m.im;
    const complex_pair inverse_m_squared = reciprocal(
        (complex_pair){m_squared_re, m_squared_im}
    );
    residual->re =
        inverse_m.re - 1.0 - covariance_sum_re / d;
    residual->im =
        inverse_m.im - covariance_sum_im / d;
    derivative->re =
        -inverse_m_squared.re + covariance_squared_sum_re / d;
    derivative->im =
        -inverse_m_squared.im + covariance_squared_sum_im / d;
}

static double residual_only(
    complex_pair m,
    double z_re,
    double z_im,
    const double *restrict covariance,
    size_t v,
    double d
) {
    double covariance_sum_re = 0.0;
    double covariance_sum_im = 0.0;
    for (size_t index = 0; index < v; ++index) {
        const double lambda = covariance[index];
        const double denominator_re = lambda * m.re - z_re;
        const double denominator_im = lambda * m.im - z_im;
        const double denominator_squared =
            denominator_re * denominator_re
            + denominator_im * denominator_im;
        covariance_sum_re +=
            lambda * denominator_re / denominator_squared;
        covariance_sum_im +=
            -lambda * denominator_im / denominator_squared;
    }
    const complex_pair inverse_m = reciprocal(m);
    const double residual_re =
        inverse_m.re - 1.0 - covariance_sum_re / d;
    const double residual_im =
        inverse_m.im - covariance_sum_im / d;
    return hypot(residual_re, residual_im);
}

static int solve_one(
    double z_re,
    double z_im,
    const double *restrict covariance,
    size_t v,
    double d,
    complex_pair initial,
    complex_pair *solution,
    double *final_residual,
    int *iterations
) {
    complex_pair m = initial;
    for (int iteration = 0; iteration < 60; ++iteration) {
        complex_pair residual;
        complex_pair derivative;
        residual_and_derivative(
            m,
            z_re,
            z_im,
            covariance,
            v,
            d,
            &residual,
            &derivative
        );
        const double residual_norm = hypot(residual.re, residual.im);
        if (residual_norm < 1.0e-12) {
            *solution = m;
            *final_residual = residual_norm;
            *iterations = iteration;
            return 0;
        }
        const double derivative_squared =
            derivative.re * derivative.re
            + derivative.im * derivative.im;
        const double step_re = (
            residual.re * derivative.re
            + residual.im * derivative.im
        ) / derivative_squared;
        const double step_im = (
            residual.im * derivative.re
            - residual.re * derivative.im
        ) / derivative_squared;
        double scale = 1.0;
        int accepted = 0;
        for (int line_iteration = 0; line_iteration < 30; ++line_iteration) {
            const complex_pair candidate = {
                m.re - scale * step_re,
                m.im - scale * step_im
            };
            const double candidate_residual = residual_only(
                candidate,
                z_re,
                z_im,
                covariance,
                v,
                d
            );
            if (
                candidate.im <= 1.0e-9
                && candidate_residual < residual_norm
            ) {
                m = candidate;
                accepted = 1;
                break;
            }
            scale *= 0.5;
        }
        if (!accepted) {
            break;
        }
    }

    for (int fixed_iteration = 0; fixed_iteration < 10000; ++fixed_iteration) {
        double covariance_sum_re = 0.0;
        double covariance_sum_im = 0.0;
        for (size_t index = 0; index < v; ++index) {
            const double lambda = covariance[index];
            const double denominator_re = lambda * m.re - z_re;
            const double denominator_im = lambda * m.im - z_im;
            const double denominator_squared =
                denominator_re * denominator_re
                + denominator_im * denominator_im;
            covariance_sum_re +=
                lambda * denominator_re / denominator_squared;
            covariance_sum_im +=
                -lambda * denominator_im / denominator_squared;
        }
        const complex_pair mapped = reciprocal(
            (complex_pair){
                1.0 + covariance_sum_re / d,
                covariance_sum_im / d
            }
        );
        const complex_pair candidate = {
            0.7 * m.re + 0.3 * mapped.re,
            0.7 * m.im + 0.3 * mapped.im
        };
        if (hypot(candidate.re - m.re, candidate.im - m.im) < 1.0e-12) {
            *solution = candidate;
            *final_residual = residual_only(
                candidate,
                z_re,
                z_im,
                covariance,
                v,
                d
            );
            *iterations = 60 + fixed_iteration;
            return 0;
        }
        m = candidate;
    }

    *solution = m;
    *final_residual = residual_only(
        m,
        z_re,
        z_im,
        covariance,
        v,
        d
    );
    *iterations = 10060;
    return 1;
}

int plrf_stieltjes_exact_density(
    const double *restrict covariance,
    const double *restrict target_squared,
    size_t v,
    size_t d,
    const double *restrict x,
    size_t point_count,
    double eta_relative,
    double eta_floor,
    double null_weight,
    double *restrict trace_density,
    double *restrict target_density,
    double *restrict point_residual,
    int *restrict point_iterations
) {
    const double pi = 3.141592653589793238462643383279502884;
    complex_pair m = {1.0, 0.0};
    for (size_t reverse_index = 0; reverse_index < point_count; ++reverse_index) {
        const size_t index = point_count - 1 - reverse_index;
        const double point = x[index];
        const double eta_candidate = eta_relative * point;
        const double eta = eta_candidate > eta_floor
            ? eta_candidate
            : eta_floor;
        double residual = 0.0;
        int iterations = 0;
        const int failed = solve_one(
            point,
            eta,
            covariance,
            v,
            (double)d,
            m,
            &m,
            &residual,
            &iterations
        );
        if (failed) {
            return (int)index + 1;
        }
        point_residual[index] = residual;
        point_iterations[index] = iterations;

        double trace_sum_im = 0.0;
        double target_sum_im = 0.0;
        for (size_t population_index = 0;
             population_index < v;
             ++population_index) {
            const double lambda = covariance[population_index];
            const double denominator_re = lambda * m.re - point;
            const double denominator_im = lambda * m.im - eta;
            const double denominator_squared =
                denominator_re * denominator_re
                + denominator_im * denominator_im;
            const double inverse_im =
                -denominator_im / denominator_squared;
            trace_sum_im += inverse_im;
            target_sum_im +=
                target_squared[population_index] * inverse_im;
        }
        const double zero_lorentzian =
            eta / (pi * (point * point + eta * eta));
        const double trace_value =
            trace_sum_im / pi
            - ((double)v - (double)d) * zero_lorentzian;
        const double target_value =
            target_sum_im / pi
            - null_weight * zero_lorentzian;
        trace_density[index] = trace_value > 0.0 ? trace_value : 0.0;
        target_density[index] = target_value > 0.0 ? target_value : 0.0;
    }
    return 0;
}

int plrf_stieltjes_real_axis_density(
    const double *restrict covariance,
    const double *restrict target_squared,
    size_t v,
    size_t d,
    const double *restrict x,
    size_t point_count,
    double seed_eta_relative,
    double seed_eta_floor,
    int homotopy_levels,
    double *restrict trace_density,
    double *restrict target_density,
    double *restrict point_residual,
    int *restrict point_iterations
) {
    const double pi = 3.141592653589793238462643383279502884;
    double *seed_m_re = malloc(point_count * sizeof(double));
    double *seed_m_im = malloc(point_count * sizeof(double));
    if (seed_m_re == NULL || seed_m_im == NULL) {
        free(seed_m_re);
        free(seed_m_im);
        return -1;
    }

    complex_pair continuation = {1.0, 0.0};
    for (size_t reverse_index = 0; reverse_index < point_count; ++reverse_index) {
        const size_t index = point_count - 1 - reverse_index;
        const double point = x[index];
        const double relative_eta = seed_eta_relative * point;
        const double seed_eta = relative_eta > seed_eta_floor
            ? relative_eta
            : seed_eta_floor;
        double residual = 0.0;
        int iterations = 0;
        const int failed = solve_one(
            point,
            seed_eta,
            covariance,
            v,
            (double)d,
            continuation,
            &continuation,
            &residual,
            &iterations
        );
        if (failed) {
            free(seed_m_re);
            free(seed_m_im);
            return (int)index + 1;
        }
        seed_m_re[index] = continuation.re;
        seed_m_im[index] = continuation.im;
    }

    for (size_t index = 0; index < point_count; ++index) {
        const double point = x[index];
        const double relative_eta = seed_eta_relative * point;
        const double seed_eta = relative_eta > seed_eta_floor
            ? relative_eta
            : seed_eta_floor;
        complex_pair m = {seed_m_re[index], seed_m_im[index]};
        double residual = 0.0;
        int total_iterations = 0;
        for (int level = 1; level <= homotopy_levels; ++level) {
            const double eta = ldexp(seed_eta, -level);
            int iterations = 0;
            const int failed = solve_one(
                point,
                eta,
                covariance,
                v,
                (double)d,
                m,
                &m,
                &residual,
                &iterations
            );
            if (failed) {
                free(seed_m_re);
                free(seed_m_im);
                return (int)index + 1;
            }
            total_iterations += iterations;
        }
        int final_iterations = 0;
        const int final_failed = solve_one(
            point,
            0.0,
            covariance,
            v,
            (double)d,
            m,
            &m,
            &residual,
            &final_iterations
        );
        if (final_failed) {
            free(seed_m_re);
            free(seed_m_im);
            return (int)index + 1;
        }
        total_iterations += final_iterations;
        point_residual[index] = residual;
        point_iterations[index] = total_iterations;

        double trace_sum_im = 0.0;
        double target_sum_im = 0.0;
        for (size_t population_index = 0;
             population_index < v;
             ++population_index) {
            const double lambda = covariance[population_index];
            const double denominator_re = lambda * m.re - point;
            const double denominator_im = lambda * m.im;
            const double denominator_squared =
                denominator_re * denominator_re
                + denominator_im * denominator_im;
            const double inverse_im =
                -denominator_im / denominator_squared;
            trace_sum_im += inverse_im;
            target_sum_im +=
                target_squared[population_index] * inverse_im;
        }
        const double trace_value = trace_sum_im / pi;
        const double target_value = target_sum_im / pi;
        trace_density[index] = trace_value > 0.0 ? trace_value : 0.0;
        target_density[index] = target_value > 0.0 ? target_value : 0.0;
    }

    free(seed_m_re);
    free(seed_m_im);
    return 0;
}
