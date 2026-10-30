#ifndef DEXRT_SC_LONGER_INTEGRATION_HPP
#define DEXRT_SC_LONGER_INTEGRATION_HPP
#include "Types.hpp"
#include <cmath>

/// Cell-mean response of a constant-medium segment:
/// I_mean = one_minus_alo * I_in + alo * S, with
/// one_minus_alo = (1 - exp(-tau)) / tau. Both terms are evaluated stably.
/// These describe the path mean, not the outgoing-intensity transmission
/// exp(-tau); the point-intensity update is in sc_integrate_longer_segment.
struct ScLongerResponse {
    fp_t alo;
    fp_t one_minus_alo;
};

YAKL_INLINE ScLongerResponse sc_longer_response(fp_t tau) {
    ScLongerResponse c;
    if (std::abs(tau) < FP(0.1)) {
        // expm1 preserves w0 = 1 - exp(-tau), but tau - w0 still
        // cancels at O(tau^2). Keep the series to recover the small ALO.
        // alo = tau/2 - tau^2/6 + tau^3/24 - tau^4/120 + tau^5/720
        c.alo = tau * (FP(0.5) - tau * (FP(1.0) / FP(6.0) -
                                        tau * (FP(1.0) / FP(24.0) -
                                               tau * (FP(1.0) / FP(120.0) - tau * FP(1.0) / FP(720.0)))));
        c.one_minus_alo = FP(1.0) - c.alo;
    } else {
        const fp_t w0 = -std::expm1(-tau);
        c.one_minus_alo = w0 / tau;
        c.alo = (tau - w0) / tau;
    }
    return c;
}

/// Pure formal integration of one constant-medium segment. The split terms
/// are weighted contributions to the cell mean; I and delta_I are unweighted.
/// No traversal, cell indexing or buffer writes belong to this operation.
struct ScLongerStepResult {
    fp_t I;
    fp_t I_ext;
    fp_t alo;
    fp_t one_minus_alo;
    fp_t delta_I;
};

YAKL_INLINE ScLongerStepResult sc_integrate_longer_segment(fp_t I_in, fp_t source, fp_t tau, fp_t cell_weight) {
    const auto coefficients = sc_longer_response(tau);
    const fp_t delta_I = -(I_in - source) * coefficients.one_minus_alo * tau;
    return {
        I_in + delta_I,
        I_in * cell_weight * coefficients.one_minus_alo,
        cell_weight * coefficients.alo,
        cell_weight * coefficients.one_minus_alo,
        delta_I,
    };
}

#endif
