#if !defined(DEXRT_SC_FORMAL_SOLVER_HPP)
#define DEXRT_SC_FORMAL_SOLVER_HPP

#include "Types.hpp"
#include "ScConfig.hpp"
#include <cmath>

struct ScStepResult {
    fp_t I;
    fp_t psi_star; /// dI/d(eta) at the node
    /// The non-local part I - psi_star eta, formed without cancellation (it is
    /// tiny in optically thick cells).
    fp_t I_ext;
    /// The node's own contribution: with I_ext, everything else at the node
    /// follows: I = I_ext + alo S_0, psi_star = alo / chi_0, and one_minus_alo = 1 - psi_star
    /// chi_0 (formed without cancellation).
    fp_t alo;
    fp_t one_minus_alo;
};

/// Coefficients of I = edt I_uw + a S_uw + alo S_0 for the piecewise linear
/// source function, with series expansions where cancellation would hurt in
/// single precision.
struct ScLinearCoeffs {
    fp_t edt;
    fp_t a;
    fp_t alo;
};

YAKL_INLINE ScLinearCoeffs sc_linear_coeffs(fp_t dtau) {
    ScLinearCoeffs c;
    if (std::abs(dtau) < FP(0.05)) {
        // Even with expm1, w0 - dtau * exp(-dtau) cancels at O(dtau^2).
        // The series preserves both O(dtau) source weights in thin cells.
        const fp_t x = dtau;
        c.edt = std::exp(-x);
        c.a = x * (FP(0.5) + x * (-FP(1.0) / FP(3.0) + x * (FP(1.0) / FP(8.0) - x * (FP(1.0) / FP(30.0)))));
        c.alo = x * (FP(0.5) + x * (-FP(1.0) / FP(6.0) + x * (FP(1.0) / FP(24.0) - x * (FP(1.0) / FP(120.0)))));
    } else {
        c.edt = std::exp(-dtau);
        const fp_t w0 = -std::expm1(-dtau);
        // Divide first so the thick limit also handles dtau = infinity.
        c.a = w0 / dtau - c.edt;
        c.alo = w0 - c.a;
    }
    return c;
}

/// Integrate from the upwind point to the node over path length ds
/// (psi_star only when requested, i.e. on the final segment into a node).
template <ScFormalSolver Solver, bool compute_psi=true>
YAKL_INLINE ScStepResult sc_formal_step(fp_t I_uw, fp_t S_uw, fp_t chi_uw, fp_t S0, fp_t chi0, fp_t ds) {
    const fp_t dtau = FP(0.5) * (chi_uw + chi0) * ds;
    ScStepResult r;
    // Negative optical depth amplifies; it must not take the transparent shortcut.
    if (dtau == FP(0.0)) {
        r.I = I_uw;
        r.I_ext = I_uw;
        r.psi_star = FP(0.0);
        r.alo = FP(0.0);
        r.one_minus_alo = FP(1.0);
        return r;
    }
    if constexpr (Solver == ScFormalSolver::PiecewiseConstant) {
        // NOTE(cmo): Piecewise constant: the step crosses the face halfway, the
        // upwind half with the upwind medium, the node half with the node's
        // (each S with its own opacity).
        const fp_t tau1 = FP(0.5) * chi_uw * ds;
        const fp_t tau2 = FP(0.5) * chi0 * ds;
        const fp_t I_face = I_uw * std::exp(-tau1) - S_uw * std::expm1(-tau1);
        const fp_t w2 = -std::expm1(-tau2);
        const fp_t e2 = std::exp(-tau2);
        r.I_ext = I_face * e2;
        r.I = r.I_ext + w2 * S0;
        r.alo = w2;
        r.one_minus_alo = e2;
        if constexpr (compute_psi) {
            r.psi_star = (chi0 != FP(0.0)) ? w2 / chi0 : FP(0.0);
        }
    } else {
        const ScLinearCoeffs c = sc_linear_coeffs(dtau);
        fp_t a = c.a;
        fp_t alo = c.alo;
        if constexpr (Solver == ScFormalSolver::PiecewiseLinearEmissivity) {
            // NOTE(cmo): The weights of the end source functions scaled by
            // their own opacities (renormalised): the standard step when
            // chi_uw == chi0, and the exact emission for linear eta in the
            // thin limit, so each S is paired with its own opacity.
            const fp_t wa = c.a * chi_uw;
            const fp_t wb = c.alo * chi0;
            const fp_t w0 = c.a + c.alo;
            a = (wa + wb != FP(0.0)) ? w0 * wa / (wa + wb) : FP(0.5) * w0;
            alo = (wa + wb != FP(0.0)) ? w0 * wb / (wa + wb) : FP(0.5) * w0;
        }
        r.I_ext = c.edt * I_uw + a * S_uw;
        const fp_t one_minus_alo = c.edt + a;
        r.I = r.I_ext + alo * S0;
        r.alo = alo;
        r.one_minus_alo = one_minus_alo;
        if constexpr (compute_psi) {
            r.psi_star = (chi0 != FP(0.0)) ? alo / chi0 : FP(0.0);
        }
    }
    return r;
}

#else
#endif
