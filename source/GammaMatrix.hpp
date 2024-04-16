#if !defined(DEXRT_GAMMA_MATRIX_HPP)
#define DEXRT_GAMMA_MATRIX_HPP

#include "Types.hpp"
#include "JasPP.hpp"
#include "EmisOpac.hpp"

struct GammaAccumState {
    fp_t eta;
    fp_t chi;
    UV uv;
    fp_t I;
    fp_t alo;
    fp_t wlamu;
    const Fp3d& Gamma;
    int i;
    int j;
    int64_t k;
};

/**
 * Add the partial integration of Gamma following the RH92 same-preconditioning approach
*/
template <bool atomic=true>
YAKL_INLINE void add_to_gamma(const GammaAccumState& args) {
    JasUnpack(args, eta, chi, uv, I, alo, wlamu, Gamma, i, j, k);
    const fp_t psi_star = alo / chi;
    const fp_t I_eff = I - psi_star * eta;

    fp_t integrand = (FP(1.0) - chi * psi_star) * uv.Uji + uv.Vji * I_eff;
    fp_t G_ij = integrand * wlamu;

    integrand = uv.Vij * I_eff;
    fp_t G_ji = integrand * wlamu;
    if constexpr (atomic) {
        yakl::atomicAdd(Gamma(i, j, k), G_ij);
        yakl::atomicAdd(Gamma(j, i, k), G_ji);
    } else {
        Gamma(i, j, k) += G_ij;
        Gamma(j, i, k) += G_ji;
    }
}

inline void fixup_gamma(const Fp3d& Gamma) {
    parallel_for(
        "Gamma fixup",
        SimpleBounds<1>(Gamma.extent(2)),
        YAKL_LAMBDA (i64 k) {
            for (int i = 0; i < Gamma.extent(1); ++i) {
                fp_t diag = FP(0.0);
                Gamma(i, i, k) = FP(0.0);
                for (int j = 0; j < Gamma.extent(0); ++j) {
                    diag += Gamma(j, i, k);
                }
                Gamma(i, i, k) = -diag;
            }
        }
    );
}

#else
#endif