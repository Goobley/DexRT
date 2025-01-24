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
    const GammaMat& Gamma;
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

    fp_t integrand = (FP(1.0) - alo) * uv.Uji + uv.Vji * I_eff;
    GammaFp G_ij = GammaFp(integrand) * GammaFp(wlamu);

    integrand = uv.Vij * I_eff;
    GammaFp G_ji = GammaFp(integrand) * GammaFp(wlamu);
    if constexpr (atomic) {
        Kokkos::atomic_add(&Gamma(i, j, k), G_ij);
        Kokkos::atomic_add(&Gamma(j, i, k), G_ji);
    } else {
        Gamma(i, j, k) += G_ij;
        Gamma(j, i, k) += G_ji;
    }
}

template <typename T>
inline void fixup_gamma(const Kokkos::View<T***>& Gamma) {
    dex_parallel_for(
        "Gamma fixup",
        FlatLoop<1>(Gamma.extent(2)),
        KOKKOS_LAMBDA (i64 k) {
            for (int i = 0; i < Gamma.extent(1); ++i) {
                T diag = FP(0.0);
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