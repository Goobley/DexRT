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
    fp_t psi_star;
    fp_t wlamu;
    const GammaMat& Gamma;
    int i;
    int j;
    int64_t k;
};

/// Add the RH92 terms for transition (i, j): Gamma(i, j) gets local * Uji +
/// Vji * I_eff and Gamma(j, i) gets Vij * I_eff, weighted by wlamu, with
/// I_eff = I - psi* eta and local = 1 - psi* chi.
struct GammaTerms {
    GammaFp ij;
    GammaFp ji;
};

/// add_gamma_terms' increments of Gamma(i, j) and Gamma(j, i).
YAKL_INLINE GammaTerms gamma_terms(const UV& uv, fp_t wlamu, fp_t I_eff, fp_t local) {
    fp_t integrand = local * uv.Uji + uv.Vji * I_eff;
    const GammaFp G_ij = GammaFp(integrand) * GammaFp(wlamu);
    integrand = uv.Vij * I_eff;
    const GammaFp G_ji = GammaFp(integrand) * GammaFp(wlamu);
    return GammaTerms{.ij = G_ij, .ji = G_ji};
}

template <bool atomic>
YAKL_INLINE void add_gamma_terms(const GammaAccumState& args, fp_t I_eff, fp_t local) {
    JasUnpack(args, uv, wlamu, Gamma, i, j, k);
    const GammaTerms terms = gamma_terms(uv, wlamu, I_eff, local);
    const GammaFp G_ij = terms.ij;
    const GammaFp G_ji = terms.ji;
    if constexpr (atomic) {
        Kokkos::atomic_add(&Gamma(i, j, k), G_ij);
        Kokkos::atomic_add(&Gamma(j, i, k), G_ji);
    } else {
        Gamma(i, j, k) += G_ij;
        Gamma(j, i, k) += G_ji;
    }
}

/**
 * Add the partial integration of Gamma following the RH92 same-preconditioning approach
*/
template <bool atomic=true>
YAKL_INLINE void add_to_gamma(const GammaAccumState& args) {
    JasUnpack(args, eta, chi, I, psi_star);
    add_gamma_terms<atomic>(args, I - psi_star * eta, FP(1.0) - chi * psi_star);
}

/// add_to_gamma from the parts of I - psi* eta and 1 - psi* chi that don't
/// cancel in optically thick cells: I_nl = I - psi* eta_total, q = 1 - psi*
/// chi_total, and the emissivity/opacity of everything other than this
/// transition, eta_other/chi_other (summed, not differenced). args.eta, chi,
/// I are unused.
template <bool atomic=true>
YAKL_INLINE void add_to_gamma_split(
    const GammaAccumState& args,
    fp_t I_nl,
    fp_t q,
    fp_t eta_other,
    fp_t chi_other
) {
    const fp_t psi_star = args.psi_star;
    add_gamma_terms<atomic>(args, I_nl + psi_star * eta_other, q + psi_star * chi_other);
}

template <typename T>
inline void fixup_gamma(const yakl::Array<T, 3, yakl::memDevice>& Gamma) {
    dex_parallel_for(
        "Gamma fixup",
        FlatLoop<1>(Gamma.extent(2)),
        YAKL_LAMBDA (i64 k) {
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