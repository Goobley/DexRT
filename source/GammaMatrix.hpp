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

struct GammaTerms {
    GammaFp ij;
    GammaFp ji;
};

YAKL_INLINE GammaTerms gamma_terms(const UV& uv, fp_t wlamu, fp_t I_eff, fp_t one_m_psi_chi) {
    // I_eff = I - psi* eta_trans and one_m_psi_chi = 1 - psi* chi_trans.
    fp_t integrand = one_m_psi_chi * uv.Uji + uv.Vji * I_eff;
    const GammaFp G_ij = GammaFp(integrand) * GammaFp(wlamu);
    integrand = uv.Vij * I_eff;
    const GammaFp G_ji = GammaFp(integrand) * GammaFp(wlamu);
    return GammaTerms{.ij = G_ij, .ji = G_ji};
}

template <bool atomic>
YAKL_INLINE void add_gamma_terms(const GammaAccumState& args, fp_t I_eff, fp_t one_m_psi_chi) {
    JasUnpack(args, uv, wlamu, Gamma, i, j, k);
    const GammaTerms terms = gamma_terms(uv, wlamu, I_eff, one_m_psi_chi);
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

template <bool atomic=true>
YAKL_INLINE void add_to_gamma(const GammaAccumState& args) {
    JasUnpack(args, eta, chi, I, psi_star);
    add_gamma_terms<atomic>(args, I - psi_star * eta, FP(1.0) - chi * psi_star);
}

/// add_to_gamma from the parts of I - psi* eta and 1 - psi* chi that don't
/// cancel in optically thick cells: I_ext = I - psi* eta_total, one_minus_alo =
/// 1 - psi* chi_total, and the emissivity/opacity of everything other than this
/// transition, eta_other/chi_other (summed, not differenced). args.eta, chi,
/// I are unused.
template <bool atomic=true>
YAKL_INLINE void add_to_gamma_split(
    const GammaAccumState& args,
    fp_t I_ext,
    fp_t one_minus_alo,
    fp_t eta_other,
    fp_t chi_other
) {
    const fp_t psi_star = args.psi_star;
    add_gamma_terms<atomic>(args, I_ext + psi_star * eta_other, one_minus_alo + psi_star * chi_other);
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