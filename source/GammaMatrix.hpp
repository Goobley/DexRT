#if !defined(DEXRT_GAMMA_MATRIX_HPP)
#define DEXRT_GAMMA_MATRIX_HPP

#include "Types.hpp"
#include "JasPP.hpp"
#include "EmisOpac.hpp"
#include "TransitionIndex.hpp"

/// Add the atom's precon_R (see atom_precon_R) at ks into the dense Gammak(i, j) (team scratch, no
/// diagonal). Ends with a team barrier.
template <typename Team, typename Mat>
KOKKOS_INLINE_FUNCTION void scatter_add_precon_R(
    const Team& team,
    const AtomicData<fp_t>& adata,
    int atom,
    const PreconRMat& precon_R,
    i64 ks,
    const Mat& Gammak
) {
    Kokkos::parallel_for(
        Kokkos::TeamVectorRange(team, precon_R.extent(0)),
        [&] (const int t) {
            const TransitionLevels lv = transition_levels(adata, atom, t);
            using GT = std::remove_cvref_t<decltype(Gammak(0, 0))>;
            Kokkos::atomic_add(&Gammak(lv.j, lv.i), GT(precon_R(t, PRECON_R_UP, ks)));
            Kokkos::atomic_add(&Gammak(lv.i, lv.j), GT(precon_R(t, PRECON_R_DOWN, ks)));
        }
    );
    team.team_barrier();
}

/// Ends with a team barrier.
template <typename Team, typename Mat>
KOKKOS_INLINE_FUNCTION void fixup_gamma(const Team& team, const Mat& Gammak) {
    using GT = std::remove_cvref_t<decltype(Gammak(0, 0))>;
    const int num_level = Gammak.extent(0);
    Kokkos::parallel_for(
        Kokkos::TeamVectorRange(team, num_level),
        [&] (const int i) {
            GT diag = GT(FP(0.0));
            Gammak(i, i) = diag;
            for (int j = 0; j < num_level; ++j) {
                diag += Gammak(j, i);
            }
            Gammak(i, i) = -diag;
        }
    );
    team.team_barrier();
}

inline void zero_precon_R(const PreconRMat& precon_R) {
    precon_R = GammaFp(0.0);
    yakl::fence();
}

struct GammaAccumState {
    fp_t eta;
    fp_t chi;
    UV uv;
    fp_t I;
    fp_t psi_star;
    fp_t wlamu;
    const PreconRMat& R;
    int t;
    int64_t k;
};

struct GammaTerms {
    GammaFp ij;
    GammaFp ji;
};

YAKL_INLINE GammaTerms gamma_terms(const UV& uv, fp_t wlamu, fp_t I_eff, fp_t one_m_psi_chi) {
    // I_eff = I - psi* eta_trans and one_m_psi_chi = 1 - psi* chi_trans.
    // .ij and .ji refer to the indices in the matrix how these were stored
    // previously, i.e. .ij is the down rate
    fp_t integrand = one_m_psi_chi * uv.Uji + uv.Vji * I_eff;
    const GammaFp G_ij = GammaFp(integrand) * GammaFp(wlamu);
    integrand = uv.Vij * I_eff;
    const GammaFp G_ji = GammaFp(integrand) * GammaFp(wlamu);
    return GammaTerms{.ij = G_ij, .ji = G_ji};
}

template <bool atomic>
YAKL_INLINE void add_gamma_terms(const GammaAccumState& args, fp_t I_eff, fp_t one_m_psi_chi) {
    JasUnpack(args, uv, wlamu, R, t, k);
    const GammaTerms terms = gamma_terms(uv, wlamu, I_eff, one_m_psi_chi);
    if constexpr (atomic) {
        Kokkos::atomic_add(&R(t, PRECON_R_DOWN, k), terms.ij);
        Kokkos::atomic_add(&R(t, PRECON_R_UP, k), terms.ji);
    } else {
        R(t, PRECON_R_DOWN, k) += terms.ij;
        R(t, PRECON_R_UP, k) += terms.ji;
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
/// transition _other.
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

#else
#endif