#if !defined(DEXRT_BACKGROUND_OPACITY_HPP)
#define DEXRT_BACKGROUND_OPACITY_HPP

#include "Types.hpp"
#include "Constants.hpp"
#include "Utils.hpp"
#include "Rad16BitStorage.hpp"
#include <string>


/// Background continuum opacities from the tables of
/// dexrt.background_tables (Lightweaver's BasicBackground, i.e. RH's): LTE
/// bound-free continua of the elements not treated as atoms, H-, H2+, H
/// free-free, H/He Rayleigh and Thomson scattering. Tables are log10 per unit
/// density as functions of (lambda, log10 T, log10 ne), resampled onto the
/// wavelength grid at load. The absorption terms emit chi B_lambda(T), the
/// scattering terms sigma J (J from the previous formal solution).
struct BackgroundOpacity {
    bool enabled = false;
    fp_t log_t0;
    fp_t dlog_t;
    i32 num_t;
    fp_t log_ne0;
    fp_t dlog_ne;
    i32 num_ne;
    Fp3d metals; /// [la, t, ne] per hydrogen nucleus
    Fp3d hminus; /// per n(H I 1s)
    Fp3d h2plus; /// per n(H I 1s) n_p
    Fp3d h_ff; /// per n_p
    Fp3d he_rayleigh; /// per hydrogen nucleus (if He isn't an atom)
    Fp1d h_rayleigh; /// [la] per n(H I 1s)
    /// Snapshot (update_background) of the hydrogen populations [ks].
    Fp1d n1s;
    Fp1d n_p;
    /// J of the previous formal solution for the scattering emissivity, for
    /// wavelengths [J_prev_la_start, + rows) (dexrt_sc: the open wavelength
    /// batch's, set by the DAG).
    Rad16Array J_prev;
    i32 J_prev_la_start = 0;
    bool have_J = false;

    YAKL_INLINE fp_t J_prev_at(i32 la, i64 ks) const {
        return J_prev(la - J_prev_la_start, ks);
    }
};

struct BackgroundTerms {
    fp_t chi_abs; /// [m-1], emits chi_abs B_lambda(T)
    fp_t sca; /// [m-1], emits sca J
};

/// Load the tables at path, leaving out the elements present as atoms in
/// state.adata_host, and allocate the snapshot storage.
template <typename StateT>
void load_background(StateT* state, const std::string& path);

/// Refresh the hydrogen population snapshot (call before a formal solution).
/// The previous J is provided per wavelength batch by the caller (dexrt_sc's
/// DAG), once have_J is set after a complete formal solution; zero until then.
template <typename StateT>
void update_background(StateT* state);

YAKL_INLINE fp_t planck_wavelength(fp_t lambda, fp_t T) {
    using namespace ConstantsFP;
    // NOTE(cmo): [kW / (m2 nm sr)], lambda [nm]
    return twohc2_kW_nm2 / (cube(lambda) * square(lambda) * FP(1e-18)) / (std::exp(hc_k_B_nm / (lambda * T)) - FP(1.0));
}

namespace BackgroundImpl {
    YAKL_INLINE fp_t bilinear(const Fp3d& tab, int la, int it, fp_t ft, int ine, fp_t fne) {
        const fp_t a = (FP(1.0) - ft) * tab(la, it, ine) + ft * tab(la, it + 1, ine);
        const fp_t b = (FP(1.0) - ft) * tab(la, it, ine + 1) + ft * tab(la, it + 1, ine + 1);
        return std::pow(FP(10.0), (FP(1.0) - fne) * a + fne * b);
    }
}

/// Background absorption and scattering opacities at wavelength index la in
/// cell ks.
YAKL_INLINE BackgroundTerms background_terms(const BackgroundOpacity& bg, int la, i64 ks, fp_t T, fp_t ne, fp_t nh) {
    using BackgroundImpl::bilinear;
    constexpr fp_t sigma_thomson = FP(6.6524587321e-29); // [m2]
    fp_t xt = (std::log10(T) - bg.log_t0) / bg.dlog_t;
    xt = std::min(std::max(xt, FP(0.0)), fp_t(bg.num_t - 1) - FP(1e-4));
    const int it = int(xt);
    const fp_t ft = xt - it;
    fp_t xne = (std::log10(ne) - bg.log_ne0) / bg.dlog_ne;
    xne = std::min(std::max(xne, FP(0.0)), fp_t(bg.num_ne - 1) - FP(1e-4));
    const int ine = int(xne);
    const fp_t fne = xne - ine;

    const fp_t n1s = bg.n1s(ks);
    const fp_t n_p = bg.n_p(ks);
    BackgroundTerms result;
    result.chi_abs = nh * bilinear(bg.metals, la, it, ft, ine, fne)
        + n1s * (bilinear(bg.hminus, la, it, ft, ine, fne) + n_p * bilinear(bg.h2plus, la, it, ft, ine, fne))
        + n_p * bilinear(bg.h_ff, la, it, ft, ine, fne);
    result.sca = ne * sigma_thomson + n1s * std::pow(FP(10.0), bg.h_rayleigh(la));
    if (bg.he_rayleigh.initialized()) {
        result.sca += nh * bilinear(bg.he_rayleigh, la, it, ft, ine, fne);
    }
    return result;
}

#else
#endif
