#if !defined(DEXRT_EMISOPAC_HPP)
#define DEXRT_EMISOPAC_HPP
#include "Config.hpp"
#include "Constants.hpp"
#include "Types.hpp"
#include "Voigt.hpp"
#include "Populations.hpp"
#include "JasPP.hpp"

struct EmisOpac {
    fp_t eta;
    fp_t chi;
};

enum class EmisOpacMode {
    /// Accumulate background + continua
    StaticOnly,
    /// Accumulate the Doppler shifted lines
    DynamicOnly,
    /// Accumulate everything
    All
};

struct UV {
    fp_t Uji;
    fp_t Vji;
    fp_t Vij;
};

struct UvOptions {
    bool include_hc_4pi = true;
};

struct LineParams {
    /// Doppler width [nm]
    fp_t dop_width;
    /// damping
    fp_t gamma;
    /// projected velocity [m / s]
    fp_t vel;
};

struct AtmosPointParams {
    fp_t temperature = FP(0.0);
    fp_t ne = FP(0.0);
    fp_t vturb = FP(0.0);
    fp_t nhtot = FP(0.0);
    fp_t vel = FP(0.0);
    /// Neutral H density
    fp_t nh0 = FP(0.0);
};

template <typename T=fp_t, typename mem_space=DefaultMemSpace>
struct EmisOpacState {
    const AtomicData<T, mem_space>& adata;
    const VoigtProfile<T, false, mem_space>& profile;
    int la;
    const KView<fp_t**, mem_space>& n;
    const KView<fp_t**, mem_space>& n_star_scratch = {};
    int64_t k;
    const AtmosPointParams& atmos;
    const KView<const u16*, mem_space>& active_set = {};
    const KView<const u16*, mem_space>& active_set_cont = {};
    bool update_n_star = true;
    EmisOpacMode mode = EmisOpacMode::All;
};


template <typename T=fp_t, typename mem_space=DefaultMemSpace>
struct EmisOpacSpecState {
    const AtomicData<T, mem_space>& adata;
    const VoigtProfile<T, false, mem_space>& profile;
    fp_t lambda;
    const KView<fp_t**, mem_space>& n;
    const KView<fp_t**, mem_space>& n_star_scratch;
    int64_t k;
    const AtmosPointParams& atmos;
};

template <typename mem_space=DefaultMemSpace>
struct SigmaInterp {
    KView<const fp_t*, mem_space> sigma;
};

template <typename mem_space=DefaultMemSpace>
struct ContParams {
    /// n_i* / n_j* exp(-h nu / (k_B T)) (gij in RH/Lw)
    fp_t thermal_ratio;
    /// Wavelength index (global wavelength array)
    int la;
    SigmaInterp<mem_space> sigma_grid;
};

template <typename mem_space=DefaultMemSpace>
YAKL_INLINE SigmaInterp<mem_space> get_sigma(const AtomicData<fp_t, mem_space>& adata, const CompCont<fp_t>& cont) {
    SigmaInterp<mem_space> result;

    result.sigma = Kokkos::subview(adata.sigma, Kokkos::make_pair(cont.sigma_start, cont.sigma_end));
    return result;
}

/** Compute mean thermal velocity
 * \param mass [u]
 * \param temperature [K]
 * \return mean thermal velocity [m/s]
*/
YAKL_INLINE fp_t thermal_vel(fp_t mass, fp_t temperature) {
    using namespace ConstantsFP;
    constexpr fp_t two_k_B_u = FP(2.0) * k_B_u;
    return std::sqrt(two_k_B_u * temperature / mass);
}

/** Compute doppler width
 * \param lambda0 wavelength [nm]
 * \param mass [u]
 * \param temperature [K]
 * \param vturb microturbulent velocity [m / s]
 * \return Doppler width [nm]
*/
YAKL_INLINE fp_t doppler_width(fp_t lambda0, fp_t mass, fp_t temperature, fp_t vturb) {
    using namespace ConstantsFP;
    constexpr fp_t two_k_B_u = FP(2.0) * k_B_u;
    return lambda0 * std::sqrt(two_k_B_u * temperature / mass + square(vturb)) / c;
}

/** Compute damping coefficient for Voigt profile from gamma
 * \param gamma [rad / s]
 * \param lambda [nm]
 * \param dop_width [nm]
 * \return gamma / (4 pi dnu_D) = gamma / (4 pi dlambda_D) * lambda**2
*/
YAKL_INLINE fp_t damping_from_gamma(fp_t gamma, fp_t lambda, fp_t dop_width) {
    using namespace ConstantsFP;
    // NOTE(cmo): Extra 1e-9 to convert c to nm / s (all lengths here in nm)
    return FP(1.0) / (four_pi * c) * gamma * FP(1e-9) * square(lambda) / dop_width;
}

/** Compute gamma from all the broadening terms for a line
 * \param line the line in question
 * \param broadeners the array of broadeners for the atom
 * \param temperature
 * \param ne
 * \param nh0
 * \return gamma [rad / s]
*/
template <typename mem_space=DefaultMemSpace>
YAKL_INLINE fp_t gamma_from_broadening(
    const CompLine<fp_t>& line,
    const KView<const ScaledExponentsBroadening<fp_t>*, mem_space>& broadeners,
    fp_t temperature,
    fp_t ne,
    fp_t nh0
) {
    fp_t gamma = line.g_natural;

    auto compute_term = [](fp_t param, fp_t exponent) {
        if (exponent == FP(0.0)) {
            return FP(1.0);
        }
        if (exponent == FP(1.0)) {
            return param;
        }
        return std::pow(param, exponent);
    };

    for (int b_idx = line.broad_start; b_idx < line.broad_end; ++b_idx) {
        const auto& b = broadeners(b_idx);
        fp_t term = (
            compute_term(temperature, b.temperature_exponent) *
            compute_term(ne, b.electron_exponent) *
            compute_term(nh0, b.hydrogen_exponent)
        );
        gamma += term * b.scaling;
    }
    return gamma;
}


template <typename mem_space=DefaultMemSpace>
YAKL_INLINE UV compute_uv(
    const CompLine<fp_t>& l,
    const VoigtProfile<fp_t, false, mem_space>& phi,
    const LineParams& params,
    fp_t lambda,
    const UvOptions& opts = UvOptions()
) {
    using namespace ConstantsFP;
    // [kJ]
    fp_t hnu_4pi = FP(1.0) / lambda;
    if (opts.include_hc_4pi) {
        hnu_4pi *= hc_kJ_nm / four_pi;
    }
    const fp_t a = damping_from_gamma(params.gamma, lambda, params.dop_width);
    const fp_t v = ((lambda - l.lambda0) + (params.vel * l.lambda0) / c) / params.dop_width;
    // [nm-1]
    const fp_t p = phi(a, v) / (sqrt_pi * params.dop_width);
    UV result;
    // [m2]
    result.Vij = hnu_4pi * l.Bij * p;
    // [m2]
    result.Vji = hnu_4pi * l.Bji * p;
    // [kW / (nm sr)]
    result.Uji = hnu_4pi * l.Aji * p;
    return result;
}

template <typename mem_space=DefaultMemSpace>
YAKL_INLINE UV compute_uv(
    const CompCont<fp_t>& cont,
    const ContParams<mem_space>& params,
    fp_t lambda,
    const UvOptions& opts = UvOptions()
) {
    using namespace ConstantsFP;
    UV result;
    // [m2]
    result.Vij = params.sigma_grid.sigma(params.la - cont.blue_idx);
    if (!opts.include_hc_4pi) {
        result.Vij /= (hc_kJ_nm / four_pi);
    }
    // [m2]
    result.Vji = params.thermal_ratio * result.Vij;
    // [kW nm2 / (nm3 m2)] = [kW nm-1] (sr implicit)
    result.Uji = twohc2_kW_nm2 / (cube(lambda) * square(lambda) * FP(1e-18)) * result.Vji;
    return result;
}

template <typename T=fp_t, typename mem_space=DefaultMemSpace>
YAKL_INLINE UV compute_uv_line(
    const EmisOpacState<T, mem_space>& args,
    int line_idx,
    const UvOptions& opts = UvOptions()
) {
    JasUnpack(args, adata, profile, la, k, atmos);
    const fp_t lambda = adata.wavelength(la);
    const auto& l = adata.lines(line_idx);

    LineParams params;
    params.dop_width = doppler_width(l.lambda0, adata.mass(l.atom), atmos.temperature, atmos.vturb);
    params.gamma = gamma_from_broadening(l, adata.broadening, atmos.temperature, atmos.ne, atmos.nh0);
    params.vel = atmos.vel;

    const UV uv = compute_uv(
        l,
        profile,
        params,
        lambda,
        opts
    );
    return uv;
}

template <typename T=fp_t, typename mem_space=DefaultMemSpace>
YAKL_INLINE UV compute_uv_cont(
    const EmisOpacState<T, mem_space>& args,
    int cont_idx,
    const UvOptions& opts = UvOptions()
) {
    JasUnpack(args, adata, profile, la, n, n_star_scratch, k, atmos, mode, active_set);
    const auto& n_star = args.n_star_scratch;
    const fp_t lambda = adata.wavelength(la);
    const auto& cont = adata.continua(cont_idx);

    using namespace ConstantsFP;
    ContParams<mem_space> params;
    params.la = la;
    const int offset = adata.level_start(cont.atom);
    params.thermal_ratio = n_star(offset + cont.i, k) / n_star(offset + cont.j, k) * std::exp(-hc_k_B_nm / (lambda * atmos.temperature));
    params.sigma_grid = get_sigma<mem_space>(adata, cont);

    const UV uv = compute_uv<mem_space>(
        cont,
        params,
        lambda,
        opts
    );
    return uv;
}


/// Computes the emissivity and opacity for for all relevant transitions in the
/// model. Lines are selected for mode All or DynamicOnly, continua for All or
/// StaticOnly.
/// N.B. This is not thread-safe for continua if multiple threads could be in
/// the same spatial cell k at a time (e.g. multiple angles). If the LTE
/// populations are precomputed, and provided it can be made thread-safe (and a
/// little faster), by setting args.update_n_star = false
template <typename T=fp_t, typename mem_space=DefaultMemSpace>
YAKL_INLINE EmisOpac emis_opac(
    const EmisOpacState<T, mem_space>& args
) {
    JasUnpack(args, adata, profile, la, n, n_star_scratch, k, atmos, mode, active_set, active_set_cont);
    EmisOpac result{FP(0.0), FP(0.0)};
    fp_t lambda = adata.wavelength(la);
    const bool lines = (mode == EmisOpacMode::All) || (mode == EmisOpacMode::DynamicOnly);
    const bool conts = (mode == EmisOpacMode::All) || (mode == EmisOpacMode::StaticOnly);

    if (lines) {
        if (active_set.is_allocated()) {
            for (int kr = 0; kr < active_set.extent(0); ++kr) {
                const auto& l = adata.lines(active_set(kr));
                const UV uv = compute_uv_line(
                    args,
                    active_set(kr)
                );
                const int offset = adata.level_start(l.atom);
                const fp_t nj = n(offset + l.j, k);
                const fp_t ni = n(offset + l.i, k);
                result.eta += nj * uv.Uji;
                result.chi += ni * uv.Vij - nj * uv.Vji;
            }
        } else {
            for (int kr = 0; kr < adata.lines.extent(0); ++kr) {
                const auto& l = adata.lines(kr);
                if (!l.is_active(la)) {
                    continue;
                }
                const UV uv = compute_uv_line(
                    args,
                    kr
                );
                const int offset = adata.level_start(l.atom);
                const fp_t nj = n(offset + l.j, k);
                const fp_t ni = n(offset + l.i, k);
                result.eta += nj * uv.Uji;
                result.chi += ni * uv.Vij - nj * uv.Vji;
            }
        }
    }

    if (conts) {
        bool any_active = false || (active_set_cont.is_allocated() && active_set_cont.extent(0) > 0);
        if (!active_set_cont.is_allocated()) {
            for (int kr = 0; kr < adata.continua.extent(0); ++kr) {
                const auto& cont = adata.continua(kr);
                any_active = any_active || cont.is_active(la);
                if (any_active) {
                    break;
                }
            }
        }
        if (!any_active) {
            return result;
        }

        if (args.update_n_star) {
            for (int ia = 0; ia < adata.num_level.extent(0); ++ia) {
                const auto n_star = slice_pops(
                    n_star_scratch,
                    adata,
                    ia
                );
                const auto lte_data = extract_lte_terms_dev(adata, ia);
                lte_pops<T, fp_t>(
                    lte_data.energy,
                    lte_data.g,
                    lte_data.stage,
                    atmos.temperature,
                    atmos.ne,
                    lte_data.abundance * atmos.nhtot,
                    n_star,
                    k
                );
            }
        }
        if (active_set_cont.is_allocated()) {
            for (int i = 0; i < active_set_cont.extent(0); ++i) {
                const int kr = active_set_cont(i);
                const auto& cont = adata.continua(kr);

                const UV uv = compute_uv_cont(
                    args,
                    kr
                );
                const int offset = adata.level_start(cont.atom);
                const fp_t nj = n(offset + cont.j, k);
                const fp_t ni = n(offset + cont.i, k);
                result.eta += nj * uv.Uji;
                result.chi += ni * uv.Vij - nj * uv.Vji;
            }
        } else {
            for (int kr = 0; kr < adata.continua.extent(0); ++kr) {
                const auto& cont = adata.continua(kr);
                if (!cont.is_active(la)) {
                    continue;
                }

                const UV uv = compute_uv_cont(
                    args,
                    kr
                );
                const int offset = adata.level_start(cont.atom);
                const fp_t nj = n(offset + cont.j, k);
                const fp_t ni = n(offset + cont.i, k);
                result.eta += nj * uv.Uji;
                result.chi += ni * uv.Vij - nj * uv.Vji;
            }
        }
    }

    return result;
}

// NOTE(cmo): An overload for a specific lambda that may not be in the atomic data grid
template <typename T=fp_t, typename mem_space=DefaultMemSpace>
YAKL_INLINE EmisOpac emis_opac(
    const EmisOpacSpecState<T, mem_space>& args
) {
    JasUnpack(args, adata, profile, lambda, n, n_star_scratch, k, atmos);
    EmisOpac result{FP(0.0), FP(0.0)};

    int la_effective = upper_bound(adata.wavelength, lambda) - 1;

    for (int kr = 0; kr < adata.lines.extent(0); ++kr) {
        const auto& l = adata.lines(kr);
        if (!l.is_active(la_effective)) {
            continue;
        }

        LineParams params;
        params.dop_width = doppler_width(l.lambda0, adata.mass(l.atom), atmos.temperature, atmos.vturb);
        params.gamma = gamma_from_broadening(l, adata.broadening, atmos.temperature, atmos.ne, atmos.nh0);
        params.vel = atmos.vel;

        const UV uv = compute_uv(
            l,
            profile,
            params,
            lambda
        );

        const int offset = adata.level_start(l.atom);
        const fp_t nj = n(offset + l.j, k);
        const fp_t ni = n(offset + l.i, k);
        result.eta += nj * uv.Uji;
        result.chi += ni * uv.Vij - nj * uv.Vji;
    }

    bool any_active = false;
    for (int kr = 0; kr < adata.continua.extent(0); ++kr) {
        const auto& cont = adata.continua(kr);
        any_active = any_active || cont.is_active(la_effective);
        if (any_active) {
            break;
        }
    }
    if (!any_active) {
        return result;
    }

    for (int ia = 0; ia < adata.num_level.extent(0); ++ia) {
        const auto n_star = slice_pops(
            n_star_scratch,
            adata,
            ia
        );
        const auto lte_data = extract_lte_terms_dev(adata, ia);
        lte_pops<T, fp_t, mem_space>(
            lte_data.energy,
            lte_data.g,
            lte_data.stage,
            atmos.temperature,
            atmos.ne,
            lte_data.abundance * atmos.nhtot,
            n_star,
            k
        );
    }
    const auto& n_star = n_star_scratch;
    for (int kr = 0; kr < adata.continua.extent(0); ++kr) {
        const auto& cont = adata.continua(kr);
        if (!cont.is_active(la_effective)) {
            continue;
        }
        int la_effective_p = std::min(la_effective + 1, cont.red_idx - 1);
        fp_t t;
        // NOTE(cmo): Weight for linear interpolation of sigma
        if (la_effective == la_effective_p) {
            t = FP(1.0);
        } else {
            t = (lambda - adata.wavelength(la_effective)) / (adata.wavelength(la_effective_p) - adata.wavelength(la_effective));
        }

        using namespace ConstantsFP;
        ContParams<mem_space> params;
        const int offset = adata.level_start(cont.atom);
        params.thermal_ratio = n_star(offset + cont.i, k) / n_star(offset + cont.j, k) * std::exp(-hc_k_B_nm / (lambda * atmos.temperature));
        params.sigma_grid = get_sigma<mem_space>(adata, cont);

        UV uv;
        uv.Vij = (
            (FP(1.0) - t) * params.sigma_grid.sigma(la_effective - cont.blue_idx)
            + t * params.sigma_grid.sigma(la_effective_p - cont.blue_idx)
        );
        uv.Vji = params.thermal_ratio * uv.Vij;
        uv.Uji = twohc2_kW_nm2 / (cube(lambda) * square(lambda) * FP(1e-18)) * uv.Vji;

        const fp_t nj = n(offset + cont.j, k);
        const fp_t ni = n(offset + cont.i, k);
        result.eta += nj * uv.Uji;
        result.chi += ni * uv.Vij - nj * uv.Vji;
    }

    return result;
}

#else
#endif