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

struct LineParams {
    /// Doppler width [nm]
    fp_t dop_width;
    /// damping
    fp_t gamma;
    /// projected velocity [m / s]
    fp_t vel;
};

struct AtmosPointParams {
    fp_t temperature;
    fp_t ne;
    fp_t vturb;
    fp_t nhtot;
    /// Neutral H density
    fp_t nh0;
};

template <typename T=fp_t, int mem_space=yakl::memDevice>
struct EmisOpacState {
    const CompAtom<T, mem_space>& atom;
    const VoigtProfile<T, false, mem_space>& profile;
    int la;
    const yakl::Array<fp_t const, 1, mem_space>& n;
    const yakl::Array<fp_t, 1, mem_space>& n_star_scratch;
    const AtmosPointParams& atmos;
    EmisOpacMode mode = EmisOpacMode::All;
};

template <int mem_space=yakl::memDevice>
struct SigmaInterp {
    yakl::Array<fp_t const, 1, mem_space> sigma;
};

template <int mem_space=yakl::memDevice>
struct ContParams {
    /// n_i* / n_j* exp(-h nu / (k_B T)) (gij in RH/Lw)
    fp_t thermal_ratio;
    /// Wavelength index (global wavelength array)
    int la;
    SigmaInterp<mem_space> sigma_grid;
};

template <int mem_space=yakl::memDevice>
YAKL_INLINE SigmaInterp<mem_space> get_sigma(const CompAtom<fp_t, mem_space>& atom, const CompCont<fp_t>& cont) {
    SigmaInterp<mem_space> result;

    result.sigma = yakl::Array<fp_t const, 1, mem_space>(
        "cont_sigma",
        &atom.sigma(cont.sigma_start),
        // TODO(cmo): Can we do away with sigma end? The array length should be the same as red_idx - blue_idx
        cont.sigma_end - cont.sigma_start
    );
    return result;
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
template <int mem_space=yakl::memDevice>
YAKL_INLINE fp_t gamma_from_broadening(
    const CompLine<fp_t>& line,
    yakl::Array<ScaledExponentsBroadening<fp_t> const, 1, mem_space> broadeners,
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


template <int mem_space=yakl::memDevice>
YAKL_INLINE UV compute_uv(
    const CompLine<fp_t>& l,
    const VoigtProfile<fp_t, false, mem_space>& phi,
    const LineParams& params,
    fp_t lambda
) {
    using namespace ConstantsFP;
    // [kJ]
    const fp_t hnu_4pi = hc_kJ_nm / (four_pi * lambda);
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

template <int mem_space=yakl::memDevice>
YAKL_INLINE UV compute_uv(
    const CompCont<fp_t>& cont,
    const ContParams<mem_space>& params,
    fp_t lambda
) {
    using namespace ConstantsFP;
    UV result;
    // [m2]
    result.Vij = params.sigma_grid.sigma(params.la - cont.blue_idx);
    // [m2]
    result.Vji = params.thermal_ratio * result.Vij;
    // [kW nm2 / (nm3 m2)] = [kW nm-1] (sr implicit)
    result.Uji = twohc2_kW_nm2 / (cube(lambda) * square(lambda) * FP(1e-18)) * result.Vji;
    return result;
}

template <typename T=fp_t, int mem_space=yakl::memDevice>
YAKL_INLINE EmisOpac emis_opac(
    const EmisOpacState<T, mem_space>& args
) {
    JasUnpack(args, atom, profile, la, n, n_star_scratch, atmos, mode);
    EmisOpac result{FP(0.0), FP(0.0)};
    fp_t lambda = atom.wavelength(la);
    const bool lines = (mode == EmisOpacMode::All) || (mode == EmisOpacMode::DynamicOnly);
    const bool conts = (mode == EmisOpacMode::All) || (mode == EmisOpacMode::StaticOnly);

    if (lines) {
        for (int kr = 0; kr < atom.lines.extent(0); ++kr) {
            const auto& l = atom.lines(kr);
            if (!l.is_active(la)) {
                continue;
            }

            LineParams params;
            params.dop_width = doppler_width(l.lambda0, atom.mass, atmos.temperature, atmos.vturb);
            params.gamma = gamma_from_broadening(l, atom.broadening, atmos.temperature, atmos.ne, atmos.nh0);
            // TODO(cmo): Come back to this!
            params.vel = FP(0.0);

            const UV uv = compute_uv(
                l,
                profile,
                params,
                lambda
            );
            result.eta += n(l.j) * uv.Uji;
            result.chi += n(l.i) * uv.Vij - n(l.j) * uv.Vji;
        }
    }

    if (conts) {
        auto& n_star = n_star_scratch;
        lte_pops<T, fp_t, mem_space>(
            atom.energy,
            atom.g,
            atom.stage,
            atmos.temperature,
            atmos.ne,
            atom.abundance * atmos.nhtot,
            n_star
        );
        for (int kr = 0; kr < atom.continua.extent(0); ++kr) {
            const auto& cont = atom.continua(kr);
            if (!cont.is_active(la)) {
                continue;
            }

            using namespace ConstantsFP;
            ContParams<mem_space> params;
            params.la = la;
            params.thermal_ratio = n_star(cont.i) / n_star(cont.j) * std::exp(-hc_k_B_nm / (lambda * atmos.temperature));
            params.sigma_grid = get_sigma<mem_space>(atom, cont);

            const UV uv = compute_uv<mem_space>(
                cont,
                params,
                lambda
            );
            result.eta += n(cont.j) * uv.Uji;
            result.chi += n(cont.i) * uv.Vij - n(cont.j) * uv.Vji;
        }
    }

    return result;
}

#else
#endif