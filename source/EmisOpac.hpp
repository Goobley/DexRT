#if !defined(DEXRT_EMISOPAC_HPP)
#define DEXRT_EMISOPAC_HPP
#include "Config.hpp"
#include "Constants.hpp"
#include "Types.hpp"
#include "Voigt.hpp"

struct EmisOpac {
    fp_t eta;
    fp_t chi;
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
    return lambda0 / c * std::sqrt(two_k_B_u * temperature / mass + square(vturb));
}

/** Compute damping coefficient for Voigt profile from gamma
 * \param gamma [rad / s]
 * \param lambda [nm]
 * \param dop_width [nm]
 * \return gamma / (4 pi dnu_D) = gamma / (4 pi dlambda_D) * lambda**2
*/
YAKL_INLINE fp_t damping_from_gamma(fp_t gamma, fp_t lambda, fp_t dop_width) {
    using namespace ConstantsFP;
    return FP(1.0) / (four_pi * c) * gamma * square(lambda) / dop_width;
}


template <int mem_space=yakl::memDevice>
YAKL_INLINE UV uv(
    const CompLine<fp_t>& l, 
    const VoigtProfile<fp_t, false, mem_space>& phi, 
    const LineParams& params,
    fp_t lambda
) {
    using namespace ConstantsFP;
    // [kJ]
    const fp_t hnu_4pi = hc_kJ_nm / (four_pi * wave);
    const fp_t a = damping_from_gamma(params.gamma, lambda, params.dop_width);
    const fp_t v = (lambda - l.lambda0 + params.vel / c * l.lambda0) / params.dop_width;
    // [nm-1]
    const fp_t p = phi(a, v) / (FP(M_PI) * params.dop_width);
    UV result;
    // [m2]
    result.Vij = hnu_4pi * l.Bij * p;
    // [m2]
    result.Vji = hnu_4pi * l.Bji * p;
    // [kW nm-1]
    result.Uji = hnu_4pi * l.Aji * p;
    return result;
} 

template <typename T=fp_t, int mem_space=yakl::memDevice>
YAKL_INLINE EmisOpac emis_opac(
    const CompAtom<T, mem_space>& c, 
    int la, 
    const yakl::Array<fp_t const, 1, mem_space> pops
) {


}

#else
#endif