#if !defined(DEXRT_PROMWEAVER_BOUNDARY)
#define DEXRT_PROMWEAVER_BOUNDARY

#include "Types.hpp"
#include "Utils.hpp"
#include "YAKL_netcdf.h"

template <int mem_space=yakl::memDevice>
struct PwBc {
    fp_t mu_min;
    fp_t mu_max;
    fp_t mu_step;
    yakl::Array<fp_t, 2, mem_space> I; // [wl, mu]
};
struct State;
void load_bc(const std::string& path, State* state);

/** Computes the outgoing mu (relative to the surface normal at the location of
 * the ray hit). Assumes the Sun is a sphere with radius 695,700 km. Looks both
 * forwards and back.
 * Radius from: 2008ApJ...675L..53H
 * \param mu The inclination of the ray in question away from the z axis
 * \param altitude_m The altitude of the ray start point [m]
 * \returns A positive mu if the ray hits, or -1.0 if it misses
*/
YAKL_INLINE fp_t chromo_ray_mu(fp_t mu, fp_t altitude_m) {
    fp_t Rs = FP(695700000.0);
    // out_mu2 = 1.0 - (1.0 - mu**2) * (Rs + altitude_m)**2 / Rs**2
    fp_t out_mu2 = (FP(1.0) - square(mu));
    // NOTE(cmo): Rearranged from (Rs + alt)^2 / Rs^2
    out_mu2 *= FP(1.0) + FP(2.0) * altitude_m / Rs + square(altitude_m / Rs);
    out_mu2 = FP(1.0) - out_mu2;
    if (out_mu2 < 0.0) {
        return FP(-1.0);
    }
    return std::sqrt(out_mu2);
}

/**
 * Sample a PwBc instance for a ray at with origin `at`, along direction `dir`,
 * at wavelength index la.  Only considers the height of the point (i.e. domain
 * size small vs solar curvature).
*/
YAKL_INLINE fp_t sample_boundary(
    const PwBc<>& bc,
    int la,
    const vec2& at,
    const vec2& dir
) {
    const fp_t alt = at(1);
    const fp_t mu = dir(1);
    const fp_t mu_sample = chromo_ray_mu(mu, alt);

    fp_t result;
    if (mu_sample < FP(0.0)) {
        result = FP(0.0);
    } else if (mu_sample <= bc.mu_min) {
        result = bc.I(la, 0);
    } else if (mu_sample >= bc.mu_max) {
        result = bc.I(la, bc.I.extent(1) - 1);
    } else {
        fp_t frac_idx = (mu_sample - bc.mu_min) / bc.mu_step;
        int idx = int(frac_idx);
        fp_t t = frac_idx - fp_t(idx);

        result = (FP(1.0) - t) * bc.I(la, idx) + t * bc.I(la, idx + 1);
    }
    return result;
}

#else
#endif