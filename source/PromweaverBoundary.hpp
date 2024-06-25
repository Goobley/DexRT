#if !defined(DEXRT_PROMWEAVER_BOUNDARY)
#define DEXRT_PROMWEAVER_BOUNDARY

#include "Types.hpp"
#include "BoundaryType.hpp"
#include "Utils.hpp"
#include "YAKL_netcdf.h"

template <int mem_space=yakl::memDevice>
struct PwBc {
    fp_t mu_min;
    fp_t mu_max;
    fp_t mu_step;
    yakl::Array<fp_t, 2, mem_space> I; // [wl, mu]
};
PwBc<> load_bc(const std::string& path, const FpConst1d& wavelength, BoundaryType type);

/** Computes the outgoing mu (relative to the surface normal at the location of
 * the ray hit). Assumes the Sun is a sphere with radius 695,700 km. Looks both
 * forwards and back.
 * Radius from: 2008ApJ...675L..53H
 * \param mu The inclination of the ray in question away from the z axis
 * \param altitude_m The altitude of the ray start point [m]
 * \returns A positive mu if the ray hits, or -1.0 if it misses
*/
YAKL_INLINE fp_t chromo_ray_mu(fp_t mu, fp_t altitude_m) {
    constexpr fp_t Rs = FP(695700000.0);
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

/** Computes the outgoing mu (relative to the surface normal at the location of
 * the ray hit). Assumes the Sun is a sphere with radius 695,700 km.
 * Radius from: 2008ApJ...675L..53H
 * \param pos The ray launch position (3D relative in local coordinate system,
 * including altitude from solar surface)
 * \param dir 3D direction vector (normalised)
 * \returns A positive mu if the ray hits, or -1.0 if it misses
*/
YAKL_INLINE fp_t vector_chromo_ray_mu(vec3 pos, vec3 dir) {
    constexpr fp_t Rs = FP(695700000.0);
    // NOTE(cmo): Solving in units of Rs
    pos(0) /= Rs;
    pos(1) /= Rs;
    pos(2) /= Rs;
    pos(2) += FP(1.0);
    // NOTE(cmo): Solve for t of intersection with sphere as per pos + t * dir
    // d @ d == 1
    constexpr fp_t a = FP(1.0);
    // 2 * (pos @ dir)
    const fp_t b = FP(2.0) * (pos(0) * dir(0) + pos(1) * dir(1) + pos(2) * dir(2));
    // pos @ pos - Rs**2
    // NOTE(cmo): Almost all of pos will be "in" z, so do (pos_z**2 - Rs**2) =
    // (pos_z + Rs) * (pos_z - Rs) and add the rest of the dot components
    // const fp_t c = (pos(2) + Rs) * (pos(2) - Rs) + square(pos(0)) + square(pos(1));
    const fp_t c = square(pos(0)) + square(pos(1)) + square(pos(2)) - FP(1.0);
    const fp_t delta = square(b) - FP(4.0) * a * c;

    if (delta < FP(0.0)) {
        return FP(-1.0);
    }

    const fp_t t_hit = (-b - std::sqrt(delta)) / (FP(2.0) * a);
    pos(0) += t_hit * dir(0);
    pos(1) += t_hit * dir(1);
    pos(2) += t_hit * dir(2);
    // NOTE(cmo): Sun-local cos(mu) = - (dir @ hit) / Rs, but Rs = 1
    const fp_t cosphi = -(dir(0) * pos(0) + dir(1) * pos(1) + dir(2) * pos(2));
    return cosphi;
}

/**
 * Sample a PwBc instance for a ray at with origin `at`, along direction `dir`,
 * at wavelength index la.  Only considers the height of the point (i.e. domain
 * size small vs solar curvature).
*/
YAKL_INLINE fp_t sample_boundary(
    const PwBc<>& bc,
    int la,
    vec3 at,
    vec3 dir
) {
    fp_t mu_sample;
    if constexpr (PWBC_USE_VECTOR_FORM) {
        if constexpr (!PWBC_CONSIDER_HORIZONTAL_OFFSET) {
            at(0) = FP(0.0);
        }
        mu_sample = vector_chromo_ray_mu(at, dir);
    } else {
        const fp_t alt = at(2);
        const fp_t mu = dir(2);
        mu_sample = chromo_ray_mu(mu, alt);
    }

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