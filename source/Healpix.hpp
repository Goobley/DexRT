#if !defined(DEXRT_HEALPIX_HPP)
#define DEXRT_HEALPIX_HPP

#include "Types.hpp"
#include "Constants.hpp"
#include "Utils.hpp"
#include "MortonCodes.hpp"

/// Location of a healpix pixel centre on the unit sphere: z = cos(theta),
/// s = sin(theta), and azimuth phi.
template <typename T=fp_t>
struct HealpixLoc {
    T z;
    T s;
    T phi;
};

/// Centre of pixel pix (nested ordering) for a given nside. Written in
/// healpix_bare (BSD-3) terminology. TXY is the precision of the in-face
/// coordinates (f32 historically in the RC code).
template <typename T=fp_t, typename TXY=f32>
YAKL_INLINE HealpixLoc<T> healpix_nest_loc(i32 nside, i32 pix) {
    constexpr T pi = T(3.14159265358979323846);
    // nest2hpd
    const i32 npface = square(nside);
    const i32 p2 = pix & (npface-1);
    const i32 px = compact_1_by_1(u32(p2));
    const i32 py = compact_1_by_1(u32(p2) >> 1);
    const i32 face = pix / npface;

    // hpd2loc
    const TXY x = (px + T(0.5)) / nside;
    const TXY y = (py + T(0.5)) / nside;
    const i32 r = 1 - face / 4;
    const T h = r - 1 + (x + y);
    const T m = T(2.0) - r * h;
    T z, s, phi;
    auto jpll = [](int face) {
        constexpr bool use_array = false;
        constexpr bool branchless = true;
        if constexpr (use_array) {
            constexpr int jpll_term[12] = { 1, 3, 5, 7, 0, 2, 4, 6, 1, 3, 5, 7 };
            return jpll_term[face];
        } else if constexpr (branchless) {
            // NOTE(cmo): in the real-world case, this profiles ~1% faster
            i32 i = face & 7;
            i32 ge4 = (i & 4) >> 2;
            return ge4 * 2 * (i & 3) + (1 - ge4) * (2 * i + 1);
        } else {
            if (face >= 8) {
                face -= 8;
            }

            if (face < 4) {
                return 2 * face + 1;
            }
            return (face - 4) * 2;
        }
    };
    if (m < T(1.0)) {
        // polar cap
        const T m23 = square(m) * (T(1.0) / T(3.0));
        z = r * (T(1.0) - m23);
        s = std::sqrt(m23 * (T(2.0) - m23));
        phi = (T(0.25) * pi) * (jpll(face) + (x - y) / m);
    } else {
        // equatorial cylinder
        z = h * (T(2.0) / T(3.0));
        s = std::sqrt((T(1.0) + z) * (T(1.0) - z));
        phi = (T(0.25) * pi) * (jpll(face) + (x - y));
    }
    return HealpixLoc<T>{
        .z = z,
        .s = s,
        .phi = phi
    };
}

#else
#endif
