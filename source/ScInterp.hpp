#if !defined(DEXRT_SC_INTERP_HPP)
#define DEXRT_SC_INTERP_HPP

#include "Types.hpp"
#include "ScConfig.hpp"
#include "ScRayEntry.hpp"
#include <cmath>

/// Limited slope [per node] from the backward and forward differences.
template <ScInterp Interp> YAKL_INLINE fp_t sc_limited_slope(fp_t dm, fp_t dp) {
    if (dm * dp <= FP(0.0)) {
        return FP(0.0);
    }
    if constexpr (Interp == ScInterp::VanLeer) {
        return FP(2.0) * dm * dp / (dm + dp);
    } else {
        const fp_t a = std::min(std::abs(FP(2.0) * dm), std::abs(FP(2.0) * dp));
        const fp_t mag = std::min(a, FP(0.5) * std::abs(dm + dp));
        return std::copysign(mag, dm);
    }
}

/// Upwind value from nodes s-2..s+1 on the upwind line, the upwind point lying
/// a distance t (node units) before s. Finite-volume remap: the average of the
/// limited piecewise-linear reconstruction over the cell-width window centred
/// on the upwind point, i.e. linear advection at Courant number t. Linear is
/// the unlimited first-order limit (plain linear interpolation). Conservative
/// and positivity preserving for t in [0, 1].
template <ScInterp Interp> YAKL_INLINE fp_t sc_interp(
    fp_t fm2,
    fp_t fm1,
    fp_t f0,
    fp_t fp1,
    fp_t t
) {
    const fp_t linear = f0 - t * (f0 - fm1);
    if constexpr (Interp == ScInterp::Linear) {
        return linear;
    } else {
        const fp_t s0 = sc_limited_slope<Interp>(f0 - fm1, fp1 - f0);
        const fp_t sm1 = sc_limited_slope<Interp>(fm1 - fm2, f0 - fm1);
        return linear - FP(0.5) * t * (FP(1.0) - t) * (s0 - sm1);
    }
}

/// Value on a plane (or face) a distance (ta, tb) before node (0, 0), from the
/// nodes fetch(i, j), i, j in -2..1: successive 1D remaps (sc_interp), along a
/// on the lines j = -2..1, then along b. The window average of a tensor-product
/// reconstruction, so it keeps the cross term; bounded and positivity
/// preserving for ta, tb in [0, 1]. Linear: bilinear. tb = 0: the 1D remap.
template <ScInterp Interp, class Fetch> YAKL_INLINE fp_t sc_interp_2d(const Fetch& fetch, fp_t ta, fp_t tb) {
    auto line = [&](i32 j) -> fp_t {
        if constexpr (Interp == ScInterp::Linear) {
            return sc_interp<Interp>(
                FP(0.0),
                fetch(-1, j),
                fetch(0, j),
                FP(0.0),
                ta
            );
        } else {
            return sc_interp<Interp>(
                fetch(-2, j),
                fetch(-1, j),
                fetch(0, j),
                fetch(1, j),
                ta
            );
        }
    };
    if constexpr (Interp == ScInterp::Linear) {
        return sc_interp<Interp>(
            FP(0.0),
            line(-1),
            line(0),
            FP(0.0),
            tb
        );
    } else {
        return sc_interp<Interp>(
            line(-2),
            line(-1),
            line(0),
            line(1),
            tb
        );
    }
}

/// sc_interp_2d of two fields from one fetch per node (fetch(i, j) -> a pair
/// with members first, second), e.g. chi and eta: half the reads.
template <ScInterp Interp, class Fetch>
YAKL_INLINE auto sc_interp_2d_pair(const Fetch& fetch, fp_t ta, fp_t tb) {
    struct Pair {
        fp_t first;
        fp_t second;
    };
    auto line = [&](i32 j) -> Pair {
        if constexpr (Interp == ScInterp::Linear) {
            const auto m1 = fetch(-1, j);
            const auto m0 = fetch(0, j);
            return Pair{
                sc_interp<Interp>(
                    FP(0.0),
                    m1.first,
                    m0.first,
                    FP(0.0),
                    ta
                ),
                sc_interp<Interp>(
                    FP(0.0),
                    m1.second,
                    m0.second,
                    FP(0.0),
                    ta
                ),
            };
        } else {
            const auto m2 = fetch(-2, j);
            const auto m1 = fetch(-1, j);
            const auto m0 = fetch(0, j);
            const auto p1 = fetch(1, j);
            return Pair{
                sc_interp<Interp>(
                    m2.first,
                    m1.first,
                    m0.first,
                    p1.first,
                    ta
                ),
                sc_interp<Interp>(
                    m2.second,
                    m1.second,
                    m0.second,
                    p1.second,
                    ta
                ),
            };
        }
    };
    if constexpr (Interp == ScInterp::Linear) {
        const Pair l1 = line(-1);
        const Pair l0 = line(0);
        return Pair{
            sc_interp<Interp>(
                FP(0.0),
                l1.first,
                l0.first,
                FP(0.0),
                tb
            ),
            sc_interp<Interp>(
                FP(0.0),
                l1.second,
                l0.second,
                FP(0.0),
                tb
            ),
        };
    } else {
        const Pair l2 = line(-2);
        const Pair l1 = line(-1);
        const Pair l0 = line(0);
        const Pair p1 = line(1);
        return Pair{
            sc_interp<Interp>(
                l2.first,
                l1.first,
                l0.first,
                p1.first,
                tb
            ),
            sc_interp<Interp>(
                l2.second,
                l1.second,
                l0.second,
                p1.second,
                tb
            ),
        };
    }
}

/// Interpolate chi and eta together on a one-dimensional upwind stencil.
template <ScInterp Interp, class Fetch>
YAKL_INLINE ScMediumPoint sc_interp_medium_1d(const Fetch& fetch, fp_t fraction) {
    constexpr bool linear = (Interp == ScInterp::Linear);
    const ScMediumPoint mm1 = fetch(-1), m0 = fetch(0);
    if (linear) {
        return sc_medium_from_eta(
            sc_interp<Interp>(
                FP(0.0),
                mm1.chi,
                m0.chi,
                FP(0.0),
                fraction
            ),
            sc_interp<Interp>(
                FP(0.0),
                mm1.chi * mm1.S,
                m0.chi * m0.S,
                FP(0.0),
                fraction
            ));
    }
    const ScMediumPoint mm2 = fetch(-2), mp1 = fetch(1);
    return sc_medium_from_eta(
        sc_interp<Interp>(
            mm2.chi,
            mm1.chi,
            m0.chi,
            mp1.chi,
            fraction
        ),
        sc_interp<Interp>(
            mm2.chi * mm2.S,
            mm1.chi * mm1.S,
            m0.chi * m0.S,
            mp1.chi * mp1.S,
            fraction
        ));
}

/// Interpolate both medium coefficients using the same tensor-product stencil.
template <ScInterp Interp, class Fetch>
YAKL_INLINE ScMediumPoint sc_interp_medium_2d(const Fetch& fetch, fp_t ta, fp_t tb) {
    struct ChiEta {
        fp_t first;
        fp_t second;
    };
    const auto r = sc_interp_2d_pair<Interp>(
        [&](i32 i, i32 j) {
            const ScMediumPoint m = fetch(i, j);
            return ChiEta{m.chi, m.chi * m.S};
        },
        ta,
        tb);
    return sc_medium_from_eta(r.first, r.second);
}

/// Upper stencil node and fractional distance before it, in node coordinates.
YAKL_INLINE void sc_interp_position(fp_t q, i32* c, fp_t* t) {
    *c = i32(std::floor(q)) + 1;
    *t = fp_t(*c) - q;
}

#endif
