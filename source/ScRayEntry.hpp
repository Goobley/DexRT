#ifndef DEXRT_SC_RAY_ENTRY_HPP
#define DEXRT_SC_RAY_ENTRY_HPP
#include "Types.hpp"
#include "ScGeometry.hpp"
#include <cmath>

struct ScMediumPoint {
    fp_t chi;
    fp_t S;
};

/// Medium from interpolated chi and eta = chi S (i.e. S interpolated with
/// opacity weights): interpolating S on its own pairs, next to a sharp
/// interface, one node's S with another's opacity (e.g. a hot, transparent
/// layer's enormous S with a cool, dense one's chi).
YAKL_INLINE ScMediumPoint sc_medium_from_eta(fp_t chi, fp_t eta) {
    // Keep the opacity sign for amplification; guard only the zero denominator.
    return ScMediumPoint{.chi = chi, .S = (chi != FP(0.0)) ? eta / chi : FP(0.0)};
}

/// Periodic fold: plain interior indexing, a mask for power-of-two extents,
/// otherwise one integer division. Negative and multi-lap indices are valid.
YAKL_INLINE i32 sc_wrap(i32 s, i32 n) {
    if (s >= 0 && s < n) {
        return s;
    }
    if ((n & (n - 1)) == 0) {
        return i32(u32(s) & u32(n - 1));
    }
    const i32 remainder = s - (s / n) * n;
    return remainder < 0 ? remainder + n : remainder;
}

#endif
