#if !defined(DEXRT_ZERO_BOUNDARY_HPP)
#define DEXRT_ZERO_BOUNDARY_HPP
#include "Types.hpp"

struct ZeroBc {

};

YAKL_INLINE fp_t sample_boundary(
    const ZeroBc& bc,
    int la,
    vec3 at,
    vec3 dir
) {
    return FP(0.0);
}

#else
#endif