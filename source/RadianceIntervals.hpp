#if !defined(DEXRT_RADIANCEINTERVALS_HPP)
#define DEXRT_RADIANCEINTERVALS_HPP
#include "Types.hpp"

YAKL_INLINE yakl::SArray<fp_t, 2, NUM_COMPONENTS, NUM_AZ> merge_intervals(
    yakl::SArray<fp_t, 2, NUM_COMPONENTS, NUM_AZ> closer, 
    yakl::SArray<fp_t, 2, NUM_COMPONENTS, NUM_AZ> further,
    yakl::SArray<fp_t, 1, NUM_AZ> az_rays
) {
    // NOTE(cmo): Storage is interleaved [intensity_1, tau_1, intensity_2...] on the first axis
    for (int i = 0; i < NUM_COMPONENTS; i += 2) {
        for (int r = 0; r < NUM_AZ; ++r) {
            if (az_rays(r) == FP(0.0)) {
                continue;
            }
            fp_t transmission = std::exp(-closer(i+1, r));
            closer(i, r) += transmission * further(i, r);
            closer(i+1, r) += further(i+1, r);
        }
    }
    return closer;
}

#else
#endif