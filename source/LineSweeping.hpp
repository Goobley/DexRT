#if !defined(DEXRT_LINE_SWEEPING_HPP)
#define DEXRT_LINE_SWEEPING_HPP
#include "Config.hpp"
#include "Types.hpp"
#include "State.hpp"
#include "CascadeState.hpp"
#include "RayMarching.hpp"

template <int RcMode>
inline void compute_line_sweep_samples(const State& state, const CascadeState& casc_state, int cascade_idx, const CascadeCalcSubset& subset, const MultiResMipChain& mip_chain = MultiResMipChain()) {

    // Do a normal cascade-like dispatch that traces from the previous probe to each probe (ensuring to extend to capture the bc for first_sample)

    // Launch co-operative groups -- one per line, and do the extensions
}

inline void interpolate_line_sweep_samples_to_cascade(const State& state, const CascadeState& casc_state, int cascade_idx, CascadeCalcSubset& subset) {
    // Do the bilinear interpolation

}


#else
#endif