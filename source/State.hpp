#if !defined(DEXRT_STATE_HPP)
#define DEXRT_STATE_HPP

#include "Types.hpp"
#include "Voigt.hpp"
#include <magma_v2.h>

struct State {
    std::vector<Fp5d> cascades;
    MipmapState raymarch_state;
    Atmosphere atmos;
    CompAtom<fp_t> atom;
    VoigtProfile<fp_t, false> phi;
    Fp3d pops; /// [x, y, num_level] TODO(cmo): Update this!
    Fp3d J; /// [num_wave, x, y]
    Fp2d alo; /// [x, y]
    Fp4d Gamma; /// [i, j, x, y]
    magma_queue_t magma_queue;
};

#else
#endif