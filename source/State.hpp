#if !defined(DEXRT_STATE_HPP)
#define DEXRT_STATE_HPP

#include "Types.hpp"
#include "Voigt.hpp"
#include "LteHPops.hpp"
#include <magma_v2.h>

struct State {
    std::vector<Fp5d> cascades;
    MipmapState raymarch_state;
    Atmosphere atmos;
    CompAtom<fp_t> atom;
    VoigtProfile<fp_t, false> phi;
    HPartFn<> nh_lte;
    FpConst1dHost wavelength_h;
    Fp3d pops; /// [num_level, x, y]
    Fp3d J; /// [num_wave, x, y]
    Fp3d alo; /// [x, y, az]
    Fp4d Gamma; /// [i, j, x, y]
    magma_queue_t magma_queue;
};

#else
#endif