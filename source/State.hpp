#if !defined(DEXRT_STATE_HPP)
#define DEXRT_STATE_HPP

#include "Types.hpp"
#include "Voigt.hpp"
#include "LteHPops.hpp"
#include "PromweaverBoundary.hpp"
#include "ZeroBoundary.hpp"
#include <magma_v2.h>

enum class BoundaryType {
    Zero,
    Promweaver,
};

struct State {
    CascadeStorage c0_size;
    Atmosphere atmos;
    InclQuadrature incl_quad;
    CompAtom<fp_t> atom;
    VoigtProfile<fp_t, false> phi;
    HPartFn<> nh_lte;
    FpConst1dHost wavelength_h;
    yakl::Array<bool, 3, yakl::memDevice> dynamic_opac; // [z, x, wave]
    Fp3d pops; /// [num_level, x, y]
    Fp3d J; /// [num_wave, x, y]
    Fp5d alo; /// [z, x, phi, wave, theta]
    Fp4d Gamma; /// [i, j, x, y]
    PwBc<> pw_bc;
    ZeroBc zero_bc;
    BoundaryType boundary;
    magma_queue_t magma_queue;
};

#else
#endif