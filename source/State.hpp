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
    AtomicData<fp_t, yakl::memDevice> adata;
    std::vector<CompAtom<fp_t, yakl::memDevice>> atoms;
    std::vector<CompAtom<fp_t, yakl::memDevice>> atoms_with_gamma;
    bool have_h;
    std::vector<int> atoms_with_gamma_mapping;
    AtomicData<fp_t, yakl::memHost> adata_host;
    VoigtProfile<fp_t, false> phi;
    HPartFn<> nh_lte;
    yakl::Array<bool, 3, yakl::memDevice> dynamic_opac; // [z, x, wave]
    Fp3d wphi; /// [kr, z, x]
    Fp3d pops; /// [num_level, x, y]
    Fp3d J; /// [num_wave, x, y]
    Fp5d alo; /// [z, x, phi, wave, theta]
    std::vector<Fp4d> Gamma; /// [i, j, x, y]
    PwBc<> pw_bc;
    ZeroBc zero_bc;
    BoundaryType boundary;
    magma_queue_t magma_queue;
};

#else
#endif