#if !defined(DEXRT_STATE_HPP)
#define DEXRT_STATE_HPP

#include "Types.hpp"
#include "Voigt.hpp"
#include "LteHPops.hpp"
#include "PromweaverBoundary.hpp"
#include "ZeroBoundary.hpp"
#include "BoundaryType.hpp"
#include "BlockMap.hpp"
#include "DexrtConfig.hpp"

#ifdef DEXRT_USE_MAGMA
    #include <magma_v2.h>
#endif

struct State {
    DexrtConfig config;
    CascadeStorage c0_size;
    GivenEmisOpac given_state;
    Atmosphere atmos;
    MultiResBlockMap<BLOCK_SIZE, ENTRY_SIZE> mr_block_map;
    yakl::Array<i32, 3, yakl::memDevice> max_block_mip;
    InclQuadrature incl_quad;
    AtomicData<fp_t, yakl::memDevice> adata;
    std::vector<CompAtom<fp_t, yakl::memDevice>> atoms;
    std::vector<CompAtom<fp_t, yakl::memDevice>> atoms_with_gamma;
    bool have_h;
    std::vector<int> atoms_with_gamma_mapping;
    AtomicData<fp_t, yakl::memHost> adata_host;
    VoigtProfile<fp_t, false> phi;
    HPartFn<> nh_lte;
    yakl::Array<bool, 2, yakl::memDevice> active; // [z, x]  -- whether this cell should be considered
    yakl::Array<i64, 2, yakl::memDevice> active_map; // [z, x] -- index of this cell in an array only allocated for active cells (-1 if not allocated)
    Fp3d wphi; /// [kr, z, x]
    Fp3d pops; /// [num_level, x, y]
    Fp3d J; /// [num_wave, x, y] -- if we're paging J to cpu, the first axis is wave_batch
    Fp3dHost J_cpu; /// [num_wave, x, y] -- The full J in host memory, if we're paging after each batch.
    Fp1d alo; /// [z, x, phi, wave, theta], but flattened. Index using Cascade operators (probe_lin_index) -- you need to fetch intensity at the same time anyway.
    std::vector<Fp4d> Gamma; /// [i, j, x, y]
    PwBc<> pw_bc;
    ZeroBc zero_bc;
    BoundaryType boundary;
#ifdef DEXRT_USE_MAGMA
    magma_queue_t magma_queue;
#endif
};

#else
#endif