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
#include "MpiWrapper.hpp"

#include <fmt/core.h>

#ifdef DEXRT_USE_MAGMA
    #include <magma_v2.h>
#endif

struct State {
    DexrtConfig config;
    MpiState mpi_state;
    CascadeStorage c0_size;
    GivenEmisOpac given_state;
    SparseAtmosphere atmos;
    MultiResBlockMap<BLOCK_SIZE, ENTRY_SIZE> mr_block_map;
    yakl::Array<i32, 3, yakl::memDevice> max_block_mip;
    InclQuadrature incl_quad;
    AtomicData<fp_t> adata;
    std::vector<CompAtom<fp_t>> atoms;
    std::vector<CompAtom<fp_t>> atoms_with_gamma;
    bool have_h;
    std::vector<int> atoms_with_gamma_mapping;
    AtomicData<fp_t, HostSpace> adata_host;
    VoigtProfile<fp_t, false> phi;
    HPartFn<> nh_lte;
    Fp2d wphi; /// [kr, ks]
    Fp2d pops; /// [num_level, ks]
    Fp2d J; /// [num_wave, ks] -- if we're paging J to cpu, the first axis is wave_batch
    KView<fp_t**, HostSpace> J_cpu; /// [num_wave, ks] -- The full J in host memory, if we're paging after each batch.
    std::vector<yakl::Array<GammaFp, 3, yakl::memDevice>> Gamma; /// [i, j, ks]
    PwBc<> pw_bc;
    ZeroBc zero_bc;
    BoundaryType boundary;
#ifdef DEXRT_USE_MAGMA
    magma_queue_t magma_queue;
#endif

    template <typename ...T>
    void println(fmt::format_string<T...> fmt, T&&... args) const {
        if (mpi_state.rank == 0) {
            fmt::println(fmt, std::forward<T>(args)...);
        }
    }
};

#else
#endif