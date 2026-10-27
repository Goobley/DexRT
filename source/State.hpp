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
#include "Rad16BitStorage.hpp"
#include "RateDiagnostics.hpp"

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
    AtomicData<fp_t, yakl::memDevice> adata;
    std::vector<CompAtom<fp_t, yakl::memDevice>> atoms;
    std::vector<CompAtom<fp_t, yakl::memDevice>> atoms_with_gamma;
    bool have_h;
    std::vector<int> atoms_with_gamma_mapping;
    AtomicData<fp_t, yakl::memHost> adata_host;
    VoigtProfile<fp_t, false> phi;
    HPartFn<> nh_lte;
    Rad16Array wphi; /// [kr, ks]
    Fp2d pops; /// [num_level, ks]
    Fp2d J; /// [wave_batch, ks] -- the current wavelength batch, paged to J_cpu after each
    Rad16ArrayHost J_cpu; /// [num_wave, ks] -- The full J in host memory.
    Rad16Array J_stage; /// J encoded for the copy to J_cpu (HALF_RADIATION_STORAGE only).
    yakl::Array<RadLossFp, 2, yakl::memDevice> rad_loss; /// [wave_batch || 1, ks] follows same logic as J, but also can be pre-integrated (i.e. all wavelength bins integrated)
    yakl::Array<RadLossFp, 2, yakl::memHost> rad_loss_cpu; /// [num_wave || 1, ks] per wavelength if not pre-integrated.
    std::vector<yakl::Array<GammaFp, 3, yakl::memDevice>> Gamma; /// [i, j, ks]
    RateDiagnostics rate_diag; /// Optional per-transition output diagnostics
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