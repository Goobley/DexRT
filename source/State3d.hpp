#if !defined(DEXRT_STATE_3D_HPP)
#define DEXRT_STATE_3D_HPP

#include "Config.hpp"
#include "DexrtConfig.hpp"
#include "RcUtilsModes3d.hpp"
#include "BlockMap.hpp"

struct GivenEmisOpac3d {
    fp_t voxel_scale;
    Fp4d emis;
    Fp4d opac;
};

struct State3d {
    DexrtConfig config;
    CascadeStorage3d c0_size;
    MultiResBlockMap<BLOCK_SIZE_3D, ENTRY_SIZE_3D, 3> mr_block_map;
    GivenEmisOpac3d given_state;
    SparseAtmosphere atmos;
    AtomicData<fp_t, yakl::memDevice> adata;
    std::vector<CompAtom<fp_t, yakl::memDevice>> atoms;
    std::vector<CompAtom<fp_t, yakl::memDevice>> atoms_with_gamma;
    bool have_h;
    std::vector<int> atoms_with_gamma_mapping;
    AtomicData<fp_t, yakl::memHost> adata_host;
    VoigtProfile<fp_t, false> phi;
    HPartFn<> nh_lte;
    Fp2d wphi; /// [kr, ks]
    Fp2d pops; /// [num_level, ks]
    Fp2d J; // [la, ks]
    Fp2dHost J_cpu; /// [num_wave, ks] -- The full J in host memory, if we're paging after each batch.
    std::vector<yakl::Array<GammaFp, 3, yakl::memDevice>> Gamma; /// [i, j, ks]
    PwBc<> pw_bc;
    ZeroBc zero_bc;
    BoundaryType boundary;

    template <typename ...T>
    void println(fmt::format_string<T...> fmt, T&&... args) const {
        // if (mpi_state.rank == 0) {
            fmt::println(fmt, std::forward<T>(args)...);
        // }
    }
};

#else
#endif