#if !defined(DEXRT_STATE_3D_HPP)
#define DEXRT_STATE_3D_HPP

#include "Config.hpp"
#include "DexrtConfig.hpp"
#include "RcUtilsModes3d.hpp"
#include "BlockMap.hpp"
#include "MpiWrapper.hpp"
#include "Rad16BitStorage.hpp"

struct GivenEmisOpac3d {
    fp_t voxel_scale;
    Fp4d emis;
    Fp4d opac;
};

struct State3d {
    MpiState mpi_state;
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
    Rad16Array wphi; /// [kr, ks]
    Fp2d pops; /// [num_level, ks]
    Fp2d J; /// [1, ks] -- the current wavelength, paged to J_cpu after each ([num_wave, ks] in GivenFs)
    Rad16ArrayHost J_cpu; /// [num_wave, ks] -- The full J in host memory.
    Rad16Array J_stage; /// J encoded for the copy to J_cpu (HALF_RADIATION_STORAGE only).
    PreconRMat precon_R; /// [t, 2, ks] the preconditioned radiative rates (GammaMatrix.hpp)
    PwBc<> pw_bc;
    ZeroBc zero_bc;
    BoundaryType boundary;

    template <typename ...T>
    void println(fmt::format_string<T...> fmt, T&&... args) const {
        if (mpi_state.rank == 0) {
            fmt::println(fmt, std::forward<T>(args)...);
        }
    }
};

#else
#endif