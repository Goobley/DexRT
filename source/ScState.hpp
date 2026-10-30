#if !defined(DEXRT_SC_STATE_HPP)
#define DEXRT_SC_STATE_HPP

#include "Types.hpp"
#include "ScCells.hpp"
#include "ScPrd.hpp"
#include "ScDecomposition.hpp"
#include "Voigt.hpp"
#include "LteHPops.hpp"
#include "PromweaverBoundary.hpp"
#include "BoundaryType.hpp"
#include "DexrtConfig.hpp"
#include "MpiWrapper.hpp"
#include "RateDiagnostics.hpp"
#include "BackgroundOpacity.hpp"
#include "Rad16BitStorage.hpp"
#include "CellList.hpp"

#include <fmt/core.h>

/// dexrt_sc's state. Shares field names with State wherever the templated
/// physics (stat_eq, conservation, Ng, ...) reads them.
struct ScState {
    DexrtConfig config;
    MpiState mpi_state;
    GivenEmisOpac given_state;
    SparseAtmosphere atmos;
    ScSlab slab; /// this rank's rows
    ScCells cells; /// dense slab index space (ks), see ScCells
    ScPrd prd; /// PRD lines (none if disabled)
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
    /// [wave, owned ks] (ks - cells.owned_offset()), if output or background
    /// scattering. Accumulated per wavelength batch in the DAG's device slots
    /// and paged here once the batch completes.
    Rad16ArrayHost J_cpu;
    /// [wave, halo rows * plane] (lower then upper): the neighbours' J_cpu
    /// rows, captured before a formal solution pages over them (background
    /// scattering only).
    Rad16ArrayHost J_halo_cpu;
    /// rad_loss (net loss, chi (S - mean I), as RC): [owned ks], summed over
    /// the wavelength batches as they complete; and on the host, [num_wave,
    /// owned ks] (PerWavelength) or [1, owned ks] (Integrated).
    yakl::Array<RadLossFp, 1, yakl::memDevice> rad_loss_integrated;
    yakl::Array<RadLossFp, 2, yakl::memHost> rad_loss_cpu;
    BackgroundOpacity background;
    PreconRMat precon_R; /// [t, 2, ks] see GammaMatrix.hpp
    RateDiagnostics rate_diag;
    PwBc<> pw_bc;
    BoundaryType boundary;

    template <typename ...T>
    void println(fmt::format_string<T...> fmt, T&&... args) const {
        if (mpi_state.rank == 0) {
            fmt::println(fmt, std::forward<T>(args)...);
        }
    }
};

/// The active owned cells.
inline ScCellList<> cell_list(const ScState& state) {
    return state.cells.list();
}

#else
#endif
