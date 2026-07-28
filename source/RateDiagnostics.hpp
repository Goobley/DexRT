#if !defined(DEXRT_RATE_DIAGNOSTICS_HPP)
#define DEXRT_RATE_DIAGNOSTICS_HPP

#include "Config.hpp"
#include "Types.hpp"
#include "DexrtConfig.hpp"
#include "MiscSparse.hpp"

#include <algorithm>
#include <cmath>
#include <string>
#include <vector>

#include <fmt/core.h>

/// [c, i, ks] -- volumetric energy exchanged by continuum i -> c
typedef yakl::Array<RadLossFp, 3, yakl::memDevice> ContEnergyMat;

/**
 * Optional per-transition diagnostics accumulated alongside the Gamma matrix.
 * None of these are used by the solver: they exist purely to be written out.
 * Each vector holds one entry per atom in adata (i.e. the same indexing as
 * State::Gamma), and is empty unless the associated output flag is set.
 */
struct RateDiagnostics {
    /// [i, j, ks] per atom. R(j, i, ks) == R_{i->j}, i.e. the same slot
    /// convention as Gamma, but accumulated with psi_star = 0, so these are the
    /// true (lambda iteration) radiative rates rather than preconditioned ones.
    /// [s-1]
    std::vector<GammaMat> radiative_rates;
    /// [c, i, ks] per atom. Energy absorbed by the photoionisation i -> c,
    /// integrated over direction and wavelength, and multiplied by n_i. [kW m-3]
    std::vector<ContEnergyMat> cont_energy_absorb;
    /// [c, i, ks] per atom. As cont_energy_absorb, but for the paired emission
    /// process (spontaneous + stimulated recombination), times n_c. [kW m-3]
    std::vector<ContEnergyMat> cont_energy_emit;
    /// [total_n_level] 0-based position of each level's ion stage within its own
    /// atom's ascending list of distinct stages. This is the `c` index of the
    /// cont_energy_* arrays, evaluated for the continuum's upper level.
    yakl::Array<i32, 1, yakl::memDevice> level_stage_idx;
    /// [num_atom] number of distinct ion stages present in each model atom, i.e.
    /// the extent of the leading axis of the cont_energy_* arrays.
    std::vector<i32> num_ion_stage;

    /// Fill level_stage_idx/num_ion_stage from the atomic data. Called once per
    /// run, before allocate.
    template <typename T>
    void init_stage_index(const AtomicData<T, yakl::memHost>& adata) {
        const int num_atom = adata.num_level.extent(0);
        const i64 total_n_level = adata.energy.extent(0);

        yakl::Array<i32, 1, yakl::memHost> idx("level_stage_idx", total_n_level);
        num_ion_stage.resize(num_atom);

        std::vector<int> stages;
        for (int ia = 0; ia < num_atom; ++ia) {
            const int start = adata.level_start(ia);
            const int n_level = adata.num_level(ia);

            // NOTE(claude): The levels are normally energy (and therefore stage)
            // ordered, but we don't rely on it here.
            stages.clear();
            for (int l = start; l < start + n_level; ++l) {
                const int stage = int(std::round(adata.stage(l)));
                if (std::find(stages.begin(), stages.end(), stage) == stages.end()) {
                    stages.push_back(stage);
                }
            }
            std::sort(stages.begin(), stages.end());
            num_ion_stage[ia] = i32(stages.size());

            for (int l = start; l < start + n_level; ++l) {
                const int stage = int(std::round(adata.stage(l)));
                const auto loc = std::lower_bound(stages.begin(), stages.end(), stage);
                idx(l) = i32(std::distance(stages.begin(), loc));
            }
        }
        level_stage_idx = idx.createDeviceCopy();
    }

    /// (Re)allocate the enabled diagnostics. The active cell count changes as
    /// the atmosphere is updated, so this may be called repeatedly from Mosscap.
    template <typename T>
    void allocate(
        const AtomicData<T, yakl::memHost>& adata,
        const DexrtOutputConfig& out_cfg,
        i64 num_active_cells
    ) {
        radiative_rates.clear();
        cont_energy_absorb.clear();
        cont_energy_emit.clear();

        if (out_cfg.cont_energy_balance && !level_stage_idx.initialized()) {
            throw std::runtime_error("RateDiagnostics::allocate called before init_stage_index");
        }

        const int num_atom = adata.num_level.extent(0);
        for (int ia = 0; ia < num_atom; ++ia) {
            const int n_level = adata.num_level(ia);
            if (out_cfg.radiative_rates) {
                radiative_rates.emplace_back(
                    GammaMat("radiative_rates", n_level, n_level, num_active_cells)
                );
            }
            if (out_cfg.cont_energy_balance) {
                const int n_stage = num_ion_stage[ia];
                cont_energy_absorb.emplace_back(
                    ContEnergyMat("cont_energy_absorb", n_stage, n_level, num_active_cells)
                );
                cont_energy_emit.emplace_back(
                    ContEnergyMat("cont_energy_emit", n_stage, n_level, num_active_cells)
                );
            }
        }
        zero();
    }

    /// Zeroed everywhere Gamma is, so the arrays hold the final formal solution.
    void zero() const {
        for (const auto& arr : radiative_rates) {
            arr = GammaFp(0.0);
        }
        for (const auto& arr : cont_energy_absorb) {
            arr = RadLossFp(0.0);
        }
        for (const auto& arr : cont_energy_emit) {
            arr = RadLossFp(0.0);
        }
        yakl::fence();
    }
};

/// The bits of the surrounding output conventions that differ between dexrt and
/// Mosscap.
struct RateDiagOutputOpts {
    /// Write on the sparse (ks) grid rather than rehydrating to (z, x)
    bool sparse = false;
    /// Appended to variable names and the ks dim (Mosscap's single-file mode)
    std::string suffix = "";
    std::string z_dim = "z";
    std::string x_dim = "x";
    /// Applied to the cont_energy_* arrays only. 1 leaves them in Dex's native
    /// kW m-3; 1e3 converts to W m-3.
    RadLossFp energy_scale = RadLossFp(1.0);
};

/// Write one [d0, d1, ks] diagnostic, following the sparse/dense conventions of
/// the surrounding save_results.
template <typename T, typename BlockMapT, typename NcT>
inline void write_rate_diag_array(
    const yakl::Array<T, 3, yakl::memDevice>& arr,
    const BlockMapT& block_map,
    NcT& nc,
    const std::string& name,
    const std::string& d0_name,
    const std::string& d1_name,
    const RateDiagOutputOpts& opts
) {
    const i64 d0 = arr.extent(0);
    const i64 d1 = arr.extent(1);
    const i64 num_active_cells = arr.extent(2);

    if (opts.sparse) {
        nc.write(arr, name, {d0_name, d1_name, "ks" + opts.suffix});
        return;
    }

    // NOTE(claude): rehydrate_sparse_quantity only handles [n, ks], so flatten
    // the two leading axes and unflatten the host result. Both reshapes are
    // contiguous no-ops.
    auto flat = arr.reshape(d0 * d1, num_active_cells);
    auto hydrated = rehydrate_sparse_quantity(block_map, flat);
    auto out = hydrated.reshape(d0, d1, hydrated.extent(1), hydrated.extent(2));
    nc.write(out, name, {d0_name, d1_name, opts.z_dim, opts.x_dim});
}

/**
 * Write the enabled rate diagnostics. `collisional_rates` is the buffer the
 * collisional rates were computed into (state.Gamma -- it is dead by output
 * time in both codes); pass nullptr to skip them.
 */
template <typename StateT, typename NcT>
inline void write_rate_diagnostics(
    const StateT& state,
    NcT& nc,
    const RateDiagOutputOpts& opts,
    const std::vector<GammaMat>* collisional_rates = nullptr
) {
    const auto& out_cfg = state.config.output;
    const auto& diag = state.rate_diag;
    const auto& block_map = state.mr_block_map.block_map;

    // NOTE(claude): The two level axes get distinct dim names even though they
    // have the same extent: xarray does not support repeated dims, and warns
    // that most of its functionality then fails silently.
    for (int ia = 0; ia < diag.radiative_rates.size(); ++ia) {
        write_rate_diag_array(
            diag.radiative_rates[ia],
            block_map,
            nc,
            fmt::format("radiative_rates_{}{}", ia, opts.suffix),
            fmt::format("level_to_{}", ia),
            fmt::format("level_from_{}", ia),
            opts
        );
    }

    if (opts.energy_scale != RadLossFp(1.0)) {
        // NOTE(claude): Scaled in place rather than into a copy: nothing else
        // reads these arrays, and they are re-zeroed before the next
        // accumulation.
        for (int ia = 0; ia < diag.cont_energy_absorb.size(); ++ia) {
            const auto& absorb = diag.cont_energy_absorb[ia];
            const auto& emit = diag.cont_energy_emit[ia];
            const RadLossFp scale = opts.energy_scale;
            dex_parallel_for(
                "scale cont energy",
                FlatLoop<3>(absorb.extent(0), absorb.extent(1), absorb.extent(2)),
                YAKL_LAMBDA (int c, int i, i64 ks) {
                    absorb(c, i, ks) *= scale;
                    emit(c, i, ks) *= scale;
                }
            );
        }
        yakl::fence();
    }

    for (int ia = 0; ia < diag.cont_energy_absorb.size(); ++ia) {
        write_rate_diag_array(
            diag.cont_energy_absorb[ia],
            block_map,
            nc,
            fmt::format("cont_energy_absorb_{}{}", ia, opts.suffix),
            fmt::format("ion_stage_{}", ia),
            fmt::format("level_{}", ia),
            opts
        );
        write_rate_diag_array(
            diag.cont_energy_emit[ia],
            block_map,
            nc,
            fmt::format("cont_energy_emit_{}{}", ia, opts.suffix),
            fmt::format("ion_stage_{}", ia),
            fmt::format("level_{}", ia),
            opts
        );
    }

    if (out_cfg.collisional_rates && collisional_rates) {
        // NOTE(claude): indexed by position in adata, like radiative_rates, so
        // collisional_rates_<ia> and radiative_rates_<ia> are the same atom.
        for (int ia = 0; ia < collisional_rates->size(); ++ia) {
            write_rate_diag_array(
                (*collisional_rates)[ia],
                block_map,
                nc,
                fmt::format("collisional_rates_{}{}", ia, opts.suffix),
                fmt::format("level_to_{}", ia),
                fmt::format("level_from_{}", ia),
                opts
            );
        }
    }
}

#else
#endif
