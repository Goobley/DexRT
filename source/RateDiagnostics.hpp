#if !defined(DEXRT_RATE_DIAGNOSTICS_HPP)
#define DEXRT_RATE_DIAGNOSTICS_HPP

#include "Config.hpp"
#include "Types.hpp"
#include "DexrtConfig.hpp"
#include "MiscSparse.hpp"
#include "TransitionIndex.hpp"

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
    /// [t, 2, ks], the same layout as precon_R (each atom's lines, then
    /// continua; slot 0 up i -> j, 1 down j -> i), but accumulated with
    /// psi_star = 0, so these are the true (lambda iteration) radiative rates
    /// rather than preconditioned ones. [s-1]
    PreconRMat radiative_rates;
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
        radiative_rates = PreconRMat();
        cont_energy_absorb.clear();
        cont_energy_emit.clear();

        if (out_cfg.cont_energy_balance && !level_stage_idx.initialized()) {
            throw std::runtime_error("RateDiagnostics::allocate called before init_stage_index");
        }

        if (out_cfg.radiative_rates) {
            const int n_trans = adata.lines.extent(0) + adata.continua.extent(0);
            radiative_rates = PreconRMat("radiative_rates", n_trans, 2, num_active_cells);
        }
        const int num_atom = adata.num_level.extent(0);
        for (int ia = 0; ia < num_atom; ++ia) {
            const int n_level = adata.num_level(ia);
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
        if (radiative_rates.initialized()) {
            radiative_rates = GammaFp(0.0);
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

struct CollisionalRatesOutput {
    std::vector<Fp3d> rates;
    std::vector<yakl::Array<i32, 2, yakl::memHost>> pairs;
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

/// The transitions of atom ia, as indexed by precon_R and radiative_rates:
/// transition_levels_<ia> [t, 2] (lower i, upper j, 0-based within the atom),
/// transition_type_<ia> [t] (0 line, 1 continuum), transition_lambda0_<ia> [t]
/// (line centre, or the edge of the continuum) [nm], the
/// level energies [eV] and statistical weights g.
template <typename AtomicDataHost, typename NcT>
inline void write_transition_table(
    const AtomicDataHost& adata,
    int ia,
    NcT& nc,
    const std::string& trans_dim,
    const RateDiagOutputOpts& opts
) {
    const int num_line = adata.num_line(ia);
    const int num_trans = num_line + adata.num_cont(ia);
    yakl::Array<i32, 2, yakl::memHost> levels("transition_levels", num_trans, 2);
    yakl::Array<i32, 1, yakl::memHost> type("transition_type", num_trans);
    yakl::Array<f64, 1, yakl::memHost> lambda0("transition_lambda0", num_trans);
    for (int t = 0; t < num_trans; ++t) {
        if (t < num_line) {
            const auto& l = adata.lines(adata.line_start(ia) + t);
            levels(t, 0) = l.i;
            levels(t, 1) = l.j;
            type(t) = 0;
            lambda0(t) = l.lambda0;
        } else {
            const auto& c = adata.continua(adata.cont_start(ia) + t - num_line);
            levels(t, 0) = c.i;
            levels(t, 1) = c.j;
            type(t) = 1;
            lambda0(t) = adata.wavelength(c.red_idx - 1);
        }
    }
    const int num_level = adata.num_level(ia);
    yakl::Array<f64, 1, yakl::memHost> energy("level_energy", num_level);
    yakl::Array<f64, 1, yakl::memHost> g("level_g", num_level);
    for (int i = 0; i < num_level; ++i) {
        energy(i) = adata.energy(adata.level_start(ia) + i);
        g(i) = adata.g(adata.level_start(ia) + i);
    }
    const std::string level_dim = fmt::format("level_{}{}", ia, opts.suffix);
    nc.write(levels, fmt::format("transition_levels_{}{}", ia, opts.suffix), {trans_dim, "lower_upper" + opts.suffix});
    nc.write(type, fmt::format("transition_type_{}{}", ia, opts.suffix), {trans_dim});
    nc.write(lambda0, fmt::format("transition_lambda0_{}{}", ia, opts.suffix), {trans_dim});
    nc.write(energy, fmt::format("level_energy_{}{}", ia, opts.suffix), {level_dim});
    nc.write(g, fmt::format("level_g_{}{}", ia, opts.suffix), {level_dim});
}

/**
 * Write the enabled rate diagnostics. `collisional_rates` are the per-pair
 * collisional rates (compute_collisional_rates_output); pass nullptr to skip
 * them.
 */
template <typename StateT, typename NcT>
inline void write_rate_diagnostics(
    const StateT& state,
    NcT& nc,
    const RateDiagOutputOpts& opts,
    const CollisionalRatesOutput* collisional_rates = nullptr
) {
    const auto& out_cfg = state.config.output;
    const auto& diag = state.rate_diag;
    const auto& block_map = state.mr_block_map.block_map;

    // NOTE(claude): The two level axes get distinct dim names even though they
    // have the same extent: xarray does not support repeated dims, and warns
    // that most of its functionality then fails silently.
    const int num_atom = diag.radiative_rates.initialized() ? state.adata_host.num_level.extent(0) : 0;
    for (int ia = 0; ia < num_atom; ++ia) {
        const std::string trans_dim = fmt::format("transition_{}{}", ia, opts.suffix);
        write_rate_diag_array(
            atom_precon_R(diag.radiative_rates, state.adata_host, ia),
            block_map,
            nc,
            fmt::format("radiative_rates_{}{}", ia, opts.suffix),
            trans_dim,
            "up_down" + opts.suffix,
            opts
        );
        write_transition_table(state.adata_host, ia, nc, trans_dim, opts);
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
        for (int ia = 0; ia < collisional_rates->rates.size(); ++ia) {
            const std::string pair_dim = fmt::format("collision_pair_{}{}", ia, opts.suffix);
            write_rate_diag_array(
                collisional_rates->rates[ia],
                block_map,
                nc,
                fmt::format("collisional_rates_{}{}", ia, opts.suffix),
                pair_dim,
                "up_down" + opts.suffix,
                opts
            );
            nc.write(
                collisional_rates->pairs[ia],
                fmt::format("collision_pair_levels_{}{}", ia, opts.suffix),
                {pair_dim, "lower_upper" + opts.suffix}
            );
        }
    }
}

#else
#endif
