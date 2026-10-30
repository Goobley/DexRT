#include "Config.hpp"
#include "Types.hpp"
#include "ScState.hpp"
#include "Utils.hpp"
#include "Atmosphere.hpp"
#include <limits>
#include "Populations.hpp"
#include "Collisions.hpp"
#include "NgAcceleration.hpp"
#include "ChargeConservation.hpp"
#include "PressureConservation.hpp"
#include "EnergyConservation.hpp"
#include "ProfileNormalisation.hpp"
#include "DexrtConfig.hpp"
#include "MiscSparse.hpp"
#include "Mipmaps.hpp"
#include "WavelengthParallelisation.hpp"
#include "InitialPops.hpp"
#include "ScStateSetup.hpp"
#include "RateDiagnostics.hpp"
#include "GitVersion.hpp"
#include "ScConfig.hpp"
#include "ScAngularQuadrature.hpp"
#include "ScGeometry.hpp"
#include "ScPreparedGeometry.hpp"
#include "ScRadiation.hpp"
#include "ScDecomposition.hpp"
#include "ScMediumCoefficients.hpp"
#include "ScOutput.hpp"
#include "ScIo.hpp"
#include "ScHalo.hpp"
#include "ScDynamicScheduler.hpp"
#include "ScFormalSolution.hpp"
#include "YAKL_netcdf.h"
#include <fmt/core.h>
#include <argparse/argparse.hpp>
#include <sstream>
#include <optional>
#include <array>

#if defined(DEXRT_SC_3D)
constexpr bool SC_3D = true;
#else
constexpr bool SC_3D = false;
#endif

int get_dexrt_dimensionality() {
    return SC_3D ? 3 : 2;
}

struct ScContext {
    ScOptions opts;
    ScSlab slab;
    ScGrid grid; /// this rank's slab
    /// Global cell-centre heights [m]: from the file's "z" (non-uniform grid),
    /// else from offset_z and dz.
    std::vector<f64> z_centres;
    bool nonuniform_z = false;
    std::vector<f64> row_steps; /// sc_row_steps for this slab
    ThermalisedBc th_bc;
    ScDirectionSet dirs;
    ScHalo halo;
    std::unique_ptr<ScFormalSolution> formal_solution;
    std::unique_ptr<ScDynamicScheduler> scheduler;
    i32 wave_batch; /// wavelengths solved together
    Fp3d emergent; /// [la, dir, plane], filled for upward directions on the top slab (3D: if output.sc_emergent)
};

/// Optional anisotropic voxel sizes, falling back to voxel_scale.
void sc_load_grid_spacing(const MpiState& mpi, const std::string& path, ScGrid* grid) {
    ScInFile in(mpi, path);
    if (in.has_var("voxel_scale_x")) {
        grid->dx = in.read_scalar<f32>("voxel_scale_x");
    }
    if (grid->dim == 3 && in.has_var("voxel_scale_y")) {
        grid->dy = in.read_scalar<f32>("voxel_scale_y");
    }
    if (in.has_var("voxel_scale_z")) {
        grid->dz = in.read_scalar<f32>("voxel_scale_z");
    }
}

/// The file's "z" cell-centre heights, if any.
std::vector<f64> sc_read_z_coord(const MpiState& mpi, const std::string& path) {
    ScInFile in(mpi, path);
    if (!in.has_var("z")) {
        return {};
    }
    if (in.shape("z").size() != 1) {
        throw std::runtime_error(fmt::format("\"z\" in {} must be 1D (cell-centre heights)", path));
    }
    return in.read_whole<f64>("z");
}

/// The z grid: uniform (dz, offset_z), or the file's "z" cell-centre heights
/// (increasing upwards; offset_z unused). Sets the slab's node heights and row
/// steps.
void sc_setup_z_grid(ScState* state, ScContext* ctx, const std::string& path) {
    ScGrid& grid = ctx->grid;
    const ScSlab& slab = ctx->slab;
    const i32 nz = slab.global_num_z;
    std::vector<f64> z = sc_read_z_coord(state->mpi_state, path);
    ctx->nonuniform_z = !z.empty();
    if (ctx->nonuniform_z) {
        if (z.size() != nz || nz < 2) {
            throw std::runtime_error(fmt::format("\"z\" in {} has {} entries, expected {}", path, z.size(), nz));
        }
        f64 min_step = std::numeric_limits<f64>::max();
        f64 max_step = 0.0;
        for (int k = 1; k < nz; ++k) {
            const f64 step = z[k] - z[k - 1];
            if (!(step > 0.0)) {
                throw std::runtime_error(fmt::format("\"z\" in {} must be strictly increasing (upwards)", path));
            }
            min_step = std::min(min_step, step);
            max_step = std::max(max_step, step);
        }
        Fp1dHost z_node("sc_z_node", grid.num_z + 2);
        for (int k = -1; k <= grid.num_z; ++k) {
            z_node(k + 1) = fp_t(sc_global_row_height(z, slab.z_start + k));
        }
        grid.z_node = z_node.createDeviceCopy();
        grid.dz = fp_t(min_step);
        ctx->row_steps = sc_row_steps(grid, z, slab.z_start);
        state->println(
            "SC grid: {} x {} (x, z), dx = {} m, non-uniform dz = {} .. {} m, z = {} .. {} m",
            grid.num_x, nz, grid.dx, min_step, max_step, z.front(), z.back()
        );
    } else {
        z.resize(nz);
        for (int k = 0; k < nz; ++k) {
            z[k] = f64(grid.offset_z) + (f64(k) + 0.5) * f64(grid.dz);
        }
        // NOTE(cmo): Positions are global
        grid.offset_z += fp_t(slab.z_start) * grid.dz;
        ctx->row_steps = sc_row_steps(grid, {}, slab.z_start);
        state->println("SC grid: {} x {} (x, z), dx = {} m, dz = {} m", grid.num_x, nz, grid.dx, grid.dz);
    }
    ctx->z_centres = std::move(z);
}

/// Thermalised lower boundary from the bottom row of the atmosphere file.
ThermalisedBc sc_thermalised_bc(const ScState& state, const ScContext& ctx, const std::string& path) {
    if (state.config.mode == DexrtMode::GivenFs) {
        throw std::runtime_error("The Thermalised boundary needs an atmosphere (not GivenFs).");
    }
    const i32 nx = ctx.grid.num_x;
    const i32 ny = ctx.grid.num_y;
    // NOTE(cmo): The bottom row (3D: plane) of a dense [z, (y,) x] temperature.
    ScInFile in(state.mpi_state, path);
    if (in.shape("temperature").size() != (SC_3D ? 3 : 2)) {
        throw std::runtime_error(fmt::format("Thermalised boundary: expected a dense [z, (y,) x] temperature in {}", path));
    }
    std::vector<f64> T(i64(nx) * ny);
    in.read_block("temperature", 0, 0, 1, T.data());
    Fp1dHost T_h("sc_th_bc_temperature", T.size());
    for (i64 i = 0; i < T.size(); ++i) {
        T_h(i) = fp_t(T[i]);
    }
    const auto& z = ctx.z_centres;
    const f64 half_step = 0.5 * (z[1] - z[0]);
    return ThermalisedBc{
        .temperature = T_h.createDeviceCopy(),
        .wavelength = state.adata.wavelength,
        .dx = ctx.grid.dx,
        .offset_x = ctx.grid.offset_x,
        .z_face = fp_t(z[0] - half_step + 0.25 * half_step),
        .num_y = ny,
        .dy = SC_3D ? ctx.grid.dy : FP(1.0),
        .offset_y = ctx.grid.offset_y
    };
}

/// The largest row step over the whole grid (all slabs), including the
/// mirrored steps beyond the ends.
f64 sc_max_global_step(const ScContext& ctx) {
    if (!ctx.nonuniform_z) {
        return f64(ctx.grid.dz);
    }
    const auto& z = ctx.z_centres;
    f64 result = 0.0;
    for (int k = 1; k < z.size(); ++k) {
        result = std::max(result, z[k] - z[k - 1]);
    }
    return result;
}

i32 sc_file_num_z(const MpiState& mpi, const std::string& path) {
    ScInFile in(mpi, path);
    if (in.has_dim("z")) {
        return in.dim("z");
    }
    i32 nz = 0;
    if (!in.att_int("num_z", &nz)) {
        throw std::runtime_error("Atmosphere has no z extent");
    }
    return nz;
}

bool sc_vy_is_zero(const ScState& state) {
    if (state.config.mode == DexrtMode::GivenFs) {
        return true;
    }
    const auto& vy = state.atmos.vy;
    fp_t max_vy = FP(0.0);
    if (vy.extent(0) > 0) {
        Kokkos::parallel_reduce(
            "max |vy|",
            Kokkos::RangePolicy<>(0, vy.extent(0)),
            KOKKOS_LAMBDA(i64 ks, fp_t & m) { m = std::max(m, std::abs(vy(ks))); },
            Kokkos::Max<fp_t>(max_vy));
    }
    return sc_allreduce_max(state.mpi_state, max_vy) == FP(0.0);
}

i32 sc_num_wave(const ScState& state) {
    return (state.config.mode == DexrtMode::GivenFs)
        ? state.given_state.emis.extent(2)
        : state.adata_host.wavelength.extent(0);
}

/// Wavelengths per batch (collective): at most what memory allows, aiming for
/// SC_PIPELINE_TASKS_PER_RANK tasks per remote stream per rank, but with
/// enough (direction x wavelength x cell) work per launch.
i32 sc_choose_wave_batch(const ScState& state, const ScPreparedGeometry& geometry,
                         const ScOptions& opts, i32 num_dirs, i32 num_wave) {
    const auto& grid = geometry.grid;
    const auto& batches = geometry.batches;
    const bool given = state.config.mode == DexrtMode::GivenFs;
    const i32 num_ranks = state.mpi_state.num_ranks;
    const i32 max_dirs = sc_max_batch_dirs(batches);
    const f64 plane = grid.plane_size();
    const f64 cells = plane * grid.num_z;
    const f64 halo_cells = plane * (grid.num_z + 2 * SC_MEDIUM_HALO);
    const f64 base_cells = state.cells.num_cells();
    const f64 M = opts.base_slots;
    const f64 lines = given ? 0 : state.adata.lines.extent(0);
    const f64 levels = given ? 0 : state.pops.extent(0);

    // Fixed: everything allocated so far, the emergent intensity and the
    // bases' per-cell line coefficients.
    f64 fixed = f64(num_wave) * num_dirs * plane * sizeof(fp_t);
    fixed += given ? 0 : M * (base_cells * (4 * lines + levels) * sizeof(fp_t) + 3 * lines * (sizeof(fp_t) + sizeof(i32)));
    if (yakl::use_pool()) {
        const auto& pool = yakl::get_pool();
        for (const auto& entry : pool.allocs) {
            fixed += f64(entry.length) * pool.blockSize;
        }
    } else {
        fixed += state.pops.totElems() * sizeof(fp_t);
        fixed += state.precon_R.totElems() * sizeof(GammaFp);
        fixed += state.cells.num_cells() * 12 * sizeof(fp_t) + state.cells.num_active * 32;
    }
    // Use the same buffer description as allocation, plus transport rings,
    // wavelength sums and coefficient spectra.
    const f64 numerical_buffers = geometry.buffer_values_per_pair();
    const f64 rows = (1 + 4.0 * opts.receive_window) * plane;
    // NOTE(cmo): J slots (and the 16-bit staging), and their previous-J
    // companions for background scattering.
    const f64 J_slots = state.J_cpu.initialized() ? (opts.j_slots ? opts.j_slots : num_ranks + opts.receive_window) : 0;
    const f64 loss_slots = state.rad_loss_integrated.initialized() ? (opts.j_slots ? opts.j_slots : num_ranks + opts.receive_window) : 0;
    const f64 per_wave = sizeof(fp_t) * (max_dirs * (numerical_buffers + rows) +
                                            (given ? 0 : 2 * M * base_cells) + (J_slots + loss_slots) * cells) +
                         (J_slots ? sizeof(Rad16Array::Storage) * cells : 0) +
                         (state.background.enabled ? J_slots * sizeof(Rad16Array::Storage) * halo_cells : 0) +
                         (given ? 0 : 3 * M * (lines + 1) * sizeof(i32));
    const f64 budget = SC_WAVE_BATCH_MEM_FRACTION * state.config.mem_pool_gb * f64(1LL << 30) - fixed;
    i32 memory_upper = budget >= per_wave ? i32(std::min(f64(num_wave), std::floor(budget / per_wave))) : 0;

    i32 min_dirs = max_dirs;
    i32 pipeline_upper = num_wave;
    std::array<i32, 2> remote_batches{};
    for (const auto& batch : batches) {
        min_dirs = std::min(min_dirs, batch.num_dirs);
        if (batch.mode == ScRayMode::Sweep) {
            ++remote_batches[batch.flip_z ? 1 : 0];
        }
    }
    if (num_ranks > 1) {
        for (auto n : remote_batches) {
            if (n) {
                const i64 wave_batches = (i64(SC_PIPELINE_TASKS_PER_RANK) * num_ranks + n - 1) / n;
                pipeline_upper = std::min(pipeline_upper, i32(std::max<i64>(1, num_wave / wave_batches)));
            }
        }
    }
    i32 work_floor = std::max(1, i32(std::ceil(32768.0 / std::max(1.0, cells * min_dirs))));
    if constexpr (HostDevSameSpace) {
        work_floor = std::max(work_floor, (2 * Kokkos::DefaultExecutionSpace().concurrency() + min_dirs - 1) / min_dirs);
    }
#ifdef HAVE_MPI
    if (num_ranks > 1) {
        MPI_Allreduce(MPI_IN_PLACE, &memory_upper, 1, MPI_INT, MPI_MIN, state.mpi_state.comm);
        MPI_Allreduce(MPI_IN_PLACE, &work_floor, 1, MPI_INT, MPI_MAX, state.mpi_state.comm);
    }
#endif
    if (memory_upper < 1) {
        throw std::runtime_error("SC memory budget cannot fit one wavelength");
    }
    const i32 batch = opts.wave_batch ? std::min(opts.wave_batch, num_wave)
                                      : std::min(memory_upper, std::max(pipeline_upper, std::min(work_floor, num_wave)));
    if (batch > memory_upper) {
        throw std::runtime_error("sc.wave_batch exceeds the collective memory bound");
    }
    state.println("SC: M={} W={}, wave batch {} (memory <= {}, pipeline <= {}, work >= {}), estimated rank-0 working set {:.3f} GB",
        opts.base_slots,
        opts.receive_window,
        batch,
        memory_upper,
        pipeline_upper,
        work_floor,
        (fixed + batch * per_wave) / f64(1LL << 30));
    return batch;
}

void sc_setup_prd(ScState* state, const ScOptions& opts);

void init_state_sc(ScState* state, ScContext* ctx, const DexrtConfig& config_in, const ScOptions& opts) {
    DexrtConfig config(config_in);
    // NOTE(cmo): No mips in SC
    config.max_cascade = 0;
    config.mip_config.mip_levels = {0};

    state->config = config;
    setup_comm(state);
    if (config.output.sparse) {
        state->println("SC: output.sparse is ignored (SC output is always dense).");
    }
    const bool distributed = state->mpi_state.num_ranks > 1;
    state->mpi_state.distributed = distributed;

    // NOTE(cmo): Equal slabs of whole activity tile rows, so each rank finds
    // its own tiles' activity.
    const i32 unit = config.mode == DexrtMode::GivenFs ? 1 : (SC_3D ? SC_TILE_3D : BLOCK_SIZE);
    ctx->slab = sc_decompose_z(sc_file_num_z(state->mpi_state, config.atmos_path), state->mpi_state, unit);
    state->slab = ctx->slab;
    const ScSlab& slab = ctx->slab;

    ScGrid& grid = ctx->grid;
    if (config.mode == DexrtMode::Lte || config.mode == DexrtMode::NonLte) {
        init_atmos_atoms(state, state->config);
        grid = ScGrid{
            .num_x = state->atmos.num_x,
            .num_y = SC_3D ? state->atmos.num_y : 1,
            .num_z = state->atmos.num_z,
            .dx = state->atmos.voxel_scale,
            .dy = SC_3D ? state->atmos.voxel_scale : FP(0.0),
            .dz = state->atmos.voxel_scale,
            .offset_x = state->atmos.offset_x,
            .offset_y = state->atmos.offset_y,
            .offset_z = state->atmos.offset_z
        };
        if (!SC_3D && grid.num_x % BLOCK_SIZE) {
            throw std::runtime_error(fmt::format("SC solver requires atmosphere dimensions to be multiples of BLOCK_SIZE ({})", BLOCK_SIZE));
        }
    } else {
        init_given_emis_opac(state, state->config);
        const i32 num_y = state->given_state.num_y;
        grid = ScGrid{
            .num_x = i32(state->given_state.emis.extent(1)) / num_y,
            .num_y = num_y,
            .num_z = slab.num_z(),
            .dx = state->given_state.voxel_scale,
            .dy = SC_3D ? state->given_state.voxel_scale : FP(0.0),
            .dz = state->given_state.voxel_scale,
            .offset_x = FP(0.0),
            .offset_y = FP(0.0),
            .offset_z = FP(0.0)
        };
    }
    grid.dim = SC_3D ? 3 : 2;
    if (config.mode == DexrtMode::GivenFs) {
        ScActiveMaskHost active("sc_given_active", grid.num_z, grid.plane_size());
        active = u8(1);
        state->cells.init(active, slab.z_start, SC_CELL_GROUP, grid.num_y);
    }
    ctx->halo.init(*state);
    ctx->halo.exchange_static(state);
    sc_load_grid_spacing(state->mpi_state, config.atmos_path, &grid);
    sc_setup_z_grid(state, ctx, config.atmos_path);
    if (config.boundary == BoundaryType::Thermalised) {
        ctx->th_bc = sc_thermalised_bc(*state, *ctx, config.atmos_path);
        const auto T = ctx->th_bc.temperature.createHostCopy();
        state->println("SC: thermalised lower boundary, T[0] = {} K, face below z = {} m", T(0), ctx->th_bc.z_face);
    }
    if (distributed) {
        state->println("SC: z decomposed over {} ranks (rank 0 rows [{}, {}))", slab.num_ranks, slab.z_start, slab.z_end);
    }

    allocate_J(state);
    allocate_rad_loss(state);
    sc_setup_prd(state, opts);

    ctx->opts = opts;
    // NOTE(cmo): 2D: folded in y when the problem is symmetric in y.
    const bool fold_y = !SC_3D && sc_vy_is_zero(*state);
    ctx->dirs = build_sc_direction_set(opts, fold_y, SC_3D ? 3 : 2);
    ScPreparedGeometry geometry(grid, slab, ctx->dirs, opts, ctx->z_centres,
                                ctx->row_steps, sc_max_global_step(*ctx));
    const i32 num_wave = sc_num_wave(*state);
    ctx->wave_batch = sc_choose_wave_batch(*state, geometry, opts, ctx->dirs.num_dirs, num_wave);
    state->println(
        "SC: {} quadrature (order {}), {} directions{}, {} batches; {} + {}, x boundary {}{}",
        ScQuadratureTypeNames[int(opts.quadrature)],
        opts.quadrature_order,
        ctx->dirs.num_dirs,
        fold_y ? " (folded in y)" : "",
        geometry.batches.size(),
        ScFormalSolverNames[int(opts.formal_solver)],
        ScInterpNames[int(opts.interpolation)],
        ScXBoundaryNames[int(opts.x_boundary)],
        SC_3D ? fmt::format(", y boundary {}", ScXBoundaryNames[int(opts.y_boundary)]) : ""
    );

    if (!SC_3D || config.output.sc_emergent) {
        ctx->emergent = Fp3d("sc_emergent", num_wave, ctx->dirs.num_dirs, grid.plane_size());
        ctx->emergent = FP(0.0);
    }
    ctx->formal_solution = std::make_unique<ScFormalSolution>(sc_make_formal_solution(
        *state,
        std::move(geometry),
        opts,
        ctx->wave_batch,
        ctx->th_bc
    ));
    ctx->scheduler = std::make_unique<ScDynamicScheduler>(state->mpi_state, opts.receive_window);

#ifdef DEXRT_USE_MAGMA
    magma_init();
    magma_queue_create(0, &state->magma_queue);
#endif
}


/// One formal solution: update the inputs (and their halo rows), then
/// run every task.
/// prd_only: only the wavelengths reaching the PRD lines, for J* (J is left
/// as it is).
void sc_formal_solution(ScState& state, ScContext& ctx, bool compute_gamma, bool lambda_iterate, bool prd_only = false) {
    if (state.config.mode != DexrtMode::GivenFs) {
        compute_nh0(state);
    }
    // NOTE(cmo): The background follows the current populations (as the
    // halo rows do) in every solve.
    update_background(&state);
    yakl::timer_start("SC halo exchange");
    ctx.halo.exchange_inputs(&state);
    yakl::timer_stop("SC halo exchange");
    if (state.prd.enabled()) {
        state.prd.J_rest.fill(FP(0.0));
    }
    yakl::fence();
    const auto plan = sc_begin_solve(
        *ctx.formal_solution,
        state,
        ctx.emergent,
        compute_gamma,
        lambda_iterate,
        prd_only
    );
    ctx.scheduler->run(plan, *ctx.formal_solution);
    yakl::fence();
    if (!prd_only) {
        state.background.have_J = true;
    }
}

/// The PRD lines (non-LTE, sc.prd: Hybrid), with the radiative rates they need.
void sc_setup_prd(ScState* state, const ScOptions& opts) {
    const bool enabled = opts.prd == ScPrdMode::Hybrid && state->config.mode == DexrtMode::NonLte;
    state->prd = sc_prd_init(*state, enabled);
    ScPrd& prd = state->prd;
    if (!prd.enabled()) {
        return;
    }
    if (!state->rate_diag.radiative_rates.initialized()) {
        DexrtOutputConfig out = state->config.output;
        out.radiative_rates = true;
        state->rate_diag.allocate(state->adata_host, out, state->cells.num_cells());
    }
    const auto flat_atmos = flatten<const fp_t>(state->atmos);
    const auto cells = cell_list(*state);
    fp_t v_max = FP(0.0);
    dex_parallel_reduce(
        "SC PRD max speed",
        FlatLoop<1>(cells.num()),
        KOKKOS_LAMBDA(i64 i, fp_t& m) {
            const i64 ks = cells(i);
            m = std::max(m, std::sqrt(square(flat_atmos.vx(ks)) + square(flat_atmos.vy(ks)) + square(flat_atmos.vz(ks))));
        },
        Kokkos::Max<fp_t>(v_max));
    prd.max_speed = sc_allreduce_max(state->mpi_state, v_max);
    state->println("SC PRD: {} lines, rho* rows {}, sub-iteration wavelengths [{}, {}), |v| <= {:.1f} km/s",
        prd.num_lines, prd.num_rows, prd.reach_lo, prd.reach_hi, prd.max_speed * FP(1e-3));
}

void add_sc_netcdf_attributes(const ScState& state, const ScContext& ctx, ScOutFile& file, i32 num_iter) {
    file.put_att("program", std::string(SC_3D ? "dexrt_sc (3d)" : "dexrt_sc (2d)"));
#ifdef DEXRT_SINGLE_PREC
    file.put_att("precision", std::string("f32"));
#else
    file.put_att("precision", std::string("f64"));
#endif
    file.put_att("sc_quadrature", std::string(ScQuadratureTypeNames[int(ctx.opts.quadrature)]));
    file.put_att("sc_quadrature_order", i32(ctx.opts.quadrature_order));
    file.put_att("sc_formal_solver", std::string(ScFormalSolverNames[int(ctx.opts.formal_solver)]));
    file.put_att("sc_interpolation", std::string(ScInterpNames[int(ctx.opts.interpolation)]));
    file.put_att("sc_x_boundary", std::string(ScXBoundaryNames[int(ctx.opts.x_boundary)]));
    if (SC_3D) {
        file.put_att("sc_y_boundary", std::string(ScXBoundaryNames[int(ctx.opts.y_boundary)]));
    }
    file.put_att("sc_characteristics", std::string(ScCharacteristicsNames[int(ctx.opts.characteristics)]));
    if (ctx.opts.characteristics == ScCharacteristics::LongerCharacteristics) {
        // NOTE(cmo): Emergent intensities are those of the characteristics
        // leaving the top face, at (canonical) x = s - 1/2 + phase.
        const std::vector<f64> faces = sc_canonical_faces(ctx.z_centres, false);
        std::vector<f64> phase(ctx.dirs.num_dirs, 0.0);
        std::vector<f64> phase_y(ctx.dirs.num_dirs, 0.0);
        for (int d = 0; d < ctx.dirs.num_dirs; ++d) {
            const f64 az = std::abs(ctx.dirs.mu_host(d, 2));
            if (az > 0.0) {
                const f64 A = faces.back() * std::abs(ctx.dirs.mu_host(d, 0)) / (az * ctx.grid.dx);
                phase[d] = A - std::floor(A);
                if (SC_3D) {
                    const f64 Ay = faces.back() * std::abs(ctx.dirs.mu_host(d, 1)) / (az * ctx.grid.dy);
                    phase_y[d] = Ay - std::floor(Ay);
                }
            }
        }
        file.put_att("sc_longer_phase_top", phase);
        if (SC_3D) {
            file.put_att("sc_longer_phase_top_y", phase_y);
        }
    }
    file.put_att("sc_folded_y", i32(ctx.dirs.folded_y));
    file.put_att("sc_boundary_cone_mu", i32(ctx.opts.boundary_cone_mu));
    file.put_att("sc_boundary_cone_phi", i32(ctx.opts.boundary_cone_phi));
    file.put_att("sc_max_row_crossings", f64(SC_MAX_ROW_CROSSINGS));
    file.put_att("num_ranks", i32(state.mpi_state.num_ranks));
    file.put_att("sc_io", std::string(file.is_parallel() ? "Parallel" : "Gather"));
    file.put_att("voxel_scale", f64(state.config.mode == DexrtMode::GivenFs ? state.given_state.voxel_scale : state.atmos.voxel_scale));
    file.put_att("voxel_scale_x", f64(ctx.grid.dx));
    if (SC_3D) {
        file.put_att("voxel_scale_y", f64(ctx.grid.dy));
    }
    if (!ctx.nonuniform_z) {
        file.put_att("voxel_scale_z", f64(ctx.grid.dz));
    }
    file.put_att("sc_uniform_z", i32(!ctx.nonuniform_z));
    file.put_att("wave_batch", i32(ctx.wave_batch));
    file.put_att("block_size", i32(BLOCK_SIZE));
    file.put_att("line_calculation_scheme", std::string(LineCoeffCalcNames[int(LINE_SCHEME)]));
    file.put_att("num_iter", i32(num_iter));
    file.put_att("output_format", std::string("full"));
    file.put_att("git_hash", std::string(GIT_HASH));
    file.put_att("config_path", state.config.own_path);

    if (state.adata_host.num_level.initialized()) {
        const auto& a = state.adata_host;
        const i32 num_atom = a.num_level.extent(0);
        file.put_att("num_atom", num_atom);
        file.put_att("num_level", std::vector<i32>(a.num_level.data(), a.num_level.data() + num_atom));
        file.put_att("num_line", std::vector<i32>(a.num_line.data(), a.num_line.data() + a.num_line.extent(0)));
        file.put_att("line_start", std::vector<i32>(a.line_start.data(), a.line_start.data() + a.line_start.extent(0)));
        std::vector<f64> lambda0(a.lines.extent(0));
        for (int i = 0; i < lambda0.size(); ++i) {
            lambda0[i] = a.lines(i).lambda0;
        }
        file.put_att("lambda0", lambda0);
    }

    // NOTE(cmo): Rank 0's timers.
    std::string timing;
    if (state.mpi_state.rank == 0) {
        auto cout_buf = std::cout.rdbuf();
        std::ostringstream timer_buffer;
        std::cout.rdbuf(timer_buffer.rdbuf());
        yakl::timer_finalize();
        std::cout.rdbuf(cout_buf);
        timing = timer_buffer.str();
    }
    file.put_att("timing", timing);
}

void save_results_sc(ScState& state, const ScContext& ctx, i32 num_iter) {
    const auto& config = state.config;
    const auto& out_cfg = config.output;
    const auto& mpi = state.mpi_state;
    const auto& slab = ctx.slab;

    state.println("Saving output to {}...", config.output_path);
    ScOutFile file(mpi, config.output_path, out_cfg.io);
    add_sc_netcdf_attributes(state, ctx, file, num_iter);
    yakl::Array<f64, 1, yakl::memHost> z("z", ctx.z_centres.size());
    for (int k = 0; k < ctx.z_centres.size(); ++k) {
        z(k) = ctx.z_centres[k];
    }
    file.write(z, "z", {"z"});

    ScOutputWriter writer(file, slab, state.cells);
    if (out_cfg.J) {
        writer.write_owned(state.J_cpu, "J", "wavelength");
    }
    if (out_cfg.rad_loss && state.rad_loss_cpu.initialized()) {
        const auto& L = state.rad_loss_cpu;
        writer.write_dense(L.reshape(L.extent(0), state.cells.num_z, state.cells.plane_size()),
            "rad_loss",
            config.rad_loss == RadLossType::Integrated ? "wavelength_integrated" : "wavelength");
    }

    // NOTE(cmo): The emergent intensity lives on the top slab.
    if (ctx.emergent.initialized()) {
        const Fp3dHost emergent = ctx.emergent.createHostCopy();
        const i32 top = mpi.num_ranks - 1;
        if (SC_3D) {
            file.write_whole("I_emergent",
                {"wavelength", "sc_dir", "y_emergent", "x_emergent"},
                {i64(emergent.extent(0)), i64(emergent.extent(1)), ctx.grid.num_y, ctx.grid.num_x},
                emergent.data(),
                top);
        } else {
            file.write_whole("I_emergent",
                {"wavelength", "sc_dir", "x_emergent"},
                {i64(emergent.extent(0)), i64(emergent.extent(1)), i64(emergent.extent(2))},
                emergent.data(),
                top);
        }
    }
    file.write(ctx.dirs.mu_host, "sc_mu", {"sc_dir", "three"});
    file.write(ctx.dirs.wmu_host, "sc_wmu", {"sc_dir"});
    if (out_cfg.wavelength && state.adata.wavelength.initialized()) {
        file.write(state.adata.wavelength.createHostCopy(), "wavelength", {"wavelength"});
        const auto& bins = state.adata_host.wavelength_bin;
        Fp1dHost bin_h("wavelength_bin", bins.extent(0));
        for (i32 la = 0; la < bins.extent(0); ++la) {
            bin_h(la) = bins(la);
        }
        file.write(bin_h, "wavelength_bin", {"wavelength"});
    }
    if (state.prd.enabled()) {
        // NOTE(cmo): rho* per PRD line on its own wavelengths: rows
        // [prd_row_start, +prd_row_count) of prd_rho are wavelengths
        // [prd_blue, +prd_row_count) of line prd_kr.
        writer.write_cells(state.prd.rho.to_fp(), "prd_rho", "prd_row");
        writer.write_cells(state.prd.J_rest.to_fp(), "prd_J_rest", "prd_row");
        const auto lines = state.prd.lines.createHostCopy();
        yakl::Array<i32, 2, yakl::memHost> table("prd_lines", state.prd.num_lines, 4);
        for (int p = 0; p < state.prd.num_lines; ++p) {
            table(p, 0) = lines(p).kr;
            table(p, 1) = lines(p).row0;
            table(p, 2) = lines(p).red - lines(p).blue;
            table(p, 3) = lines(p).blue;
        }
        file.write(table, "prd_lines", {"prd_line", "prd_kr_row_start_count_blue"});
    }
    if (config.mode == DexrtMode::GivenFs) {
        return;
    }

    if (out_cfg.pops) {
        writer.write_cells(state.pops, "pops", "level");
    }
    if (out_cfg.lte_pops) {
        Fp2d lte_pops = state.pops.createDeviceObject();
        compute_lte_pops(&state, lte_pops);
        yakl::fence();
        writer.write_cells(lte_pops, "lte_pops", "level");
    }
    if (out_cfg.temperature) {
        writer.write_scalar(state.atmos.temperature, "temperature");
    }
    if (out_cfg.ne) {
        writer.write_scalar(state.atmos.ne, "ne");
    }
    if (out_cfg.nh_tot) {
        writer.write_scalar(state.atmos.nh_tot, "nh_tot");
    }
    if (out_cfg.active) {
        writer.write_active();
    }
    const auto& diag = state.rate_diag;
    const i32 num_atom = (out_cfg.radiative_rates && diag.radiative_rates.initialized()) ? state.adata_host.num_level.extent(0) : 0;
    for (i32 ia = 0; ia < num_atom; ++ia) {
        const std::string trans_dim = fmt::format("transition_{}", ia);
        writer.write_tensor(atom_precon_R(diag.radiative_rates, state.adata_host, ia),
            fmt::format("radiative_rates_{}", ia),
            trans_dim,
            "up_down");
        write_transition_table(state.adata_host, ia, file, trans_dim, RateDiagOutputOpts{});
    }
    for (i32 ia = 0; ia < diag.cont_energy_absorb.size(); ++ia) {
        writer.write_tensor(diag.cont_energy_absorb[ia],
            fmt::format("cont_energy_absorb_{}", ia),
            fmt::format("ion_stage_{}", ia),
            fmt::format("level_{}", ia));
        writer.write_tensor(diag.cont_energy_emit[ia],
            fmt::format("cont_energy_emit_{}", ia),
            fmt::format("ion_stage_{}", ia),
            fmt::format("level_{}", ia));
    }
    if (out_cfg.collisional_rates) {
        const CollisionalRatesOutput coll = compute_collisional_rates_output(state);
        for (i32 ia = 0; ia < coll.rates.size(); ++ia) {
            const std::string pair_dim = fmt::format("collision_pair_{}", ia);
            writer.write_tensor(coll.rates[ia], fmt::format("collisional_rates_{}", ia), pair_dim, "up_down");
            file.write(coll.pairs[ia], fmt::format("collision_pair_levels_{}", ia), {pair_dim, "lower_upper"});
        }
    }
}

int main(int argc, char** argv) {
    init_mpi(argc, argv);

    argparse::ArgumentParser program("DexRT SC");
    program.add_argument("--config")
        .default_value(std::string("dexrt.yaml"))
        .help("Path to config file")
        .metavar("FILE");
    program.add_argument("--restart-from")
        .nargs(1)
        .help("Path to snapshot file")
        .metavar("FILE");
    program.add_epilog(SC_3D ? "DexRT short characteristics non-LTE solver (3d)." : "DexRT short characteristics non-LTE solver (2d).");
    program.parse_known_args(argc, argv);

    std::optional<std::string> restart_path = program.present("--restart-from");
    const std::string config_path = program.get<std::string>("--config");
    const DexrtConfig config = load_and_parse_dexrt_config(config_path);
    const ScOptions sc_opts = parse_sc_options(YAML::LoadFile(config_path));

    Kokkos::initialize(argc, argv);
    yakl::init(
        yakl::InitConfig()
            .set_pool_size_mb(config.mem_pool_gb * 1024)
    );

    {
        ScState state;
        ScContext ctx;
        init_state_sc(&state, &ctx, config, sc_opts);
        const auto& mpi = state.mpi_state;
        yakl::timer_start("DexRT SC");

        if (config.mode == DexrtMode::GivenFs) {
            sc_formal_solution(state, ctx, false, true);
            yakl::timer_stop("DexRT SC");
            save_results_sc(state, ctx, 1);
        } else {
            compute_lte_pops(&state);
            if (state.config.initial_pops_path.size() > 0) {
                load_initial_pops(&state, state.config.initial_pops_path);
            }
            const bool non_lte = config.mode == DexrtMode::NonLte;
            const bool do_restart = bool(restart_path);
            const bool pops_provided = do_restart || (config.initial_pops_path.size() > 0);
            const bool conserve_charge = config.conserve_charge;
            const bool actually_conserve_charge = state.have_h && conserve_charge;
            if (!actually_conserve_charge && conserve_charge) {
                throw std::runtime_error("Charge conservation enabled without a model H!");
            }
            const bool conserve_pressure = config.conserve_pressure;
            const bool conserve_energy = config.conserve_energy;
            if (
                (conserve_pressure && !conserve_charge)
                || (conserve_energy && !conserve_charge)
            ) {
                throw std::runtime_error("Cannot enable pressure or energy conservation without charge conservation.");
            }
            const bool actually_conserve_pressure = actually_conserve_charge && conserve_pressure;
            const bool actually_conserve_energy = actually_conserve_charge && conserve_energy;
            const int initial_lambda_iterations = config.initial_lambda_iterations;
            const int max_iters = config.max_iter;
            const fp_t non_lte_tol = config.pop_tol;
            fp_t max_change = FP(1.0);
            int num_iter = 1;

            if (non_lte) {
                int i = 0;
                if (actually_conserve_charge && !pops_provided) {
                    state.println("-- Iterating LTE n_e/pressure --");
                    fp_t lte_max_change = FP(1.0);
                    int lte_i = 0;
                    while ((lte_max_change > FP(1e-5) || lte_i < 6) && lte_i < max_iters) {
                        lte_i += 1;
                        compute_nh0(state);
                        lte_max_change = sc_allreduce_max(mpi, stat_eq(&state, StatEqOptions{
                            .ignore_change_below_ntot_frac=FP(1e-7)
                        }));
                        if (lte_i < 2) {
                            continue;
                        }
                        fp_t nr_update = sc_allreduce_max(mpi, nr_post_update(&state, NrPostUpdateOptions{
                            .ignore_change_below_ntot_frac = FP(1e-7),
                            .conserve_pressure = false
                        }));
                        lte_max_change = nr_update;
                        if (actually_conserve_pressure) {
                            fp_t nh_tot_update = sc_allreduce_max(mpi, simple_conserve_pressure(&state));
                            lte_max_change = std::max(nh_tot_update, lte_max_change);
                        }
                    }
                    state.println("Ran for {} iterations", lte_i);
                }

                if (!pops_provided) {
                    set_initial_pops_special(&state);
                }
                if (do_restart) {
                    i = handle_restart(&state, *restart_path);
                }

                state.println("-- Non-LTE Iterations ({} wavelengths) --", state.adata_host.wavelength.extent(0));
                NgAccelerator ng;
                if (config.ng.enable) {
                    ng.init(
                        NgAccelArgs{
                            .num_level=(i64)state.pops.extent(0),
                            .num_space=(i64)state.pops.extent(1),
                            .accel_tol=config.ng.threshold,
                            .lower_tol=config.ng.lower_threshold,
                            .distributed=(mpi.num_ranks > 1)
                        }
                    );
                    ng.accelerate(state, FP(1.0));
                }
                bool first_iter = true;
                bool accelerated = false;
                while (((max_change > non_lte_tol || i < (initial_lambda_iterations+1)) && i < max_iters) || accelerated) {
                    state.println("==== FS {} ====", i);
                    compute_nh0(state);
                    zero_precon_R(state.precon_R);
                    if (state.prd.enabled()) {
                        sc_prd_capture_collisions(state, state.prd);
                    }
                    state.rate_diag.zero();
                    compute_profile_normalisation(state, ctx.dirs.mu, ctx.dirs.wmu, first_iter);

                    const bool lambda_iterate = i < initial_lambda_iterations;
                    const StatEqOptions stat_eq_opts{
                        .ignore_change_below_ntot_frac=std::min(FP(1e-6), non_lte_tol)
                    };
                    sc_formal_solution(state, ctx, true, lambda_iterate);
                    state.println("  == Statistical equilibrium ==");
                    max_change = sc_allreduce_max(mpi, stat_eq(&state, stat_eq_opts));
                    if (i > 0 && actually_conserve_charge) {
                        fp_t nr_update = sc_allreduce_max(mpi, nr_post_update(
                            &state,
                            NrPostUpdateOptions{
                                .ignore_change_below_ntot_frac = std::min(FP(1e-6), non_lte_tol),
                                .conserve_pressure = CONSERVE_PRESSURE_NR && actually_conserve_pressure
                            }
                        ));
                        max_change = std::max(nr_update, max_change);
                        if (!CONSERVE_PRESSURE_NR && actually_conserve_pressure) {
                            fp_t nh_tot_update = sc_allreduce_max(mpi, simple_conserve_pressure(&state));
                            max_change = std::max(nh_tot_update, max_change);
                        }
                        if (actually_conserve_energy) {
                            fp_t temp_update = sc_allreduce_max(mpi, simple_conserve_energy(&state));
                            max_change = std::max(temp_update, max_change);
                        }
                    }
                    if (config.ng.enable) {
                        accelerated = ng.accelerate(state, max_change);
                        if (accelerated) {
                            state.println("  ~~ Ng Acceleration! (📉 or 💣) ~~");
                        }
                    }
                    if (state.prd.enabled() && i >= ctx.opts.prd_start_iter) {
                        // NOTE(cmo): Redistribute with the populations fixed,
                        // re-solving only the wavelengths reaching the PRD lines.
                        yakl::timer_start("SC PRD");
                        for (int s = 0; s < ctx.opts.prd_subiterations; ++s) {
                            const fp_t d_rho = sc_allreduce_max(mpi, sc_prd_redistribute(state, state.prd));
                            state.println("  == PRD sub-iteration {}: max drho/rho = {:.3e} ==", s, d_rho);
                            sc_formal_solution(state, ctx, false, false, true);
                            if (s + 1 >= SC_PRD_MIN_SUBITERATIONS && d_rho < ctx.opts.prd_tolerance) {
                                break;
                            }
                        }
                        yakl::timer_stop("SC PRD");
                    }
                    i += 1;
                    if (
                        (state.config.snapshot_frequency != 0) &&
                        (i % state.config.snapshot_frequency == 0)
                    ) {
                        save_snapshot(state, i);
                    }
                    first_iter = false;
                }
                if (state.config.sparse_calculation && state.config.final_dense_fs) {
                    state.config.sparse_calculation = false;
                    state.println("Final FS (dense)");
                    state.rate_diag.zero();
                    sc_formal_solution(state, ctx, false, true);
                    state.config.sparse_calculation = true;
                }
                num_iter = i;
            } else {
                compute_nh0(state);
                sc_formal_solution(state, ctx, false, true);
            }
            yakl::timer_stop("DexRT SC");
            save_results_sc(state, ctx, num_iter);
        }
    }
    yakl::finalize();
    finalise_mpi();
    Kokkos::finalize();
    return 0;
}
