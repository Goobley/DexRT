#include "ScFormalSolution.hpp"
#include "ScRadiation.hpp"
#include <chrono>

namespace {
/// Keep each timing scope around one complete numerical step.
template <class Run>
void sc_timed(const char* label, const Run& run) {
    yakl::timer_start(label);
    run();
    yakl::timer_stop(label);
}
/// Time completed device work only when requested; the normal path adds no fences.
template <class Run>
void sc_measure(bool enabled, f64& total, const Run& run) {
    if (!enabled) {
        run();
        return;
    }
    Kokkos::fence();
    const auto start = std::chrono::steady_clock::now();
    run();
    Kokkos::fence();
    total += std::chrono::duration<f64>(std::chrono::steady_clock::now() - start).count();
}
} // namespace

/// Allocate task and wavelength payloads before binding any borrowed solve state.
ScFormalSolution sc_make_formal_solution(
    const ScState& state,
    ScPreparedGeometry geometry,
    const ScOptions& opts,
    i32 wave_batch,
    ThermalisedBc thermalised,
    ScSolveMode mode
) {
    auto work = sc_make_buffers(geometry, wave_batch);
    Fp1d staging(
        "sc_task_row_staging",
        i64(sc_max_batch_dirs(geometry.batches)) * wave_batch * geometry.grid.plane_size()
    );
    auto wavelengths = sc_make_wavelength_storage(
        state,
        geometry.slab.num_ranks,
        opts,
        wave_batch,
        mode == ScSolveMode::Diagnostic
    );
    auto cache = sc_make_coefficient_cache(state, opts.base_slots, wave_batch);
    return {
        .geometry = std::move(geometry),
        .opts = opts,
        .wave_batch = wave_batch,
        .thermalised = thermalised,
        .work = std::move(work),
        .staging = std::move(staging),
        .wavelengths = std::move(wavelengths),
        .medium_coefficient_cache = std::move(cache),
        .epoch = {},
    };
}

/// Bind a solve epoch and choose its boundary/J policy before admitting tasks.
ScTaskPlan sc_begin_solve(
    ScFormalSolution& fs,
    ScState& state,
    const Fp3d& emergent_,
    bool compute_gamma_,
    bool lambda_iterate_,
    ScSolveMode mode
) {
    JasUnpack(fs, epoch, wavelengths, medium_coefficient_cache, geometry, wave_batch, thermalised);
    if (epoch) {
        throw std::logic_error("SC begin_solve called before finish_solve");
    }
    const bool prd_only = mode == ScSolveMode::Prd;
    sc_begin_wavelength_solve(wavelengths, mode != ScSolveMode::Formal);
    sc_reset_coefficient_cache(medium_coefficient_cache);
    const bool given = state.config.mode == DexrtMode::GivenFs;
    ScJPath J_path = ScJPath::None;
    if (wavelengths.store_J) {
        if (!state.config.sparse_calculation) {
            J_path = ScJPath::Dense;
        } else {
            J_path = compute_gamma_ ? ScJPath::InGamma : ScJPath::Active;
        }
    }
    epoch.emplace(ScSolveEpoch{
        .state = state,
        .emergent = emergent_,
        .compute_gamma = mode != ScSolveMode::Diagnostic && compute_gamma_,
        .lambda_iterate = lambda_iterate_,
        .given = given,
        .J_path = J_path,
        .boundary = sc_resolve_boundary(state.boundary, state.pw_bc, thermalised),
        .mode = mode,
        .timings = {},
    });
    if (wavelengths.store_loss) {
        state.rad_loss_integrated = RadLossFp(0.0);
    }
    const i32 nwave = given ? state.given_state.emis.extent(2) : state.adata.wavelength.extent(0);
    return sc_make_task_plan(
        geometry,
        nwave,
        wave_batch,
        wavelengths.slot_count(),
        prd_only ? std::optional(std::pair(state.prd.reach_lo, state.prd.reach_hi)) : std::nullopt
    );
}

/// Prepare inflow, produce the medium, sweep, then pack optional outflow.
/// Angular accumulation follows in the scheduler after starting the send.
void sc_sweep(
    ScFormalSolution& fs,
    const ScScheduledTask& t,
    i32 slot,
    const Fp1dHost* input,
    const Fp1dHost* output
) {
    JasUnpack(fs, geometry, work, staging, wavelengths, medium_coefficient_cache, opts);
    auto& solve = fs.epoch.value();
    auto& state = solve.state;
    const auto& batch = geometry.batches[t.batch];
    const i32 width = t.end - t.la;
    const bool local = batch.mode == ScRayMode::AlongY;
    const bool detailed = opts.scheduler_stats;
    auto& timings = solve.timings;
    ScLossSink loss;
    if (wavelengths.store_loss) {
        loss = ScLossSink{
            .loss = wavelengths.loss(slot),
            .wavelength_bin = state.adata.wavelength_bin,
            .wmu = geometry.dirs.wmu,
            .dir_idx = batch.dir_idx,
            .la_start = t.la,
        };
    }
    const ScSweepTask task{
        .batch = batch,
        .la_start = t.la,
        .wave_batch = width,
        .has_upstream = input != nullptr,
        .has_downstream = output != nullptr,
        .loss = loss,
    };

    if (input) {
        sc_measure(detailed, timings.unpack, [&] {
            sc_unpack_intensity(
                *input,
                staging,
                work.incoming,
                batch.num_dirs,
                width
            );
        });
    }
    if (!local) {
        sc_measure(detailed, timings.boundary, [&] {
            sc_prepare_boundaries(geometry, work, task, solve.boundary);
        });
    }
    sc_timed("SC medium", [&] {
        const auto previous_J = wavelengths.previous_J(slot);
        const ScMediumCoefficients* coefficients = nullptr;
        if (!solve.given) {
            sc_measure(detailed, timings.coefficients, [&] {
                coefficients = &sc_get_medium_coefficients(
                    medium_coefficient_cache,
                    state,
                    previous_J,
                    t.la,
                    t.end
                );
            });
        }
        sc_measure(detailed, timings.medium, [&] {
            sc_prepare_medium(
                state,
                geometry.dirs,
                batch,
                coefficients,
                work.medium,
                t.la,
                width
            );
        });
    });
    sc_timed("SC sweep", [&] {
        sc_measure(detailed, timings.sweep, [&] {
            if (local) {
                sc_along_y_2d(geometry, work, task);
            } else {
                sc_solve(geometry, work, task, opts);
            }
        });
    });
    if (output) {
        sc_measure(detailed, timings.pack, [&] {
            sc_pack_intensity(
                work.outgoing,
                staging,
                *output,
                batch.num_dirs,
                width
            );
        });
    }
}

/// Consume this task before workspace reuse; wavelength sums survive across directions.
void sc_accumulate(ScFormalSolution& fs, const ScScheduledTask& t, i32 slot) {
    JasUnpack(fs, geometry, work, wavelengths, medium_coefficient_cache);
    auto& solve = fs.epoch.value();
    auto& state = solve.state;
    const auto& batch = geometry.batches[t.batch];
    const i32 wb = t.end - t.la;
    const Fp2d J_slot = wavelengths.J(slot);
    const ScPrd& prd = state.prd;
    if (solve.compute_gamma) {
        yakl::timer_start("SC Gamma");
        sc_accumulate_gamma(
            state,
            medium_coefficient_cache.current(),
            geometry.dirs,
            batch,
            work.medium,
            work.solution,
            t.la,
            wb,
            solve.lambda_iterate,
            solve.J_path == ScJPath::InGamma ? J_slot : Fp2d(),
            wavelengths.previous_J(slot)
        );
        yakl::timer_stop("SC Gamma");
    }
    if (solve.J_path == ScJPath::Dense || solve.J_path == ScJPath::Active) {
        yakl::timer_start("SC J");
        if (solve.J_path == ScJPath::Dense) {
            sc_accumulate_J<true>(
                state,
                geometry.dirs,
                batch,
                work.medium,
                work.solution,
                J_slot,
                wb
            );
        } else {
            sc_accumulate_J<false>(
                state,
                geometry.dirs,
                batch,
                work.medium,
                work.solution,
                J_slot,
                wb
            );
        }
        yakl::timer_stop("SC J");
    }
    if (solve.mode != ScSolveMode::Diagnostic && prd.enabled() && t.end > prd.reach_lo && t.la < prd.reach_hi) {
        yakl::timer_start("SC PRD J*");
        sc_prd_accumulate_J_rest(
            state,
            prd,
            geometry.dirs,
            batch,
            work.medium,
            work.solution,
            t.la,
            wb
        );
        yakl::timer_stop("SC PRD J*");
    }
    if (!geometry.slab.has_upper() && solve.emergent.initialized()) {
        sc_store_emergent(
            geometry.grid,
            geometry.dirs,
            batch,
            work.medium,
            work.solution,
            work.outgoing,
            solve.emergent,
            t.la,
            wb
        );
    }
}

void sc_open_wavelength_batch(ScFormalSolution& fs, const ScScheduledTask& task, i32 slot) {
    sc_open_wavelength_slot(fs.wavelengths, fs.epoch.value().state, task, slot);
}

void sc_close_wavelength_batch(ScFormalSolution& fs, const ScScheduledTask& task, i32 slot) {
    sc_close_wavelength_slot(fs.wavelengths, fs.epoch.value().state, task, slot);
}

void sc_finish_solve(ScFormalSolution& fs) {
    auto& state = fs.epoch.value().state;
    sc_finish_wavelength_solve(fs.wavelengths, state);
    state.println(
        "SC task coefficients: base builds {} hits {}",
        fs.medium_coefficient_cache.builds,
        fs.medium_coefficient_cache.hits
    );
    fs.epoch.reset();
}
