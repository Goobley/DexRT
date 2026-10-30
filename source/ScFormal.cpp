#include "ScFormal.hpp"
#include "ScPreparedGeometry.hpp"
#include "LoopUtils.hpp"
#include <type_traits>

/// Allocate one reusable task workspace with solver-specific scratch and a common outgoing plane.
ScBlockBuffers sc_make_buffers(const ScPreparedGeometry& geometry, i32 width) {
    JasUnpack(geometry, grid, batches, method, storage, longer_layout);
    ScBlockBuffers result;
    const i32 nd = sc_max_batch_dirs(batches);
    const i64 plane = grid.plane_size();
    const i64 cells = plane * grid.num_z;
    const i64 medium = plane * (grid.num_z + 2 * SC_MEDIUM_HALO);
    result.medium = {
        .chi = Fp3d("sc_chi", nd, width, medium),
        .S = Fp3d("sc_S", nd, width, medium),
    };
    result.solution = {
        .I_ext = Fp3d("sc_I_ext", nd, width, cells),
        .alo = Fp3d("sc_alo", nd, width, cells),
        .one_minus_alo = Fp3d("sc_one_minus_alo", nd, width, cells),
    };
    result.incoming = Fp3d("sc_incoming", nd, width, plane);
    result.outgoing = Fp3d("sc_outgoing", nd, width, plane);
    const auto allocate = [&](const char* name, i64 count) {
        return count ? Fp3d(name, nd, width, count) : Fp3d();
    };
    if (method == ScCharacteristics::Short) {
        auto& scratch = result.scratch.emplace<ScShortStorage>();
        scratch.side_x = storage.side_x;
        scratch.side_y = storage.side_y;
        const auto& sizes = storage.short_inflow;
        scratch.bottom = allocate("sc_short_bottom", sizes.bottom);
        scratch.edge = allocate("sc_short_edge", sizes.edge);
        scratch.ghost = allocate("sc_short_ghost", sizes.ghost);
        scratch.side = allocate("sc_short_side", sizes.side);
        scratch.xface = allocate("sc_short_xface", sizes.xface);
        scratch.yface = allocate("sc_short_yface", sizes.yface);
        scratch.ghost_x = allocate("sc_short_ghost_x", sizes.ghost_x);
        scratch.ghost_y = allocate("sc_short_ghost_y", sizes.ghost_y);
        scratch.intensity_planes = Fp3d("sc_intensity_planes", nd, width, 2 * plane);
    } else {
        auto& scratch = result.scratch.emplace<ScLongerStorage>();
        scratch.bottom = allocate("sc_longer_bottom", storage.longer_inflow.bottom);
        scratch.side_x = allocate("sc_longer_side_x", storage.longer_inflow.side_x);
        scratch.side_y = allocate("sc_longer_side_y", storage.longer_inflow.side_y);
        if constexpr (HostDevSameSpace) {
            scratch.host_lines = Fp3d("sc_host_lines", nd, width, longer_layout.line_count);
        }
    }
    return result;
}

/// Along the invariant y axis the medium is homogeneous: I=S if opaque, zero otherwise.
void sc_along_y_2d(const ScPreparedGeometry& geometry, const ScBlockBuffers& work, const ScSweepTask& task) {
    JasUnpack(geometry, grid);
    JasUnpack(task, wave_batch);
    JasUnpack(work.medium, chi);
    JasUnpack(work.solution, I_ext, alo, one_minus_alo);
    constexpr i32 halo = SC_MEDIUM_HALO;
    const i32 Ns = grid.num_x;
    // NOTE(cmo): Along y in a 2.5D medium (an infinite homogeneous medium): the
    // source function, I = S, psi_star = 1 / chi.
    dex_parallel_for(
        "SC along y",
        FlatLoop<3>(task.batch.num_dirs, wave_batch, grid.num_z * i64(Ns)),
        YAKL_LAMBDA(i32 d, i32 w, i64 idx) {
            const bool opaque = chi(d, w, idx + halo * Ns) > FP(0.0);
            I_ext(d, w, idx) = FP(0.0);
            alo(d, w, idx) = opaque ? FP(1.0) : FP(0.0);
            one_minus_alo(d, w, idx) = opaque ? FP(0.0) : FP(1.0);
        });
    Kokkos::fence();
}

/// Dispatch the prepared workspace to its solver, outside all device kernels.
void sc_solve(
    const ScPreparedGeometry& geometry,
    const ScBlockBuffers& work,
    const ScSweepTask& task,
    const ScOptions& opts) {
    std::visit(
        [&](const auto& scratch) {
            using Storage = std::decay_t<decltype(scratch)>;
            if constexpr (std::is_same_v<Storage, ScShortStorage>) {
                sc_solve_short(
                    geometry,
                    work,
                    task,
                    opts.formal_solver,
                    opts.interpolation
                );
            } else {
                sc_solve_longer(geometry, work, task);
            }
        },
        work.scratch);
}
