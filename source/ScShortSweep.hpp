#ifndef DEXRT_SC_SHORT_SWEEP_HPP
#define DEXRT_SC_SHORT_SWEEP_HPP
#include "ScShortView.hpp"
#include "ScShortNode.hpp"
#include <limits>
#include <stdexcept>

#if !defined(SC_3D_LB_THREADS)
#define SC_3D_LB_THREADS 512
#define SC_3D_LB_BLOCKS 2
#endif

/// Short's plane driver is the same in both dimensions. The node solver and
/// storage view supply the dimension-specific interpolation/boundary geometry.
template <int Dim, ScFormalSolver Solver, ScInterp Interp, bool Loss>
void sc_short_sweep(const ScPreparedGeometry& geometry, const ScBlockBuffers& work, const ScSweepTask& task) {
    const ScShortBatch<Dim> storage(geometry, work, task);
    const i64 plane = geometry.grid.plane_size();
    if constexpr (Dim == 3) {
        if (plane * (geometry.grid.num_z + 2 * SC_MEDIUM_HALO) > std::numeric_limits<i32>::max() ||
            geometry.storage.short_inflow.total() > std::numeric_limits<i32>::max()) {
            throw std::runtime_error("SC 3D: a slab's planes exceed the i32 offsets of the sweep");
        }
    }
    const i32 team_size = dex_team_size(sc_sweep_team_size(plane));
    i32 chunks = 1;
    if constexpr (!HostDevSameSpace) {
        if (geometry.grid.dim == 3) {
            const i64 pairs = i64(task.batch.num_dirs) * task.wave_batch;
            const i64 resident = Kokkos::DefaultExecutionSpace().concurrency() / std::max(team_size, 1);
            if (pairs < resident) {
                chunks = i32(std::max<i64>(
                    1,
                    std::min((resident + pairs - 1) / pairs, (plane + team_size - 1) / team_size)
                ));
            }
        }
    }
    const i32 wave_batch = task.wave_batch;
    const i32 cells = plane;
    const i32 chunk_len = (cells + chunks - 1) / chunks;
    JasUnpack(task.batch, frames, row_t, row_ty, row_ds, row_cell_ds);
    JasUnpack(task, loss);
    JasUnpack(work, outgoing);
    const i32 last_plane = geometry.grid.num_z - 1;

    const auto launch_planes = [&](i32 p0, i32 p1) {
        const auto kernel = KOKKOS_LAMBDA(const KTeam& team) {
            const i32 pair = team.league_rank() / chunks;
            const i32 chunk = team.league_rank() - pair * chunks;
            const i32 d = pair / wave_batch;
            const i32 w = pair - d * wave_batch;
            const auto view = storage.bind(d, w);
            JasUse(frames); // nvcc requires first capture outside constexpr-if.
            const i32 c0 = chunk * chunk_len;
            const i32 c1 = std::min(c0 + chunk_len, cells);
            const fp_t loss_weight = Loss ? loss.weight(d, w) : FP(0.0);
            for (i32 p = p0; p < p1; ++p) {
                const ScRowStep step{
                    .tx = row_t(d, p),
                    .ty = Dim == 3 ? row_ty(d, p) : FP(0.0),
                    .ds = row_ds(d, p),
                };
                const fp_t inv_cell_ds = Loss ? FP(1.0) / row_cell_ds(d, p) : FP(0.0);
                Kokkos::parallel_for(Kokkos::TeamThreadRange(team, c0, c1), [&](i32 c) {
                    const i32 sy = Dim == 3 ? c / view.medium.Nx : 0;
                    const i32 s = Dim == 3 ? c - sy * view.medium.Nx : c;
                    const auto solved = sc_solve_short_node<Dim, Solver, Interp, Loss>(
                        step,
                        p,
                        s,
                        sy,
                        view
                    );
                    const auto& result = solved.result;
                    view.intensity.store(view.medium, p, c, result);
                    if constexpr (Loss) {
                        loss.add(
                            frames(d),
                            w,
                            p,
                            s,
                            sy,
                            loss_weight * (result.I - solved.I_uw) * inv_cell_ds
                        );
                    }
                });
                team.team_barrier();
            }
            if (p1 == last_plane + 1) {
                // The last plane is complete; each chunk copies its own cells.
                Kokkos::parallel_for(Kokkos::TeamThreadRange(team, c0, c1), [&](i32 c) {
                    outgoing(d, w, c) = view.intensity.ring[(last_plane & 1) * view.medium.plane + c];
                });
            }
        };
        // Retain the original dimension-specific launch bounds and team cap.
        using Bounds = std::conditional_t<
            Dim == 2,
            Kokkos::LaunchBounds<512, 2>,
            Kokkos::LaunchBounds<SC_3D_LB_THREADS, SC_3D_LB_BLOCKS>>;
        using Policy = Kokkos::TeamPolicy<Kokkos::DefaultExecutionSpace, Bounds>;
        const i32 league = task.batch.num_dirs * wave_batch * chunks;
        const i32 max_team = Policy(league, 1).team_size_max(kernel, Kokkos::ParallelForTag());
        Kokkos::parallel_for(
            Dim == 2 ? "SC sweep 2d" : "SC sweep 3d",
            Policy(league, std::min(team_size, max_team)),
            kernel
        );
    };
    // A team owns the slab unless a large 3D plane needs device-wide ordering.
    if (chunks == 1) {
        launch_planes(0, geometry.grid.num_z);
    } else {
        for (i32 p = 0; p < geometry.grid.num_z; ++p) {
            launch_planes(p, p + 1);
        }
    }
    Kokkos::fence();
}
#endif
