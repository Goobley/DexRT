#include "ScBoundary.hpp"
#include <type_traits>
#include "ScLongerGeometry.hpp"
#include "ScRayEntry.hpp"
#include "ScShortNode.hpp"
#include "ScBoundaryCone.hpp"
#include "LoopUtils.hpp"

namespace {
struct ScShortBoundaryPoint {
    fp_t z = 0, x = 0, y = 0;
    bool active = true;
};

/// Each named array has one launch and one coordinate rule, independent of the medium.
template <int Dim, class Bc, class Locate>
void sc_fill_short_boundary(
    const ScPreparedGeometry& geometry,
    const ScSweepTask& task,
    const Bc& bc,
    const Fp3d& output,
    const char* label,
    const Locate& locate) {
    JasUnpack(geometry, grid);
    JasUnpack(task.batch, frames, dir_idx);
    JasUnpack(geometry.dirs, mu);
    const ScConeQuad cone(geometry.dirs);
    const i32 la = task.la_start;
    dex_parallel_for(
        label,
        FlatLoop<3>(task.batch.num_dirs, task.wave_batch, output.extent(2)),
        YAKL_LAMBDA(i32 d, i32 w, i64 e) {
            const auto point = locate(d, e);
            if (point.active) {
                const auto sample =
                    sc_boundary_sampler<Dim>(
                        grid,
                        frames(d),
                        bc,
                        cone,
                        mu,
                        dir_idx(d),
                        la + w
                    );
                output(d, w, e) = sample(point.z, point.x, point.y);
            }
        });
}

/// Sample the four independent 2D boundary arrays in canonical coordinates.
template <class Bc>
void sc_boundary_values(
    const ScPreparedGeometry& geometry,
    const ScBlockBuffers& work,
    const ScSweepTask& task,
    const Bc& bc) {
    JasUnpack(geometry, periodic_x);
    JasUnpack(task.batch, row_t);
    JasUnpack(work.short_storage(), bottom, edge, ghost, side, side_x);
    const bool has_upstream = task.has_upstream;
    const i32 nx = geometry.grid.num_x;
    if (!has_upstream) {
        sc_fill_short_boundary<2>(
            geometry,
            task,
            bc,
            bottom,
            "SC Short bottom",
            YAKL_LAMBDA(i32 d, i64 s) {
                const auto entry =
                    sc_node_entry<2>(
                        row_t(d, 0),
                        FP(0.0),
                        0,
                        s,
                        0,
                        periodic_x,
                        false,
                        has_upstream
                    );
                return ScShortBoundaryPoint{.z = -FP(0.5), .x = entry.x_start};
            });
    }
    if (!periodic_x) {
        sc_fill_short_boundary<2>(
            geometry,
            task,
            bc,
            edge,
            "SC Short edge",
            YAKL_LAMBDA(i32 d, i64 p) { return ScShortBoundaryPoint{.z = fp_t(p - 1), .x = -FP(0.5)}; });
        sc_fill_short_boundary<2>(
            geometry,
            task,
            bc,
            ghost,
            "SC Short ghost",
            YAKL_LAMBDA(i32 d, i64 g) {
                const i32 p = g / 2 - 1;
                const i32 s = -(g - 2 * (g / 2)) - 1;
                return ScShortBoundaryPoint{.z = fp_t(p), .x = fp_t(s)};
            });
        sc_fill_short_boundary<2>(
            geometry,
            task,
            bc,
            side,
            "SC Short side",
            YAKL_LAMBDA(i32 d, i64 e) {
                const i32 p = e / side_x;
                const i32 s = e - i64(p) * side_x;
                if (s >= nx) {
                    return ScShortBoundaryPoint{.active = false};
                }
                const auto entry =
                    sc_node_entry<2>(
                        row_t(d, p),
                        FP(0.0),
                        p,
                        s,
                        0,
                        periodic_x,
                        false,
                        has_upstream
                    );
                if (!entry.side) {
                    return ScShortBoundaryPoint{.active = false};
                }
                return ScShortBoundaryPoint{
                    .z = fp_t(p - 1) + entry.h_side,
                    .x = -FP(0.5),
                    .active = entry.side,
                };
            });
    }
}

/// Sample bottom, physical faces, ghosts and side entries into independent 3D arrays.
template <class Bc>
void sc_boundary_values_3d(
    const ScPreparedGeometry& geometry,
    const ScBlockBuffers& work,
    const ScSweepTask& task,
    const Bc& bc) {
    JasUnpack(geometry, periodic_x, periodic_y);
    JasUnpack(task.batch, row_t, row_ty);
    JasUnpack(work.short_storage(), bottom, xface, yface, ghost_x, ghost_y, side, side_x, side_y);
    const bool has_upstream = task.has_upstream;
    const i32 nx = geometry.grid.num_x, ny = geometry.grid.num_y;
    if (!has_upstream) {
        sc_fill_short_boundary<3>(
            geometry,
            task,
            bc,
            bottom,
            "SC Short bottom 3d",
            YAKL_LAMBDA(i32 d, i64 e) {
                const i32 sy = e / nx;
                const i32 s = e - i64(sy) * nx;
                const fp_t tx = row_t(d, 0), ty = row_ty(d, 0);
                return ScShortBoundaryPoint{
                    .z = -FP(0.5),
                    .x = fp_t(s) - FP(0.5) * tx,
                    .y = fp_t(sy) - FP(0.5) * ty,
                };
            });
    }
    if (!periodic_x) {
        sc_fill_short_boundary<3>(
            geometry,
            task,
            bc,
            xface,
            "SC Short xface",
            YAKL_LAMBDA(i32 d, i64 e) {
                const i32 p = e / (ny + 1);
                const i32 sy = e - i64(p) * (ny + 1) - 1;
                return ScShortBoundaryPoint{
                    .z = fp_t(p - 1),
                    .x = -FP(0.5),
                    .y = sy < 0 ? -FP(0.5) : fp_t(sy),
                };
            });
        sc_fill_short_boundary<3>(
            geometry,
            task,
            bc,
            ghost_x,
            "SC Short ghost x",
            YAKL_LAMBDA(i32 d, i64 e) {
                const i64 row = 2 * (i64(ny) + 2);
                const i32 p = e / row;
                const i64 g = e - i64(p) * row;
                const i32 gx = -1 - i32(g / (ny + 2));
                i32 sy = i32(g % (ny + 2)) - 2;
                if (periodic_y) {
                    sy = sc_wrap(sy, ny);
                }
                return ScShortBoundaryPoint{
                    .z = fp_t(p - 1),
                    .x = fp_t(gx),
                    .y = fp_t(sy),
                };
            });
    }
    if (!periodic_y) {
        sc_fill_short_boundary<3>(
            geometry,
            task,
            bc,
            yface,
            "SC Short yface",
            YAKL_LAMBDA(i32 d, i64 e) {
                const i32 p = e / nx;
                const i32 s = e - i64(p) * nx;
                return ScShortBoundaryPoint{
                    .z = fp_t(p - 1),
                    .x = fp_t(s),
                    .y = -FP(0.5),
                };
            });
        sc_fill_short_boundary<3>(
            geometry,
            task,
            bc,
            ghost_y,
            "SC Short ghost y",
            YAKL_LAMBDA(i32 d, i64 e) {
                const i64 row = 2 * i64(nx);
                const i32 p = e / row;
                const i64 g = e - i64(p) * row;
                const i32 gy = -1 - i32(g / nx);
                const i32 s = g % nx;
                return ScShortBoundaryPoint{
                    .z = fp_t(p - 1),
                    .x = fp_t(s),
                    .y = fp_t(gy),
                };
            });
    }
    if (!periodic_x || !periodic_y) {
        sc_fill_short_boundary<3>(
            geometry,
            task,
            bc,
            side,
            "SC Short side 3d",
            YAKL_LAMBDA(i32 d, i64 e) {
                const i64 row = i64(ny) * side_x + i64(side_y) * nx;
                const i32 p = e / row;
                const i64 q = e - i64(p) * row;
                i32 s, sy;
                if (q < i64(ny) * side_x) {
                    sy = q / side_x;
                    s = q - i64(sy) * side_x;
                } else {
                    const i64 rest = q - i64(ny) * side_x;
                    sy = rest / nx;
                    s = rest - i64(sy) * nx;
                }
                if (s >= nx || sy >= ny) {
                    return ScShortBoundaryPoint{.active = false};
                }
                const fp_t tx = row_t(d, p), ty = row_ty(d, p);
                const auto entry = sc_node_entry<3>(
                    tx,
                    ty,
                    p,
                    s,
                    sy,
                    periodic_x,
                    periodic_y,
                    has_upstream
                );
                if (!entry.side) {
                    return ScShortBoundaryPoint{.active = false};
                }
                const fp_t h = entry.h_side;
                return ScShortBoundaryPoint{
                    .z = fp_t(p - 1) + h,
                    .x = std::max(entry.x0 + h * tx, -FP(0.5)),
                    .y = std::max(entry.y0 + h * ty, -FP(0.5)),
                    .active = entry.side,
                };
            });
    }
}
} // namespace

void sc_prepare_short_boundaries(
    const ScPreparedGeometry& geometry,
    const ScBlockBuffers& work,
    const ScSweepTask& task,
    const ScBoundary& boundary) {
    std::visit(
        [&](const auto& bc) {
            if (geometry.grid.dim == 2) {
                sc_boundary_values(geometry, work, task, bc);
            } else {
                sc_boundary_values_3d(geometry, work, task, bc);
            }
        },
        boundary);
}

namespace {
/// Fill one physical characteristic-entry array using the existing global lattice labels.
template <int Dim, bool UniformZ, bool Face, class Bc>
void prepare_longer_array(
    const ScPreparedGeometry& geometry,
    const ScSweepTask& task,
    const Bc& bc,
    const Fp3d& output,
    i64 row_size,
    i64 side_offset,
    const char* label) {
    JasUnpack(geometry, grid);
    JasUnpack(task.batch, frames, longer, dir_idx);
    JasUnpack(longer, layout);
    JasUnpack(geometry.dirs, mu);
    const ScConeQuad cone(geometry.dirs);
    const i32 la = task.la_start;
    dex_parallel_for(
        label,
        FlatLoop<3>(task.batch.num_dirs, task.wave_batch, output.extent(2)),
        YAKL_LAMBDA(i32 d, i32 w, i64 e) {
            JasUse(layout, row_size, side_offset); // nvcc needs first capture outside constexpr-if.
            i64 ray = e;
            if constexpr (!Face) {
                const i32 p = e / row_size;
                const i64 local = e - i64(p) * row_size;
                ray = layout.plane_size + p * layout.side_size + side_offset + local;
            }
            const auto entry = sc_longer_entry(layout, ray);
            const auto path = sc_longer_plane_path<UniformZ>(longer, longer.directions(d), entry.plane);
            const auto position = sc_longer_entry_position(layout, path, entry);
            const auto sample = sc_boundary_sampler<Dim>(
                grid,
                frames(d),
                bc,
                cone,
                mu,
                dir_idx(d),
                la + w
            );
            output(d, w, e) = sample(position.z, position.x, position.y);
        });
}

/// Bottom and each lateral face have independent storage and launches.
template <int Dim, bool UniformZ, class Bc>
void prepare_boundaries(
    const ScPreparedGeometry& geometry,
    const ScBlockBuffers& work,
    const ScSweepTask& task,
    const Bc& bc) {
    JasUnpack(work.longer_storage(), bottom, side_x, side_y);
    JasUnpack(task.batch.longer, layout);
    if (!task.has_upstream) {
        prepare_longer_array<Dim, UniformZ, true>(
            geometry,
            task,
            bc,
            bottom,
            layout.plane_size,
            0,
            "SC Longer bottom");
    }
    if (layout.side_rows) {
        prepare_longer_array<Dim, UniformZ, false>(
            geometry,
            task,
            bc,
            side_y,
            layout.side_rows,
            0,
            "SC Longer side y");
    }
    const i64 x_rows = layout.side_size - layout.side_rows;
    if (x_rows) {
        prepare_longer_array<Dim, UniformZ, false>(
            geometry,
            task,
            bc,
            side_x,
            x_rows,
            layout.side_rows,
            "SC Longer side x");
    }
}

template <bool UniformZ>
void prepare_spacing_boundaries(
    const ScPreparedGeometry& geometry,
    const ScBlockBuffers& work,
    const ScSweepTask& task,
    const ScBoundary& boundary) {
    std::visit(
        [&](const auto& bc) {
            if (geometry.grid.dim == 2) {
                prepare_boundaries<2, UniformZ>(geometry, work, task, bc);
            } else {
                prepare_boundaries<3, UniformZ>(geometry, work, task, bc);
            }
        },
        boundary);
}
} // namespace

void sc_prepare_longer_boundaries(
    const ScPreparedGeometry& geometry,
    const ScBlockBuffers& work,
    const ScSweepTask& task,
    const ScBoundary& boundary) {
    if (task.batch.longer.uniform_z) {
        prepare_spacing_boundaries<true>(geometry, work, task, boundary);
    } else {
        prepare_spacing_boundaries<false>(geometry, work, task, boundary);
    }
}

namespace {
using ScDevSpan = Kokkos::View<fp_t*, DefaultMemSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>>;
using ScHostSpan = Kokkos::View<fp_t*, Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>>;

} // namespace

/// Consume the scheduler receive row synchronously before it can be reposted.
void sc_unpack_intensity(
    const Fp1dHost& row,
    const Fp1d& staging,
    const Fp3d& incoming,
    i32 num_dirs,
    i32 wave_batch) {
    const i32 ns = incoming.extent(2);
    Kokkos::deep_copy(
        ScDevSpan(staging.data(), i64(num_dirs) * wave_batch * ns),
        ScHostSpan(row.data(), i64(num_dirs) * wave_batch * ns));
    dex_parallel_for(
        "SC unpack I_in",
        FlatLoop<3>(num_dirs, wave_batch, ns),
        YAKL_LAMBDA(i32 d, i32 w, i32 s) {
            incoming(d, w, s) = staging((i64(d) * wave_batch + w) * ns + s);
        });
    Kokkos::fence();
}

/// Finish the device-to-host copy before MPI borrows the independent send row.
void sc_pack_intensity(
    const Fp3d& outgoing,
    const Fp1d& staging,
    const Fp1dHost& row,
    i32 num_dirs,
    i32 wave_batch) {
    const i32 ns = outgoing.extent(2);
    dex_parallel_for(
        "SC pack I row",
        FlatLoop<3>(num_dirs, wave_batch, ns),
        YAKL_LAMBDA(i32 d, i32 w, i32 s) {
            staging((i64(d) * wave_batch + w) * ns + s) = outgoing(d, w, s);
        });
    Kokkos::fence();
    Kokkos::deep_copy(
        ScHostSpan(row.data(), i64(num_dirs) * wave_batch * ns),
        ScDevSpan(staging.data(), i64(num_dirs) * wave_batch * ns));
}

/// Select the concrete boundary once; visits dispatch only inflow kernels.
ScBoundary sc_resolve_boundary(BoundaryType type, const PwBc<>& pw_bc, const ThermalisedBc& thermalised) {
    switch (type) {
    case BoundaryType::Zero:
        return ZeroBc{};
    case BoundaryType::Promweaver:
        return pw_bc;
    case BoundaryType::Thermalised:
        return thermalised;
    default:
        throw std::runtime_error("Unhandled boundary type in SC solve");
    }
}

/// Select the solver's physical inflow preparation before medium production.
void sc_prepare_boundaries(
    const ScPreparedGeometry& geometry,
    const ScBlockBuffers& work,
    const ScSweepTask& task,
    const ScBoundary& boundary) {
    std::visit(
        [&](const auto& scratch) {
            using Storage = std::decay_t<decltype(scratch)>;
            if constexpr (std::is_same_v<Storage, ScShortStorage>) {
                sc_prepare_short_boundaries(geometry, work, task, boundary);
            } else {
                sc_prepare_longer_boundaries(geometry, work, task, boundary);
            }
        },
        work.scratch);
}
